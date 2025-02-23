import pyzed.sl as sl
import cv2
import numpy as np
import time
from ultralytics import YOLO
import os
import sys
import supervision as sv
import torch
import torchvision

# Constants
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1
COLOR_RED = (0, 0, 255)
THICKNESS = 2
DEPTH_CENTER_COLOR = (255, 0, 0)
DEPTH_CENTER_RADIUS = 5
MAP_SIZE = 500  # Size of the map in pixels
MAP_SCALE = 50  # Scale factor for converting real-world coordinates to map coordinates
COLOR_ROBOT = (0, 255, 0)
COLOR_PATH = (0, 0, 255)
COLOR_OBJECT_RED = (0, 0, 255)    # Kırmızı nesne rengi
COLOR_OBJECT_GREEN = (0, 255, 0)   # Yeşil nesne rengi
COLOR_OBJECT_YELLOW = (0, 255, 255) # Sarı nesne rengi

manual_mode = False

# Load YOLO model
model = YOLO("balonx50.engine")

# Initialize annotators
bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

def initialize_camera():
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080
    init_params.camera_fps = 30
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL
    init_params.coordinate_units = sl.UNIT.METER
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE
    init_params.depth_minimum_distance = 0.20
    init_params.depth_maximum_distance = 40
    init_params.camera_disable_self_calib = True
    init_params.depth_stabilization = 80
    init_params.sensors_required = False
    init_params.enable_image_enhancement = True
    init_params.async_grab_camera_recovery = False
    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        raise Exception("Failed to open ZED camera. Exiting.")
    return zed

# Convert quaternion to specified angle (yaw, roll, pitch)
def quaternion_to_angle(quaternion, angle_type):
    ox, oy, oz, ow = quaternion
    if angle_type == "yaw":
        siny_cosp = 2 * (ow * oz + ox * oy)
        cosy_cosp = 1 - 2 * (oy * oy + oz * oz)
        return np.degrees(np.arctan2(siny_cosp, cosy_cosp))
    elif angle_type == "pitch":
        sinp = 2 * (ow * oy - oz * ox)
        return np.degrees(np.arcsin(sinp)) if abs(sinp) < 1 else np.copysign(90, sinp)
    elif angle_type == "roll":
        sinr_cosp = 2 * (ow * ox + oy * oz)
        cosr_cosp = 1 - 2 * (ox * ox + oy * oy)
        return np.degrees(np.arctan2(sinr_cosp, cosr_cosp))
    return 0

# Initialize positional tracking
def initialize_positional_tracking(zed):
    py_transform = sl.Transform()
    tracking_parameters = sl.PositionalTrackingParameters(_init_pos=py_transform)
    err = zed.enable_positional_tracking(tracking_parameters)
    if err != sl.ERROR_CODE.SUCCESS:
        print("Enable positional tracking : " + repr(err) + ". Exit program.")
        zed.close()
        exit()

# Initialize spatial mapping
def initialize_spatial_mapping(zed):
    mapping_parameters = sl.SpatialMappingParameters(map_type=sl.SPATIAL_MAP_TYPE.MESH)
    mapping_parameters.resolution_meter = 0.10
    mapping_parameters.range_meter = 10
    mapping_parameters.use_chunk_only = True

    error = zed.enable_spatial_mapping(mapping_parameters)
    if error != sl.ERROR_CODE.SUCCESS:
        raise Exception(f"Spatial mapping initialization failed: {error}")

# Draw robot and detected objects on the 2D map
def draw_robot_on_map(map_image, position, yaw, trajectory, detected_objects):
    # Convert position to the map scale
    x = int(MAP_SIZE / 2 + position[0] * MAP_SCALE)
    y = int(MAP_SIZE / 2 - position[2] * MAP_SCALE)  # Flip Y-axis

    # Save the position in the trajectory
    trajectory.append((x, y))

    # Draw the trajectory (path)
    for i in range(1, len(trajectory)):
        cv2.line(map_image, trajectory[i - 1], trajectory[i], COLOR_PATH, 1)

    # Draw detected objects on the map
    for obj in detected_objects:
        obj_x = int(MAP_SIZE / 2 + obj['position'][0] * MAP_SCALE)
        obj_y = int(MAP_SIZE / 2 - obj['position'][2] * MAP_SCALE)
        
        # Nesne türüne göre renk seç
        if obj['class_id'] == 3:  # Kırmızı
            color = COLOR_OBJECT_RED
        elif obj['class_id'] == 2:  # Yeşil
            color = COLOR_OBJECT_GREEN
        elif obj['class_id'] == 4:  # Sarı
            color = COLOR_OBJECT_YELLOW
        
        # Nesneyi haritada işaretle
        cv2.circle(map_image, (obj_x, obj_y), 3, color, -1)

    # Draw the robot's current position
    cv2.circle(map_image, (x, y), 5, COLOR_ROBOT, -1)

    # Draw the robot's orientation as a line
    direction_x = int(x + 10 * np.cos(np.radians(yaw)))
    direction_y = int(y - 10 * np.sin(np.radians(yaw)))
    cv2.line(map_image, (x, y), (direction_x, direction_y), COLOR_ROBOT, 2)

# Render text over the frame
def render_text(frame, text, position):
    cv2.putText(frame, text, position, FONT, FONT_SCALE, COLOR_RED, THICKNESS)

# Constants for depth stability checking
DEPTH_CHECK_HEIGHT = 20  # Ekran merkezinin üstündeki piksel sayısı
DEPTH_STABILITY_THRESHOLD = 0.15  # Ani derinlik değişimi eşiği (metre)
DEPTH_CHECK_INTERVAL = 20  # Yatay tarama aralığı (piksel)
DEPTH_HISTORY_SIZE = 3  # Nesne takibi için derinlik geçmişi boyutu

class DetectionFilter:
    def __init__(self):
        self.history = {}  # {class_id: {'positions': [], 'depths': [], 'last_seen': timestamp}}
    
    def update(self, class_id, position, depth_val):
        """
        Nesne tespitini filtrele ve stabil olup olmadığını kontrol et.
        Ani derinlik değişimleri olan nesneleri (su yüzeyi) filtrele.
        """
        current_time = time.time()
        
        if class_id not in self.history:
            self.history[class_id] = {
                'positions': [], 
                'depths': [],
                'last_seen': current_time
            }
        
        history = self.history[class_id]
        history['positions'].append(position)
        history['depths'].append(depth_val)
        history['last_seen'] = current_time
        
        # Geçmiş boyutunu sınırla
        if len(history['positions']) > DEPTH_HISTORY_SIZE:
            history['positions'].pop(0)
            history['depths'].pop(0)
        
        # Derinlik stabilitesini kontrol et
        if len(history['depths']) < 2:
            return False
            
        # Ardışık derinlik değerleri arasındaki değişimi kontrol et
        max_diff = max(abs(d1 - d2) 
                      for d1, d2 in zip(history['depths'][:-1], 
                                      history['depths'][1:]))
        
        # Stabil derinlik değişimi varsa nesneyi kabul et
        return max_diff <= DEPTH_STABILITY_THRESHOLD
    
    def clean_old_detections(self, timeout=5.0):
        """5 saniyedir görünmeyen nesneleri temizle"""
        current_time = time.time()
        to_remove = []
        
        for class_id, history in self.history.items():
            if current_time - history['last_seen'] > timeout:
                to_remove.append(class_id)
        
        for class_id in to_remove:
            del self.history[class_id]

def detect_obstacles(depth, center_x, center_y):
    """
    Ekranın üst kısmında yatay bir çizgi boyunca engelleri tespit eder.
    Su yüzeyini tespit etmek için ani derinlik değişimlerini kontrol eder.
    """
    obstacles = []
    surface_y = center_y - DEPTH_CHECK_HEIGHT  # Merkez üstündeki kontrol çizgisi
    
    # Yatay çizgi boyunca derinlik kontrolü
    for x in range(0, width, DEPTH_CHECK_INTERVAL):
        # Komşu noktaların derinlik değerlerini al
        depths = []
        for dx in [-DEPTH_CHECK_INTERVAL, 0, DEPTH_CHECK_INTERVAL]:
            if 0 <= x + dx < width:
                depth_val = depth.get_value(x + dx, surface_y)[1]
                if not np.isnan(depth_val):
                    depths.append(depth_val)
        
        # En az 2 geçerli derinlik değeri varsa stabilite kontrolü yap
        if len(depths) >= 2:
            # Ardışık derinlik değerleri arasındaki maksimum farkı hesapla
            max_diff = max(abs(d1 - d2) for d1, d2 in zip(depths[:-1], depths[1:]))
            
            # Derinlik değişimi ani değilse (su yüzeyi değilse) engel olarak işaretle
            if max_diff <= DEPTH_STABILITY_THRESHOLD:
                point_cloud = sl.Mat()
                zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
                point3D = point_cloud.get_value(x, surface_y)
                
                # Geçerli 3D koordinat varsa engel listesine ekle
                if not np.isnan(point3D[2]):
                    obstacles.append(point3D[:3])
    
    return obstacles

# Basic class to handle the timestamp of the different sensors
class TimestampHandler:
    def __init__(self):
        self.t_imu = sl.Timestamp()
    
    def is_new(self, sensor):
        if isinstance(sensor, sl.IMUData):
            new_ = (sensor.timestamp.get_microseconds() > self.t_imu.get_microseconds())
            if new_:
                self.t_imu = sensor.timestamp
            return new_

def main():
    zed = initialize_camera()

    # Initialize positional tracking and spatial mapping
    initialize_positional_tracking(zed)
    print("initialized positional tracking..")
    initialize_spatial_mapping(zed)
    print("initialized spatial mapping..")

    # Initialize a blank 2D map
    map_image = np.ones((MAP_SIZE, MAP_SIZE, 3), dtype=np.uint8) * 255

    # Store the robot's trajectory
    trajectory = []

    # Used to store the sensors timestamp
    ts_handler = TimestampHandler()
    
    # Initialize detection filter
    detection_filter = DetectionFilter()

    # Create containers
    image = sl.Mat()
    depth = sl.Mat()
    pose = sl.Pose()
    mesh = sl.Mesh()
    sensors_data = sl.SensorsData()

    # Get camera resolution
    camera_info = zed.get_camera_information()
    width = camera_info.camera_configuration.resolution.width
    height = camera_info.camera_configuration.resolution.height
    center_x = width // 2
    center_y = height // 2

    print("Kamera çözünürlüğü: ", width, "x", height)
    print("Görüntü orta noktası: ", (center_x, center_y))

    # For FPS calculation
    fps_previous_time = 0
    mesh_timer = 0

    while True:
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            # Get image and depth
            zed.retrieve_image(image, sl.VIEW.LEFT)
            zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
            frame = cv2.cvtColor(image.get_data(), cv2.COLOR_BGRA2BGR)

            # YOLO object detection
            results = model(frame, conf=0.50)[0]
            detections = sv.Detections.from_ultralytics(results)
            frame = bounding_box_annotator.annotate(scene=frame, detections=detections)
            frame = label_annotator.annotate(scene=frame, detections=detections)

            # Get object positions and classes
            coordinates = detections.xyxy.tolist()
            class_ids = detections.class_id.tolist()
            detected_objects = []

            # Process detected objects
            for i, (box, class_id) in enumerate(zip(coordinates, class_ids)):
                x1, y1, x2, y2 = map(int, box)
                depth_val = depth.get_value(x2, y1)[1]
                
                if not np.isnan(depth_val):
                    point_cloud = sl.Mat()
                    zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
                    point3D = point_cloud.get_value(x2, y1)
                    
                    # Derinlik stabilitesi kontrolü
                    if detection_filter.update(class_id, point3D[:3], depth_val):
                        # Stabil tespitleri detected_objects'e ekle
                        filtered_position = np.mean(detection_filter.history[class_id]['positions'], axis=0)
                        
                        detected_objects.append({
                            'position': filtered_position,
                            'class_id': class_id
                        })
                    
                    # Display depth on frame
                    text = f"{depth_val:.2f} m"
                    cv2.putText(frame, text, (x2 - 60, y1 + 20), FONT, 0.7, COLOR_RED, 2)
            
            # Eski tespitleri temizle
            detection_filter.clean_old_detections()

            # Get sensor data and position
            if zed.get_sensors_data(sensors_data, sl.TIME_REFERENCE.IMAGE) and zed.get_position(pose):
                if ts_handler.is_new(sensors_data.get_imu_data()):
                    # Get position and orientation
                    translation = pose.get_translation(sl.Translation()).get()
                    quaternion = sensors_data.get_imu_data().get_pose().get_orientation().get()
                    magnetometer_data = sensors_data.get_magnetometer_data()
                    magnetic_heading = magnetometer_data.magnetic_heading

                    # Engelleri tespit et
                    obstacles = detect_obstacles(depth, center_x, center_y)
                    
                    # Tespit edilen engelleri detected_objects'e ekle
                    for obstacle_pos in obstacles:
                        detected_objects.append({
                            'position': obstacle_pos,
                            'class_id': -1  # Engeller için özel sınıf ID'si
                        })

                    # Update map with robot position and detected objects
                    draw_robot_on_map(map_image, translation, magnetic_heading, trajectory, detected_objects)
                    cv2.imshow("2D Map", map_image)

                    # Display orientation information
                    magnetic_heading_info = (
                        f"Magnetic Heading: {magnetic_heading:.0f} "
                        f"({magnetometer_data.magnetic_heading_state}) "
                        f"[{magnetometer_data.magnetic_heading_accuracy:.0f}]"
                    )

                    yaw = quaternion_to_angle(quaternion, "yaw")
                    roll = quaternion_to_angle(quaternion, "roll")
                    pitch = quaternion_to_angle(quaternion, "pitch")

                    render_text(frame, f"Yaw: {yaw:.0f}", (frame.shape[1] - 200, 30))
                    render_text(frame, f"Roll: {roll:.0f}", (frame.shape[1] - 200, 60))
                    render_text(frame, f"Pitch: {pitch:.0f}", (frame.shape[1] - 200, 90))
                    render_text(frame, magnetic_heading_info, (frame.shape[1] - 1300, 30))

                    # Display spatial mapping state
                    state = zed.get_spatial_mapping_state()
                    render_text(frame, f"Spatial Mapping State: {state.name}", (frame.shape[1] - 1300, 60))

            # Display center depth
            depth_value = depth.get_value(center_x, center_y)[1]
            depth_info_text = f"Center Depth: {depth_value:.2f} m" if not np.isnan(depth_value) else "Couldn't Calculate..: NaN"
            render_text(frame, depth_info_text, (10, 50))
            cv2.circle(frame, (center_x, center_y), DEPTH_CENTER_RADIUS, DEPTH_CENTER_COLOR, -1)

            # Calculate and display FPS
            fps_current_time = time.time()
            fps = 1 / (fps_current_time - fps_previous_time)
            fps_previous_time = fps_current_time
            cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Update spatial map periodically
            if mesh_timer % 30 == 0:
                zed.request_spatial_map_async()

            if zed.get_spatial_map_request_status_async() == sl.ERROR_CODE.SUCCESS and mesh_timer > 0:
                zed.retrieve_spatial_map_async(mesh)

            mesh_timer += 1

            # Display frame
            frame_resized = cv2.resize(frame, (960, 540))
            cv2.imshow("ZED Camera", frame_resized)

            # Check for exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Cleanup
    cv2.destroyAllWindows()
    zed.disable_positional_tracking()
    zed.close()

if __name__ == "__main__":
    main()
