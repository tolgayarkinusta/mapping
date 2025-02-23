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
manual_mode = False
# Load a model
model = YOLO("balonx50.engine")
#model.to("cuda")


bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

# Check if the script is running with root privileges
#if os.geteuid() != 0:
#    print("This script requires elevated privileges. Please run it with `sudo` or as root.")
#    #sys.exit(1)

from MainSystem import USVController

controller = USVController("COM10", baud=57600)
print("Arming vehicle...")
controller.arm_vehicle()
print("Vehicle armed!")
print("Setting mode...")
controller.set_mode("MANUAL")
print("Mode set!")

# Constants
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1
COLOR_RED = (0, 0, 255)
THICKNESS = 2
DEPTH_CENTER_COLOR = (255, 0, 0)
DEPTH_CENTER_RADIUS = 5
width = None  # Başlangıçta tanımlayın

def initialize_camera():
    # ZED kamera nesnesi oluştur
    zed = sl.Camera()
    # ZED başlatma parametreleri ayarla
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720  # 720p çözünürlük
    init_params.camera_fps = 30  # 30 FPS
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL # depth mode best quality at neural_plus
    init_params.coordinate_units = sl.UNIT.METER #using metric system
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE # default for the opencv
    init_params.depth_minimum_distance = 0.20
    init_params.depth_maximum_distance = 40
    init_params.camera_disable_self_calib = True
    init_params.depth_stabilization = 80 #titreme azaltıcı
    init_params.sensors_required = False # true yaparsan imu açılmadan kamera açılmaz
    init_params.enable_image_enhancement = True #true was always the default
    init_params.async_grab_camera_recovery = False #set true if u want to keep processing if cam gets shutdown
    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        raise Exception("Failed to open ZED camera. Exiting.")
    return zed

# Render text over the frame
def render_text(frame, text, position):
    cv2.putText(frame, text, position, FONT, FONT_SCALE, COLOR_RED, THICKNESS)

# Basic class to handle the timestamp of the different sensors to know if it is a new sensors_data or an old one
class TimestampHandler:
    def __init__(self):
        self.t_imu = sl.Timestamp()
    # check if the new timestamp is higher than the reference one, and if yes, save the current as reference
    def is_new(self, sensor):
        if (isinstance(sensor, sl.IMUData)):
            new_ = (sensor.timestamp.get_microseconds() > self.t_imu.get_microseconds())
            if new_:
                self.t_imu = sensor.timestamp
            return new_

def main():
    zed = initialize_camera()
    global width
    # Kamera çözünürlüğünü al
    camera_info = zed.get_camera_information()
    width = camera_info.camera_configuration.resolution.width
    print(width)
    height = camera_info.camera_configuration.resolution.height
    print(height)
    # Görüntüde merkez noktasını hesapla
    center_x = width // 2
    center_y = height // 2
    print("Kamera çözünürlüğü: ", width, "x", height)
    print("Görüntü orta noktası: ", (center_x, center_y))

    # Used to store the sensors timestamp to know if the sensors_data is a new one or not
    ts_handler = TimestampHandler()

    # Görüntü ve derinlik verilerini almak için Mat nesneleri oluştur
    image = sl.Mat()
    depth = sl.Mat()
    # Sensör verisi al
    sensors_data = sl.SensorsData()

    # For FPS calculation
    fps_previous_time = 0

    # Sonsuz bir döngüde görüntü akışı
    while True:
        # Kameradan bir yeni kare alın
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            # Görüntü ve derinlik verilerini al
            zed.retrieve_image(image, sl.VIEW.LEFT)
            zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
            # OpenCV formatına dönüştür
            frame = cv2.cvtColor(image.get_data(), cv2.COLOR_BGRA2BGR)  # BGRA -> BGR
            results = model(frame, conf=0.50)[0]

            # yolo sonuçlarının sv.Detections formatına dönüştürülmesi
            detections = sv.Detections.from_ultralytics(results)

            # tespitlerin sınırlarının ve etiketlerinin oluşturulması
            frame = bounding_box_annotator.annotate(scene=frame, detections=detections)
            frame = label_annotator.annotate(scene=frame, detections=detections)

            # tespitlerin koordinatlarının sınıflarının alınması
            coordinates = detections.xyxy.tolist()
            class_ids = detections.class_id.tolist()

            # Her tespit kutusunun sağ üst köşesine derinlik değerini yazdırmak için:
            for box in coordinates:
                x1, y1, x2, y2 = map(int, box)  # tamsayıya çeviriyoruz
                # Sağ üst köşe koordinatları: (x2, y1)
                depth_val = depth.get_value(x2, y1)[1]  # İkinci değer derinlik (metre cinsinden)
                # Eğer depth değeri geçerliyse (NaN değilse) yazdır
                if not np.isnan(depth_val):
                    text = f"{depth_val:.2f} m"
                    # Yazıyı kutunun sağ üst köşesine ekleyelim; konum ayarını isteğinize göre değiştirebilirsiniz
                    cv2.putText(frame, text, (x2 - 60, y1 + 20), FONT, 0.7, COLOR_RED, 2)

            red_detected = False
            green_detected = False
            yellow_detected = False

            red_positions = []
            green_positions = []
            yellow_positions = []


            for i, class_id in enumerate(class_ids):
                if class_id == 3:  # Kırmızı
                    red_detected = True
                    red_positions.append(coordinates[i])
                elif class_id == 4:  # Sarı
                    yellow_detected = True
                    yellow_positions.append(coordinates[i])
                elif class_id == 2:  # Yeşil
                    green_detected = True
                    green_positions.append(coordinates[i])


            # Tuş kontrolü: 'm' tuşu ile modlar arasında geçiş yapılır.
            key = cv2.waitKey(1) & 0xFF
            if key == ord('m'):
                global manual_mode
                manual_mode = not manual_mode
                if manual_mode:
                    print("Manuel mod aktif. Otomatik sürüş durdu.")
                else:
                    print("Otomatik mod aktif. Manuel kontrol devre dışı.")
                # Küçük bir gecikme, tuşun sürekli algılanmasını önlemek için
                time.sleep(0.2)

            # Manuel mod aktifse, WASD tuşlarıyla kontrol yapılır.
            if manual_mode:
                cv2.putText(frame, "MANUEL MOD", (50, 50), FONT, 1, (0, 255, 255), 2)
                if key == ord('w'):
                    # İleri hareket: her iki motor ileri
                    controller.set_servo(5, 1550,1)
                    controller.set_servo(6, 1550,1)
                elif key == ord('s'):
                    # Geri hareket: her iki motor geri
                    controller.set_servo(5, 1440,1)
                    controller.set_servo(6, 1440,1)
                elif key == ord('a'):
                    # Sola dönüş: sol motor yavaş, sağ motor hızlı
                    controller.set_servo(5, 1440,1)
                    controller.set_servo(6, 1550,1)
                elif key == ord('d'):
                    # Sağa dönüş: sol motor hızlı, sağ motor yavaş
                    controller.set_servo(5, 1550,1)
                    controller.set_servo(6, 1440,1)
                else:
                    # Tuşlara basılmadığında motorlar nötr konumda kalır
                    controller.set_servo(5, 1500,1)
                    controller.set_servo(6, 1500,1)
            else:


                # retrieve the current sensors sensors_data
                if zed.get_sensors_data(sensors_data, sl.TIME_REFERENCE.IMAGE): #time_reference.image for synchorinzed timestamps
                    # Check if the data has been updated since the last time
                    # IMU is the sensor with the highest rate
                    if ts_handler.is_new(sensors_data.get_imu_data()):

                        # Filtered orientation quaternion
                        quaternion = sensors_data.get_imu_data().get_pose().get_orientation().get()
                        # Access the magnetometer data
                        magnetometer_data = sensors_data.get_magnetometer_data()

                        # Access the magnetic heading and state
                        magnetic_heading_info = (
                            f"Magnetic Heading: {magnetometer_data.magnetic_heading:.0f} "
                            f"({magnetometer_data.magnetic_heading_state}) "
                            f"[{magnetometer_data.magnetic_heading_accuracy:.0f}]"
                        )

                        render_text(frame, magnetic_heading_info, (frame.shape[1] - 1000, 30))

                # Orta noktanın derinlik bilgisini al
                #depth_value = depth.get_value(center_x, center_y)[1]  # Sadece metre cinsinden değeri al

            # Calculate FPS
            fps_current_time = time.time()
            fps = 1 / (fps_current_time - fps_previous_time)
            fps_previous_time = fps_current_time
            cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Görüntüyü göster
            frame_resized = cv2.resize(frame, (960, 540))  # Resize the frame to desired dimensions960, 540
            cv2.imshow("ZED Camera", frame_resized)

            k = cv2.waitKey(1)

            if k % 256 == 27:
                print("Esc tuşuna basıldı.. Kapatılıyor..")
                controller.stop_motors()
                break

    # Kaynakları serbest bırak ve kamerayı kapat
    cv2.destroyAllWindows()
    zed.close()

if __name__ == "__main__":
    main()
