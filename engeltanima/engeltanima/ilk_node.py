import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class RobotaksiEngelNode(Node):
    def __init__(self):
        super().__init__('engeltanima_yonetici')
        
        self.KRITIK_MESAFE = 1.5
        self.ACIL_FREN_MESAFE = 0.6
        self.ACI_ARALIGI = 15
        
        self.ALT_SARI = np.array([15, 100, 100])
        self.UST_SARI = np.array([35, 255, 255])
        
        self.lidar_sub = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        self.kamera_sub = self.create_subscription(Image, '/camera/image_raw', self.kamera_callback, 10)
        
        self.bridge = CvBridge()
        self.en_yakin_mesafe = float('inf')
        self.get_logger().info(f"İzleme Açısı: ±{self.ACI_ARALIGI} derece.")

    def lidar_callback(self, msg):
        veriler = msg.ranges
        toplam_veri = len(veriler)
        bir_dereceye_dusan_indeks = toplam_veri / 360.0
        
        indeks_araligi = int(self.ACI_ARALIGI * bir_dereceye_dusan_indeks)
        on_bolge = veriler[0:indeks_araligi] + veriler[-indeks_araligi:]
        
        gecerli_veriler = [r for r in on_bolge if msg.range_min < r < msg.range_max]
        
        if gecerli_veriler:
            self.en_yakin_mesafe = min(gecerli_veriler)
            
            if self.en_yakin_mesafe < self.ACIL_FREN_MESAFE:
                self.get_logger().error(f"{self.en_yakin_mesafe:.2f}m mesafede engel var!")
            elif self.en_yakin_mesafe < self.KRITIK_MESAFE:
                self.get_logger().warn(f"Engel yaklaşıyor ({self.en_yakin_mesafe:.2f}m). Kamera devrede.")

    def kamera_callback(self, data):
        try:
            frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except Exception as e:
            return

        if self.en_yakin_mesafe < self.KRITIK_MESAFE:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            maske = cv2.inRange(hsv, self.ALT_SARI, self.UST_SARI)
            maske = cv2.medianBlur(maske, 5)
            
            konturlar, _ = cv2.findContours(maske, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            statik_tespit = False
            for kontur in konturlar:
                if cv2.contourArea(kontur) > 400:
                    statik_tespit = True
                    x, y, w, h = cv2.boundingRect(kontur)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, "STATIK ENGEL", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if statik_tespit:
                self.get_logger().info("KONİ/BARİYER(Statik Engel) algılandı.")
            else:
                self.get_logger().info("TANIMLANAMAYAN/DİNAMİK ENGEL.")
                cv2.putText(frame, "BILINMEYEN ENGEL", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.putText(frame, f"Mesafe: {self.en_yakin_mesafe:.2f}m", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Robotaksi Analiz Sistemi", frame)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = RobotaksiEngelNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()