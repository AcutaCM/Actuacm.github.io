import cv2
import torch
import numpy as np
import mysql.connector
from deepface import DeepFace
from djitellopy import Tello, TelloException
import sys
import os
import datetime
import time
import warnings
from typing import Tuple, Optional, List
import requests  # 用于将图像上传到web平台

warnings.filterwarnings("ignore", category=UserWarning)

# ========== 增强配置 ==========
class Config:
    ALERT = {
        "snapshot_dir": os.path.join("Resources", "Images"),
        "cooldown_sec": 8,
        "alarm_sound": "alarm.wav",
        "min_face_size": (120, 120),
        "web_upload_url": "http://your-web-platform/upload"  # 更改为实际的web平台上传URL
    }

    DATABASE = {
        "host": "localhost",
        "user": "root",
        "password": "root",
        "db_name": "face_db",
        "table": "face_embeddings",
        "embed_size": 128  # 与Facenet返回的特征维度匹配
    }

    FACE = {
        "detector": "retinaface",
        "model_name": "Facenet",  # Use Facenet
        "threshold": 0.65,
        "normalization": "base"
    }

    DRONE = {
        "cruise_speed": 40,
        "hover_height": 120,
        "approach_dist": 80,
        "max_rotate": 45,
        "pid_gains": (0.35, 0.005, 0.08),
        "safe_ratio": 0.65,
        # 巡航轨迹：前进后旋转（顺时针）
        "cruise_pattern": [
            ("forward", 80),
            ("rotate", 90)
        ],
        "obstacle_threshold": 50,  # 避障距离阈值(cm)
        "avoid_distance": 30       # 避障时平移的距离(cm)
    }

    MODEL = {
        "yolo_weights": "yolov5s.pt",
        "conf_thres": 0.7,
        "iou_thres": 0.4,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }


# ========== 核心系统组件 ==========

class FaceDatabase:
    def __init__(self):
        self.conn = mysql.connector.connect(
            host=Config.DATABASE["host"],
            user=Config.DATABASE["user"],
            password=Config.DATABASE["password"],
            database=Config.DATABASE["db_name"]
        )
        self.init_db()  # 初始化数据库表
        self.cache = self._load_embeddings()
        print(f"Loaded {len(self.cache)} registered faces")

    def init_db(self):
        """初始化数据库表，若不存在则创建"""
        try:
            cursor = self.conn.cursor()
            create_table_query = f"""
            CREATE TABLE IF NOT EXISTS {Config.DATABASE["table"]} (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                embedding BLOB NOT NULL
            )
            """
            cursor.execute(create_table_query)
            self.conn.commit()
            cursor.close()
            print("Database table initialized successfully.")
        except mysql.connector.Error as err:
            print("Error initializing database:", err)

    def _load_embeddings(self) -> List[Tuple[str, np.ndarray]]:
        cursor = self.conn.cursor()
        try:
            cursor.execute(f"SELECT name, embedding FROM {Config.DATABASE['table']}")
            rows = cursor.fetchall()
            embeddings = [(name, np.frombuffer(emb, np.float32)) for name, emb in rows]
            return embeddings
        except mysql.connector.Error as err:
            print("Error loading embeddings:", err)
            return []
        finally:
            cursor.close()

    def verify(self, embedding: np.ndarray) -> Tuple[str, float]:
        embedding = self._normalize(embedding)
        best_match = ("unknown", 0.0)
        for name, db_emb in self.cache:
            if embedding.shape == db_emb.shape:  # Ensure shapes are compatible
                similarity = np.dot(embedding, db_emb)
                if similarity > best_match[1]:
                    best_match = (name, similarity)
        # 返回识别到的名字及相似度
        return best_match[0], best_match[1]

    @staticmethod
    def _normalize(embedding: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(embedding)
        return embedding / norm if norm > 0 else embedding

    def close(self):
        self.conn.close()


class AlertSystem:
    def __init__(self):
        os.makedirs(Config.ALERT["snapshot_dir"], exist_ok=True)
        self.last_alert = 0

    def trigger(self):
        if time.time() - self.last_alert < Config.ALERT["cooldown_sec"]:
            return
        self._play_sound()
        self.last_alert = time.time()

    def capture(self, frame: np.ndarray, face_img: np.ndarray):
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        full_path = os.path.join(Config.ALERT["snapshot_dir"], f"full_{ts}.jpg")
        face_path = os.path.join(Config.ALERT["snapshot_dir"], f"face_{ts}.jpg")
        cv2.imwrite(full_path, frame)
        cv2.imwrite(face_path, face_img)
        print(f"Saved alert images: {full_path}, {face_path}")
        self.upload_to_web(face_path)

    def upload_to_web(self, face_path: str):
        with open(face_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(Config.ALERT["web_upload_url"], files=files)
            if response.status_code == 200:
                print("Image uploaded successfully")
            else:
                print("Failed to upload image")

    @staticmethod
    def _play_sound():
        if sys.platform == "darwin":
            os.system(f"afplay {Config.ALERT['alarm_sound']}")
        elif sys.platform == "win32":
            import winsound
            winsound.PlaySound(Config.ALERT["alarm_sound"], winsound.SND_FILENAME)
        else:
            print("\a")


class PIDController:
    def __init__(self, kp: float, ki: float, kd: float):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.reset()

    def reset(self):
        self.integral = 0.0
        self.last_err = 0.0

    def update(self, error: float, dt: float) -> float:
        self.integral += error * dt
        derivative = (error - self.last_err) / dt if dt > 0 else 0.0
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.last_err = error
        return output


class DroneController:
    def __init__(self):
        self.tello = Tello()
        self.pid_x = PIDController(*Config.DRONE["pid_gains"])
        self.pid_y = PIDController(*Config.DRONE["pid_gains"])
        self.current_target = None
        self.cruise_index = 0

    def connect(self):
        try:
            self.tello.connect()
            self.tello.streamon()
            print(f"Battery: {self.tello.get_battery()}%")
        except TelloException as e:
            print(f"Connection error: {str(e)}")
            self.emergency_stop()

    def takeoff(self):
        try:
            self.tello.takeoff()
            self.tello.set_speed(Config.DRONE["cruise_speed"])
            self._adjust_height(Config.DRONE["hover_height"])
        except TelloException as e:
            print(f"Takeoff error: {str(e)}")
            self.emergency_stop()

    def land(self):
        try:
            self.tello.land()
        except TelloException as e:
            print(f"Land error: {str(e)}")

    def avoid_obstacle(self) -> bool:
        try:
            distance = self.tello.get_distance_tof()
            if distance is not None and distance < Config.DRONE["obstacle_threshold"]:
                print(f"Obstacle detected at {distance} cm, avoiding...")
                self.tello.send_rc_control(0, 0, 0, 0)
                time.sleep(0.1)
                self.tello.move_right(Config.DRONE["avoid_distance"])
                return True
            return False
        except Exception as e:
            print("Error in obstacle detection:", e)
            return False

    def perform_cruise(self):
        # 执行巡航动作前检查障碍物
        if self.avoid_obstacle():
            return
        action, param = Config.DRONE["cruise_pattern"][self.cruise_index]
        try:
            if action == "forward":
                self.tello.move_forward(param)
            elif action == "back":
                self.tello.move_back(param)
            elif action == "rotate":
                self.tello.rotate_clockwise(param)
            self.cruise_index = (self.cruise_index + 1) % len(Config.DRONE["cruise_pattern"])
        except TelloException as e:
            print(f"Cruise error: {str(e)}")
            self.emergency_stop()

    def smooth_move_forward(self, distance: int):
        step_size = 20  # Move in smaller steps
        steps = distance // step_size
        remainder = distance % step_size

        for _ in range(steps):
            self.tello.move_forward(step_size)
            time.sleep(0.5)  # Small delay between steps

        if remainder > 0:
            self.tello.move_forward(remainder)

    def track_target(self, bbox: tuple, frame_size: tuple):
        # 跟踪目标时同时进行避障
        if self.avoid_obstacle():
            return
        img_h, img_w = frame_size[:2]
        x1, y1, x2, y2 = bbox
        t_cx, t_cy = (x1 + x2) // 2, (y1 + y2) // 2

        err_x = (t_cx - img_w / 2) / (img_w / 2)
        err_y = (t_cy - img_h / 2) / (img_h / 2)

        speed_x = int(self.pid_x.update(err_x, 0.1))
        speed_y = int(self.pid_y.update(err_y, 0.1))

        self.tello.send_rc_control(speed_y, speed_x, 0, 0)

    def _adjust_height(self, target: int):
        current = self.tello.get_height()
        diff = target - current
        if abs(diff) > 20:
            try:
                if diff > 0:
                    self.tello.move_up(diff)
                else:
                    self.tello.move_down(-diff)
            except TelloException as e:
                print(f"Height adjustment error: {str(e)}")
                self.emergency_stop()

    def emergency_stop(self):
        self.tello.send_rc_control(0, 0, 0, 0)
        self.land()
        raise RuntimeError("Emergency landing activated")


class VisionProcessor:
    def __init__(self):
        self.model = self._init_yolo()
        # 构建人脸模型（Facenet）
        self.face_model = DeepFace.build_model(Config.FACE["model_name"])

    def _init_yolo(self):
        model = torch.hub.load('ultralytics/yolov5', 'custom',
                               path=Config.MODEL["yolo_weights"],
                               device=Config.MODEL["device"])
        model.conf = Config.MODEL["conf_thres"]
        model.iou = Config.MODEL["iou_thres"]
        model.classes = [0]  # 只检测人体
        return model

    def detect_people(self, frame: np.ndarray) -> np.ndarray:
        results = self.model(frame[..., ::-1])
        return results.xyxy[0].cpu().numpy()

    def extract_face(self, frame: np.ndarray) -> Optional[np.ndarray]:
        # 确保传入的数据是有效的图像数组
        if not isinstance(frame, np.ndarray) or frame.size == 0:
            print("Invalid input to extract_face: not a valid image array")
            return None
        try:
            face_data = DeepFace.represent(
                frame,
                detector_backend=Config.FACE["detector"],
                model_name=Config.FACE["model_name"],
                enforce_detection=False
            )
            if face_data and 'embedding' in face_data[0]:
                return face_data[0]["embedding"]
            else:
                print("Face data does not contain 'embedding'")
                return None
        except Exception as e:
            print(f"Face error: {str(e)}")
        return None


class SurveillanceSystem:
    def __init__(self):
        self.drone = DroneController()
        self.vision = VisionProcessor()
        self.database = FaceDatabase()
        self.alerts = AlertSystem()
        self.cruise_mode = True

    def run(self):
        try:
            self.drone.connect()
            self.drone.takeoff()
            while True:
                frame = self._get_reliable_frame()
                if frame is None:
                    continue

                detections = self.vision.detect_people(frame)

                if self.cruise_mode:
                    self._handle_cruise_mode(detections)
                else:
                    self._handle_tracking_mode(frame, detections)

                self._display_interface(frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('l'):
                    print("Emergency landing triggered by user.")
                    self.drone.land()
                    break
                elif key == ord('q'):
                    break
        finally:
            self._safe_shutdown()

    def _get_reliable_frame(self) -> Optional[np.ndarray]:
        """增强视频流获取，包含重试机制"""
        for _ in range(5):
            frame_read = self.drone.tello.get_frame_read()
            if frame_read is not None:
                frame = frame_read.frame
                if frame is not None:
                    return cv2.resize(frame, (640, 480))
            time.sleep(0.1)
        print("Video stream unavailable")
        return None

    def _handle_cruise_mode(self, detections: np.ndarray):
        # 巡航模式下若检测到人体，则切换到跟踪模式
        if detections.shape[0] > 0:
            self.cruise_mode = False
            self.drone.current_target = detections[0][:4].astype(int)
            print("Switched to tracking mode")
        else:
            self.drone.perform_cruise()

    def _handle_tracking_mode(self, frame: np.ndarray, detections: np.ndarray):
        if detections.shape[0] == 0:
            self._return_to_cruise()
            return

        target = detections[0][:4].astype(int)
        x1, y1, x2, y2 = target
        width = x2 - x1

        # 若检测框宽度较小，则目标可能较远，尝试靠近
        if width < 150:
            print("Approaching target...")
            try:
                self.drone.smooth_move_forward(20)
            except Exception as e:
                print("Approach error:", e)
        else:
            # 保持目标居中跟踪
            self.drone.track_target(target, frame.shape)
            # 提取目标区域的人脸信息
            face_roi = frame[y1:y2, x1:x2]
            if not (isinstance(face_roi, np.ndarray) and face_roi.size > 0):
                print("Invalid face ROI, skipping face extraction")
                return
            embedding = self.vision.extract_face(face_roi)
            if embedding is not None:
                name, _ = self.database.verify(embedding)
                if name == "unknown":
                    self._handle_unknown_face(frame, face_roi)
                else:
                    print(f"Recognized: {name}")

    def _return_to_cruise(self):
        self.cruise_mode = True
        self.drone.tello.send_rc_control(0, 0, 0, 0)
        print("Returning to cruise mode")

    def _handle_unknown_face(self, frame: np.ndarray, face_img: np.ndarray):
        self.alerts.trigger()
        self.alerts.capture(frame, face_img)
        self.drone.tello.send_rc_control(0, 0, 0, 0)
        time.sleep(2)

    def _display_interface(self, frame: np.ndarray):
        display_frame = frame.copy()
        mode_text = f"Mode: {'CRUISE' if self.cruise_mode else 'TRACKING'}"
        cv2.putText(display_frame, mode_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("Drone Surveillance", display_frame)

    def _safe_shutdown(self):
        self.drone.land()
        self.drone.tello.streamoff()
        self.database.close()
        cv2.destroyAllWindows()
        print("System shutdown complete")


if __name__ == "__main__":
    system = SurveillanceSystem()
    system.run()