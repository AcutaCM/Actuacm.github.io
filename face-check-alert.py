import cv2
import torch
import numpy as np
import mysql.connector
from deepface import DeepFace
from djitellopy import Tello
import sys
import os
import datetime
import time
# 新增预警配置参数
ALERT_CONFIG = {
    "snapshot_dir": "alerts/snapshots",
    "cooldown_time": 5  # 拍照冷却时间（秒）
}

# 合并配置参数
DATABASE_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "root",
    "database": "face_db",
    "table": "face_embeddings"
}

FACE_CONFIG = {
    "threshold": 0.55,
    "haar_scale": 1.08,
    "min_neighbors": 5,
    "target_size": (160, 160)
}
MODEL_CONFIG = {
    "model_path": "PersonTrack-master/best.pt",
    "repo_path": "PersonTrack-master",  # 本地克隆的YOLOv5仓库路径
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "half_precision": True  # 启用半精度推理
}

class AlertSystem:
    def __init__(self):
        # 创建警报目录
        os.makedirs(ALERT_CONFIG["snapshot_dir"], exist_ok=True)
        self.last_alert_time = 0

    def take_snapshot(self, frame, face_roi):
        """保存陌生人快照"""
        current_time = time.time()
        if current_time - self.last_alert_time > ALERT_CONFIG["cooldown_time"]:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"stranger_{timestamp}.jpg"
            full_path = os.path.join(ALERT_CONFIG["snapshot_dir"], filename)

            # 保存完整画面和人脸区域
            cv2.imwrite(full_path, frame)
            cv2.imwrite(full_path.replace(".jpg", "_face.jpg"), face_roi)

            print(f"警报快照已保存：{filename}")
            self.last_alert_time = current_time

class FaceDatabase:

    def __init__(self):
        self.conn = mysql.connector.connect(
            host=DATABASE_CONFIG["host"],
            user=DATABASE_CONFIG["user"],
            password=DATABASE_CONFIG["password"],
            database=DATABASE_CONFIG["database"]
        )

    def _debug_embedding(self, embedding: np.ndarray):
        """特征向量调试"""
        print(f"当前特征向量: \n维度: {embedding.shape}\n样例值: {embedding[:5]}...")

    def verify_face(self, embedding: np.ndarray) -> tuple:
        cursor = self.conn.cursor()
        cursor.execute(f"SELECT name, embedding FROM {DATABASE_CONFIG['table']}")

        max_similarity = 0.0
        best_match = "stranger"
        self._debug_embedding(embedding)  # 调试输出

        for (name, db_emb) in cursor:
            db_embedding = np.frombuffer(db_emb, dtype=np.float32)
            if db_embedding.shape != embedding.shape:
                print(f"维度不匹配: 数据库{db_embedding.shape} vs 当前{embedding.shape}")
                continue

            # 余弦相似度计算优化
            norm = np.linalg.norm(embedding) * np.linalg.norm(db_embedding)
            similarity = np.dot(embedding, db_embedding) / norm if norm != 0 else 0.0

            print(f"比对 {name}: 相似度 {similarity:.4f}")  # 调试输出

            if similarity > max_similarity:
                max_similarity = similarity
                best_match = name

        print(f"最高相似度: {max_similarity:.4f}")  # 调试输出
        return (best_match, max_similarity) if max_similarity > FACE_CONFIG["threshold"] else ("stranger", 0.0)

    def close(self):
        self.conn.close()


def load_yolov5():
    """优化后的模型加载函数"""
    # 记录加载时间用于调试
    start_time = time.time()

    # 设置本地仓库路径
    sys.path.insert(0, MODEL_CONFIG["repo_path"])

    # 从本地加载模型（确保已经执行 git clone https://github.com/ultralytics/yolov5 到指定路径）
    model = torch.hub.load(MODEL_CONFIG["repo_path"],
                           'custom',
                           path=MODEL_CONFIG["model_path"],
                           source='local',
                           force_reload=False,  # 禁止强制重载
                           device=MODEL_CONFIG["device"])

    # 应用优化配置
    model.conf = 0.6
    model.classes = [0]

    # 启用半精度（如果支持）
    if MODEL_CONFIG["half_precision"] and MODEL_CONFIG["device"] == "cuda":
        model = model.half()

    # 预热模型
    _ = model(torch.zeros(1, 3, 640, 640).to(MODEL_CONFIG["device"]))

    print(f"模型加载完成，耗时：{time.time() - start_time:.2f}秒")
    return model


def preprocess_face(face_img: np.ndarray) -> np.ndarray:
    """标准化预处理流程"""
    # 颜色空间转换 BGR -> RGB
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

    # 尺寸调整
    face_img = cv2.resize(face_img, FACE_CONFIG["target_size"])

    # 归一化处理
    face_img = face_img.astype(np.float32) / 255.0
    return face_img


def detect_and_track():
    global face_db
    tello = Tello()
    alert_system = AlertSystem()  # 初始化预警系统

    try:
        tello.connect()
        tello.streamon()
        print(f"电池电量: {tello.get_battery()}%")
        tello.takeoff()
        tello.move_up(30)
        yolo_model = load_yolov5()
        face_db = FaceDatabase()
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        alert_active = False  # 预警状态标志
        last_alert_time = 0

        while True:
            frame = tello.get_frame_read().frame
            if frame is None:
                continue

            frame = cv2.resize(frame, (640, 480))
            results = yolo_model(frame)
            stranger_detected = False

            for det in results.xyxy[0].cpu().numpy():
                if det[4] < 0.6:
                    continue

                x1, y1, x2, y2 = map(int, det[:4])
                person_roi = frame[y1:y2, x1:x2]

                gray = cv2.cvtColor(person_roi, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=FACE_CONFIG["haar_scale"],
                    minNeighbors=FACE_CONFIG["min_neighbors"]
                )

                for (fx, fy, fw, fh) in faces:
                    try:
                        face_img = person_roi[fy:fy + fh, fx:fx + fw]
                        if face_img.size == 0:
                            continue

                        processed_face = preprocess_face(face_img)
                        embedding_obj = DeepFace.represent(
                            processed_face,
                            model_name="Facenet",
                            enforce_detection=False,
                            detector_backend="skip"
                        )
                        embedding = np.array(embedding_obj[0]['embedding'], dtype=np.float32)

                        name, confidence = face_db.verify_face(embedding)
                        result = name if confidence > FACE_CONFIG["threshold"] else "Stranger"

                        # 陌生人预警处理
                        if result.lower() == "stranger":
                            stranger_detected = True
                            alert_system.take_snapshot(frame, face_img)
                            cv2.putText(frame, "ALERT! STRANGER DETECTED", (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                        # 可视化标记
                        color = (0, 255, 0) if name != "stranger" else (0, 0, 255)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, f"{name} ({confidence:.2f})", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                    except Exception as e:
                        print(f"人脸处理异常: {str(e)}")

            # 用户交互处理
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('a'):  # 触发声音警报
                print("声音警报已激活")
            elif key == ord('s'):  # 无人机闪避动作
                tello.move_left(50)
                tello.rotate_clockwise(180)
                print("执行闪避动作")
            elif key == ord('l'):  # 开启LED警示灯
                tello.turn_motor_on()
                time.sleep(1)
                tello.turn_motor_off()
                print("LED警示灯已激活")

            cv2.imshow("Drone View", frame)

    finally:
        face_db.close()
        tello.land()
        tello.streamoff()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    detect_and_track()
    detect_and_track()
    detect_and_track()