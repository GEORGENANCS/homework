import os
import sys
import cv2
import time
import traceback
import sqlite3
import json
from datetime import datetime
from PIL import Image

# 界面库
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QGroupBox,
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QUrl
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWebEngineWidgets import QWebEngineView

# 算法库
from transformers import pipeline

try:
    import torch
except Exception:
    torch = None


# --- 专注度计算逻辑 ---
EMOTION_SCORE_MAP = {
    "neutral": 90,
    "surprise": 95,
    "happy": 75,
    "sad": 60,
    "fear": 30,
    "angry": 20,
    "disgust": 20,
}

EMOTION_CN_MAP = {
    "neutral": "平静",
    "surprise": "惊讶",
    "happy": "开心",
    "sad": "低落",
    "fear": "担忧",
    "angry": "生气",
    "disgust": "厌恶",
}


def normalize_emotion_label(emotion_label):
    label = emotion_label.lower()
    for k in EMOTION_SCORE_MAP:
        if k in label:
            return k
    return "unknown"


def emotion_to_score(emotion_label):
    key = normalize_emotion_label(emotion_label)
    return EMOTION_SCORE_MAP.get(key, 60)


def emotion_to_cn(emotion_label):
    key = normalize_emotion_label(emotion_label)
    return EMOTION_CN_MAP.get(key, "未知")


def weighted_score_from_preds(preds):
    """
    使用概率加权而非仅 top1：
    score = Σ(情绪分值 * 置信度)
    """
    if not preds:
        return 0.0

    weighted_sum = 0.0
    conf_sum = 0.0
    for item in preds:
        label = item.get("label", "")
        conf = float(item.get("score", 0.0))
        weighted_sum += emotion_to_score(label) * conf
        conf_sum += conf

    if conf_sum <= 0:
        return 0.0
    return weighted_sum / conf_sum




class SQLiteStorage:
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self._create_tables()

    def _create_tables(self):
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS frame_summary (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                ts TEXT NOT NULL,
                face_count INTEGER NOT NULL,
                avg_focus_score REAL,
                smoothed_focus_score REAL
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS face_emotion_event (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                ts TEXT NOT NULL,
                face_idx INTEGER NOT NULL,
                bbox_x INTEGER NOT NULL,
                bbox_y INTEGER NOT NULL,
                bbox_w INTEGER NOT NULL,
                bbox_h INTEGER NOT NULL,
                top_emotion TEXT NOT NULL,
                top_confidence REAL NOT NULL,
                focus_score REAL NOT NULL,
                raw_preds_json TEXT NOT NULL
            )
            """
        )
        self.conn.commit()

    def insert_frame_summary(self, session_id, ts, face_count, avg_focus_score, smoothed_focus_score):
        self.conn.execute(
            """
            INSERT INTO frame_summary(session_id, ts, face_count, avg_focus_score, smoothed_focus_score)
            VALUES (?, ?, ?, ?, ?)
            """,
            (session_id, ts, face_count, avg_focus_score, smoothed_focus_score),
        )
        self.conn.commit()

    def insert_face_events(self, events):
        if not events:
            return

        normalized_events = []
        for e in events:
            normalized_events.append((
                str(e[0]),           # session_id
                str(e[1]),           # ts
                int(e[2]),           # face_idx
                int(e[3]),           # bbox_x
                int(e[4]),           # bbox_y
                int(e[5]),           # bbox_w
                int(e[6]),           # bbox_h
                str(e[7]),           # top_emotion
                float(e[8]),         # top_confidence
                float(e[9]),         # focus_score
                str(e[10]),          # raw_preds_json
            ))

        self.conn.executemany(
            """
            INSERT INTO face_emotion_event(
                session_id, ts, face_idx, bbox_x, bbox_y, bbox_w, bbox_h,
                top_emotion, top_confidence, focus_score, raw_preds_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            normalized_events,
        )
        self.conn.commit()

    def close(self):
        self.conn.close()

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)
    update_chart_signal = pyqtSignal(str, float)

    def __init__(self, video_source, session_id, db_path):
        super().__init__()
        self.video_source = video_source
        self.session_id = session_id
        self.storage = SQLiteStorage(db_path)
        self.is_running = True

        # 参数：偏向减少误检 + 提升流畅度
        self.process_every_n_frames = 4
        self.min_face_size = 48
        self.min_face_conf = 0.45
        self.ema_alpha = 0.35
        self.smoothed_score = None
        self.last_detections = []  # [(x,y,w,h,label_text), ...]

        face_model = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        self.face_detector = cv2.CascadeClassifier(face_model)
        if self.face_detector.empty():
            print("!!! Haar人脸检测器加载失败")
        else:
            print(f">>> 人脸检测器加载成功: {face_model}")

        print(">>> 正在加载表情识别模型 (首次运行需下载)...")
        device_id = 0 if (torch is not None and torch.cuda.is_available()) else -1
        print(f">>> 表情模型推理设备: {'CUDA:0' if device_id == 0 else 'CPU'}")
        try:
            self.emotion_pipe = pipeline(
                "image-classification",
                model="dima806/facial_emotions_image_detection",
                top_k=None,
                device=device_id,
            )
            print(">>> 表情模型加载成功")
        except Exception as e:
            print(f"!!! 表情模型加载失败: {e}")
            self.emotion_pipe = None

    def _is_reasonable_face_box(self, w, h):
        if w < self.min_face_size or h < self.min_face_size:
            return False
        ratio = w / float(h)
        return 0.7 <= ratio <= 1.4

    def _draw_label(self, frame, x, y, text):
        # 使用抗锯齿文本 + 自适应标签宽度，提升清晰度
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.58
        thickness = 2
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

        pad_x = 6
        pad_y = 4
        left = max(0, x)
        top = max(0, y - text_h - baseline - pad_y * 2)
        right = min(frame.shape[1] - 1, left + text_w + pad_x * 2)
        bottom = max(0, y)

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), -1)
        cv2.putText(
            frame,
            text,
            (left + pad_x, bottom - baseline - pad_y),
            font,
            font_scale,
            (15, 15, 15),
            thickness,
            cv2.LINE_AA,
        )

    def run(self):
        source = self.video_source
        if isinstance(source, str) and source.isdigit():
            source = int(source)

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"!!! 无法打开视频源: {source}，尝试回退到摄像头0")
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("!!! 摄像头0也无法打开，线程退出")
                return

        frame_count = 0

        while self.is_running and cap.isOpened():
            try:
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

                frame_count += 1
                should_infer = (frame_count % self.process_every_n_frames == 0)

                # 默认用上一轮检测结果叠加，保证每帧都能刷新显示
                current_detections = self.last_detections

                if should_infer:
                    frame_scores = []
                    face_events = []
                    current_detections = []

                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = self.face_detector.detectMultiScale(
                        gray,
                        scaleFactor=1.08,
                        minNeighbors=8,
                        minSize=(self.min_face_size, self.min_face_size),
                    )

                    for (x, y, w, h) in faces:
                        if not self._is_reasonable_face_box(w, h):
                            continue

                        face_img = frame[y:y + h, x:x + w]
                        if face_img.size == 0 or self.emotion_pipe is None:
                            continue

                        try:
                            pil_img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
                            preds = self.emotion_pipe(pil_img)
                            if not preds:
                                continue

                            top_pred = max(preds, key=lambda item: item.get("score", 0.0))
                            top_label = top_pred.get("label", "unknown")
                            top_conf = float(top_pred.get("score", 0.0))

                            # 低置信度直接过滤，减少“把背景当人脸”
                            if top_conf < self.min_face_conf:
                                continue

                            score = weighted_score_from_preds(preds)
                            frame_scores.append(score)

                            emotion_en = normalize_emotion_label(top_label)
                            label_text = f"{emotion_en} focus:{score:.0f}"
                            current_detections.append((x, y, w, h, label_text))
                            face_events.append((
                                self.session_id,
                                datetime.now().isoformat(timespec="seconds"),
                                int(len(face_events)),
                                int(x), int(y), int(w), int(h),
                                str(emotion_en),
                                float(top_conf),
                                float(score),
                                json.dumps(preds, ensure_ascii=False),
                            ))
                        except Exception as e:
                            print(f"识别出错: {e}")

                    self.last_detections = current_detections

                    avg_score = None
                    if frame_scores:
                        avg_score = sum(frame_scores) / len(frame_scores)
                        if self.smoothed_score is None:
                            self.smoothed_score = avg_score
                        else:
                            self.smoothed_score = self.ema_alpha * avg_score + (1 - self.ema_alpha) * self.smoothed_score

                        now_dt = datetime.now()
                        now_str = now_dt.strftime("%H:%M:%S")
                        self.update_chart_signal.emit(now_str, float(self.smoothed_score))
                    else:
                        now_dt = datetime.now()

                    # 数据持久化：明细 + 帧级汇总
                    self.storage.insert_face_events(face_events)
                    self.storage.insert_frame_summary(
                        session_id=self.session_id,
                        ts=now_dt.isoformat(timespec="seconds"),
                        face_count=len(face_events),
                        avg_focus_score=avg_score,
                        smoothed_focus_score=self.smoothed_score,
                    )

                for (x, y, w, h, label_text) in current_detections:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    self._draw_label(frame, x, y, label_text)

                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                qt_img = QImage(rgb_image.data, w, h, ch * w, QImage.Format_RGB888).copy()
                self.change_pixmap_signal.emit(qt_img)
            except Exception:
                print("!!! 视频线程发生异常:")
                traceback.print_exc()
                time.sleep(0.05)

        cap.release()

    def stop(self):
        self.is_running = False
        self.wait()
        self.storage.close()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("智慧课堂专注度分析系统")
        self.resize(1600, 900)

        self.setStyleSheet("""
            QMainWindow { background-color: #2b2b2b; }
            QGroupBox { color: white; font-weight: bold; border: 1px solid #555; margin-top: 10px; }
            QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top center; padding: 0 5px; }
        """)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)

        video_group = QGroupBox(" 实时视频监控 (Real-time Monitor) ")
        video_layout = QVBoxLayout()
        self.lbl_video = QLabel()
        self.lbl_video.setAlignment(Qt.AlignCenter)
        self.lbl_video.setStyleSheet("background-color: #1a1a1a; border-radius: 5px;")
        self.lbl_video.setText("正在初始化模型，请稍候...")
        video_layout.addWidget(self.lbl_video)
        video_group.setLayout(video_layout)

        data_group = QGroupBox(" 专注度时序分析 (Engagement Trend) ")
        data_layout = QVBoxLayout()
        self.web_view = QWebEngineView()
        self.web_view.setStyleSheet("background-color: white;")

        html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chart.html").replace('\\', '/')
        self.web_view.load(QUrl(f"file:///{html_path}"))
        self.page_loaded = False
        self.web_view.loadFinished.connect(self.on_page_loaded)

        data_layout.addWidget(self.web_view)
        data_group.setLayout(data_layout)

        main_layout.addWidget(video_group, 7)
        main_layout.addWidget(data_group, 3)

        video_source = self._resolve_video_source(os.environ.get("VIDEO_SOURCE", "0"))
        session_id = datetime.now().strftime("session_%Y%m%d_%H%M%S")
        db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "focus_data.db")
        print(f">>> 当前视频源: {video_source}")
        print(f">>> 当前会话ID: {session_id}")
        print(f">>> 数据库存储: {db_path}")
        self.thread = VideoThread(video_source, session_id, db_path)
        self.thread.change_pixmap_signal.connect(self.update_video_ui)
        self.thread.update_chart_signal.connect(self.update_chart_ui)
        self.thread.start()

    def _resolve_video_source(self, raw_source):
        source = str(raw_source).strip()

        if source.isdigit():
            return source

        if os.path.exists(source):
            return source

        script_dir = os.path.dirname(os.path.abspath(__file__))
        local_candidate = os.path.join(script_dir, source)
        if os.path.exists(local_candidate):
            return local_candidate

        if "chassroom" in source.lower():
            print("!!! 检测到可能的拼写错误: 'chassroom'，你可能想写 'classroom'")

        print(f"!!! 视频源不存在: {source}，已自动回退到摄像头 0")
        return "0"

    def on_page_loaded(self, ok):
        print(f">>> 页面加载: {'成功' if ok else '失败'}")
        self.page_loaded = bool(ok)

    def update_video_ui(self, qt_img):
        target_size = self.lbl_video.size()
        if target_size.width() <= 0 or target_size.height() <= 0:
            self.lbl_video.setPixmap(QPixmap.fromImage(qt_img))
            return
        scaled_img = qt_img.scaled(target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.lbl_video.setPixmap(QPixmap.fromImage(scaled_img))

    def update_chart_ui(self, time_str, score):
        if not self.page_loaded:
            return
        js = f"""
        if (typeof updateChart === 'function') {{
            updateChart('{time_str}', {score});
        }}
        """
        self.web_view.page().runJavaScript(js)

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
