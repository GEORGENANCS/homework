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
    QPushButton,
    QComboBox,
    QFileDialog,
    QMessageBox,
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QUrl
from PyQt5.QtGui import QImage, QPixmap, QDesktopServices
from PyQt5.QtWebEngineWidgets import QWebEngineView

# 算法库
from transformers import pipeline
from ultralytics import YOLO

# 兼容两种运行方式：
# 1) 包方式: python -m ncsfocus.app
# 2) 脚本方式: python app.py
if __package__:
    from .focus_mapping import FocusEstimator, normalize_emotion_label
else:
    from focus_mapping import FocusEstimator, normalize_emotion_label

import torch


# --- 专注度计算逻辑 ---
EMOTION_CN_MAP = {
    "neutral": "平静",
    "surprise": "惊讶",
    "happy": "开心",
    "sad": "低落",
    "fear": "担忧",
    "angry": "生气",
    "disgust": "厌恶",
}


def emotion_to_cn(emotion_label):
    key = normalize_emotion_label(emotion_label)
    return EMOTION_CN_MAP.get(key, "未知")


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
    status_signal = pyqtSignal(str)

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

        mapping_path = os.environ.get("FOCUS_MAPPING_PATH", "focus_mapping_calibrated.json")
        self.focus_estimator, loaded = FocusEstimator.from_json_path(mapping_path)
        if loaded:
            print(f">>> 专注度映射: 已加载标定文件 {mapping_path}")
        else:
            print(">>> 专注度映射: 使用默认规则映射")

        self.face_detector_name = "yolo11"
        yolo_model = os.environ.get("YOLO_FACE_MODEL", "yolo11n-face.pt")
        self.yolo_conf = float(os.environ.get("YOLO_FACE_CONF", "0.25"))
        self.yolo_iou = float(os.environ.get("YOLO_FACE_IOU", "0.45"))
        self.face_detector = YOLO(yolo_model)
        print(f">>> 人脸检测器: YOLO11 ({yolo_model}), conf={self.yolo_conf}, iou={self.yolo_iou}")

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

    def _detect_faces(self, frame):
        results = self.face_detector.predict(
            source=frame,
            conf=self.yolo_conf,
            iou=self.yolo_iou,
            verbose=False,
        )
        if not results:
            return []

        boxes = results[0].boxes
        if boxes is None or boxes.xyxy is None:
            return []

        faces = []
        for box in boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = [int(v) for v in box[:4]]
            w = max(0, x2 - x1)
            h = max(0, y2 - y1)
            if w <= 0 or h <= 0:
                continue
            if not self._is_reasonable_face_box(w, h):
                continue
            faces.append((x1, y1, w, h))
        return faces

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
            msg = f"无法打开视频源: {source}"
            print(f"!!! {msg}")
            self.status_signal.emit(msg)
            return

        frame_count = 0

        while self.is_running and cap.isOpened():
            try:
                ret, frame = cap.read()
                if not ret:
                    if isinstance(source, int):
                        continue
                    break

                frame_count += 1
                should_infer = (frame_count % self.process_every_n_frames == 0)

                # 默认用上一轮检测结果叠加，保证每帧都能刷新显示
                current_detections = self.last_detections

                if should_infer:
                    frame_scores = []
                    face_events = []
                    current_detections = []

                    faces = self._detect_faces(frame)

                    for (x, y, w, h) in faces:
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

                            score = self.focus_estimator.weighted_score_from_preds(preds)
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
        self.status_signal.emit("视频处理已停止")

    def stop(self):
        self.is_running = False
        self.wait()
        self.storage.close()



class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("智慧课堂专注度分析系统")
        self.resize(1600, 900)

        self.thread = None
        self.selected_video_file = ""
        self.current_source = "0"
        self.db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "focus_data.db")

        self.setStyleSheet("""
            QMainWindow { background-color: #2b2b2b; }
            QGroupBox { color: white; font-weight: bold; border: 1px solid #555; margin-top: 10px; }
            QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top center; padding: 0 5px; }
            QPushButton { padding: 8px 12px; }
            QComboBox { padding: 6px 8px; background: #fff; }
        """)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        root_layout = QVBoxLayout()
        central_widget.setLayout(root_layout)

        # 顶部控制栏
        ctrl_group = QGroupBox(" 控制面板 (Control Panel) ")
        ctrl_layout = QHBoxLayout()
        self.source_combo = QComboBox()
        self.source_combo.addItems(["摄像头 0", "视频文件"])

        self.btn_pick_file = QPushButton("选择视频文件")
        self.btn_pick_file.clicked.connect(self.pick_video_file)

        self.btn_start = QPushButton("开始分析")
        self.btn_start.clicked.connect(self.start_analysis)

        self.btn_stop = QPushButton("停止分析")
        self.btn_stop.clicked.connect(self.stop_analysis)
        self.btn_stop.setEnabled(False)

        self.btn_open_report_dir = QPushButton("打开报告目录")
        self.btn_open_report_dir.clicked.connect(self.open_report_dir)

        self.btn_generate_report = QPushButton("生成离线报告")
        self.btn_generate_report.clicked.connect(self.generate_report)

        self.status_label = QLabel("状态：就绪")
        self.status_label.setStyleSheet("color: #ddd;")

        ctrl_layout.addWidget(self.source_combo)
        ctrl_layout.addWidget(self.btn_pick_file)
        ctrl_layout.addWidget(self.btn_start)
        ctrl_layout.addWidget(self.btn_stop)
        ctrl_layout.addWidget(self.btn_generate_report)
        ctrl_layout.addWidget(self.btn_open_report_dir)
        ctrl_layout.addWidget(self.status_label, 1)
        ctrl_group.setLayout(ctrl_layout)
        root_layout.addWidget(ctrl_group)

        # 主显示区域
        main_layout = QHBoxLayout()

        video_group = QGroupBox(" 实时视频监控 (Real-time Monitor) ")
        video_layout = QVBoxLayout()
        self.lbl_video = QLabel()
        self.lbl_video.setAlignment(Qt.AlignCenter)
        self.lbl_video.setStyleSheet("background-color: #1a1a1a; border-radius: 5px;")
        self.lbl_video.setText("请选择视频源并点击“开始分析”")
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
        root_layout.addLayout(main_layout)

    def pick_video_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择视频文件",
            os.path.dirname(os.path.abspath(__file__)),
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)",
        )
        if file_path:
            self.selected_video_file = file_path
            self.source_combo.setCurrentText("视频文件")
            self.status_label.setText(f"状态：已选择视频 {os.path.basename(file_path)}")

    def _build_source(self):
        if self.source_combo.currentText() == "摄像头 0":
            return "0"
        if not self.selected_video_file:
            raise ValueError("请选择视频文件")
        return self.selected_video_file

    def start_analysis(self):
        if self.thread is not None and self.thread.isRunning():
            QMessageBox.information(self, "提示", "分析已在运行")
            return
        try:
            source = self._build_source()
        except ValueError as e:
            QMessageBox.warning(self, "缺少输入", str(e))
            return

        session_id = datetime.now().strftime("session_%Y%m%d_%H%M%S")
        self.current_source = source
        self.thread = VideoThread(source, session_id, self.db_path)
        self.thread.change_pixmap_signal.connect(self.update_video_ui)
        self.thread.update_chart_signal.connect(self.update_chart_ui)
        self.thread.status_signal.connect(self.update_status)
        self.thread.finished.connect(self.on_analysis_finished)
        self.thread.start()

        self.status_label.setText(f"状态：运行中 | session={session_id}")
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)

    def stop_analysis(self):
        if self.thread is None:
            return
        self.thread.stop()

    def on_analysis_finished(self):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

    def update_status(self, msg):
        self.status_label.setText(f"状态：{msg}")

    def open_report_dir(self):
        report_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "report_out")
        os.makedirs(report_dir, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(report_dir))

    def generate_report(self):
        try:
            from generate_report import main as report_main
            old = list(sys.argv)
            sys.argv = ["generate_report.py", "--db", self.db_path, "--out_dir", os.path.join(os.path.dirname(os.path.abspath(__file__)), "report_out")]
            report_main()
            sys.argv = old
            QMessageBox.information(self, "完成", "报告已生成到 report_out")
        except Exception as e:
            QMessageBox.critical(self, "报告失败", str(e))


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
