import os
import sys
import cv2
import time
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


def emotion_to_score(emotion_label):
    label = emotion_label.lower()
    for k, v in EMOTION_SCORE_MAP.items():
        if k in label:
            return v
    return 60


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


# --- 视频处理线程 ---
class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)     # 视频流信号
    update_chart_signal = pyqtSignal(str, float)  # 图表数据信号

    def __init__(self, video_source):
        super().__init__()
        self.video_source = video_source
        self.is_running = True

        # 参数
        self.process_every_n_frames = 5
        self.min_face_size = 40
        self.ema_alpha = 0.35
        self.smoothed_score = None

        # 1) 加载 OpenCV Haar 人脸检测器（不依赖额外下载）
        face_model = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        self.face_detector = cv2.CascadeClassifier(face_model)
        if self.face_detector.empty():
            print("!!! Haar人脸检测器加载失败")
        else:
            print(f">>> 人脸检测器加载成功: {face_model}")

        # 2) 加载表情识别模型
        print(">>> 正在加载表情识别模型 (首次运行需下载)...")
        try:
            self.emotion_pipe = pipeline(
                "image-classification",
                model="dima806/facial_emotions_image_detection",
                top_k=None,
            )
            print(">>> 表情模型加载成功")
        except Exception as e:
            print(f"!!! 表情模型加载失败: {e}")
            self.emotion_pipe = None

    def run(self):
        # 兼容摄像头ID(0)或视频路径
        source = self.video_source
        if isinstance(source, str) and source.isdigit():
            source = int(source)

        cap = cv2.VideoCapture(source)
        frame_count = 0

        while self.is_running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 视频结束则循环
                continue

            frame_count += 1
            if frame_count % self.process_every_n_frames != 0:
                time.sleep(0.01)
                continue

            frame_scores = []

            # Haar 检测需要灰度图
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(self.min_face_size, self.min_face_size),
            )

            for (x, y, w, h) in faces:
                if w < self.min_face_size or h < self.min_face_size:
                    continue

                # 裁脸
                face_img = frame[y:y + h, x:x + w]
                if face_img.size == 0 or self.emotion_pipe is None:
                    continue

                try:
                    pil_img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
                    preds = self.emotion_pipe(pil_img)

                    score = weighted_score_from_preds(preds)
                    top_pred = max(preds, key=lambda item: item.get("score", 0.0)) if preds else {"label": "unknown", "score": 0.0}
                    label = top_pred.get("label", "unknown")
                    conf = float(top_pred.get("score", 0.0))

                    # 低置信度结果降权
                    if conf < 0.5:
                        score *= 0.7

                    frame_scores.append(score)

                    # 绘制框 + 标签
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.rectangle(frame, (x, y - 24), (x + 170, y), (0, 255, 0), -1)
                    cv2.putText(
                        frame,
                        f"{label}:{conf:.2f} {score:.0f}",
                        (x + 4, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.48,
                        (0, 0, 0),
                        1,
                    )
                except Exception as e:
                    print(f"识别出错: {e}")

            # 计算当帧均值 + EMA 平滑
            if frame_scores:
                avg_score = sum(frame_scores) / len(frame_scores)
                if self.smoothed_score is None:
                    self.smoothed_score = avg_score
                else:
                    self.smoothed_score = self.ema_alpha * avg_score + (1 - self.ema_alpha) * self.smoothed_score

                now_str = datetime.now().strftime("%H:%M:%S")
                self.update_chart_signal.emit(now_str, float(self.smoothed_score))

            # 显示画面
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            qt_img = QImage(rgb_image.data, w, h, ch * w, QImage.Format_RGB888)
            p = qt_img.scaled(800, 600, Qt.KeepAspectRatio)
            self.change_pixmap_signal.emit(p)

        cap.release()

    def stop(self):
        self.is_running = False
        self.wait()


# --- 主窗口 ---
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("智慧课堂专注度分析系统")
        self.resize(1200, 750)

        self.setStyleSheet("""
            QMainWindow { background-color: #2b2b2b; }
            QGroupBox { color: white; font-weight: bold; border: 1px solid #555; margin-top: 10px; }
            QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top center; padding: 0 5px; }
        """)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)

        # 左侧：视频监控
        video_group = QGroupBox(" 实时视频监控 (Real-time Monitor) ")
        video_layout = QVBoxLayout()
        self.lbl_video = QLabel()
        self.lbl_video.setAlignment(Qt.AlignCenter)
        self.lbl_video.setStyleSheet("background-color: #1a1a1a; border-radius: 5px;")
        self.lbl_video.setText("正在初始化模型，请稍候...")
        video_layout.addWidget(self.lbl_video)
        video_group.setLayout(video_layout)

        # 右侧：数据分析
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

        # 布局比例 2:1
        main_layout.addWidget(video_group, 6)
        main_layout.addWidget(data_group, 4)

        # 支持环境变量传入视频源：
        # - 数字(如 "0") -> 摄像头
        # - 文件路径 -> 视频文件
        # 如果文件不存在，会自动回退到摄像头0
        video_source = self._resolve_video_source(os.environ.get("VIDEO_SOURCE", "0"))
        print(f">>> 当前视频源: {video_source}")
        self.thread = VideoThread(video_source)
        self.thread.change_pixmap_signal.connect(self.update_video_ui)
        self.thread.update_chart_signal.connect(self.update_chart_ui)
        self.thread.start()

    def _resolve_video_source(self, raw_source):
        source = str(raw_source).strip()

        # 摄像头 id
        if source.isdigit():
            return source

        # 绝对路径或当前工作目录下路径
        if os.path.exists(source):
            return source

        # 脚本目录下路径
        script_dir = os.path.dirname(os.path.abspath(__file__))
        local_candidate = os.path.join(script_dir, source)
        if os.path.exists(local_candidate):
            return local_candidate

        # 常见拼写错误提醒
        if "chassroom" in source.lower():
            print("!!! 检测到可能的拼写错误: 'chassroom'，你可能想写 'classroom'")

        print(f"!!! 视频源不存在: {source}，已自动回退到摄像头 0")
        return "0"

    def on_page_loaded(self, ok):
        print(f">>> 页面加载: {'成功' if ok else '失败'}")
        self.page_loaded = bool(ok)

    def update_video_ui(self, qt_img):
        self.lbl_video.setPixmap(QPixmap.fromImage(qt_img))

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
