import sys
import cv2
import time
import os
import numpy as np
from datetime import datetime
from PIL import Image

# 界面库
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QGroupBox
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QUrl
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWebEngineWidgets import QWebEngineView

# 算法库
from ultralytics import YOLO
from transformers import pipeline # 替代 DeepFace

# --- 专注度计算逻辑 ---
def calculate_score(emotion_label):
    # 不同的模型输出标签可能略有不同，做模糊匹配
    label = emotion_label.lower()
    if 'neutral' in label: return 90    # 专注
    if 'surprise' in label: return 95   # 惊讶（高优）
    if 'happy' in label: return 75      # 快乐
    if 'sad' in label: return 90        # 悲伤
    if 'fear' in label: return 30       # 焦虑
    if 'angry' in label: return 20      # 愤怒
    if 'disgust' in label: return 20    # 厌恶
    return 60 # 默认

# --- 视频处理线程 ---
class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)     # 视频流信号
    update_chart_signal = pyqtSignal(str, float)  # 图表数据信号

    def __init__(self, video_source):
        super().__init__()
        self.video_source = video_source
        self.is_running = True
        
        # 1. 加载 YOLOv11
        print(">>> 正在加载 YOLOv11 模型...")
        try:
            self.detector = YOLO('yolo11n.pt') 
            print(">>> YOLOv11 加载成功")
        except Exception as e:
            print(f"!!! YOLO加载失败: {e}")

        # 2. 加载表情识别 (Hugging Face Transformers)
        print(">>> 正在加载表情识别模型 (首次运行需下载)...")
        try:
            # 使用基于 ViT 的轻量级表情模型
            self.emotion_pipe = pipeline("image-classification", model="dima806/facial_emotions_image_detection")
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
        
        # 计数器，用于跳帧处理
        frame_count = 0
        
        while self.is_running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # 视频结束则循环
                continue

            frame_count += 1
            # 每 5 帧处理一次算法，保证界面流畅
            if frame_count % 5 != 0:
                time.sleep(0.01) 
                continue

            # --- 核心处理流程 ---
            # 1. YOLO 检测 (只检测人 class=0)
            results = self.detector.predict(frame, classes=[0], verbose=False, conf=0.5)
            
            frame_scores = []
            
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # 简单的人脸定位策略：
                    # YOLO检测到的是全身，我们截取上半部分作为"人脸区域"输入表情模型
                    # (如果有 yolov8-face 模型更好，但在通用环境下这样最稳)
                    face_h = int((y2 - y1) * 0.5) # 取顶部 1/4 高度
                    real_y2 = y1 + face_h
                    
                    # 截取图像
                    face_img = frame[y1:real_y2, x1:x2]
                    
                    if face_img.size == 0 or self.emotion_pipe is None:
                        continue

                    try:
                        # 2. 表情识别
                        # OpenCV(BGR) 转 PIL(RGB)
                        pil_img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
                        
                        # 推理
                        preds = self.emotion_pipe(pil_img)
                        # preds 格式如: [{'label': 'neutral', 'score': 0.9}, ...]
                        top_emotion = preds[0]['label']
                        score = calculate_score(top_emotion)
                        frame_scores.append(score)

                        # 3. 绘制 AR 效果
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) # 全身框
                        cv2.rectangle(frame, (x1, y1), (x2, real_y2), (255, 0, 0), 1) # 头部框
                        
                        # 标签背景条
                        cv2.rectangle(frame, (x1, y1-25), (x1+150, y1), (0,255,0), -1)
                        cv2.putText(frame, f"{top_emotion} {score}", (x1+5, y1-5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                                  
                    except Exception as e:
                        print(f"识别出错: {e}")

            # 4. 更新图表数据
            avg_score = 0
            if frame_scores:
                avg_score = sum(frame_scores) / len(frame_scores)
                now_str = datetime.now().strftime("%H:%M:%S")
                self.update_chart_signal.emit(now_str, avg_score)

            # 5. 显示画面
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            # 缩放显示
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
        
        # 样式美化
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
        
        # 加载 HTML
       
        html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chart.html").replace('\\', '/')

        self.web_view.load(QUrl(f"file:///{html_path}"))
        self.page_loaded = False  
        # 第一处
        self.web_view.loadFinished.connect(self.on_page_loaded)
        
        data_layout.addWidget(self.web_view)
        data_group.setLayout(data_layout)

        # 布局比例 2:1
        main_layout.addWidget(video_group, 6)
        main_layout.addWidget(data_group, 4)

        # 启动线程 (参数 '0' 代表摄像头，改成 'video.mp4' 代表视频文件)
        self.thread = VideoThread(r"C:\Users\17387\my-app\focussys\ss\classroom1.mp4")
        self.thread.change_pixmap_signal.connect(self.update_video_ui)
        self.thread.update_chart_signal.connect(self.update_chart_ui)
        self.thread.start()

    def on_page_loaded(self, ok):
        print(f">>> 页面加载: {'成功' if ok else '失败'}")
        self.page_loaded = True
    # 测试 echarts 是否真的加载了
        self.web_view.page().runJavaScript(
        "console.log('echarts loaded:', typeof echarts);"
    )

    def update_chart_ui(self, time_str, score):
        if not self.page_loaded:
           return
        js = f"""
        if (typeof updateChart === 'function') {{
            updateChart('{time_str}', {score});
        }}
        """
        self.web_view.page().runJavaScript(js)
    def on_page_loaded(self, ok):
        print(f">>> 页面加载: {'成功' if ok else '失败'}")
        self.page_loaded = True
            #  第二处

    def update_video_ui(self, qt_img):
        self.lbl_video.setPixmap(QPixmap.fromImage(qt_img))

    # def update_chart_ui(self, time_str, score):
    #     # 调用 JS 更新图表
    #     js = f"updateChart('{time_str}', {score});"
    #     self.web_view.page().runJavaScript(js)  第三处

    def update_chart_ui(self, time_str, score):
        if not self.page_loaded:
           return
        js = f"updateChart('{time_str}', {score});"
        self.web_view.page().runJavaScript(js)

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())