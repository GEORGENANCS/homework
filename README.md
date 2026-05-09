# ncsfocus

基于**面部表情**的课堂教学专注度分析系统（不包含头部姿态特征）。

## 模块
- `ncsfocus/app.py`：实时分析主程序（YOLO11 人脸检测 + 表情识别 + 专注度估计 + SQLite 落库）
- `ncsfocus/extract_frames.py`：视频抽帧工具
- `ncsfocus/quick_label_faces.py`：半自动标注人脸框与情绪（仅 YOLO11）
- `ncsfocus/label_focus.py`：帧级专注度标注（high/mid/low）
- `ncsfocus/fit_focus_mapping.py`：基于标注拟合 emotion->focus 映射
- `ncsfocus/generate_report.py`：离线报告生成
- `ncsfocus/focus_mapping.py`：专注度估计器

## 快速开始
1. 抽帧：`python extract_frames.py --video your.mp4 --out_dir frames_out --every_n 10`
2. 标注情绪：`python quick_label_faces.py --image_dir frames_out`
3. 标注专注度：`python label_focus.py --image_dir frames_out`
4. 拟合映射：`python fit_focus_mapping.py --face_csv face_emotions_manual.csv --frame_csv frame_focus_manual.csv`
5. 实时运行：`python main.py`
6. 生成报告：`python generate_report.py --db focus_data.db`

> 说明：项目已移除 Haar 检测分支，统一使用 YOLO11。
