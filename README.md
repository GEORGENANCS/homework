# ncsfocus

基于**面部表情**的课堂教学专注度分析系统（不包含头部姿态特征）。

## 模块
- `ncsfocus/app.py`：实时分析主程序（YOLO11 人脸检测 + 表情识别 + 专注度估计 + SQLite 落库）
- `ncsfocus/quick_label_faces.py`：半自动标注（仅 YOLO11）
- `ncsfocus/fit_focus_mapping.py`：基于标注拟合 emotion->focus 映射
- `ncsfocus/generate_report.py`：离线报告生成
- `ncsfocus/focus_mapping.py`：专注度估计器

## 快速开始
1. 运行标注（可选）
2. 拟合映射：`python fit_focus_mapping.py --face_csv ... --frame_csv ...`
3. 实时运行：`python main.py`
4. 生成报告：`python generate_report.py --db focus_data.db`

> 说明：项目已移除 Haar 检测分支，统一使用 YOLO11。
