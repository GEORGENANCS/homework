# ncsfocus

毕业设计项目：基于课堂视频的专注度与情绪分析原型。

## 项目目标
- 从课堂视频中抽帧并检测人脸。
- 对样本进行半自动标注（是否在场、情绪、专注度）。
- 拟合专注度映射规则并输出统计报告。

## 主要脚本
- `extract_frames.py`：从视频按时间间隔抽帧。
- `quick_label_faces.py`：快速标注人脸样本，可选专注度标注。
- `label_focus.py`：补充/修订专注度标注。
- `prepare_filtered_csv.py`（若在上层目录）：清洗与过滤标注数据。
- `fit_focus_mapping.py`：拟合专注度映射参数。
- `focus_mapping.py`：加载与应用专注度映射。
- `generate_report.py`：生成统计图表与报告。
- `app.py`：应用入口（本地运行分析流程）。

## 典型流程
1. 准备课堂视频。
2. 运行抽帧脚本生成 `frame/` 图像。
3. 使用标注脚本生成/修订 CSV 标注文件。
4. 拟合映射并导出 `focus_mapping_calibrated*.json`。
5. 运行报告脚本，在 `report_out/` 下查看结果。

## 输出说明
- `report_out/summary.csv`：总体统计。
- `report_out/trend_by_minute.csv`：分钟级趋势。
- `report_out/focus_trend.png`：专注度趋势图。
- `report_out/emotion_distribution.*`：情绪分布数据与图。

## 备注
当前仓库包含实验数据与模型文件（如 `*.pt`、`*.csv`、`frame/*.jpg`），建议按需在 `.gitignore` 中管理大文件。
