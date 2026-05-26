# 附录：核心代码流程说明（人脸检测—筛选—情绪识别—专注度计算—平滑—存储）

本文系统在运行时按如下主链路执行：

1. 人脸检测（YOLO11）
2. 人脸框筛选（最小尺寸 + 宽高比）
3. 情绪识别（Hugging Face 模型）
4. 专注度计算（情绪概率加权）
5. 时间平滑（EMA）
6. 结果存储（SQLite：帧级汇总 + 人脸级事件）

---

## A. 流程总览

- 线程启动后先加载映射配置、YOLO 人脸检测器和情绪识别模型；
- 在视频循环中按帧执行检测与识别；
- 将每张人脸的情绪概率映射为专注度分数；
- 对同帧多人分数求平均得到帧级分数；
- 用 EMA 做时序平滑，得到稳定趋势；
- 将帧级与人脸级结果写入数据库，供离线报告统计。

---

## B. 核心步骤与代码对应

## B.1 人脸检测（YOLO11）

系统通过 Ultralytics YOLO 执行人脸检测，关键阈值为：
- 置信度阈值：`YOLO_FACE_CONF`（默认 0.25）
- NMS IoU 阈值：`YOLO_FACE_IOU`（默认 0.45）

实现要点：
- 通过 `self.face_detector.predict(source=frame, conf=..., iou=...)` 获取候选框。
- 读取 `boxes.xyxy` 转成 `(x, y, w, h)`。

---

## B.2 人脸框筛选（几何约束）

为减少课堂远景误检，检测框在进入情绪识别前需通过几何筛选：
- 最小尺寸阈值：`self.min_face_size`（代码当前默认 48）
- 宽高比约束：`0.7 <= w/h <= 1.4`

筛选函数：`_is_reasonable_face_box(w, h)`。

---

## B.3 情绪识别

对通过筛选的人脸 ROI：
1. 裁剪 `face_img = frame[y:y+h, x:x+w]`
2. 转 RGB 并喂入情绪模型 `dima806/facial_emotions_image_detection`
3. 读取类别概率分布与 top-1 情绪
4. 若 top-1 置信度低于 `self.min_face_conf`（默认 0.45），则跳过该脸，降低噪声

---

## B.4 专注度计算（情绪概率加权）

系统使用 `FocusEstimator` 将情绪映射为分值，并采用概率加权：

\[
\text{focus}(face)=\frac{\sum_i p_i\cdot s_i}{\sum_i p_i}
\]

其中：
- \(p_i\)：第 \(i\) 个情绪类别概率
- \(s_i\)：该类别对应专注度映射分值

同一帧多人时，取所有有效人脸分数均值作为帧级专注度。

---

## B.5 时间平滑（EMA）

为降低单帧波动，使用指数滑动平均：

\[
S_t=\alpha X_t+(1-\alpha)S_{t-1}
\]

- \(X_t\)：当前帧原始专注度
- \(S_t\)：平滑后专注度
- \(\alpha\)：平滑系数（代码当前 `self.ema_alpha=0.35`）

该步骤显著提升趋势曲线可读性与稳定性。

---

## B.6 结果存储（SQLite）

系统落库两类表：

1. `frame_summary`（帧级）
- `session_id`
- `ts`
- `face_count`
- `avg_focus_score`
- `smoothed_focus_score`

2. `face_emotion_event`（人脸级）
- `session_id`
- `ts`
- `face_idx`
- `bbox_x/y/w/h`
- `top_emotion`
- `top_confidence`
- `focus_score`
- `raw_preds_json`

对应写入接口：
- `insert_frame_summary(...)`
- `insert_face_events(...)`

---

## C. 流程伪代码（附录可直接引用）

```text
初始化：加载映射、YOLO、人脸情绪模型、数据库连接
while 视频仍可读取:
    读入一帧 frame
    检测人脸 boxes = YOLO(frame, conf, iou)
    对每个 box:
        若不满足 min_size / ratio: continue
        裁剪 ROI 并做情绪识别
        若 top_conf < min_face_conf: continue
        用情绪概率加权计算 face_focus
        记录人脸事件
    若存在有效人脸:
        frame_focus = mean(face_focus_list)
        smooth_focus = EMA(frame_focus, alpha)
    否则:
        smooth_focus 维持上次状态或置空
    写入 frame_summary 与 face_emotion_event
    更新界面显示与趋势图
结束：释放资源
```

---

## D. 说明（论文附录建议）

- 若论文正文实验最终采用 `min_size=32`，需同步修改代码默认值或在文中注明“实验参数覆盖运行默认值”。
- 建议在附录中同时给出参数配置来源（环境变量 / 启动参数）以增强可复现性。
