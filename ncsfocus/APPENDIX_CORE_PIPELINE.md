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


## E. SQLite 表结构规范（附录可直接引用）

以下给出推荐的规范化表结构，用于保证“帧级汇总表 + 人脸事件表”的可追溯性、可检索性与可复现性。

### E.1 帧级汇总表：`frame_summary`

**建表 SQL（建议版）**

```sql
CREATE TABLE IF NOT EXISTS frame_summary (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    ts TEXT NOT NULL,
    face_count INTEGER NOT NULL,
    avg_focus_score REAL,
    smoothed_focus_score REAL,
    created_at TEXT NOT NULL DEFAULT (datetime('now','localtime'))
);
```

**字段说明**

| 字段名 | 类型 | 约束 | 说明 |
|---|---|---|---|
| id | INTEGER | PRIMARY KEY AUTOINCREMENT | 自增主键 |
| session_id | TEXT | NOT NULL | 一次分析会话唯一标识 |
| ts | TEXT | NOT NULL | 帧时间戳（建议 ISO8601） |
| face_count | INTEGER | NOT NULL | 当前帧有效人脸数 |
| avg_focus_score | REAL | NULLABLE | 当前帧原始平均专注度 |
| smoothed_focus_score | REAL | NULLABLE | EMA 平滑后专注度 |
| created_at | TEXT | NOT NULL | 入库时间 |

**索引建议**

```sql
CREATE INDEX IF NOT EXISTS idx_frame_summary_session_ts
ON frame_summary(session_id, ts);

CREATE INDEX IF NOT EXISTS idx_frame_summary_session
ON frame_summary(session_id);
```

**存储样例**

| id | session_id | ts | face_count | avg_focus_score | smoothed_focus_score | created_at |
|---:|---|---|---:|---:|---:|---|
| 1 | session_20260526_101530 | 2026-05-26T10:15:31.240 | 18 | 68.42 | 67.95 | 2026-05-26 10:15:31 |
| 2 | session_20260526_101530 | 2026-05-26T10:15:31.440 | 17 | 66.90 | 67.58 | 2026-05-26 10:15:31 |

---

### E.2 人脸事件表：`face_emotion_event`

**建表 SQL（建议版）**

```sql
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
    focus_score REAL,
    raw_preds_json TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now','localtime'))
);
```

**字段说明**

| 字段名 | 类型 | 约束 | 说明 |
|---|---|---|---|
| id | INTEGER | PRIMARY KEY AUTOINCREMENT | 自增主键 |
| session_id | TEXT | NOT NULL | 对应分析会话ID |
| ts | TEXT | NOT NULL | 事件时间戳（帧时刻） |
| face_idx | INTEGER | NOT NULL | 同一帧内人脸序号 |
| bbox_x / bbox_y | INTEGER | NOT NULL | 人脸框左上角坐标 |
| bbox_w / bbox_h | INTEGER | NOT NULL | 人脸框宽高 |
| top_emotion | TEXT | NOT NULL | top-1 情绪标签 |
| top_confidence | REAL | NOT NULL | top-1 情绪置信度 |
| focus_score | REAL | NULLABLE | 该人脸专注度分数 |
| raw_preds_json | TEXT | NULLABLE | 全部情绪概率分布 JSON |
| created_at | TEXT | NOT NULL | 入库时间 |

**索引建议**

```sql
CREATE INDEX IF NOT EXISTS idx_face_event_session_ts
ON face_emotion_event(session_id, ts);

CREATE INDEX IF NOT EXISTS idx_face_event_session_emotion
ON face_emotion_event(session_id, top_emotion);

CREATE INDEX IF NOT EXISTS idx_face_event_session_faceidx
ON face_emotion_event(session_id, face_idx);
```

**存储样例**

| id | session_id | ts | face_idx | bbox_x | bbox_y | bbox_w | bbox_h | top_emotion | top_confidence | focus_score | raw_preds_json |
|---:|---|---|---:|---:|---:|---:|---:|---|---:|---:|---|
| 1 | session_20260526_101530 | 2026-05-26T10:15:31.240 | 0 | 412 | 186 | 64 | 64 | neutral | 0.81 | 71.34 | [{"label":"neutral","score":0.81},{"label":"happy","score":0.12},...] |
| 2 | session_20260526_101530 | 2026-05-26T10:15:31.240 | 1 | 528 | 194 | 58 | 58 | sad | 0.67 | 49.86 | [{"label":"sad","score":0.67},{"label":"neutral","score":0.19},...] |

---

### E.3 主外键与一致性建议

- 主键：两表均采用 `id INTEGER PRIMARY KEY AUTOINCREMENT`。
- 逻辑关联键：`session_id + ts` 可将某一帧的汇总信息与该帧所有人脸事件关联。
- 若需强约束，可增加唯一键：

```sql
CREATE UNIQUE INDEX IF NOT EXISTS uq_face_event_frame_face
ON face_emotion_event(session_id, ts, face_idx);
```

---

### E.4 查询示例（论文可附）

**查询某会话的分钟级专注度趋势**

```sql
SELECT SUBSTR(ts,1,16) AS minute_bucket,
       AVG(smoothed_focus_score) AS minute_focus,
       AVG(face_count) AS minute_faces
FROM frame_summary
WHERE session_id = ?
GROUP BY minute_bucket
ORDER BY minute_bucket;
```

**查询某会话情绪分布**

```sql
SELECT top_emotion, COUNT(*) AS cnt
FROM face_emotion_event
WHERE session_id = ?
GROUP BY top_emotion
ORDER BY cnt DESC;
```
