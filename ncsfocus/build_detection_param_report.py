#!/usr/bin/env python3
"""
生成“检测参数与实验依据”文档模板：
1) 自动从 app.py / quick_label_faces.py 读取默认参数（conf/iou/min_size/ratio）。
2) 生成可直接粘贴到论文的 Markdown 报告，包含：
   - 参数最终取值表
   - 对比实验空表（conf / iou / min_size / ratio）
   - 单标注者重标一致性空表
   - 可直接使用的结论段落

用法：
  python ncsfocus/build_detection_param_report.py \
      --out ncsfocus/PARAM_EXPERIMENT_TEMPLATE.md
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--app", default="ncsfocus/app.py", help="主程序路径")
    p.add_argument("--label", default="ncsfocus/quick_label_faces.py", help="标注脚本路径")
    p.add_argument("--out", default="ncsfocus/PARAM_EXPERIMENT_TEMPLATE.md", help="输出 markdown 文件")
    return p.parse_args()


def find_first(pattern: str, text: str, default: str = "N/A") -> str:
    m = re.search(pattern, text, flags=re.MULTILINE)
    return m.group(1) if m else default


def extract_defaults(app_text: str, label_text: str) -> dict:
    data = {
        "yolo_conf": find_first(r'YOLO_FACE_CONF",\s*"([0-9.]+)"', app_text),
        "yolo_iou": find_first(r'YOLO_FACE_IOU",\s*"([0-9.]+)"', app_text),
        "min_face_size": find_first(r"self\.min_face_size\s*=\s*([0-9]+)", app_text),
        "min_face_conf": find_first(r"self\.min_face_conf\s*=\s*([0-9.]+)", app_text),
        "ratio_low": find_first(r"return\s*([0-9.]+)\s*<=\s*ratio\s*<=\s*([0-9.]+)", app_text, default="N/A"),
        "ratio_high": "N/A",
        "label_conf": find_first(r"--yolo_conf" + r".*default=([0-9.]+)", label_text),
        "label_iou": find_first(r"--yolo_iou" + r".*default=([0-9.]+)", label_text),
    }
    m = re.search(r"return\s*([0-9.]+)\s*<=\s*ratio\s*<=\s*([0-9.]+)", app_text)
    if m:
        data["ratio_low"] = m.group(1)
        data["ratio_high"] = m.group(2)
    return data


def render_md(v: dict) -> str:
    return f"""# 检测参数与实验依据模板

> 自动提取自代码默认值，可直接用于论文第4章（参数设置）与第6章（实验依据）。

## 1. 当前代码默认参数（自动提取）

| 参数 | 当前默认值 | 代码来源 |
|---|---:|---|
| YOLO置信度阈值 conf | {v['yolo_conf']} | `app.py` + `quick_label_faces.py` |
| NMS IoU阈值 iou | {v['yolo_iou']} | `app.py` + `quick_label_faces.py` |
| 最小人脸尺寸 min(w,h) | {v['min_face_size']} px | `app.py` |
| 人脸框宽高比 w/h | [{v['ratio_low']}, {v['ratio_high']}] | `app.py` |
| 情绪最低置信度过滤 | {v['min_face_conf']} | `app.py` |

## 2. 参数实验对比表（填写实测结果）

### 表A：conf阈值对比（固定 iou={v['yolo_iou']}, min={v['min_face_size']}, ratio=[{v['ratio_low']},{v['ratio_high']}])

| conf | Precision | Recall | F1 | 误检率(%) | 漏检率(%) | FPS |
|---|---:|---:|---:|---:|---:|---:|
| 0.20 |  |  |  |  |  |  |
| {v['yolo_conf']}（当前） |  |  |  |  |  |  |
| 0.30 |  |  |  |  |  |  |
| 0.35 |  |  |  |  |  |  |

### 表B：NMS IoU阈值对比（固定 conf={v['yolo_conf']}, min={v['min_face_size']}, ratio=[{v['ratio_low']},{v['ratio_high']}])

| IoU | Precision | Recall | F1 | 重复框率(%) | FPS |
|---|---:|---:|---:|---:|---:|
| 0.40 |  |  |  |  |  |
| {v['yolo_iou']}（当前） |  |  |  |  |  |
| 0.50 |  |  |  |  |  |
| 0.55 |  |  |  |  |  |

### 表C：最小尺寸阈值对比（固定 conf={v['yolo_conf']}, iou={v['yolo_iou']}, ratio=[{v['ratio_low']},{v['ratio_high']}])

| min(w,h) px | 有效人脸保留率(%) | 误检率(%) | 情绪可识别率(%) | FPS |
|---|---:|---:|---:|---:|
| 32 |  |  |  |  |
| 40 |  |  |  |  |
| {v['min_face_size']}（当前） |  |  |  |  |
| 56 |  |  |  |  |

### 表D：宽高比阈值对比（固定 conf={v['yolo_conf']}, iou={v['yolo_iou']}, min={v['min_face_size']})

| ratio范围(w/h) | 误检率(%) | 有效人脸保留率(%) | F1 | FPS |
|---|---:|---:|---:|---:|
| [0.6, 1.6] |  |  |  |  |
| [{v['ratio_low']}, {v['ratio_high']}]（当前） |  |  |  |  |
| [0.8, 1.3] |  |  |  |  |

## 3. 单标注者一致性检验（你是单人标注适用）

| 指标 | 数值 |
|---|---:|
| 抽样量（帧/人脸） |  |
| 间隔天数 |  |
| 一致率(%) |  |
| Cohen’s Kappa（可选） |  |

## 4. 可直接粘贴到论文的描述

### 4.2 参数设置（方法章节）
本研究在人脸检测模块采用以下默认参数：置信度阈值 conf={v['yolo_conf']}、NMS IoU阈值 iou={v['yolo_iou']}、最小人脸尺寸 min(w,h)≥{v['min_face_size']}px、宽高比约束 w/h∈[{v['ratio_low']},{v['ratio_high']}]。其中 conf 与 iou 用于控制候选框置信度筛选与重叠框抑制强度；最小尺寸与宽高比约束用于过滤远景课堂中的噪声框与畸形框，以提升后续情绪识别输入质量。

### 6.4 参数依据（实验章节）
在固定数据与硬件环境下，本文对 conf、iou、最小尺寸与宽高比进行单因素扫描。以 Precision、Recall、F1、误检率及FPS为综合评价指标，最终选择 conf={v['yolo_conf']}、iou={v['yolo_iou']}、min(w,h)≥{v['min_face_size']}、w/h∈[{v['ratio_low']},{v['ratio_high']}] 作为默认配置。该组参数在精度、误检抑制与实时性之间取得较优折中。
"""


def main() -> None:
    args = parse_args()
    app_text = Path(args.app).read_text(encoding="utf-8")
    label_text = Path(args.label).read_text(encoding="utf-8")
    values = extract_defaults(app_text, label_text)
    out = Path(args.out)
    out.write_text(render_md(values), encoding="utf-8")
    print(f"已生成: {out}")


if __name__ == "__main__":
    main()
