#!/usr/bin/env python3
"""
控制变量参数实验脚本（人脸检测）

功能：
- 在控制台通过参数指定模型、数据、真值、扫描维度与候选值
- 自动跑对比实验并输出：Precision / Recall / F1 / 误检率 / 漏检率 / FPS
- 支持单因素控制变量：conf / iou / min_size / ratio

示例：
python ncsfocus/run_param_sweep.py \
  --image_dir ncsfocus/frame \
  --gt_csv ncsfocus/face_emotions_manual.csv \
  --model yolo11n-face.pt \
  --sweep conf --values 0.20,0.25,0.30,0.35 \
  --base_conf 0.25 --base_iou 0.45 --base_min_size 48 --base_ratio 0.7,1.4 \
  --iou_match 0.5 --save_csv ncsfocus/exp_conf.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from ultralytics import YOLO


BBox = Tuple[int, int, int, int]  # x,y,w,h


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--image_dir", required=True)
    p.add_argument("--gt_csv", required=True, help="需含字段: image,x,y,w,h")
    p.add_argument("--model", default="yolo11n-face.pt")

    p.add_argument("--sweep", choices=["conf", "iou", "min_size", "ratio"], required=True)
    p.add_argument("--values", required=True, help="逗号分隔。ratio格式示例: 0.6:1.6,0.7:1.4")

    p.add_argument("--base_conf", type=float, default=0.25)
    p.add_argument("--base_iou", type=float, default=0.45)
    p.add_argument("--base_min_size", type=int, default=48)
    p.add_argument("--base_ratio", default="0.7,1.4")

    p.add_argument("--iou_match", type=float, default=0.5, help="预测框与GT匹配阈值")
    p.add_argument("--limit", type=int, default=0, help="限制图片数量，0表示全部")
    p.add_argument("--save_csv", default="")
    return p.parse_args()


def parse_ratio(s: str) -> Tuple[float, float]:
    a, b = s.split(",") if "," in s else s.split(":")
    return float(a), float(b)


def parse_values(sweep: str, raw: str):
    arr = [x.strip() for x in raw.split(",") if x.strip()]
    if sweep == "ratio":
        return [parse_ratio(x) for x in arr]
    if sweep == "min_size":
        return [int(x) for x in arr]
    return [float(x) for x in arr]


def load_gt(path: str) -> Dict[str, List[BBox]]:
    out: Dict[str, List[BBox]] = {}
    with open(path, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        required = {"image", "x", "y", "w", "h"}
        if not required.issubset(set(rd.fieldnames or [])):
            raise ValueError(f"gt_csv缺少字段，至少需要: {required}")
        for r in rd:
            name = (r.get("image") or "").strip()
            if not name:
                continue
            x, y, w, h = int(float(r["x"])), int(float(r["y"])), int(float(r["w"])), int(float(r["h"]))
            if w <= 0 or h <= 0:
                continue
            out.setdefault(name, []).append((x, y, w, h))
    return out


def iou(a: BBox, b: BBox) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def filter_box(b: BBox, min_size: int, ratio_range: Tuple[float, float]) -> bool:
    _, _, w, h = b
    if w < min_size or h < min_size:
        return False
    r = w / float(h)
    return ratio_range[0] <= r <= ratio_range[1]


def detect(model: YOLO, img_path: str, conf: float, iou_th: float, min_size: int, ratio_range: Tuple[float, float]) -> List[BBox]:
    r = model.predict(source=img_path, conf=conf, iou=iou_th, verbose=False)
    if not r:
        return []
    boxes = r[0].boxes
    if boxes is None or boxes.xyxy is None:
        return []
    out = []
    for b in boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = [int(v) for v in b[:4]]
        w, h = max(0, x2 - x1), max(0, y2 - y1)
        box = (x1, y1, w, h)
        if w > 0 and h > 0 and filter_box(box, min_size, ratio_range):
            out.append(box)
    return out


@dataclass
class Metrics:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    seconds: float = 0.0
    images: int = 0

    def to_row(self, label: str):
        p = self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0
        r = self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        fp_rate = self.fp / (self.tp + self.fp) * 100 if (self.tp + self.fp) else 0.0
        fn_rate = self.fn / (self.tp + self.fn) * 100 if (self.tp + self.fn) else 0.0
        fps = self.images / self.seconds if self.seconds > 0 else 0.0
        return {
            "setting": label,
            "Precision": round(p, 4),
            "Recall": round(r, 4),
            "F1": round(f1, 4),
            "误检率(%)": round(fp_rate, 2),
            "漏检率(%)": round(fn_rate, 2),
            "FPS": round(fps, 2),
            "TP": self.tp,
            "FP": self.fp,
            "FN": self.fn,
        }


def match_counts(preds: List[BBox], gts: List[BBox], iou_th: float) -> Tuple[int, int, int]:
    used = set()
    tp = 0
    for p in preds:
        best_iou = 0.0
        best_j = -1
        for j, g in enumerate(gts):
            if j in used:
                continue
            v = iou(p, g)
            if v > best_iou:
                best_iou = v
                best_j = j
        if best_j >= 0 and best_iou >= iou_th:
            tp += 1
            used.add(best_j)
    fp = max(0, len(preds) - tp)
    fn = max(0, len(gts) - tp)
    return tp, fp, fn


def main():
    args = parse_args()
    gt = load_gt(args.gt_csv)
    images = sorted([p for p in os.listdir(args.image_dir) if p.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))])
    if args.limit > 0:
        images = images[: args.limit]
    if not images:
        raise RuntimeError("image_dir下无可用图片")

    ratio_base = parse_ratio(args.base_ratio)
    values = parse_values(args.sweep, args.values)

    model = YOLO(args.model)
    rows = []

    for val in values:
        conf = args.base_conf
        iou_th = args.base_iou
        min_size = args.base_min_size
        ratio = ratio_base

        if args.sweep == "conf":
            conf = float(val)
            label = f"conf={conf}"
        elif args.sweep == "iou":
            iou_th = float(val)
            label = f"iou={iou_th}"
        elif args.sweep == "min_size":
            min_size = int(val)
            label = f"min_size={min_size}"
        else:
            ratio = val
            label = f"ratio={ratio[0]}:{ratio[1]}"

        m = Metrics()
        t0 = time.perf_counter()
        for name in images:
            img_path = str(Path(args.image_dir) / name)
            preds = detect(model, img_path, conf=conf, iou_th=iou_th, min_size=min_size, ratio_range=ratio)
            gts = gt.get(name, [])
            tp, fp, fn = match_counts(preds, gts, args.iou_match)
            m.tp += tp
            m.fp += fp
            m.fn += fn
            m.images += 1
        m.seconds = time.perf_counter() - t0
        row = m.to_row(label)
        rows.append(row)

    # 控制台输出
    header = ["setting", "Precision", "Recall", "F1", "误检率(%)", "漏检率(%)", "FPS", "TP", "FP", "FN"]
    print("\n" + " | ".join(header))
    print("-" * 110)
    for r in rows:
        print(" | ".join(str(r[h]) for h in header))

    if args.save_csv:
        with open(args.save_csv, "w", encoding="utf-8", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=header)
            wr.writeheader()
            wr.writerows(rows)
        print(f"\n已保存: {args.save_csv}")


if __name__ == "__main__":
    main()
