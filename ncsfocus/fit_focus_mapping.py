#!/usr/bin/env python3
"""根据人工标注数据拟合 emotion->focus score 映射。"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from typing import Dict, List

FOCUS_NUMERIC = {
    "high": 85.0,
    "mid": 65.0,
    "low": 40.0,
}


def read_csv(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="拟合情绪到专注度映射")
    parser.add_argument("--face_csv", default="templates/face_labels_template.csv")
    parser.add_argument("--frame_csv", default="templates/frame_labels_template.csv")
    parser.add_argument("--out", default="focus_mapping_calibrated.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    face_rows = read_csv(args.face_csv)
    frame_rows = read_csv(args.frame_csv)

    frame_focus: Dict[str, float] = {}
    for r in frame_rows:
        image = (r.get("image") or "").strip()
        lvl = (r.get("focus_level_gt") or "").strip().lower()
        if image and lvl in FOCUS_NUMERIC:
            frame_focus[image] = FOCUS_NUMERIC[lvl]

    emo_vals = defaultdict(list)
    for r in face_rows:
        image = (r.get("image") or "").strip()
        emo = (r.get("emotion_gt") or "unknown").strip().lower()
        fv = frame_focus.get(image)
        if fv is not None:
            emo_vals[emo].append(fv)

    if not emo_vals:
        raise RuntimeError("未找到可用于拟合的标注样本")

    emotion_score_map: Dict[str, float] = {}
    for emo, vals in emo_vals.items():
        emotion_score_map[emo] = round(sum(vals) / len(vals), 3)

    out = {
        "method": "annotation_calibrated_mean",
        "focus_numeric": FOCUS_NUMERIC,
        "sample_count": int(sum(len(v) for v in emo_vals.values())),
        "emotion_score_map": emotion_score_map,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"已输出: {args.out}")
    for k in sorted(emotion_score_map):
        print(f"{k}: {emotion_score_map[k]:.3f} (n={len(emo_vals[k])})")


if __name__ == "__main__":
    main()
