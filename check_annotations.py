#!/usr/bin/env python3
"""
标注质量检查脚本
- 检查 face_labels / frame_labels 的列完整性
- 检查值域（情绪标签、专注等级、遮挡等级）
- 检查 bbox 合法性、重复样本、缺失值
- 可选检查图片文件是否存在
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

VALID_EMOTIONS = {"neutral", "happy", "sad", "angry", "fear", "surprise", "disgust", "unknown"}
VALID_OCCLUSIONS = {"none", "light", "heavy"}
VALID_FOCUS = {"high", "mid", "low"}


FACE_REQUIRED = [
    "image", "face_id", "x", "y", "w", "h", "face_valid", "emotion_gt", "occlusion_level", "annotator"
]
FRAME_REQUIRED = ["image", "focus_level_gt", "reason_tag", "annotator"]


def read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def ensure_columns(rows: List[Dict[str, str]], required: List[str], table_name: str) -> List[str]:
    if not rows:
        return [f"[{table_name}] 文件为空"]
    cols = set(rows[0].keys())
    missing = [c for c in required if c not in cols]
    if missing:
        return [f"[{table_name}] 缺少列: {missing}"]
    return []


def to_int(v: str, default: int = -1) -> int:
    try:
        return int(v)
    except Exception:
        return default


def check_face_rows(rows: List[Dict[str, str]], image_dir: str | None) -> Tuple[List[str], Dict[str, int]]:
    errors: List[str] = []
    stats = Counter()

    seen = set()
    emotions = Counter()
    occs = Counter()

    for i, r in enumerate(rows, start=2):
        key = (r.get("image", ""), r.get("face_id", ""))
        if key in seen:
            errors.append(f"[face_labels:L{i}] 重复主键 (image, face_id): {key}")
        seen.add(key)

        image = (r.get("image") or "").strip()
        if not image:
            errors.append(f"[face_labels:L{i}] image 为空")

        if image_dir and image:
            if not os.path.exists(os.path.join(image_dir, image)):
                errors.append(f"[face_labels:L{i}] 图片不存在: {image}")

        face_valid = to_int((r.get("face_valid") or "").strip(), default=-1)
        if face_valid not in (0, 1):
            errors.append(f"[face_labels:L{i}] face_valid 非法: {r.get('face_valid')}")

        x = to_int((r.get("x") or "").strip())
        y = to_int((r.get("y") or "").strip())
        w = to_int((r.get("w") or "").strip())
        h = to_int((r.get("h") or "").strip())

        # 只要 face_valid=1，就要求 bbox 合法
        if face_valid == 1:
            if x < 0 or y < 0 or w <= 0 or h <= 0:
                errors.append(f"[face_labels:L{i}] bbox 非法: x={x}, y={y}, w={w}, h={h}")

        emotion = (r.get("emotion_gt") or "").strip().lower()
        if emotion not in VALID_EMOTIONS:
            errors.append(f"[face_labels:L{i}] emotion_gt 不在允许集合: {emotion}")
        emotions[emotion] += 1

        occ = (r.get("occlusion_level") or "").strip().lower()
        if occ not in VALID_OCCLUSIONS:
            errors.append(f"[face_labels:L{i}] occlusion_level 不在允许集合: {occ}")
        occs[occ] += 1

        if not (r.get("annotator") or "").strip():
            errors.append(f"[face_labels:L{i}] annotator 为空")

        stats["face_rows"] += 1

    stats.update({f"emotion_{k}": v for k, v in emotions.items()})
    stats.update({f"occlusion_{k}": v for k, v in occs.items()})
    return errors, dict(stats)


def check_frame_rows(rows: List[Dict[str, str]], image_dir: str | None) -> Tuple[List[str], Dict[str, int]]:
    errors: List[str] = []
    stats = Counter()

    seen = set()
    focus_counter = Counter()

    for i, r in enumerate(rows, start=2):
        image = (r.get("image") or "").strip()
        if not image:
            errors.append(f"[frame_labels:L{i}] image 为空")

        if image in seen:
            errors.append(f"[frame_labels:L{i}] 重复 image: {image}")
        seen.add(image)

        if image_dir and image:
            if not os.path.exists(os.path.join(image_dir, image)):
                errors.append(f"[frame_labels:L{i}] 图片不存在: {image}")

        focus = (r.get("focus_level_gt") or "").strip().lower()
        if focus not in VALID_FOCUS:
            errors.append(f"[frame_labels:L{i}] focus_level_gt 不在允许集合: {focus}")
        focus_counter[focus] += 1

        if not (r.get("reason_tag") or "").strip():
            errors.append(f"[frame_labels:L{i}] reason_tag 为空")

        if not (r.get("annotator") or "").strip():
            errors.append(f"[frame_labels:L{i}] annotator 为空")

        stats["frame_rows"] += 1

    stats.update({f"focus_{k}": v for k, v in focus_counter.items()})
    return errors, dict(stats)


def cross_checks(face_rows: List[Dict[str, str]], frame_rows: List[Dict[str, str]]) -> List[str]:
    errors: List[str] = []

    face_images = {((r.get("image") or "").strip()) for r in face_rows}
    frame_images = {((r.get("image") or "").strip()) for r in frame_rows}

    only_in_face = sorted([x for x in face_images - frame_images if x])
    only_in_frame = sorted([x for x in frame_images - face_images if x])

    if only_in_face:
        errors.append(f"[cross] 仅在 face_labels 中出现的 image 数量: {len(only_in_face)}")
    if only_in_frame:
        errors.append(f"[cross] 仅在 frame_labels 中出现的 image 数量: {len(only_in_frame)}")

    return errors


def print_stats(title: str, stats: Dict[str, int]) -> None:
    print(f"\n=== {title} ===")
    for k in sorted(stats.keys()):
        print(f"{k}: {stats[k]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="标注质量检查")
    parser.add_argument("--face_csv", default="templates/face_labels_template.csv", help="face 标注 CSV")
    parser.add_argument("--frame_csv", default="templates/frame_labels_template.csv", help="frame 标注 CSV")
    parser.add_argument("--image_dir", default=None, help="可选：图片目录，用于检查 image 文件是否存在")
    args = parser.parse_args()

    face_rows = read_csv(args.face_csv)
    frame_rows = read_csv(args.frame_csv)

    errors = []
    errors += ensure_columns(face_rows, FACE_REQUIRED, "face_labels")
    errors += ensure_columns(frame_rows, FRAME_REQUIRED, "frame_labels")

    if errors:
        print("\n".join(errors))
        raise SystemExit(1)

    face_errors, face_stats = check_face_rows(face_rows, args.image_dir)
    frame_errors, frame_stats = check_frame_rows(frame_rows, args.image_dir)
    cross_errors = cross_checks(face_rows, frame_rows)

    all_errors = face_errors + frame_errors + cross_errors

    print_stats("face_labels 统计", face_stats)
    print_stats("frame_labels 统计", frame_stats)

    if all_errors:
        print("\n=== 发现问题 ===")
        for e in all_errors[:200]:
            print(e)
        if len(all_errors) > 200:
            print(f"... 其余 {len(all_errors) - 200} 条省略")
        raise SystemExit(2)

    print("\n✅ 标注检查通过：未发现结构性问题")


if __name__ == "__main__":
    main()
