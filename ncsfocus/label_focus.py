#!/usr/bin/env python3
"""帧级专注度标注工具（high/mid/low）。"""

from __future__ import annotations

import argparse
import csv
import os
from datetime import datetime
from typing import List

import cv2

FOCUS_KEYS = {
    "h": "high",
    "m": "mid",
    "l": "low",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="帧级专注度标注")
    p.add_argument("--image_dir", required=True, help="待标注图片目录")
    p.add_argument("--out_csv", default="frame_focus_manual.csv", help="输出 CSV")
    p.add_argument("--annotator", default="annotator_a", help="标注人 ID")
    p.add_argument("--start_index", type=int, default=0)
    p.add_argument("--max_images", type=int, default=0)
    return p.parse_args()


def list_images(image_dir: str) -> List[str]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted([n for n in os.listdir(image_dir) if os.path.splitext(n.lower())[1] in exts])


def ensure_csv(path: str) -> None:
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "focus_level_gt", "annotator", "labeled_at"])


def ask_focus() -> str:
    while True:
        key = input("专注度键(h:high / m:mid / l:low / q:quit): ").strip().lower()
        if key == "q":
            return "quit"
        if key in FOCUS_KEYS:
            return FOCUS_KEYS[key]
        print("无效输入，请重试")


def main() -> None:
    args = parse_args()
    if not os.path.isdir(args.image_dir):
        raise FileNotFoundError(f"图片目录不存在: {args.image_dir}")

    images = list_images(args.image_dir)
    if not images:
        raise RuntimeError("目录中没有可标注图片")

    images = images[args.start_index:]
    if args.max_images > 0:
        images = images[:args.max_images]

    ensure_csv(args.out_csv)
    cv2.namedWindow("label_focus", cv2.WINDOW_NORMAL)

    for idx, name in enumerate(images, start=1):
        path = os.path.join(args.image_dir, name)
        img = cv2.imread(path)
        if img is None:
            print(f"[跳过] 读取失败: {name}")
            continue

        cv2.imshow("label_focus", img)
        cv2.waitKey(1)
        print(f"\n[{idx}/{len(images)}] {name}")
        lvl = ask_focus()
        if lvl == "quit":
            print("用户终止标注")
            break

        now = datetime.now().isoformat(timespec="seconds")
        with open(args.out_csv, "a", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow([name, lvl, args.annotator, now])

    cv2.destroyAllWindows()
    print(f"已输出: {args.out_csv}")


if __name__ == "__main__":
    main()
