#!/usr/bin/env python3
"""从视频抽帧，供后续人工标注使用。"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="从视频抽帧")
    p.add_argument("--video", required=True, help="输入视频路径")
    p.add_argument("--out_dir", default="frames_out", help="输出目录")
    p.add_argument("--every_n", type=int, default=10, help="每 N 帧抽 1 帧")
    p.add_argument("--max_frames", type=int, default=0, help="最多导出多少帧，0=不限")
    p.add_argument("--prefix", default="frame", help="输出文件名前缀")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.every_n <= 0:
        raise ValueError("every_n 必须 > 0")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频: {args.video}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    saved = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if total % args.every_n == 0:
            ts_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            name = f"{args.prefix}_{total:06d}_{ts_ms:09d}ms.jpg"
            cv2.imwrite(str(out_dir / name), frame)
            saved += 1
            if args.max_frames > 0 and saved >= args.max_frames:
                break

        total += 1

    cap.release()
    print(f"总读取帧数: {total}")
    print(f"导出帧数: {saved}")
    print(f"输出目录: {out_dir}")


if __name__ == "__main__":
    main()
