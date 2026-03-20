#!/usr/bin/env python3
"""
自动抽帧脚本
- 支持按总帧数抽样
- 支持随机/分层（前中后）策略
- 输出文件名包含原始帧号与时间戳

示例：
  python extract_frames.py --video classroom1.mp4 --out sampled_frames --total 600 --strategy stratified
  python extract_frames.py --video classroom1.mp4 --out sampled_frames --total 300 --strategy random --seed 42
"""

from __future__ import annotations

import argparse
import math
import os
import random
from typing import List, Tuple

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从课堂视频自动抽帧")
    parser.add_argument("--video", required=True, help="输入视频路径")
    parser.add_argument("--out", default="sampled_frames", help="输出目录（默认 sampled_frames）")
    parser.add_argument("--total", type=int, default=600, help="总抽帧数量（默认 600）")
    parser.add_argument(
        "--strategy",
        choices=["stratified", "random"],
        default="stratified",
        help="抽样策略：stratified=前中后分层，random=全局随机",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子（默认 42）")
    parser.add_argument(
        "--dedupe_seconds",
        type=float,
        default=0.0,
        help="去重时间窗（秒），>0 时会尽量避免过密抽样，例如 0.5",
    )
    return parser.parse_args()


def split_ranges(total_frames: int) -> List[Tuple[int, int]]:
    one_third = total_frames // 3
    return [
        (0, one_third),
        (one_third, one_third * 2),
        (one_third * 2, total_frames),
    ]


def sample_indices_random(total_frames: int, n: int, seed: int) -> List[int]:
    random.seed(seed)
    n = min(n, total_frames)
    return sorted(random.sample(range(total_frames), n))


def sample_indices_stratified(total_frames: int, n: int, seed: int) -> List[int]:
    random.seed(seed)
    ranges = split_ranges(total_frames)

    counts = [n // 3, n // 3, n - 2 * (n // 3)]
    sampled: List[int] = []

    for (start, end), k in zip(ranges, counts):
        segment_len = max(0, end - start)
        if segment_len == 0 or k <= 0:
            continue
        k = min(k, segment_len)
        sampled.extend(random.sample(range(start, end), k))

    return sorted(set(sampled))


def enforce_time_dedup(indices: List[int], fps: float, dedupe_seconds: float) -> List[int]:
    if dedupe_seconds <= 0 or fps <= 0:
        return indices

    min_gap_frames = max(1, int(math.ceil(dedupe_seconds * fps)))
    out: List[int] = []
    last = -10**12
    for idx in indices:
        if idx - last >= min_gap_frames:
            out.append(idx)
            last = idx
    return out


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.video):
        raise FileNotFoundError(f"视频不存在: {args.video}")

    os.makedirs(args.out, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {args.video}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)

    if total_frames <= 0:
        cap.release()
        raise RuntimeError("读取到的视频总帧数无效")

    target_n = min(args.total, total_frames)

    if args.strategy == "random":
        indices = sample_indices_random(total_frames, target_n, args.seed)
    else:
        indices = sample_indices_stratified(total_frames, target_n, args.seed)

    indices = enforce_time_dedup(indices, fps, args.dedupe_seconds)

    saved = 0
    for i, fidx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fidx))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        ts = fidx / fps
        out_name = f"f_{saved:05d}_idx{fidx:07d}_t{ts:09.3f}.jpg"
        out_path = os.path.join(args.out, out_name)
        if cv2.imwrite(out_path, frame):
            saved += 1

    cap.release()

    print("=== 抽帧完成 ===")
    print(f"视频: {args.video}")
    print(f"策略: {args.strategy}")
    print(f"请求数量: {args.total}")
    print(f"实际保存: {saved}")
    print(f"输出目录: {args.out}")


if __name__ == "__main__":
    main()
