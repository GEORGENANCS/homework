#!/usr/bin/env python3
"""
快速人工标注脚本（半自动）：
- 自动检测人脸框（仅 YOLO11）
- 你只需输入：
  1) 本图是否有人脸
  2) 每张人脸的情绪标签
  3) （可选）本图专注度等级 high/mid/low

输出：
- face_emotions_manual.csv   (每张人脸一行)
- frame_presence_manual.csv  (每张图一行)

默认图片目录：D:\\picture
"""

from __future__ import annotations

import argparse
import csv
import os
from datetime import datetime
from typing import List, Tuple

import cv2
from ultralytics import YOLO

EMOTION_KEYS = {
    "n": "neutral",
    "h": "happy",
    "s": "sad",
    "a": "angry",
    "f": "fear",
    "u": "surprise",
    "d": "disgust",
    "x": "unknown",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="快速人工标注：是否有人脸 + 人脸情绪")
    parser.add_argument("--image_dir", default=r"D:\\picture", help="图片目录，默认 D:\\picture")
    parser.add_argument("--face_csv", default="face_emotions_manual.csv", help="人脸情绪标注输出")
    parser.add_argument("--frame_csv", default="frame_presence_manual.csv", help="图片是否有人脸输出")
    parser.add_argument("--annotator", default="annotator_a", help="标注人ID")
    parser.add_argument("--start_index", type=int, default=0, help="从第几张开始（断点续标）")
    parser.add_argument("--max_images", type=int, default=0, help="最多标注多少张，0=全部")
    parser.add_argument("--detector", choices=["yolo11"], default="yolo11", help="人脸检测器（仅支持yolo11）")
    parser.add_argument("--yolo_model", default="yolo11n-face.pt", help="YOLO11 人脸模型路径或名称")
    parser.add_argument("--yolo_conf", type=float, default=0.25, help="YOLO 置信度阈值")
    parser.add_argument("--yolo_iou", type=float, default=0.45, help="YOLO NMS IoU 阈值")
    parser.add_argument("--label_focus", action="store_true", help="在帧级标注中追加专注度 high/mid/low")
    return parser.parse_args()


def list_images(image_dir: str) -> List[str]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    names = []
    for n in os.listdir(image_dir):
        if os.path.splitext(n.lower())[1] in exts:
            names.append(n)
    return sorted(names)


def ensure_csv(path: str, header: List[str]) -> None:
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)


def append_row(path: str, row: List[object]) -> None:
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def detect_faces_yolo(img_bgr, detector, conf: float, iou: float) -> List[Tuple[int, int, int, int]]:
    results = detector.predict(source=img_bgr, conf=conf, iou=iou, verbose=False)
    if not results:
        return []
    boxes = results[0].boxes
    if boxes is None or boxes.xyxy is None:
        return []

    out: List[Tuple[int, int, int, int]] = []
    for box in boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = [int(v) for v in box[:4]]
        w = max(0, x2 - x1)
        h = max(0, y2 - y1)
        if w > 0 and h > 0:
            out.append((x1, y1, w, h))
    return out


def draw_faces(img, faces):
    canvas = img.copy()
    for i, (x, y, w, h) in enumerate(faces):
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(canvas, f"id:{i}", (x, max(15, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return canvas


def wait_preview_key() -> bool:
    """
    在标注前等待用户确认，避免 OpenCV 窗口事件循环与终端 input() 交替时卡住。
    返回 True 表示继续，False 表示退出。
    """
    print("  在图片窗口按 Enter/Space 开始标注，按 q 退出。")
    while True:
        key = cv2.waitKey(50) & 0xFF
        if key in (13, 32):  # Enter / Space
            return True
        if key in (ord("q"), 27):  # q / Esc
            return False


def ask_yes_no(prompt: str) -> bool:
    while True:
        v = input(prompt).strip().lower()
        if v in ("y", "yes", "1"):
            return True
        if v in ("n", "no", "0"):
            return False
        print("请输入 y 或 n")


def ask_emotion(face_idx: int) -> str:
    tips = " / ".join([f"{k}:{v}" for k, v in EMOTION_KEYS.items()])
    while True:
        key = input(f"  人脸 {face_idx} 情绪键({tips})：").strip().lower()
        if key in EMOTION_KEYS:
            return EMOTION_KEYS[key]
        print("  无效键，请重输")


def ask_focus_level() -> str:
    """返回 high/mid/low，或空字符串(跳过)。"""
    while True:
        key = input("本图专注度键(h:high / m:mid / l:low / Enter跳过): ").strip().lower()
        if key == "":
            return ""
        if key == "h":
            return "high"
        if key == "m":
            return "mid"
        if key == "l":
            return "low"
        print("  无效键，请重输")


def main() -> None:
    args = parse_args()

    if not os.path.isdir(args.image_dir):
        raise FileNotFoundError(f"图片目录不存在: {args.image_dir}")

    images = list_images(args.image_dir)
    if not images:
        raise RuntimeError("目录中未找到可标注图片")

    if args.start_index < 0 or args.start_index >= len(images):
        raise ValueError("start_index 超出范围")

    if args.max_images > 0:
        images = images[args.start_index: args.start_index + args.max_images]
    else:
        images = images[args.start_index:]

    ensure_csv(args.face_csv, [
        "image", "face_id", "x", "y", "w", "h", "emotion_gt", "annotator", "labeled_at"
    ])
    ensure_csv(args.frame_csv, [
        "image", "has_face", "detected_face_count", "focus_level_gt", "annotator", "labeled_at"
    ])

    detector = YOLO(args.yolo_model)
    print(f"检测器: YOLO11 ({args.yolo_model}), conf={args.yolo_conf}, iou={args.yolo_iou}")

    print(f"开始标注，共 {len(images)} 张图片")
    print("提示：先在窗口按 Enter/Space，再到终端输入标签。")
    cv2.namedWindow("quick_label_faces", cv2.WINDOW_NORMAL)

    for i, name in enumerate(images, start=1):
        path = os.path.join(args.image_dir, name)
        img = cv2.imread(path)
        if img is None:
            print(f"[跳过] 读取失败: {name}")
            continue

        faces = detect_faces_yolo(img, detector, args.yolo_conf, args.yolo_iou)

        canvas = draw_faces(img, faces)
        cv2.imshow("quick_label_faces", canvas)
        if not wait_preview_key():
            print("\n用户主动结束标注。")
            break

        print(f"\n[{i}/{len(images)}] {name} | 自动检测到人脸数: {len(faces)}")
        has_face = ask_yes_no("本图是否有人脸? (y/n): ")

        now = datetime.now().isoformat(timespec="seconds")
        focus_level_gt = ask_focus_level() if args.label_focus else ""
        append_row(args.frame_csv, [name, int(has_face), len(faces), focus_level_gt, args.annotator, now])

        if has_face and faces:
            for fid, (x, y, w, h) in enumerate(faces):
                emo = ask_emotion(fid)
                append_row(args.face_csv, [name, fid, x, y, w, h, emo, args.annotator, now])
        elif has_face and not faces:
            print("  你判定有人脸，但自动检测为0，建议后续人工补框或重跑检测器。")

        print("  已保存当前图片标注。")

    cv2.destroyAllWindows()
    print("\n✅ 标注完成")
    print(f"face标注: {args.face_csv}")
    print(f"frame标注: {args.frame_csv}")


if __name__ == "__main__":
    main()
