#!/usr/bin/env python3
"""
快速人工标注脚本（半自动）：
- 自动检测人脸框（Haar）
- 你只需输入：
  1) 本图是否有人脸
  2) 每张人脸的情绪标签

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


def detect_faces(gray_img, detector) -> List[Tuple[int, int, int, int]]:
    faces = detector.detectMultiScale(
        gray_img,
        scaleFactor=1.1,
        minNeighbors=6,
        minSize=(40, 40),
    )
    return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]


def draw_faces(img, faces):
    canvas = img.copy()
    for i, (x, y, w, h) in enumerate(faces):
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(canvas, f"id:{i}", (x, max(15, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return canvas


def wait_preview_key(window_name: str) -> bool:
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
        "image", "has_face", "detected_face_count", "annotator", "labeled_at"
    ])

    model_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    detector = cv2.CascadeClassifier(model_path)
    if detector.empty():
        raise RuntimeError(f"无法加载人脸检测器: {model_path}")

    print(f"开始标注，共 {len(images)} 张图片")
    print("提示：先在窗口按 Enter/Space，再到终端输入标签。")
    cv2.namedWindow("quick_label_faces", cv2.WINDOW_NORMAL)

    for i, name in enumerate(images, start=1):
        path = os.path.join(args.image_dir, name)
        img = cv2.imread(path)
        if img is None:
            print(f"[跳过] 读取失败: {name}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detect_faces(gray, detector)

        canvas = draw_faces(img, faces)
        cv2.imshow("quick_label_faces", canvas)
        if not wait_preview_key("quick_label_faces"):
            print("\n用户主动结束标注。")
            break

        print(f"\n[{i}/{len(images)}] {name} | 自动检测到人脸数: {len(faces)}")
        has_face = ask_yes_no("本图是否有人脸? (y/n): ")

        now = datetime.now().isoformat(timespec="seconds")
        append_row(args.frame_csv, [name, int(has_face), len(faces), args.annotator, now])

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
