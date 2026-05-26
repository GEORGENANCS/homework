#!/usr/bin/env python3
"""
基于 SQLite 的离线分析报表生成脚本。

默认读取项目根目录 focus_data.db，输出：
- report_out/summary.csv
- report_out/trend_by_minute.csv
- report_out/emotion_distribution.csv
- report_out/report.md
- report_out/focus_trend.png (可选，依赖 matplotlib)
- report_out/emotion_distribution.png (可选，依赖 matplotlib)
"""

from __future__ import annotations

import argparse
import csv
import os
import sqlite3
from datetime import datetime
from typing import Dict, List, Sequence, Tuple

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从 focus_data.db 生成离线分析报告")
    parser.add_argument("--db", default="focus_data.db", help="SQLite 数据库路径")
    parser.add_argument("--session_id", default="", help="指定会话ID，留空则使用最近会话")
    parser.add_argument("--out_dir", default="report_out", help="报告输出目录")
    parser.add_argument("--risk_threshold", type=float, default=(65.0 + 40.0) / 2.0, help="低专注阈值，默认取(mid+low)/2=52.5")
    return parser.parse_args()


def query_all(conn: sqlite3.Connection, sql: str, params: Sequence[object] = ()) -> List[Tuple]:
    cur = conn.cursor()
    cur.execute(sql, params)
    return cur.fetchall()


def ensure_out_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def pick_session_id(conn: sqlite3.Connection, session_id: str) -> str:
    if session_id:
        rows = query_all(conn, "SELECT 1 FROM frame_summary WHERE session_id = ? LIMIT 1", (session_id,))
        if not rows:
            raise ValueError(f"指定 session_id 不存在: {session_id}")
        return session_id

    rows = query_all(
        conn,
        """
        SELECT session_id
        FROM frame_summary
        GROUP BY session_id
        ORDER BY MAX(ts) DESC
        LIMIT 1
        """,
    )
    if not rows:
        raise RuntimeError("frame_summary 为空，无法生成报告")
    return str(rows[0][0])


def write_csv(path: str, header: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def build_summary(conn: sqlite3.Connection, session_id: str, risk_threshold: float) -> Dict[str, object]:
    row = query_all(
        conn,
        """
        SELECT
          MIN(ts),
          MAX(ts),
          COUNT(*),
          AVG(avg_focus_score),
          AVG(smoothed_focus_score),
          AVG(face_count)
        FROM frame_summary
        WHERE session_id = ?
        """,
        (session_id,),
    )[0]
    start_ts, end_ts, frame_rows, avg_focus, avg_smooth, avg_faces = row

    low_rows = query_all(
        conn,
        """
        SELECT COUNT(*)
        FROM frame_summary
        WHERE session_id = ? AND smoothed_focus_score IS NOT NULL AND smoothed_focus_score < ?
        """,
        (session_id, float(risk_threshold)),
    )[0][0]
    valid_rows = query_all(
        conn,
        """
        SELECT COUNT(*)
        FROM frame_summary
        WHERE session_id = ? AND smoothed_focus_score IS NOT NULL
        """,
        (session_id,),
    )[0][0]

    emotion_top = query_all(
        conn,
        """
        SELECT top_emotion, COUNT(*) AS cnt
        FROM face_emotion_event
        WHERE session_id = ?
        GROUP BY top_emotion
        ORDER BY cnt DESC
        LIMIT 1
        """,
        (session_id,),
    )
    top_emotion = emotion_top[0][0] if emotion_top else "N/A"

    low_ratio = (float(low_rows) / float(valid_rows) * 100.0) if valid_rows else 0.0
    return {
        "session_id": session_id,
        "start_ts": start_ts or "",
        "end_ts": end_ts or "",
        "frame_rows": int(frame_rows or 0),
        "avg_focus_score": float(avg_focus) if avg_focus is not None else None,
        "avg_smoothed_focus_score": float(avg_smooth) if avg_smooth is not None else None,
        "avg_face_count": float(avg_faces) if avg_faces is not None else None,
        "risk_threshold": float(risk_threshold),
        "low_focus_rows": int(low_rows or 0),
        "low_focus_ratio_pct": low_ratio,
        "top_emotion": top_emotion,
    }


def build_trend_by_minute(conn: sqlite3.Connection, session_id: str) -> List[Tuple[str, float, float]]:
    rows = query_all(
        conn,
        """
        SELECT
          SUBSTR(ts, 1, 16) AS minute_bucket,
          AVG(smoothed_focus_score) AS minute_focus,
          AVG(face_count) AS minute_faces
        FROM frame_summary
        WHERE session_id = ? AND smoothed_focus_score IS NOT NULL
        GROUP BY minute_bucket
        ORDER BY minute_bucket
        """,
        (session_id,),
    )
    return [(str(t), float(f), float(c)) for (t, f, c) in rows if f is not None and c is not None]


def build_emotion_distribution(conn: sqlite3.Connection, session_id: str) -> List[Tuple[str, int, float]]:
    rows = query_all(
        conn,
        """
        SELECT top_emotion, COUNT(*) AS cnt
        FROM face_emotion_event
        WHERE session_id = ?
        GROUP BY top_emotion
        ORDER BY cnt DESC
        """,
        (session_id,),
    )
    total = sum(int(r[1]) for r in rows)
    out: List[Tuple[str, int, float]] = []
    for emo, cnt in rows:
        c = int(cnt)
        pct = (c / total * 100.0) if total else 0.0
        out.append((str(emo), c, pct))
    return out


def build_low_focus_segments(
    conn: sqlite3.Connection, session_id: str, risk_threshold: float, limit: int = 50
) -> List[Tuple[str, float, int]]:
    rows = query_all(
        conn,
        """
        SELECT ts, smoothed_focus_score, face_count
        FROM frame_summary
        WHERE session_id = ? AND smoothed_focus_score IS NOT NULL AND smoothed_focus_score < ?
        ORDER BY smoothed_focus_score ASC
        LIMIT ?
        """,
        (session_id, float(risk_threshold), int(limit)),
    )
    return [(str(ts), float(score), int(fc)) for (ts, score, fc) in rows]


def save_plots(
    out_dir: str,
    trend_rows: List[Tuple[str, float, float]],
    emotion_rows: List[Tuple[str, int, float]],
) -> Dict[str, str]:
    paths: Dict[str, str] = {}
    if plt is None:
        return paths

    if trend_rows:
        x = [r[0] for r in trend_rows]
        y = [r[1] for r in trend_rows]
        plt.figure(figsize=(10, 4))
        plt.plot(x, y, marker="o")
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Smoothed Focus")
        plt.title("Focus Trend by Minute")
        plt.tight_layout()
        trend_png = os.path.join(out_dir, "focus_trend.png")
        plt.savefig(trend_png, dpi=150)
        plt.close()
        paths["focus_trend_png"] = trend_png

    if emotion_rows:
        labels = [r[0] for r in emotion_rows]
        sizes = [r[1] for r in emotion_rows]
        plt.figure(figsize=(6, 6))
        plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
        plt.title("Emotion Distribution")
        plt.tight_layout()
        emo_png = os.path.join(out_dir, "emotion_distribution.png")
        plt.savefig(emo_png, dpi=150)
        plt.close()
        paths["emotion_png"] = emo_png

    return paths


def format_float(v: object) -> str:
    if v is None:
        return "N/A"
    return f"{float(v):.2f}"


def write_report_md(
    path: str,
    summary: Dict[str, object],
    trend_rows: List[Tuple[str, float, float]],
    emotion_rows: List[Tuple[str, int, float]],
    low_focus_rows: List[Tuple[str, float, int]],
    plot_paths: Dict[str, str],
) -> None:
    lines: List[str] = []
    lines.append("# 课堂专注度离线分析报告")
    lines.append("")
    lines.append(f"- 生成时间: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- 会话ID: {summary['session_id']}")
    lines.append(f"- 时段: {summary['start_ts']} ~ {summary['end_ts']}")
    lines.append("")
    lines.append("## 1) 总览指标")
    lines.append("")
    lines.append(f"- 抽样帧数: {summary['frame_rows']}")
    lines.append(f"- 平均专注度(avg_focus_score): {format_float(summary['avg_focus_score'])}")
    lines.append(f"- 平滑平均专注度(smoothed): {format_float(summary['avg_smoothed_focus_score'])}")
    lines.append(f"- 平均人脸数: {format_float(summary['avg_face_count'])}")
    lines.append(
        f"- 低专注占比(<{summary['risk_threshold']}): "
        f"{summary['low_focus_rows']} / {summary['frame_rows']} "
        f"({format_float(summary['low_focus_ratio_pct'])}%)"
    )
    lines.append(f"- 主导情绪: {summary['top_emotion']}")
    lines.append("")
    lines.append("## 2) 分钟级趋势")
    lines.append("")
    lines.append(f"- 统计分钟点数: {len(trend_rows)}")
    if plot_paths.get("focus_trend_png"):
        png_name = os.path.basename(plot_paths["focus_trend_png"])
        lines.append(f"![focus trend]({png_name})")
    else:
        lines.append("- 未生成趋势图（可能未安装 matplotlib 或无可用数据）")
    lines.append("")
    lines.append("## 3) 情绪分布")
    lines.append("")
    if emotion_rows:
        for emo, cnt, pct in emotion_rows:
            lines.append(f"- {emo}: {cnt} ({pct:.2f}%)")
    else:
        lines.append("- 无情绪数据")
    if plot_paths.get("emotion_png"):
        png_name = os.path.basename(plot_paths["emotion_png"])
        lines.append(f"![emotion distribution]({png_name})")
    else:
        lines.append("- 未生成情绪分布图（可能未安装 matplotlib 或无可用数据）")
    lines.append("")
    lines.append("## 4) 低专注片段（Top 50）")
    lines.append("")
    if low_focus_rows:
        lines.append("| ts | smoothed_focus_score | face_count |")
        lines.append("|---|---:|---:|")
        for ts, score, fc in low_focus_rows:
            lines.append(f"| {ts} | {score:.2f} | {fc} |")
    else:
        lines.append("- 未发现低于阈值的片段")
    lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    args = parse_args()
    ensure_out_dir(args.out_dir)

    if not os.path.exists(args.db):
        raise FileNotFoundError(f"数据库不存在: {args.db}")

    conn = sqlite3.connect(args.db)
    try:
        session_id = pick_session_id(conn, args.session_id)
        summary = build_summary(conn, session_id, args.risk_threshold)
        trend_rows = build_trend_by_minute(conn, session_id)
        emotion_rows = build_emotion_distribution(conn, session_id)
        low_focus_rows = build_low_focus_segments(conn, session_id, args.risk_threshold, limit=50)

        summary_csv = os.path.join(args.out_dir, "summary.csv")
        write_csv(
            summary_csv,
            header=list(summary.keys()),
            rows=[list(summary.values())],
        )

        trend_csv = os.path.join(args.out_dir, "trend_by_minute.csv")
        write_csv(trend_csv, header=["minute_bucket", "minute_focus", "minute_faces"], rows=trend_rows)

        emotion_csv = os.path.join(args.out_dir, "emotion_distribution.csv")
        write_csv(emotion_csv, header=["top_emotion", "count", "pct"], rows=emotion_rows)

        plot_paths = save_plots(args.out_dir, trend_rows, emotion_rows)

        report_md = os.path.join(args.out_dir, "report.md")
        write_report_md(report_md, summary, trend_rows, emotion_rows, low_focus_rows, plot_paths)

        print("=== 报告生成完成 ===")
        print(f"session_id: {session_id}")
        print(f"summary_csv: {summary_csv}")
        print(f"trend_csv: {trend_csv}")
        print(f"emotion_csv: {emotion_csv}")
        print(f"report_md: {report_md}")
        if plot_paths:
            for k, v in plot_paths.items():
                print(f"{k}: {v}")
        else:
            print("plots: skipped (matplotlib not available or no data)")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
