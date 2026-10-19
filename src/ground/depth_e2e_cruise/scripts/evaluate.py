#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""运行日志评测：解析推理节点输出的 CSV，生成多场景性能指标表（Markdown）.

指标定义:
    duration        巡航时长 s（日志首末时间差）
    mean_v/max_v    平均/最大线速度 m/s
    mean_abs_w      平均角速度绝对值 rad/s（过大说明画龙）
    turn_rate_jerk  单位时间角速度变化量均值 rad/s^2（越小越平滑）
    min_depth_*     全程最近障碍距离的最小/均值 m（越大越安全）
    collisions      贴障事件次数（min_depth 低于阈值的状态切换计数）

用法:
    python3 evaluate.py --log runs/run_world_20260101_120000.csv
    python3 evaluate.py --dir runs --out runs/eval_table.md
"""
from __future__ import annotations

import argparse
import csv
import glob
import os

HEADER = ["t", "v", "w", "min_depth"]


def load_rows(path):
    """读取运行日志 CSV -> [(t, v, w, min_depth), ...]（纯 float，不依赖 numpy）."""
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        head = next(reader, None)
        if head != HEADER:
            raise ValueError("日志表头不匹配: %s (%s)" % (head, path))
        for line in reader:
            if len(line) < 4:
                continue
            rows.append((float(line[0]), float(line[1]),
                         float(line[2]), float(line[3])))
    return rows


def compute_metrics(rows, collision_depth=0.12):
    """日志行 -> 指标字典（纯函数，便于单元测试）."""
    if len(rows) < 2:
        return None
    ts = [r[0] for r in rows]
    vs = [r[1] for r in rows]
    ws = [r[2] for r in rows]
    ds = [r[3] for r in rows]
    duration = ts[-1] - ts[0]
    n = len(rows)
    mean_abs_w = sum(abs(w) for w in ws) / n
    jerk = sum(abs(ws[i] - ws[i - 1]) for i in range(1, n)) / max(duration, 1e-6)
    collisions = 0
    in_collision = ds[0] < collision_depth
    for d in ds[1:]:
        now = d < collision_depth
        if now and not in_collision:
            collisions += 1
        in_collision = now
    return {
        "duration": duration,
        "samples": n,
        "mean_v": sum(vs) / n,
        "max_v": max(vs),
        "mean_abs_w": mean_abs_w,
        "turn_rate_jerk": jerk,
        "min_depth_min": min(ds),
        "min_depth_mean": sum(ds) / n,
        "collisions": collisions,
    }


FIELDS = [
    ("duration", "巡航时长 (s)", "%.1f"),
    ("samples", "日志样本数", "%d"),
    ("mean_v", "平均线速度 (m/s)", "%.3f"),
    ("max_v", "最大线速度 (m/s)", "%.3f"),
    ("mean_abs_w", "平均角速度绝对值 (rad/s)", "%.3f"),
    ("turn_rate_jerk", "角速度变化率 (rad/s^2)", "%.3f"),
    ("min_depth_min", "最近障碍距离最小值 (m)", "%.3f"),
    ("min_depth_mean", "最近障碍距离均值 (m)", "%.3f"),
    ("collisions", "贴障事件次数", "%d"),
]


def format_table(metrics_by_label, collision_depth=0.12):
    """{场景: 指标dict} -> Markdown 表（第一个场景列作为指标名列）."""
    labels = list(metrics_by_label.keys())
    lines = ["| 指标 | " + " | ".join(labels) + " |",
             "|---" * (len(labels) + 1) + "|"]
    for key, name, fmt in FIELDS:
        cells = []
        for label in labels:
            value = metrics_by_label[label].get(key)
            cells.append(fmt % value if value is not None else "-")
        lines.append("| %s | %s |" % (name, " | ".join(cells)))
    lines.append("")
    lines.append("贴障判据：全程最近障碍距离 < %.2f m 记一次事件。" % collision_depth)
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="运行日志性能评测")
    ap.add_argument("--log", nargs="*", default=[], help="单个或多个日志 CSV 路径")
    ap.add_argument("--dir", default="", help="批量评测目录下所有 run_*.csv")
    ap.add_argument("--collision", type=float, default=0.12, help="贴障判据 m")
    ap.add_argument("--out", default="", help="结果 Markdown 输出路径（可选）")
    args = ap.parse_args()

    paths = list(args.log)
    if args.dir:
        paths += sorted(glob.glob(os.path.join(args.dir, "run_*.csv")))
    if not paths:
        raise SystemExit("未指定日志：用 --log <csv> 或 --dir <runs目录>")

    metrics_by_label = {}
    for path in paths:
        label = os.path.splitext(os.path.basename(path))[0]
        metrics_by_label[label] = compute_metrics(load_rows(path), args.collision)

    table = format_table(metrics_by_label, args.collision)
    print(table)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(table + "\n")
        print("已写入: %s" % args.out)


if __name__ == "__main__":
    main()
