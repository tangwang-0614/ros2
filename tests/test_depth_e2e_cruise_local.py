#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""depth_e2e_cruise 模块本地测试（桩依赖，不需要 ROS / Gazebo / torch）.

A. 深度图预处理：32FC1/16UC1 解码、无效值处理、min-pooling 网格化、归一化
B. 动作解码与低通平滑：限幅、混合系数边界
C. 运行日志评测：指标计算与贴障事件去重计数
D. 工程文件：package.xml / CMakeLists / main.launch / depth_e2e.yaml

用法: python3 tests/test_depth_e2e_cruise_local.py
"""
from __future__ import annotations

import os
import sys
import types
import xml.etree.ElementTree as ET

PASS, FAIL = [], []


def check(cond, label):
    (PASS if cond else FAIL).append(label)
    print("  [%s] %s" % ("PASS" if cond else "FAIL", label))


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
PKG = os.path.join(ROOT, "src", "ground", "depth_e2e_cruise")
SCRIPTS = os.path.join(PKG, "scripts")
sys.path.insert(0, SCRIPTS)


# ---------------------------------------------------------------- torch 桩
def install_torch_stub():
    """core 只在推理时才真正用 torch，测试用最小桩满足 import 即可."""
    if "torch" in sys.modules:
        return
    t = types.ModuleType("torch")
    nn = types.ModuleType("torch.nn")

    class _Module(object):
        def __init__(self, *args, **kwargs):
            pass

    nn.Module = _Module
    nn.Linear = lambda *a, **k: None
    nn.ReLU = lambda *a, **k: None
    nn.Sequential = lambda *a, **k: None
    t.nn = nn
    t.sigmoid = lambda x: x
    t.tanh = lambda x: x
    t.stack = lambda *a, **k: None
    t.from_numpy = lambda x: x
    t.load = lambda *a, **k: {}
    t.save = lambda *a, **k: None
    t.no_grad = lambda: types.SimpleNamespace(
        __enter__=lambda s: None, __exit__=lambda s, *a: None)
    t.manual_seed = lambda seed: None
    sys.modules["torch"] = t
    sys.modules["torch.nn"] = nn


install_torch_stub()

try:
    import numpy as np
except ImportError:                                  # pragma: no cover
    np = None

import depth_e2e_core as core  # noqa: E402
import evaluate as ev  # noqa: E402  （evaluate.py 与本测试同在 sys.path 的 scripts 下）

MAX_RANGE = 5.0


# ================================================================ A. 预处理
def test_depth_to_array():
    print("== A1. depth_to_array 解码与无效值 ==")
    if np is None:
        check(False, "本机无 numpy，跳过（Ubuntu 环境必装）")
        return
    w, h = 640, 480
    arr = np.full((h, w), 2.0, dtype=np.float32)
    arr[0, 0] = np.inf
    arr[0, 1] = np.nan
    arr[0, 2] = 0.0                       # 无回波
    raw = arr.tobytes()
    out = core.depth_to_array(raw, "32FC1", w, h, MAX_RANGE)
    check(out.shape == (h, w), "输出形状 (480, 640)")
    check(abs(float(out[10, 10]) - 2.0) < 1e-6, "正常值保持不变")
    check(float(out[0, 0]) == MAX_RANGE, "inf -> max_range")
    check(float(out[0, 1]) == MAX_RANGE, "nan -> max_range")
    check(float(out[0, 2]) == MAX_RANGE, "0(无回波) -> max_range")

    # 16UC1 毫米转米
    arr16 = np.full((h, w), 1500, dtype=np.uint16)
    out16 = core.depth_to_array(arr16.tobytes(), "16UC1", w, h, MAX_RANGE)
    check(abs(float(out16[10, 10]) - 1.5) < 1e-6, "16UC1 毫米 -> 米 (1500 -> 1.5)")

    try:
        core.depth_to_array(raw, "8UC3", w, h, MAX_RANGE)
        check(False, "未知编码应报错")
    except ValueError:
        check(True, "未知编码 -> ValueError")


def test_depth_to_features():
    print("== A2. depth_to_features 网格化与归一化 ==")
    if np is None:
        check(False, "本机无 numpy，跳过（Ubuntu 环境必装）")
        return
    w, h, rows, cols = 640, 480, 24, 32
    gh, gw = h // rows, w // cols
    arr = np.full((h, w), 1.0, dtype=np.float32)      # 全场 1 m
    raw = arr.tobytes()
    feats = core.depth_to_features(raw, "32FC1", w, h, rows, cols, 0.05, MAX_RANGE)
    check(feats.shape == (rows * cols,), "特征维度 24x32=768")
    check(abs(float(feats.min()) - 0.2) < 1e-6 and abs(float(feats.max()) - 0.2) < 1e-6,
          "均匀 1m -> 全部归一化为 0.2")

    # 最近障碍保留：左上角格子里放一个 0.3m 的障碍，min-pooling 后该格 = 0.3/5
    arr2 = np.full((h, w), 4.0, dtype=np.float32)
    arr2[0:gh, 0:gw] = 0.3
    f2 = core.depth_to_features(arr2.tobytes(), "32FC1", w, h, rows, cols, 0.05, MAX_RANGE)
    check(abs(float(f2[0]) - 0.3 / MAX_RANGE) < 1e-6, "格内最近障碍被 min-pooling 保留")
    check(abs(float(f2[1]) - 4.0 / MAX_RANGE) < 1e-6, "相邻格不受影响")

    # 越界截断：0.01m 噪点 clip 到 min_range
    arr3 = np.full((h, w), 0.01, dtype=np.float32)
    f3 = core.depth_to_features(arr3.tobytes(), "32FC1", w, h, rows, cols, 0.05, MAX_RANGE)
    check(abs(float(f3.min()) - 0.05 / MAX_RANGE) < 1e-6, "低于 min_range 截断")


# ================================================================ B. 动作
def test_action():
    print("== B. decode_action / low_pass ==")
    v, w = core.decode_action([0.5, -9.9], 0.22, 2.84)
    check(abs(v - 0.22) < 1e-9 and abs(w + 2.84) < 1e-9, "越界输出被限幅")
    v0, w0 = core.decode_action([-1.0, 10.0], 0.22, 2.84)
    check(v0 == 0.0 and abs(w0 - 2.84) < 1e-9, "负线速度截到 0，正角速度限幅")

    prev, new = (1.0, -1.0), (0.0, 1.0)
    check(core.low_pass(prev, new, 1.0) == new, "alpha=1 -> 完全跟随新值")
    s = core.low_pass(prev, new, 0.5)
    check(abs(s[0] - 0.5) < 1e-9 and abs(s[1]) < 1e-9, "alpha=0.5 -> 中点混合")
    check(core.low_pass(prev, new, 0.0) == prev, "alpha=0 -> 保持旧值")


# ================================================================ C. 评测
def test_evaluate():
    print("== C. evaluate 指标与贴障计数 ==")
    # 10 Hz x 5 s：v=0.2，w 先 +1 后 -1；min_depth 两段低于阈值（中间隔开）
    rows = []
    for i in range(50):
        t = i * 0.1
        w = 1.0 if i < 25 else -1.0
        d = 0.10 if i in (5, 6, 30) else 0.5
        rows.append((t, 0.2, w, d))
    m = ev.compute_metrics(rows, 0.12)
    check(m is not None, "指标可计算")
    check(abs(m["duration"] - 4.9) < 1e-6, "时长 = 首末时间差 4.9 s")
    check(m["samples"] == 50, "样本数 50")
    check(abs(m["mean_v"] - 0.2) < 1e-9, "平均线速度 0.2")
    check(abs(m["mean_abs_w"] - 1.0) < 1e-9, "平均角速度绝对值 1.0")
    check(m["min_depth_min"] == 0.10, "最近障碍最小值 0.10")
    check(m["collisions"] == 2, "贴障事件去重计数 = 2（连续帧算一次）")
    check(ev.compute_metrics(rows[:1], 0.12) is None, "样本不足返回 None")

    table = ev.format_table({"case_a": m}, 0.12)
    check("| 指标 | case_a |" in table, "Markdown 表含场景列")
    check("贴障判据" in table, "表尾注明贴障判据")


# ================================================================ D. 工程文件
def test_project_files():
    print("== D. 工程文件 ==")
    pkg = ET.parse(os.path.join(PKG, "package.xml")).getroot()
    check(pkg.findtext("name").strip() == "depth_e2e_cruise", "package.xml 包名正确")
    deps = {d.text.strip() for d in pkg.findall("exec_depend")}
    for dep in ("rospy", "sensor_msgs", "geometry_msgs", "turtlebot3_gazebo", "turtlebot3_teleop"):
        check(dep in deps, "package.xml 声明依赖 %s" % dep)

    cmake = open(os.path.join(PKG, "CMakeLists.txt"), encoding="utf-8").read()
    for name in ("main.py", "depth_e2e_core.py", "depth_e2e_node.py",
                 "collect_data.py", "train.py", "evaluate.py"):
        check(name in cmake, "CMakeLists 安装 %s" % name)

    launch = ET.parse(os.path.join(PKG, "launch", "main.launch")).getroot()
    nodes = [n.get("type") for n in launch.iter("node")]   # 含 group 内嵌套节点
    check("main.py" in nodes, "main.launch 启动入口 main.py")
    check("collect_data.py" in nodes, "main.launch 支持采集节点")
    args = [a.get("name") for a in launch.findall("arg")]
    for a in ("mode", "scenario", "model_path", "log_enable", "use_xterm"):
        check(a in args, "main.launch 暴露参数 %s" % a)
    launch_text = open(os.path.join(PKG, "launch", "main.launch"), encoding="utf-8").read()
    check("depth_e2e.yaml" in launch_text, "main.launch 加载配置 yaml")
    check("TURTLEBOT3_MODEL" in launch_text, "main.launch 指定 waffle（深度相机型号）")
    check(launch_text.count("turtlebot3_world.launch")
          + launch_text.count("turtlebot3_house.launch")
          + launch_text.count("turtlebot3_empty_world.launch") >= 3,
          "main.launch 支持 world/house/empty 三场景")

    yaml_text = open(os.path.join(PKG, "config", "depth_e2e.yaml"), encoding="utf-8").read()
    for key in ("depth", "cmd_vel", "grid_rows", "max_range", "max_lin",
                "max_ang", "publish_hz", "smooth_alpha", "model", "collision_depth",
                "record_hz"):
        check(key in yaml_text, "depth_e2e.yaml 含 %s" % key)

    for script in ("main.py", "collect_data.py", "train.py", "evaluate.py",
                   "depth_e2e_core.py", "depth_e2e_node.py"):
        path = os.path.join(PKG, "scripts", script)
        check(os.path.isfile(path), "脚本存在 %s" % script)
        if os.path.isfile(path):
            head = open(path, encoding="utf-8").read(200)
            check(head.startswith("#!/usr/bin/env python3"), "%s 有可执行 shebang" % script)


def main():
    test_depth_to_array()
    test_depth_to_features()
    test_action()
    test_evaluate()
    test_project_files()
    print("\n==================== 结果 ====================")
    print("通过 %d 项，失败 %d 项" % (len(PASS), len(FAIL)))
    if FAIL:
        for f in FAIL:
            print("  FAIL: %s" % f)
        return 1
    print("全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
