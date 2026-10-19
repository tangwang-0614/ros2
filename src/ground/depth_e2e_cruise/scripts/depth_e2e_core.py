#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""depth_e2e_cruise 核心逻辑：深度图预处理、MLP 网络、动作解码、日志工具.

不依赖 rospy，可在无 ROS 环境下单独测试（见 tests/test_depth_e2e_cruise_local.py）。
torch 仅用于网络定义与权重加载，推理在 CPU 上即可运行。
"""
from __future__ import annotations

import csv
import os

import numpy as np
import torch
import torch.nn as nn

# 深度图编码 -> numpy dtype（Gazebo waffle 的 RealSense 输出 32FC1，单位米）
_DTYPES = {
    "32FC1": np.float32,
    "16UC1": np.uint16,
}


# ================================================================ 预处理
def depth_to_array(data, encoding, width, height, max_range=5.0):
    """原始深度图字节 -> (H, W) float32 米制数组，无效值（inf/nan/<=0）填为 max_range."""
    dtype = _DTYPES.get(encoding)
    if dtype is None:
        raise ValueError("不支持的深度图编码: %s" % encoding)
    arr = np.frombuffer(data, dtype=dtype)
    if arr.size != width * height:
        raise ValueError("深度图尺寸不匹配: %d != %dx%d" % (arr.size, width, height))
    arr = arr.reshape(height, width).astype(np.float32)
    if encoding == "16UC1":          # 16UC1 单位为毫米
        arr = arr / 1000.0
    arr[~np.isfinite(arr)] = max_range
    arr[arr <= 0.0] = max_range      # 0 表示无回波，视为远处而不是贴障
    return arr


def depth_to_features(data, encoding, width, height,
                      rows=24, cols=32, min_range=0.05, max_range=5.0):
    """深度图 -> 网格特征向量（网络输入）.

    网格化方式：每格 (H/rows x W/cols) 像素取最小值（min-pooling，保留最近障碍），
    再线性归一化到 [0, 1]（clip 到 [min_range, max_range] 后除以 max_range）。
    返回 shape=(rows*cols,) 的 float32 向量，行优先。
    """
    arr = depth_to_array(data, encoding, width, height, max_range)
    if height % rows == 0 and width % cols == 0:
        gh, gw = height // rows, width // cols
        grid = arr.reshape(rows, gh, cols, gw).min(axis=(1, 3))
    else:                            # 不能整除时按步长切片退化处理
        ys = np.linspace(0, height, rows + 1).astype(int)
        xs = np.linspace(0, width, cols + 1).astype(int)
        grid = np.empty((rows, cols), dtype=np.float32)
        for r in range(rows):
            for c in range(cols):
                grid[r, c] = arr[ys[r]:ys[r + 1], xs[c]:xs[c + 1]].min()
    grid = np.clip(grid, min_range, max_range) / max_range
    return grid.astype(np.float32).flatten()


# ================================================================ 网络
class CruiseMLP(nn.Module):
    """行为克隆 MLP：网格深度特征 -> (线速度 v, 角速度 w).

    输出经过 sigmoid/tanh 隐式限幅，物理值域 [0, max_lin] 与 [-max_ang, max_ang]，
    因此推理输出永远合法，不需要额外安全裁剪层。
    """

    def __init__(self, n_in=768, hidden=64, max_lin=0.22, max_ang=2.84):
        super().__init__()
        self.max_lin = float(max_lin)
        self.max_ang = float(max_ang)
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x):
        raw = self.net(x)
        v = torch.sigmoid(raw[..., 0]) * self.max_lin
        w = torch.tanh(raw[..., 1]) * self.max_ang
        return torch.stack([v, w], dim=-1)


def load_model(path, n_in, hidden, max_lin, max_ang, device="cpu"):
    """加载训练好的权重；文件不存在时给出明确报错（提示先运行 train.py）."""
    if not os.path.isfile(path):
        raise IOError(
            "未找到模型权重 %s，请先用 scripts/train.py 训练生成（见文档运行步骤）" % path)
    model = CruiseMLP(n_in=n_in, hidden=hidden, max_lin=max_lin, max_ang=max_ang)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


# ================================================================ 动作
def decode_action(out, max_lin=0.22, max_ang=2.84):
    """网络输出 (2,) -> (v, w) 物理值，并做安全限幅."""
    v = min(max(float(out[0]), 0.0), max_lin)
    w = min(max(float(out[1]), -max_ang), max_ang)
    return v, w


def low_pass(prev, new, alpha):
    """输出低通平滑：alpha=1 不平滑，越小越平滑."""
    a = min(max(float(alpha), 0.0), 1.0)
    return (prev[0] * (1.0 - a) + new[0] * a,
            prev[1] * (1.0 - a) + new[1] * a)


# ================================================================ 运行日志
class RunLogger(object):
    """推理过程 CSV 记录器（文本文件，可入库作为评测证据）."""

    HEADER = ["t", "v", "w", "min_depth"]

    def __init__(self, path):
        self._f = open(path, "w", newline="")
        self._w = csv.writer(self._f)
        self._w.writerow(self.HEADER)

    def log(self, t, v, w, min_depth):
        self._w.writerow(["%.3f" % t, "%.4f" % v, "%.4f" % w, "%.3f" % min_depth])

    def close(self):
        try:
            self._f.close()
        except Exception:
            pass
