#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""键盘示范采集节点：记录 (深度网格特征, 当前 /cmd_vel) 数据对 -> data/*.npz.

配合键盘遥操作使用（main.launch mode:=collect 会自动在 xterm 中拉起
turtlebot3_teleop_keyboard）：人以"示范教师"身份遥控小车平稳巡航，
节点按 collect/record_hz 采样，Ctrl+C 退出时保存数据集并打印统计。

数据集（npz 二进制）不入库：data/ 目录已在仓库 .gitignore 中忽略，
正式提交时上传网盘并在文档中提供下载链接。
"""
from __future__ import annotations

import os
import time

import numpy as np
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image

import depth_e2e_core as core

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CMD_STALE = 0.5         # /cmd_vel 超过该时长未更新则暂停采集 s


class Collector(object):

    def __init__(self):
        self.depth_topic = rospy.get_param("topic/depth", "/camera/depth/image_raw")
        self.cmd_topic = rospy.get_param("topic/cmd_vel", "/cmd_vel")
        self.rows = int(rospy.get_param("preprocess/grid_rows", 24))
        self.cols = int(rospy.get_param("preprocess/grid_cols", 32))
        self.min_range = float(rospy.get_param("preprocess/min_range", 0.05))
        self.max_range = float(rospy.get_param("preprocess/max_range", 5.0))
        self.hz = float(rospy.get_param("collect/record_hz", 10.0))
        self.min_v = float(rospy.get_param("collect/min_v", 0.02))
        self.min_w = float(rospy.get_param("collect/min_w", 0.05))
        self.max_depth_age = float(rospy.get_param("collect/max_depth_age", 0.5))
        out_dir = str(rospy.get_param("collect/out", "data"))
        self.out_path = os.path.join(PKG_ROOT, out_dir, "depth_e2e_dataset.npz")

        self.features = None
        self.last_depth_t = None
        self.last_cmd = (0.0, 0.0)
        self.last_cmd_t = None
        self.feat_buf, self.act_buf = [], []
        self.t_start = None

        rospy.Subscriber(self.depth_topic, Image, self.on_depth, queue_size=1)
        rospy.Subscriber(self.cmd_topic, Twist, self.on_cmd, queue_size=1)
        self.timer = rospy.Timer(rospy.Duration(1.0 / self.hz), self.tick)
        rospy.on_shutdown(self.on_stop)
        rospy.loginfo("[%s] 采集就绪：请在 xterm 键盘窗口中遥控小车平稳巡航，"
                      "Ctrl+C 结束并保存 -> %s", rospy.get_name(), self.out_path)

    # ---------------------------------------------------------------- 回调
    def on_depth(self, msg):
        try:
            self.features = core.depth_to_features(
                msg.data, msg.encoding, msg.width, msg.height,
                self.rows, self.cols, self.min_range, self.max_range)
        except (ValueError, TypeError):
            return
        self.last_depth_t = rospy.get_time()
        if self.t_start is None:
            self.t_start = rospy.get_time()

    def on_cmd(self, msg):
        self.last_cmd = (float(msg.linear.x), float(msg.angular.z))
        self.last_cmd_t = rospy.get_time()

    def tick(self, _event):
        now = rospy.get_time()
        if self.features is None or self.last_depth_t is None or self.last_cmd_t is None:
            return
        if now - self.last_depth_t > self.max_depth_age:   # 深度流断续 -> 暂停
            return
        if now - self.last_cmd_t > CMD_STALE:
            return
        v, w = self.last_cmd
        if abs(v) < self.min_v and abs(w) < self.min_w:    # 静止样本剔除
            return
        self.feat_buf.append(self.features.copy())
        self.act_buf.append((v, w))

    # ---------------------------------------------------------------- 保存
    def on_stop(self):
        n = len(self.feat_buf)
        if n < 10:
            rospy.logwarn("[%s] 有效样本仅 %d 个（<10），未保存。请重新采集。",
                          rospy.get_name(), n)
            return
        out_dir = os.path.dirname(self.out_path)
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        feats = np.asarray(self.feat_buf, dtype=np.float32)
        acts = np.asarray(self.act_buf, dtype=np.float32)
        np.savez_compressed(self.out_path, features=feats, actions=acts)
        duration = rospy.get_time() - (self.t_start or 0.0)
        rospy.loginfo("[%s] 数据集已保存: %s", rospy.get_name(), self.out_path)
        rospy.loginfo("[%s] 样本数 %d | 时长 %.1f s | v∈[%.3f, %.3f] m/s | "
                      "w∈[%.3f, %.3f] rad/s | 特征维度 %d",
                      rospy.get_name(), n, duration,
                      float(acts[:, 0].min()), float(acts[:, 0].max()),
                      float(acts[:, 1].min()), float(acts[:, 1].max()),
                      feats.shape[1])


def main():
    rospy.init_node("depth_e2e_collect")
    Collector()
    rospy.spin()


if __name__ == "__main__":
    main()
