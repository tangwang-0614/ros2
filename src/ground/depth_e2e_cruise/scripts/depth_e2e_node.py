#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""端到端推理节点：深度图 -> MLP -> /cmd_vel 自主巡航.

订阅 /camera/depth/image_raw（TurtleBot3 waffle RealSense），每帧降采样为
网格特征，送入训练好的 CruiseMLP 得到 (v, w)，低通平滑后发布到 /cmd_vel；
同时把 (t, v, w, min_depth) 写入运行日志，供 scripts/evaluate.py 评测。

深度流中断超过 1 s 自动停车（安全兜底）。
"""
from __future__ import annotations

import os
import time

import numpy as np
import torch
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image

import depth_e2e_core as core

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEPTH_STALE = 1.0        # 深度流中断判据 s
MIN_DEPTH_FLOOR = 1e-3   # 日志中最小深度的下限，避免 0


class DepthE2ECruise(object):

    def __init__(self):
        # ---------------- 参数（与 config/depth_e2e.yaml 对应）
        self.depth_topic = rospy.get_param("topic/depth", "/camera/depth/image_raw")
        self.cmd_topic = rospy.get_param("topic/cmd_vel", "/cmd_vel")
        self.rows = int(rospy.get_param("preprocess/grid_rows", 24))
        self.cols = int(rospy.get_param("preprocess/grid_cols", 32))
        self.min_range = float(rospy.get_param("preprocess/min_range", 0.05))
        self.max_range = float(rospy.get_param("preprocess/max_range", 5.0))
        self.max_lin = float(rospy.get_param("control/max_lin", 0.22))
        self.max_ang = float(rospy.get_param("control/max_ang", 2.84))
        self.hz = float(rospy.get_param("control/publish_hz", 10.0))
        self.alpha = float(rospy.get_param("control/smooth_alpha", 0.5))
        hidden = int(rospy.get_param("model/hidden", 64))
        self.collision_depth = float(rospy.get_param("log/collision_depth", 0.12))

        model_path = rospy.get_param("model/path", "models/model.pth")
        if not os.path.isabs(model_path):
            model_path = os.path.join(PKG_ROOT, model_path)
        self.model = core.load_model(model_path, self.rows * self.cols,
                                     hidden, self.max_lin, self.max_ang)

        # ---------------- 运行日志
        self.logger = None
        if bool(rospy.get_param("log/enable", True)):
            tag = str(rospy.get_param("log/tag", "run"))
            log_dir = os.path.join(PKG_ROOT, str(rospy.get_param("log/dir", "runs")))
            if not os.path.isdir(log_dir):
                os.makedirs(log_dir)
            log_path = os.path.join(
                log_dir, "run_%s_%s.csv" % (tag, time.strftime("%Y%m%d_%H%M%S")))
            self.logger = core.RunLogger(log_path)
            rospy.loginfo("[%s] 运行日志: %s", rospy.get_name(), log_path)

        # ---------------- 状态与接口
        self.features = None          # 最近一帧网格特征
        self.last_depth_t = None
        self.smooth = (0.0, 0.0)
        self.pub = rospy.Publisher(self.cmd_topic, Twist, queue_size=1)
        rospy.Subscriber(self.depth_topic, Image, self.on_depth, queue_size=1)
        self.timer = rospy.Timer(rospy.Duration(1.0 / self.hz), self.tick)
        rospy.on_shutdown(self.on_stop)
        rospy.loginfo("[%s] 模型已加载，订阅 %s，发布 %s（%.1f Hz）",
                      rospy.get_name(), self.depth_topic, self.cmd_topic, self.hz)

    # ---------------------------------------------------------------- 回调
    def on_depth(self, msg):
        try:
            self.features = core.depth_to_features(
                msg.data, msg.encoding, msg.width, msg.height,
                self.rows, self.cols, self.min_range, self.max_range)
        except (ValueError, TypeError) as exc:
            rospy.logwarn_throttle(5.0, "[%s] 深度帧处理失败: %s", rospy.get_name(), exc)
            return
        self.last_depth_t = rospy.get_time()

    def tick(self, _event):
        now = rospy.get_time()
        stale = (self.features is None or self.last_depth_t is None
                 or now - self.last_depth_t > DEPTH_STALE)
        if stale:
            v, w = 0.0, 0.0           # 深度流异常 -> 停车兜底
        else:
            with torch.no_grad():
                out = self.model(torch.from_numpy(self.features).unsqueeze(0))[0].numpy()
            v, w = core.decode_action(out, self.max_lin, self.max_ang)

        self.smooth = core.low_pass(self.smooth, (v, w), self.alpha)
        cmd = Twist()
        cmd.linear.x = self.smooth[0]
        cmd.angular.z = self.smooth[1]
        self.pub.publish(cmd)

        if self.logger is not None and self.features is not None:
            min_depth = max(float(self.features.min()) * self.max_range, MIN_DEPTH_FLOOR)
            self.logger.log(now, self.smooth[0], self.smooth[1], min_depth)

    def on_stop(self):
        cmd = Twist()
        self.pub.publish(cmd)         # 退出前发零速
        if self.logger is not None:
            self.logger.close()
            rospy.loginfo("[%s] 已保存运行日志，可用 scripts/evaluate.py 评测", rospy.get_name())


def main():
    rospy.init_node("depth_e2e_cruise")
    DepthE2ECruise()
    rospy.spin()


if __name__ == "__main__":
    main()
