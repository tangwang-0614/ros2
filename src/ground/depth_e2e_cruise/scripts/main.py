#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""depth_e2e_cruise 入口：端到端推理自主巡航.

用法:
    roslaunch depth_e2e_cruise main.launch                    # 默认 turtlebot3_world 场景
    roslaunch depth_e2e_cruise main.launch scenario:=house    # 多场景切换
"""
from depth_e2e_node import main

if __name__ == "__main__":
    main()
