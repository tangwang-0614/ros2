# depth_e2e_cruise 深度相机端到端巡航（行为克隆）

以 **TurtleBot3 waffle 的 RealSense 深度图像**为唯一输入，用 PyTorch 训练的
MLP 直接输出 `/cmd_vel` 速度指令，替代"建图-规划-控制"流水线。
完整闭环：**键盘示范采集 → 离线训练 → 端到端推理 → 日志评测**，
支持 3 个 Gazebo 场景的多场景测试与性能评价。

详细原理、算法流程与源码解析见文档：`docs/ground/depth_e2e_cruise.md`

## 运行环境

- Ubuntu 20.04 + ROS Noetic（Python 3.8）
- Gazebo Classic 11、TurtleBot3 仿真包
- Python 依赖（见 requirements.txt）：numpy、torch（CPU 版即可）

```bash
sudo apt install ros-noetic-turtlebot3-gazebo ros-noetic-turtlebot3-teleop xterm
pip3 install numpy torch --index-url https://download.pytorch.org/whl/cpu
```

## 编译

在本仓库根目录执行：

```bash
catkin_make
source devel/setup.bash
```

## 快速开始（四步闭环）

```bash
# 1. 键盘示范采集：弹出 xterm 键盘窗口，遥控小车平稳巡航几分钟，Ctrl+C 保存
roslaunch depth_e2e_cruise main.launch mode:=collect

# 2. 离线训练（CPU 数分钟内完成），产出 models/model.pth
rosrun depth_e2e_cruise train.py

# 3. 端到端自主巡航（默认 turtlebot3_world 场景）
roslaunch depth_e2e_cruise main.launch

# 4. 运行日志评测（多场景各跑一遍，批量对比）
rosrun depth_e2e_cruise evaluate.py --dir src/ground/depth_e2e_cruise/runs --out eval_table.md
```

## 多场景测试

```bash
roslaunch depth_e2e_cruise main.launch scenario:=world   # 圆柱障碍世界
roslaunch depth_e2e_cruise main.launch scenario:=house   # 室内家居场景
roslaunch depth_e2e_cruise main.launch scenario:=empty   # 空旷开阔场景
```

每次推理运行会在 `runs/` 目录留下 `run_<场景>_<时间>.csv` 日志
（文本文件，可入库作为评测证据），`evaluate.py` 汇总为指标表。

## 注意

- 数据集 `data/` 与权重 `models/model.pth` 为二进制不入库：
  数据集走网盘链接（见文档），权重用 `train.py` 一键复现。
- 训练与推理的场景分布要保持一致（在哪采的示范就在哪测），这是行为
  克隆方法的固有边界，文档"多场景测试"一节有详细说明。
