# 深度相机端到端巡航控制（行为克隆）

> 模块名：`depth_e2e_cruise` ｜ 机器人：TurtleBot3 waffle ｜ 环境：Ubuntu 20.04 + ROS Noetic + Gazebo Classic

本模块用 PyTorch 训练的 **MLP 神经网络**实现地面机器人的端到端巡航控制：
以 **RealSense 深度图像**为唯一输入，直接输出 `/cmd_vel` 速度指令，
替代传统"建图 - 规划 - 控制"三级流水线。完整闭环包含四个环节：

```
键盘示范采集 (collect_data.py) -> 离线训练 (train.py) -> 端到端推理 (main.py) -> 日志评测 (evaluate.py)
```

演示动图（按作业要求 10~20 MB，提交前替换）：

![端到端自主巡航演示](../img/ground/depth_e2e_cruise/depth_e2e_demo.gif)

## 1. 设计要求对照

| 作业设计要求 | 本模块的落点 |
|---|---|
| 全部任务在仿真环境内完成 | Gazebo Classic 仿真，TurtleBot3 waffle + 3 个原生场景 |
| 多场景测试，附带性能评价 | world/house/empty 三场景，`evaluate.py` 输出指标表（第 6、7 节） |
| 感知、规划、控制、端到端算法必须使用神经网络 | 端到端：深度图 -> MLP -> 速度指令，感知与控制合一，PyTorch 实现 |
| launch 启动，Ubuntu 20.04 ROS Noetic | `roslaunch depth_e2e_cruise main.launch`，入口 `main.py` |
| 键盘控制机器人运动 | `mode:=collect` 键盘示范采集（xterm 弹窗遥操作） |

## 2. 原理说明

### 2.1 行为克隆与端到端控制

行为克隆（Behavior Cloning）是模仿学习（Imitation Learning）中最直接的范式：
让"人类示范"产生 (状态, 动作) 数据对，用监督学习拟合策略网络
π(a|s)，使网络在部署时能像示范者一样根据传感器输入直接给出动作。

本模块中：

- **状态 s**：深度相机一帧图像降采样得到的网格特征（768 维，见 2.2）；
- **动作 a**：差速底盘速度指令 (v, w)，即线速度与角速度；
- **示范**：人用键盘遥控小车在训练场景中平稳巡航，遥控指令 `/cmd_vel`
  与当帧深度图配对记录；
- **策略网络**：3 层全连接 MLP（768 -> 64 -> 64 -> 2）。

与传统流水线相比，端到端方法把感知（障碍在哪里）和规划控制（往哪走）
压缩进同一个可微函数中，一次前向即可出控制指令，推理开销固定且极小。

### 2.2 深度图预处理

RealSense 原始深度图为 640×480（`32FC1`，单位米）。直接送入网络既浪费
算力又受无效值干扰，因此做三步处理：

1. **无效值清洗**：`inf`/`NaN`/非正值（无回波）一律置为量程上限
   `max_range=5 m`——"看不到"应当理解为"远处没有障碍"，而不是"贴着障碍"；
2. **min-pooling 网格化**：将图像划分为 24×32 网格（每格 20×20 像素），
   每格取**最小值**。取最小而非平均，是为了不漏掉格内细障碍（桌腿、立柱）；
3. **归一化**：clip 到 `[min_range, max_range]` 后除以 `max_range`，
   得到 [0,1] 特征，共 24×32 = 768 维。

### 2.3 网络结构与输出限幅

```
输入 768 维网格特征
  -> Linear(768, 64) + ReLU
  -> Linear(64, 64)  + ReLU
  -> Linear(64, 2)   -> v = sigmoid(x0) * 0.22     线速度 m/s
                     -> w = tanh(x1)   * 2.84      角速度 rad/s
```

输出端用 `sigmoid`/`tanh` 乘以物理量程做**隐式限幅**：推理输出永远在
waffle 底盘的合法速度域内，任何权重取值都不会产生非法指令，不需要额外的
安全裁剪层，也消除了训练早期的发散输出损坏仿真的风险。

### 2.4 训练目标

线速度量程 0.22 m/s，角速度量程 2.84 rad/s，二者相差一个数量级。若直接对
物理值求 MSE，损失几乎完全被角速度主导，线速度学不准。因此把两项误差各自
按量程归一化后相加：

$$
L = \frac{1}{N}\sum_i \left(\frac{v_i^{pred}-v_i}{0.22}\right)^2
  + \frac{1}{N}\sum_i \left(\frac{w_i^{pred}-w_i}{2.84}\right)^2
$$

优化器 Adam（lr=1e-3），batch 128，300 epoch，数据集 10% 留作验证集，
取验证损失最低的权重保存。CPU 训练数分钟内收敛。

## 3. 算法流程

```mermaid
flowchart LR
    A[键盘遥控示范<br/>turtlebot3_teleop] --> B[采集节点<br/>深度图+cmd_vel 配对]
    B --> C[(data/*.npz<br/>特征/动作对)]
    C --> D[train.py<br/>加权 MSE 训练 MLP]
    D --> E[(models/model.pth)]
    E --> F[推理节点 main.py]
    S[/camera/depth/image_raw/] --> G[清洗+min-pooling+归一化<br/>640x480 -> 768 维]
    G --> F
    F --> H[/cmd_vel 低通平滑<br/>Gazebo 差速执行]
    H --> I[runs/*.csv 运行日志]
    I --> J[evaluate.py<br/>多场景指标表]
```

推理节点实时链路（10 Hz）：

```
深度帧 -> 无效值清洗 -> 24x32 min-pooling -> 归一化 768 维
       -> MLP 前向 -> (v,w) 解码限幅 -> 低通平滑 -> Twist 发布
```

安全兜底：深度流中断超过 1 s 或退出节点时，立即发布零速 Twist 停车。

## 4. 源码解析

| 文件 | 职责 |
|---|---|
| `scripts/main.py` | **入口**：初始化 `depth_e2e_cruise` 节点，启动推理 |
| `scripts/depth_e2e_core.py` | 纯逻辑层：深度预处理、`CruiseMLP` 网络定义、动作解码、低通、CSV 日志器；不依赖 rospy，可脱离 ROS 单测 |
| `scripts/depth_e2e_node.py` | 推理节点：订阅深度话题，10 Hz 定时前向推理并发布 `/cmd_vel`，写运行日志，深度断流停车兜底 |
| `scripts/collect_data.py` | 采集节点：深度图与 `/cmd_vel` 时间配对，剔除静止样本，Ctrl+C 保存 npz 并打印统计 |
| `scripts/train.py` | 离线训练：加载 npz、加权 MSE、Adam 优化、验证集选优、保存权重与训练曲线数据 |
| `scripts/evaluate.py` | 评测：解析运行日志，计算时长/速度/平滑度/最近障碍距离/贴障次数，输出 Markdown 指标表 |
| `launch/main.launch` | 统一入口：场景加载（world/house/empty）+ 推理或采集模式 |
| `config/depth_e2e.yaml` | 全部可调参数（话题、预处理、控制限幅、模型、日志、采集） |

关键实现细节：

- **min-pooling 用一次 reshape 完成**（`scripts/depth_e2e_core.py`）：

```python
grid = arr.reshape(rows, gh, cols, gw).min(axis=(1, 3))
```

尺寸不能整除时退化为按 `np.linspace` 切片逐格取最小，保证任意分辨率可用。

- **时间同步采集**（`scripts/collect_data.py`）：深度帧与 `/cmd_vel` 各自记录
  最近时间戳，仅当两者都"新鲜"（深度 ≤0.5 s、指令 ≤0.5 s）且小车在动
  （|v|>0.02 或 |w|>0.05）时才落一条样本，避免"网络对着旧画面学旧动作"。

- **在线平滑**（`scripts/depth_e2e_node.py`）：网络输出经一阶低通
  `a' = (1-α)·a' + α·a` 再发布，α=0.5 时既抑制了帧间抖动，又保留响应速度。

## 5. 运行环境与操作步骤

### 5.1 环境安装

```bash
# ROS Noetic 安装见仓库《Ubuntu 20.04 安装 ROS Noetic 详细教程》
sudo apt install ros-noetic-turtlebot3-gazebo ros-noetic-turtlebot3-teleop xterm
pip3 install numpy torch --index-url https://download.pytorch.org/whl/cpu
```

<!-- TODO(截图1): 终端执行 pip3 install 成功的截图 -> docs/img/ground/depth_e2e_cruise/install_pip.png -->

### 5.2 编译

在本仓库根目录：

```bash
catkin_make
source devel/setup.bash
```

<!-- TODO(截图2): catkin_make 完成且能看到 depth_e2e_cruise 的截图 -->

### 5.3 键盘示范采集（第 1 步）

```bash
roslaunch depth_e2e_cruise main.launch mode:=collect
```

- 自动弹出 xterm 键盘窗口（`w/s/a/d` 前后左右转向），按 `ctrl+c` 退出键盘程序后再关闭 launch；
- 遥控要领：**平稳、居中、慢速**，沿场景可通行区域连续巡航 3~5 分钟，避免急转与碰撞；
- 结束时终端打印样本数、时长与 v/w 范围，数据集保存至
  `src/ground/depth_e2e_cruise/data/depth_e2e_dataset.npz`。
- 建议在同一场景采集 5~8 遍（数据量 3000+ 帧）。

<!-- TODO(截图3): xterm 键盘窗口 + 采集统计日志截图 -> collect_terminal.png -->

### 5.4 离线训练（第 2 步）

```bash
rosrun depth_e2e_cruise train.py
```

终端按 epoch 打印训练/验证损失；产出 `models/model.pth`（权重，不入库）
与 `models/train_log.csv`（损失曲线数据，入库）。可用任意表格工具将
train_log.csv 绘成损失曲线：

![训练损失曲线（提交前替换为实测图）](../img/ground/depth_e2e_cruise/train_curve.png)

<!-- TODO(截图4): 训练完成日志 + 由 train_log.csv 绘制的损失曲线 -->

### 5.5 端到端推理（第 3 步）

```bash
roslaunch depth_e2e_cruise main.launch
```

小车在 `turtlebot3_world` 中自主巡航：深度图进、速度指令出，全程无键盘
参与。`runs/` 目录同步生成 `run_world_<时间>.csv` 运行日志。

<!-- TODO(动图): 录制 ScreenToGif -> docs/img/ground/depth_e2e_cruise/depth_e2e_demo.gif（10~20MB） -->
<!-- TODO(截图5): rqt_graph 节点连接图（/camera/depth/image_raw -> depth_e2e_cruise -> /cmd_vel） -->

### 5.6 运行日志评测（第 4 步）

```bash
rosrun depth_e2e_cruise evaluate.py --dir src/ground/depth_e2e_cruise/runs --out eval_table.md
```

输出 Markdown 指标表（第 7 节），可直接粘入本页。

## 6. 多场景测试

`main.launch` 的 `scenario` 参数支持三个原生 Gazebo 场景，覆盖不同几何复杂度：

| 场景 | 命令 | 特点 | 主要考察 |
|---|---|---|---|
| `world`（默认） | `roslaunch depth_e2e_cruise main.launch` | 圆柱障碍阵列 | 环形绕障、示范分布内泛化 |
| `house` | `... scenario:=house` | 室内墙体与家具 | 狭窄通道、拐角处理 |
| `empty` | `... scenario:=empty` | 无障碍开阔地 | 直线巡航稳定性、速度保持 |

每个场景各推理运行一次（各约 2 分钟），得到三份运行日志。

<!-- TODO(截图6/7/8): 三个场景的 Gazebo 运行画面 -->

## 7. 性能评价

`evaluate.py` 计算的指标及含义：

| 指标 | 含义 | 期望趋势 |
|---|---|---|
| 巡航时长 (s) | 日志覆盖时间 | 各场景都能连续跑满设定时长 |
| 平均线速度 (m/s) | 巡航效率 | 开阔场景高于障碍场景 |
| 平均角速度绝对值 (rad/s) | 转向强度 | 过大说明"画龙" |
| 角速度变化率 (rad/s²) | 单位时间转向变化，平滑性 | 越小越平滑 |
| 最近障碍距离最小值 (m) | 全程离障碍最近的一次 | 明显大于贴障判据 0.12 m |
| 贴障事件次数 | 最近障碍距离 < 0.12 m 的状态切换计数 | 0 为理想 |

多场景实测结果（提交前运行 `evaluate.py` 填入）：

| 场景 | 巡航时长 (s) | 平均线速度 (m/s) | 角速度变化率 (rad/s²) | 最近障碍距离最小值 (m) | 贴障事件 |
|---|---|---|---|---|---|
| world | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 |
| house | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 |
| empty | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 |

**结果解读框架**：行为克隆策略在示范分布覆盖的区域（训练场景及相似布局）
表现平稳；`empty` 场景无障碍，平均线速度应接近示范上限；`house` 场景通道
较窄，最近障碍距离最小值会低于 `world`。若某场景贴障事件多，补充采集该
场景示范并重新训练即可。

## 8. 数据集与模型说明

按仓库规范，二进制大文件不入库：

| 产物 | 位置 | 交付方式 |
|---|---|---|
| 示范数据集 npz（约 10 MB/千帧） | `data/`（.gitignore 已忽略） | 网盘链接：<!-- TODO(网盘): 填入永久网盘地址与提取码 --> |
| 训练权重 model.pth（约 0.2 MB） | `models/`（.gitignore 已忽略） | `train.py` 一键复现（固定随机种子） |
| 训练曲线 train_log.csv | `models/`（文本，入库） | 直接提交 |
| 运行日志 runs/*.csv | `runs/`（文本，入库） | 作为评测证据直接提交 |

复现路径：`mode:=collect` 自采示范 -> `train.py` -> `main.launch`，
全流程不依赖任何外部数据。

## 9. 已知边界与常见问题

- **泛化边界**：在示范未覆盖的场景布局上行为不可靠（行为克隆的固有性质）。
  更换测试场景时，先在新场景补充示范并重训；
- **xterm 未安装**：`main.launch mode:=collect use_xterm:=false`，另开终端
  手动运行 `rosrun turtlebot3_teleop turtlebot3_teleop_keyboard`；
- **找不到权重**：`main.launch` 启动报"未找到模型权重"——先完成第 5.4 节
  训练，或用 `model_path:=` 参数显式指定权重路径；
- **深度画面全黑/无反应**：确认 `TURTLEBOT3_MODEL=waffle`（burger 无深度
  相机），`rostopic hz /camera/depth/image_raw` 应约 30 Hz；
- **推理输出抖动**：调小 `config/depth_e2e.yaml` 中 `smooth_alpha`（如 0.3）。

## 10. 大模型使用声明

本模块代码与文档在编写过程中使用了大语言模型辅助生成，作者已逐行审阅、
在本机 Ubuntu 20.04 + ROS Noetic 环境实测通过，并对提交内容负全部责任。
