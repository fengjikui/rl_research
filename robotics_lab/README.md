# 用 PPO 让宇树 Go2 在仿真里走路

这一部分沿用官方现成环境，不重新编写物理仿真器。先看懂一只机器狗怎样观察、给关节下指令、拿到奖励，再对照官方 PPO 代码。本文提供源码导读和 NVIDIA 服务器上的准备步骤；**尚未在本项目实际启动 Isaac Lab 训练，也没有验证 4090 或 A3 的运行结果。**

建议按这个顺序阅读：

1. [从一只站不稳的狗开始](Go2_PPO逐步讲解.md)：具体例子 → 输入输出 → 奖励 → 一批数据 → PPO → 源码位置。
2. [NVIDIA 服务器运行步骤](NVIDIA服务器运行.md)：固定版本、列出任务、64 个仿真环境短测、训练和录像。
3. 需要核查服务器时运行 [preflight.py](preflight.py)。它只读系统与安装信息，不安装软件、不启动仿真或训练。

## 选哪一套官方代码

本教程主线使用 [NVIDIA Isaac Lab](https://github.com/isaac-sim/IsaacLab) 自带的 Go2 平地速度跟踪任务：

```text
Isaac-Velocity-Flat-Unitree-Go2-v0
```

它的好处是任务、机器人资产配置、PPO 入口都在同一个官方框架里。进阶再换成崎岖地形任务 `Isaac-Velocity-Rough-Unitree-Go2-v0`。

为避免读文档与运行代码混版，教程固定 **Isaac Lab v2.3.2**，commit `37ddf626871758333d6ed89cf64ad702aef127d0`。对应可选依赖固定 **RSL-RL 3.1.2**，commit `c49573e4d1eb6662d7b047fe935522c23665f6f3`。这套版本的配置与训练入口均已逐项阅读核对。它是一个明确的教学版本，不声称是最新版本；当前 `main` 已调整目录和训练入口，不能把本教程的旧入口直接套到 `main`。[任务注册](https://github.com/isaac-sim/IsaacLab/blob/37ddf626871758333d6ed89cf64ad702aef127d0/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2/__init__.py)、[RSL-RL 依赖约束](https://github.com/isaac-sim/IsaacLab/blob/37ddf626871758333d6ed89cf64ad702aef127d0/source/isaaclab_rl/setup.py)。

另一个官方选择是 [宇树 Unitree RL Lab](https://github.com/unitreerobotics/unitree_rl_lab)。核对 commit `4960b84732b0c2ec593dccbfe963fda1bcd7b1e3` 时，它确实注册了 **`Unitree-Go2-Velocity`**，同时支持其他机器人；README 中的 `Unitree-G1-29dof-Velocity` 示例则是人形机器人，不能拿来当 Go2 的命令。宇树扩展还需要自己的资产准备；本次先沿 Isaac Lab 自带任务学习。[宇树 Go2 注册代码](https://github.com/unitreerobotics/unitree_rl_lab/blob/4960b84732b0c2ec593dccbfe963fda1bcd7b1e3/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2/__init__.py)。

源码核对日期：**2026-11-03**。上游文件通过官方仓库读取；本目录不包含第三方仓库副本或机器人资产，也不包含实体机器人控制命令。

## 和我们已有游戏 PPO 怎么连接

| 已有视觉游戏 | Go2 平地行走 |
|---|---|
| 观察来自游戏图像 | 观察来自身体速度、姿态、关节与指令 |
| 动作是一个离散按钮 | 动作是 12 个关节位置目标的偏移量 |
| 环境执行按钮后给新画面与奖励 | 物理仿真执行电机作用后给新身体状态与奖励 |
| Categorical 分布采样 | 对角高斯分布采样连续动作 |
| 保存 old log-prob、优势、价值目标 | 同样保存这些量，再进行 PPO 更新 |

最值得迁移的是你已经学过的采样与损失流程。最值得重新看的则是动作单位、控制频率、奖励各项以及仿真模型；这些决定“高分”究竟对应什么动作。

A3 目前没有可调试环境，不进行移植。此处固定的 Isaac Sim/PhysX CUDA 与 NVIDIA 图形运行链路不能直接当作 A3 NPU 程序运行；这不表示 A3 不能训练强化学习策略，只是那需要另选或移植仿真与学习后端。
