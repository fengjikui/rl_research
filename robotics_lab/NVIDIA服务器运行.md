# 在 NVIDIA 服务器上准备 Go2 仿真训练

本页保留通用的 NVIDIA Linux 运行流程。AutoDL 实例上的安装细节和实测步骤见 [AutoDL 配置教程](01_AutoDL从开机到IsaacLab.md) 和 [Go2 训练与回放教程](02_Go2从短测到回放.md)。

## 1 固定一套版本再安装

本教程选择 Isaac Lab **v2.3.2**（commit `37ddf626871758333d6ed89cf64ad702aef127d0`）与其依赖的 RSL-RL **3.1.2**。对应官方 pip 安装指南使用 **Isaac Sim 5.1.0、Python 3.11**；Linux x86_64 示例为 PyTorch 2.7.0 + CUDA 12.8。先按这一版本的官方安装步骤核验驱动、系统、Python 与包版本，不直接把本项目 Mac 上的 `.venv` 复制过去。[版本安装文档](https://isaac-sim.github.io/IsaacLab/v2.3.2/source/setup/quickstart.html)。

单张 4090 可作为这种任务的候选工作站显卡，但可开多少只狗、多少时间达到好效果，要由所选场景、驱动、可用显存和实际吞吐决定。开始先用少量环境检查；不要一上来照搬默认 4096 个环境，也不承诺半小时一定学会。

首次启动可能下载 Kit 扩展与机器人资产，编译或建立缓存。安装、下载、初始化时间应与正式学习时间分开记录。没有桌面不代表没有图形依赖；`--headless` 关闭可见窗口，录视频仍需要对应渲染与相机设置。Isaac Sim 的许可由服务器操作者阅读并按官方流程接受。

## 2 先检查服务器

在本项目仓库根目录执行：

```bash
python robotics_lab/preflight.py --out robot_preflight.json
```

如果已检出 Isaac Lab，还可检查固定源码：

```bash
python robotics_lab/preflight.py \
  --isaaclab-root /path/to/IsaacLab \
  --out robot_preflight.json
```

该脚本读取 CPU 架构、内存、安装版本、`nvidia-smi` 输出与可选源码路径，不安装、不启动模拟器、不申请训练。输出“找到 GPU”只是一项检查，不等于 Vulkan、资产和模拟器已经能运行。

安装按上面官方指南完成后，检出明确版本并安装 RSL-RL 集成：

```bash
git clone --branch v2.3.2 https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
git rev-parse HEAD
./isaaclab.sh -i rsl_rl
```

`git rev-parse HEAD` 应与本页 commit 一致。不要同时混用 `main` 的文档、v2.3.2 的代码和最新版 RSL-RL：当前 `main` 的目录和训练入口已发生变化。

## 3 先列出任务再训练

以下命令均在 **IsaacLab 仓库根目录**、已配置好的环境里执行：

```bash
./isaaclab.sh -p scripts/environments/list_envs.py --keyword Go2
```

确认列表包含 `Isaac-Velocity-Flat-Unitree-Go2-v0`。如果没有，先解决版本或安装问题，而不是把 task 名改成 G1；G1 是另一种机器人。[官方任务表](https://isaac-sim.github.io/IsaacLab/v2.3.2/source/overview/environments.html)。

## 4 用 64 个环境做启动检查

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Flat-Unitree-Go2-v0 \
  --headless --num_envs 64 --max_iterations 10
```

这 10 个 iteration 用于检查模拟器、资产、CUDA 和更新是否正常，不是步态效果验收。检查日志中的观察与动作维数、每秒采样量、loss 是否为有限值、GPU 负载/显存、机器的实际温度与错误日志。若环境启动失败，优先排查缺失资产、版本和图形依赖，不能把它归因于 PPO 理论。

初始化与 10 轮学习均正常后，再把环境数逐步加到 256、1024 等，比较每秒数据量与资源占用。更大的并行数未必在每台机器上更快；不与同机其他用户的任务抢占全部资源。

## 5 正式训练与模型保存

例如先使用 1024 个环境，保留该任务默认训练轮数：

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Flat-Unitree-Go2-v0 \
  --headless --num_envs 1024 --max_iterations 300
```

300 是本教程固定版本的平地任务默认 iteration 数，不是 300 次环境 step。默认每只狗每轮收 24 个动作，再做 PPO 更新。1024 个环境与 300 轮大约对应 `1024×24×300 = 7,372,800` 条转移，不等于训练花了多少分钟。[训练配置](https://github.com/isaac-sim/IsaacLab/blob/37ddf626871758333d6ed89cf64ad702aef127d0/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2/agents/rsl_rl_ppo_cfg.py)。

检查点与环境、算法配置写入 `logs/rsl_rl/unitree_go2_flat/<运行时间>/`。配置保留下来，才知道某段视频究竟是怎样训练出来的。要接着某个运行继续学习，可查看该版本训练脚本的 `--resume`、`--load_run`、`--checkpoint` 参数；明确指定检查点，不默认把不同实验混在一起。

训练时先不录视频，以免把额外渲染开销混入吞吐。需要定期录像时，训练入口也支持 `--video`、`--video_length`、`--video_interval`，应把这种设置记入实验配置。[实际训练脚本](https://github.com/isaac-sim/IsaacLab/blob/37ddf626871758333d6ed89cf64ad702aef127d0/scripts/reinforcement_learning/rsl_rl/train.py)。

## 6 用保存的模型推理并录视频

将下面路径换成刚才实际生成的检查点：

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Velocity-Flat-Unitree-Go2-Play-v0 \
  --checkpoint /absolute/path/to/model_299.pt \
  --num_envs 1 --headless --video --video_length 1000
```

`model_299.pt` 只是路径示意，实际文件名以输出为准。这个版本 `--video` 会启用相机，保存指定步数的视频后退出；录像通常写到检查点所在运行目录的 `videos/play/`。1 只狗便于观看。1000 个控制步在 50 Hz 配置下约对应 20 秒仿真时间，实际编码耗时另算。[推理与视频代码](https://github.com/isaac-sim/IsaacLab/blob/37ddf626871758333d6ed89cf64ad702aef127d0/scripts/reinforcement_learning/rsl_rl/play.py)。

Play 配置会关闭部分观察噪声或随机扰动。它适合展示，但不要把更容易的 Play 录像当作严格等条件评价。真正比较训练前后，应固定指令分布、地形、随机化、环境种子和评价时长，报告跌倒率、速度跟踪误差与原始日志，而不是只挑最漂亮的一段。

## 7 A3 与实体机器人暂不开展

本页固定的是 NVIDIA Isaac Sim/PhysX 和 CUDA 学习链路；A3 NPU 不能直接替代其中的图形与物理后端。当前没有 A3 调试环境，因此不提供未经验证的“改成 npu 即可运行”命令。以后可另选 CPU 物理环境或移植学习后端，属于另一项工作。

本教程只到仿真训练与推理。仿真里跑起来不等于可以直接把输出发送到实体 Go2。宇树官方提供独立部署路线，这次以源码逻辑为学习目标，不执行实体控制。
