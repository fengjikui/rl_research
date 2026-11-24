# 01 在 AutoDL 上从零配置 Isaac Lab

目标是在单卡 Linux 实例上跑通宇树 Go2 平地仿真。选用仓库的教学版本 **Isaac Sim 5.1.0、Isaac Lab 2.3.2、RSL-RL 3.1.2、Python 3.11、PyTorch 2.7.0/cu128**。先验证 CUDA 和 Vulkan 两条链路，再进入第二篇启动模拟器和 PPO。[Isaac Lab 2.3.2 安装指南](https://isaac-sim.github.io/IsaacLab/v2.3.2/source/setup/quickstart.html)

| 软件 | 这次用它做什么 |
| --- | --- |
| NVIDIA 驱动、CUDA、Vulkan/EGL | 驱动由宿主机提供。PyTorch 用 CUDA 做计算，Isaac Sim 用 Vulkan/EGL 做无桌面渲染 |
| Isaac Sim | 加载 Go2 的 USD 资产，用 PhysX 模拟关节、接触和重力，渲染相机视频 |
| Isaac Lab | 提供 Go2 平地任务、速度指令、观察、动作、奖励和终止条件 |
| RSL-RL + PyTorch | 批量采集轨迹，计算优势，更新网络并保存检查点 |
| uv | 准备独立的 Python 3.11 环境并安装依赖 |

## 1. 选实例并检查两块盘

本次实例使用 Ubuntu 22.04 x86_64、单张 RTX 4090 24 GB、30 GB 系统盘和 50 GB 数据盘。开机后在 AutoDL 终端执行以下命令，或在本机用控制台给出的 SSH 命令登录。后续命令均在服务器执行。

```bash
uname -m
cat /etc/os-release
ldd --version | head -1
nvidia-smi
df -h / /root/autodl-tmp
cat /sys/fs/cgroup/cpu.max /sys/fs/cgroup/memory.max
head -1 /sys/fs/cgroup/cpu.pressure
```

确认能看到 RTX 4090，并从 `nvidia-smi` 读取实际驱动、显存、负载和温度。控制台截图与实际实例的驱动版本可能不同，本次实测为 **580.76.05**，高于 Isaac Sim 5.1 在 Linux 上的推荐值 580.65.06。`nvidia-smi` 的 *CUDA Version* 表示宿主机驱动支持的 CUDA 版本，稍后用 `torch.version.cuda` 检查 PyTorch 用的版本。容器中的 `free -h` 可能显示宿主机的总内存，实例限额以 cgroup 为准。[Isaac Sim 5.1 系统要求](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html)

把实验放到 AutoDL 数据盘，给系统盘留下安装系统库的空间。50 GB 先用于单次实验，装完再用 `df -h` 复核空间。释放实例前下载需要保留的记录。[AutoDL 存储说明](https://backup.autodl.com/docs/env/)

```bash
export GO2_ROOT=/root/autodl-tmp/go2-ppo
mkdir -p "$GO2_ROOT"/{cache,tmp,runtime}
chmod 700 "$GO2_ROOT/runtime"
cd "$GO2_ROOT"
```

## 2. 验证 NVIDIA 的无桌面 Vulkan

AutoDL 实例可以无显示器运行。录像前仍需检查容器能否通过 NVIDIA EGL 枚举显卡。安装测试命令和图形库，对实验目录生成独立 ICD 配置。

```bash
apt-get update
apt-get install -y --no-install-recommends \
  vulkan-tools libvulkan1 libegl1 libsm6 libxt6 libglu1-mesa

ls -l /lib/x86_64-linux-gnu/libEGL_nvidia.so.0
sed 's|libGLX_nvidia.so.0|/lib/x86_64-linux-gnu/libEGL_nvidia.so.0|' \
  /etc/vulkan/icd.d/nvidia_icd.json > "$GO2_ROOT/nvidia-egl.json"

export VK_ICD_FILENAMES="$GO2_ROOT/nvidia-egl.json"
export XDG_RUNTIME_DIR="$GO2_ROOT/runtime"
vulkaninfo --summary
```

本次输出包含 `deviceName = NVIDIA GeForce RTX 4090` 和 `driverName = NVIDIA`。无桌面时跳过 surface 信息的提示可以接受。若只看到软件渲染设备，检查 EGL 库与 ICD 路径。若实例没有现成 NVIDIA ICD，按 AutoDL 指南创建。上面的 `libxt6` 和 `libglu1-mesa` 提供 Isaac Sim 启动时要加载的 `libXt.so.6` 与 `libGLU.so.1`。最后以第二篇成功生成 MP4 为准。[AutoDL Vulkan 配置](https://api.autodl.com/docs/vulkan/)

## 3. 建独立的 Python 3.11 环境

AutoDL 基础镜像附带的 PyTorch 用 Python 3.12。这里用镜像里的 Miniconda Python 启动 uv，再把 Python 3.11 和新环境装到数据盘。已有 uv 时，可用现成的可执行文件替代。

```bash
cd "$GO2_ROOT"
export UV_CACHE_DIR="$GO2_ROOT/cache/uv"
export UV_PYTHON_INSTALL_DIR="$GO2_ROOT/python"
export TMPDIR="$GO2_ROOT/tmp"
export UV_CONCURRENT_DOWNLOADS=4 UV_CONCURRENT_INSTALLS=2 UV_CONCURRENT_BUILDS=1
export UV_LINK_MODE=hardlink

/root/miniconda3/bin/python -m pip install \
  --index-url https://mirrors.aliyun.com/pypi/simple \
  --no-cache-dir --target "$GO2_ROOT/bootstrap-mirror" uv==0.13.9
uv="$GO2_ROOT/bootstrap-mirror/bin/uv"
"$uv" venv --python 3.11 --managed-python "$GO2_ROOT/venv"
python="$GO2_ROOT/venv/bin/python"
"$python" -V
```

从新终端继续时，先恢复上面的变量。下面用 `"$uv" pip install --python "$python"` 明确安装目的地。

## 4. 安装并实测 PyTorch CUDA

官方 cu128 下载慢时，可以从阿里云 PyTorch 专用镜像取同一构建。本次使用下面这条命令，两个 wheel 的哈希取自官方索引。普通 Python 依赖用国内 PyPI 镜像，清华源也可作为这部分的备选。GitHub 能直连时，保持默认网络设置。[PyTorch 历史版本](https://pytorch.org/get-started/previous-versions/) · [AutoDL 学术资源加速](https://www.autodl.com/docs/network_turbo/)

```bash
"$uv" pip install --python "$python" \
  --default-index https://mirrors.aliyun.com/pypi/simple \
  'https://mirrors.aliyun.com/pytorch-wheels/cu128/torch-2.7.0%2Bcu128-cp311-cp311-manylinux_2_28_x86_64.whl#sha256=c4bbc0b4be60319ba1cefc90be9557b317f0b3c261eeceb96ca6e0343eec56bf' \
  'https://mirrors.aliyun.com/pytorch-wheels/cu128/torchvision-0.22.0%2Bcu128-cp311-cp311-manylinux_2_28_x86_64.whl#sha256=f3ac527d58b4c2043eb8d9e29fc56cd1751f36f2aaa6dc75e34ec54c951bcb9c'

"$python" - <<'PY'
import torch

x = torch.ones(1, device="cuda")
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(), (x + 1).item())
PY
```

本次实测输出 `2.7.0+cu128 12.8 NVIDIA GeForce RTX 4090 2.0`。这一步实际在 GPU 上执行了运算。

## 5. 安装 Isaac Sim、Isaac Lab 和 RSL-RL

Isaac Sim 的额外索引指向 NVIDIA。包含 `extscache` 会下载比较大的 Omniverse 扩展缓存，首次安装需要时间。首次使用需由操作者阅读并决定是否接受 Omniverse 许可协议。[Isaac Sim Python 安装和许可](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_python.html)

```bash
"$uv" pip install --python "$python" \
  --default-index https://mirrors.aliyun.com/pypi/simple \
  --index https://pypi.nvidia.com 'isaacsim[all,extscache]==5.1.0'

git clone --depth 1 --branch v2.3.2 https://github.com/isaac-sim/IsaacLab.git
git -C IsaacLab rev-parse HEAD

printf '%s\n' 'torch==2.7.0+cu128' 'torchvision==0.22.0+cu128' \
  'isaacsim==5.1.0' > constraints.txt
printf '%s\n' 'setuptools==80.9.0' > build-constraints.txt
"$uv" pip install --python "$python" \
  --default-index https://mirrors.aliyun.com/pypi/simple -c constraints.txt \
  --build-constraint build-constraints.txt \
  -e IsaacLab/source/isaaclab \
  -e IsaacLab/source/isaaclab_assets \
  -e IsaacLab/source/isaaclab_tasks \
  -e 'IsaacLab/source/isaaclab_rl[rsl-rl]' moviepy
"$uv" pip check --python "$python"
```

预期 Git 提交是 `37ddf626871758333d6ed89cf64ad702aef127d0`。可编辑安装让训练时导入这份源码，方便在源码里查看观察、奖励和运行配置。RSL-RL 版本由 Isaac Lab 的 `rsl-rl` 扩展锁定。构建环境固定 `setuptools` 80.9.0，是因为 `flatdict` 4.0.1 的构建脚本仍要导入 `pkg_resources`。

这组固定版本的 `pip check` 会报告一条已知冲突。Isaac Sim 锁定的 FastAPI 要求较早的 Starlette，Isaac Lab 锁定 Starlette 0.49.1。本实验保留 Isaac Lab 的 Starlette 版本，使用无桌面仿真和文件录像，验收依赖实际短测。遇到其他依赖冲突时，查看具体包名和版本再处理。[Isaac Lab 上游记录](https://github.com/isaac-sim/IsaacLab/issues/4145)

下面的命令直接读取包元数据来核对版本。

```bash
"$python" - <<'PY'
from importlib.metadata import version

for package in ("torch", "isaacsim", "isaaclab", "rsl-rl-lib"):
    print(package, version(package))
PY
df -h / "$GO2_ROOT"
```

本次安装日志可用 `tail -f /root/autodl-tmp/go2-ppo/setup.log` 实时查看。退出 `tail` 不影响安装。阅读时留意 `Downloading` 与 `Downloaded`，缓存大小包括解包文件，无法直接换算成完成比例。

继续到 [02 跑短测、正式训练和回放](02_Go2从短测到回放.md)。
