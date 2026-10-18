# Linux GPU 持续负载与编号辨认

脚本 `tests/manual/linux_gpu_identify.py` 可单独复制到服务器，无需安装 FlyAgent。
使用你已有的 CUDA 版 PyTorch 环境，额外依赖仅为 `psutil`，系统需有 `nvidia-smi`。
脚本不安装依赖、不调用 API、不下载模型、不修改驱动功耗和温控。

## 运行

```bash
conda activate 你的CUDA环境
# 仅缺少 psutil 时安装：
python -m pip install psutil

# 在仓库根目录执行：
python tests/manual/linux_gpu_identify.py --gpus 0,2,3

# 如果只复制了这个文件，在文件所在目录执行：
python linux_gpu_identify.py --gpus 0,2,3
```

GPU 参数必填，是当前 `nvidia-smi` 的物理编号。脚本启动时打印编号、UUID、PCI 地址，
然后按 UUID 隔离每个子进程；不依赖之前运行的编号。没有指定的卡不启动负载。
每张卡持续执行矩阵乘法，三个矩阵共约 972 MiB，CUDA 上下文和计算库还会额外占用显存。
目标是高计算利用率，实际利用率和温升以监控为准，不保证始终 100%。

**不设置运行期限或温度停止阈值。按 Ctrl+C 停止并释放本次负载。**
正常退出时显示 `All owned load workers exited.`；显卡自身的温控保护保持原样。
worker 会记录解释器、PyTorch/CUDA 版本及 CUDA 可用性；不可用时退出，不回退 CPU。

每两秒显示目标卡的利用率、温度、功耗和显存。功耗字段不支持时显示 None，
关键监控失效、worker 异常或发现新的外部实验占用时，停止本次全部负载并报告错误。
目标卡存在其他实验时拒绝启动；仅明确识别的 G 类型 Xorg/Xwayland 桌面进程允许共存。
检查不是 GPU 独占锁，运行期间仍会持续检查外部进程。

## SSH 断开与停止

前台运行时，终端挂断通常会触发清理；需要断开 SSH 后继续，先进入专用 tmux：

```bash
tmux new -s gpu-identify
python linux_gpu_identify.py --gpus 0,2,3
# Ctrl+B，然后 D：离开 tmux，负载继续。
# 返回后按 Ctrl+C 停止：
tmux attach -t gpu-identify
```

手动终止父进程时，子进程也会检查父进程身份并退出；设备或驱动卡死时不能保证
Python 立即释放 GPU。脚本只终止自己创建的 worker，不按名称或显卡批量杀进程。

用于辨认实体显卡时，将 UUID、PCI 插槽信息与现场位置交叉核对。
相邻卡会互相传热，不能仅凭“最凉的一张”就确认是未加载的 GPU 1。
