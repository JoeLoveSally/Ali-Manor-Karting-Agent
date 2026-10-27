# 安装、训练与复现

## 环境

使用 Linux / WSL2、Python 3.10+。创建虚拟环境后安装根目录的 `requirements.txt`。GPU 训练需要与驱动匹配的 CUDA 版 PyTorch，确认 `torch.cuda.is_available()` 返回真。

手机开启 USB 调试并授权；WSL2 用户需先完成 USB 转发。当前客户端使用 scrcpy 4.1 协议，匹配的 `scrcpy-server` 必须位于解析后的 scrcpy 可执行文件同目录。通过 `adb devices` 确认设备在线，保持亮屏并停在出发页。

现有标定来自1440×3200竖屏设备，传输帧为720×1600；其他设备布局需核对几何标定与界面识别。

## 单局运行

从仓库根目录执行：

```bash
python tools/drive.py models --checkpoint best.pt --serial YOUR_DEVICE_SERIAL --max-time 150
```

如果停在上一局结算页，加入 `--prepare`，程序识别后领取普通奖励并进入下一局。每次只跑一局，结束停在结算页，记录写入 `runs/drive-时间戳/`。

## 从零训练与恢复

```bash
DEVICE=cuda ENVS=256 ITERS=2000 OUT=outputs/my-run bash scripts/train.sh
```

默认使用256个环境、每轮64步，最多2000轮。脚本拒绝复用已有输出目录，保证新实验从随机权重开始。可用 `GPU_MEMORY_GB` 调整本进程的 PyTorch CUDA 分配器上限，默认2 GiB；这不包含全部 CUDA 上下文显存，也不会替其他程序预留资源。共享显卡上先检查剩余资源。

同一实验中断后，可按脚本内相同参数调用 `python -m kart.ppo`，添加 `--resume 输出目录/checkpoint.pt` 恢复模型、优化器和环境状态。新实验不导入旧实验权重。

`configs/train-params.json` 用于仿真训练；`artifacts/sim_params.json` 用于真机状态估计。两者用途不同。

## 独立评测与回放

```bash
python tools/eval_survival.py \
  --params configs/train-params.json --device cuda --envs 128 --seed 20261101 \
  --models baseline=models/best.pt best=models/best.pt \
  --suites standard16 long30 bends16 ramp relief middle16 level645 \
  --case configs/level645-case.json --robust-scenes --sampled-diagnostics \
  --out outputs/evaluation

python tools/collect_policy_replays.py \
  --evaluation outputs/evaluation --model models/best.pt \
  --params configs/train-params.json --device cuda --balanced \
  --out outputs/replays

python tools/render_policy_replays.py outputs/replays
```

工具要求一个名为 `baseline` 的对照。示例用同一权重核验管线，不能据此声称相对提升；正式比较请替换基线权重。命令中的局部场景标识是兼容现有代码的内部名称。

渲染需要中文字体，默认使用系统的 `NotoSansCJK-Regular.ttc`；也可设置 `KART_FONT=/字体路径/字体文件`。回放输出包括 HTML、MP4 和输入记录。

## 真机记录复算

```bash
python tools/export_drive.py runs/drive-时间戳
python tools/replay_drive.py runs/drive-时间戳 --out outputs/phone-replay
```

复算使用记录的画面与动作验证可重现性，不等于一次新的闭环测试。

## 测试

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```
