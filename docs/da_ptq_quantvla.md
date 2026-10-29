# π₀.₅：QuantVLA 基础上的伪量化对照（2026-10-28，v2）

> 2026-10-29：`--profile cogact` 已更新为 `cogact_reference_v3`，直接复用本地 CogACT 数值函数，改为通道回退、W-QDQ 学生探测、原版 CSRC 和 inverse-beta OHB；需要新目录重新校准。架构适配、预算含义与 GPU 3 运行命令见 [CogACT 对照审计](da_ptq_cogact_alignment.md)。本文的 `quantvla` 仍是旧协议，不会自动切换。

本次更新补齐旧 π₀.₅ 实现没有包含的 QuantVLA 基础。默认 `--profile quantvla`；旧 artifact 仍可加载，旧算法可显式用 `--profile legacy`。**旧校准结果不能作为新配置的对照，原始 BF16 checkpoint 和数据无需重新下载。**

这是一份依据论文与公开 GR00T 实现编写的 **π₀.₅ 适配**。公开仓库没有可直接复用的 π₀.₅ adapter，因此不能称为官方 π₀.₅ 逐算子复现，也不能用这份新实现代表原论文已经报告过的结果。

依据：[QuantVLA v3](https://arxiv.org/abs/2602.20309v3)、[公开量化预处理](https://github.com/AIoT-MLSys-Lab/QuantVLA/blob/main/gr00t/quantization/duquant_preprocess.py)、[公开 ATM/OHB 实现](https://github.com/AIoT-MLSys-Lab/QuantVLA/blob/main/gr00t/atm/dit_atm.py)、[校准脚本](https://github.com/AIoT-MLSys-Lab/QuantVLA/blob/main/tools/calibrate_atm_dit.py)。同时检查了本地 `D:/VLA/QuantVLA.pdf`。

## 新的公共基础与参数

| 项目 | 新默认与含义 | 依据/适配说明 |
| --- | --- | --- |
| VLM 范围 | 语言模型所有 Q/K/V/O 和 MLP 线性层 | QuantVLA 的选择性布局 |
| action expert 范围 | 只量化 MLP；所有 Q/K/V/O 保持浮点 | QuantVLA 的关键注意力保护；W/A 均不做 QDQ |
| 固定边界保护 | expert 首 1 个 block 的 MLP；尾部至少 2 个完整 block | 边界数来自本地 adapter，**不是 QuantVLA 原论文规定的 block 数**。最后两个 block 的所有线性层均不做 W/A QDQ；尾部保护不可关闭，可用 skip 参数增加 |
| 其他浮点模块 | vision、embedding、norm、动作/时间输入输出投影 | 保留原 checkpoint 的 dtype（包括本来就是 FP32 的模块） |
| 混合精度回退 | 其余 expert MLP 候选层中 ceil(30% × 候选数)，默认 W16A16 | DA、等权、随机相同参数预算；30% 不等于全模型 BF16 比例 |
| 位宽 | W2A8，signed INT codebook 的浮点 QDQ | 用户要求的更低位宽；QuantVLA 附录 D 默认 W4A8，可用 `--weight-bits 4` |
| 旋转 | 输入/输出 block=64，zigzag 通道置换，输出 restore | 公开代码；用小协方差的 eigh 得到 SVD 等价正交基，符号/退化子空间不保证与原代码逐比特相同 |
| smoothing | 0.15 的权重通道能量混合，用于置换排序 | 跟随已公开代码；未将论文描述但该预处理代码未实现的对角 Λ 变换冒充为已实现 |
| W scale | 每输出通道，5 点 MSE clipping，scale=base/{0.5,0.75,1,1.25,1.5} | 公开代码；可切换 RTN |
| A scale | 99.9 percentile，每输入通道，最多 32 batches | 附录 D 的分位数与次数；signed/per-channel 跟随公开代码，与附录另一处 unsigned/per-token 文字有差异 |
| ATM/OHB 统计 | 最多 128 个校准 batches；log clamp=0.30，neutral band=0.03 | 附录 D；使用相同示范帧的无标签统计，非原论文在线仿真缓冲区 |
| 每任务数据限制 | 最多使用前 5 个示范 episodes，再采样 frame+chunk | 附录 D 的 5 trials 在离线示范上的适配，不能将 512 帧称作 512 轨迹 |
| DA/CSRC | 512 帧、batch1、16 gradient probes、相同结构先验/CSRC 参数 | 沿用现有 A.3 配置与架构适配，见旧说明中的公式、solver 和 Ω 定义 |

固定保护先应用，再在剩余等成本 MLP 候选集合中分配 30%。新默认的回退层 W16A16 会同时旁路旋转和 A8；`--fallback-activation-bits 8` 才是 W16A8。所有实验采用同一设置。

2026-10-28 补充硬约束：所有 profile/mode 均至少保护 action expert 最后两个完整 Transformer block，即使 `--skip-last-expert-blocks` 设为 0/1，或自定义了选层正则，也不能将它们重新量化。它们不占额外30%分配预算。实际π₀.₅加载策略保留 attention/MLP 的BF16权重，归一化及其原FP32参数仍沿用openpi精度策略；不是强制把归一化降为BF16。此约束保护精度，ATM/OHB仍可折叠到保持BF16的Q/O投影中。

新manifest记录 `mandatory_expert_tail_blocks`。加载时若发现最后两个block仍有W/A低比特分配会拒绝加载；把已QDQ的权重转成BF16不能恢复原值，需要从原checkpoint重新导出。默认v2原本已排除末尾两个MLP及所有expert注意力投影，这些合规产物可继续使用，不需仅为本硬约束重新校准。旧legacy产物若量化了末尾注意力则不符合新约束。

ATM 观察 **RoPE 后的真实 suffix-query/prefix+suffix-key logits**，支持 GQA 并排除 mask 元素。按 teacher/student 标准差比值缩放 action-expert Q 的每个 head；π₀.₅ 共享 prefix KV，因此只缩放 Q，避免修改共享 K。OHB 按 teacher/student 输出 RMS 比值缩放 O 的 weight/bias。两个比值均采用公开代码的乘法方向（论文部分公式的除法方向与公开实现不一致），使用附录 D 的 clamp。先折叠 ATM，再重测并折叠 OHB，是本适配的显式顺序。

CSRC 在**恢复原坐标后的** prefix K/V 输出上拟合；折叠时将校正矩阵共轭变换回存储的旋转坐标，之后按原分配位宽重新 QDQ。每个精度分配变体分别校准 CSRC、A scales、ATM/OHB，因为它们产生的学生误差不同；它们复用相同帧列表，不能无条件共享同一份校正系数。

最终保存浮点 QDQ weights、激活尺度、置换、旋转矩阵、融合 bias 和完整配置。ATM/OHB 已折叠到浮点注意力投影。重载不再做第二次 W-QDQ。**没有 packed INT2 storage/整数 GEMM；成功率可测，不能从此流程声称原生低比特加速或压缩比例。**

## 模式

| 模式 | 公共 QuantVLA 基础 | 动态回退 | CSRC |
| --- | --- | --- | --- |
| `quantvla` | 有 | 无额外 DA 回退，仅固定保护 | 无 |
| `da_ptq` | 有 | 结构加权梯度排名 | 有 |
| `equal_weight` | 有 | 等权梯度排名 | 有 |
| `random_seed_0/1/2` | 有 | 固定种子的随机等预算分配 | 有 |
| `uniform` | 仅相同固定保护/范围；无旋转、ATM/OHB | 无 | 无 |

双卡 runner 默认运行后五个有 DA 回退/CSRC 的变体：DA、等权和三个随机种子。需独立 QuantVLA 基线时在 `--variants` 中加 `quantvla`；该基线无额外 30% 回退，不能叫 matched-cost 基线。

## 当前服务器可执行命令

先把本地代码提交、推送到自己的 fork，再在服务器 `git pull --ff-only` 更新 `feat/da-ptq-w2a8-libero`。以下以已安装完成的 `~/openpi/.venv` 和 `libero-cpython` 为准；**不要再执行 uv sync 覆盖 cu128/Transformers 补丁**。

在 Bash 中设置路径与 CPU renderer。两张卡分别做模型运算，各自只看见一张卡，进程内编号都是 `cuda:0`。

```bash
cd /home/xusiyuan/openpi
conda activate libero-cpython

export LIBERO_PYTHON="$CONDA_PREFIX/bin/python"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export LIBGL_ALWAYS_SOFTWARE=1
export LP_NUM_THREADS=4
export OMP_NUM_THREADS=4
export JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export LIBERO_CONFIG_PATH="$PWD/.libero_config"
unset MUJOCO_EGL_DEVICE_ID

export PTQ_DATA_ROOT=/data2/xusiyuan
export RUNS="$PTQ_DATA_ROOT/runs/quantvla_controls_v2"
export PYTHONPATH="$PWD/src:$PWD/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"

test -f "$PTQ_DATA_ROOT/checkpoints/pi05_libero_pytorch/model.safetensors"
test -f "$PTQ_DATA_ROOT/datasets/physical-intelligence/libero/meta/info.json"
"$LIBERO_PYTHON" -c 'import platform; assert platform.python_implementation() == "CPython"'

COMMON=(
  --checkpoint-dir "$PTQ_DATA_ROOT/checkpoints/pi05_libero_pytorch"
  --dataset-root "$PTQ_DATA_ROOT/datasets/physical-intelligence/libero"
  --output-dir "$RUNS"
  --eval-python "$LIBERO_PYTHON"
  --gpus 6 7 --base-port 2414
  --profile quantvla --weight-bits 2 --activation-bits 8
  --bf16-ratio 0.30 --fallback-activation-bits 16
  --skip-first-expert-blocks 1 --skip-last-expert-blocks 2
  --calibration-samples 512 --batch-size 1 --probe-steps 16
  --rotation-block-size 64 --rotation-output-block-size 64
  --activation-percentile 99.9 --activation-calibration-batches 32
  --quantvla-calibration-steps 128 --max-calibration-episodes-per-task 5
  --quantvla-smoothing 0.15 --atm-log-clamp 0.30 --ohb-log-clamp 0.30
  --matching-neutral-band 0.03
)
```

先检查任务划分，然后校准五个变体（建议在 tmux 中执行以防 SSH 断开）：

```bash
.venv/bin/python scripts/run_da_ptq_controls.py "${COMMON[@]}" --dry-run
.venv/bin/python -u scripts/run_da_ptq_controls.py "${COMMON[@]}" --stage calibrate
```

校准结束先验证 BF16 与三个代表变体，Spatial 每任务 1 次并保存视频。BF16 也失败时先排查评估链路，不能直接将失败归因于量化。

```bash
.venv/bin/python -u scripts/run_da_ptq_controls.py "${COMMON[@]}" \
  --stage evaluate --variants da_ptq equal_weight random_seed_0 \
  --include-bf16 --suites libero_spatial --num-trials 1 --seeds 7 \
  --save-video --video-max-trials 1
```

检查视频与成功率后，正式跑四个 suite、每任务 50 次，默认五个变体共 10,000 episodes；每任务前两次保存视频：

```bash
.venv/bin/python -u scripts/run_da_ptq_controls.py "${COMMON[@]}" \
  --stage evaluate --num-trials 50 --seeds 7 \
  --save-video --video-max-trials 2
```

该正式测评命令**不会再校准**。用 `--stage all` 可在同一调用中校准缺失变体后评估；完整产物会检查 config、source 路径和 selection，一致才复用。源权重和数据必须保持不变，路径校验不等于内容哈希验证。改位宽、保护范围、预算、CSRC 或 matching 参数时用新的 `--output-dir`。`--stage evaluate` 遇到缺失/不匹配校准会报错。

## 每卡多个模型并发

增加 `--workers-per-gpu 3`，允许 6、7 号卡每卡同时运行最多三个独立模型。默认五个变体分配如下；若添加 `--include-bf16`，第六个 worker 同时评估原始 BF16，无需校准它。

| GPU | worker 0 | worker 1 | worker 2 |
| --- | --- | --- | --- |
| 6 | DA-PTQ | random seed 0 | random seed 2 |
| 7 | equal weight | random seed 1 | BF16（可选） |

```bash
# 尚未开始校准时：新启动的校准可以直接五个变体并发
.venv/bin/python -u scripts/run_da_ptq_controls.py "${COMMON[@]}" \
  --workers-per-gpu 3 --stage calibrate

# 完整校准已保存后：并发正式测评，复用原有产物
nohup .venv/bin/python -u scripts/run_da_ptq_controls.py "${COMMON[@]}" \
  --workers-per-gpu 3 --stage evaluate \
  --num-trials 50 --seeds 7 --save-video --video-max-trials 2 \
  > "$RUNS/formal_launcher.log" 2>&1 &
```

端口自动独立分配（默认2414起），日志为 `gpu6_worker0.log` 等，评估结果在相同名字的子目录。worker数只影响调度，不改变校准参数，已有完整产物可直接复用。默认五个变体只有五个活跃worker；添加BF16才会用满六个。

正在运行的旧launcher不会动态增加worker；**不要对同一校准目录重复启动第二套任务**。可让当前校准完成，再使用上面的并发测评。并发会同时增加CPU渲染与主存负载，总体耗时需要实际观察，不能保证随worker数线性缩短。

## 已完成校准：4、5、6、7号卡各3个实例

仅设置更多GPU/worker时，五个模型最多只有五个活跃实例。加 `--split-suites` 后，五个模型 × 四个suite变成20个独立评估任务，由12个GPU槽位动态领取；长suite优先，完成一个再取下一个。每个模型/suite只执行一次，50 trials/任务和配对seed保持不变。此模式只支持 `--stage evaluate`，不会加载校准数据或重新校准。

更新代码后，在已配置好OSMesa的服务器执行（`RUNS`必须指向已校准的目录，不要创建新的校准根目录）：

```bash
cd /home/xusiyuan/openpi
conda activate libero-cpython
export LIBERO_PYTHON="$CONDA_PREFIX/bin/python"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LIBERO_CONFIG_PATH="$PWD/.libero_config"
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export LIBGL_ALWAYS_SOFTWARE=1
export LP_NUM_THREADS=2
export OMP_NUM_THREADS=2
export JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONPATH="$PWD/src:$PWD/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"
export PTQ_DATA_ROOT=/data2/xusiyuan
export RUNS="$PTQ_DATA_ROOT/runs/quantvla_controls_v2"

LOG="$RUNS/launcher_12workers_$(date +%Y%m%d_%H%M%S).log"
nohup .venv/bin/python -u scripts/run_da_ptq_controls.py \
  --checkpoint-dir "$PTQ_DATA_ROOT/checkpoints/pi05_libero_pytorch" \
  --dataset-root "$PTQ_DATA_ROOT/datasets/physical-intelligence/libero" \
  --output-dir "$RUNS" --eval-python "$LIBERO_PYTHON" \
  --stage evaluate --split-suites \
  --gpus 4 5 6 7 --workers-per-gpu 3 --base-port 2514 \
  --num-trials 50 --seeds 7 --save-video --video-max-trials 2 \
  > "$LOG" 2>&1 &
echo "Launcher PID: $!"
echo "Launcher log: $LOG"
```

上述量化设置使用本页标准默认值；若校准时改过量化参数，仍需传入相同参数。五个变体总计10,000 episodes，与分片前完全相同。`--include-bf16`是额外的完整BF16对照，增加到24个分片/12,000 episodes，并非自动必选。

```bash
LOG_DIR=$(ls -dt "$RUNS"/evaluation_* | head -n 1)
tail -n 30 -F "$LOG_DIR"/gpu*_worker*.log
# 总汇总：$LOG_DIR/results.json
# 分片/视频：$LOG_DIR/shards/<variant>__<suite>/<variant>/<suite>/
```

GPU测得单实例8GB是当前占用，不等于全程峰值；每卡3个实例按该观测约24GB，保留余量。12个实例同时运行CPU渲染，线程先设为每进程2个，观察CPU利用率/吞吐再调整。并行评估的延迟不能作为独占GPU的推理效率对比。新评估目录不恢复旧评估中的部分episode；不要同时启动同一比较组的另一套正式测评。

```text
quantvla_controls_v2/
  quantized/
    calibration_selection.json
    da_ptq/ equal_weight/ random_seed_0/ random_seed_1/ random_seed_2/
      model.safetensors
      da_ptq.safetensors
      da_ptq.json
      assets/
  calibration_<UTC日期_PID>/
    request.json  gpu6.log  gpu7.log
  evaluation_<UTC日期_PID>/
    request.json  gpu6.log  gpu7.log  results.json
    gpu6/<variant>/<suite>/summary.json、*.npz、*.mp4
    gpu7/<variant>/<suite>/summary.json、*.npz、*.mp4
```

`--save-video` 现在会透传给仿真客户端，`--video-max-trials 0` 保存所有 trial。每次评估使用新目录；本 runner **不恢复**中断的单个 episode，但可以继续复用已经完整保存的校准模型。部分校准目录不会被自动覆盖。

本地验证是 CPU 小模型数值、实际 joint-Gemma 控制流和序列化测试；完整 π₀.₅、CUDA、OSMesa 与 LIBERO 成功率需在服务器验证。
