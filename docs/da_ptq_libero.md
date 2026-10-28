# π₀.₅ / LIBERO：W2A8 伪量化与 DA-PTQ 对照实验

> **2026-10-28 更新：本文保留 v1/legacy 的公式与适配说明。新默认已改为包含 QuantVLA 的 v2；层保护、旋转、ATM/OHB 参数和当前双卡命令见 [v2 运行说明](da_ptq_quantvla.md)。复现本文旧流程时，所有 `quantize_da_ptq.py` 命令均须添加 `--profile legacy`；旧结果不可与新默认混用。**

本实现以本地 openpi 的 PyTorch π₀.₅ 为基础。提供四种模式：

服务器上下载模型与数据的逐步命令见 [服务器下载说明](da_ptq_server_download.md)。

| 模式 | 权重精度分配 | CSRC |
| --- | --- | --- |
| `uniform` | 所有被选中的投影 W2，固定保护部分保留源精度 | 关闭 |
| `da_ptq` | 结构加权 flow loss 的梯度排名，前 30% 候选线性层保留 BF16 | 开启 |
| `equal_weight` | 七维损失权重全部为 1，仍按梯度排名且保留相同 BF16 预算 | 开启 |
| `random` | 同一候选集合中随机分配相同 BF16 预算 | 开启 |

所有模式使用同一个固定的校准采样列表、原始 checkpoint 和动作处理流程。
随机模式默认分配种子为 0、1、2。`equal_weight` 是去除结构先验的对照，仍然执行混合精度分配。

## 参数与论文的对应

按用户指定的补充材料 A.3 / 表 1 迁移以下默认值：

| 参数 | 默认值 |
| --- | --- |
| 量化 | W2A8（π₀.₅ 设置） |
| 校准样本、batch size | 512、1 |
| 空间 bins / warmup 样本数 | 6 / 128 |
| 梯度 probe batches | 16 |
| 候选范围 | action expert 的 MLP，保护最后 2 个 block 的 MLP |
| BF16 保留比例 | 候选线性层的 30%，数量取向上取整 |
| 阻尼 | 3e-4 |
| 两个 planar-mode 权重 / control-mode 权重 | 1.8 / 0.15 |
| 敏感性 gain | 1.6 |
| CSRC λf / λi / rank | 1.0 / 8.0 / 8 |
| 通道 scale clip | [0.5, 2.0] |

这些数值来自以 CogACT 为主的补充表，现按用户要求迁移到 π₀.₅；不能据此声称它们就是历史 π₀.₅ 实验的已核实配置。

实现中有以下明确的适配选择，均写入输出 manifest：

- **校准单位是 frame + action chunk**，不是 512 条完整轨迹；LIBERO action horizon 沿用 `pi05_libero` 的 10。任务之间轮转，再按末端 XYZ 的 6×6×6 空间 bins 平衡抽样。128 个随机 warmup 帧仅确定空间边界。`--unbalanced-sampling` 可改成普通随机帧采样。
- 量化对象是 PaliGemma 语言部分和 Gemma action expert 的 attention/MLP 线性投影。vision tower、embedding、动作输入/输出和时间投影维持源精度；最后两个 expert block 的 MLP 在所有模式中固定保护。`uniform` 也是同一量化范围内的 uniform，不能称为全模型所有参数均为 INT2。
- DA-MPA 按正文的**线性层**排名，默认不使用旧 CogACT 代码的逐输出通道排名。候选层必须具有相同权重参数量，保证所有模式高精度参数预算相同。报告实际候选比例和全模型高精度/排除参数比例，不能把 30% 当成全模型 BF16 比例。
- J 按正文的六个虚拟平面段与一个 generic-mode 构造；没有旧代码的物理 link-length 列表。先对校准集取平均再归一化。表 1 未定义 gain 的公式，这里显式采用 `clip(1 + 1.6*(s_hat-1), 0.05, 20)` 再归一化，manifest 同时保存原 prior 与实际 loss weights。
- π₀.₅ 使用原模型的 flow-matching unreduced MSE，只加权前七个真实动作维度；padding 维度不参与敏感性损失。校准时关闭图像增强，固定每个 batch 的 noise/time，policy 权重不做 optimizer update。
- CSRC 纠正 prefix 的 K/V 线性投影输出，位置在 RoPE 和 KV cache 插入之前。这是 π₀.₅ 接口的适配位置，不能照搬 CogACT projector。只收集有效 prefix tokens。
- CSRC 按正文 Eq.14 的 quartic 协方差目标进行 dense 优化，然后对 M-I 做 rank-8 SVD。正文未给出 solver/Ω 定义，这里使用 CPU float64 LBFGS（identity 初始化、strong-Wolfe、最多 64 iter），`Ω_ij=sqrt(r_i*r_j)`、`r=clip(var_FP/var_aligned_Q, 0.25, 4)`。这些是可追踪的新实现选择，不是捏造的历史配置。
- CSRC 同时拟合、逐投影折叠。折叠后重新量化到原分配的位宽，不保留低秩推理旁路。biasless 投影需要增加一个融合在线性算子内的 bias 才能实现非零平移；新增 bias 会单独记录，不声称参数数量完全不变。
- 本实现没有默认移植附录 A.4 的额外 pre-rotation；当前对照统一关闭该额外变换。若原 π₀.₅ 实验启用了旋转，应先明确其实现再加入独立消融，不能宣称当前实现逐算子复现了原 checkpoint。

## 运行前需要的文件

在支持 openpi 的 GPU 环境中，沿用该版本的依赖和 `transformers_replace` 补丁。CogACT 环境不能直接代替。

1. 已在 LIBERO 微调的 `pi05_libero` checkpoint，转换为 PyTorch 格式，包含 `model.safetensors` 与原始 `assets/`。
2. LeRobot 格式的 `physical-intelligence/libero` 数据。可用 `--dataset-root` 指定本地包含 `meta/info.json` 的数据目录。
3. LIBERO 评估环境（可使用 openpi 官方的独立 Python 环境）与初始化状态文件。

若源 checkpoint 为 JAX，使用官方转换脚本：

```bash
uv run examples/convert_jax_model_to_pytorch.py \
  --checkpoint_dir /data/checkpoints/pi05_libero \
  --config_name pi05_libero \
  --output_path /data/checkpoints/pi05_libero_pytorch
```

校准沿用 checkpoint 的归一化统计，不重新估计或替换原策略的统计。

## 校准与导出

先检查配置（不加载模型、不运行推理）：

```bash
uv run scripts/quantize_da_ptq.py \
  --checkpoint-dir /data/checkpoints/pi05_libero_pytorch \
  --output-dir runs/da_ptq_w2a8 \
  --mode all --dry-run
```

正式运行全部模式：

```bash
uv run scripts/quantize_da_ptq.py \
  --checkpoint-dir /data/checkpoints/pi05_libero_pytorch \
  --dataset-root /data/lerobot/physical-intelligence/libero \
  --output-dir runs/da_ptq_w2a8 \
  --mode all --random-seeds 0 1 2 --device cuda
```

`--mode all` 依次重新加载同一个原始 checkpoint，输出：

```text
runs/da_ptq_w2a8/
  calibration_selection.json
  uniform/
  da_ptq/
  equal_weight/
  random_seed_0/
  random_seed_1/
  random_seed_2/
```

如果已有可比的完整 DA-PTQ artifact，只新增对照即可：

```bash
uv run scripts/quantize_da_ptq.py \
  --checkpoint-dir /data/checkpoints/pi05_libero_pytorch \
  --dataset-root /data/lerobot/physical-intelligence/libero \
  --selection-file runs/reference/calibration_selection.json \
  --output-dir runs/controls --mode equal_weight

uv run scripts/quantize_da_ptq.py \
  --checkpoint-dir /data/checkpoints/pi05_libero_pytorch \
  --dataset-root /data/lerobot/physical-intelligence/libero \
  --selection-file runs/reference/calibration_selection.json \
  --output-dir runs/controls --mode random --random-seeds 0 1 2
```

旧模型若不是同一实现、位宽范围、精度预算和采样协议，不能直接拼入新对照表。脚本拒绝覆盖已经存在的实验目录。

每个 artifact 包含：

- `model.safetensors`：浮点存储的 QDQ 权重，保留标准模型 state 的键结构。
- `da_ptq.safetensors`：静态 A8 scales、必要时的融合 bias。
- `assets/`：原策略归一化统计。
- `da_ptq.json`：配置、分配图、原/有效动作权重、梯度分数、校准索引、CSRC 诊断和分阶段时间/峰值内存。

标准 `policy_config.create_trained_policy` 已接入自动识别/恢复，不会在加载时重复量化。已有普通 checkpoint 加载流程保留。

## 配对 LIBERO 评估

自动启动每个策略 server，使用相同任务、初态、seed 和逐请求噪声评估四个 suite：

```bash
uv run scripts/evaluate_da_ptq_matrix.py \
  --runs-dir runs/da_ptq_w2a8 \
  --output-dir runs/da_ptq_w2a8_eval \
  --bf16-checkpoint /data/checkpoints/pi05_libero_pytorch \
  --eval-python /path/to/libero/environment/bin/python \
  --num-trials 50 --seeds 7 --device cuda
```

排错时可以先用 `--variants equal_weight random_seed_0 --suites libero_spatial --num-trials 2`，但不能把这种小规模检查当成正式基准结果。

评估输出每个 suite 的 `summary.json`、每 episode 的动作/末端位置/姿态/夹爪/初态 `.npz`、policy noise seeds 和 round-trip 耗时，最终汇总 `results.json`。失败/超时保留在分母；服务器或环境异常单独标记并中止，不伪装成 0% 成功率。轨迹保存到各自终止时刻，后续漂移分析需明确定义终止后的统计，不自动补零或只筛成功例。

也可手动启动单个 server：

```bash
uv run scripts/serve_da_ptq.py --checkpoint-dir runs/da_ptq_w2a8/equal_weight --port 8000
```

随后在 LIBERO 环境中：

```bash
python examples/libero/eval_da_ptq.py \
  --suite libero_spatial --output-dir runs/equal_weight_eval --num-trials 50 --seeds 7
```

## 伪量化与真实部署指标

W2 使用 signed [-2,1] 代码范围、zero-point=0、逐输出通道五点 MSE scale；A8 使用逐输入通道静态 absmax scale，推理时 QDQ。**权重仍以浮点数保存，矩阵乘法仍是浮点 GEMM。** 因而 checkpoint 文件大小和实际 GPU 显存不会呈现 packed INT2 的压缩比，运行时间也不能当作原生 INT2/INT8 kernel 的加速结果。

本实现适合验证数值影响、DA-MPA / CSRC 消融与闭环成功率。真实部署效率需要单独接入低比特存储和执行后端。日志中的耗时和内存只描述当前伪量化实现，不与论文原生部署数字混用。

## 本地检查

数值核心只依赖 PyTorch、NumPy、safetensors 和 pytest；在仓库根目录：

```bash
python -m pytest tests/quantization --confcutdir=tests/quantization -q
```

覆盖 INT2 码本、A8、折叠后再量化、padding 梯度屏蔽、结构代理、同预算控制、空间采样、Eq.14 求解、四模式校准/导出/重载，以及实际 PI0Pytorch forward/sampler 与小型替代 backend 的集成。这个测试不替代真实 Gemma/SigLIP checkpoint 或 LIBERO 成功率测试。
