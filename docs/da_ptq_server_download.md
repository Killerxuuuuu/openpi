# 服务器下载 π₀.₅ checkpoint 与 LIBERO 数据

更新：2026-10-26。此文档只提供命令；本地未执行大文件下载或正式实验。

将当前**修改后的** `openpi/` 目录同步到 Linux 服务器；仅重新 clone 官方仓库不会包含本次加入的量化与对照脚本。下面命令在该服务器的 openpi 根目录执行。将 `/data/da-ptq` 换成你的服务器数据盘路径。

## 0. 准备 openpi 环境（服务器已有环境可以跳过）

使用官方支持的 Linux / Python 3.11 环境和可用的 PyTorch CUDA 安装。如果还没有 uv，可先在服务器当前 Python 环境安装：

```bash
python -m pip install --upgrade uv
python -m uv --version
```

下面的 `uv` 也可写成 `python -m uv`，避免 PATH 没包含命令安装目录的问题。安装后：

```bash
cd /path/to/your/modified/openpi
export UV_LINK_MODE=copy
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

按照该版本 openpi 的 PyTorch 要求安装 transformers 补丁。`UV_LINK_MODE=copy` 避免修改共享 uv hardlink 缓存。

```bash
uv run python -c "import pathlib, shutil, transformers; source=pathlib.Path('src/openpi/models_pytorch/transformers_replace'); target=pathlib.Path(transformers.__file__).parent; shutil.copytree(source,target,dirs_exist_ok=True); print(target)"
```

校准使用 PyTorch，JAX 只处理配置、数据和转换；让 JAX 留在 CPU：

```bash
export JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PTQ_DATA_ROOT=/data/da-ptq
mkdir -p "$PTQ_DATA_ROOT/checkpoints" "$PTQ_DATA_ROOT/datasets"
```

## 1. 下载已在 LIBERO 微调的 π₀.₅

官方来源：`gs://openpi-assets/checkpoints/pi05_libero/`。这对应 `pi05_libero` config；不需要先训练 `pi05_base`。

用 gsutil 隔离工具环境下载到明确的本地目录：

```bash
mkdir -p "$PTQ_DATA_ROOT/checkpoints/pi05_libero"
uv tool run --from gsutil gsutil -m cp -r \
  "gs://openpi-assets/checkpoints/pi05_libero/*" \
  "$PTQ_DATA_ROOT/checkpoints/pi05_libero/"
```

`uv tool run --from gsutil` 首次会安装 gsutil 工具依赖，不会把它混进模型环境。它等价于先前写法 `uvx --from gsutil`，不要求单独存在 uvx 命令。若服务器已安装 gsutil，可直接用 `gsutil -m cp -r ...`。

下载后应有：

```text
/data/da-ptq/checkpoints/pi05_libero/
  params/
  assets/
```

也可使用 openpi 自带下载器，路径由缓存设置决定：

```bash
export OPENPI_DATA_HOME="$PTQ_DATA_ROOT/openpi_cache"
uv run python -c "from openpi.shared import download; print(download.maybe_download('gs://openpi-assets/checkpoints/pi05_libero'))"
```

以上替代命令打印的目录是后续 `--checkpoint_dir` 参数；它和手动 gsutil 路径二选一即可。

## 2. 转换为当前量化脚本需要的 PyTorch checkpoint

```bash
uv run examples/convert_jax_model_to_pytorch.py \
  --checkpoint_dir "$PTQ_DATA_ROOT/checkpoints/pi05_libero" \
  --config_name pi05_libero \
  --output_path "$PTQ_DATA_ROOT/checkpoints/pi05_libero_pytorch" \
  --precision bfloat16
```

如果采用上一节的 openpi 下载器，替换这里的 `--checkpoint_dir` 为它打印的路径。转换脚本已修复 root/assets 拷贝位置，并按模型 config 判断 pi05，不再依赖目录名称。

转换后应有：

```text
/data/da-ptq/checkpoints/pi05_libero_pytorch/
  model.safetensors
  config.json
  assets/
    physical-intelligence/libero/norm_stats.json
```

以真实 assets 结构为准；必须保留源 checkpoint 中的全部 assets。不要用重新计算的统计覆盖它。

## 3. 下载 LIBERO 的 LeRobot 版本

数据仓库：`physical-intelligence/libero`，包含 Spatial、Object、Goal、LIBERO-10 四个 suite，网页总大小约 34.9 GB。

```bash
uv run python -c "import os; from huggingface_hub import snapshot_download; snapshot_download(repo_id='physical-intelligence/libero',repo_type='dataset',local_dir=os.path.join(os.environ['PTQ_DATA_ROOT'],'datasets','physical-intelligence','libero'))"
```

下载目录：

```text
/data/da-ptq/datasets/physical-intelligence/libero/
  meta/
    info.json
    episodes.jsonl
    tasks.jsonl
    ...
  data/
    chunk-000/
    chunk-001/
    ...
```

本版本的图像存储在 Parquet 中，`total_videos=0`，不要求存在 videos 目录。`--dataset-root` 指向包含 `meta/info.json` 的这一层。现有量化脚本使用 LeRobot 数据接口，不能把官方原始 HDF5 的目录直接传进去。

## 4. 检查完整性与路径

```bash
uv run python -c "import os,pathlib,json; root=pathlib.Path(os.environ['PTQ_DATA_ROOT']); ckpt=root/'checkpoints/pi05_libero_pytorch'; ds=root/'datasets/physical-intelligence/libero'; assert (ckpt/'model.safetensors').is_file(); assert list((ckpt/'assets').rglob('norm_stats.json')); info=json.loads((ds/'meta/info.json').read_text()); print('CHECKPOINT:',ckpt); print('DATASET:',ds); print('episodes:',info['total_episodes'],'frames:',info['total_frames'],'tasks:',info['total_tasks'])"
```

当前数据页面记录 1693 episodes、273465 frames、40 tasks。与本地下载版本不一致时，应核对数据 revision，不要直接混入已有实验。

## 5. 之后运行实验时传入这两个路径

```bash
uv run scripts/quantize_da_ptq.py \
  --checkpoint-dir "$PTQ_DATA_ROOT/checkpoints/pi05_libero_pytorch" \
  --dataset-root "$PTQ_DATA_ROOT/datasets/physical-intelligence/libero" \
  --output-dir "$PTQ_DATA_ROOT/runs/w2a8" \
  --mode all --dry-run
```

上面的 `--dry-run` 只展示配置；准备好正式执行时再去掉它。完整量化/评估命令见 [DA-PTQ 运行说明](da_ptq_libero.md)。闭环评估还需要另外准备 LIBERO 仿真环境及子模块；下载演示数据不等于安装了模拟器。

## 官方来源

- openpi PyTorch / 转换：https://github.com/Physical-Intelligence/openpi#pytorch-support
- pi05_libero checkpoint / 基准：https://github.com/Physical-Intelligence/openpi/blob/main/examples/libero/README.md
- 演示数据：https://huggingface.co/datasets/physical-intelligence/libero
