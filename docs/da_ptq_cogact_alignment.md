# CogACT / openpi Implementation Audit and Partial Alignment

Date: 2026-10-29. Reference: the local `DA-PTQ/vla/quantvla_ptq.py`, `DA-PTQ/scripts/quantize_quantvla.py`, and `DA-PTQ/README.md`. This is a comparison of executable local code, not a reconstruction of historical paper results.

## Main finding

The CogACT LLM wrapper explicitly excludes Q/K/V/O projections (`apply_quantvla_wrappers_to_llm`), despite its include-regex and comment mentioning attention. The earlier openpi `quantvla` profile quantized these language projections at W2. This is a substantial experimental difference and a plausible contributor to the observed collapse. It is not yet a demonstrated cause: only server evaluation can establish whether the new layout improves success.

`--profile cogact` now provides a separate, partially aligned configuration. The existing `quantvla` and `legacy` profiles remain explicit and loadable when they satisfy the mandatory expert-tail guard. Do not mix their calibration artifacts or results with the new profile.

## Code comparison

| Component | Local CogACT code / README recipe | Earlier openpi `quantvla` profile | New openpi `cogact` profile |
| --- | --- | --- | --- |
| Precision | W4A8 recipe | W2A8 experiment | Configurable; validate W4A8 first, then test W2A8 |
| Language Q/K/V/O | Excluded by the actual wrapper predicate | Quantized | Excluded from W/A QDQ |
| Expert attention | Floating | Floating | Floating |
| Expert boundaries | README: first 0, last 2 MLP blocks; dataclass: first 1, last 2 | First 1, last 2 | First 0 by default; final 2 complete blocks always protected |
| Input/output rotation block | README: 16/16, both rotations enabled, restore output | 64/64 | 16/16, both rotations enabled, restore output |
| Activation scale | Signed per-input-channel, 99.9th percentile, first 64 observed calls, at most 4096 rows/call | First 32 calls, no 4096-row cap | 64 calls and the same 4096-row cap |
| Non-adaptive weight QDQ | Per-output-channel RTN or MSE, with scales `base / [0.5,0.75,1,1.25,1.5]` | Same reciprocal MSE grid | Same shared formula; parity-tested against the local source |
| Sensitivity-adaptive weight path | Per-output-row RTN, even when the general mode is MSE | Per-whole-layer MSE | Whole-layer path retained; not the CogACT adaptive-row RTN path |
| High-bit fallback | W16 rows retain A8 | W16A16 layers | W16A8 layers by default; fixed protected blocks bypass both W/A QDQ |
| Statistical matching budget | Full calibration pass; README requests 512 batches | At most 128 batches | Full configured frame/batch pass by default (512 batches for 512 frames, batch size 1) |
| Structural surrogate | Seven virtual segments with link lengths `0.35,0.30,0.25,0.20,0.15,0.10,0.08`; per-batch weights | Six equal planar segments plus generic channel; dataset-average weights | Local seven-segment formula and per-batch loss weights, parity-tested |
| Surrogate gain | Clip `1 + gain * (normalized_score - 1)` to `[0.05,20]`, no subsequent mean normalization | Additional mean normalization after gain | Matches the local gain behavior |
| Allocation unit | Global output-channel ranking; different row widths can imply different parameter costs | Equal-cost whole linear layers | Equal-cost whole layers retained for the existing rebuttal controls |
| Sensitivity probe model | Already W-QDQ-wrapped student, A quantization disabled | Full-precision teacher | Full-precision teacher retained; not an exact CogACT ranking reproduction |
| ATM clipping | Ratio clipped to `[0.5,2]`, linear neutrality threshold 0.02 | Log clamp 0.30, log neutrality 0.03 | Ratio clipped to `[0.5,2]`, linear neutrality 0.02 |
| ATM folding | Multiply fused Q and K by `sqrt(alpha)` | Scale expert Q only | Scale Q only: pi0.5 uses GQA and shared prefix keys, so scaling expert K alone would miss prefix-key logits |
| OHB direction | Computes teacher/student RMS, then folds **inverse beta** into O | Folds **beta** into O | Retains multiplication toward the teacher; does not copy the inverse-beta behavior |
| Attention statistic aggregation | Mean of per-call standard deviations/RMS | Pooled moments over captured calls | Pooled moments retained |
| Compensation target | Separate condition projection `action_model.net.z_embedder.linear` | Quantized prefix K/V before RoPE/cache | Floating prefix K/V before RoPE/cache; compensates input drift from quantized language MLPs |
| Compensation solver | Optional grouped affine or optional dense linear-system solver | Dense quartic covariance fit using LBFGS, followed by rank truncation | Existing covariance solver retained and explicitly recorded; not identical to either local solver |
| Actual compensation enablement | README defaults affine/dense to false; entrypoint forcibly disables dense compensation and its FP16 bypass | CSRC enabled for the three allocation strategies | CSRC remains enabled for the requested controls; it is not silently disabled to match a local entrypoint default |
| Rotation numerics | SVD bases, usually stored/operated in source dtype | Covariance-eigh bases and FP32 rotation math | Existing eigh/FP32 path retained; floating equivalence is tested, but basis signs and low-bit outputs need not match SVD bit-for-bit |
| Loss and conditioning | Diffusion epsilon prediction with a separate action-conditioning interface | pi0.5 flow matching with joint/cached attention | pi0.5 architecture and flow target retained |

The retained differences are intentional and visible. In particular, this profile does **not** claim that channel ranking and layer ranking are equivalent, that the two compensation solvers solve the same objective, or that the historical CogACT paper results are reproduced. No original CogACT algorithm code was changed by this audit.

## Why floating K/V still receives compensation

Protecting a K/V projection from QDQ does not prevent its inputs from changing when preceding language MLPs are quantized. The new profile collects teacher/student moments at these floating projections and folds the fitted affine correction into their floating weights and bias. It does not wrap them in A8/W2 again. Any added fused biases are serialized separately and restored without introducing a low-rank inference branch.

## Validation

- CPU tests import the actual local CogACT reference file and compare W2/W4/W8 RTN/MSE QDQ, percentile observers (including the 4096-row cap), and the per-batch structural weighting formula.
- Tiny models exercise the actual openpi joint-attention control flow, gradients, mixed allocation, floating-K/V compensation, export and exact reload for the old and new profiles.
- The tests are numerical and implementation checks. They do not establish full-checkpoint accuracy, LIBERO success, native INT2 execution, or GPU latency.
- Tests that require the separate local CogACT checkout are skipped if that reference file is absent, such as in a standalone openpi checkout.

## Server instructions: fresh calibration required

Update the fork on the server first. Keep the existing CPython/torch-cu128/Transformers patch and separate LIBERO environment. Use fresh commands rather than the old `COMMON` array: its explicit block=64, first-block=1, or fallback-A16 arguments would override the new defaults.

```bash
cd /home/xusiyuan/openpi
export PTQ_DATA_ROOT=/data2/xusiyuan
export JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Inspect the resolved new settings without loading a model.
.venv/bin/python scripts/quantize_da_ptq.py \
  --checkpoint-dir "$PTQ_DATA_ROOT/checkpoints/pi05_libero_pytorch" \
  --dataset-root "$PTQ_DATA_ROOT/datasets/physical-intelligence/libero" \
  --output-dir "$PTQ_DATA_ROOT/runs/cogact_layout_w4a8" \
  --profile cogact --weight-bits 4 --mode da_ptq --dry-run

# First validate the shared path at the local CogACT recipe's W4A8 precision.
CUDA_VISIBLE_DEVICES=6 .venv/bin/python -u scripts/quantize_da_ptq.py \
  --checkpoint-dir "$PTQ_DATA_ROOT/checkpoints/pi05_libero_pytorch" \
  --dataset-root "$PTQ_DATA_ROOT/datasets/physical-intelligence/libero" \
  --output-dir "$PTQ_DATA_ROOT/runs/cogact_layout_w4a8" \
  --profile cogact --weight-bits 4 --mode da_ptq --device cuda:0
```

After calibration, use the existing LIBERO CPU-rendering environment:

```bash
conda activate libero-cpython
export LIBERO_PYTHON="$CONDA_PREFIX/bin/python"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LIBERO_CONFIG_PATH="$PWD/.libero_config"
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export LIBGL_ALWAYS_SOFTWARE=1
export LP_NUM_THREADS=8
export OMP_NUM_THREADS=2
export PYTHONPATH="$PWD/src:$PWD/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"

CUDA_VISIBLE_DEVICES=6 .venv/bin/python scripts/evaluate_da_ptq_matrix.py \
  --runs-dir "$PTQ_DATA_ROOT/runs/cogact_layout_w4a8" \
  --output-dir "$PTQ_DATA_ROOT/runs/cogact_layout_w4a8/spatial_smoke" \
  --variants da_ptq \
  --bf16-checkpoint "$PTQ_DATA_ROOT/checkpoints/pi05_libero_pytorch" \
  --eval-python "$LIBERO_PYTHON" --suites libero_spatial \
  --num-trials 1 --seeds 7 --port 2914 --device cuda:0 \
  --save-video --video-max-trials 1
```

If W4A8 also fails while BF16 succeeds, stop and inspect the common quantization/compensation path rather than running a full experiment. W4A8 success would narrow the investigation but would not establish that W2A8 works.

Once the shared path has been validated, calibrate the requested W2A8 controls into another new directory:

```bash
export RUNS="$PTQ_DATA_ROOT/runs/cogact_layout_w2a8_controls"
.venv/bin/python -u scripts/run_da_ptq_controls.py \
  --checkpoint-dir "$PTQ_DATA_ROOT/checkpoints/pi05_libero_pytorch" \
  --dataset-root "$PTQ_DATA_ROOT/datasets/physical-intelligence/libero" \
  --output-dir "$RUNS" --profile cogact --weight-bits 2 \
  --gpus 6 7 --workers-per-gpu 1 --stage calibrate

# Check W2A8 before scheduling the formal run; each GPU hosts at most two models.
.venv/bin/python -u scripts/run_da_ptq_controls.py \
  --checkpoint-dir "$PTQ_DATA_ROOT/checkpoints/pi05_libero_pytorch" \
  --dataset-root "$PTQ_DATA_ROOT/datasets/physical-intelligence/libero" \
  --output-dir "$RUNS" --profile cogact --weight-bits 2 \
  --eval-python "$LIBERO_PYTHON" --gpus 6 7 --workers-per-gpu 2 \
  --stage evaluate --split-suites --include-bf16 \
  --suites libero_spatial --num-trials 1 --seeds 7 --base-port 2920 \
  --save-video --video-max-trials 1
```

Only after the W2A8 checks pass, use the same final command with all four suites, `--num-trials 50`, and `--video-max-trials 2`. Reuse the **new** calibration artifacts and the **same profile**. Keep old all-zero runs for diagnosis; do not merge them into a successful run or present this change as a verified recovery before measurement.
