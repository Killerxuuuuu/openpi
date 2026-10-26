"""Post-training fake quantization for the PyTorch pi0.5 policy.

The numerical core requires only PyTorch. Dataset/model integration is imported
lazily so the algorithms can also be verified on a CPU-only machine.
"""
