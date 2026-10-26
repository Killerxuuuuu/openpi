from pathlib import Path
import sys

# The numerical tests intentionally do not import src/openpi/conftest.py, whose
# JAX GPU fixtures are unrelated to the PyTorch-only quantization core.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
