python
"""
Centralised random seed control for reproducibility.

All training and evaluation scripts import set_seed() from this module
to ensure that every stochastic process in the pipeline uses the same
seed (42).

Usage:
    from set_seed import set_seed
    set_seed(42)
"""

import os
import random

import numpy as np


def set_seed(seed: int = 42) -> None:
    """
    Set the random seed for Python, NumPy, and PyTorch (CPU + CUDA).

    Also configures environment variables to reduce non-determinism in
    CUDA operations. Note that complete determinism on GPU may require
    additional flags (see PyTorch documentation on reproducibility).
    """
    # Python
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch (imported lazily so the module works even if torch is not installed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Reduce non-determinism (may slightly slow down training)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Reproducible CUDA ops (available in torch >= 1.8)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            # Not all operations support deterministic mode;
            # warn_only=True prevents a hard failure.
            pass
    except ImportError:
        # torch is not installed in the classical-baselines-only environment
        pass


if __name__ == "__main__":
    set_seed(42)
    print("Random seed set to 42 for Python, NumPy, and PyTorch.")