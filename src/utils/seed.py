"""Fixed random seed utilities to ensure reproducible runs across all modules."""

import os
import random
import numpy as np

def set_seed(seed: int = 42) -> None:
    """Set random seeds for python random, numpy, os, and PyTorch (if available).
    
    Args:
        seed: Integer seed value (default: 42).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
