"""Configuration constants and paths for Second Take.

Rules adhered to:
- 16 kHz mono audio
- 10 ms hop (160 samples at 16 kHz)
- 25 ms window (400 samples at 16 kHz)
- CPU by default, GPU optional
- Fixed random seed: 42
- Anomaly threshold: |z| >= 2.0
"""

from pathlib import Path
import os

# Project Root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data Directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
CORRUPTED_DATA_DIR = DATA_DIR / "corrupted"
HELD_OUT_DATA_DIR = DATA_DIR / "held_out"

# Audio Settings
TARGET_SAMPLE_RATE: int = 16000  # 16 kHz mono
CHANNELS: int = 1               # Mono

# Frame parameters (STFT / Feature extraction)
WINDOW_MS: float = 25.0         # 25 ms analysis window
HOP_MS: float = 10.0            # 10 ms hop size

WINDOW_SAMPLES: int = int(TARGET_SAMPLE_RATE * (WINDOW_MS / 1000.0))  # 400 samples
HOP_SAMPLES: int = int(TARGET_SAMPLE_RATE * (HOP_MS / 1000.0))        # 160 samples

# Seed & Reproducibility
RANDOM_SEED: int = 42

# Device Configuration
DEFAULT_DEVICE: str = "cpu"     # Laptop light by default
COMPUTE_TYPE: str = "int8"      # Light CPU inference (or float16 for CUDA)

# Detection Thresholds
Z_SCORE_THRESHOLD: float = 2.0  # Flag flaw regions only if |z| >= 2.0
