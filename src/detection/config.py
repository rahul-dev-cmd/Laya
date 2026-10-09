"""Configuration constants and thresholds for pause flaw corruption and detection.

Rules adhered to:
- Minimum natural pause duration: 150 ms (0.150 s)
- Anomaly criterion: |z| >= 2.0 in either direction ('too short' / 'too long')
- 10 ms crossfade at audio cuts
- Room noise acoustic fill for lengthened pauses
- Tiers:
  - rushed: 0.85, 0.65, 0.40, 0.20, 0.00
  - draggy: 1.15, 1.40, 1.80, 2.50, 3.00
- Modes:
  - Mode A: Contrastive word-boundary z-score relative to baseline take
  - Mode B: Speaker population z-score relative to speaker median & MAD
"""

from typing import Dict, Tuple

# Pause duration threshold for identifying phrase boundaries
PAUSE_MIN_DURATION_SECONDS: float = 0.150  # 150 ms minimum

# Anomaly threshold: flag when |z| >= 2.0
Z_SCORE_THRESHOLD: float = 2.0

# Scale factor for normal-consistent Median Absolute Deviation (MAD)
MAD_SCALE_FACTOR: float = 1.4826

# Crossfade duration at audio splice joins
CROSSFADE_MS: float = 10.0

# Time-stretch / duration scaling multipliers by tier
RUSHED_TIERS: Dict[int, float] = {
    1: 0.85,
    2: 0.65,
    3: 0.40,
    4: 0.20,
    5: 0.00,
}

DRAGGY_TIERS: Dict[int, float] = {
    1: 1.15,
    2: 1.40,
    3: 1.80,
    4: 2.50,
    5: 3.00,
}

CONTROL_MULTIPLIER: float = 1.00

# Primary detection mode ('A' for contrastive word-boundary, 'B' for speaker median & MAD)
PRIMARY_MODE: str = "A"

# Target regions: 3-4 s region containing exactly one prominent phrase boundary
# Clip 01: words 84 ("that") to 91 ("brothers"), target pause between word 86 ("county") and 87 ("There")
CLIP_01_CHOSEN_WORDS: Tuple[int, int] = (84, 91)
CLIP_01_TARGET_PAUSE: Tuple[int, int] = (86, 87)

# Clip 02: words 156 ("in") to 164 ("this"), target pause between word 158 ("hand") and 159 ("I")
CLIP_02_CHOSEN_WORDS: Tuple[int, int] = (156, 164)
CLIP_02_TARGET_PAUSE: Tuple[int, int] = (158, 159)

# =====================================================================
# Pitch Detection Frozen Constants & Thresholds
# =====================================================================
PITCH_WINDOW_DURATION: float = 3.0          # 3.0 s sliding window
PITCH_HOP_DURATION: float = 0.1             # 100 ms step between sliding windows
PITCH_Z_SCORE_THRESHOLD: float = -2.0       # Window anomaly criterion: z <= -2.0 (flat intonation)
PITCH_WORD_Z_THRESHOLD: float = -1.5        # Word pitch spread compression criterion: z <= -1.5
PITCH_MIN_VOICED_FRAMES: int = 30           # At least 300 ms voiced speech in a 3s window
PITCH_CONTRASTIVE_RATIO: float = 0.85       # At least 15% reduction in SD relative to baseline
PITCH_CONTRASTIVE_DROP: float = 0.4         # At least 0.4 semitones drop relative to baseline
PITCH_MERGE_GAP_SECONDS: float = 0.35       # Max gap (s) between overlapping/adjacent flagged windows
PITCH_STAGE2_MARGIN_SECONDS: float = 0.5    # Search margin around flagged windows for word edge refinement


def print_detection_config() -> None:
    """Print active detection thresholds at the start of each run."""
    print("=" * 70)
    print("  PAUSE DETECTION CONFIGURATION THRESHOLDS")
    print("=" * 70)
    print(f"  Minimum Pause Duration:     {PAUSE_MIN_DURATION_SECONDS:.3f} s ({PAUSE_MIN_DURATION_SECONDS * 1000:.0f} ms)")
    print(f"  Z-Score Anomaly Criterion:  |z| >= {Z_SCORE_THRESHOLD:.1f}")
    print(f"  MAD Normal Scale Factor:    {MAD_SCALE_FACTOR:.4f}")
    print(f"  Acoustic Splice Crossfade:  {CROSSFADE_MS:.1f} ms")
    print(f"  Primary Evaluation Mode:    Mode {PRIMARY_MODE} (contrastive word-boundary)")
    print(f"  Rushed Tier Multipliers:    {RUSHED_TIERS}")
    print(f"  Draggy Tier Multipliers:    {DRAGGY_TIERS}")
    print("=" * 70)


def print_pitch_detection_config() -> None:
    """Print active pitch detection thresholds at the start of each run."""
    print("=" * 70)
    print("  PITCH DETECTION CONFIGURATION THRESHOLDS")
    print("=" * 70)
    print(f"  Sliding Window Duration:    {PITCH_WINDOW_DURATION:.1f} s")
    print(f"  Window Hop Duration:        {PITCH_HOP_DURATION:.2f} s (100 ms)")
    print(f"  Window Z-Score Threshold:   z <= {PITCH_Z_SCORE_THRESHOLD:.1f}")
    print(f"  Word Z-Score Threshold:     z <= {PITCH_WORD_Z_THRESHOLD:.1f}")
    print(f"  Minimum Voiced Frames:      {PITCH_MIN_VOICED_FRAMES} frames")
    print(f"  Contrastive Drop Criterion: SD <= {PITCH_CONTRASTIVE_RATIO*100:.0f}% baseline & drop >= {PITCH_CONTRASTIVE_DROP:.2f} st")
    print(f"  Window Merge Gap:           {PITCH_MERGE_GAP_SECONDS:.2f} s")
    print(f"  Stage 2 Search Margin:      {PITCH_STAGE2_MARGIN_SECONDS:.2f} s")
    print("=" * 70)

