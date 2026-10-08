# Second Take: Rate Flaw Ladder (`librivox_01`)

This directory contains the synthetic speech rate / tempo flaw ladder generated for **Track C (Contrastive Speech Analytics & Temporal Flaw Grounding)** using the reference speaker clip `librivox_01.wav`.

---

## 🎯 Target Speech Phrase
All rate flaw variants modify the **exact same phrase** near the middle of `librivox_01.wav`, anchored at word boundaries and bounded by natural acoustic pauses:
- **Words**: 174 through 191 (18 words)
- **Text**: *"However this may be it is certain that he soon became domesticated in the family of Colonel Syme"*
- **Original Region**: `53.90s` to `58.84s` (duration: `4.94s`)
- **Bounding Pauses**: `440 ms` before (following word 173 *"Virginia"*), `740 ms` after (preceding word 192 *"The"*)
- **Acoustic Splice**: 10.0 ms linear crossfade at both joins to eliminate click artifacts.

---

## 📊 Rate Flaw Ladder Specification & Mapping Table

| File Name | Tier Name | Direction | Factor | Description | Flaw Region (Corrupted Timeline) | Flaw Duration | Audio Timeline Shift |
| :--- | :---: | :---: | :---: | :--- | :---: | :---: | :---: |
| `librivox_01_rate_tier1.wav` | Tier 1 | Speed-up | **1.05** | Subtle speed-up (+5%) | `[53.900s, 58.605s]` | 4.705 s | -0.235 s (earlier) |
| `librivox_01_rate_tier2.wav` | Tier 2 | Speed-up | **1.10** | Mild speed-up (+10%) | `[53.900s, 58.391s]` | 4.491 s | -0.449 s (earlier) |
| `librivox_01_rate_tier3.wav` | Tier 3 | Speed-up | **1.25** | Moderate speed-up (+25%) | `[53.900s, 57.852s]` | 3.952 s | -0.988 s (earlier) |
| `librivox_01_rate_tier4.wav` | Tier 4 | Speed-up | **1.60** | Strong speed-up (+60%) | `[53.900s, 56.988s]` | 3.088 s | -1.853 s (earlier) |
| `librivox_01_rate_tier5.wav` | Tier 5 | Speed-up | **2.00** | Extreme speed-up (+100%) | `[53.900s, 56.370s]` | 2.470 s | -2.470 s (earlier) |
| `librivox_01_rate_slow_tier1.wav` | Slow Tier 1 | Slow-down | **0.95** | Subtle slow-down (-5%) | `[53.900s, 59.100s]` | 5.200 s | +0.260 s (later) |
| `librivox_01_rate_slow_tier2.wav` | Slow Tier 2 | Slow-down | **0.80** | Moderate slow-down (-20%) | `[53.900s, 60.075s]` | 6.175 s | +1.235 s (later) |
| `librivox_01_rate_slow_tier3.wav` | Slow Tier 3 | Slow-down | **0.60** | Strong slow-down (-40%) | `[53.900s, 62.133s]` | 8.233 s | +3.293 s (later) |
| `librivox_01_control.wav` | Control | Neutral | **1.00** | Phase-vocoder resynthesis | *None* | — | 0.000 s |

### Legacy File Mapping
To maintain strict backward compatibility without modifying or overwriting existing dataset files:
- **`librivox_01_rate_t3.wav`** & `.json`: Factor **1.25** (identical audio to `librivox_01_rate_tier3.wav`).
- **`librivox_01_rate_t5.wav`** & `.json`: Factor **1.60** (identical audio to `librivox_01_rate_tier4.wav`).
- **`librivox_01_rate_slow1..3.wav`**: Aliased copies of `librivox_01_rate_slow_tier1..3.wav` to support alternate naming conventions.

---

## 📈 Baseline Acoustic Statistics (`librivox_01.wav`)
- **Analysis Window**: 3.0 seconds sliding window
- **Hop Size**: 0.1 seconds (100 ms)
- **Baseline Mean Speech Rate**: `3.180` words/second
- **Baseline Standard Deviation**: `0.600` words/second
- **Flaw Criterion**: $|z| \ge 2.0$ with contrastive delivery deviation relative to baseline take ($\ge 10\%$ speedup/slowdown).

---

## 🔍 Detection & Grounding Results (`rate_ladder_results.json`)

| File | Factor | Detected | Peak $z$ | Start Error | End Error | IoU | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `librivox_01_rate_tier1.wav` | **1.05** | **no** | $+1.92$ | N/A | N/A | 0.0000 | Sub-threshold ($z < 2.0$) |
| `librivox_01_rate_tier2.wav` | **1.10** | **yes** | $+2.48$ | 0.8460 s | 0.6239 s | 0.6727 | DETECTED |
| `librivox_01_rate_tier3.wav` | **1.25** | **yes** | $+3.59$ | 0.0370 s | 0.0350 s | 0.9819 | DETECTED (IoU 98.2%) |
| `librivox_01_rate_tier4.wav` | **1.60** | **yes** | $+4.70$ | 0.0040 s | 0.0375 s | 0.9867 | DETECTED (IoU 98.7%) |
| `librivox_01_rate_tier5.wav` | **2.00** | **yes** | $+5.25$ | 0.0140 s | 0.0360 s | 0.9802 | DETECTED (IoU 98.0%) |
| `librivox_01_rate_slow_tier1.wav` | **0.95** | **no** | $-1.41$ | N/A | N/A | 0.0000 | Sub-threshold ($|z| < 2.0$) |
| `librivox_01_rate_slow_tier2.wav` | **0.80** | **no** | $-1.97$ | N/A | N/A | 0.0000 | Borderline ($z = -1.97$) |
| `librivox_01_rate_slow_tier3.wav` | **0.60** | **yes** | $-2.52$ | 3.3080 s | 0.0253 s | 0.5951 | DETECTED |
| `librivox_01_control.wav` | **1.00** | **no** | $+1.92$ | N/A | N/A | 0.0000 | **PASS (0 false positives)** |
