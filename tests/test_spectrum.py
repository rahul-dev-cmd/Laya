"""Tests for spectrogram and voice clarity feature extraction.

Tests cover:
- output shapes for spectrogram and voice clarity
- determinism of spectrogram and voice clarity
- pure tone peaks in the right frequency band
- silence gives null clarity
- noisy signal gives lower clarity than clean tone

No long dashes in any text or comments.
"""

import numpy as np
import pytest

from src.features.spectrum import compute_spectrogram, compute_voice_clarity


def test_spectrogram_output_shapes():
    """Verify spectrogram output structure and dimensions."""
    sr = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    signal = 0.5 * np.sin(2 * np.pi * 440.0 * t)

    res = compute_spectrogram(signal, target_sr=sr)

    assert "times" in res
    assert "frequencies" in res
    assert "values" in res

    times = res["times"]
    frequencies = res["frequencies"]
    values = res["values"]

    assert len(frequencies) == 48
    assert len(values) == 48

    num_frames = len(times)
    # 2.0 seconds with hop ~40 ms (~640 samples at 16 kHz) gives ~50 frames
    assert num_frames > 0
    for row in values:
        assert len(row) == num_frames
        assert all(isinstance(v, int) for v in row)

    # Center frequencies should be monotonically increasing up to Nyquist (8000 Hz)
    assert frequencies[0] < frequencies[-1]
    assert frequencies[-1] <= 8000


def test_voice_clarity_output_shapes():
    """Verify voice clarity output structure and dimensions."""
    sr = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    signal = 0.5 * np.sin(2 * np.pi * 200.0 * t)

    res = compute_voice_clarity(signal, target_sr=sr)

    assert "times" in res
    assert "hnr_db" in res
    assert "median_db" in res
    assert "reference_median_db" in res

    times = res["times"]
    hnr_db = res["hnr_db"]
    # 2.0 seconds at 10 ms hop has 200 frames
    assert len(times) == 200
    assert len(hnr_db) == 200
    assert res["reference_median_db"] is None


def test_spectrogram_and_clarity_determinism():
    """Verify that feature computations produce identical outputs on repeated runs."""
    sr = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    signal = 0.4 * np.sin(2 * np.pi * 300.0 * t)

    spec1 = compute_spectrogram(signal, target_sr=sr)
    spec2 = compute_spectrogram(signal, target_sr=sr)
    assert spec1 == spec2

    clarity1 = compute_voice_clarity(signal, target_sr=sr)
    clarity2 = compute_voice_clarity(signal, target_sr=sr)
    assert clarity1 == clarity2


def test_pure_tone_peaks_in_right_frequency_band():
    """Verify that a pure sinusoidal tone peaks in the corresponding frequency band."""
    sr = 16000
    duration = 1.0
    target_freq = 1000.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    tone = 0.6 * np.sin(2 * np.pi * target_freq * t)

    res = compute_spectrogram(tone, target_sr=sr)
    times = res["times"]
    frequencies = res["frequencies"]
    values = res["values"]

    # Check a frame near the center
    mid_idx = len(times) // 2
    band_powers = [values[b][mid_idx] for b in range(len(frequencies))]
    peak_band = int(np.argmax(band_powers))
    peak_freq = frequencies[peak_band]

    # Peak band center frequency should be close to 1000 Hz (within 150 Hz band tolerance)
    assert abs(peak_freq - target_freq) <= 150.0


def test_silence_gives_null_clarity():
    """Verify that zero-amplitude silence produces null voice clarity frames and median."""
    sr = 16000
    duration = 1.0
    silence = np.zeros(int(sr * duration))

    res = compute_voice_clarity(silence, target_sr=sr)
    assert res["median_db"] is None
    assert all(v is None for v in res["hnr_db"])


def test_noisy_signal_gives_lower_clarity_than_clean_tone():
    """Verify that adding noise substantially degrades harmonicity and clarity."""
    sr = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    f0 = 160.0

    # Voiced harmonic series
    clean = np.zeros_like(t)
    for harmonic in range(1, 6):
        clean += (1.0 / harmonic) * np.sin(2 * np.pi * harmonic * f0 * t)
    clean /= np.max(np.abs(clean))

    clarity_clean = compute_voice_clarity(clean, target_sr=sr)
    assert clarity_clean["median_db"] is not None

    # Add Gaussian noise
    rng = np.random.RandomState(42)
    noisy = clean + 0.4 * rng.randn(len(clean))

    clarity_noisy = compute_voice_clarity(noisy, target_sr=sr)

    if clarity_noisy["median_db"] is not None:
        assert clarity_noisy["median_db"] < clarity_clean["median_db"]
    else:
        # If noise destroyed all voicing detection, that is also strictly lower clarity
        assert clarity_clean["median_db"] is not None
