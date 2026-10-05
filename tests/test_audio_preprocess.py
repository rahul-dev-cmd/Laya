"""Tests for audio preprocessing and 16 kHz mono compliance."""

import tempfile
from pathlib import Path
import numpy as np
import pytest

from src.config import (
    TARGET_SAMPLE_RATE,
    WINDOW_SAMPLES,
    HOP_SAMPLES,
)
from src.audio.preprocess import (
    load_audio,
    save_audio,
    compute_frame_rms,
    get_audio_metadata,
)

def test_audio_save_and_load():
    """Test saving a synthetic sine wave and reloading it as 16 kHz mono."""
    sr = TARGET_SAMPLE_RATE
    duration_sec = 1.0
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    # Generate 440 Hz test tone
    sine_wave = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

    with tempfile.TemporaryDirectory() as tmpdir:
        test_wav = Path(tmpdir) / "test_sine.wav"
        save_audio(test_wav, sine_wave, sr=sr)
        assert test_wav.exists()

        loaded_audio, loaded_sr = load_audio(test_wav, target_sr=TARGET_SAMPLE_RATE)
        assert loaded_sr == TARGET_SAMPLE_RATE
        assert loaded_audio.ndim == 1  # Mono
        assert len(loaded_audio) == int(sr * duration_sec)
        assert np.max(np.abs(loaded_audio)) <= 1.0

def test_frame_rms_dimensions():
    """Verify RMS frame counts match 25ms window and 10ms hop."""
    sr = TARGET_SAMPLE_RATE
    duration_sec = 2.0
    num_samples = int(sr * duration_sec)
    audio = np.random.randn(num_samples).astype(np.float32) * 0.1

    rms = compute_frame_rms(audio, frame_length=WINDOW_SAMPLES, hop_length=HOP_SAMPLES)
    # At 2.0s with 10ms hop (160 samples), expected frames ~ 200
    expected_frames = int(np.ceil(num_samples / HOP_SAMPLES))
    assert abs(len(rms) - expected_frames) <= 2
    assert np.all(rms >= 0.0)

def test_audio_metadata():
    """Test get_audio_metadata structure and calculations."""
    sr = 16000
    audio = np.ones(16000, dtype=np.float32) * 0.5
    meta = get_audio_metadata(audio, sr=sr)

    assert meta["sample_rate"] == 16000
    assert meta["num_samples"] == 16000
    assert meta["duration_seconds"] == 1.0
    assert pytest.approx(meta["peak_amplitude"], 0.01) == 0.5
    assert pytest.approx(meta["mean_rms_energy"], 0.01) == 0.5
