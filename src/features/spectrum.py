"""Spectrogram and voice clarity feature extraction module for Laya.

Computes:
1. Log-magnitude spectrogram from STFT (25 ms window, 10 ms hop reduced to
   about 40 ms hop and 48 frequency bands up to 8 kHz, in whole decibels).
2. Voice clarity track: Harmonics-to-Noise Ratio (HNR in dB) with Parselmouth,
   sampled per 10 ms where voiced and null elsewhere, with median clarity.

No long dashes in any text or comments.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import librosa
import numpy as np
import parselmouth

from src.audio.preprocess import load_audio
from src.config import HOP_SAMPLES, TARGET_SAMPLE_RATE, WINDOW_SAMPLES


def compute_spectrogram(
    audio_input: Union[str, Path, np.ndarray],
    target_sr: int = TARGET_SAMPLE_RATE,
    n_fft: int = 512,
    win_samples: int = WINDOW_SAMPLES,
    hop_samples: int = HOP_SAMPLES,
    num_bands: int = 48,
    time_downsample: int = 4,
    top_db: float = 80.0,
) -> Dict[str, Any]:
    """Compute reduced log-magnitude spectrogram in whole decibels.

    Args:
        audio_input: Audio filepath or 1D numpy array.
        target_sr: Target sample rate in Hz (default: 16000).
        n_fft: FFT window size (default: 512).
        win_samples: STFT analysis window in samples (default: 400 for 25 ms).
        hop_samples: STFT hop size in samples (default: 160 for 10 ms).
        num_bands: Number of output frequency bands up to Nyquist (default: 48).
        time_downsample: Frame subsampling factor (default: 4 for ~40 ms hop).
        top_db: Dynamic range cutoff in decibels (default: 80.0).

    Returns:
        Dictionary containing:
            - times: list of frame center timestamps in seconds (~40 ms step)
            - frequencies: list of center frequencies for the 48 bands (Hz)
            - values: 2D list of shape [num_bands, num_frames] in whole decibels
    """
    if isinstance(audio_input, (str, Path)):
        audio, sr = load_audio(audio_input, target_sr=target_sr)
    else:
        audio = np.asarray(audio_input, dtype=np.float32)
        sr = target_sr

    if len(audio) == 0:
        return {"times": [], "frequencies": [], "values": []}

    # 1. Compute Short-Time Fourier Transform (STFT)
    stft = librosa.stft(
        y=audio,
        n_fft=n_fft,
        win_length=win_samples,
        hop_length=hop_samples,
    )
    mag = np.abs(stft)

    # 2. Subsample along the time axis to reduce hop to about 40 ms
    mag_reduced = mag[:, ::time_downsample]
    num_frames = mag_reduced.shape[1]
    hop_sec = (hop_samples * time_downsample) / float(sr)
    times = [round(float(i * hop_sec), 3) for i in range(num_frames)]

    # 3. Bin FFT frequencies into about 48 bands up to 8 kHz (Nyquist)
    nyquist = sr / 2.0
    band_edges = np.linspace(0.0, nyquist, num_bands + 1)
    fft_freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    center_freqs: List[int] = []
    band_values: List[np.ndarray] = []

    for k in range(num_bands):
        f_low = band_edges[k]
        f_high = band_edges[k + 1]
        center_freqs.append(int(round((f_low + f_high) / 2.0)))

        # Find matching FFT bins
        idx = np.where((fft_freqs >= f_low) & (fft_freqs < f_high))[0]
        if len(idx) == 0:
            closest = int(np.argmin(np.abs(fft_freqs - (f_low + f_high) / 2.0)))
            b_mag = mag_reduced[closest, :]
        else:
            b_mag = np.mean(mag_reduced[idx, :], axis=0)

        band_values.append(b_mag)

    band_array = np.array(band_values, dtype=np.float32)

    # 4. Convert magnitude to whole decibels relative to peak
    max_val = float(np.max(band_array))
    if max_val <= 0.0 or np.isnan(max_val):
        db_array = np.full(band_array.shape, -int(top_db), dtype=np.int32)
    else:
        db_raw = librosa.amplitude_to_db(band_array, ref=np.max, top_db=top_db)
        db_array = np.round(db_raw).astype(np.int32)

    values = db_array.tolist()

    return {
        "times": times,
        "frequencies": center_freqs,
        "values": values,
    }


def compute_voice_clarity(
    audio_input: Union[str, Path, np.ndarray],
    target_sr: int = TARGET_SAMPLE_RATE,
    reference_wav: Optional[Union[str, Path]] = None,
    hop_sec: float = 0.010,
    pitch_floor: float = 75.0,
    pitch_ceiling: float = 500.0,
) -> Dict[str, Any]:
    """Compute frame-level voice clarity (HNR in dB) and median clarity.

    Args:
        audio_input: Audio filepath or 1D numpy array.
        target_sr: Target sample rate in Hz (default: 16000).
        reference_wav: Optional reference WAV filepath for paired mode.
        hop_sec: Uniform frame step in seconds (default: 0.010 for 10 ms).
        pitch_floor: Minimum pitch in Hz for voicing (default: 75.0).
        pitch_ceiling: Maximum pitch in Hz for voicing (default: 500.0).

    Returns:
        Dictionary containing:
            - times: list of timestamps at 10 ms hop
            - hnr_db: list of float HNR values where voiced and null elsewhere
            - median_db: float median voiced clarity for participant
            - reference_median_db: float median clarity for reference, or null
    """
    if isinstance(audio_input, (str, Path)):
        snd = parselmouth.Sound(str(audio_input))
    else:
        audio = np.asarray(audio_input, dtype=np.float64)
        snd = parselmouth.Sound(audio, sampling_frequency=target_sr)

    duration = float(snd.duration)
    num_frames = int(round(duration / hop_sec))
    times = [round(float(i * hop_sec), 3) for i in range(num_frames)]

    # Compute pitch (voicing detection) and harmonicity (HNR)
    pitch = snd.to_pitch(time_step=hop_sec, pitch_floor=pitch_floor, pitch_ceiling=pitch_ceiling)
    harm = snd.to_harmonicity(time_step=hop_sec, minimum_pitch=pitch_floor)

    # Vectorized extraction: get all pitch and HNR values at once via numpy
    times_arr = np.array(times, dtype=np.float64)
    pitch_values = np.array(
        [pitch.get_value_at_time(t) for t in times_arr], dtype=np.float64
    )
    harm_values = np.array(
        [harm.get_value(t) for t in times_arr], dtype=np.float64
    )

    # Build masks for voiced frames with valid HNR
    voiced_mask = ~np.isnan(pitch_values) & (pitch_values > 0.0)
    valid_hnr_mask = voiced_mask & ~np.isnan(harm_values) & (harm_values > -100.0)

    hnr_db: List[Optional[float]] = [None] * num_frames
    for idx in np.where(valid_hnr_mask)[0]:
        hnr_db[idx] = round(float(harm_values[idx]), 1)

    voiced_hnrs = harm_values[valid_hnr_mask]

    median_db: Optional[float] = (
        round(float(np.median(voiced_hnrs)), 1) if len(voiced_hnrs) > 0 else None
    )

    # Reference audio median clarity (if provided in paired mode)
    reference_median_db: Optional[float] = None
    if reference_wav is not None:
        ref_path = Path(reference_wav)
        if ref_path.exists() and ref_path.is_file():
            ref_snd = parselmouth.Sound(str(ref_path))
            ref_pitch = ref_snd.to_pitch(
                time_step=hop_sec, pitch_floor=pitch_floor, pitch_ceiling=pitch_ceiling
            )
            ref_harm = ref_snd.to_harmonicity(
                time_step=hop_sec, minimum_pitch=pitch_floor
            )
            ref_frames = int(round(ref_snd.duration / hop_sec))
            ref_times = [round(float(i * hop_sec), 3) for i in range(ref_frames)]
            ref_voiced_hnrs: List[float] = []

            for rt in ref_times:
                rp_val = ref_pitch.get_value_at_time(rt)
                if rp_val is not None and not np.isnan(rp_val) and rp_val > 0.0:
                    rh_val = ref_harm.get_value(rt)
                    if rh_val is not None and not np.isnan(rh_val) and rh_val > -100.0:
                        ref_voiced_hnrs.append(round(float(rh_val), 1))

            if ref_voiced_hnrs:
                reference_median_db = round(float(np.median(ref_voiced_hnrs)), 1)

    return {
        "times": times,
        "hnr_db": hnr_db,
        "median_db": median_db,
        "reference_median_db": reference_median_db,
    }
