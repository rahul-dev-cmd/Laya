#!/usr/bin/env python3
"""Acoustic feature extraction module for Second Take.

Extracts time-aligned speech features for a speech recording and its word alignment:
1. F0 contour (parselmouth Praat pitch), converted to semitones relative to speaker median F0.
2. Energy (RMS in dB), z-scored over the clip.
3. Speech rate: words per second in a sliding 3-second window, from forced word alignment.
4. Pauses: gaps between consecutive words longer than 150 ms, as (start, end) intervals.
5. MFCCs (13 coefficients) and log-magnitude spectrogram (FFT).

Standard:
- 16,000 Hz sample rate (mono)
- 25 ms analysis window (400 samples)
- 10 ms hop size (160 samples)
- Reproducible random seed (42)
"""

import sys
import os
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Prefer project's virtual environment if available and running under an external python
venv_py = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
if venv_py.exists() and sys.executable.lower() != str(venv_py).lower():
    try:
        import parselmouth
    except ImportError:
        import subprocess
        result = subprocess.call([str(venv_py), __file__] + sys.argv[1:])
        sys.exit(result)

import csv
import argparse
import subprocess
from typing import Dict, List, Tuple, Any, Union

import numpy as np
import librosa
import parselmouth
from scipy.signal import medfilt

from src.config import (
    TARGET_SAMPLE_RATE,
    WINDOW_SAMPLES,
    HOP_SAMPLES,
    WINDOW_MS,
    HOP_MS,
    RANDOM_SEED,
)
from src.utils.seed import set_seed
from src.audio.preprocess import load_audio, compute_frame_rms

def compute_f0_contour(
    wav_path: Union[str, Path],
    times: np.ndarray,
    pitch_floor: float = 80.0,
    pitch_ceiling: float = 350.0,
    median_filter_size: int = 5,
    max_semitones_from_median: float = 12.0,
    min_voiced_duration_ms: float = 50.0,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute and clean F0 contour using Parselmouth (Praat), converted to semitones.
    
    Cleaning pipeline:
    1. Restrict pitch search to plausible speaker range [pitch_floor, pitch_ceiling].
    2. Apply a 5-frame median filter across the contour.
    3. Treat values beyond +-12 semitones from the median as unvoiced (NaN).
    4. Ignore/remove voiced frames shorter than 50 ms (<5 consecutive frames at 10ms hop).
    
    Formula: semitones = 12 * log2(f0 / median_f0) for voiced frames.
    
    Args:
        wav_path: Path to the audio file.
        times: Array of frame center timestamps in seconds.
        pitch_floor: Minimum pitch frequency in Hz (default: 80.0 Hz).
        pitch_ceiling: Maximum pitch frequency in Hz (default: 350.0 Hz).
        median_filter_size: Median filter size in frames (default: 5 frames).
        max_semitones_from_median: Semitone cutoff relative to median (default: 12.0).
        min_voiced_duration_ms: Minimum continuous voiced duration in ms (default: 50.0 ms).
        
    Returns:
        Tuple of (f0_hz, f0_semitones, median_f0).
        Unvoiced frames contain np.nan.
    """
    sound = parselmouth.Sound(str(wav_path))
    # 1. Restrict pitch search range to plausible range for this speaker
    pitch = sound.to_pitch_ac(
        time_step=HOP_MS / 1000.0,
        pitch_floor=pitch_floor,
        pitch_ceiling=pitch_ceiling,
    )

    # Evaluate F0 at uniform frame center timestamps
    f0_hz = np.array([pitch.get_value_at_time(float(t)) for t in times], dtype=np.float64)
    # Ensure unvoiced is represented as NaN
    f0_hz[f0_hz <= 0.0] = np.nan

    # 2. Apply a 5-frame median filter
    f0_clean = np.where(~np.isnan(f0_hz) & (f0_hz > 0.0), f0_hz, 0.0)
    f0_filt = medfilt(f0_clean, kernel_size=median_filter_size)
    f0_hz = np.where(f0_filt > 0.0, f0_filt, np.nan)

    # 3. Treat values beyond +-12 semitones from the median as unvoiced (NaN)
    voiced_mask_pre = ~np.isnan(f0_hz) & (f0_hz > 0.0)
    if np.any(voiced_mask_pre):
        median_pre = float(np.median(f0_hz[voiced_mask_pre]))
        semitones_pre = 12.0 * np.log2(f0_hz / median_pre)
        outliers = np.abs(semitones_pre) > max_semitones_from_median
        f0_hz[outliers] = np.nan

    # 4. Ignore voiced frames shorter than 50 ms (5 frames at 10 ms hop)
    min_frames = int(round(min_voiced_duration_ms / HOP_MS))
    is_voiced = ~np.isnan(f0_hz) & (f0_hz > 0.0)
    diff = np.diff(np.pad(is_voiced.astype(int), (1, 1), "constant"))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    for s, e in zip(starts, ends):
        if (e - s) < min_frames:
            f0_hz[s:e] = np.nan

    # Compute final median F0 and semitones relative to median
    voiced_mask_final = ~np.isnan(f0_hz) & (f0_hz > 0.0)
    if np.any(voiced_mask_final):
        median_f0 = float(np.median(f0_hz[voiced_mask_final]))
        f0_semitones = np.full_like(f0_hz, np.nan)
        f0_semitones[voiced_mask_final] = 12.0 * np.log2(f0_hz[voiced_mask_final] / median_f0)
        # Guarantee strict bound
        outliers_final = np.abs(f0_semitones) > max_semitones_from_median
        f0_semitones[outliers_final] = np.nan
        f0_hz[outliers_final] = np.nan
    else:
        median_f0 = 0.0
        f0_semitones = np.full_like(f0_hz, np.nan)

    return f0_hz, f0_semitones, median_f0

def compute_energy_contour(
    audio: np.ndarray,
    frame_length: int = WINDOW_SAMPLES,
    hop_length: int = HOP_SAMPLES,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute frame-level RMS energy in dB and its z-score over the clip.
    
    Args:
        audio: 1D normalized audio array.
        frame_length: Window size in samples (400 samples = 25 ms).
        hop_length: Hop size in samples (160 samples = 10 ms).
        
    Returns:
        Tuple of (energy_db, energy_z).
    """
    rms = compute_frame_rms(audio, frame_length=frame_length, hop_length=hop_length)
    # Energy in decibels (clipped at -160 dB floor for silence)
    energy_db = 20.0 * np.log10(np.maximum(rms, 1e-8))
    # Z-score normalization across the clip
    mean_db = float(np.mean(energy_db))
    std_db = float(np.std(energy_db))
    energy_z = (energy_db - mean_db) / (std_db + 1e-8)

    return energy_db, energy_z

def compute_speech_rate_and_pauses(
    alignment_csv_path: Union[str, Path],
    times: np.ndarray,
    total_duration: float,
    window_duration: float = 3.0,
    pause_threshold_ms: float = 150.0,
) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
    """Calculate sliding speech rate and detect inter-word pauses.
    
    Args:
        alignment_csv_path: CSV with columns word, start_seconds, end_seconds.
        times: Array of frame timestamps.
        total_duration: Total audio duration in seconds.
        window_duration: Sliding window size in seconds (default: 3.0 s).
        pause_threshold_ms: Minimum gap between consecutive words for a pause (default: 150 ms).
        
    Returns:
        Tuple of (speech_rate: np.ndarray, pauses: List[Tuple[start, end]]).
    """
    rows: List[Dict[str, Any]] = []
    with open(alignment_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                start = float(r["start_seconds"])
                end = float(r["end_seconds"])
                word = r["word"].strip()
                if end > start:
                    rows.append({"word": word, "start": start, "end": end, "mid": (start + end) / 2.0})
            except (ValueError, KeyError):
                continue

    # 1. Detect pauses: gaps between consecutive words > 150 ms
    pauses: List[Tuple[float, float]] = []
    threshold_sec = pause_threshold_ms / 1000.0
    for i in range(len(rows) - 1):
        gap_start = rows[i]["end"]
        gap_end = rows[i + 1]["start"]
        if gap_end - gap_start > threshold_sec:
            pauses.append((round(gap_start, 3), round(gap_end, 3)))

    # 2. Compute speech rate: words per second in sliding 3-second window
    midpoints = np.array([r["mid"] for r in rows], dtype=np.float64)
    half_win = window_duration / 2.0
    speech_rate = np.zeros(len(times), dtype=np.float64)

    for i, t in enumerate(times):
        w_start = max(0.0, t - half_win)
        w_end = min(total_duration, t + half_win)
        w_dur = max(0.1, w_end - w_start)
        # Count words whose midpoint falls inside the window
        count = np.sum((midpoints >= w_start) & (midpoints <= w_end))
        speech_rate[i] = count / w_dur

    return speech_rate, pauses

def compute_spectral_features(
    audio: np.ndarray,
    sr: int = TARGET_SAMPLE_RATE,
    n_mfcc: int = 13,
    n_fft: int = 512,
    window_samples: int = WINDOW_SAMPLES,
    hop_samples: int = HOP_SAMPLES,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute 13 MFCCs and log-magnitude STFT spectrogram.
    
    Args:
        audio: 1D audio waveform.
        sr: Sample rate in Hz.
        n_mfcc: Number of MFCC coefficients (default: 13).
        n_fft: FFT size (default: 512).
        window_samples: Window length (400 samples = 25 ms).
        hop_samples: Hop length (160 samples = 10 ms).
        
    Returns:
        Tuple of (mfccs: (13, T), spectrogram: (257, T)).
    """
    mfccs = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        win_length=window_samples,
        hop_length=hop_samples,
    )
    stft = librosa.stft(
        y=audio,
        n_fft=n_fft,
        win_length=window_samples,
        hop_length=hop_samples,
    )
    # Log-magnitude spectrogram: log(|STFT| + eps)
    spectrogram = np.log(np.abs(stft) + 1e-8)

    return mfccs, spectrogram

def extract_clip_features(
    wav_path: Union[str, Path],
    alignment_csv_path: Union[str, Path],
    output_npz_path: Union[str, Path],
    pitch_floor: float = 80.0,
    pitch_ceiling: float = 350.0,
    seed: int = RANDOM_SEED,
) -> Dict[str, Any]:
    """Extract all acoustic features for a clip and save as compressed .npz archive.
    
    Args:
        wav_path: Path to input WAV file.
        alignment_csv_path: Path to word alignment CSV file.
        output_npz_path: Destination path for .npz feature file.
        pitch_floor: Pitch search floor frequency in Hz (default: 80.0 Hz).
        pitch_ceiling: Pitch search ceiling frequency in Hz (default: 350.0 Hz).
        seed: Random seed for reproducibility.
        
    Returns:
        Dictionary of extracted feature arrays.
    """
    set_seed(seed)
    wav_path = Path(wav_path)
    alignment_csv_path = Path(alignment_csv_path)
    output_npz_path = Path(output_npz_path)

    # 1. Load audio (16 kHz mono normalized)
    print(f"[Features] Loading audio from '{wav_path}'...")
    audio, sr = load_audio(wav_path, target_sr=TARGET_SAMPLE_RATE)
    duration = float(len(audio)) / float(sr)

    # 2. Frame-level energy (RMS in dB, z-scored)
    print("[Features] Computing RMS energy (25ms window, 10ms hop)...")
    energy_db, energy_z = compute_energy_contour(
        audio, frame_length=WINDOW_SAMPLES, hop_length=HOP_SAMPLES
    )
    num_frames = len(energy_db)
    times = np.arange(num_frames) * (HOP_SAMPLES / sr)

    # 3. F0 contour via Parselmouth & semitones relative to median
    print(f"[Features] Computing F0 contour (floor={pitch_floor:.1f}Hz, ceiling={pitch_ceiling:.1f}Hz) & cleaning...")
    f0_hz, f0_semitones, median_f0 = compute_f0_contour(
        wav_path,
        times,
        pitch_floor=pitch_floor,
        pitch_ceiling=pitch_ceiling,
    )
    print(f"[Features] Speaker median F0: {median_f0:.2f} Hz | Voiced frames: {np.sum(~np.isnan(f0_semitones))}/{num_frames}")

    # 4. Speech rate & pause detection from alignment
    print("[Features] Computing sliding speech rate (3s window) and pauses (>150ms)...")
    speech_rate, pauses = compute_speech_rate_and_pauses(
        alignment_csv_path, times, total_duration=duration, window_duration=3.0, pause_threshold_ms=150.0
    )
    print(f"[Features] Detected {len(pauses)} pauses (>150ms) | Mean speech rate: {np.mean(speech_rate):.2f} wps")

    # 5. MFCCs (13) and log-magnitude spectrogram
    print("[Features] Computing 13 MFCCs and log-magnitude spectrogram (FFT)...")
    mfccs, spectrogram = compute_spectral_features(
        audio, sr=sr, n_mfcc=13, n_fft=512, window_samples=WINDOW_SAMPLES, hop_samples=HOP_SAMPLES
    )

    # 6. Ensure exact frame synchronization across all arrays
    T = min(num_frames, mfccs.shape[1], spectrogram.shape[1])
    times = times[:T]
    f0_hz = f0_hz[:T]
    f0_semitones = f0_semitones[:T]
    energy_db = energy_db[:T]
    energy_z = energy_z[:T]
    speech_rate = speech_rate[:T]
    mfccs = mfccs[:, :T]
    spectrogram = spectrogram[:, :T]

    pauses_arr = np.array(pauses, dtype=np.float32) if len(pauses) > 0 else np.empty((0, 2), dtype=np.float32)

    # 7. Save to compressed NPZ
    output_npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz_path,
        times=times,
        f0_hz=f0_hz,
        f0_semitones=f0_semitones,
        f0_median=median_f0,
        energy_db=energy_db,
        energy_z=energy_z,
        speech_rate=speech_rate,
        pauses=pauses_arr,
        mfccs=mfccs,
        spectrogram=spectrogram,
        sample_rate=sr,
        hop_samples=HOP_SAMPLES,
        window_samples=WINDOW_SAMPLES,
    )
    print(f"[Features] Successfully saved features to '{output_npz_path}' ({T} frames, {output_npz_path.stat().st_size / 1024:.1f} KB)")

    return {
        "times": times,
        "f0_hz": f0_hz,
        "f0_semitones": f0_semitones,
        "f0_median": median_f0,
        "energy_db": energy_db,
        "energy_z": energy_z,
        "speech_rate": speech_rate,
        "pauses": pauses_arr,
        "mfccs": mfccs,
        "spectrogram": spectrogram,
    }

def plot_feature_contours(
    features_source: Union[str, Path, Dict[str, Any]],
    output_png_path: Union[str, Path],
    clip_title: str = "librivox_01.wav",
) -> Path:
    """Plot F0, energy, and speech rate with shaded pauses as stacked panels.
    
    Args:
        features_source: Path to .npz file or dictionary of features.
        output_png_path: Destination PNG image path.
        clip_title: Title string for the figure.
        
    Returns:
        Path to the saved PNG image.
    """
    output_png_path = Path(output_png_path)
    output_png_path.parent.mkdir(parents=True, exist_ok=True)

    # Load data if path was passed
    if isinstance(features_source, (str, Path)):
        npz_file = str(features_source)
        data = np.load(npz_file)
    else:
        # Save temporary npz if passed as dictionary to allow external rendering
        npz_file = str(output_png_path.with_suffix(".temp.npz"))
        np.savez(npz_file, **features_source)
        data = features_source

    times = data["times"]
    f0_semitones = data["f0_semitones"]
    energy_z = data["energy_z"]
    speech_rate = data["speech_rate"]
    pauses = data["pauses"]
    f0_median = float(data["f0_median"])

    # Try plotting in the current process
    try:
        import matplotlib
        import matplotlib.pyplot as plt

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

        c_f0 = "#1f77b4"      # Blue
        c_energy = "#ff7f0e"  # Orange
        c_rate = "#2ca02c"    # Green
        c_pause = "#7f7f7f"   # Gray

        # Panel 1: F0 contour
        ax1.plot(times, f0_semitones, color=c_f0, lw=1.2, label=f"F0 (Median: {f0_median:.1f} Hz)")
        ax1.axhline(0, color="gray", linestyle=":", alpha=0.7)
        ax1.set_ylabel("F0 (semitones)", fontsize=11, fontweight="bold")
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.set_title(f"Speech Feature Contours: {clip_title}", fontsize=13, fontweight="bold", pad=10)

        # Panel 2: Energy contour
        ax2.plot(times, energy_z, color=c_energy, lw=1.2, label="Energy (RMS dB z-score)")
        ax2.axhline(0, color="gray", linestyle=":", alpha=0.7)
        ax2.set_ylabel("Energy (z-score)", fontsize=11, fontweight="bold")
        ax2.grid(True, linestyle="--", alpha=0.5)

        # Panel 3: Speech Rate contour
        ax3.plot(times, speech_rate, color=c_rate, lw=1.2, label="Speech Rate (3s window)")
        mean_rate = float(np.mean(speech_rate))
        ax3.axhline(mean_rate, color="gray", linestyle=":", alpha=0.7, label=f"Mean Rate ({mean_rate:.2f} wps)")
        ax3.set_ylabel("Rate (words/s)", fontsize=11, fontweight="bold")
        ax3.set_xlabel("Time (seconds)", fontsize=11, fontweight="bold")
        ax3.grid(True, linestyle="--", alpha=0.5)

        # Shading pauses across all panels
        for i, (p_start, p_end) in enumerate(pauses):
            lbl = "Pause (>150ms)" if i == 0 else None
            for ax in (ax1, ax2, ax3):
                ax.axvspan(p_start, p_end, color=c_pause, alpha=0.25, label=lbl if ax == ax1 else None)

        # Add legends
        for ax in (ax1, ax2, ax3):
            ax.legend(loc="upper right", framealpha=0.9)

        ax3.set_xlim(0, float(np.max(times)))
        plt.tight_layout()
        plt.savefig(output_png_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"[Features] Saved stacked plot to '{output_png_path}'")

    except Exception:
        # If current Python environment encounters native font/DLL policy restrictions (e.g. ft2font in virtualenv),
        # delegate to system Python which has verified matplotlib rendering capability.
        system_py = None
        for candidate in [r"C:\Python314\python.exe", "py", "python"]:
            try:
                chk = subprocess.run([candidate, "-c", "import matplotlib; print('ok')"], capture_output=True, text=True)
                if chk.returncode == 0 and "ok" in chk.stdout:
                    system_py = candidate
                    break
            except Exception:
                continue

        if not system_py:
            print("[Features] Warning: Could not find Python interpreter with working matplotlib.")
            return output_png_path

        plot_script = f"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

data = np.load(r'{npz_file}')
times = data['times']
f0_semitones = data['f0_semitones']
energy_z = data['energy_z']
speech_rate = data['speech_rate']
pauses = data['pauses']
f0_median = float(data['f0_median'])

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
c_f0 = '#1f77b4'
c_energy = '#ff7f0e'
c_rate = '#2ca02c'
c_pause = '#7f7f7f'

ax1.plot(times, f0_semitones, color=c_f0, lw=1.2, label=f'F0 (Median: {{f0_median:.1f}} Hz)')
ax1.axhline(0, color='gray', linestyle=':', alpha=0.7)
ax1.set_ylabel('F0 (semitones)', fontsize=11, fontweight='bold')
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.set_title(r'Speech Feature Contours: {clip_title}', fontsize=13, fontweight='bold', pad=10)

ax2.plot(times, energy_z, color=c_energy, lw=1.2, label='Energy (RMS dB z-score)')
ax2.axhline(0, color='gray', linestyle=':', alpha=0.7)
ax2.set_ylabel('Energy (z-score)', fontsize=11, fontweight='bold')
ax2.grid(True, linestyle='--', alpha=0.5)

ax3.plot(times, speech_rate, color=c_rate, lw=1.2, label='Speech Rate (3s window)')
mean_rate = float(np.mean(speech_rate))
ax3.axhline(mean_rate, color='gray', linestyle=':', alpha=0.7, label=f'Mean Rate ({{mean_rate:.2f}} wps)')
ax3.set_ylabel('Rate (words/s)', fontsize=11, fontweight='bold')
ax3.set_xlabel('Time (seconds)', fontsize=11, fontweight='bold')
ax3.grid(True, linestyle='--', alpha=0.5)

for i, (p_start, p_end) in enumerate(pauses):
    lbl = 'Pause (>150ms)' if i == 0 else None
    for ax in (ax1, ax2, ax3):
        ax.axvspan(p_start, p_end, color=c_pause, alpha=0.25, label=lbl if ax == ax1 else None)

for ax in (ax1, ax2, ax3):
    ax.legend(loc='upper right', framealpha=0.9)

ax3.set_xlim(0, float(np.max(times)))
plt.tight_layout()
Path(r'{output_png_path}').parent.mkdir(parents=True, exist_ok=True)
plt.savefig(r'{output_png_path}', dpi=200, bbox_inches='tight')
plt.close()
"""
        res = subprocess.run([system_py, "-c", plot_script], capture_output=True, text=True)
        if res.returncode != 0:
            print(f"[Features] Warning: Plot rendering failed: {res.stderr}")
        else:
            print(f"[Features] Saved stacked plot to '{output_png_path}'")

    # Clean up temporary npz if created
    if not isinstance(features_source, (str, Path)):
        try:
            os.remove(npz_file)
        except OSError:
            pass

    return output_png_path

def main():
    parser = argparse.ArgumentParser(
        description="Extract acoustic speech features for a clip and alignment file."
    )
    parser.add_argument(
        "--audio",
        default="dataset/sources/librivox_01.wav",
        help="Path to WAV audio file (default: dataset/sources/librivox_01.wav)",
    )
    parser.add_argument(
        "--alignment",
        default="dataset/sources/librivox_01_alignment.csv",
        help="Path to alignment CSV file (default: dataset/sources/librivox_01_alignment.csv)",
    )
    parser.add_argument(
        "--output-npz",
        default="dataset/features/librivox_01_features.npz",
        help="Destination path for .npz feature file (default: dataset/features/librivox_01_features.npz)",
    )
    parser.add_argument(
        "--output-plot",
        default="scratch/features/librivox_01.png",
        help="Destination path for stacked feature plot PNG (default: scratch/features/librivox_01.png)",
    )
    parser.add_argument(
        "--pitch-floor",
        type=float,
        default=80.0,
        help="Pitch search floor in Hz (default: 80.0 Hz)",
    )
    parser.add_argument(
        "--pitch-ceiling",
        type=float,
        default=350.0,
        help="Pitch search ceiling in Hz (default: 350.0 Hz)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    audio_path = Path(args.audio)
    alignment_path = Path(args.alignment)
    out_npz = Path(args.output_npz)
    out_plot = Path(args.output_plot)

    if not audio_path.exists():
        print(f"Error: Audio file not found at {audio_path}", file=sys.stderr)
        sys.exit(1)

    if not alignment_path.exists():
        print(f"Error: Alignment file not found at {alignment_path}", file=sys.stderr)
        sys.exit(1)

    print("\n" + "=" * 60)
    print(f"FEATURE EXTRACTION: {audio_path.name}")
    print("=" * 60)
    
    extract_clip_features(
        wav_path=audio_path,
        alignment_csv_path=alignment_path,
        output_npz_path=out_npz,
        pitch_floor=args.pitch_floor,
        pitch_ceiling=args.pitch_ceiling,
        seed=args.seed,
    )

    print("\n" + "=" * 60)
    print(f"RENDERING FEATURE VISUALIZATION")
    print("=" * 60)
    plot_feature_contours(
        features_source=out_npz,
        output_png_path=out_plot,
        clip_title=audio_path.name,
    )
    print("\n[Done] Feature extraction and visualization complete!")

if __name__ == "__main__":
    main()
