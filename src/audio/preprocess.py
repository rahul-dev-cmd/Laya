"""Audio preprocessing module for Second Take.

Standard:
- Mono audio
- 16,000 Hz target sample rate
- Float32 array in range [-1.0, 1.0]
- Hop size: 10 ms (160 samples at 16 kHz)
- Window length: 25 ms (400 samples at 16 kHz)
"""

from pathlib import Path
from typing import Tuple, Dict, Any, Union
import numpy as np
import soundfile as sf
import librosa

from src.config import TARGET_SAMPLE_RATE, WINDOW_SAMPLES, HOP_SAMPLES

def load_audio(
    file_path: Union[str, Path],
    target_sr: int = TARGET_SAMPLE_RATE,
) -> Tuple[np.ndarray, int]:
    """Load audio file, convert to mono, and resample to target_sr.
    
    Args:
        file_path: Path to the input audio file (WAV, MP3, FLAC, etc.).
        target_sr: Target sample rate in Hz (default: 16,000 Hz).
        
    Returns:
        Tuple of (audio_signal: np.ndarray, sample_rate: int).
        audio_signal is 1D float32 normalized between -1.0 and 1.0.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")

    # librosa.load handles multi-channel to mono conversion and resampling
    y, sr = librosa.load(str(path), sr=target_sr, mono=True)
    y = y.astype(np.float32)

    # Normalize amplitude if non-silent
    max_val = np.max(np.abs(y))
    if max_val > 1.0:
        y = y / max_val

    return y, sr

def save_audio(
    file_path: Union[str, Path],
    audio: np.ndarray,
    sr: int = TARGET_SAMPLE_RATE,
) -> None:
    """Save audio array as a 16 kHz mono 16-bit PCM WAV file.
    
    Args:
        file_path: Destination WAV file path.
        audio: 1D audio array.
        sr: Sample rate in Hz (default: 16,000 Hz).
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr, subtype="PCM_16")

def compute_frame_rms(
    audio: np.ndarray,
    frame_length: int = WINDOW_SAMPLES,
    hop_length: int = HOP_SAMPLES,
) -> np.ndarray:
    """Compute Root Mean Square (RMS) energy per frame.
    
    Window: 25 ms (400 samples at 16 kHz)
    Hop: 10 ms (160 samples at 16 kHz)
    
    Args:
        audio: 1D audio waveform.
        frame_length: Analysis window in samples.
        hop_length: Hop length in samples.
        
    Returns:
        1D array of RMS energy per frame.
    """
    rms = librosa.feature.rms(
        y=audio,
        frame_length=frame_length,
        hop_length=hop_length,
        center=True,
    )
    return rms.squeeze(0)

def get_audio_metadata(
    audio: np.ndarray,
    sr: int = TARGET_SAMPLE_RATE,
) -> Dict[str, Any]:
    """Calculate key summary metrics of the audio array.
    
    Args:
        audio: 1D audio array.
        sr: Audio sample rate.
        
    Returns:
        Dictionary containing duration, num_samples, rms, peak, and silence ratio.
    """
    duration = float(len(audio)) / float(sr) if sr > 0 else 0.0
    peak = float(np.max(np.abs(audio))) if len(audio) > 0 else 0.0
    rms = float(np.sqrt(np.mean(audio**2))) if len(audio) > 0 else 0.0

    return {
        "sample_rate": sr,
        "num_samples": len(audio),
        "duration_seconds": round(duration, 4),
        "peak_amplitude": round(peak, 4),
        "mean_rms_energy": round(rms, 6),
    }
