"""Audio format conversion and validation utilities for Laya.

Provides ffmpeg conversion to 16 kHz mono WAV, container magic byte validation,
and duration/size limit enforcement.

No long dashes in any text or comments.
"""

from pathlib import Path
import json
import os
import shutil
import subprocess
from typing import Optional, Tuple, Union

MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MB
MAX_DURATION_SECONDS = 150.0         # 150 seconds


def is_ffmpeg_available() -> bool:
    """Check if ffmpeg is available on system PATH."""
    return shutil.which("ffmpeg") is not None


def get_audio_stream_info(file_path: Union[str, Path]) -> dict:
    """Get audio stream details (sample_rate, channels, duration)."""
    ffprobe_bin = shutil.which("ffprobe")
    if not ffprobe_bin:
        import soundfile as sf
        info = sf.info(str(file_path))
        return {
            "sample_rate": int(info.samplerate),
            "channels": int(info.channels),
            "duration": float(info.duration),
        }

    cmd = [
        ffprobe_bin,
        "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=sample_rate,channels,duration:format=duration",
        "-of", "json",
        str(file_path),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError("ffprobe failed to probe audio stream.")

    data = json.loads(proc.stdout or "{}")
    streams = data.get("streams", [])
    if not streams:
        raise RuntimeError("No audio stream found in file.")

    st = streams[0]
    dur_str = st.get("duration") or data.get("format", {}).get("duration")
    duration = float(dur_str) if dur_str else 0.0
    return {
        "sample_rate": int(st.get("sample_rate", 0)),
        "channels": int(st.get("channels", 0)),
        "duration": duration,
    }


def check_audio_magic_bytes(header: bytes) -> bool:
    """Verify that file header contains recognized audio container signatures.
    
    Checks for:
    - WAV (RIFF....WAVE)
    - WebM / Matroska (EBML ID 0x1A45DFA3)
    - Ogg (OggS)
    - MP4 / M4A / MOV (ftyp or moov box)
    - MP3 (ID3 tag or frame sync header)
    - FLAC (fLaC)
    """
    if len(header) < 4:
        return False

    # RIFF WAV
    if header.startswith(b"RIFF") and len(header) >= 12 and header[8:12] == b"WAVE":
        return True

    # WebM / MKV (EBML identifier)
    if header.startswith(b"\x1a\x45\xdf\xa3"):
        return True

    # Ogg container
    if header.startswith(b"OggS"):
        return True

    # MP4 / M4A / MOV container
    if len(header) >= 8 and (header[4:8] == b"ftyp" or header[4:8] == b"moov"):
        return True

    # MP3 ID3v2 tag
    if header.startswith(b"ID3"):
        return True

    # MP3 sync word (11 bits set: 0xFF followed by high 3 bits set in second byte)
    if len(header) >= 2 and header[0] == 0xFF and (header[1] & 0xE0) == 0xE0:
        return True

    # FLAC
    if header.startswith(b"fLaC"):
        return True

    return False


def probe_audio_stream(file_path: Union[str, Path]) -> Tuple[bool, float, Optional[str]]:
    """Probe audio stream codec and duration using ffprobe.
    
    Returns:
        (is_valid, duration_seconds, error_message)
    """
    ffprobe_bin = shutil.which("ffprobe")
    if not ffprobe_bin:
        # Fallback to soundfile if ffprobe is absent
        try:
            import soundfile as sf
            info = sf.info(str(file_path))
            return True, float(info.duration), None
        except Exception as e:
            return False, 0.0, f"Cannot probe audio: {str(e)}"

    cmd = [
        ffprobe_bin,
        "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=codec_type,duration:format=duration",
        "-of", "json",
        str(file_path),
    ]

    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode != 0:
            return False, 0.0, "Invalid or corrupt audio file."

        data = json.loads(proc.stdout or "{}")
        streams = data.get("streams", [])
        if not streams:
            return False, 0.0, "No audio stream found in uploaded file."

        dur_str = streams[0].get("duration") or data.get("format", {}).get("duration")
        duration = float(dur_str) if dur_str else 0.0
        return True, duration, None
    except Exception as exc:
        return False, 0.0, f"Probe error: {str(exc)}"


def validate_audio_content(
    file_path: Union[str, Path],
    max_bytes: int = MAX_UPLOAD_BYTES,
    max_duration: float = MAX_DURATION_SECONDS,
) -> Tuple[bool, float, Optional[str]]:
    """Validate uploaded audio by size, header content, and probed duration.
    
    Returns:
        (is_valid, duration_seconds, error_message)
    """
    path = Path(file_path)
    if not path.exists() or not path.is_file():
        return False, 0.0, "File does not exist."

    # 1. Size check
    file_size = path.stat().st_size
    if file_size > max_bytes:
        return False, 0.0, f"File size ({file_size / (1024 * 1024):.1f} MB) exceeds maximum allowed limit of {max_bytes // (1024 * 1024)} MB."

    if file_size < 16:
        return False, 0.0, "File is too small to contain valid audio data."

    # 2. Magic bytes check
    with open(path, "rb") as f:
        header = f.read(64)

    if not check_audio_magic_bytes(header):
        return False, 0.0, "Invalid audio file content. The file does not match any supported audio signature."

    # 3. Probing duration and stream
    is_valid, duration, err = probe_audio_stream(path)
    if not is_valid:
        return False, 0.0, err or "Invalid audio stream."

    if duration > max_duration:
        return False, duration, f"Audio duration ({duration:.1f} s) exceeds maximum allowed limit of {max_duration:.0f} s."

    return True, duration, None


def convert_to_wav16k_mono(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
) -> Path:
    """Convert any audio file (webm, ogg, mp4, wav, etc.) to 16 kHz mono WAV using ffmpeg.
    
    Args:
        input_path: Path to source audio file.
        output_path: Destination WAV path. If omitted, creates a sibling file with .wav suffix.
        
    Returns:
        Path to converted 16 kHz mono WAV file.
        
    Raises:
        RuntimeError: If ffmpeg is missing or conversion fails.
    """
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        raise RuntimeError("ffmpeg is required for audio conversion but was not found on PATH. Please install ffmpeg.")

    in_p = Path(input_path)
    if not in_p.exists():
        raise FileNotFoundError(f"Input file not found: {in_p}")

    out_p = Path(output_path) if output_path else in_p.with_suffix(".wav")
    out_p.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        ffmpeg_bin,
        "-y",
        "-i", str(in_p),
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        str(out_p),
    ]

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        err_msg = proc.stderr.strip() or "Unknown ffmpeg error."
        raise RuntimeError(f"ffmpeg conversion failed: {err_msg}")

    if not out_p.exists() or out_p.stat().st_size == 0:
        raise RuntimeError("ffmpeg conversion did not produce a valid output WAV file.")

    return out_p
