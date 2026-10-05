"""WhisperX word-level alignment module.

Follows project requirements:
- Audio at 16 kHz mono.
- CPU by default, GPU optional via flag.
- Fixed random seeds for reproducible runs.
- Extracts word-level start and end timestamps and alignment scores.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np

from src.config import (
    DEFAULT_DEVICE,
    COMPUTE_TYPE,
    RANDOM_SEED,
    TARGET_SAMPLE_RATE,
)
from src.utils.seed import set_seed
from src.utils.io import save_json
from src.audio.preprocess import load_audio

def align_speech(
    audio_path: Union[str, Path],
    transcript_text: Optional[str] = None,
    output_json_path: Optional[Union[str, Path]] = None,
    device: str = DEFAULT_DEVICE,
    compute_type: Optional[str] = None,
    model_size: str = "base.en",
    batch_size: int = 4,
    seed: int = RANDOM_SEED,
) -> Dict[str, Any]:
    """Perform word-level speech alignment using WhisperX.
    
    Args:
        audio_path: Path to the audio file.
        transcript_text: Optional reference text transcript.
        output_json_path: Path to save output alignment JSON.
        device: 'cpu' or 'cuda' (default: 'cpu').
        compute_type: 'int8', 'float32', or 'float16' (defaults to int8 on CPU, float16 on GPU).
        model_size: Whisper model size ('tiny.en', 'base.en', 'small.en', etc.).
        batch_size: Batch size for Whisper inference (default: 4 for light CPU).
        seed: Random seed for reproducibility.
        
    Returns:
        Dictionary containing aligned words, segments, and metadata.
    """
    # 1. Enforce reproducibility
    set_seed(seed)

    # 2. Determine device and compute type
    device = device.lower()
    if compute_type is None:
        compute_type = "int8" if device == "cpu" else "float16"

    # 3. Load & preprocess audio (16 kHz mono)
    audio_arr, sr = load_audio(audio_path, target_sr=TARGET_SAMPLE_RATE)
    duration_seconds = round(float(len(audio_arr)) / float(sr), 4)

    import whisperx

    # 4. Transcribe or prepare segments
    # WhisperX transcription model
    print(f"[Aligner] Loading Whisper model '{model_size}' on {device} ({compute_type})...")
    asr_model = whisperx.load_model(
        model_size,
        device=device,
        compute_type=compute_type,
        language="en",
    )

    print(f"[Aligner] Transcribing audio ({duration_seconds:.2f}s)...")
    transcribe_result = asr_model.transcribe(
        audio_arr,
        batch_size=batch_size,
        language="en",
    )

    # 5. Load alignment model and align to get word-level timestamps
    print(f"[Aligner] Loading alignment model for English on {device}...")
    align_model, align_metadata = whisperx.load_align_model(
        language_code="en",
        device=device,
    )

    print("[Aligner] Aligning words to audio frames...")
    aligned_result = whisperx.align(
        transcribe_result["segments"],
        align_model,
        align_metadata,
        audio_arr,
        device,
        return_char_alignments=False,
    )

    # 6. Extract structured word list
    words_list: List[Dict[str, Any]] = []
    cleaned_segments: List[Dict[str, Any]] = []

    for seg in aligned_result.get("segments", []):
        seg_entry = {
            "start": round(float(seg.get("start", 0.0)), 3),
            "end": round(float(seg.get("end", 0.0)), 3),
            "text": seg.get("text", "").strip(),
            "words": [],
        }

        for w in seg.get("words", []):
            if "start" in w and "end" in w:
                word_entry = {
                    "word": w.get("word", "").strip(),
                    "start": round(float(w["start"]), 3),
                    "end": round(float(w["end"]), 3),
                    "score": round(float(w.get("score", 1.0)), 4),
                }
                words_list.append(word_entry)
                seg_entry["words"].append(word_entry)

        cleaned_segments.append(seg_entry)

    # 7. Build standardized output payload
    output_payload: Dict[str, Any] = {
        "audio_file": str(Path(audio_path).resolve()),
        "duration_seconds": duration_seconds,
        "sample_rate": sr,
        "device": device,
        "model": model_size,
        "total_words": len(words_list),
        "reference_transcript": transcript_text if transcript_text else None,
        "words": words_list,
        "segments": cleaned_segments,
    }

    # 8. Save output if requested
    if output_json_path:
        out_path = Path(output_json_path)
        save_json(output_payload, out_path)
        print(f"[Aligner] Saved alignment JSON to {out_path} ({len(words_list)} words)")

    return output_payload
