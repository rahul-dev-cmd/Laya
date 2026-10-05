"""CLI tool to run WhisperX word-level alignment on a speech recording and transcript.

Usage:
    python scripts/run_alignment.py
    python scripts/run_alignment.py --audio data/processed/gettysburg_address_16k.wav --transcript data/transcripts/gettysburg_address.txt --device cpu
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    PROCESSED_DATA_DIR,
    TRANSCRIPTS_DIR,
    DEFAULT_DEVICE,
    RANDOM_SEED,
)
from src.utils.io import read_transcript
from src.alignment.aligner import align_speech

def main():
    parser = argparse.ArgumentParser(
        description="Run WhisperX word-level alignment on a speech audio file and transcript."
    )
    parser.add_argument(
        "--audio",
        type=str,
        default=str(PROCESSED_DATA_DIR / "gettysburg_address_16k.wav"),
        help="Path to audio file (default: data/processed/gettysburg_address_16k.wav)",
    )
    parser.add_argument(
        "--transcript",
        type=str,
        default=str(TRANSCRIPTS_DIR / "gettysburg_address.txt"),
        help="Path to reference transcript file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output JSON file (default: data/processed/<audio_stem>_aligned.json)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
        choices=["cpu", "cuda"],
        help="Inference device (default: cpu)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="base.en",
        help="Whisper model size (default: base.en)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.is_file():
        print(f"Error: Audio file not found at {audio_path}")
        print("Tip: Run `python scripts/download_sample_speech.py` first.")
        sys.exit(1)

    transcript_text = None
    if args.transcript:
        transcript_path = Path(args.transcript)
        if transcript_path.is_file():
            transcript_text = read_transcript(transcript_path)
            print(f"[Run] Loaded transcript from {transcript_path} ({len(transcript_text.split())} words)")
        else:
            print(f"[Run] Warning: Transcript file {transcript_path} not found. Running transcription directly.")

    out_json = args.output
    if out_json is None:
        out_json = PROCESSED_DATA_DIR / f"{audio_path.stem}_aligned.json"

    print(f"[Run] Aligning: {audio_path.name}")
    print(f"[Run] Device: {args.device} | Model: {args.model} | Seed: {args.seed}")

    result = align_speech(
        audio_path=audio_path,
        transcript_text=transcript_text,
        output_json_path=out_json,
        device=args.device,
        model_size=args.model,
        seed=args.seed,
    )

    total_words = result.get("total_words", 0)
    duration = result.get("duration_seconds", 0.0)
    print("\n" + "=" * 50)
    print(f"ALIGNMENT SUCCESSFUL!")
    print(f"Duration: {duration:.2f}s | Words Aligned: {total_words}")
    print(f"Output saved to: {out_json}")
    
    # Print sample of first 8 words
    print("\nSample Aligned Words:")
    for w in result.get("words", [])[:8]:
        print(f"  [{w['start']:6.2f}s - {w['end']:6.2f}s]  {w['word']:<15} (score: {w['score']:.2f})")
    if total_words > 8:
        print(f"  ... and {total_words - 8} more words.")
    print("=" * 50)

if __name__ == "__main__":
    main()
