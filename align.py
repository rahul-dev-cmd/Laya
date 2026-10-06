#!/usr/bin/env python3
"""Word-level forced alignment tool for speech audio and reference transcripts.

Uses Whisper word-level acoustic alignment matched against the reference transcript
via dynamic sequence alignment to produce precise word start and end timestamps.

Output format:
    CSV with columns: word, start_seconds, end_seconds
"""

import sys
import os
from pathlib import Path

# Prefer project's virtual environment if available.
venv_py = Path(__file__).resolve().parent / ".venv" / "Scripts" / "python.exe"
if venv_py.exists() and sys.executable.lower() != str(venv_py).lower():
    import subprocess
    result = subprocess.call([str(venv_py), __file__] + sys.argv[1:])
    sys.exit(result)

import faster_whisper

import re
import csv
import argparse
import difflib
from typing import List, Tuple, Dict, Any

def clean_token(word: str) -> str:
    """Normalize word for robust text-to-audio sequence matching."""
    return re.sub(r'[^a-zA-Z0-9]', '', word).lower()

def strip_punctuation(word: str) -> str:
    """Strip leading and trailing punctuation while preserving internal hyphens."""
    cleaned = re.sub(r'^[^\w]+|[^\w]+$', '', word)
    return cleaned if cleaned else word

def align_transcript_to_audio(
    wav_path: str | Path,
    transcript_path: str | Path,
    output_csv_path: str | Path | None = None,
    model_size: str = "base.en",
    device: str = "cpu",
    compute_type: str = "int8",
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Align reference transcript words to audio using Whisper word timestamps.
    
    Args:
        wav_path: Path to input WAV file.
        transcript_path: Path to reference transcript text file.
        output_csv_path: Path to save the alignment CSV. If None, saves next to wav.
        model_size: Whisper model size (default: base.en).
        device: Device to run inference on (default: cpu).
        compute_type: Computation type (default: int8).
        
    Returns:
        Tuple of (aligned_words_list, unplaced_words_list)
    """
    wav_path = Path(wav_path)
    transcript_path = Path(transcript_path)
    
    if output_csv_path is None:
        output_csv_path = wav_path.parent / f"{wav_path.stem}_alignment.csv"
    else:
        output_csv_path = Path(output_csv_path)

    # 1. Read reference transcript
    with open(transcript_path, "r", encoding="utf-8") as f:
        raw_words = f.read().split()

    print(f"[Aligner] Loading Whisper model '{model_size}' on {device} ({compute_type})...")
    model = faster_whisper.WhisperModel(model_size, device=device, compute_type=compute_type)

    print(f"[Aligner] Transcribing audio '{wav_path.name}' with word-level timestamps...")
    segments, info = model.transcribe(str(wav_path), word_timestamps=True)
    
    whisper_words = []
    for s in segments:
        if s.words:
            for w in s.words:
                whisper_words.append({
                    "word": w.word.strip(),
                    "start": round(float(w.start), 3),
                    "end": round(float(w.end), 3),
                    "probability": round(float(w.probability), 4),
                })

    print(f"[Aligner] Target words: {len(raw_words)}, Whisper recognized words: {len(whisper_words)}")

    # 2. Sequence alignment between transcript tokens and Whisper tokens
    t_clean = [clean_token(w) for w in raw_words]
    w_clean = [clean_token(w["word"]) for w in whisper_words]

    matcher = difflib.SequenceMatcher(None, t_clean, w_clean)
    aligned_rows: List[Dict[str, Any]] = []
    unplaced_words: List[str] = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for t_idx, w_idx in zip(range(i1, i2), range(j1, j2)):
                aligned_rows.append({
                    "word": strip_punctuation(raw_words[t_idx]),
                    "start_seconds": whisper_words[w_idx]["start"],
                    "end_seconds": whisper_words[w_idx]["end"],
                })
        elif tag == "delete":
            # Words present in reference transcript but completely missing/omitted in speech
            for t_idx in range(i1, i2):
                unplaced = strip_punctuation(raw_words[t_idx])
                unplaced_words.append(unplaced)
                print(f"[Aligner] Word could not be placed in audio: '{unplaced}' (transcript index {t_idx})")
        elif tag == "insert":
            # Audio has extra recognized token not in transcript; ignore
            pass
        elif tag == "replace":
            t_sub = raw_words[i1:i2]
            w_sub = whisper_words[j1:j2]
            if len(w_sub) == 0:
                for w in t_sub:
                    unplaced_words.append(strip_punctuation(w))
                continue

            span_start = w_sub[0]["start"]
            span_end = w_sub[-1]["end"]
            span_dur = max(0.01, span_end - span_start)

            # Distribute time among target words proportionally to length
            lengths = [max(1, len(clean_token(w))) for w in t_sub]
            total_len = sum(lengths)
            curr_start = span_start

            for idx, w in enumerate(t_sub):
                w_dur = round(span_dur * (lengths[idx] / total_len), 3)
                w_end = round(curr_start + w_dur, 3) if idx < len(t_sub) - 1 else span_end
                aligned_rows.append({
                    "word": strip_punctuation(w),
                    "start_seconds": curr_start,
                    "end_seconds": w_end,
                })
                curr_start = w_end

    # Post-process: ensure all words have a strictly positive duration
    for k in range(len(aligned_rows)):
        s = aligned_rows[k]["start_seconds"]
        e = aligned_rows[k]["end_seconds"]
        if e <= s:
            prev_e = aligned_rows[k-1]["end_seconds"] if k > 0 else 0.0
            next_s = aligned_rows[k+1]["start_seconds"] if k + 1 < len(aligned_rows) else s + 0.25
            if next_s > e:
                aligned_rows[k]["end_seconds"] = round(min(next_s, s + 0.25), 3)
            elif prev_e < s:
                aligned_rows[k]["start_seconds"] = round(max(prev_e, s - 0.15), 3)
            else:
                aligned_rows[k]["end_seconds"] = round(s + 0.08, 3)

    # 3. Write CSV output
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["word", "start_seconds", "end_seconds"])
        writer.writeheader()
        for row in aligned_rows:
            writer.writerow(row)

    print(f"[Aligner] Successfully wrote {len(aligned_rows)} aligned words to '{output_csv_path}'")
    if unplaced_words:
        print(f"[Aligner] Total unplaced words: {len(unplaced_words)} -> {unplaced_words}")
    else:
        print("[Aligner] All transcript words were placed successfully!")

    return aligned_rows, unplaced_words

def main():
    parser = argparse.ArgumentParser(
        description="Forced alignment of speech WAV and transcript text to word-level timestamps."
    )
    parser.add_argument("audio", nargs="?", default=None, help="Path to audio WAV file")
    parser.add_argument("transcript", nargs="?", default=None, help="Path to transcript TXT file")
    parser.add_argument("output", nargs="?", default=None, help="Path to output CSV file")
    parser.add_argument("--model", default="base.en", help="Whisper model size (default: base.en)")
    parser.add_argument("--device", default="cpu", help="Inference device (default: cpu)")
    parser.add_argument("--all", action="store_true", help="Process both default dataset clips (librivox_01 and librivox_02)")

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    dataset_dir = project_root / "dataset" / "sources"

    # Default to running both clips if no specific clip provided or if --all
    if args.all or (args.audio is None and args.transcript is None):
        clips = [
            (dataset_dir / "librivox_01.wav", dataset_dir / "librivox_01.txt", dataset_dir / "librivox_01_alignment.csv"),
            (dataset_dir / "librivox_02.wav", dataset_dir / "librivox_02.txt", dataset_dir / "librivox_02_alignment.csv"),
        ]
        all_unplaced = {}
        for wav_p, txt_p, csv_p in clips:
            print("\n" + "=" * 60)
            print(f"Aligning {wav_p.name}...")
            print("=" * 60)
            _, unplaced = align_transcript_to_audio(
                wav_path=wav_p,
                transcript_path=txt_p,
                output_csv_path=csv_p,
                model_size=args.model,
                device=args.device,
            )
            all_unplaced[wav_p.name] = unplaced
            
        print("\n" + "=" * 60)
        print("SUMMARY OF UNPLACED WORDS:")
        print("=" * 60)
        for clip_name, unplaced in all_unplaced.items():
            if unplaced:
                print(f"  {clip_name}: {unplaced}")
            else:
                print(f"  {clip_name}: None (all words placed)")
    else:
        if args.audio is None or args.transcript is None:
            parser.error("Both audio and transcript paths must be provided when not running with --all.")
        align_transcript_to_audio(
            wav_path=args.audio,
            transcript_path=args.transcript,
            output_csv_path=args.output,
            model_size=args.model,
            device=args.device,
        )

if __name__ == "__main__":
    main()
