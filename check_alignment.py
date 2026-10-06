#!/usr/bin/env python3
"""Alignment verification tool.

Picks 10 random words from an alignment CSV file, cuts each one out of the corresponding
WAV file as a short audio snippet, and writes a listing text file.

Outputs:
    scratch/check/clip01_word_<n>.wav (and clip02_word_<n>.wav)
    scratch/check/clip01_words.txt    (and clip02_words.txt)
"""

import sys
import os
import csv
import wave
import random
import argparse
from pathlib import Path

def extract_word_snippets(
    wav_path: str | Path,
    alignment_csv_path: str | Path,
    output_dir: str | Path = "scratch/check",
    clip_prefix: str = "clip01",
    num_words: int = 10,
    seed: int = 42,
    padding_seconds: float = 0.0,
) -> Path:
    """Extract random word audio snippets based on alignment timestamps.
    
    Args:
        wav_path: Path to source WAV audio file.
        alignment_csv_path: Path to alignment CSV with columns word, start_seconds, end_seconds.
        output_dir: Output directory to store snippets (default: scratch/check).
        clip_prefix: Prefix for snippet filenames (e.g., 'clip01' or 'clip02').
        num_words: Number of random words to sample (default: 10).
        seed: Random seed for reproducibility (default: 42).
        padding_seconds: Extra padding on each side of the cut in seconds (default: 0.0).
        
    Returns:
        Path to the generated words listing text file.
    """
    wav_path = Path(wav_path)
    alignment_csv_path = Path(alignment_csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Read alignment rows
    rows = []
    with open(alignment_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                start = float(r["start_seconds"])
                end = float(r["end_seconds"])
                word = r["word"].strip()
                if word and end > start:
                    rows.append({
                        "word": word,
                        "start_seconds": start,
                        "end_seconds": end,
                    })
            except (ValueError, KeyError):
                continue

    if len(rows) < num_words:
        raise ValueError(f"Alignment CSV contains only {len(rows)} valid words, cannot pick {num_words}.")

    # 2. Pick random words (seeded for reproducibility)
    rng = random.Random(seed)
    sampled = rng.sample(rows, num_words)

    # 3. Read source WAV file
    with wave.open(str(wav_path), "rb") as wf:
        nchannels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        nframes = wf.getnframes()
        total_duration = nframes / framerate
        audio_frames = wf.readframes(nframes)

    # Bytes per full sample frame (channels * sample width)
    frame_size = nchannels * sampwidth

    listing_lines = []
    print(f"\n[Check] Extracting {num_words} snippets from '{wav_path.name}' to '{output_dir}' (prefix: {clip_prefix}):")

    for idx, item in enumerate(sampled, start=1):
        word = item["word"]
        orig_start = item["start_seconds"]
        orig_end = item["end_seconds"]

        # Apply padding and clamp
        start_t = max(0.0, orig_start - padding_seconds)
        end_t = min(total_duration, orig_end + padding_seconds)

        start_frame = int(round(start_t * framerate))
        end_frame = int(round(end_t * framerate))
        if end_frame <= start_frame:
            end_frame = min(nframes, start_frame + 1)

        start_byte = start_frame * frame_size
        end_byte = end_frame * frame_size
        snippet_data = audio_frames[start_byte:end_byte]

        snippet_name = f"{clip_prefix}_word_{idx}.wav"
        snippet_path = output_dir / snippet_name

        with wave.open(str(snippet_path), "wb") as out_wf:
            out_wf.setnchannels(nchannels)
            out_wf.setsampwidth(sampwidth)
            out_wf.setframerate(framerate)
            out_wf.writeframes(snippet_data)

        dur = end_t - start_t
        line = f"{snippet_name}: {word} [{orig_start:.3f}s - {orig_end:.3f}s, dur={dur:.3f}s]"
        listing_lines.append(line)
        print(f"  {line}")

    # 4. Save listing text file
    words_txt_path = output_dir / f"{clip_prefix}_words.txt"
    with open(words_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(listing_lines) + "\n")

    print(f"[Check] Saved listing to '{words_txt_path}'")
    return words_txt_path

def main():
    parser = argparse.ArgumentParser(
        description="Verify word alignments by extracting random word audio snippets."
    )
    parser.add_argument("--audio", default=None, help="Path to audio WAV file")
    parser.add_argument("--alignment", default=None, help="Path to alignment CSV file")
    parser.add_argument("--output-dir", default="scratch/check", help="Directory to save snippets (default: scratch/check)")
    parser.add_argument("--prefix", default=None, help="Prefix for snippet filenames (e.g. clip01 or clip02)")
    parser.add_argument("--count", type=int, default=10, help="Number of random words (default: 10)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--padding", type=float, default=0.0, help="Padding in seconds around each word (default: 0.0)")
    parser.add_argument("--all", action="store_true", help="Process both default dataset clips (clip01 and clip02)")

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    dataset_dir = project_root / "dataset" / "sources"

    if args.all or (args.audio is None and args.alignment is None):
        # Run both clips
        tasks = [
            (
                dataset_dir / "librivox_01.wav",
                dataset_dir / "librivox_01_alignment.csv",
                "clip01",
            ),
            (
                dataset_dir / "librivox_02.wav",
                dataset_dir / "librivox_02_alignment.csv",
                "clip02",
            ),
        ]
        for wav_p, csv_p, prefix in tasks:
            if not wav_p.exists():
                print(f"Error: WAV file not found: {wav_p}", file=sys.stderr)
                continue
            if not csv_p.exists():
                print(f"Error: Alignment file not found: {csv_p}. Run align.py first.", file=sys.stderr)
                continue
            extract_word_snippets(
                wav_path=wav_p,
                alignment_csv_path=csv_p,
                output_dir=args.output_dir,
                clip_prefix=prefix,
                num_words=args.count,
                seed=args.seed,
                padding_seconds=args.padding,
            )
    else:
        if args.audio is None or args.alignment is None:
            parser.error("Both --audio and --alignment must be specified when not running on all clips.")
        prefix = args.prefix if args.prefix else Path(args.audio).stem
        extract_word_snippets(
            wav_path=args.audio,
            alignment_csv_path=args.alignment,
            output_dir=args.output_dir,
            clip_prefix=prefix,
            num_words=args.count,
            seed=args.seed,
            padding_seconds=args.padding,
        )

if __name__ == "__main__":
    main()
