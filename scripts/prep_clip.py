#!/usr/bin/env python3
"""Preparation script for new speech clips in Laya dataset.

Inputs:
    1. mp3 path: Path to source MP3 recording
    2. start time: Start offset (e.g., '00:00:21', '00:00:25.5', '21')
    3. length: Duration in seconds (e.g., '95', '91.0')
    4. output name: Output clip identifier (e.g., 'librivox_03')
    5. source-text file: Path to reference book / text file

Actions:
    - Checks for existing files and strictly refuses to overwrite existing audio,
      transcripts, alignments, diff reports, or sources_table rows.
    - Cuts a 16 kHz mono 16-bit PCM WAV into dataset/sources/.
    - Runs existing WhisperX alignment by reusing scripts/run_alignment.py.
    - Generates a diff report (like diff_01.txt) documenting discrepancies
      between Whisper transcription and source text passage.
    - Prints the first and last 5 spoken words and any numerals in the text
      that need converting to words.
    - Appends one row (with blank license and URL fields, no guessing)
      to dataset/sources_table.csv.

Usage:
    python scripts/prep_clip.py raw_03.mp3 00:00:21 95 librivox_03 source_text_03.txt
    python scripts/prep_clip.py --mp3 dataset/sources/raw_03.mp3 --start 00:00:21 --length 95 --name librivox_03 --source-text dataset/sources/source_text_03.txt
"""

import sys
import os
import re
import csv
import difflib
import argparse
import subprocess
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

# Prefer project virtual environment if available when invoked directly
if __name__ == "__main__":
    venv_py = Path(__file__).resolve().parent.parent / ".venv" / "Scripts" / "python.exe"
    if venv_py.exists() and sys.executable.lower() != str(venv_py).lower():
        result = subprocess.call([str(venv_py), __file__] + sys.argv[1:])
        sys.exit(result)

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    DEFAULT_DEVICE,
    RANDOM_SEED,
    TARGET_SAMPLE_RATE,
)
from scripts.run_alignment import run_alignment


TABLE_COLUMNS = [
    "clip_id",
    "wav_file",
    "raw_mp3",
    "start_time",
    "duration_seconds",
    "word_count",
    "source_text_file",
    "audio_url",
    "text_url",
    "audio_license",
    "text_license",
]


def parse_time_seconds(time_val: Any) -> float:
    """Parse time representations like '00:00:21', '00:00:25.5', '21', '21.0' into float seconds."""
    s = str(time_val).strip()
    if ":" in s:
        parts = s.split(":")
        if len(parts) == 3:
            h, m, sec = parts
            return int(h) * 3600 + int(m) * 60 + float(sec)
        elif len(parts) == 2:
            m, sec = parts
            return int(m) * 60 + float(sec)
    return float(s)


def format_time_mmss(seconds: float) -> str:
    """Format seconds into MM:SS.SS format."""
    m = int(seconds // 60)
    s = seconds % 60
    return f"{m:02d}:{s:05.2f}"


def clean_token(word: str) -> str:
    """Normalize token for robust diff comparison."""
    return re.sub(r'[^a-zA-Z0-9]', '', word).lower()


def strip_punctuation(word: str) -> str:
    """Strip edge punctuation while preserving inner hyphens/apostrophes."""
    return re.sub(r'^[^\w]+|[^\w]+$', '', word)


def cut_audio_wav(
    mp3_path: Path,
    start_time: str,
    length_sec: float,
    output_wav: Path,
) -> None:
    """Cut 16 kHz mono 16-bit PCM WAV using ffmpeg, falling back to librosa if needed."""
    output_wav.parent.mkdir(parents=True, exist_ok=True)

    # Primary: ffmpeg
    cmd = [
        "ffmpeg",
        "-ss", str(start_time),
        "-i", str(mp3_path),
        "-t", str(length_sec),
        "-ar", str(TARGET_SAMPLE_RATE),
        "-ac", "1",
        "-c:a", "pcm_s16le",
        str(output_wav),
    ]

    use_fallback = False
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0 or not output_wav.is_file() or output_wav.stat().st_size == 0:
            print(f"[PrepClip] ffmpeg failed (code {proc.returncode}), using Python audio fallback...")
            use_fallback = True
    except FileNotFoundError:
        print("[PrepClip] ffmpeg not found in PATH, using Python audio fallback...")
        use_fallback = True

    if use_fallback:
        import librosa
        import soundfile as sf
        start_sec = parse_time_seconds(start_time)
        y, sr = librosa.load(
            str(mp3_path),
            sr=TARGET_SAMPLE_RATE,
            mono=True,
            offset=start_sec,
            duration=length_sec,
        )
        sf.write(str(output_wav), y, TARGET_SAMPLE_RATE, subtype="PCM_16")

    print(f"[PrepClip] Cut 16 kHz mono WAV saved: {output_wav} ({output_wav.stat().st_size / 1024:.1f} KB)")


def extract_passage_from_source(
    source_tokens: List[str],
    spoken_words: List[str],
) -> Tuple[List[str], int, int]:
    """Find corresponding passage slice in source text using SequenceMatcher."""
    source_clean = [clean_token(w) for w in source_tokens]
    spoken_clean = [clean_token(w) for w in spoken_words]

    # If source text is already roughly the size of spoken words, use entire source
    if len(source_tokens) <= len(spoken_words) * 1.5:
        return source_tokens, 0, len(source_tokens)

    sm = difflib.SequenceMatcher(None, source_clean, spoken_clean)
    blocks = [b for b in sm.get_matching_blocks() if b.size > 2]

    if not blocks:
        # Fallback to entire text if no large blocks
        return source_tokens, 0, len(source_tokens)

    start_idx = max(0, blocks[0].a - blocks[0].b)
    end_idx = min(len(source_tokens), blocks[-1].a + (len(spoken_clean) - blocks[-1].b))
    return source_tokens[start_idx:end_idx], start_idx, end_idx


def find_numerals_in_passage(passage_tokens: List[str]) -> List[Tuple[str, str]]:
    """Find numerals/digits in source passage that require conversion to words."""
    numerals: List[Tuple[str, str]] = []
    for idx, tok in enumerate(passage_tokens):
        if re.search(r'\d', tok):
            cleaned = strip_punctuation(tok)
            if cleaned:
                ctx_start = max(0, idx - 2)
                ctx_end = min(len(passage_tokens), idx + 3)
                context = " ".join(passage_tokens[ctx_start:ctx_end])
                numerals.append((cleaned, context))
    return numerals


def classify_discrepancy(
    tag: str,
    s_tokens: List[str],
    w_tokens: List[str],
) -> str:
    """Generate detailed classification message for diff report."""
    s_text = " ".join(s_tokens)
    w_text = " ".join(w_tokens)
    s_clean = clean_token(s_text)
    w_clean = clean_token(w_text)

    if tag == "insert" or not s_tokens:
        return "Extraneous word insertion. Spoken/Whisper output contains additional word(s) absent in source passage."
    elif tag == "delete" or not w_tokens:
        return "Word omission / elision. Source text word(s) omitted or not recognized in speech audio."

    has_num_s = bool(re.search(r'\d', s_text))
    has_num_w = bool(re.search(r'\d', w_text))
    if has_num_s or has_num_w:
        return "Number formatting. Source text contains digits that speaker pronounced as words (or Whisper normalized to numerals)."

    if s_text.lower() == w_text.lower():
        if s_clean == w_clean and s_text != w_text:
            return "Punctuation / apostrophe discrepancy. Differs in punctuation or apostrophe usage."
        return "Capitalization difference. Differs in uppercase/lowercase casing."

    if s_clean == w_clean:
        return "Punctuation / apostrophe discrepancy. Spoken identically; differs in punctuation conventions."

    if re.sub(r'\s+', '', s_text.lower()) == re.sub(r'\s+', '', w_text.lower()):
        return "Orthographic compounding / spacing. Spoken identically; differs in closed compound vs separate words."

    return "Phonetic substitution / discrepancy between printed source and Whisper speech recognition."


def generate_diff_report(
    clip_id: str,
    wav_filename: str,
    raw_mp3_filename: str,
    start_str: str,
    length_sec: float,
    source_text_filename: str,
    passage_tokens: List[str],
    whisper_words_entries: List[Dict[str, Any]],
    output_diff_path: Path,
) -> str:
    """Generate and write a diff report formatted like diff_01.txt."""
    spoken_words = [w["word"].strip() for w in whisper_words_entries]
    passage_clean = [clean_token(w) for w in passage_tokens]
    whisper_clean = [clean_token(w) for w in spoken_words]

    # Calculate coverage
    coverage_end = length_sec
    if whisper_words_entries:
        coverage_end = max(coverage_end, whisper_words_entries[-1].get("end", length_sec))

    lines: List[str] = []
    lines.append(f"DIFF REPORT: {clip_id.upper()} ({wav_filename}, {length_sec:.1f} s cut from {raw_mp3_filename} with -ss {start_str} -t {length_sec})")
    lines.append(f"Source Text File: {source_text_filename}")
    lines.append(f"Audio File: {wav_filename}")
    lines.append(f"Passage Coverage: 00:00.00 - {format_time_mmss(coverage_end)} ({length_sec:.1f} s WAV file)")
    lines.append("")
    lines.append("=" * 100)
    lines.append("DISCREPANCIES BETWEEN WHISPER OUTPUT AND SOURCE PASSAGE")
    lines.append("=" * 100)
    lines.append("")

    matcher = difflib.SequenceMatcher(None, passage_clean, whisper_clean)
    diff_count = 0

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue

        diff_count += 1
        s_chunk = " ".join(passage_tokens[i1:i2]) if i1 < i2 else "[omitted]"
        w_chunk = " ".join(spoken_words[j1:j2]) if j1 < j2 else "[omitted]"

        # Determine timestamps
        if j1 < len(whisper_words_entries) and j2 > j1:
            t_s = whisper_words_entries[j1]["start"]
            t_e = whisper_words_entries[min(j2 - 1, len(whisper_words_entries) - 1)]["end"]
        elif j1 < len(whisper_words_entries):
            t_s = whisper_words_entries[j1]["start"]
            t_e = whisper_words_entries[j1]["end"]
        elif whisper_words_entries:
            t_s = whisper_words_entries[-1]["end"]
            t_e = t_s
        else:
            t_s, t_e = 0.0, 0.0

        detail = classify_discrepancy(tag, passage_tokens[i1:i2], spoken_words[j1:j2])

        lines.append(f"{diff_count}. [{format_time_mmss(t_s)} - {format_time_mmss(t_e)}]")
        lines.append(f"   - Whisper: {w_chunk}")
        lines.append(f"   - Source:  {s_chunk}")
        lines.append(f"   - Detail:  {detail}")
        lines.append("")

    if diff_count == 0:
        lines.append("No discrepancies found between Whisper transcription and source text.")
        lines.append("")

    report_text = "\n".join(lines)
    output_diff_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_diff_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"[PrepClip] Diff report saved to: {output_diff_path} ({diff_count} discrepancies)")
    return report_text


def update_sources_table(
    table_path: Path,
    row_dict: Dict[str, Any],
) -> None:
    """Add one row to dataset/sources_table.csv, creating file with header if not present."""
    table_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = table_path.is_file()

    # If file exists, check for duplicate clip_id or wav_file
    if file_exists:
        with open(table_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("clip_id") == row_dict["clip_id"]:
                    raise FileExistsError(
                        f"Refusing to overwrite: clip_id '{row_dict['clip_id']}' is already registered in {table_path}"
                    )
                if row.get("wav_file") == row_dict["wav_file"]:
                    raise FileExistsError(
                        f"Refusing to overwrite: wav_file '{row_dict['wav_file']}' is already registered in {table_path}"
                    )

    with open(table_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=TABLE_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)

    print(f"[PrepClip] Added row to {table_path}: {row_dict['clip_id']}")


def main():
    parser = argparse.ArgumentParser(
        description="Cut speech clip to 16 kHz mono WAV, align with Whisper, produce diff report, and register in sources_table.csv."
    )
    # Positional or optional flags
    parser.add_argument("pos_mp3", nargs="?", default=None, help="Input MP3 file path")
    parser.add_argument("pos_start", nargs="?", default=None, help="Start time (e.g. '00:00:21', '21')")
    parser.add_argument("pos_length", nargs="?", default=None, help="Length in seconds (e.g. '95')")
    parser.add_argument("pos_name", nargs="?", default=None, help="Output clip name (e.g. 'librivox_03')")
    parser.add_argument("pos_source_text", nargs="?", default=None, help="Source-text file path")

    parser.add_argument("--mp3", "-i", dest="opt_mp3", default=None, help="Input MP3 file path")
    parser.add_argument("--start", "-ss", "--start-time", dest="opt_start", default=None, help="Start time")
    parser.add_argument("--length", "-t", "--duration", dest="opt_length", default=None, help="Length in seconds")
    parser.add_argument("--name", "-n", "--output-name", dest="opt_name", default=None, help="Output clip name")
    parser.add_argument("--source-text", "-s", "--text", dest="opt_source_text", default=None, help="Source-text file path")

    parser.add_argument("--device", default=DEFAULT_DEVICE, choices=["cpu", "cuda"], help="Inference device (default: cpu)")
    parser.add_argument("--model", default="base.en", help="Whisper model size (default: base.en)")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed (default: 42)")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "dataset" / "sources"), help="Output directory for WAV and diffs")
    parser.add_argument("--table", default=str(PROJECT_ROOT / "dataset" / "sources_table.csv"), help="Path to sources_table.csv")

    args = parser.parse_args()

    # Resolve arguments
    mp3_arg = args.opt_mp3 or args.pos_mp3
    start_arg = args.opt_start or args.pos_start
    length_arg = args.opt_length or args.pos_length
    name_arg = args.opt_name or args.pos_name
    source_text_arg = args.opt_source_text or args.pos_source_text

    if not all([mp3_arg, start_arg, length_arg, name_arg, source_text_arg]):
        parser.error(
            "All 5 parameters are required: mp3, start, length, name, source-text.\n"
            "Example: python scripts/prep_clip.py raw_01.mp3 00:00:21 95 librivox_01 source_text_01.txt"
        )

    mp3_path = Path(mp3_arg)
    if not mp3_path.is_file():
        # Check relative to dataset/sources if not found directly
        alt_mp3 = PROJECT_ROOT / "dataset" / "sources" / mp3_arg
        if alt_mp3.is_file():
            mp3_path = alt_mp3
        else:
            print(f"Error: Input MP3 file not found: {mp3_path}")
            sys.exit(1)

    source_text_path = Path(source_text_arg)
    if not source_text_path.is_file():
        alt_txt = PROJECT_ROOT / "dataset" / "sources" / source_text_arg
        if alt_txt.is_file():
            source_text_path = alt_txt
        else:
            print(f"Error: Source-text file not found: {source_text_path}")
            sys.exit(1)

    clip_name = Path(name_arg).stem
    length_sec = parse_time_seconds(length_arg)
    output_dir = Path(args.output_dir)
    table_path = Path(args.table)

    # 1. Target files
    target_wav = output_dir / f"{clip_name}.wav"
    target_csv = output_dir / f"{clip_name}_alignment.csv"
    target_json = output_dir / f"{clip_name}_aligned.json"

    num_match = re.search(r'(\d+)$', clip_name)
    if num_match:
        target_diff = output_dir / f"diff_{num_match.group(1)}.txt"
    else:
        target_diff = output_dir / f"diff_{clip_name}.txt"

    # 2. Strict check: Refuse to overwrite any existing files
    existing_conflicts: List[str] = []
    if target_wav.exists():
        existing_conflicts.append(str(target_wav))
    if target_csv.exists():
        existing_conflicts.append(str(target_csv))
    if target_json.exists():
        existing_conflicts.append(str(target_json))
    if target_diff.exists():
        existing_conflicts.append(str(target_diff))

    if table_path.exists():
        with open(table_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if r.get("clip_id") == clip_name:
                    existing_conflicts.append(f"Row '{clip_name}' in {table_path}")
                    break
                if r.get("wav_file") == target_wav.name:
                    existing_conflicts.append(f"Row '{target_wav.name}' in {table_path}")
                    break

    if existing_conflicts:
        print("\n[ERROR] Refusing to overwrite existing files / registrations:")
        for conflict in existing_conflicts:
            print(f"  - {conflict}")
        print("Aborting. Existing audio, alignments, diffs, and tables remain strictly unmodified.")
        sys.exit(1)

    print(f"\n[PrepClip] Starting preparation for clip '{clip_name}'...")
    print(f"  Input MP3:    {mp3_path}")
    print(f"  Start Time:   {start_arg}")
    print(f"  Length:       {length_sec}s")
    print(f"  Source Text:  {source_text_path}")
    print(f"  Output WAV:   {target_wav}")

    # 3. Cut 16 kHz mono WAV
    cut_audio_wav(
        mp3_path=mp3_path,
        start_time=start_arg,
        length_sec=length_sec,
        output_wav=target_wav,
    )

    # 4. Run existing alignment reusing scripts/run_alignment.py
    print(f"[PrepClip] Running speech alignment...")
    alignment_result = run_alignment(
        audio_path=target_wav,
        transcript_text=None,
        output_path=target_json,
        device=args.device,
        model_size=args.model,
        seed=args.seed,
    )

    whisper_words_entries = alignment_result.get("words", [])
    spoken_words = [w["word"].strip() for w in whisper_words_entries]

    # Save alignment CSV matching librivox_01_alignment.csv schema
    with open(target_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["word", "start_seconds", "end_seconds"])
        writer.writeheader()
        for w in whisper_words_entries:
            writer.writerow({
                "word": w["word"],
                "start_seconds": w["start"],
                "end_seconds": w["end"],
            })
    print(f"[PrepClip] Alignment CSV saved: {target_csv} ({len(whisper_words_entries)} words)")

    # 5. Read source text and extract passage
    with open(source_text_path, "r", encoding="utf-8", errors="replace") as f:
        raw_source_text = f.read()
    source_tokens = raw_source_text.split()

    passage_tokens, p_start, p_end = extract_passage_from_source(source_tokens, spoken_words)
    print(f"[PrepClip] Extracted passage tokens [{p_start}:{p_end}] ({len(passage_tokens)} words)")

    # 6. Find numerals in passage needing conversion to words
    numerals = find_numerals_in_passage(passage_tokens)

    # 7. Print first and last 5 spoken words and numerals
    print("\n" + "=" * 80)
    print("SPOKEN WORDS SUMMARY")
    print("=" * 80)
    print(f"First 5 spoken words: {' '.join(spoken_words[:5])}")
    print(f"Last 5 spoken words:  {' '.join(spoken_words[-5:])}")
    print("-" * 80)
    print("NUMERALS IN SOURCE TEXT NEEDING CONVERSION TO WORDS")
    print("-" * 80)
    if numerals:
        print(f"Found {len(numerals)} numeral(s) in source text passage:")
        for num, ctx in numerals:
            print(f"  • '{num}' in context: \"{ctx}\"")
    else:
        print("None detected (no digit numerals in source passage).")
    print("=" * 80 + "\n")

    # 8. Generate diff report like diff_01.txt
    generate_diff_report(
        clip_id=clip_name,
        wav_filename=target_wav.name,
        raw_mp3_filename=mp3_path.name,
        start_str=str(start_arg),
        length_sec=length_sec,
        source_text_filename=source_text_path.name,
        passage_tokens=passage_tokens,
        whisper_words_entries=whisper_words_entries,
        output_diff_path=target_diff,
    )

    # 9. Add one row (blank license and URL fields) to dataset/sources_table.csv
    row_entry = {
        "clip_id": clip_name,
        "wav_file": target_wav.name,
        "raw_mp3": mp3_path.name,
        "start_time": str(start_arg),
        "duration_seconds": round(length_sec, 2),
        "word_count": len(spoken_words),
        "source_text_file": source_text_path.name,
        "audio_url": "",      # Blank - no guessing of URLs
        "text_url": "",       # Blank - no guessing of URLs
        "audio_license": "",  # Blank - no guessing of licenses
        "text_license": "",   # Blank - no guessing of licenses
    }
    update_sources_table(table_path, row_entry)

    print("\n" + "=" * 80)
    print(f"CLIP PREPARATION COMPLETED SUCCESSFULLY: {clip_name}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
