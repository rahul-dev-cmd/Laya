"""Download a verified public-domain LibriVox speech and prepare transcript.

Speech: Abraham Lincoln's Gettysburg Address
Reader: John Greenman
Source: Internet Archive (LibriVox Collection)
URL: https://archive.org/download/gettysburg_johng_librivox/gettysburg_address_64kb.mp3
License: Public Domain / LibriVox Public Domain Dedication
"""

import sys
import urllib.request
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    TRANSCRIPTS_DIR,
    TARGET_SAMPLE_RATE,
)
from src.utils.io import save_json
from src.audio.preprocess import load_audio, save_audio, get_audio_metadata

AUDIO_SOURCE_URL = (
    "https://archive.org/download/gettysburg_johng_librivox/gettysburg_address_64kb.mp3"
)

GETTYSBURG_TRANSCRIPT = (
    "Four score and seven years ago our fathers brought forth on this continent, "
    "a new nation, conceived in Liberty, and dedicated to the proposition that all "
    "men are created equal. "
    "Now we are engaged in a great civil war, testing whether that nation, or any "
    "nation so conceived and so dedicated, can long endure. We are met on a great "
    "battle-field of that war. We have come to dedicate a portion of that field, "
    "as a final resting place for those who here gave their lives that that nation "
    "might live. It is altogether fitting and proper that we should do this. "
    "But, in a larger sense, we can not dedicate, we can not consecrate, we can not "
    "hallow, this ground. The brave men, living and dead, who struggled here, have "
    "consecrated it, far above our poor power to add or detract. The world will little "
    "note, nor long remember what we say here, but it can never forget what they "
    "did here. It is for us the living, rather, to be dedicated here to the unfinished "
    "work which they who fought here have thus far so nobly advanced. It is rather "
    "for us to be here dedicated to the great task remaining before us, that from "
    "these honored dead we take increased devotion to that cause for which they "
    "gave the last full measure of devotion, that we here highly resolve that these "
    "dead shall not have died in vain, that this nation, under God, shall have a new "
    "birth of freedom, and that government of the people, by the people, for the "
    "people, shall not perish from the earth."
)

def download_and_preprocess() -> Path:
    """Download the sample LibriVox speech, write transcript, and preprocess to 16 kHz mono WAV."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

    raw_mp3 = RAW_DATA_DIR / "gettysburg_address.mp3"
    processed_wav = PROCESSED_DATA_DIR / "gettysburg_address_16k.wav"
    transcript_txt = TRANSCRIPTS_DIR / "gettysburg_address.txt"
    metadata_json = TRANSCRIPTS_DIR / "gettysburg_address_metadata.json"

    # 1. Download raw MP3 if not already present
    if not raw_mp3.is_file():
        print(f"[Download] Fetching verified LibriVox audio from:\n  {AUDIO_SOURCE_URL}")
        headers = {"User-Agent": "SecondTakeSpeechAnalytics/1.0"}
        req = urllib.request.Request(AUDIO_SOURCE_URL, headers=headers)
        with urllib.request.urlopen(req) as resp, open(raw_mp3, "wb") as out_f:
            out_f.write(resp.read())
        print(f"[Download] Saved raw audio: {raw_mp3} ({raw_mp3.stat().st_size / 1024:.1f} KB)")
    else:
        print(f"[Download] Raw audio already exists: {raw_mp3}")

    # 2. Write verified transcript
    with open(transcript_txt, "w", encoding="utf-8") as f:
        f.write(GETTYSBURG_TRANSCRIPT.strip() + "\n")
    print(f"[Transcript] Written to {transcript_txt}")

    # 3. Preprocess to standard 16 kHz mono float32 WAV
    print(f"[Preprocess] Loading & converting to {TARGET_SAMPLE_RATE} Hz mono WAV...")
    audio_arr, sr = load_audio(raw_mp3, target_sr=TARGET_SAMPLE_RATE)
    save_audio(processed_wav, audio_arr, sr=sr)
    meta = get_audio_metadata(audio_arr, sr=sr)
    print(f"[Preprocess] Saved 16 kHz mono WAV: {processed_wav}")
    print(f"             Duration: {meta['duration_seconds']}s, RMS: {meta['mean_rms_energy']}")

    # 4. Save metadata record
    metadata = {
        "title": "The Gettysburg Address",
        "author": "Abraham Lincoln",
        "reader": "John Greenman",
        "source": "LibriVox / Internet Archive",
        "source_url": AUDIO_SOURCE_URL,
        "license": "Public Domain",
        "language": "en",
        "audio_raw": str(raw_mp3.name),
        "audio_processed": str(processed_wav.name),
        "transcript_file": str(transcript_txt.name),
        "audio_metadata": meta,
    }
    save_json(metadata, metadata_json)
    print(f"[Metadata] Saved to {metadata_json}")

    return processed_wav

if __name__ == "__main__":
    download_and_preprocess()
