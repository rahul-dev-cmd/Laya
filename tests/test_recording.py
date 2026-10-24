"""Tests for audio recording, conversion, passage listing, and manifest hiding.

Requirements:
1. Conversion of a short webm or ogg sample to 16 kHz mono WAV.
2. Rejection of an oversized file.
3. Rejection of a file with the wrong content.
4. Temporary files removed after the analysis.
5. The passages list only includes passages with both files.
6. The manifest hides the rate example.
"""

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest
import scipy.io.wavfile
from starlette.testclient import TestClient

from backend.app import SCRATCH_DIR, app
from src.audio.convert import (
    convert_to_wav16k_mono,
    get_audio_stream_info,
    is_ffmpeg_available,
    probe_audio_stream,
    validate_audio_content,
)

client = TestClient(app)


def generate_short_media_sample(fmt="webm", duration=0.5):
    """Helper to synthesize a clean short audio sample using ffmpeg."""
    temp_file = tempfile.NamedTemporaryFile(suffix=f".{fmt}", delete=False)
    temp_file.close()
    out_path = temp_file.name

    codec = "libopus" if fmt in ("webm", "ogg") else "aac"
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"sine=frequency=440:duration={duration}",
        "-c:a",
        codec,
        out_path,
    ]
    res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if res.returncode != 0 or not os.path.exists(out_path):
        if os.path.exists(out_path):
            os.remove(out_path)
        pytest.skip(f"ffmpeg could not synthesize test sample format {fmt}")
    return out_path


def test_conversion_short_webm_and_ogg_to_wav16k_mono():
    """Verify conversion of a short webm or ogg sample to 16 kHz mono WAV."""
    assert is_ffmpeg_available(), "ffmpeg must be installed on PATH"

    for fmt in ["webm", "ogg"]:
        sample_path = generate_short_media_sample(fmt=fmt, duration=0.6)
        out_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name

        try:
            # Validate source media content
            validate_audio_content(sample_path)

            # Convert to 16 kHz mono WAV
            convert_to_wav16k_mono(sample_path, out_wav)
            assert os.path.exists(out_wav), f"Converted WAV was not created for {fmt}"

            # Check stream metadata using ffprobe
            stream_info = get_audio_stream_info(out_wav)
            assert stream_info["sample_rate"] == 16000, f"Expected 16 kHz, got {stream_info['sample_rate']}"
            assert stream_info["channels"] == 1, f"Expected 1 channel, got {stream_info['channels']}"
            assert stream_info["duration"] > 0.0

            # Double check via scipy wavfile reader
            sr, data = scipy.io.wavfile.read(out_wav)
            assert sr == 16000
            assert data.ndim == 1
            assert len(data) > 0
        finally:
            if os.path.exists(sample_path):
                os.remove(sample_path)
            if os.path.exists(out_wav):
                os.remove(out_wav)


def test_rejection_of_oversized_file():
    """Verify rejection of files exceeding the 25 MB limit."""
    # Test through analyze endpoint with a payload larger than 25 MB (25 MB + 1 KB)
    oversized_bytes = b"0" * (25 * 1024 * 1024 + 1024)
    response = client.post(
        "/api/analyze",
        files={"audio_file": ("huge_recording.webm", oversized_bytes, "audio/webm")},
    )
    assert response.status_code == 400
    detail = response.json().get("detail", "")
    assert "25 MB" in detail or "large" in detail.lower()


def test_rejection_of_wrong_content():
    """Verify rejection of files with invalid content or disguised non-audio data."""
    # Case 1: Plain text file pretending to be webm
    text_payload = b"Hello, this is just plain text content pretending to be a webm file."
    response = client.post(
        "/api/analyze",
        files={"audio_file": ("fake_audio.webm", text_payload, "audio/webm")},
    )
    assert response.status_code == 400
    detail = response.json().get("detail", "")
    assert "content" in detail.lower() or "format" in detail.lower() or "invalid" in detail.lower()

    # Case 2: Random invalid binary bytes
    random_bytes = os.urandom(1024)
    response2 = client.post(
        "/api/analyze",
        files={"audio_file": ("garbage.ogg", random_bytes, "audio/ogg")},
    )
    assert response2.status_code == 400


def test_temporary_files_removed_after_analysis():
    """Verify that temporary files and folders are cleaned up after POST /api/analyze."""
    webm_sample = generate_short_media_sample(fmt="webm", duration=0.8)
    try:
        # Record directory listing in SCRATCH_DIR before the request
        before_entries = set(os.listdir(SCRATCH_DIR)) if SCRATCH_DIR.exists() else set()

        with open(webm_sample, "rb") as f:
            file_bytes = f.read()

        # Submit to analyze endpoint
        response = client.post(
            "/api/analyze",
            files={"audio_file": ("temp_test.webm", file_bytes, "audio/webm")},
        )
        # Even if analysis succeeds or fails on alignment due to synthetic tone, temp files must be removed
        after_entries = set(os.listdir(SCRATCH_DIR)) if SCRATCH_DIR.exists() else set()
        new_entries = after_entries - before_entries

        # Verify no orphaned directories or files were left in SCRATCH_DIR
        assert len(new_entries) == 0, f"Lingering files found in {SCRATCH_DIR}: {new_entries}"
    finally:
        if os.path.exists(webm_sample):
            os.remove(webm_sample)


def test_passages_list_includes_only_both_files():
    """Verify that GET /api/passages only returns passages having both .wav and .txt in dataset/sources."""
    response = client.get("/api/passages")
    assert response.status_code == 200
    passages = response.json()
    assert isinstance(passages, list)
    assert len(passages) >= 2

    sources_dir = Path("dataset/sources")
    returned_ids = set()

    for p in passages:
        pid = p["id"]
        returned_ids.add(pid)
        wav_path = sources_dir / f"{pid}.wav"
        txt_path = sources_dir / f"{pid}.txt"

        assert wav_path.exists(), f"WAV file {wav_path} does not exist"
        assert txt_path.exists(), f"Transcript file {txt_path} does not exist"
        assert p["title"]
        assert p["source"]
        assert p["duration"] > 0
        assert p["text"]
        assert len(p["text"].strip()) > 0

    # librivox_01 and librivox_02 have both .wav and .txt
    assert "librivox_01" in returned_ids
    assert "librivox_02" in returned_ids

    # librivox_05 only has .wav in dataset/sources, so it must NOT be included
    wav_05 = sources_dir / "librivox_05.wav"
    txt_05 = sources_dir / "librivox_05.txt"
    if wav_05.exists() and not txt_05.exists():
        assert "librivox_05" not in returned_ids


def test_manifest_hides_rate_example():
    """Verify that dataset/examples/manifest.json hides librivox_01_rate_tier4."""
    manifest_path = Path("dataset/examples/manifest.json")
    assert manifest_path.exists(), "dataset/examples/manifest.json must exist"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    if isinstance(manifest, list):
        examples_list = manifest
    else:
        examples_list = manifest.get("examples", [])
    examples_by_id = {item["name"]: item for item in examples_list}

    # librivox_01_rate_tier4 must be hidden with the exact note
    assert "librivox_01_rate_tier4" in examples_by_id
    rate_item = examples_by_id["librivox_01_rate_tier4"]
    assert rate_item["visible"] is False
    assert rate_item["note"] == "Hidden: the full pipeline raised flags outside the damaged region, still being investigated."

    # librivox_01_pitch_tier5 must be visible
    assert "librivox_01_pitch_tier5" in examples_by_id
    pitch_item = examples_by_id["librivox_01_pitch_tier5"]
    assert pitch_item["visible"] is True

    # GET /api/examples must not include hidden examples by default
    res = client.get("/api/examples")
    assert res.status_code == 200
    active_examples = res.json()
    active_ids = [ex["id"] for ex in active_examples]

    assert "librivox_01_pitch_tier5" in active_ids
    assert "librivox_01_rate_tier4" not in active_ids
