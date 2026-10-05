"""Tests for I/O and reproducibility utilities."""

import tempfile
from pathlib import Path
import numpy as np

from src.utils.seed import set_seed
from src.utils.io import read_transcript, save_json, load_json

def test_set_seed_reproducibility():
    """Verify set_seed produces reproducible numpy random draws."""
    set_seed(42)
    draw1 = np.random.randn(10)

    set_seed(42)
    draw2 = np.random.randn(10)

    np.testing.assert_array_equal(draw1, draw2)

def test_json_and_transcript_io():
    """Verify reading transcripts and saving/loading JSON."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        txt_path = tmp_path / "sample.txt"
        txt_path.write_text("  This is a   multiline\n transcript test.  ", encoding="utf-8")

        cleaned = read_transcript(txt_path)
        assert cleaned == "This is a multiline transcript test."

        json_path = tmp_path / "sample.json"
        data = {"speech": "sample", "words": 5}
        save_json(data, json_path)
        loaded = load_json(json_path)
        assert loaded == data
