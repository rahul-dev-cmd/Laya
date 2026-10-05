"""I/O utilities for reading/writing transcripts and alignment JSON files."""

import json
from pathlib import Path
from typing import Any, Dict

def read_transcript(file_path: str | Path) -> str:
    """Read a plain text speech transcript.
    
    Args:
        file_path: Path to the transcript file.
        
    Returns:
        Cleaned transcript string with normalized whitespace.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Transcript file not found: {path}")
    
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    
    # Normalize excessive whitespace but preserve word boundaries
    return " ".join(text.strip().split())

def save_json(data: Dict[str, Any], output_path: str | Path, indent: int = 2) -> None:
    """Save dictionary to JSON with directory creation.
    
    Args:
        data: Dictionary data to serialize.
        output_path: Destination JSON file path.
        indent: Indentation level for pretty-printing.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)

def load_json(file_path: str | Path) -> Dict[str, Any]:
    """Load JSON file safely.
    
    Args:
        file_path: Path to the JSON file.
        
    Returns:
        Parsed JSON as dictionary.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
