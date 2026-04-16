import json
import re
import sys

from pathlib import Path


def load_mapping(mapping_path: str | Path) -> dict:
    """Load a JSON mapping file."""
    try:
        with open(mapping_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading mapping file: {e}")
        sys.exit(1)


def generalize_label(label: str, mapping: dict) -> str:
    """Map a BIO‑prefixed label to its high‑level category."""
    if label == "O":
        return "O"
    base_label = re.sub(r"^[BI]-", "", label)
    for prefix, category in mapping.items():
        if base_label.startswith(prefix):
            return category
    return "MISC"


def strip_bio_prefix(label: str) -> str:
    """Remove the B‑/I‑ prefix; keep 'O' unchanged."""
    return label if label == "O" else re.sub(r"^[BI]-", "", label)
