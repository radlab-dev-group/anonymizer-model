"""
Universal PII Class Distribution Analyzer
----------------------------------------
This script reads JSONL files and generalizes NER tags based on an
external JSON mapping file. It supports prefix-based mapping to
handle fine-grained labels.

Example usage:
    python analyzer.py -i train.jsonl val.jsonl -m mapping.json -o report.xlsx
"""

import json
import argparse
import pandas as pd
from collections import Counter
import sys
import re


def load_mapping(mapping_path):
    """Loads the mapping dictionary from a JSON file."""
    try:
        with open(mapping_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading mapping file: {e}")
        sys.exit(1)


def generalize_label(label, mapping):
    """
    Generalizes a label based on the provided mapping dictionary.
    Matches the base label against prefixes defined in the JSON.
    """
    if label == "O":
        return "O"

    # 1. Remove BIO prefix (B- or I-)
    base_label = re.sub(r"^[BI]-", "", label)

    # 2. Iterate through mapping.
    # The order in the JSON file determines the priority.
    for prefix, category in mapping.items():
        if base_label.startswith(prefix):
            return category

    return "MISC"  # Fallback if no prefix matches


def analyze_pii_distribution(input_files, mapping_path, output_path):
    """
    Reads JSONL files, maps the labels using the provided mapping,
    and writes a new JSONL file containing the converted dataset.
    """
    # Load the universal mapping
    mapping = load_mapping(mapping_path)
    all_labels = []
    mapped_records = []  # <-- new container for converted records

    print(f"Loading files: {input_files}...")

    try:
        for file_path in input_files:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    original_labels = data.get("labels", [])
                    all_labels.extend(original_labels)

                    # Convert labels using the mapping
                    mapped_labels = [
                        generalize_label(l, mapping) for l in original_labels
                    ]
                    data["labels"] = mapped_labels
                    mapped_records.append(data)  # <-- store the converted record

        if not all_labels:
            print("Error: No labels found.")
            return

        total_tokens = len(all_labels)

        # --- Statistics (optional, kept for console feedback) ---
        full_counts = Counter(all_labels)
        gen_counts = Counter(
            [lbl for rec in mapped_records for lbl in rec.get("labels", [])]
        )

        print(f"Analysis complete. Found {total_tokens} tokens.")
        print("Original label distribution:")
        for lbl, cnt in full_counts.most_common():
            print(f"  {lbl}: {cnt}")

        print("\nGeneralized label distribution:")
        for lbl, cnt in gen_counts.most_common():
            print(f"  {lbl}: {cnt}")

        # --- Write converted dataset to JSONL ---
        print(f"Saving converted dataset to: {output_path}")
        with open(output_path, "w", encoding="utf-8") as out_f:
            for record in mapped_records:
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Conversion and saving completed successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Generalize PII labels using a JSON map and output a converted JSONL file."
    )
    parser.add_argument(
        "-i", "--input", nargs="+", required=True, help="Paths to JSONL files"
    )
    parser.add_argument(
        "-m", "--mapping", required=True, help="Path to the mapping JSON file"
    )
    parser.add_argument(
        "-o", "--output", required=True, help="Path to output converted .jsonl file"
    )

    args = parser.parse_args()
    analyze_pii_distribution(args.input, args.mapping, args.output)


if __name__ == "__main__":
    main()
