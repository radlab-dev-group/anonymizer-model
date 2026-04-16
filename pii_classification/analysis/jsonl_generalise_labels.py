import sys
import json
import argparse

from .label_generalizer import load_mapping, generalize_label


def analyze_pii_distribution(input_files, mapping_path, output_path):
    mapping = load_mapping(mapping_path)
    all_labels = []
    mapped_records = []

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

                    mapped_labels = [
                        generalize_label(l, mapping) for l in original_labels
                    ]
                    data["labels"] = mapped_labels
                    mapped_records.append(data)

        if not all_labels:
            print("Error: No labels found.")
            return

        total_tokens = len(all_labels)
        print(f"Analysis complete. Found {total_tokens} tokens.")

        # Write converted dataset
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
