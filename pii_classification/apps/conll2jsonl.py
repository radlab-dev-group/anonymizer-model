import json
import argparse
import os
import sys


def convert_conll_to_jsonl(input_file, output_file):
    """
    Converts a CONLL formatted file to JSONL format.
    Extracts the first column (text) and the last column (NER).
    """
    sentences = []
    current_tokens = []
    current_labels = []

    try:
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if not line:
                    if current_tokens:
                        sentences.append(
                            {"text": current_tokens, "labels": current_labels}
                        )
                        current_tokens = []
                        current_labels = []
                    continue

                if line.startswith("-DOCSTART"):
                    continue

                parts = line.split()
                if len(parts) >= 2:
                    word = parts[0]
                    label = parts[-1]

                    current_tokens.append(word)
                    current_labels.append(label)

            if current_tokens:
                sentences.append({"text": current_tokens, "labels": current_labels})

        with open(output_file, "w", encoding="utf-8") as f:
            for sentence in sentences:
                f.write(json.dumps(sentence, ensure_ascii=False) + "\n")

        print(f"Sukces: Przetworzono {len(sentences)} zdań.")
        print(f"Plik zapisany w: {output_file}")

    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Converter from CONLL format to JSONL format for NER tasks."
    )
    parser.add_argument(
        "-i", "--input", required=True, help="Path to the input file in CONLL format"
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Path to the output file in JSONL format",
    )

    args = parser.parse_args()
    convert_conll_to_jsonl(args.input, args.output)


if __name__ == "__main__":
    main()
