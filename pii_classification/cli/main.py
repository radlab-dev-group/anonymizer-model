import sys
import argparse

from pii_classification.converters.conll2jsonl import convert_conll_to_jsonl
from pii_classification.analysis.jsonl_generalise_labels import (
    analyze_pii_distribution as generalise_labels,
)
from pii_classification.analysis.labels_distribution_report import (
    analyze_pii_distribution as generate_report,
)


def _add_common_arguments(parser):
    """Add arguments that are shared across sub‑commands (e.g., --verbose)."""
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output"
    )


def main():
    root_parser = argparse.ArgumentParser(
        prog="pii-classifier",
        description="Utilities for converting, generalising and reporting PII NER data.",
    )
    subparsers = root_parser.add_subparsers(dest="command", required=True)

    # ---- convert ----
    parser_convert = subparsers.add_parser(
        "convert", help="Convert a CONLL file to JSONL."
    )
    parser_convert.add_argument(
        "-i", "--input", required=True, help="Path to CONLL file"
    )
    parser_convert.add_argument(
        "-o", "--output", required=True, help="Path to JSONL output"
    )
    _add_common_arguments(parser_convert)

    # ---- generalise ----
    parser_gen = subparsers.add_parser(
        "generalise", help="Generalise fine‑grained NER labels using a mapping file."
    )
    parser_gen.add_argument(
        "-i", "--input", nargs="+", required=True, help="JSONL files to process"
    )
    parser_gen.add_argument(
        "-m", "--mapping", required=True, help="Mapping JSON file"
    )
    parser_gen.add_argument(
        "-o", "--output", required=True, help="Output JSONL file"
    )
    _add_common_arguments(parser_gen)

    # ---- report ----
    parser_report = subparsers.add_parser(
        "report", help="Generate an Excel report with label distribution."
    )
    parser_report.add_argument(
        "-i", "--input", nargs="+", required=True, help="JSONL files"
    )
    parser_report.add_argument(
        "-o", "--output", required=True, help="Excel file to create"
    )
    _add_common_arguments(parser_report)

    args = root_parser.parse_args()

    if args.command == "convert":
        convert_conll_to_jsonl(args.input, args.output)

    elif args.command == "generalise":
        generalise_labels(args.input, args.mapping, args.output)

    elif args.command == "report":
        generate_report(args.input, args.output)

    else:
        root_parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
