"""
PII Class Distribution Analyzer
------------------------------
This script reads one or multiple JSONL files containing NER labels,
calculates the distribution of each class, and generates a formatted
Excel report. It provides both a detailed BIO-tag distribution and
a grouped class summary.

Example usage:
    python analyzer.py -i train.jsonl val.jsonl -o report.xlsx
"""

import json
import argparse
import pandas as pd
from collections import Counter
import sys
import re


def strip_bio_prefix(label):
    """
    Removes B- or I- prefix from NER labels to get the base class.
    Example: 'B-nam_org_company' -> 'nam_org_company', 'O' -> 'O'
    """
    if label == "O":
        return "O"
    return re.sub(r"^[BI]-", "", label)


def analyze_pii_distribution(input_files, output_excel):
    """
    Reads JSONL files, counts NER labels, and generates a formatted
    Excel report with two sheets: Full Distribution and Class Summary.
    """
    all_labels = []

    print(f"Loading files: {input_files}...")

    try:
        for file_path in input_files:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    all_labels.extend(data.get("labels", []))

        if not all_labels:
            print("Error: No labels found in the provided files.")
            return

        # --- 1. Full Distribution (Exact labels) ---
        full_counts = Counter(all_labels)
        df_full = pd.DataFrame(full_counts.items(), columns=["Label", "Count"])
        df_full = df_full.sort_values(by="Count", ascending=False).reset_index(
            drop=True
        )
        total_tokens = df_full["Count"].sum()
        df_full["Percentage"] = (df_full["Count"] / total_tokens * 100).round(
            2
        ).astype(str) + "%"

        # --- 2. Class Summary (Grouped B/I labels) ---
        grouped_labels = [strip_bio_prefix(l) for l in all_labels]
        group_counts = Counter(grouped_labels)
        df_grouped = pd.DataFrame(group_counts.items(), columns=["Class", "Count"])
        df_grouped = df_grouped.sort_values(by="Count", ascending=False).reset_index(
            drop=True
        )
        df_grouped["Percentage"] = (df_grouped["Count"] / total_tokens * 100).round(
            2
        ).astype(str) + "%"

        print(f"Analysis complete. Found {total_tokens} tokens.")

        # --- Excel Export ---
        writer = pd.ExcelWriter(output_excel, engine="xlsxwriter")

        df_full.to_excel(writer, sheet_name="Full Distribution", index=False)
        df_grouped.to_excel(writer, sheet_name="Class Summary", index=False)

        workbook = writer.book

        header_format = workbook.add_format(
            {
                "bold": True,
                "text_wrap": True,
                "valign": "top",
                "fg_color": "#D7E4BC",
                "border": 1,
            }
        )
        cell_format = workbook.add_format({"border": 1})

        for sheet_name, df in [
            ("Full Distribution", df_full),
            ("Class Summary", df_grouped),
        ]:
            worksheet = writer.sheets[sheet_name]

            for col_num, value in enumerate(df.columns.values):
                worksheet.write(0, col_num, value, header_format)

            for row_num in range(1, len(df) + 1):
                for col_num in range(len(df.columns)):
                    val = df.iloc[row_num - 1, col_num]
                    worksheet.write(row_num, col_num, val, cell_format)

        # --- Chart Generation (Based on Class Summary, excluding 'O') ---
        df_chart = df_grouped[df_grouped["Class"] != "O"].reset_index(drop=True)
        chart_data_sheet_name = "ChartData"
        df_chart.to_excel(writer, sheet_name=chart_data_sheet_name, index=False)
        writer.sheets[chart_data_sheet_name].hide()

        chart = workbook.add_chart({"type": "column"})
        chart.add_series(
            {
                "name": "Occurrences",
                "categories": [chart_data_sheet_name, 1, 0, len(df_chart), 0],
                "values": [chart_data_sheet_name, 1, 1, len(df_chart), 1],
                "fill": {"color": "#4F81BD"},
            }
        )

        chart.set_title({"name": "PII Class Distribution (excluding Outside)"})
        chart.set_x_axis({"name": "PII Class"})
        chart.set_y_axis({"name": "Count"})
        chart.set_style(11)

        summary_sheet = writer.sheets["Class Summary"]
        summary_sheet.insert_chart("E2", chart, {"x_scale": 1.5, "y_scale": 1.5})

        writer.close()
        print(f"Report successfully saved to: {output_excel}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except json.JSONDecodeError:
        print("Error: One of the files is not in a valid JSONL format.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze class distribution from JSONL files and generate an Excel report."
    )
    parser.add_argument(
        "-i", "--input", nargs="+", required=True, help="Paths to JSONL files"
    )
    parser.add_argument(
        "-o", "--output", required=True, help="Path to the output .xlsx file"
    )

    args = parser.parse_args()
    analyze_pii_distribution(args.input, args.output)


if __name__ == "__main__":
    main()
