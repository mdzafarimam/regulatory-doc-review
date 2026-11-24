"""Export the golden evaluation dataset to a CSV file for manual upload to
LangSmith (Datasets & Experiments -> + New Dataset -> Upload CSV).

LangSmith's CSV upload lets you map one or more columns to "Input" fields and
one or more columns to "Output" fields. This script writes one column per
input/output key required by the graph's state schema (`ReviewState`) plus
the reference output fields used in `dataset.py`:

    messages          -> input field "messages" (empty list, JSON-encoded)
    document          -> input field "document"
    reviews           -> input field "reviews" (empty list, JSON-encoded)
    final_report      -> input field "final_report" (empty string)
    reference_notes   -> output field "reference_notes"
    expected_verdict  -> output field "expected_verdict"

Usage:
    python -m agents.regulatory_review.evaluation.export_csv
    python -m agents.regulatory_review.evaluation.export_csv --out path/to/file.csv

After generating the CSV, in the LangSmith UI:
    1. Go to Datasets & Experiments -> "+ New Dataset" -> "Upload a CSV file".
    2. Select the generated CSV.
    3. Map "messages", "document", "reviews", and "final_report" columns as
       Input, and "reference_notes" / "expected_verdict" columns as Output.
    4. Name the dataset (e.g. "regulatory-review-eval") and create it.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from agents.regulatory_review.evaluation.dataset import get_dataset_examples

DEFAULT_OUT_PATH = Path(__file__).resolve().parent / "golden_dataset.csv"

FIELDNAMES = [
    "messages",
    "document",
    "reviews",
    "final_report",
    "reference_notes",
    "expected_verdict",
]


def export_to_csv(out_path: Path = DEFAULT_OUT_PATH) -> Path:
    """Write all golden dataset examples to a CSV file at `out_path`."""
    examples = get_dataset_examples()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for example in examples:
            inputs = example.get("inputs", {})
            outputs = example.get("outputs", {})
            writer.writerow(
                {
                    "messages": json.dumps(inputs.get("messages", [])),
                    "document": inputs.get("document", ""),
                    "reviews": json.dumps(inputs.get("reviews", [])),
                    "final_report": inputs.get("final_report", ""),
                    "reference_notes": outputs.get("reference_notes", ""),
                    "expected_verdict": outputs.get("expected_verdict", ""),
                }
            )

    print(f"Wrote {len(examples)} example(s) to {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT_PATH,
        help="Output CSV file path (default: evaluation/golden_dataset.csv)",
    )
    args = parser.parse_args()
    export_to_csv(args.out)


if __name__ == "__main__":
    main()
