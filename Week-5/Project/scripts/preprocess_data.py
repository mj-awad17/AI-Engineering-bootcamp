import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.categories import CATEGORY_NAMES


def normalize_dataset(input_path: Path, output_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(input_path)
    normalized_columns = {column.casefold().strip(): column for column in frame.columns}

    text_column = next(
        (normalized_columns[name] for name in ("ticket_text", "text", "description") if name in normalized_columns),
        None,
    )
    subject_column = normalized_columns.get("subject")
    category_column = next(
        (normalized_columns[name] for name in ("category", "label", "ticket_type") if name in normalized_columns),
        None,
    )
    if text_column is None or category_column is None:
        raise ValueError(
            "CSV must have a ticket_text (or text/description) column and a category (or label/ticket_type) column."
        )

    text = frame[text_column].fillna("").astype(str).str.strip()
    if subject_column and subject_column != text_column:
        subject = frame[subject_column].fillna("").astype(str).str.strip()
        text = (subject + "\n" + text).str.strip()

    category_lookup = {name.casefold(): name for name in CATEGORY_NAMES}
    categories = frame[category_column].fillna("").astype(str).str.strip()
    mapped_categories = categories.str.casefold().map(category_lookup)
    if mapped_categories.isna().any():
        unsupported = sorted(categories[mapped_categories.isna()].unique().tolist())
        raise ValueError(
            "Unsupported category labels: "
            + ", ".join(repr(label) for label in unsupported)
            + ". Map source labels to the ten supported categories before training."
        )

    result = pd.DataFrame({"ticket_text": text, "category": mapped_categories})
    result = result[result["ticket_text"].str.len() > 0].drop_duplicates()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize labeled ticket CSV data.")
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data" / "raw" / "demo_tickets.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "processed" / "tickets.csv",
    )
    args = parser.parse_args()
    result = normalize_dataset(args.input, args.output)
    print(f"Saved {len(result)} tickets to {args.output}")


if __name__ == "__main__":
    main()

