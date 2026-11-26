import argparse
import json
import os
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]


def download_csv(
    url: str,
    output_path: Path,
    source_name: str,
    license_name: str,
) -> None:
    if not url.lower().startswith(("https://", "http://")):
        raise ValueError("Dataset URL must use HTTP or HTTPS.")

    response = requests.get(url, timeout=(10, 60))
    response.raise_for_status()
    frame = pd.read_csv(StringIO(response.content.decode("utf-8-sig")))
    if frame.empty:
        raise ValueError("Downloaded dataset contains no rows.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    metadata_path = output_path.with_suffix(".metadata.json")
    metadata_path.write_text(
        json.dumps(
            {
                "source_name": source_name,
                "source_url": url,
                "license": license_name,
                "downloaded_rows": len(frame),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Downloaded {len(frame)} rows to {output_path}")
    print(f"Recorded dataset attribution in {metadata_path}")


def download_json(
    url: str,
    output_path: Path,
    source_name: str,
    license_name: str,
) -> None:
    if not url.lower().startswith(("https://", "http://")):
        raise ValueError("Dataset URL must use HTTP or HTTPS.")

    response = requests.get(url, timeout=(10, 60))
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict) or not isinstance(payload.get("rows"), list):
        raise ValueError("Expected a Hugging Face datasets-server JSON response.")
    if not payload["rows"]:
        raise ValueError("Downloaded dataset contains no rows.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    flat_rows = []
    for item in payload["rows"]:
        if isinstance(item, dict) and isinstance(item.get("row"), dict):
            flat_rows.append(item["row"])
        elif isinstance(item, dict):
            flat_rows.append(item)

    csv_path = output_path.with_suffix(".csv")
    pd.DataFrame(flat_rows).to_csv(csv_path, index=False)

    metadata_path = output_path.with_suffix(".metadata.json")
    metadata_path.write_text(
        json.dumps(
            {
                "source_name": source_name,
                "source_url": url,
                "license": license_name,
                "downloaded_rows": len(payload["rows"]),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Downloaded {len(payload['rows'])} rows to {output_path}")
    print(f"Saved flattened CSV data to {csv_path}")
    print(f"Recorded dataset attribution in {metadata_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a CSV dataset you are licensed to use."
    )
    parser.add_argument("--url", default=os.getenv("TICKET_DATASET_URL"))
    parser.add_argument("--source-name", default=os.getenv("DATASET_SOURCE_NAME"))
    parser.add_argument("--license", default=os.getenv("DATASET_LICENSE"))
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "external" / "tickets_download.csv",
    )
    args = parser.parse_args()
    if not args.url or not args.source_name or not args.license:
        parser.error(
            "Provide --url, --source-name, and --license (or set their DATASET_* values in .env)."
        )
    if "datasets-server.huggingface.co/rows" in args.url:
        json_output = args.output.with_suffix(".json")
        download_json(args.url, json_output, args.source_name, args.license)
    else:
        download_csv(args.url, args.output, args.source_name, args.license)


if __name__ == "__main__":
    main()

