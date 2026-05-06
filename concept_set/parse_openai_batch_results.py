#!/usr/bin/env python3
"""Parse OpenAI Batch API results into the standard filtering artifacts."""

import argparse
import json
from pathlib import Path

from batch_result_utils import build_outputs_from_joined


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def extract_text(row: dict) -> str | None:
    response = row.get("response", {}).get("body", {})
    choices = response.get("choices", [])
    if not choices:
        return None
    return choices[0].get("message", {}).get("content")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("output/openai"))
    args = parser.parse_args()

    rows = load_jsonl(args.results)
    with open(args.manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    rows_by_id = {row["custom_id"]: row for row in rows}
    joined = []
    for item in manifest["requests"]:
        row = rows_by_id.get(item["custom_id"], {})
        joined.append({
            "custom_id": item["custom_id"],
            "concepts": item["concepts"],
            "response_text": extract_text(row) or "",
        })

    args.output.mkdir(parents=True, exist_ok=True)
    with open(args.output / "05_raw_responses.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    build_outputs_from_joined(joined, args.output)


if __name__ == "__main__":
    main()
