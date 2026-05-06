#!/usr/bin/env python3
"""Parse Google Gemini batch prediction results into the standard filtering artifacts."""

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
    response = row.get("response", {})
    candidates = response.get("candidates") or row.get("candidates") or []
    for candidate in candidates:
        parts = candidate.get("content", {}).get("parts", [])
        texts = [part.get("text", "") for part in parts if "text" in part]
        if texts:
            return "".join(texts)
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("output/google"))
    args = parser.parse_args()

    rows = load_jsonl(args.results)
    with open(args.manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    rows_by_key = {row.get("key"): row for row in rows}
    joined = []
    for item in manifest["requests"]:
        row = rows_by_key.get(item["key"], {})
        joined.append({
            "key": item["key"],
            "concepts": item["concepts"],
            "response_text": extract_text(row) or "",
        })

    args.output.mkdir(parents=True, exist_ok=True)
    with open(args.output / "05_raw_responses.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    build_outputs_from_joined(joined, args.output)


if __name__ == "__main__":
    main()
