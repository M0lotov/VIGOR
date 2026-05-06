#!/usr/bin/env python3
"""Shared helpers for parsing batch outputs into filtering artifacts."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

log = logging.getLogger("batch_result_utils")


def parse_llm_response(text: str) -> Optional[list[dict]]:
    """Parse a JSON array from a model response, tolerating markdown fences."""
    text = (text or "").strip()
    if not text:
        return None

    if text.startswith("```"):
        lines = text.split("\n")
        start = 1 if lines[0].startswith("```") else 0
        end = -1 if lines[-1].strip() == "```" else len(lines)
        text = "\n".join(lines[start:end])

    first_bracket = text.find("[")
    last_bracket = text.rfind("]")
    if first_bracket == -1 or last_bracket == -1:
        return None

    json_str = text[first_bracket:last_bracket + 1]
    try:
        result = json.loads(json_str)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        json_str = json_str.replace(",\n]", "\n]").replace(",]", "]")
        try:
            result = json.loads(json_str)
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            return None

    return None


def write_category_txts(directory: Path, by_category: dict[str, list[dict]]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for category in sorted(by_category.keys()):
        concepts = sorted(item["concept"] for item in by_category[category])
        path = directory / f"{category}.txt"
        with open(path, "w", encoding="utf-8") as f:
            if concepts:
                f.write("\n".join(concepts) + "\n")
            else:
                f.write("")


def write_rejected_txt(path: Path, results: list[dict]) -> None:
    rejected = sorted(
        r["concept"]
        for r in results
        if r.get("decision") == "reject"
    )
    with open(path, "w", encoding="utf-8") as f:
        if rejected:
            f.write("\n".join(rejected) + "\n")
        else:
            f.write("")


def build_outputs_from_joined(joined_rows: list[dict], output_dir: Path) -> None:
    """Build the same output artifacts as the live filter script."""
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for row in joined_rows:
        concepts = row.get("concepts", [])
        parsed = parse_llm_response(row.get("response_text", ""))
        if not parsed:
            for concept_meta in concepts:
                all_results.append({
                    "concept": concept_meta["concept"],
                    "original_axis": concept_meta.get("original_axis"),
                    "decision": "unfiltered",
                    "llm_error": "parse_failure",
                })
            continue

        for idx, concept_meta in enumerate(concepts):
            if idx < len(parsed):
                judgment = parsed[idx]
                all_results.append({
                    "concept": concept_meta["concept"],
                    "original_axis": concept_meta.get("original_axis"),
                    "category": judgment.get("category", "unknown"),
                    "decision": judgment.get("decision", "keep"),
                })
            else:
                all_results.append({
                    "concept": concept_meta["concept"],
                    "original_axis": concept_meta.get("original_axis"),
                    "decision": "unfiltered",
                    "llm_error": "missing_in_response",
                })

    filtered = [r for r in all_results if r.get("decision") == "keep"]

    by_category: dict[str, list[dict]] = {}
    for row in filtered:
        category = row.get("category", "unknown")
        by_category.setdefault(category, []).append({"concept": row["concept"]})

    with open(output_dir / "06_llm_judgments_full.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    with open(output_dir / "07_filtered_concept_pool.json", "w", encoding="utf-8") as f:
        json.dump(filtered, f, indent=2, ensure_ascii=False)

    with open(output_dir / "08_by_visual_category.json", "w", encoding="utf-8") as f:
        json.dump(by_category, f, indent=2, ensure_ascii=False)

    write_category_txts(output_dir / "08_filtered_by_category_txt", by_category)
    write_rejected_txt(output_dir / "09_rejected_concepts.txt", all_results)

