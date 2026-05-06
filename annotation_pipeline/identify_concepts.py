#!/usr/bin/env python3
"""
Identify visual concepts in images from ``cleaned_annotations.json`` using
the OpenAI API and a predefined concept set.

The script sends each image plus the concept-identification prompt to an
OpenAI vision-capable model, writes one JSONL record per processed image,
and can optionally materialize a merged JSON summary at the end.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from openai import OpenAI


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("identify_concepts")

DEFAULT_ANNOTATIONS = Path("cleaned_annotations.json")
DEFAULT_CONCEPT_SET = Path(
    "../concept_set/output/"
    "final_concepts_cleaned/all_concepts.txt"
)
DEFAULT_OUTPUT_JSONL = Path("identified_concepts_openai_full.jsonl")
DEFAULT_OUTPUT_JSON = Path("identified_concepts_openai_full.json")
DEFAULT_MODEL = "gpt-5.4"


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a precise visual concept annotator for a vision research benchmark. \
You will be given an image and a set of visual concepts. Your task is to \
identify which concepts from the set are present in the image based on \
clear visual evidence. Be accurate and exhaustive - report every concept \
you can confidently identify, but never report concepts that lack clear \
visual support in the image. A concept must be visually grounded, not \
inferred from general knowledge about the world.\
"""


def build_prompt(
    concept_set: list[str] | None = None,
    annotated_concepts: list[str] | None = None,
) -> str:
    """
    Build the user prompt for visual concept identification.

    Args:
        concept_set: Flat list of concept names. Uses EXAMPLE_CONCEPT_SET if None.
        annotated_concepts: Concepts already annotated for this image.

    Returns:
        The user prompt string (to be sent alongside the image).
    """
    if annotated_concepts is None:
        annotated_concepts = []

    concepts_str = ", ".join(concept_set)
    annotated_str = ", ".join(annotated_concepts) if annotated_concepts else "(none)"

    return f"""\
Here is a set of visual concepts:

[{concepts_str}]

These concepts are already annotated for this image:

[{annotated_str}]

Look at the provided image and identify which additional concepts from the \
full concept set are clearly present in the image but missing from the \
already annotated list. A concept is present only if there is clear visual \
evidence for it in the image. Do not report a concept based on assumption \
or world knowledge - only report it if you can directly see the evidence. \
Do not repeat concepts that are already annotated.

Respond ONLY with a JSON object in this exact format, no other text:

{{"missing": ["concept_1", "concept_2", ...]}}

The list should contain only concepts from the set above. Do not add any \
concepts outside the set. If none are missing, return {{"missing": []}}.\
"""


def build_verification_prompt(
    concept_set: list[str],
    candidate_concepts: list[str],
) -> str:
    """
    Build a second-pass verification prompt. A different model checks
    whether each candidate concept truly has visual evidence.
    """
    concepts_str = ", ".join(concept_set)
    candidates_str = ", ".join(candidate_concepts)

    return f"""\
A previous model analyzed this image and identified the following \
concepts as present:

[{candidates_str}]

These were selected from this full concept set:

[{concepts_str}]

For each candidate concept, verify whether there is clear visual \
evidence in the image. Also check if the previous model missed any \
concepts that are clearly visible.

Respond ONLY with a JSON object in this exact format, no other text:

{{
  "confirmed": ["<concepts with clear visual evidence>"],
  "rejected": ["<concepts without sufficient evidence>"],
  "missed": ["<concepts from the full set that are clearly present but were not listed>"]
}}\
"""


def load_concept_set(path: Path) -> list[str]:
    concepts: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            concept = line.strip().strip("'\"")
            if concept:
                concepts.append(concept)
    return concepts


def load_annotations(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def extract_annotated_concepts(image: dict[str, Any]) -> list[str]:
    concepts: set[str] = set()
    for level_entries in image.get("concepts", {}).values():
        if not isinstance(level_entries, list):
            continue
        for entry in level_entries:
            category = entry.get("category")
            if isinstance(category, str):
                normalized = category.strip()
                if normalized:
                    concepts.add(normalized)
    return sorted(concepts)


def normalize_category(concept: dict[str, Any]) -> str:
    category = concept.get("category", "")
    if not isinstance(category, str):
        return ""
    return category.strip().lower()


def count_unique_concepts(image: dict[str, Any]) -> int:
    concepts = image.get("concepts", {})
    if not isinstance(concepts, dict):
        return 0

    unique_concepts: set[str] = set()
    for level in ("object", "part", "attribute", "relation"):
        level_entries = concepts.get(level, [])
        if not isinstance(level_entries, list):
            continue
        for concept in level_entries:
            if not isinstance(concept, dict):
                continue
            category = normalize_category(concept)
            if category:
                unique_concepts.add(category)
    return len(unique_concepts)


def load_processed_ids(path: Path) -> set[int]:
    if not path.exists():
        return set()

    processed_ids: set[int] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            coco_image_id = record.get("coco_image_id")
            if isinstance(coco_image_id, int):
                processed_ids.add(coco_image_id)
    return processed_ids


def sdk_response_to_dict(response: Any) -> Any:
    """Best-effort serialization for SDK response objects."""
    if response is None:
        return None

    if hasattr(response, "model_dump"):
        try:
            return response.model_dump()
        except Exception:
            pass

    if hasattr(response, "dict"):
        try:
            return response.dict()
        except Exception:
            pass

    if isinstance(response, (dict, list, str, int, float, bool)) or response is None:
        return response

    return {"repr": repr(response)}


def parse_json_object(text: str) -> dict[str, Any] | None:
    """Parse a JSON object from raw model text, tolerating markdown fences."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    first = text.find("{")
    last = text.rfind("}")
    if first == -1 or last == -1 or last < first:
        return None

    candidate = text[first:last + 1]
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        candidate = candidate.replace(",\n}", "\n}").replace(",}", "}")
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            return None

    if isinstance(parsed, dict):
        return parsed
    return None


def normalize_concepts(
    payload: dict[str, Any],
    concept_set_lookup: set[str],
    key: str,
) -> list[str]:
    values = payload.get(key, [])
    if not isinstance(values, list):
        return []

    normalized: list[str] = []
    seen: set[str] = set()
    for item in values:
        if not isinstance(item, str):
            continue
        concept = item.strip()
        if concept in concept_set_lookup and concept not in seen:
            seen.add(concept)
            normalized.append(concept)
    return normalized


def call_openai_for_image(
    client: OpenAI,
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    image_url: str,
    max_retries: int,
    retry_sleep: float,
) -> tuple[str | None, Any]:
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_prompt},
                            {"type": "image_url", "image_url": {"url": image_url}},
                        ],
                    },
                ],
            )
            text = response.choices[0].message.content if response.choices else None
            return text, response
        except Exception as exc:
            wait_seconds = retry_sleep * (2 ** attempt)
            log.warning(
                "OpenAI request failed on attempt %s/%s for %s: %s",
                attempt + 1,
                max_retries,
                image_url,
                exc,
            )
            if attempt + 1 < max_retries:
                time.sleep(wait_seconds)

    return None, None


def iter_target_images(
    annotations: dict[str, Any],
    start_index: int,
    limit: int | None,
) -> list[dict[str, Any]]:
    images = annotations.get("images", [])
    if not isinstance(images, list):
        raise ValueError("Expected annotations['images'] to be a list.")

    if start_index < 0:
        raise ValueError("--start-index must be >= 0")

    sorted_images = sorted(
        images,
        key=count_unique_concepts,
        reverse=True,
    )

    sliced = sorted_images[start_index:]
    if limit is not None:
        sliced = sliced[:limit]
    return sliced


def write_merged_output(
    *,
    annotations: dict[str, Any],
    jsonl_path: Path,
    output_path: Path,
) -> None:
    predictions_by_id: dict[int, dict[str, Any]] = {}
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            coco_image_id = record.get("coco_image_id")
            if isinstance(coco_image_id, int):
                predictions_by_id[coco_image_id] = record

    merged_images: list[dict[str, Any]] = []
    for image in annotations.get("images", []):
        coco_image_id = image.get("coco_image_id")
        merged = dict(image)
        if coco_image_id in predictions_by_id:
            pred = predictions_by_id[coco_image_id]
            merged["openai_identified_concepts"] = pred.get("present", [])
            merged["openai_identification_meta"] = {
                "model": pred.get("model"),
                "concept_set_size": pred.get("concept_set_size"),
                "image_url": pred.get("image_url"),
                "error": pred.get("error"),
            }
        merged_images.append(merged)

    payload = {
        "info": annotations.get("info", {}),
        "statistics": annotations.get("statistics", {}),
        "identification": {
            "output_jsonl": str(jsonl_path),
            "total_predictions": len(predictions_by_id),
        },
        "images": merged_images,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Identify concepts in cleaned_annotations.json with OpenAI."
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=DEFAULT_ANNOTATIONS,
        help="Path to cleaned_annotations.json",
    )
    parser.add_argument(
        "--concept-set",
        type=Path,
        default=DEFAULT_CONCEPT_SET,
        help="Path to the flat concept set text file.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=DEFAULT_OUTPUT_JSONL,
        help="Incremental JSONL output path.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_OUTPUT_JSON,
        help="Optional merged JSON output path.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="OpenAI vision-capable model name.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("OPENAI_API_KEY", ""),
        help="OpenAI API key. Defaults to OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start from this image index after sorting by unique concept count.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most this many images.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retries per API request.",
    )
    parser.add_argument(
        "--retry-sleep",
        type=float,
        default=2.0,
        help="Base sleep in seconds before retry backoff.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the JSONL output instead of resuming.",
    )
    parser.add_argument(
        "--skip-merged-json",
        action="store_true",
        help="Do not write the merged JSON summary file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the prompt and exit without calling the API.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.api_key and not args.dry_run:
        raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY or pass --api-key.")

    concept_set = load_concept_set(args.concept_set)
    concept_lookup = set(concept_set)
    annotations = load_annotations(args.annotations)
    target_images = iter_target_images(annotations, args.start_index, args.limit)
    if args.dry_run:
        example_annotated = extract_annotated_concepts(target_images[0]) if target_images else []
        prompt = build_prompt(concept_set, example_annotated)
        print("SYSTEM PROMPT:")
        print(SYSTEM_PROMPT)
        print()
        print("USER PROMPT:")
        print(prompt)
        print()
        print(f"Concept set size: {len(concept_set)}")
        print(f"Images selected: {len(target_images)}")
        return

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if args.output_json and not args.skip_merged_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)

    if args.overwrite and args.output_jsonl.exists():
        args.output_jsonl.unlink()

    processed_ids = set() if args.overwrite else load_processed_ids(args.output_jsonl)
    client = OpenAI(api_key=args.api_key)

    total = len(target_images)
    pending = sum(
        1 for image in target_images
        if image.get("coco_image_id") not in processed_ids
    )
    log.info("Loaded %s concepts from %s", len(concept_set), args.concept_set)
    log.info("Selected %s images (%s pending after resume check)", total, pending)
    log.info("Writing incremental results to %s", args.output_jsonl)

    with args.output_jsonl.open("a", encoding="utf-8") as out_handle:
        for idx, image in enumerate(target_images, start=args.start_index):
            coco_image_id = image.get("coco_image_id")
            image_info = image.get("image_info", {})
            image_url = image_info.get("coco_url", "")
            file_name = image_info.get("file_name", "")

            if not isinstance(coco_image_id, int):
                log.warning("Skipping image at index %s with invalid coco_image_id", idx)
                continue

            if coco_image_id in processed_ids:
                continue

            record: dict[str, Any] = {
                "coco_image_id": coco_image_id,
                "file_name": file_name,
                "image_url": image_url,
                "model": args.model,
                "concept_set_size": len(concept_set),
                "annotated_concepts": extract_annotated_concepts(image),
                "missing_concepts": [],
                "present": [],
                "error": None,
            }

            if not image_url:
                record["error"] = "missing_coco_url"
                out_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_handle.flush()
                processed_ids.add(coco_image_id)
                log.warning("Image %s has no coco_url; recorded error", coco_image_id)
                continue

            log.info("Processing image %s (%s)", coco_image_id, file_name)
            prompt = build_prompt(concept_set, record["annotated_concepts"])
            raw_text, raw_response = call_openai_for_image(
                client,
                model=args.model,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=prompt,
                image_url=image_url,
                max_retries=args.max_retries,
                retry_sleep=args.retry_sleep,
            )

            record["raw_response"] = sdk_response_to_dict(raw_response)
            record["raw_text"] = raw_text

            if raw_text is None:
                record["error"] = "request_failed"
            else:
                parsed = parse_json_object(raw_text)
                if parsed is None:
                    record["error"] = "json_parse_failed"
                else:
                    record["missing_concepts"] = normalize_concepts(
                        parsed,
                        concept_lookup,
                        "missing",
                    )
                    record["present"] = list(record["missing_concepts"])
                    record["dropped_out_of_set"] = [
                        item for item in parsed.get("missing", [])
                        if isinstance(item, str) and item.strip() not in concept_lookup
                    ] if isinstance(parsed.get("missing", []), list) else []

            out_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_handle.flush()
            processed_ids.add(coco_image_id)

    if not args.skip_merged_json:
        write_merged_output(
            annotations=annotations,
            jsonl_path=args.output_jsonl,
            output_path=args.output_json,
        )
        log.info("Wrote merged JSON to %s", args.output_json)


if __name__ == "__main__":
    main()
