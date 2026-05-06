#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import shutil
from collections import defaultdict
from pathlib import Path

from postprocess_annotations import slugify


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("finalize_annotations")

DEFAULT_SUMMARY_JSON = Path("postprocessed_annotations_summary_full.json")
DEFAULT_VERIFIED_JSONL = Path("verified_annotations_voting_full.jsonl")
DEFAULT_ANNOTATED_MASKS_DIR = Path("annotated_masks_full")
DEFAULT_OUTPUT_DIR = Path("final_annotations_full")
DEFAULT_CONCEPTS_JSONL = Path("identified_concepts_openai_full.jsonl")
DEFAULT_CONCEPT_SET_DIR = Path("../concept_set/output/final_concepts_cleaned")
DEFAULT_FINAL_ANNOTATION_JSONL = Path("final_annotation_full.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Construct final concept annotations by combining postprocessed masks "
            "with verified tool selections."
        )
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=DEFAULT_SUMMARY_JSON,
        help="Path to postprocessed_annotations_summary.json.",
    )
    parser.add_argument(
        "--verified-jsonl",
        type=Path,
        default=DEFAULT_VERIFIED_JSONL,
        help="Path to verified_annotations_voting.jsonl.",
    )
    parser.add_argument(
        "--annotated-masks-dir",
        type=Path,
        default=DEFAULT_ANNOTATED_MASKS_DIR,
        help="Directory containing per-tool candidate masks.",
    )
    parser.add_argument(
        "--concepts-jsonl",
        type=Path,
        default=DEFAULT_CONCEPTS_JSONL,
        help="JSONL file used to recover image metadata such as image_url.",
    )
    parser.add_argument(
        "--postprocessed-dir",
        type=Path,
        default=None,
        help="Directory containing masks from postprocess_annotations.py. Defaults to summary['output_dir'].",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Destination directory for final annotation masks.",
    )
    parser.add_argument(
        "--concept-set-dir",
        type=Path,
        default=DEFAULT_CONCEPT_SET_DIR,
        help="Directory containing cleaned concept-set category text files.",
    )
    parser.add_argument(
        "--final-annotation-jsonl",
        type=Path,
        default=DEFAULT_FINAL_ANNOTATION_JSONL,
        help="Path to write the finalized per-image annotation manifest JSONL.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite destination files if they already exist.",
    )
    return parser.parse_args()


def load_summary(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_identified_concepts(path: Path) -> dict[int, dict]:
    rows: dict[int, dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            image_id = record.get("coco_image_id")
            if isinstance(image_id, int):
                rows[image_id] = record
    return rows


def load_verifications(path: Path) -> dict[tuple[int, str], dict]:
    rows: dict[tuple[int, str], dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            image_id = record.get("coco_image_id")
            concept_slug = record.get("concept_slug")
            if isinstance(image_id, int) and isinstance(concept_slug, str):
                rows[(image_id, concept_slug)] = record
    return rows


def load_concept_categories(path: Path) -> dict[str, str]:
    category_by_concept: dict[str, str] = {}
    for txt_path in sorted(path.glob("*.txt")):
        if txt_path.name == "all_concepts.txt":
            continue
        category = txt_path.stem
        with txt_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                concept = line.strip().lower()
                if concept:
                    category_by_concept[concept] = category
    return category_by_concept


def copy_file(src: Path, dst: Path, overwrite: bool) -> bool:
    if not src.exists():
        return False
    if dst.exists() and not overwrite:
        return True
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def concept_output_path(output_dir: Path, image_id: int, concept: str) -> Path:
    return output_dir / str(image_id) / f"{slugify(concept)}.png"


def postprocessed_mask_path(postprocessed_dir: Path, image_id: int, concept: str) -> Path:
    return postprocessed_dir / str(image_id) / f"{slugify(concept)}.png"


def verified_mask_path(
    annotated_masks_dir: Path,
    image_id: int,
    concept: str,
    selected_tool: str,
) -> Path:
    return annotated_masks_dir / str(image_id) / slugify(concept) / selected_tool / "mask.png"


def main() -> None:
    args = parse_args()

    summary = load_summary(args.summary_json)
    identified_concepts = load_identified_concepts(args.concepts_jsonl)
    verifications = load_verifications(args.verified_jsonl)
    concept_categories = load_concept_categories(args.concept_set_dir)
    postprocessed_dir = args.postprocessed_dir or Path(summary["output_dir"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.final_annotation_jsonl.parent.mkdir(parents=True, exist_ok=True)

    copied_from_postprocessed = 0
    copied_from_verified = 0
    skipped_missing_postprocessed = 0
    skipped_unverified = 0
    skipped_rejected = 0
    skipped_missing_verified_mask = 0
    skipped_missing_category = 0
    final_annotation_records: list[dict] = []

    for image in summary.get("images", []):
        image_id = image["coco_image_id"]
        image_url = identified_concepts.get(image_id, {}).get("image_url")
        categorized_annotations: dict[str, list[dict[str, str]]] = defaultdict(list)

        for concept in image.get("concepts_with_mask", []):
            src = postprocessed_mask_path(postprocessed_dir, image_id, concept)
            dst = concept_output_path(args.output_dir, image_id, concept)
            if copy_file(src, dst, args.overwrite):
                copied_from_postprocessed += 1
                category = concept_categories.get(concept.lower())
                if category is None:
                    skipped_missing_category += 1
                    log.warning("No category found for %s / %s", image_id, concept)
                else:
                    categorized_annotations[category].append(
                        {"concept": concept, "mask_path": str(dst)}
                    )
            else:
                skipped_missing_postprocessed += 1
                log.warning("Missing postprocessed mask for %s / %s: %s", image_id, concept, src)

        remaining_concepts = list(image.get("concepts_with_bbox", [])) + list(
            image.get("concepts_without_annotation", [])
        )
        for concept in remaining_concepts:
            concept_slug = slugify(concept)
            verification = verifications.get((image_id, concept_slug))
            if verification is None:
                skipped_unverified += 1
                log.warning("No verification found for %s / %s", image_id, concept)
                continue

            selected_tool = verification.get("selected_tool")
            if not isinstance(selected_tool, str) or not selected_tool:
                skipped_rejected += 1
                continue

            src = verified_mask_path(args.annotated_masks_dir, image_id, concept, selected_tool)
            dst = concept_output_path(args.output_dir, image_id, concept)
            if copy_file(src, dst, args.overwrite):
                copied_from_verified += 1
                category = concept_categories.get(concept.lower())
                if category is None:
                    skipped_missing_category += 1
                    log.warning("No category found for %s / %s", image_id, concept)
                else:
                    categorized_annotations[category].append(
                        {"concept": concept, "mask_path": str(dst)}
                    )
            else:
                skipped_missing_verified_mask += 1
                log.warning(
                    "Missing verified mask for %s / %s via %s: %s",
                    image_id,
                    concept,
                    selected_tool,
                    src,
                )

        final_annotation_records.append(
            {
                "coco_image_id": image_id,
                "image_url": image_url,
                "annotated_concepts": {
                    category: sorted(entries, key=lambda entry: entry["concept"])
                    for category, entries in sorted(categorized_annotations.items())
                },
            }
        )

    with args.final_annotation_jsonl.open("w", encoding="utf-8") as handle:
        for record in final_annotation_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    log.info("Copied %s masks directly from %s", copied_from_postprocessed, postprocessed_dir)
    log.info("Copied %s masks from verified tool selections", copied_from_verified)
    log.info("Skipped %s concepts with missing postprocessed masks", skipped_missing_postprocessed)
    log.info("Skipped %s concepts with no verification record", skipped_unverified)
    log.info("Skipped %s concepts rejected by voting", skipped_rejected)
    log.info("Skipped %s concepts with missing verified mask files", skipped_missing_verified_mask)
    log.info("Skipped %s finalized concepts with no concept-set category", skipped_missing_category)
    log.info("Final annotations written to %s", args.output_dir)
    log.info("Final annotation manifest written to %s", args.final_annotation_jsonl)


if __name__ == "__main__":
    main()
