import argparse
import json
import re
import time
from collections import defaultdict
from io import BytesIO
from pathlib import Path

import numpy as np
import requests
import torch
from PIL import Image
from pycocotools import mask as mask_utils
from transformers import Sam3Model, Sam3Processor


DEFAULT_CONCEPTS_JSONL = Path("identified_concepts_openai_full.jsonl")
DEFAULT_ANNOTATIONS_JSON = Path("cleaned_annotations.json")
DEFAULT_OUTPUT_DIR = Path("postprocessed_annotations_full")
DEFAULT_SUMMARY_JSON = Path("postprocessed_annotations_summary_full.json")
DEFAULT_RELATION_CONCEPTS_TXT = Path(
    "../concept_set/output/final_concepts_cleaned/relation.txt"
)

CONCEPT_LEVELS = ("object", "part", "attribute", "relation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Postprocess concept annotations into image-level binary masks for the "
            "concepts identified in identified_concepts_openai.jsonl."
        )
    )
    parser.add_argument(
        "--concepts-jsonl",
        type=Path,
        default=DEFAULT_CONCEPTS_JSONL,
        help="JSONL file containing per-image identified concepts.",
    )
    parser.add_argument(
        "--annotations-json",
        type=Path,
        default=DEFAULT_ANNOTATIONS_JSON,
        help="Merged/cleaned annotation JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where binary mask PNGs will be written.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=DEFAULT_SUMMARY_JSON,
        help="JSON file storing which concepts have mask, bbox, or no annotation.",
    )
    parser.add_argument(
        "--relation-concepts-txt",
        type=Path,
        default=DEFAULT_RELATION_CONCEPTS_TXT,
        help="Text file containing one relation concept per line.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of images to process, for debugging.",
    )
    return parser.parse_args()


def load_identified_concepts(path: Path) -> dict[int, dict]:
    records = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            image_id = record["coco_image_id"]
            records[image_id] = record
    return records


def load_annotations(path: Path) -> dict[int, dict]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return {image["coco_image_id"]: image for image in data["images"]}


def load_relation_concepts(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8") as handle:
        return {line.strip().lower() for line in handle if line.strip()}


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug or "concept"


def concept_annotations_by_name(image_record: dict) -> dict[str, list[dict]]:
    grouped = defaultdict(list)
    for level in CONCEPT_LEVELS:
        for annotation in image_record["concepts"].get(level, []):
            concept = annotation["category"].strip().lower()
            grouped[concept].append(annotation)
    return grouped


def determine_concept_buckets(
    image_record: dict,
    identified_record: dict,
    relation_concepts: set[str],
) -> tuple[list[str], list[str], list[str]]:
    mask_concepts = set()
    bbox_concepts = set()

    for level_entries in image_record["concepts"].values():
        for entry in level_entries:
            concept = entry["category"].strip().lower()
            has_mask = "segmentation" in entry and entry["segmentation"] not in [None, [], {}]
            has_bbox = "bbox" in entry and entry["bbox"] not in [None, [], {}]
            if entry.get("concept_level") == "relation":
                if relation_bboxes(entry):
                    has_mask = True
                has_bbox = has_bbox or bool(relation_bboxes(entry))
            if has_mask:
                mask_concepts.add(concept)
            elif has_bbox:
                bbox_concepts.add(concept)

    llm_concepts = [concept.strip().lower() for concept in identified_record.get("present", [])]
    concepts_with_mask = sorted(mask_concepts)
    concepts_with_bbox = sorted(concept for concept in bbox_concepts if concept not in mask_concepts)
    concepts_without_annotation = sorted(
        concept
        for concept in llm_concepts
        if concept not in relation_concepts and concept not in mask_concepts and concept not in bbox_concepts
    )
    return concepts_with_mask, concepts_with_bbox, concepts_without_annotation


def decode_segmentation(segmentation, height: int, width: int) -> np.ndarray:
    if not segmentation:
        return np.zeros((height, width), dtype=bool)

    if isinstance(segmentation, list):
        rles = mask_utils.frPyObjects(segmentation, height, width)
        rle = mask_utils.merge(rles)
    elif isinstance(segmentation, dict) and isinstance(segmentation.get("counts"), list):
        rle = mask_utils.frPyObjects(segmentation, height, width)
    else:
        rle = segmentation

    decoded = mask_utils.decode(rle)
    if decoded.ndim == 3:
        decoded = np.any(decoded, axis=2)
    return decoded.astype(bool)


def apply_bbox(mask: np.ndarray, bbox: list[float] | None) -> None:
    if bbox is None or len(bbox) != 4:
        return

    height, width = mask.shape
    x, y, w, h = bbox
    x0 = max(0, int(np.floor(x)))
    y0 = max(0, int(np.floor(y)))
    x1 = min(width, int(np.ceil(x + w)))
    y1 = min(height, int(np.ceil(y + h)))
    if x1 > x0 and y1 > y0:
        mask[y0:y1, x0:x1] = True


def relation_bboxes(annotation: dict) -> list[list[float]]:
    boxes = []
    subject_bbox = annotation.get("subject", {}).get("bbox")
    object_bbox = annotation.get("object", {}).get("bbox")
    if subject_bbox is not None:
        boxes.append(subject_bbox)
    if object_bbox is not None:
        boxes.append(object_bbox)
    return boxes


def bbox_xywh_to_xyxy(bbox: list[float] | None) -> list[float] | None:
    if bbox is None or len(bbox) != 4:
        return None
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


def resolve_image_url(identified_record: dict, annotation_record: dict | None) -> str | None:
    annotation_image_info = annotation_record.get("image_info", {}) if annotation_record else {}
    return annotation_image_info.get("coco_url") or identified_record.get("image_url")


class Sam3RelationMaskGenerator:
    def __init__(
        self,
        model_id: str = "facebook/sam3",
        device: str | None = None,
        instance_threshold: float = 0.5,
        mask_threshold: float = 0.5,
    ) -> None:
        self.device = device or ("cuda:1" if torch.cuda.is_available() else "cpu")
        self.instance_threshold = instance_threshold
        self.mask_threshold = mask_threshold
        self.processor = Sam3Processor.from_pretrained(model_id)
        self.model = Sam3Model.from_pretrained(model_id).to(self.device)
        self.model.eval()

    def get_mask(self, image_url: str, boxes: list[list[float]]) -> np.ndarray | None:
        if not boxes:
            return None

        response = requests.get(image_url, timeout=30)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
        xyxy_boxes = [bbox_xywh_to_xyxy(bbox) for bbox in boxes]
        xyxy_boxes = [bbox for bbox in xyxy_boxes if bbox is not None]
        if not xyxy_boxes:
            return None

        inputs = self.processor(
            images=image,
            input_boxes=[xyxy_boxes],
            return_tensors="pt",
        ).to(self.device)

        with torch.inference_mode():
            outputs = self.model(**inputs)

        results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=self.instance_threshold,
            mask_threshold=self.mask_threshold,
            target_sizes=[(image.height, image.width)],
        )[0]
        masks = results.get("masks")
        if masks is None or len(masks) == 0:
            return None

        combined_mask = masks.to(torch.bool).any(dim=0).cpu().numpy()
        return combined_mask if combined_mask.any() else None


def build_mask_for_concept(
    annotations: list[dict],
    height: int,
    width: int,
    image_url: str | None = None,
    relation_mask_generator: Sam3RelationMaskGenerator | None = None,
) -> tuple[np.ndarray | None, str]:
    is_relation = any(annotation.get("concept_level") == "relation" for annotation in annotations)
    if is_relation:
        boxes = []
        for annotation in annotations:
            boxes.extend(relation_bboxes(annotation))
        if relation_mask_generator is None or image_url is None:
            return None, "none"
        mask = relation_mask_generator.get_mask(image_url, boxes)
        if mask is not None:
            return mask, "mask"
        return None, "none"

    has_segmentation = any(annotation.get("segmentation") is not None for annotation in annotations)
    if has_segmentation:
        mask = np.zeros((height, width), dtype=bool)
        for annotation in annotations:
            segmentation = annotation.get("segmentation")
            if segmentation is None:
                continue
            mask |= decode_segmentation(segmentation, height, width)
        return mask, "mask" if mask.any() else "none"

    has_bbox = False
    mask = np.zeros((height, width), dtype=bool)
    for annotation in annotations:
        if annotation.get("concept_level") == "relation":
            boxes = relation_bboxes(annotation)
        else:
            bbox = annotation.get("bbox")
            boxes = [bbox] if bbox is not None else []

        for bbox in boxes:
            apply_bbox(mask, bbox)
            has_bbox = True

    if has_bbox and mask.any():
        return mask, "bbox"
    return None, "none"


def save_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    image.save(path)


def format_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def main() -> None:
    args = parse_args()

    identified_records = load_identified_concepts(args.concepts_jsonl)
    annotation_records = load_annotations(args.annotations_json)
    relation_concepts = load_relation_concepts(args.relation_concepts_txt)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    relation_mask_generator = None

    summary = {
        "concepts_jsonl": str(args.concepts_jsonl),
        "annotations_json": str(args.annotations_json),
        "output_dir": str(args.output_dir),
        "images": [],
    }

    image_ids = sorted(identified_records)
    if args.limit is not None:
        image_ids = image_ids[: args.limit]
    total_images = len(image_ids)
    start_time = time.time()

    for index, image_id in enumerate(image_ids, start=1):
        identified = identified_records[image_id]
        annotation_record = annotation_records.get(image_id)

        image_summary = {
            "coco_image_id": image_id,
            "file_name": identified.get("file_name"),
            "concepts_with_mask": [],
            "concepts_with_bbox": [],
            "concepts_without_annotation": [],
        }

        if annotation_record is None:
            image_summary["concepts_without_annotation"] = sorted(
                concept.strip().lower()
                for concept in identified.get("present", [])
                if concept.strip().lower() not in relation_concepts
            )
            summary["images"].append(image_summary)
            continue

        image_info = annotation_record["image_info"]
        height = image_info["height"]
        width = image_info["width"]
        image_url = resolve_image_url(identified, annotation_record)
        annotations_by_concept = concept_annotations_by_name(annotation_record)
        concepts_with_mask, concepts_with_bbox, concepts_without_annotation = determine_concept_buckets(
            annotation_record,
            identified,
            relation_concepts,
        )

        image_summary["concepts_without_annotation"] = concepts_without_annotation

        for concept in concepts_with_mask + concepts_with_bbox:
            annotations = annotations_by_concept.get(concept, [])
            if not annotations:
                continue

            if any(annotation.get("concept_level") == "relation" for annotation in annotations):
                if relation_mask_generator is None:
                    relation_mask_generator = Sam3RelationMaskGenerator()
            mask, annotation_type = build_mask_for_concept(
                annotations,
                height,
                width,
                image_url=image_url,
                relation_mask_generator=relation_mask_generator,
            )
            if annotation_type == "none" or mask is None:
                continue

            mask_name = f"{slugify(concept)}.png"
            mask_path = args.output_dir / str(image_id) / mask_name
            save_mask(mask, mask_path)

            if annotation_type == "mask":
                image_summary["concepts_with_mask"].append(concept)
            elif annotation_type == "bbox":
                image_summary["concepts_with_bbox"].append(concept)

        image_summary["concepts_with_mask"].sort()
        image_summary["concepts_with_bbox"].sort()

        summary["images"].append(image_summary)
        elapsed = time.time() - start_time
        average_per_image = elapsed / index
        remaining_images = total_images - index
        eta_seconds = average_per_image * remaining_images
        print(
            f"[{index}/{total_images}] image {image_id} processed | "
            f"elapsed {format_duration(elapsed)} | "
            f"eta {format_duration(eta_seconds)}"
        )

    with args.summary_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Processed {len(summary['images'])} images")
    print(f"Wrote masks to {args.output_dir}")
    print(f"Wrote summary to {args.summary_json}")


if __name__ == "__main__":
    main()
