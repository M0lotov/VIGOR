from __future__ import annotations

import argparse
import json
import time
import traceback
import urllib.request
from io import BytesIO
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from annotators import (
    Annotator,
    Attention,
    Chefer,
    ClipSeg,
    GroundedSAM,
    Sam3,
    _load_image,
)
from postprocess_annotations import slugify


DEFAULT_SUMMARY_JSON = Path("postprocessed_annotations_summary_full.json")
DEFAULT_MASKS_DIR = Path("postprocessed_annotations_full")
DEFAULT_OUTPUT_DIR = Path("annotated_masks_full")
DEFAULT_ANNOTATIONS_JSON = Path("cleaned_annotations.json")
COLOR_CONCEPTS = {
    "red",
    "blue",
    "green",
    "yellow",
    "black",
    "white",
    "brown",
    "gray",
    "grey",
    "orange",
    "pink",
    "purple",
    "beige",
    "tan",
    "golden",
    "silver",
    "dark",
    "light",
    "bright",
    "colorful",
    "multicolored",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate concept masks for each image in postprocessed_annotations_summary.json. "
            "Concepts with masks are skipped, bbox concepts are masked by the bbox mask PNG, "
            "and concepts without annotation use thresholded model outputs."
        )
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=DEFAULT_SUMMARY_JSON,
        help="Path to postprocessed_annotations_summary.json.",
    )
    parser.add_argument(
        "--masks-dir",
        type=Path,
        default=DEFAULT_MASKS_DIR,
        help="Directory containing per-image concept mask PNGs from postprocess_annotations.py.",
    )
    parser.add_argument(
        "--annotations-json",
        type=Path,
        default=DEFAULT_ANNOTATIONS_JSON,
        help="Annotation JSON used to look up image URLs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where mask and overlay images will be written.",
    )
    parser.add_argument(
        "--tools",
        nargs="+",
        choices=("chefer","sam3", "clipseg", "attention", "grounded_sam"),
        default=("chefer","sam3", "clipseg", "attention", "grounded_sam"),
        help="Annotators to run.",
    )
    parser.add_argument("--limit-images", type=int, default=None, help="Optional maximum number of images.")
    parser.add_argument(
        "--limit-concepts",
        type=int,
        default=None,
        help="Optional maximum number of concepts per bucket per image.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Default device passed to generators unless overridden by a tool-specific option.",
    )
    parser.add_argument("--grounded-sam-device", default='cuda:0', help="Device override for GroundedSAM.")
    parser.add_argument("--chefer-device", default='cuda:1', help="Device override for Chefer.")
    parser.add_argument("--sam3-device", default='cuda:2', help="Device override for Sam3.")
    parser.add_argument("--clipseg-device", default='cuda:3', help="Device override for ClipSeg.")
    parser.add_argument(
        "--overlay-alpha",
        type=float,
        default=1,
        help="Blend alpha for overlay images.",
    )
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=0.5,
        help="Threshold applied to normalized model heatmaps to produce binary masks.",
    )
    parser.add_argument(
        "--log-errors",
        action="store_true",
        help="Print full tracebacks when a tool/concept fails.",
    )
    return parser.parse_args()


def load_summary(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_annotation_image_info(path: Path) -> dict[int, dict]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return {image["coco_image_id"]: image["image_info"] for image in data["images"]}


def load_image_from_url(image_id: int, image_url: str) -> Image.Image:
    with urllib.request.urlopen(image_url) as response:
        image_bytes = response.read()
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image.filename = f"{image_id}{Path(image_url).suffix or '.jpg'}"
    return image


def concept_to_annotator_prompt(concept: str) -> str:
    concept_normalized = concept.strip().lower()
    if concept_normalized in COLOR_CONCEPTS:
        return f"{concept} object"
    return concept


def format_duration(seconds: float) -> str:
    seconds_int = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds_int, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes:d}m {secs:02d}s"
    return f"{secs:d}s"


def load_binary_mask(mask_path: Path, target_shape: tuple[int, int]) -> np.ndarray:
    if not mask_path.exists():
        raise FileNotFoundError(f"Missing bbox mask PNG: {mask_path}")

    mask = Image.open(mask_path).convert("L")
    width = target_shape[1]
    height = target_shape[0]
    if mask.size != (width, height):
        mask = mask.resize((width, height), Image.Resampling.NEAREST)
    return np.asarray(mask, dtype=np.uint8) > 0


def save_mask_png(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    binary = (np.asarray(mask, dtype=np.float32) > 0).astype(np.uint8) * 255
    Image.fromarray(binary, mode="L").save(path)


def save_mask_npy(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, mask.astype(np.float32))


def save_overlay_png(path: Path, overlay: Image.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(path)


def save_original_image(path: Path, image: Image.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def instantiate_generators(args: argparse.Namespace) -> tuple[dict[str, Annotator], list[str]]:
    factories: dict[str, Callable[[], Annotator]] = {
        "chefer": lambda: Chefer(
            device=args.chefer_device or args.device,
            overlay_alpha=args.overlay_alpha,
            mask_threshold=0.6,
        ),
        "sam3": lambda: Sam3(
            device=args.sam3_device or args.device,
            overlay_alpha=args.overlay_alpha,
            mask_threshold=0.3,
            instance_threshold=0.3
        ),
        "clipseg": lambda: ClipSeg(
            device=args.clipseg_device or args.device,
            overlay_alpha=args.overlay_alpha,
            mask_threshold=0.3,
        ),
        "attention": lambda: Attention(
            overlay_alpha=args.overlay_alpha,
            mask_threshold=0.6,
        ),
        "grounded_sam": lambda: GroundedSAM(
            device=args.grounded_sam_device or args.device,
            overlay_alpha=args.overlay_alpha,
        )
    }
    generators: dict[str, Annotator] = {}
    failures: list[str] = []
    for tool_name in args.tools:
        try:
            generators[tool_name] = factories[tool_name]()
        except Exception as exc:  # noqa: BLE001
            failures.append(f"tool={tool_name} initialization failed: {exc}")
            if args.log_errors:
                traceback.print_exc()
    return generators, failures


def concept_output_dir(output_dir: Path, image_id: int, concept: str, tool_name: str) -> Path:
    return output_dir / str(image_id) / slugify(concept) / tool_name


def image_output_dir(output_dir: Path, image_id: int) -> Path:
    return output_dir / str(image_id)


def main() -> None:
    args = parse_args()
    summary = load_summary(args.summary_json)
    annotation_image_info = load_annotation_image_info(args.annotations_json)

    generators, failures = instantiate_generators(args)
    if not generators:
        raise RuntimeError("No annotators were initialized successfully.")

    images = summary["images"]
    if args.limit_images is not None:
        images = images[: args.limit_images]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    total_tasks = 0
    for image_record in images:
        bbox_concepts = image_record.get("concepts_with_bbox", [])
        raw_concepts = image_record.get("concepts_without_annotation", [])
        if args.limit_concepts is not None:
            bbox_concepts = bbox_concepts[: args.limit_concepts]
            raw_concepts = raw_concepts[: args.limit_concepts]
        total_tasks += (len(bbox_concepts) + len(raw_concepts)) * len(generators)

    started_at = time.time()
    completed_tasks = 0

    for image_record in images:
        image_id = image_record["coco_image_id"]
        file_name = image_record["file_name"]
        image_info = annotation_image_info.get(image_id, {})
        image_url = image_info.get("coco_url")
        try:
            if not image_url:
                raise FileNotFoundError(f"No image URL found for image_id={image_id}.")
            source_image = load_image_from_url(image_id, image_url)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"image_id={image_id} file={file_name!r}: {exc}")
            if args.log_errors:
                traceback.print_exc()
            continue

        save_original_image(image_output_dir(args.output_dir, image_id) / "original.png", source_image)

        bbox_concepts = image_record.get("concepts_with_bbox", [])
        raw_concepts = image_record.get("concepts_without_annotation", [])
        if args.limit_concepts is not None:
            bbox_concepts = bbox_concepts[: args.limit_concepts]
            raw_concepts = raw_concepts[: args.limit_concepts]

        concepts_to_process = (
            [("bbox", concept) for concept in bbox_concepts]
            + [("raw", concept) for concept in raw_concepts]
        )

        for concept_kind, concept in concepts_to_process:
            bbox_mask_path = None
            if concept_kind == "bbox":
                bbox_mask_path = args.masks_dir / str(image_id) / f"{slugify(concept)}.png"
            annotator_prompt = concept_to_annotator_prompt(concept)

            for tool_name, generator in generators.items():
                try:
                    pil_image = _load_image(source_image)
                    mask = generator.get_mask(pil_image, annotator_prompt)

                    if concept_kind == "bbox":
                        bbox_mask = load_binary_mask(bbox_mask_path, mask.shape)
                        mask = mask * bbox_mask.astype(np.float32)

                    # overlay = _overlay_mask(
                    #     image=pil_image,
                    #     mask=mask,
                    #     image_alpha=args.overlay_alpha,
                    #     contour_width=3
                    # )

                    target_dir = concept_output_dir(args.output_dir, image_id, concept, tool_name)
                    save_mask_npy(target_dir / "mask.npy", mask)
                    save_mask_png(target_dir / "mask.png", mask)
                    # save_overlay_png(target_dir / "overlay.png", overlay)
                except Exception as exc:  # noqa: BLE001
                    failures.append(
                        f"image_id={image_id} concept={concept!r} kind={concept_kind} tool={tool_name}: {exc}"
                    )
                    if args.log_errors:
                        traceback.print_exc()
                finally:
                    completed_tasks += 1
                    elapsed = time.time() - started_at
                    remaining_tasks = max(total_tasks - completed_tasks, 0)
                    eta_seconds = (elapsed / completed_tasks) * remaining_tasks if completed_tasks else 0.0
                    print(
                        f"[{completed_tasks}/{total_tasks}] image_id={image_id} "
                        f"concept={concept!r} tool={tool_name} "
                        f"elapsed={format_duration(elapsed)} "
                        f"eta={format_duration(eta_seconds)}",
                        flush=True,
                    )

    if failures:
        print("Completed with failures:")
        for failure in failures:
            print(f"- {failure}")
    else:
        print("Completed successfully.")


if __name__ == "__main__":
    main()
