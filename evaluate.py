from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import json
import math
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
from pycocotools import mask as coco_mask
from transformers import AutoModel, AutoModelForImageTextToText, AutoProcessor
from transformers.image_utils import load_image


CONCEPT_TYPES = ("primitive", "intermediate", "object", "scene")

DEFAULT_MODEL = "Qwen/Qwen2-VL-2B-Instruct"
IMAGE_TOKEN_STRINGS = (
    "<|image_pad|>",
    "<image>",
    "<start_of_image>",
    "<IMG_CONTEXT>",
    "[IMG]",
    "<image_soft_token>",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model ID to evaluate.")
    parser.add_argument(
        "--annotation-path",
        type=Path,
        default=Path("train.json"),
        help="COCO-style train.json with captions, coco_url values, categories, and segmentations.",
    )
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu", help="Torch device.")
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"), help="Model dtype.")
    parser.add_argument("--attn-implementation", default="eager", help="Attention implementation. Use eager for attentions.")
    parser.add_argument("--max-images", type=int, default=None, help="Optional cap for smoke tests.")
    parser.add_argument(
        "--resize-images",
        choices=("auto", "never", "always"),
        default="auto",
        help="Resize images before processing. Auto resizes InternVL inputs only.",
    )
    parser.add_argument("--image-size", type=int, default=None, help="Square image size used when resizing is enabled.")
    parser.add_argument("--allow-cpu", action="store_true", help="Allow CPU execution despite very high runtime/memory cost.")
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def model_family(model_id: str) -> str:
    normalized = model_id.lower()
    if "qwen" in normalized:
        return "qwen"
    if "internvl" in normalized:
        return "internvl"
    if "gemma" in normalized:
        return "gemma"
    if "ministral" in normalized:
        return "mistral"
    return "generic"


def load_processor_and_model(model_id: str, args: argparse.Namespace) -> tuple[AutoProcessor, torch.nn.Module]:
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    base_kwargs = {
        "attn_implementation": args.attn_implementation,
        "device_map": "auto",
        "trust_remote_code": True,
    }
    dtype = dtype_from_name(args.dtype)
    for model_cls in (AutoModelForImageTextToText, AutoModel):
        for dtype_kwarg in ("dtype", "torch_dtype"):
            try:
                model = model_cls.from_pretrained(model_id, **base_kwargs, **{dtype_kwarg: dtype})
                model.eval()
                return processor, model
            except Exception:
                continue
    model = AutoModel.from_pretrained(model_id, **base_kwargs)
    model.eval()
    return processor, model


def maybe_resize_image(image, family: str, args: argparse.Namespace):
    should_resize = args.resize_images == "always" or (args.resize_images == "auto" and family == "internvl")
    if not should_resize:
        return image

    image_size = args.image_size or (448 if family == "internvl" else 512)
    if image.size == (image_size, image_size):
        return image
    return image.resize((image_size, image_size))


def apply_chat_template(processor: AutoProcessor, messages: list[dict]) -> str:
    if hasattr(processor, "apply_chat_template"):
        return processor.apply_chat_template(messages, add_generation_prompt=True)
    return "<image>\nDescribe the image."


def config_image_token_id(model: torch.nn.Module) -> int:
    image_token_id = getattr(model.config, "image_token_id", None)
    return image_token_id


def model_device(model: torch.nn.Module) -> torch.device:
    device = getattr(model, "device", None)
    if device is not None:
        return torch.device(device)
    return next(model.parameters()).device


def normalize_heatmap(heatmap: torch.Tensor) -> torch.Tensor:
    heatmap = heatmap.float()
    heatmap = heatmap - heatmap.min()
    return heatmap / heatmap.max().clamp_min(1e-8)


def compute_auprc(score_map: torch.Tensor, gt_mask: torch.Tensor) -> float:
    gt_flat = gt_mask.bool().flatten()
    total_positive = gt_flat.sum().item()
    if total_positive == 0:
        return float("nan")

    score_flat = score_map.flatten()
    sorted_indices = torch.argsort(score_flat, descending=True)
    sorted_gt = gt_flat[sorted_indices].float()
    true_positives = torch.cumsum(sorted_gt, dim=0)
    ranks = torch.arange(1, sorted_gt.numel() + 1, device=score_map.device, dtype=torch.float32)
    precision_at_rank = true_positives / ranks
    return (precision_at_rank * sorted_gt).sum().item() / total_positive


def compute_pointing_accuracy(attn_map: torch.Tensor, gt_mask: torch.Tensor) -> float:
    if gt_mask.sum().item() == 0 or attn_map.numel() == 0:
        return float("nan")
    point_idx = int(attn_map.argmax().item())
    return float(gt_mask.flatten()[point_idx].item())


def find_token_spans(token_texts: list[str]) -> list[tuple[int, int]]:
    spans = []
    cursor = 0
    for token_text in token_texts:
        start = cursor
        cursor += len(token_text)
        spans.append((start, cursor))
    return spans


def load_annotations(annotation_path: Path) -> list[dict]:
    with annotation_path.open() as f:
        data = json.load(f)

    categories = {category["id"]: category for category in data["categories"]}
    annotations_by_image_id: dict[int, list[dict]] = {}
    for annotation in data["annotations"]:
        annotations_by_image_id.setdefault(annotation["image_id"], []).append(annotation)

    records = []
    for image in data["images"]:
        annotated_concepts: dict[str, list[dict]] = {}
        for annotation in annotations_by_image_id.get(image["id"], []):
            category = categories[annotation["category_id"]]
            concept_type = category.get("supercategory", "")
            annotated_concepts.setdefault(concept_type, []).append(
                {
                    "concept": category["name"],
                    "segmentation": annotation.get("segmentation"),
                    "bbox": annotation.get("bbox"),
                }
            )

        records.append(
            {
                "coco_image_id": image["id"],
                "coco_url": image["coco_url"],
                "caption": image.get("caption", ""),
                "width": image["width"],
                "height": image["height"],
                "annotated_concepts": annotated_concepts,
            }
        )
    return records


def flatten_concepts(annotation: dict) -> list[str]:
    concepts = []
    for concept_group in annotation["annotated_concepts"].values():
        for item in concept_group:
            concepts.append(item["concept"])
    return list(dict.fromkeys(concepts))


def build_concept_metadata(annotation: dict) -> dict[str, dict]:
    concept_metadata = {}
    for concept_type, concept_group in annotation["annotated_concepts"].items():
        for item in concept_group:
            concept_entry = concept_metadata.setdefault(
                item["concept"],
                {
                    "concept_type": concept_type,
                    "instances": [],
                },
            )
            concept_entry["instances"].append(item)
    return concept_metadata


def decode_concept_mask(
    instances: list[dict],
    height: int,
    width: int,
    device: torch.device,
) -> torch.Tensor:
    rles = []
    fallback_mask = np.zeros((height, width), dtype=bool)

    for instance in instances:
        segmentation = instance.get("segmentation")
        if isinstance(segmentation, list):
            rles.extend(coco_mask.frPyObjects(segmentation, height, width))
        elif isinstance(segmentation, dict):
            if isinstance(segmentation.get("counts"), list):
                encoded = coco_mask.frPyObjects(segmentation, height, width)
                if isinstance(encoded, list):
                    rles.extend(encoded)
                else:
                    rles.append(encoded)
            else:
                rles.append(segmentation)
        else:
            bbox = instance.get("bbox")
            if bbox:
                x, y, box_w, box_h = bbox
                x0 = max(0, int(x))
                y0 = max(0, int(y))
                x1 = min(width, int(math.ceil(x + box_w)))
                y1 = min(height, int(math.ceil(y + box_h)))
                fallback_mask[y0:y1, x0:x1] = True

    if rles:
        decoded = coco_mask.decode(rles)
        if decoded.ndim == 3:
            decoded = np.any(decoded, axis=2)
        gt_mask = decoded.astype(bool)
    else:
        gt_mask = fallback_mask

    return torch.from_numpy(np.ascontiguousarray(gt_mask)).to(device)


def image_token_mask(
    inputs: dict,
    processor: AutoProcessor,
    model: torch.nn.Module,
    prompt_len: int,
    device: torch.device,
) -> torch.Tensor:
    if "mm_token_type_ids" in inputs:
        return (inputs["mm_token_type_ids"][0] == 1).to(device)

    image_token_id = config_image_token_id(model)
    input_ids = inputs["input_ids"][0, :prompt_len].to(device)
    mask = input_ids == image_token_id
    return mask


def factor_pair_closest_to_aspect(num_tokens: int, aspect_ratio: float) -> tuple[int, int]:
    if num_tokens <= 0:
        raise ValueError("Image token count must be positive.")

    best_h = 1
    best_w = num_tokens
    best_error = float("inf")
    for h in range(1, int(math.sqrt(num_tokens)) + 1):
        if num_tokens % h != 0:
            continue
        w = num_tokens // h
        for candidate_h, candidate_w in ((h, w), (w, h)):
            error = abs((candidate_w / candidate_h) - aspect_ratio)
            if error < best_error:
                best_h = candidate_h
                best_w = candidate_w
                best_error = error
    return best_h, best_w


def image_aspect_ratio(image, inputs: dict) -> float:
    pixel_values = inputs.get("pixel_values")
    if isinstance(pixel_values, torch.Tensor) and pixel_values.ndim >= 4:
        height = int(pixel_values.shape[-2])
        width = int(pixel_values.shape[-1])
        if height > 0 and width > 0:
            return width / height
    width, height = image.size
    return width / height


def attention_grid_size(
    inputs: dict,
    processor: AutoProcessor,
    image,
    image_token_count: int,
    family: str,
) -> tuple[int, int]:
    if family == "qwen" and "image_grid_thw" in inputs:
        merge_size = 2
        image_processor = getattr(processor, "image_processor", None)
        if image_processor is not None:
            merge_size = getattr(image_processor, "merge_size", merge_size)
        grid_t, grid_h, grid_w = map(int, inputs["image_grid_thw"][0].tolist())
        return max(1, grid_t * grid_h // merge_size), max(1, grid_w // merge_size)

    square_size = int(math.sqrt(image_token_count))
    if square_size * square_size == image_token_count:
        return square_size, square_size
    return factor_pair_closest_to_aspect(image_token_count, image_aspect_ratio(image, inputs))


def collect_generated_image_attentions(
    coco_url: str,
    caption: str,
    processor: AutoProcessor,
    model: torch.nn.Module,
    family: str,
    args: argparse.Namespace,
) -> dict:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Describe the image."},
            ],
        },
    ]

    image = maybe_resize_image(load_image(coco_url).convert("RGB"), family, args)
    text_prompt = apply_chat_template(processor, messages)
    device = model_device(model)
    inputs = processor(text=[text_prompt], images=[image], padding=True, return_tensors="pt").to(device)
    forward_inputs = processor(text=[text_prompt + caption], images=[image], padding=True, return_tensors="pt").to(
        device
    )

    prompt_len = inputs["input_ids"].shape[1]
    generated_ids = forward_inputs["input_ids"][:, prompt_len:]
    if generated_ids.shape[1] == 0:
        raise ValueError("The model did not generate any new tokens.")

    prompt_image_token_mask = image_token_mask(inputs, processor, model, prompt_len, device)
    image_token_count = int(prompt_image_token_mask.sum().item())
    grid_h, grid_w = attention_grid_size(inputs, processor, image, image_token_count, family)
    if grid_h * grid_w != image_token_count:
        raise ValueError(
            f"Image token count {image_token_count} does not match inferred attention grid {grid_h}x{grid_w}."
        )
    tokenizer = getattr(processor, "tokenizer", processor)
    generated_token_texts = [
        tokenizer.decode([token_id], skip_special_tokens=False) for token_id in generated_ids[0].tolist()
    ]
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=False)

    with torch.inference_mode():
        forward_output = model(
            **forward_inputs,
            output_attentions=True,
            return_dict=True,
            use_cache=False,
        )

    layer_image_attentions = []
    attentions = getattr(forward_output, "attentions", None)
    if not attentions:
        raise ValueError("The model did not return attentions.")
    for layer_attention in attentions:
        if layer_attention is None:
            continue
        generated_attention = layer_attention[0, :, prompt_len:, :]
        prompt_attention = generated_attention[:, :, :prompt_len]
        image_attention = prompt_attention[:, :, prompt_image_token_mask]
        layer_image_attentions.append(image_attention)
    if not layer_image_attentions:
        raise ValueError("The model returned no language-layer attentions.")
    generated_image_attentions = torch.stack(layer_image_attentions).permute(2, 0, 1, 3)
    _, topk_indices = generated_image_attentions.mean((0, 1, 2)).topk(min(3, image_token_count))
    for idx in topk_indices:
        generated_image_attentions[:, :, :, idx] = 0

    return {
        "generated_text": generated_text,
        "generated_token_texts": generated_token_texts,
        "generated_image_attentions": generated_image_attentions,
        "grid_h": grid_h,
        "grid_w": grid_w,
    }


def match_concepts_to_heatmaps(
    concepts: list[str],
    generated_text: str,
    generated_token_texts: list[str],
    generated_image_attentions: torch.Tensor,
    grid_h: int,
    grid_w: int,
) -> list[dict]:
    token_spans = find_token_spans(generated_token_texts)
    generated_text_lower = generated_text.lower()
    matched_concepts = {}

    for concept in concepts:
        concept_lower = concept.lower()
        search_start = 0
        while True:
            char_start = generated_text_lower.find(concept_lower, search_start)
            if char_start == -1:
                break

            char_end = char_start + len(concept_lower)
            matched_token_indices = [
                token_idx
                for token_idx, (token_start, token_end) in enumerate(token_spans)
                if token_start < char_end and token_end > char_start
            ]

            if matched_token_indices:
                concept_attention = generated_image_attentions[matched_token_indices].mean(dim=(0, 1, 2))
                concept_entry = matched_concepts.setdefault(concept, {"concept": concept, "attentions": []})
                concept_entry["attentions"].append(concept_attention)

            search_start = char_end

    matched = list(matched_concepts.values())
    for item in matched:
        item["attention_map"] = torch.stack(item["attentions"]).mean(dim=0).reshape(grid_h, grid_w).float()
        item["heatmap"] = normalize_heatmap(item["attention_map"])
    return matched


def mean(values: list[float]) -> float:
    clean = [value for value in values if value == value]
    return sum(clean) / len(clean) if clean else float("nan")


def new_metric_bucket() -> dict[str, list[float]]:
    return {"auprcs": [], "pointing_accuracies": []}


def add_metric(
    bucket: dict[str, list[float]],
    auprc: float,
    pointing_accuracy: float,
) -> None:
    if auprc == auprc:
        bucket["auprcs"].append(auprc)
    if pointing_accuracy == pointing_accuracy:
        bucket["pointing_accuracies"].append(pointing_accuracy)


def metric_bucket_to_image_result(image_id: int, bucket: dict[str, list[float]]) -> dict | None:
    if not any(bucket.values()):
        return None
    return {
        "image_id": image_id,
        "mean_auprc": mean(bucket["auprcs"]),
        "pointing_accuracy": mean(bucket["pointing_accuracies"]),
        "num_concepts": max(len(bucket["auprcs"]), len(bucket["pointing_accuracies"])),
    }


def metrics_summary(
    image_results: list[dict],
) -> dict:
    return {
        "mean_auprc": mean([item["mean_auprc"] for item in image_results]),
        "pointing_accuracy": mean([item["pointing_accuracy"] for item in image_results]),
    }


def concept_type_metrics(
    concept_type_results: dict[str, dict[str, list[float]]],
) -> dict:
    metrics = {}
    for concept_type in CONCEPT_TYPES:
        auprcs = concept_type_results.get(concept_type, {}).get("auprcs", [])
        pointing_accuracies = concept_type_results.get(concept_type, {}).get("pointing_accuracies", [])
        metrics[concept_type] = {
            "mean_auprc": mean(auprcs),
            "pointing_accuracy": mean(pointing_accuracies),
        }
    return metrics


def format_metric(value: float) -> str:
    if value != value:
        return "nan"
    return f"{value:.3f}"


def print_metrics_table(metrics: dict) -> None:
    rows = [("overall", metrics["overall"])]
    rows.extend((concept_type, metrics["concept_type"][concept_type]) for concept_type in CONCEPT_TYPES)

    headers = ("level", "mean_auprc", "pointing_accuracy")
    table = [
        (
            level,
            format_metric(level_metrics["mean_auprc"]),
            format_metric(level_metrics["pointing_accuracy"]),
        )
        for level, level_metrics in rows
    ]
    widths = [
        max(len(headers[column_idx]), *(len(row[column_idx]) for row in table))
        for column_idx in range(len(headers))
    ]

    print(" | ".join(header.ljust(widths[column_idx]) for column_idx, header in enumerate(headers)))
    print("-+-".join("-" * width for width in widths))
    for row in table:
        print(" | ".join(value.ljust(widths[column_idx]) for column_idx, value in enumerate(row)))


def evaluate_model(
    model_id: str,
    annotations: list[dict],
    args: argparse.Namespace,
) -> dict:
    family = model_family(model_id)
    processor, model = load_processor_and_model(model_id, args)
    device = model_device(model)

    image_results = []
    concept_type_results = {concept_type: new_metric_bucket() for concept_type in CONCEPT_TYPES}
    processed = 0

    for annotation in tqdm(annotations):
        if args.max_images is not None and processed >= args.max_images:
            break

        image_id = annotation["coco_image_id"]
        coco_url = annotation.get("coco_url")
        caption = annotation.get("caption")
        if not coco_url:
            continue
        if not caption:
            continue

        concepts = flatten_concepts(annotation)
        concept_metadata = build_concept_metadata(annotation)

        try:
            attention_result = collect_generated_image_attentions(coco_url, caption, processor, model, family, args)
        except Exception:
            continue

        matched_concepts = match_concepts_to_heatmaps(
            concepts,
            attention_result["generated_text"],
            attention_result["generated_token_texts"],
            attention_result["generated_image_attentions"],
            attention_result["grid_h"],
            attention_result["grid_w"],
        )

        image_metric_bucket = new_metric_bucket()
        for item in matched_concepts:
            concept_info = concept_metadata.get(item["concept"])
            if concept_info is None:
                continue

            gt_mask = decode_concept_mask(
                concept_info["instances"],
                annotation["height"],
                annotation["width"],
                device,
            )
            resized_heatmap = torch.nn.functional.interpolate(
                item["heatmap"].unsqueeze(0).unsqueeze(0).float(),
                size=gt_mask.shape,
                mode="bilinear",
                align_corners=False,
            )[0, 0]
            resized_attention_map = torch.nn.functional.interpolate(
                item["attention_map"].unsqueeze(0).unsqueeze(0).float(),
                size=gt_mask.shape,
                mode="bilinear",
                align_corners=False,
            )[0, 0].clamp_min(0)
            auprc = compute_auprc(resized_heatmap, gt_mask)
            pointing_accuracy = compute_pointing_accuracy(resized_attention_map, gt_mask)
            concept_type = concept_info["concept_type"]
            concept_type_entry = concept_type_results.setdefault(concept_type, new_metric_bucket())
            add_metric(image_metric_bucket, auprc, pointing_accuracy)
            add_metric(concept_type_entry, auprc, pointing_accuracy)

        image_result = metric_bucket_to_image_result(image_id, image_metric_bucket)
        if image_result is not None:
            image_results.append(image_result)
        processed += 1

    return {
        "overall": metrics_summary(image_results),
        "concept_type": concept_type_metrics(concept_type_results),
    }


def main() -> None:
    args = parse_args()
    if str(args.device).startswith("cpu") and not args.allow_cpu:
        raise SystemExit("CUDA is not available; rerun on a GPU node or pass --allow-cpu explicitly.")

    annotations = load_annotations(args.annotation_path)
    metrics = evaluate_model(args.model, annotations, args)
    print_metrics_table(metrics)


if __name__ == "__main__":
    main()
