#!/usr/bin/env python3
"""
Generate image captions from ``final_annotation.jsonl`` using the OpenAI Batch API.

Each batch request sends the image plus its annotated concepts to a vision-capable
model and asks for a concise caption grounded in both the visible image content
and the provided concept annotations.
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

from identify_concepts import parse_json_object, sdk_response_to_dict


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("generate_captions")

DEFAULT_ANNOTATIONS = Path("final_annotation_full.jsonl")
DEFAULT_OUTPUT_JSONL = Path("generated_captions_full.jsonl")
DEFAULT_MODEL = "gpt-5.4"
DEFAULT_BATCH_DIR = Path("captions_openai_batch")

SYSTEM_PROMPT = """\
You are a vision-language captioning assistant for an image annotation dataset.
Your job is to write captions that stay faithful to the actual visible image content. 
Use the provided concept list as guidance, but do not invent details that are not visually supported by the image.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate captions for final_annotation.jsonl with the OpenAI Batch API."
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=DEFAULT_ANNOTATIONS,
        help="Path to final_annotation.jsonl",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=DEFAULT_OUTPUT_JSONL,
        help="Final caption JSONL output path.",
    )
    parser.add_argument(
        "--batch-dir",
        type=Path,
        default=DEFAULT_BATCH_DIR,
        help="Directory for batch request/manifest/output artifacts.",
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
        "--base-url",
        type=str,
        default=os.environ.get("OPENAI_BASE_URL"),
        help="Optional base URL override. Defaults to OPENAI_BASE_URL.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start from this line index in final_annotation.jsonl.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most this many images.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the final JSONL output instead of resuming.",
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Poll the batch job until it reaches a terminal status.",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=30.0,
        help="Seconds between batch status polls when --wait is set.",
    )
    parser.add_argument(
        "--completion-window",
        type=str,
        default="24h",
        choices=["24h"],
        help="Batch completion window.",
    )
    parser.add_argument(
        "--batch-id",
        type=str,
        default="",
        help="Existing batch id to resume instead of creating a new batch.",
    )
    parser.add_argument(
        "--caption-style",
        type=str,
        default="freeform",
        choices=["freeform"],
        help="Caption style to request from the model.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write requests/manifest and print one example request without calling the API.",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Use the OpenAI Batch API. If omitted, use the normal chat API.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Call the API directly for a single image instead of using the Batch API.",
    )
    return parser.parse_args()


def load_annotations_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object on line {line_number} of {path}")
            rows.append(row)
    return rows


def load_completed_caption_ids(path: Path) -> set[int]:
    if not path.exists():
        return set()

    completed_ids: set[int] = set()
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
            if not isinstance(coco_image_id, int):
                continue

            caption = record.get("caption")
            error = record.get("error")
            if (isinstance(caption, str) and caption.strip()) or error == "missing_image_url":
                completed_ids.add(coco_image_id)

    return completed_ids


def flatten_annotated_concepts(annotated_concepts: dict[str, Any]) -> tuple[dict[str, list[str]], list[str]]:
    grouped: dict[str, list[str]] = {}
    flat: list[str] = []
    seen_flat: set[str] = set()

    for concept_type in sorted(annotated_concepts):
        entries = annotated_concepts.get(concept_type, [])
        if not isinstance(entries, list):
            continue

        concepts_for_type: list[str] = []
        seen_type: set[str] = set()
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            concept = entry.get("concept")
            if not isinstance(concept, str):
                continue
            normalized = concept.strip()
            if not normalized or normalized in seen_type:
                continue
            seen_type.add(normalized)
            concepts_for_type.append(normalized)
            if normalized not in seen_flat:
                seen_flat.add(normalized)
                flat.append(normalized)

        if concepts_for_type:
            grouped[concept_type] = concepts_for_type

    return grouped, flat


def build_prompt(*, flat_concepts: list[str]) -> str:
    concepts_text = ", ".join(flat_concepts) if flat_concepts else "(none)"
    return f"""\
Here is the list of annotated concepts for this image:

[{concepts_text}]

Task:
- Look at the image.
- Write a detailed natural-language caption for the image.
- Use the concept annotations as required content to cover in the caption.
- Include all annotated concepts in the caption.
- Keep the caption faithful to the visible image content.
- Do not add details that are not visible in the image.

Respond ONLY with a JSON object in this exact format:

{{"caption": "your caption here"}}\
"""


def build_request(
    *,
    custom_id: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    image_url: str,
) -> dict[str, Any]:
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                },
            ],
        },
    }


def batch_paths(batch_dir: Path) -> dict[str, Path]:
    return {
        "requests": batch_dir / "requests.jsonl",
        "manifest": batch_dir / "requests_manifest.json",
        "batch_id": batch_dir / "batch_id.txt",
        "status": batch_dir / "batch_status.json",
        "output": batch_dir / "batch_output.jsonl",
        "errors": batch_dir / "batch_errors.jsonl",
    }


def prepare_requests(
    *,
    annotations: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], set[int]]:
    processed_ids = set() if args.overwrite else load_completed_caption_ids(args.output_jsonl)

    selected_rows = annotations[args.start_index :]
    if args.limit is not None:
        selected_rows = selected_rows[: args.limit]

    requests: list[dict[str, Any]] = []
    manifest_items: list[dict[str, Any]] = []
    missing_url_items: list[dict[str, Any]] = []

    for row in selected_rows:
        coco_image_id = row.get("coco_image_id")
        image_url = row.get("image_url", "")
        annotated_concepts = row.get("annotated_concepts", {})

        if not isinstance(coco_image_id, int):
            log.warning("Skipping row with invalid coco_image_id: %r", coco_image_id)
            continue
        if coco_image_id in processed_ids:
            continue

        grouped_concepts, flat_concepts = flatten_annotated_concepts(annotated_concepts)
        manifest_item = {
            "coco_image_id": coco_image_id,
            "image_url": image_url,
            "annotated_concepts": annotated_concepts,
            "grouped_concepts": grouped_concepts,
            "flat_concepts": flat_concepts,
        }

        if not isinstance(image_url, str) or not image_url.strip():
            missing_url_items.append(manifest_item)
            continue

        custom_id = f"caption-{coco_image_id}"
        prompt = build_prompt(flat_concepts=flat_concepts)

        requests.append(
            build_request(
                custom_id=custom_id,
                model=args.model,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=prompt,
                image_url=image_url.strip(),
            )
        )
        manifest_item["custom_id"] = custom_id
        manifest_items.append(manifest_item)

    return selected_rows, requests, manifest_items, missing_url_items, processed_ids


def write_request_artifacts(
    *,
    requests: list[dict[str, Any]],
    manifest_items: list[dict[str, Any]],
    missing_url_items: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Path]:
    paths = batch_paths(args.batch_dir)
    args.batch_dir.mkdir(parents=True, exist_ok=True)

    with paths["requests"].open("w", encoding="utf-8") as handle:
        for request in requests:
            handle.write(json.dumps(request, ensure_ascii=False) + "\n")

    manifest = {
        "annotations": str(args.annotations),
        "output_jsonl": str(args.output_jsonl),
        "model": args.model,
        "caption_style": args.caption_style,
        "start_index": args.start_index,
        "limit": args.limit,
        "num_requests": len(requests),
        "missing_image_url": missing_url_items,
        "requests": manifest_items,
    }
    paths["manifest"].write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return paths


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_file_content_text(client: OpenAI, file_id: str) -> str:
    content = client.files.content(file_id)
    text = getattr(content, "text", None)
    if isinstance(text, str):
        return text
    if callable(text):
        value = text()
        if isinstance(value, str):
            return value
    if hasattr(content, "read"):
        raw = content.read()
        if isinstance(raw, bytes):
            return raw.decode("utf-8")
        if isinstance(raw, str):
            return raw
    raise ValueError(f"Unable to read file content for {file_id}")


def upload_and_create_batch(client: OpenAI, requests_path: Path, args: argparse.Namespace) -> Any:
    with requests_path.open("rb") as handle:
        uploaded = client.files.create(file=handle, purpose="batch")
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window=args.completion_window,
        metadata={"script": "generate_captions", "model": args.model},
    )
    return batch


def persist_batch_status(paths: dict[str, Path], batch: Any) -> None:
    paths["status"].write_text(
        json.dumps(sdk_response_to_dict(batch), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def poll_batch(client: OpenAI, batch_id: str, poll_interval: float) -> Any:
    terminal_statuses = {"completed", "failed", "expired", "cancelled"}
    while True:
        batch = client.batches.retrieve(batch_id)
        status = getattr(batch, "status", None)
        request_counts = getattr(batch, "request_counts", None)
        counts_repr = sdk_response_to_dict(request_counts)
        log.info("Batch %s status=%s request_counts=%s", batch_id, status, counts_repr)
        if status in terminal_statuses:
            return batch
        time.sleep(poll_interval)


def download_batch_files(client: OpenAI, batch: Any, paths: dict[str, Path]) -> None:
    if getattr(batch, "output_file_id", None):
        output_text = read_file_content_text(client, batch.output_file_id)
        paths["output"].write_text(output_text, encoding="utf-8")
        log.info("Wrote batch output to %s", paths["output"])
    if getattr(batch, "error_file_id", None):
        error_text = read_file_content_text(client, batch.error_file_id)
        paths["errors"].write_text(error_text, encoding="utf-8")
        log.info("Wrote batch errors to %s", paths["errors"])


def run_test_request(
    *,
    client: OpenAI,
    request: dict[str, Any],
    manifest_item: dict[str, Any],
    output_jsonl: Path,
    overwrite: bool,
) -> dict[str, Any]:
    body = request["body"]
    response = client.chat.completions.create(**body)
    response_dict = sdk_response_to_dict(response)
    raw_text = None

    if isinstance(response_dict, dict):
        choices = response_dict.get("choices", [])
        if choices:
            raw_text = choices[0].get("message", {}).get("content")

    caption = None
    if isinstance(raw_text, str):
        parsed = parse_json_object(raw_text)
        if isinstance(parsed, dict) and isinstance(parsed.get("caption"), str):
            caption = " ".join(parsed["caption"].split()) or None

    record: dict[str, Any] = {
        "coco_image_id": manifest_item.get("coco_image_id"),
        "image_url": manifest_item.get("image_url", ""),
        "annotated_concepts": manifest_item.get("annotated_concepts", {}),
        "caption": caption,
        "error": None if caption is not None else "invalid_caption_response",
        "raw_text": raw_text,
        "raw_response": response_dict,
    }

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if overwrite and output_jsonl.exists():
        output_jsonl.unlink()
    with output_jsonl.open("a", encoding="utf-8") as out_handle:
        out_handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    return record


def print_test_summary(manifest_item: dict[str, Any], record: dict[str, Any]) -> None:
    concepts = manifest_item.get("flat_concepts", [])
    caption = record.get("caption")

    print("Concepts:")
    if isinstance(concepts, list) and concepts:
        for concept in concepts:
            print(f"- {concept}")
    else:
        print("- (none)")

    print()
    print("Caption:")
    print(caption if isinstance(caption, str) and caption else "(no caption)")


def run_direct_request(client: OpenAI, request: dict[str, Any]) -> tuple[str | None, str | None, Any]:
    body = request["body"]
    response = client.chat.completions.create(**body)
    response_dict = sdk_response_to_dict(response)
    raw_text = None

    if isinstance(response_dict, dict):
        choices = response_dict.get("choices", [])
        if choices:
            raw_text = choices[0].get("message", {}).get("content")

    caption = None
    if isinstance(raw_text, str):
        parsed = parse_json_object(raw_text)
        if isinstance(parsed, dict) and isinstance(parsed.get("caption"), str):
            caption = " ".join(parsed["caption"].split()) or None

    return caption, raw_text, response_dict


def build_caption_record(
    *,
    item: dict[str, Any],
    caption: str | None,
    error: str | None,
    raw_text: str | None = None,
    raw_response: Any = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "coco_image_id": item.get("coco_image_id"),
        "image_url": item.get("image_url", ""),
        "annotated_concepts": item.get("annotated_concepts", {}),
        "caption": caption,
        "error": error,
    }
    if raw_text is not None:
        record["raw_text"] = raw_text
    if raw_response is not None:
        record["raw_response"] = raw_response
    return record


def materialize_direct_results(
    *,
    client: OpenAI,
    requests: list[dict[str, Any]],
    manifest_items: list[dict[str, Any]],
    missing_url_items: list[dict[str, Any]],
    output_jsonl: Path,
    overwrite: bool,
) -> None:
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if overwrite and output_jsonl.exists():
        output_jsonl.unlink()
    processed_ids = set() if overwrite else load_completed_caption_ids(output_jsonl)

    with output_jsonl.open("a", encoding="utf-8") as out_handle:
        for item in missing_url_items:
            coco_image_id = item["coco_image_id"]
            if coco_image_id in processed_ids:
                continue
            record = {
                "coco_image_id": coco_image_id,
                "image_url": item.get("image_url", ""),
                "annotated_concepts": item.get("annotated_concepts", {}),
                "caption": None,
                "error": "missing_image_url",
            }
            out_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_handle.flush()
            processed_ids.add(coco_image_id)

        for request, item in zip(requests, manifest_items):
            coco_image_id = item["coco_image_id"]
            if coco_image_id in processed_ids:
                continue

            caption = None
            raw_text = None
            raw_response = None
            error = None

            try:
                caption, raw_text, raw_response = run_direct_request(client, request)
                if caption is None:
                    error = "invalid_caption_response"
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"

            record: dict[str, Any] = {
                "coco_image_id": coco_image_id,
                "image_url": item.get("image_url", ""),
                "annotated_concepts": item.get("annotated_concepts", {}),
                "caption": caption,
                "error": error,
            }
            if raw_text is not None:
                record["raw_text"] = raw_text
            if raw_response is not None:
                record["raw_response"] = raw_response

            out_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_handle.flush()
            processed_ids.add(coco_image_id)
            log.info("Processed image %s", coco_image_id)


def extract_caption_from_row(row: dict[str, Any]) -> tuple[str | None, str | None]:
    response = row.get("response", {})
    response_body = response.get("body", {}) if isinstance(response, dict) else {}
    choices = response_body.get("choices", []) if isinstance(response_body, dict) else []
    raw_text = None
    if choices:
        raw_text = choices[0].get("message", {}).get("content")
    if not isinstance(raw_text, str):
        return None, None

    parsed = parse_json_object(raw_text)
    if not isinstance(parsed, dict):
        return None, raw_text

    caption = parsed.get("caption")
    if not isinstance(caption, str):
        return None, raw_text

    cleaned = " ".join(caption.split())
    return cleaned or None, raw_text


def describe_batch_error(row: dict[str, Any]) -> str | None:
    if not row:
        return "missing_batch_result"

    if row.get("error"):
        return str(row["error"])

    response = row.get("response", {})
    if not isinstance(response, dict):
        return "missing_batch_response"

    status_code = response.get("status_code")
    if status_code != 200:
        body = response.get("body", {})
        if isinstance(body, dict):
            error_body = body.get("error")
            if isinstance(error_body, dict):
                message = error_body.get("message")
                if isinstance(message, str) and message.strip():
                    return f"status_code:{status_code}: {message.strip()}"
        return f"status_code:{status_code}"

    return None


def retry_caption_request_direct(
    client: OpenAI,
    request: dict[str, Any],
) -> tuple[str | None, str | None, Any, str | None]:
    caption, raw_text, raw_response = run_direct_request(client, request)
    error = None if caption is not None else "invalid_caption_response"
    return caption, raw_text, raw_response, error


def materialize_results(
    *,
    client: OpenAI,
    output_jsonl: Path,
    overwrite: bool,
    requests_path: Path,
    manifest_path: Path,
    batch_output_path: Path,
    batch_error_path: Path | None = None,
) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = load_jsonl(batch_output_path) if batch_output_path.exists() else []
    error_rows = load_jsonl(batch_error_path) if batch_error_path and batch_error_path.exists() else []
    request_rows = load_jsonl(requests_path)
    rows_by_custom_id = {row.get("custom_id"): row for row in rows}
    error_rows_by_custom_id = {row.get("custom_id"): row for row in error_rows}
    requests_by_custom_id = {row.get("custom_id"): row for row in request_rows}

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if overwrite and output_jsonl.exists():
        output_jsonl.unlink()
    processed_ids = set() if overwrite else load_completed_caption_ids(output_jsonl)

    with output_jsonl.open("a", encoding="utf-8") as out_handle:
        for item in manifest.get("missing_image_url", []):
            coco_image_id = item["coco_image_id"]
            if coco_image_id in processed_ids:
                continue

            record = {
                "coco_image_id": coco_image_id,
                "image_url": item.get("image_url", ""),
                "annotated_concepts": item.get("annotated_concepts", {}),
                "caption": None,
                "error": "missing_image_url",
            }
            out_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_handle.flush()
            processed_ids.add(coco_image_id)

        for item in manifest["requests"]:
            coco_image_id = item["coco_image_id"]
            if coco_image_id in processed_ids:
                continue

            custom_id = item["custom_id"]
            row = rows_by_custom_id.get(custom_id) or error_rows_by_custom_id.get(custom_id, {})
            caption, raw_text = extract_caption_from_row(row)
            batch_error = describe_batch_error(row)

            if batch_error is None and caption is not None:
                record = build_caption_record(
                    item=item,
                    caption=caption,
                    error=None,
                    raw_text=raw_text,
                )
            else:
                request = requests_by_custom_id.get(custom_id)
                if request is None:
                    log.warning("Missing original request for failed batch item %s", custom_id)
                    record = build_caption_record(
                        item=item,
                        caption=caption,
                        error=batch_error or "invalid_caption_response",
                        raw_text=raw_text,
                    )
                else:
                    log.info(
                        "Retrying failed batch item %s for image %s via normal chat API",
                        custom_id,
                        coco_image_id,
                    )
                    try:
                        retry_caption, retry_raw_text, retry_raw_response, retry_error = (
                            retry_caption_request_direct(client, request)
                        )
                        record = build_caption_record(
                            item=item,
                            caption=retry_caption,
                            error=retry_error,
                            raw_text=retry_raw_text,
                            raw_response=retry_raw_response,
                        )
                        record["batch_error"] = batch_error or "invalid_caption_response"
                        record["retried_via_chat_api"] = True
                        record["retry_succeeded"] = retry_error is None
                    except Exception as exc:
                        record = build_caption_record(
                            item=item,
                            caption=caption,
                            error=f"{type(exc).__name__}: {exc}",
                            raw_text=raw_text,
                        )
                        record["batch_error"] = batch_error or "invalid_caption_response"
                        record["retried_via_chat_api"] = True
                        record["retry_succeeded"] = False

            out_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_handle.flush()
            processed_ids.add(coco_image_id)


def main() -> None:
    args = parse_args()

    if not args.annotations.exists():
        raise FileNotFoundError(f"Annotation file not found: {args.annotations}")

    annotations = load_annotations_jsonl(args.annotations)
    selected_rows, requests, manifest_items, missing_url_items, processed_ids = prepare_requests(
        annotations=annotations,
        args=args,
    )

    paths = write_request_artifacts(
        requests=requests,
        manifest_items=manifest_items,
        missing_url_items=missing_url_items,
        args=args,
    )

    log.info("Loaded %s annotation rows from %s", len(annotations), args.annotations)
    log.info(
        "Selected %s rows; prepared %s batch requests; skipped %s already processed; %s missing image_url",
        len(selected_rows),
        len(requests),
        len(processed_ids),
        len(missing_url_items),
    )
    log.info("Wrote batch requests to %s", paths["requests"])
    log.info("Wrote request manifest to %s", paths["manifest"])

    if args.dry_run:
        if requests:
            print(json.dumps(requests[0], indent=2, ensure_ascii=False))
        else:
            print("No batch requests to submit.")
        return

    if not args.api_key:
        raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY or pass --api-key.")

    client_kwargs: dict[str, Any] = {"api_key": args.api_key}
    if args.base_url:
        client_kwargs["base_url"] = args.base_url
    client = OpenAI(**client_kwargs)

    if args.test:
        if not requests:
            raise ValueError("No test request available. Check filters or existing processed outputs.")
        record = run_test_request(
            client=client,
            request=requests[0],
            manifest_item=manifest_items[0],
            output_jsonl=args.output_jsonl,
            overwrite=args.overwrite,
        )
        print_test_summary(manifest_items[0], record)
        return

    if not args.batch:
        materialize_direct_results(
            client=client,
            requests=requests,
            manifest_items=manifest_items,
            missing_url_items=missing_url_items,
            output_jsonl=args.output_jsonl,
            overwrite=args.overwrite,
        )
        return

    if args.batch_id:
        batch_id = args.batch_id
        log.info("Resuming existing batch %s", batch_id)
        batch = client.batches.retrieve(batch_id)
    else:
        batch = upload_and_create_batch(client, paths["requests"], args)
        batch_id = batch.id
        paths["batch_id"].write_text(f"{batch_id}\n", encoding="utf-8")
        log.info("Created batch %s", batch_id)

    persist_batch_status(paths, batch)

    if args.wait:
        batch = poll_batch(client, batch_id, args.poll_interval)
        persist_batch_status(paths, batch)
    else:
        status = getattr(batch, "status", None)
        log.info("Batch %s status=%s", batch_id, status)

    if getattr(batch, "status", None) == "completed":
        download_batch_files(client, batch, paths)
        if paths["output"].exists() or paths["errors"].exists():
            materialize_results(
                client=client,
                output_jsonl=args.output_jsonl,
                overwrite=args.overwrite,
                requests_path=paths["requests"],
                manifest_path=paths["manifest"],
                batch_output_path=paths["output"],
                batch_error_path=paths["errors"],
            )
    else:
        log.info(
            "Batch %s is not complete. Re-run with --batch-id %s --wait to finish and materialize outputs.",
            batch_id,
            batch_id,
        )


if __name__ == "__main__":
    main()
