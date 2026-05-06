#!/usr/bin/env python3
"""
Prepare Google Gemini Batch API requests for concept filtering.

This script does not submit the batch. It creates:
  - a JSONL input file with one keyed GenerateContentRequest per line
  - a manifest JSON file mapping each request line to its concepts

Usage:
    python prepare_google_batch_requests.py \
        --input output/05_all_concepts.txt \
        --output output/google_batch \
        --model gemini-3.1-pro-preview
"""

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("prepare_google_batch")

SYSTEM_PROMPT = """\
You are an expert in computer vision, visual perception, and ontology design. \
Your task is to evaluate visual concepts for inclusion in a minimal, \
comprehensive vocabulary that can describe any natural image through composition.

You apply rigorous criteria and return structured JSON. You never explain \
or hedge — just evaluate and output the requested format.\
"""


def build_batch_prompt(batch: list[dict]) -> str:
    concept_list = "\n".join(
        f"  {i + 1:3d}. {entry['concept']}"
        for i, entry in enumerate(batch)
    )

    return f"""\
# Task
 
Evaluate each of the candidate visual concepts listed below. These concepts were extracted from computer vision dataset annotations and are candidates for a **minimal, comprehensive visual concept vocabulary**. The vocabulary should contain only concepts that are atomic, visually grounded, and useful for describing natural images through composition.
 
For each concept, apply five criteria and assign a visual category.
 
---
 
# Criteria
 
## C1 — Lexical well-formedness
Is this entry a clean, correctly spelled, standalone concept term?
 
REJECT if ANY of these apply:
- **Misspelling or typo**: "balck" (→ black), "brwon" (→ brown), "celing" (→ ceiling), "widshield" (→ windshield), "aentenna" (→ antenna).
- **Sentence fragment**: "is red", "are blue", "in black", "written in white", "color black", "colored green". A concept is a noun, adjective, or verb — not a copula+adjective or preposition+adjective.
- **Formatting artifact**: "black+white", "red,white,blue", "red-and-white". Conjunctions with symbols are data artifacts, not concepts.
- **Definition or synonym list**: "bannister, banister, balustrade, balusters, handrail", "door, double door", "wiper (for windshield/screen)", "cap (container lid)". Parenthetical glosses and comma-separated alternatives are dictionary entries, not concept terms.
- **Dangling phrase**: "top of", "arm of a person", "back of chair", "side of building". If it requires a following noun to be meaningful, it's a fragment.
 
PASS if it's a real word or short phrase (1-3 words), correctly spelled, that can stand alone as a concept label.
 
## C2 — Visual groundability
Can this concept be identified **purely from pixel-level appearance** in a photograph, without requiring external knowledge, text, audio, or reasoning about non-visible properties?
 
- PASS: "red", "furry", "sitting", "round", "kitchen" — directly observable.
- FAIL: "expensive", "famous", "illegal", "tuesday", "owned", "imaginary" — require inference beyond pixels.
 
Edge cases: "old" passes (wrinkles, patina, wear are visible). "happy" passes (facial expression). "dangerous" fails (requires reasoning). "heavy" fails (weight not visible). "frozen" passes (ice crystals, frost are visible surface properties).
 
## C3 — Compositional atomicity
Is this concept a **single, indivisible visual primitive**, or is it a conjunction of two or more simpler concepts that each belong to a recognized visual category?
 
A concept is COMPOUND (reject) if it can be decomposed as:
- **modifier + base** where both parts are independently valid concepts:
  - color + color: "black and white" → black + white
  - brightness + color: "dark blue" → dark + blue, "bright red" → bright + red, "pale pink" → pale + pink
  - color + part: "red lips" → red + lips, "blue eyes" → blue + eyes, "white belly" → white + belly, "yellow wheel" → yellow + wheel
  - material + part: "metal handle" → metal + handle, "wood beam" → wood + beam, "glass pane" → glass + pane
  - shape + part: "long sleeve" → long + sleeve, "short mane" → short + mane, "pointy ear" → pointy + ear
  - state + part: "closed eye" → closed + eye, "bent knees" → bent + knees, "bare branch" → bare + branch
  - object + part: "airplane door" → airplane + door, "car hood" → car + hood, "dog ear" → dog + ear, "banana stem" → banana + stem, "pizza crust" → pizza + crust, "zebra's leg" → zebra + leg
  - object + color: "painted green" → painted + green, "blonde hair" → blonde + hair
  - color + color compound: "reddish brown" → red + brown, "greenish yellow" → green + yellow, "blue-gray" → blue + gray
 
A concept is ATOMIC (keep) if:
- It names a **single** thing, property, or relationship: "red", "sleeve", "door", "rough", "sitting".
- Even if linguistically analyzable, the visual gestalt is distinct from its parts: "rainbow" (not just many colors side by side — has arc shape), "camouflage" (not just green+brown+pattern — has specific military-derived appearance).
 
The key test: **can you express the same visual content by listing the components separately?** If yes, it's compound. If something is lost, it's atomic.
 
## C4 — Semantic level
Is the concept at the right level of abstraction?
 
- TOO_ABSTRACT: overly general, nearly every image contains it. "Thing", "entity", "stuff", "area", "object", "item", "place".
- BASIC: the natural, default level humans use. "Dog" not "mammal" or "golden retriever". "Red" not "color" or "crimson". "Running" not "moving" or "jogging at 6mph". This is the target level.
- TOO_SPECIFIC: fine-grained subtype not visually distinct from parent, or requires domain expertise. "Boeing 747-400", "Labrador retriever" (vs "dog"), "Hepplewhite chair" (vs "chair"), "terra cotta" (vs "orange-brown"), "teal blue" (vs "teal").
 
## C5 — Positive assertion
Does this concept describe the **presence** of a visual property, or the **absence** of one?
 
- POSITIVE (keep): "sleeve", "door", "striped", "bearded" — describes something that IS visible.
- NEGATIVE (reject): "sleeveless", "doorless", "leafless", "headless", "strapless", "armless", "frameless", "unpeeled", "with no leaves" — describes something that is NOT there.
 
Absence concepts are linguistically useful but are not visual primitives. You describe what you see, not what you don't see. The vocabulary can express "a shirt without sleeves" as "shirt" + (no "sleeve"), rather than needing "sleeveless" as its own concept.
 
Exception: concepts where the "-less" form names a visually distinctive state that is not simply the parent minus the part: "wireless" (names a technology category with its own visual signature), "seamless" (describes a continuous surface — a positive visual property, not just the absence of seams). These are rare.
 
---
 
# Visual category assignment
 
Assign each concept to exactly ONE of these categories:
 
| Category | Definition | Examples |
|----------|------------|----------|
| color | Hue, saturation, brightness, color patterns | red, blue, golden, pale, dark, pastel, multicolored |
| edge | Boundary quality, contour properties, sharpness | sharp, blurry, smooth-edged, jagged, curved |
| texture | Surface pattern, tactile appearance, material finish | rough, smooth, woven, striped, dotted, furry, metallic |
| shape | Geometric form, spatial extent, proportions | round, rectangular, tall, thin, flat, elongated |
| part | Component of a larger object | wheel, handle, leg, roof, screen, wing, door |
| object | Discrete nameable entity or stuff region | dog, car, sky, grass, person, table, water |
| motion | Action, pose, dynamic state, body configuration | running, sitting, flying, falling, parked, open, closed |
| relation | Spatial, possessive, or interactive relationship | on, in, next to, holding, wearing, behind, above |
 
Disambiguation: "Furry" → texture. "Wheel" → part. "Standing" → motion. "Wooden" → texture. "Tall" → shape. "Behind" → relation.
 
---
 
# Concepts to evaluate
 
{concept_list}
 
---
 
# Output format
 
Return a JSON array with exactly {len(batch)} objects, one per concept, in the same order. No text outside the JSON.
 
```json
[
  {{
    "id": 1,
    "concept": "<concept name>",
    "category": "<color|edge|texture|shape|part|object|motion|relation>",
    "c1_wellformed": true,
    "c2_groundable": true,
    "c3_atomic": true,
    "c3_decomposes_to": null,
    "c4_level": "basic",
    "c5_positive": true,
    "decision": "<keep|reject>",
    "reject_reasons": null
  }}
]
```
 
**Decision rules:**
- "keep" if ALL: c1=true, c2=true, c3=true, c4=basic, c5=true.
- "reject" if ANY fails. List all failed criteria in reject_reasons (e.g., ["C1","C3"]).
 
When c3_atomic is false, always populate c3_decomposes_to with the components (e.g., "dark + blue" or "airplane + door").
 
Be aggressive. When in doubt, reject.
 
Output ONLY the JSON array.
"""


def load_concepts(input_path: Path) -> list[dict]:
    entries = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            concept = line.strip().strip("'\"")
            if not concept:
                continue
            entries.append({"concept": concept, "original_axis": "all"})
    return entries


def batch_concepts(entries: list[dict], batch_size: int) -> list[list[dict]]:
    return [
        entries[i:i + batch_size]
        for i in range(0, len(entries), batch_size)
    ]


def make_request(key: str, prompt: str, max_output_tokens: int) -> dict:
    return {
        "key": key,
        "request": {
            "systemInstruction": {
                "parts": [{"text": SYSTEM_PROMPT}],
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ],
            "generationConfig": {
                "temperature": 0.0,
                # "maxOutputTokens": max_output_tokens,
            },
        }
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare Google Gemini Batch API input for concept filtering",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("output/05_all_concepts.txt"),
        help="Input newline-delimited concept text file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/google_batch"),
        help="Directory for generated batch request files",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-3.1-pro-preview",
        help="Gemini model for the batch job",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Concepts per request",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=8000,
        help="maxOutputTokens per request",
    )
    args = parser.parse_args()

    entries = load_concepts(args.input)
    batches = batch_concepts(entries, args.batch_size)
    args.output.mkdir(parents=True, exist_ok=True)

    requests_path = args.output / "requests.jsonl"
    manifest_path = args.output / "requests_manifest.json"

    manifest = {
        "input": str(args.input),
        "model": args.model,
        "batch_size": args.batch_size,
        "max_output_tokens": args.max_output_tokens,
        "total_concepts": len(entries),
        "num_requests": len(batches),
        "requests": [],
    }

    with open(requests_path, "w", encoding="utf-8") as requests_file:
        for batch_idx, batch in enumerate(batches, start=1):
            key = f"concept-filter-batch-{batch_idx:04d}"
            request = make_request(key, build_batch_prompt(batch), args.max_output_tokens)
            requests_file.write(json.dumps(request, ensure_ascii=False) + "\n")
            manifest["requests"].append({
                "key": key,
                "line_number": batch_idx,
                "size": len(batch),
                "concepts": [
                    {
                        "concept": entry["concept"],
                        "original_axis": entry.get("original_axis"),
                    }
                    for entry in batch
                ],
            })

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    log.info(f"Saved batch input     → {requests_path}")
    log.info(f"Saved batch manifest  → {manifest_path}")
    log.info(f"Prepared {len(batches)} requests for {len(entries)} concepts")
    log.info(
        "Submit this file with Gemini Batch API file input for model %s",
        args.model,
    )


if __name__ == "__main__":
    main()
