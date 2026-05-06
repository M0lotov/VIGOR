#!/usr/bin/env python3
"""
LLM-based Visual Concept Filtering
====================================
Sends batches of 50 concepts to an LLM for quality filtering based on:
  - Visual groundability
  - Atomicity (non-composability)
  - Visual distinctiveness
  - Semantic level (basic-level preference)

Also categorizes each concept into a visual primitive type:
  color | edge | texture | shape | part | object | motion | relation

Usage:
    python llm_filter_concepts.py \
        --input output/05_all_concepts.txt \
        --output output/06_llm_filtered.json \
        --model claude-sonnet-4-20250514 \
        --api-key $ANTHROPIC_API_KEY

    # Dry-run: just print the prompt for the first batch
    python llm_filter_concepts.py \
        --input output/05_all_concepts.txt \
        --dry-run
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from tqdm import tqdm
from typing import Any, Optional
from openai import OpenAI
from anthropic import Anthropic
from google import genai
from google.genai import types

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("llm_filter")

DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-4-20250514",
    "openai": "gpt-5.4",
    "google": "gemini-2.5-pro",
}

# =========================================================================
# PROMPT CONSTRUCTION
# =========================================================================

SYSTEM_PROMPT = """\
You are an expert in computer vision, visual perception, and ontology design. \
Your task is to evaluate visual concepts for inclusion in a minimal, \
comprehensive vocabulary that can describe any natural image through composition.

You apply rigorous criteria and return structured JSON. You never explain \
or hedge — just evaluate and output the requested format.\
"""


def build_batch_prompt(
    batch: list[dict],
) -> str:
    """
    Build the user prompt for one batch of ~50 concepts.

    Args:
        batch: list of {"concept": str, "count": int, "sources": list[str]}
    """

    # Format the batch as a numbered list
    concept_list = "\n".join(
        f"  {i+1:3d}. {c['concept']}"
        for i, c in enumerate(batch)
    )

    prompt = f"""\
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

    return prompt


# =========================================================================
# API CALLING
# =========================================================================

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


def extract_anthropic_text(response: Any) -> str:
    """Extract text content from an Anthropic SDK response."""
    parts = []
    for block in getattr(response, "content", []) or []:
        if getattr(block, "type", None) == "text":
            parts.append(getattr(block, "text", ""))
    return "".join(parts)


def call_anthropic(
    system: str,
    user: str,
    model: str = "claude-sonnet-4-20250514",
    api_key: str = "",
    max_tokens: int = 8000,
    temperature: float = 0.0,
) -> tuple[Optional[str], Optional[Any]]:
    """Call Anthropic using the official SDK."""

    client = Anthropic(api_key=api_key)

    for attempt in range(3):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return extract_anthropic_text(response), response
        except Exception as e:
            log.error(f"Request failed: {e}")
            if attempt < 2:
                wait = 2 ** (attempt + 1)
                time.sleep(wait)

    return None, None


def call_openai(
    system: str,
    user: str,
    model: str = "gpt-4o",
    api_key: str = "",
    temperature: float = 0.0,
) -> tuple[Optional[str], Optional[Any]]:
    """Call OpenAI using the official SDK."""

    client = OpenAI(api_key=api_key)

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            text = response.choices[0].message.content if response.choices else None
            return text, response
        except Exception as e:
            log.error(f"Request failed: {e}")
            if attempt < 2:
                wait = 2 ** (attempt + 1)
                time.sleep(wait)

    return None, None


def call_google(
    system: str,
    user: str,
    model: str = "gemini-2.5-pro",
    api_key: str = "",
    max_tokens: int = 8000,
    temperature: float = 0.0,
) -> tuple[Optional[str], Optional[Any]]:
    """Call Google Gemini using the official SDK."""

    client = genai.Client(api_key=api_key)

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=model,
                contents=user,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                ),
            )
            return getattr(response, "text", None), response
        except Exception as e:
            log.error(f"Request failed: {e}")
            if attempt < 2:
                wait = 2 ** (attempt + 1)
                time.sleep(wait)

    return None, None


def parse_llm_response(text: str) -> Optional[list[dict]]:
    """Parse the JSON array from the LLM response, tolerating some noise."""
    text = text.strip()
    # Strip markdown fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last fence lines
        start = 1 if lines[0].startswith("```") else 0
        end = -1 if lines[-1].strip() == "```" else len(lines)
        text = "\n".join(lines[start:end])

    # Find the JSON array boundaries
    first_bracket = text.find("[")
    last_bracket = text.rfind("]")
    if first_bracket == -1 or last_bracket == -1:
        log.error("No JSON array found in response")
        return None

    json_str = text[first_bracket:last_bracket + 1]

    try:
        result = json.loads(json_str)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError as e:
        log.error(f"JSON parse error: {e}")
        # Try to fix common issues: trailing commas
        json_str = json_str.replace(",\n]", "\n]").replace(",]", "]")
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

    return None


# =========================================================================
# BATCH PROCESSING
# =========================================================================

def load_concept_pool(input_path: Path) -> dict[str, list[dict]]:
    """Load one concept per line from a text file into a single pooled axis."""
    entries = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            concept = line.strip().strip("'\"")
            if not concept:
                continue
            entries.append({"concept": concept})
    return {"all": entries}


def flatten_pool(pool_data: dict) -> list[dict]:
    """Flatten the axis-keyed pool into one concept list while preserving origin."""
    flattened = []
    for axis, entries in pool_data.items():
        for entry in entries:
            flattened.append({
                **entry,
                "original_axis": axis,
            })
    return flattened


def batch_concepts(entries: list[dict], batch_size: int = 50) -> list[list[dict]]:
    """Split concept entries into batches."""
    return [
        entries[i:i + batch_size]
        for i in range(0, len(entries), batch_size)
    ]


def write_category_txts(directory: Path, by_category: dict[str, list[dict]]) -> None:
    """Write one text file per visual category."""
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
    """Write all rejected concepts to a flat text file."""
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


def process_pool(
    pool_data: dict,
    api_provider: str,
    model: str,
    api_key: str,
    output_path: Path,
    batch_size: int = 50,
    dry_run: bool = False,
):
    """Process the entire concept pool through LLM filtering."""

    all_entries = flatten_pool(pool_data)
    all_results = []
    raw_responses = []
    total_kept = 0
    total_rejected = 0
    if not all_entries:
        log.warning("No concepts found in input pool")
        return

    provider_output_path = output_path / api_provider
    provider_output_path.mkdir(parents=True, exist_ok=True)

    log.info(f"\n{'='*50}")
    log.info(
        f"Processing combined pool ({len(all_entries)} concepts) "
        f"with provider={api_provider}"
    )
    log.info(f"{'='*50}")

    batches = batch_concepts(all_entries, batch_size)

    for batch_idx, batch in tqdm(enumerate(batches[:2]), total=len(batches[:2])):
        log.info(
            f"  Batch {batch_idx + 1}/{len(batches)} "
            f"({len(batch)} concepts)"
        )

        prompt = build_batch_prompt(batch)

        if dry_run:
            log.info(f"\n--- DRY RUN: Prompt for batch {batch_idx + 1} ---")
            print(f"\n{'='*70}")
            print(f"SYSTEM:\n{SYSTEM_PROMPT}")
            print(f"\nUSER:\n{prompt}")
            print(f"{'='*70}\n")
            if batch_idx == 0:
                # Only print one batch in dry-run
                log.info("(Showing first batch only in dry-run mode)")
                break
            continue

        # Call LLM
        if api_provider == "anthropic":
            response_text, raw_response = call_anthropic(
                SYSTEM_PROMPT, prompt,
                model=model, api_key=api_key,
            )
        elif api_provider == "openai":
            response_text, raw_response = call_openai(
                SYSTEM_PROMPT, prompt,
                model=model, api_key=api_key,
            )
        elif api_provider == "google":
            response_text, raw_response = call_google(
                SYSTEM_PROMPT, prompt,
                model=model, api_key=api_key,
            )
        else:
            log.error(f"Unknown provider: {api_provider}")
            return

        raw_responses.append({
            # "provider": api_provider,
            # "model": model,
            "batch_index": batch_idx + 1,
            # "num_batches": len(batches),
            "batch_size": len(batch),
            # "concepts": [entry["concept"] for entry in batch],
            "response_text": response_text,
            # "raw_response": sdk_response_to_dict(raw_response),
        })

        if not response_text:
            log.error(f"  No response for batch {batch_idx + 1}, skipping")
            for entry in batch:
                all_results.append({
                    'concept': entry['concept'],
                    "decision": "unfiltered",
                    "llm_error": "no_response",
                })
            continue

        # Parse response
        parsed = parse_llm_response(response_text)
        if not parsed:
            log.error(f"  Failed to parse batch {batch_idx + 1}, skipping")
            for entry in batch:
                all_results.append({
                    'concept': entry['concept'],
                    "decision": "unfiltered",
                    "llm_error": "parse_failure",
                })
            continue

        # Match responses to input concepts
        for i, entry in enumerate(batch):
            if i < len(parsed):
                judgment = parsed[i]
                decision = judgment.get("decision", "keep")

                result = {
                    'concept': entry['concept'],
                    "category": judgment.get("category", "unknown"),
                    "decision": decision,
                }
                all_results.append(result)

                if decision == "keep":
                    total_kept += 1
                elif decision == "reject":
                    total_rejected += 1
            else:
                all_results.append({
                    'concept': entry['concept'],
                    "decision": "unfiltered",
                    "llm_error": "missing_in_response",
                })

        # Rate limit courtesy
        time.sleep(1.0)

    # ---- Save full results (every concept with its judgment) ----
    raw_path = provider_output_path / "05_raw_responses.json"
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(raw_responses, f, indent=2, ensure_ascii=False)
    log.info(f"\nSaved raw responses → {raw_path}")

    full_path = provider_output_path / "06_llm_judgments_full.json"
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    log.info(f"\nSaved full judgments → {full_path}")

    if dry_run:
        return

    # ---- Save filtered pool ----
    filtered = [
        r for r in all_results
        if r.get("decision") == "keep"
    ]

    filtered_path = provider_output_path / "07_filtered_concept_pool.json"
    with open(filtered_path, "w", encoding="utf-8") as f:
        json.dump(filtered, f, indent=2, ensure_ascii=False)
    log.info(f"Saved filtered pool → {filtered_path}")

    # ---- Save categorized view (grouped by visual category) ----
    by_category = {}
    for r in all_results:
        if r.get("decision") != "keep":
            continue
        cat = r.get("category", "unknown")
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append({
            "concept": r["concept"],
        })

    cat_path = provider_output_path / "08_by_visual_category.json"
    with open(cat_path, "w", encoding="utf-8") as f:
        json.dump(by_category, f, indent=2, ensure_ascii=False)
    log.info(f"Saved categorized view → {cat_path}")

    filtered_txt_dir = provider_output_path / "08_filtered_by_category_txt"
    write_category_txts(filtered_txt_dir, by_category)
    log.info(f"Saved filtered txts  → {filtered_txt_dir}")

    rejected_txt_path = provider_output_path / "09_rejected_concepts.txt"
    write_rejected_txt(rejected_txt_path, all_results)
    log.info(f"Saved rejected txt   → {rejected_txt_path}")

    # ---- Summary ----
    log.info(f"\n{'='*50}")
    log.info("FILTERING SUMMARY")
    log.info(f"{'='*50}")
    log.info(f"  Kept:       {total_kept:>5d}")
    log.info(f"  Rejected:   {total_rejected:>5d}")
    log.info(f"  Total:      {total_kept + total_rejected:>5d}")

    log.info(f"\nFiltered pool by visual category:")
    for cat in sorted(by_category.keys()):
        log.info(f"  {cat:12s}: {len(by_category[cat]):>5d}")


# =========================================================================
# MAIN
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Filter visual concepts using an LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        # "--input", type=Path, default='output/05_all_concepts.txt',
        "--input", type=Path, default='output/openai/08_filtered_by_category_txt/part.txt',
        help="Path to newline-delimited concept pool text file",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("./output"),
        help="Output directory",
    )
    parser.add_argument(
        "--provider", choices=["anthropic", "openai", "google"], default="openai",
        help="LLM API provider",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Model name (defaults depend on provider)",
    )
    parser.add_argument(
        "--api-key", type=str, default=None,
        help="API key (or set the provider-specific env var)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=50,
        help="Concepts per LLM call (default: 50)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the prompt for the first batch without calling the API",
    )
    args = parser.parse_args()

    # Resolve API key
    api_key = args.api_key
    if not api_key:
        if args.provider == "anthropic":
            api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        elif args.provider == "openai":
            api_key = os.environ.get("OPENAI_API_KEY", "")
        else:
            api_key = (
                os.environ.get("GOOGLE_API_KEY", "")
                or os.environ.get("GEMINI_API_KEY", "")
            )

    model = args.model or DEFAULT_MODELS[args.provider]

    if not api_key and not args.dry_run:
        log.error("No API key provided. Use --api-key or set env var.")
        sys.exit(1)

    # Load pool
    pool_data = load_concept_pool(args.input)

    total = sum(len(v) for v in pool_data.values())
    log.info(f"Loaded {total} concepts from {args.input}")

    args.output.mkdir(parents=True, exist_ok=True)

    process_pool(
        pool_data=pool_data,
        api_provider=args.provider,
        model=model,
        api_key=api_key,
        output_path=args.output,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
