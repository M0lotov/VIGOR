#!/usr/bin/env python3
"""Use an LLM to perform the final pool-level cleaning pass."""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

from anthropic import Anthropic
from google import genai
from google.genai import types
from openai import OpenAI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("final_cleaning")

DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-4-20250514",
    "openai": "gpt-5.4",
    "google": "gemini-2.5-pro",
}

SYSTEM_PROMPT = """\
You are an expert in computer vision, visual perception, and ontology design. You are performing the final cleaning pass on a visual concept vocabulary. The entire concept pool is provided below — your job is to identify and resolve redundancies, duplicates, and remaining quality issues that can only be detected when looking at the pool as a whole.

Return structured JSON. No explanation outside the JSON.
"""

PROMPT_TEMPLATE = """\
# Task

Below is the complete filtered visual concept pool, organized by visual category. Each category has already been through per-concept quality filtering (visual groundability, semantic level, atomicity, etc.). What remains are **pool-level** problems that require seeing multiple concepts together to detect:

- Spelling and hyphenation variants of the same concept
- Singular/plural duplicates
- Morphological variants (same root, different inflection)
- Near-synonyms (different words, visually indistinguishable)
- Concepts that decompose into other concepts IN THIS POOL
- Concepts that don't belong in their assigned category

Your job: produce a cleaned version of the pool by identifying every problematic cluster and resolving it.

---

# Rules

## R1 — Spelling / hyphenation variants
If the same concept appears with different spacing or hyphenation, keep exactly ONE canonical form (the most common English spelling, no hyphens when avoidable).

Examples: "band aid" / "band-aid" / "bandaid" → keep "bandaid". "cell phone" / "cellphone" → keep "cellphone". "t shirt" / "t-shirt" → keep "t-shirt".

## R2 — Singular / plural
Keep only the SINGULAR form. The vocabulary describes concept types, not quantities — plurality is a modifier, not a separate concept.

Examples: "cloud" / "clouds" → keep "cloud". "egg" / "eggs" → keep "egg". "biker" / "bikers" → keep "biker".

Exception: concepts that are inherently plural (the singular form is rarely used or means something different): "glasses" (eyewear), "scissors", "pants", "binoculars". Keep these as-is.

## R3 — Morphological variants
When multiple inflected forms of the same root appear, keep only the BASE form that best functions as a standalone concept label:

- For colors: keep the base adjective. "gray" / "graying" / "grayish" → keep "gray". "black" / "blackened" / "blackish" → keep "black". "yellow" / "yellowed" / "yellowish" → keep "yellow".
- For states: keep the form that names the observable state. "fade" / "faded" / "fading" → keep "faded" (the visible state). "shade" / "shaded" / "shading" / "shadowed" / "shadowing" / "shadowy" / "shady" → keep "shadowed" (the visible state in an image).
- For textures: keep the adjective form. "rust" / "rusted" / "rusty" → keep "rusty".

## R4 — Near-synonyms (visually indistinguishable)
If two concepts would produce indistinguishable photographs in most contexts, merge them. Keep the more common / general term.

Examples:
- "path" / "pathway" / "footpath" → keep "path"
- "shore" / "shoreline" / "coastline" / "coast" → keep "shore"
- "bathroom" / "restroom" / "washroom" → keep "bathroom"
- "couch" / "sofa" → keep "couch"
- "grey" / "gray" → keep "gray"
- "monochromatic" / "monochrome" → keep "monochrome"

Do NOT merge concepts that share a word but look different: "road" and "street" are both kept. "creek" and "river" are both kept.

## R5 — Reference-dependent composability
Now that you can see the full pool, check: is this concept expressible as a simple combination of 2-3 OTHER concepts that are IN THIS POOL?

The most important case: **person + role/activity compounds**. If both "person" and "running" are in the pool, then "runner" = person + running → remove "runner".

BUT keep role concepts that are visually distinctive through costume/equipment that ISN'T captured by the activity alone: "clown", "bride", "cowboy", "soldier", "firefighter".

Also check for object compounds: "alarm clock" = alarm + clock? No — "alarm" isn't the visual part. Keep it.

## R6 — Category misassignment
Flag concepts that are in the wrong category, or that don't cleanly belong to ANY category.

## R7 — Redundant age/gender person splits
Check the person-related concepts and remove aggregation terms like "people", "crowd", "group", "couple" if they are just person + quantity.

---

# The full concept pool

{POOL}

---

# Output format

Return a JSON object with two sections:

```json
{{
  "actions": [
    {{
      "action": "merge",
      "rule": "R1",
      "cluster": ["band aid", "band-aid", "bandaid"],
      "keep": "bandaid",
      "remove": ["band aid", "band-aid"]
    }},
    {{
      "action": "remove",
      "rule": "R6",
      "concepts": ["backlit", "dim", "gloomy"],
      "reason": "lighting conditions, not colors"
    }},
    {{
      "action": "recategorize",
      "rule": "R6",
      "concept": "shadow",
      "from": "object",
      "to": "edge",
      "reason": "shadow is a visual phenomenon affecting edges/boundaries"
    }}
  ],
  "summary": {{
    "total_input": 0,
    "merges": 0,
    "removals": 0,
    "recategorizations": 0,
    "total_output": 0
  }}
}}
```

Be thorough. Scan every concept in every category. The goal is to reduce the pool to a truly minimal, non-redundant set where every concept earns its place and no two concepts are visually interchangeable.

Output ONLY the JSON object.
"""


def load_concept_pool(input_dir: Path) -> dict[str, list[str]]:
    pool = {}
    for path in sorted(input_dir.glob("*.txt")):
        concepts = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                concept = line.strip()
                if concept:
                    concepts.append(concept)
        pool[path.stem] = concepts
    return pool


def format_pool(pool: dict[str, list[str]]) -> str:
    sections = []
    for category in sorted(pool.keys()):
        sections.append(f"## {category}")
        sections.extend(f"- {concept}" for concept in pool[category])
        sections.append("")
    return "\n".join(sections).strip()


def parse_json_object(text: str) -> Optional[dict]:
    text = (text or "").strip()
    if not text:
        return None
    if text.startswith("```"):
        lines = text.split("\n")
        start = 1 if lines[0].startswith("```") else 0
        end = -1 if lines[-1].strip() == "```" else len(lines)
        text = "\n".join(lines[start:end])

    first = text.find("{")
    last = text.rfind("}")
    if first == -1 or last == -1:
        return None

    candidate = text[first:last + 1]
    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        return None
    return None


def call_openai(system: str, user: str, model: str, api_key: str) -> tuple[Optional[str], Optional[Any]]:
    client = OpenAI(api_key=api_key)
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            text = response.choices[0].message.content if response.choices else None
            return text, response
        except Exception as e:
            log.error(f"OpenAI request failed: {e}")
            if attempt < 2:
                time.sleep(2 ** (attempt + 1))
    return None, None


def call_anthropic(system: str, user: str, model: str, api_key: str) -> tuple[Optional[str], Optional[Any]]:
    client = Anthropic(api_key=api_key)
    for attempt in range(3):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=8000,
                temperature=0.0,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            text = "".join(
                block.text for block in getattr(response, "content", [])
                if getattr(block, "type", None) == "text"
            )
            return text, response
        except Exception as e:
            log.error(f"Anthropic request failed: {e}")
            if attempt < 2:
                time.sleep(2 ** (attempt + 1))
    return None, None


def call_google(system: str, user: str, model: str, api_key: str) -> tuple[Optional[str], Optional[Any]]:
    client = genai.Client(api_key=api_key)
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=model,
                contents=user,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    temperature=0.0,
                    max_output_tokens=8000,
                ),
            )
            return getattr(response, "text", None), response
        except Exception as e:
            log.error(f"Google request failed: {e}")
            if attempt < 2:
                time.sleep(2 ** (attempt + 1))
    return None, None


def apply_actions(pool: dict[str, list[str]], actions: list[dict]) -> dict[str, list[str]]:
    category_sets = {category: set(concepts) for category, concepts in pool.items()}

    def remove_from_all(concept: str) -> None:
        for concepts in category_sets.values():
            concepts.discard(concept)

    def find_category(concept: str) -> Optional[str]:
        for category, concepts in category_sets.items():
            if concept in concepts:
                return category
        return None

    for action in actions:
        action_type = action.get("action")
        if action_type == "merge":
            keep = action.get("keep")
            remove = action.get("remove", [])
            cluster = action.get("cluster", [])
            target_category = find_category(keep) if keep else None
            if not target_category:
                for concept in cluster:
                    target_category = find_category(concept)
                    if target_category:
                        break
            if keep and target_category:
                category_sets[target_category].add(keep)
            for concept in remove:
                if concept != keep:
                    remove_from_all(concept)
        elif action_type == "remove":
            for concept in action.get("concepts", []):
                remove_from_all(concept)
        elif action_type == "recategorize":
            concept = action.get("concept")
            source = action.get("from")
            target = action.get("to")
            if not concept or not target:
                continue
            if source in category_sets:
                category_sets[source].discard(concept)
            else:
                remove_from_all(concept)
            category_sets.setdefault(target, set()).add(concept)

    return {
        category: sorted(concepts)
        for category, concepts in sorted(category_sets.items())
    }


def write_pool(output_dir: Path, pool: dict[str, list[str]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for category, concepts in pool.items():
        with open(output_dir / f"{category}.txt", "w", encoding="utf-8") as f:
            if concepts:
                f.write("\n".join(concepts) + "\n")
            else:
                f.write("")


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-based final concept pool cleaning")
    parser.add_argument("--input-dir", type=Path, default=Path("output/final_concepts"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/final_concepts_cleaned"))
    parser.add_argument("--provider", choices=["anthropic", "openai", "google"], default="openai")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--actions-path", type=Path, default=Path("output/final_cleaning_actions.json"))
    parser.add_argument("--raw-response-path", type=Path, default=Path("output/final_cleaning_raw_response.txt"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    api_key = args.api_key
    if not api_key:
        if args.provider == "anthropic":
            api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        elif args.provider == "openai":
            api_key = os.environ.get("OPENAI_API_KEY", "")
        else:
            api_key = os.environ.get("GOOGLE_API_KEY", "") or os.environ.get("GEMINI_API_KEY", "")

    if not api_key and not args.dry_run:
        log.error("No API key provided. Use --api-key or set the provider env var.")
        sys.exit(1)

    model = args.model or DEFAULT_MODELS[args.provider]
    pool = load_concept_pool(args.input_dir)
    total_input = sum(len(v) for v in pool.values())
    log.info(f"Loaded {total_input} concepts from {args.input_dir}")

    prompt = PROMPT_TEMPLATE.format(POOL=format_pool(pool))

    if args.dry_run:
        print("SYSTEM:\n")
        print(SYSTEM_PROMPT)
        print("\nUSER:\n")
        print(prompt)
        return

    if args.provider == "openai":
        response_text, _ = call_openai(SYSTEM_PROMPT, prompt, model, api_key)
    elif args.provider == "anthropic":
        response_text, _ = call_anthropic(SYSTEM_PROMPT, prompt, model, api_key)
    else:
        response_text, _ = call_google(SYSTEM_PROMPT, prompt, model, api_key)

    if not response_text:
        log.error("No response from provider")
        sys.exit(1)

    args.raw_response_path.parent.mkdir(parents=True, exist_ok=True)
    args.raw_response_path.write_text(response_text, encoding="utf-8")

    parsed = parse_json_object(response_text)
    if not parsed:
        log.error("Failed to parse cleaning response as JSON object")
        sys.exit(1)

    args.actions_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.actions_path, "w", encoding="utf-8") as f:
        json.dump(parsed, f, indent=2, ensure_ascii=False)

    cleaned_pool = apply_actions(pool, parsed.get("actions", []))
    write_pool(args.output_dir, cleaned_pool)

    total_output = sum(len(v) for v in cleaned_pool.values())
    log.info(f"Saved actions        → {args.actions_path}")
    log.info(f"Saved raw response   → {args.raw_response_path}")
    log.info(f"Saved cleaned pool   → {args.output_dir}")
    log.info(f"Final size: {total_input} -> {total_output}")


if __name__ == "__main__":
    main()
