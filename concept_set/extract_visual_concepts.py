#!/usr/bin/env python3
"""
Visual Concept Set Extractor
=============================
Extracts, normalizes, and prunes visual concepts from major CV datasets
to construct a minimal yet comprehensive concept vocabulary.

Usage:
    python extract_visual_concepts.py --data-root ./data --output ./output

Each dataset must be downloaded separately (see download_datasets.sh).
The script gracefully skips any missing dataset and reports what it found.

Requirements:
    pip install nltk spacy tqdm requests
    python -m spacy download en_core_web_sm
    python -m nltk.downloader wordnet omw-1.4
"""

import argparse
import collections
import csv
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Optional imports (degrade gracefully)
# ---------------------------------------------------------------------------
try:
    from nltk.corpus import wordnet as wn
    import nltk
    HAS_WORDNET = True
except ImportError:
    HAS_WORDNET = False

try:
    import spacy
    HAS_SPACY = True
except ImportError:
    HAS_SPACY = False

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):
        return it

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("concept_extractor")


# =========================================================================
# UTILITIES
# =========================================================================

def normalize(term: str) -> str:
    """Lowercase, strip, collapse whitespace, remove trailing punctuation."""
    t = term.lower().strip()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[.,;:!?]+$", "", t)
    return t


def load_json(path: Path) -> Optional[dict | list]:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> Optional[list]:
    if not path.exists():
        return None
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class ConceptPool:
    """Accumulates raw concepts per axis with provenance tracking."""

    AXES = [
        "objects", "attributes", "relations",
        "actions", "scenes", "materials", "parts",
    ]

    def __init__(self):
        # axis -> {normalized_term -> {"sources": set, "count": int, "raw_forms": set}}
        self.data: Dict[str, Dict[str, dict]] = {
            ax: {} for ax in self.AXES
        }

    def add(self, axis: str, term: str, source: str, count: int = 1):
        assert axis in self.AXES, f"Unknown axis: {axis}"
        t = normalize(term)
        if not t or len(t) < 2:
            return
        bucket = self.data[axis]
        if t not in bucket:
            bucket[t] = {"sources": set(), "count": 0, "raw_forms": set()}
        bucket[t]["sources"].add(source)
        bucket[t]["count"] += count
        bucket[t]["raw_forms"].add(term.strip())

    def summary(self) -> dict:
        out = {}
        for ax in self.AXES:
            out[ax] = len(self.data[ax])
        return out

    def to_serializable(self) -> dict:
        out = {}
        for ax in self.AXES:
            entries = []
            for term, info in sorted(
                self.data[ax].items(), key=lambda x: -x[1]["count"]
            ):
                entries.append({
                    "concept": term,
                    "count": info["count"],
                    "sources": sorted(info["sources"]),
                    "raw_forms": sorted(info["raw_forms"]),
                })
            out[ax] = entries
        return out


# =========================================================================
# AXIS 1: OBJECTS
# =========================================================================

def extract_lvis(data_root: Path, pool: ConceptPool):
    """Extract object categories from LVIS v1 annotations."""
    source = "LVIS"
    # Try multiple possible locations
    candidates = [
        data_root / "lvis" / "lvis_v1_val.json",
        data_root / "lvis" / "lvis_v1_train.json",
    ]
    ann = None
    for c in candidates:
        ann = load_json(c)
        if ann:
            break
    if not ann:
        log.warning(f"[{source}] Not found. Expected: {candidates[0]}")
        return

    categories = ann.get("categories", [])
    log.info(f"[{source}] Found {len(categories)} categories")
    for cat in categories:
        name = cat.get("name", "")
        # LVIS names use underscore separators
        name = name.replace("_", " ")
        synonyms = cat.get("synonyms", [])
        freq = cat.get("instance_count", cat.get("image_count", 1))
        pool.add("objects", name, source, count=freq)
        for syn in synonyms:
            pool.add("objects", syn.replace("_", " "), source, count=1)


def extract_coco_panoptic(data_root: Path, pool: ConceptPool):
    """Extract thing + stuff categories from COCO panoptic."""
    source = "COCO-Panoptic"
    candidates = [
        data_root / "coco" / "annotations" / "panoptic_val2017.json",
        data_root / "coco" / "annotations" / "panoptic_train2017.json",
        data_root / "coco" / "panoptic_coco_categories.json",
    ]
    ann = None
    for c in candidates:
        ann = load_json(c)
        if ann:
            break
    if not ann:
        log.warning(f"[{source}] Not found. Expected: {candidates[0]}")
        return

    # Categories may be top-level or nested
    categories = ann if isinstance(ann, list) else ann.get("categories", [])
    log.info(f"[{source}] Found {len(categories)} categories")
    for cat in categories:
        name = cat.get("name", "")
        is_stuff = cat.get("isthing", 1) == 0
        pool.add("objects", name, source + ("-stuff" if is_stuff else "-thing"))


def extract_visual_genome_objects(data_root: Path, pool: ConceptPool):
    """Extract object names from Visual Genome objects.json."""
    source = "VisualGenome"
    path = data_root / "visual_genome" / "objects.json"
    data = load_json(path)
    if not data:
        # Try alternate: object_synsets.json or objects.json.gz
        log.warning(f"[{source}] objects.json not found at {path}")
        return

    log.info(f"[{source}] Processing object annotations...")
    name_counts = collections.Counter()
    for img_entry in tqdm(data, desc="VG objects"):
        for obj in img_entry.get("objects", []):
            for name in obj.get("names", []):
                name_counts[name] += 1
            # Some entries use 'name' instead of 'names'
            if "name" in obj and "names" not in obj:
                name_counts[obj["name"]] += 1

    log.info(f"[{source}] Found {len(name_counts)} unique object names")
    for name, count in name_counts.items():
        if count >= 5:  # Filter noise
            pool.add("objects", name, source, count=count)


def extract_open_images(data_root: Path, pool: ConceptPool):
    """Extract boxable class names from Open Images V7."""
    source = "OpenImages"
    path = data_root / "open_images" / "class-descriptions-boxable.csv"
    if not path.exists():
        log.warning(f"[{source}] Not found at {path}")
        return

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        count = 0
        for row in reader:
            if len(row) >= 2:
                label_name = row[1]
                pool.add("objects", label_name, source)
                count += 1
    log.info(f"[{source}] Found {count} boxable classes")


def extract_paco_lvis(data_root: Path, pool: ConceptPool):
    """Extract objects, parts, and attributes from PACO-LVIS."""
    source = "PACO-LVIS"
    candidates = [
        data_root / "paco_lvis" / "paco_lvis_v1_val.json",
        data_root / "paco_lvis" / "paco_lvis_v1_train.json",
    ]
    ann = None
    for c in candidates:
        ann = load_json(c)
        if ann:
            break
    if not ann:
        log.warning(f"[{source}] Not found. Expected: {candidates[0]}")
        return

    categories = ann.get("categories", [])
    obj_count = part_count = 0
    for cat in categories:
        name = cat.get("name", "")
        supercat = cat.get("supercategory", "")
        if not name:
            continue
        if supercat == "PART":
            # Strip "object:" prefix, keep only the part name
            part_name = name.split(":", 1)[-1] if ":" in name else name
            pool.add("parts", part_name.replace("_", " "), source)
            part_count += 1
        else:
            # OBJECT categories
            freq = cat.get("instance_count", cat.get("image_count", 1))
            pool.add("objects", name.replace("_", " "), source, count=freq)
            obj_count += 1

    # Attributes (color, pattern_marking, material, transparency)
    attributes = ann.get("attributes", [])
    attr_count = 0
    for attr in attributes:
        name = attr.get("name", "")
        if not name:
            continue
        supercat = attr.get("supercategory", "")
        if supercat in ("material", "transparency"):
            pool.add("materials", name.replace("_", " "), source)
        else:
            pool.add("attributes", name.replace("_", " "), source)
        attr_count += 1

    log.info(
        f"[{source}] Found {obj_count} objects, {part_count} parts, "
        f"{attr_count} attributes"
    )


# =========================================================================
# AXIS 2: ATTRIBUTES
# =========================================================================

def extract_vaw(data_root: Path, pool: ConceptPool):
    """Extract attribute vocabulary from VAW dataset."""
    source = "VAW"
    # VAW provides attribute_index.json or the main annotation files
    candidates = [
        data_root / "vaw" / "attribute_index.json",
        data_root / "vaw" / "train_part1.json",
    ]

    # Try attribute index first (cleanest)
    attr_index = load_json(candidates[0])
    if attr_index:
        log.info(f"[{source}] Found attribute index with {len(attr_index)} attributes")
        for attr_name, idx in attr_index.items():
            pool.add("attributes", attr_name, source)
        return

    # Fall back to mining from annotation files
    for split in ["train_part1.json", "train_part2.json", "val.json", "test.json"]:
        path = data_root / "vaw" / split
        data = load_json(path)
        if not data:
            continue
        for entry in data:
            for attr in entry.get("positive_attributes", []):
                pool.add("attributes", attr, source)
            for attr in entry.get("negative_attributes", []):
                pool.add("attributes", attr, source, count=0)

    attr_count = len(pool.data["attributes"])
    if attr_count > 0:
        log.info(f"[{source}] Extracted {attr_count} unique attributes from annotations")
    else:
        log.warning(f"[{source}] Not found. Expected: {candidates[0]}")


def extract_mit_states(data_root: Path, pool: ConceptPool):
    """Extract state/attribute vocabulary from MIT-States."""
    source = "MIT-States"
    base = data_root / "mit_states"

    # MIT-States organizes by adj_noun folders or provides a metadata file
    metadata = load_json(base / "metadata.json")
    if metadata:
        for entry in metadata:
            adj = entry.get("adjective", entry.get("state", ""))
            if adj:
                pool.add("attributes", adj, source)
        log.info(f"[{source}] Extracted attributes from metadata.json")
        return

    # Fall back: parse directory structure (release2.0/images/adj_noun/)
    img_dir = base / "release_dataset" / "images"
    if not img_dir.exists():
        img_dir = base / "images"
    if img_dir.exists():
        states = set()
        for subdir in img_dir.iterdir():
            if subdir.is_dir():
                parts = subdir.name.split("_", 1)
                if len(parts) == 2:
                    state = parts[0].replace("-", " ")
                    states.add(state)
        for s in states:
            pool.add("attributes", s, source)
        log.info(f"[{source}] Parsed {len(states)} states from directory names")
    else:
        log.warning(f"[{source}] Not found at {base}")


def extract_vg_attributes(data_root: Path, pool: ConceptPool):
    """Extract attributes from Visual Genome attributes.json."""
    source = "VG-Attributes"
    path = data_root / "visual_genome" / "attributes.json"
    data = load_json(path)
    if not data:
        log.warning(f"[{source}] attributes.json not found at {path}")
        return

    attr_counts = collections.Counter()
    for img_entry in tqdm(data, desc="VG attributes"):
        for obj in img_entry.get("attributes", []):
            for attr in obj.get("attributes", []):
                attr_counts[attr] += 1

    log.info(f"[{source}] Found {len(attr_counts)} unique attributes")
    for attr, count in attr_counts.items():
        if count >= 10:  # Filter rare noise
            pool.add("attributes", attr, source, count=count)


# =========================================================================
# AXIS 3: RELATIONS
# =========================================================================

def extract_vg_relationships(data_root: Path, pool: ConceptPool):
    """Extract relationship predicates from Visual Genome."""
    source = "VG-Relations"
    path = data_root / "visual_genome" / "relationships.json"
    data = load_json(path)
    if not data:
        log.warning(f"[{source}] relationships.json not found at {path}")
        return

    rel_counts = collections.Counter()
    for img_entry in tqdm(data, desc="VG relations"):
        for rel in img_entry.get("relationships", []):
            predicate = rel.get("predicate", "")
            if predicate:
                rel_counts[predicate] += 1

    log.info(f"[{source}] Found {len(rel_counts)} unique predicates")
    for pred, count in rel_counts.items():
        if count >= 20:  # Relations are very noisy in VG; threshold higher
            pool.add("relations", pred, source, count=count)


def extract_gqa_relations(data_root: Path, pool: ConceptPool):
    """Extract relationship predicates from GQA scene graphs."""
    source = "GQA"
    sg_dir = data_root / "gqa" / "sceneGraphs"
    candidates = [
        sg_dir / "train_sceneGraphs.json",
        sg_dir / "val_sceneGraphs.json",
    ]

    rel_counts = collections.Counter()
    found = False
    for path in candidates:
        data = load_json(path)
        if not data:
            continue
        found = True
        for img_id, graph in tqdm(data.items(), desc=f"GQA {path.stem}"):
            objects = graph.get("objects", {})
            for obj_id, obj_info in objects.items():
                for rel in obj_info.get("relations", []):
                    pred = rel.get("name", "")
                    if pred:
                        rel_counts[pred] += 1

    if not found:
        log.warning(f"[{source}] Not found at {sg_dir}")
        return

    log.info(f"[{source}] Found {len(rel_counts)} unique predicates")
    for pred, count in rel_counts.items():
        if count >= 50:
            pool.add("relations", pred, source, count=count)


# =========================================================================
# AXIS 4: ACTIONS / STATES
# =========================================================================

def extract_hico_det(data_root: Path, pool: ConceptPool):
    """Extract action verbs from HICO-DET."""
    source = "HICO-DET"

    # HICO-DET provides a list_action.csv or anno.json
    csv_path = data_root / "hico_det" / "list_action.csv"
    if csv_path.exists():
        verbs = set()
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                # Format varies: (id, verb, object) or (verb, object)
                for cell in row:
                    cell = cell.strip()
                    if cell and not cell.isdigit() and len(cell) > 1:
                        # Heuristic: verbs are usually short and lowercase
                        if cell == cell.lower() and " " not in cell:
                            verbs.add(cell)
        for v in verbs:
            pool.add("actions", v, source)
        log.info(f"[{source}] Extracted {len(verbs)} verbs from CSV")
        return

    # Try JSON annotations
    json_path = data_root / "hico_det" / "anno.json"
    data = load_json(json_path)
    if not data:
        # Try the HF-style metadata
        json_path = data_root / "hico_det" / "metadata.json"
        data = load_json(json_path)

    if data:
        verbs = set()
        if isinstance(data, dict) and "categories" in data:
            for cat in data["categories"]:
                verb = cat.get("verb", cat.get("action", ""))
                if verb:
                    verbs.add(verb)
        elif isinstance(data, list):
            for entry in data:
                for hoi in entry.get("hoi_annotation", []):
                    cat_id = hoi.get("category_id", -1)
                    # Would need a category mapping; just collect what we can
                pass
        for v in verbs:
            pool.add("actions", v, source)
        log.info(f"[{source}] Extracted {len(verbs)} verbs")
    else:
        # Provide the 117 well-known HICO-DET verbs as a fallback
        _inject_hico_verbs_fallback(pool, source)


def _inject_hico_verbs_fallback(pool: ConceptPool, source: str):
    """Inject the canonical 117 HICO-DET verbs when data files are unavailable."""
    verbs = [
        "adjust", "assemble", "block", "blow", "board", "break", "brush",
        "buy", "carry", "catch", "chase", "check", "clean", "control",
        "cook", "cut", "direct", "drag", "dribble", "drink", "drive",
        "dry", "eat", "enter", "exit", "extract", "feed", "fill", "flip",
        "flush", "fly", "greet", "grind", "groom", "herd", "hit", "hold",
        "hop_on", "hose", "hug", "hunt", "inspect", "install", "jump",
        "kick", "kiss", "lasso", "launch", "lick", "lie_on", "lift",
        "light", "load", "lose", "make", "milk", "move", "no_interaction",
        "open", "operate", "pack", "paint", "park", "pay", "peel",
        "pet", "pick", "pick_up", "point", "pour", "press", "pull",
        "push", "race", "read", "release", "repair", "ride", "row",
        "run", "sail", "scratch", "serve", "set", "shear", "sign",
        "sip", "sit_at", "sit_on", "slide", "smell", "spin", "squeeze",
        "stab", "stand_on", "stand_under", "stick", "stir", "stop",
        "straddle", "swing", "tag", "talk_on", "teach", "text_on",
        "throw", "tie", "toast", "train", "turn", "type_on", "walk",
        "wash", "watch", "wave", "wear", "wield", "zip",
    ]
    for v in verbs:
        pool.add("actions", v.replace("_", " "), source)
    log.info(f"[{source}] Injected {len(verbs)} canonical verbs (fallback)")


def extract_vcoco(data_root: Path, pool: ConceptPool):
    """Extract action categories from V-COCO."""
    source = "V-COCO"
    path = data_root / "vcoco" / "vcoco_test.json"
    if not path.exists():
        path = data_root / "vcoco" / "vcoco_train.json"
    data = load_json(path)
    if not data:
        # Fallback: the 26 known V-COCO actions
        actions = [
            "hold", "stand", "sit", "ride", "walk", "look", "hit", "eat",
            "jump", "lay", "talk on phone", "carry", "throw", "catch",
            "cut", "run", "work on computer", "ski", "surf", "skateboard",
            "smile", "drink", "kick", "read", "snowboard",
        ]
        for a in actions:
            pool.add("actions", a, source)
        log.info(f"[{source}] Injected {len(actions)} canonical actions (fallback)")
        return

    actions = set()
    if isinstance(data, list):
        for entry in data:
            action = entry.get("action_name", entry.get("action", ""))
            if action:
                actions.add(action)
    for a in actions:
        pool.add("actions", a, source)
    log.info(f"[{source}] Extracted {len(actions)} actions")


def extract_imsitu(data_root: Path, pool: ConceptPool):
    """Extract verb frames from imSitu."""
    source = "imSitu"
    path = data_root / "imsitu" / "imsitu_space.json"
    data = load_json(path)
    if not data:
        log.warning(f"[{source}] Not found at {path}")
        return

    verbs = data.get("verbs", {})
    log.info(f"[{source}] Found {len(verbs)} verb frames")
    for verb_name, verb_info in verbs.items():
        # imSitu verb names are FrameNet-style (e.g., "galloping")
        pool.add("actions", verb_name.replace("_", " "), source)
        # Also extract the abstract (human-readable description)
        abstract = verb_info.get("abstract", "")
        # The roles give us semantic frame info — useful metadata
        # but not directly concepts for the pool


def extract_ava_actions(data_root: Path, pool: ConceptPool):
    """Extract atomic actions from AVA dataset."""
    source = "AVA"
    path = data_root / "ava" / "ava_action_list_v2.2.pbtxt"
    if not path.exists():
        path = data_root / "ava" / "ava_action_list.csv"

    if path.exists() and path.suffix == ".csv":
        with open(path, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    pool.add("actions", row[1], source)
        log.info(f"[{source}] Extracted actions from CSV")
    elif path.exists():
        # Parse the protobuf text format
        with open(path, "r") as f:
            text = f.read()
        names = re.findall(r'name:\s*"([^"]+)"', text)
        for name in names:
            pool.add("actions", name, source)
        log.info(f"[{source}] Extracted {len(names)} actions from pbtxt")
    else:
        # Fallback: 80 well-known AVA atomic actions
        log.warning(f"[{source}] Not found at {path}")


def extract_kinetics_actions(data_root: Path, pool: ConceptPool):
    """Extract action class names from Kinetics-700."""
    source = "Kinetics-700"
    # Kinetics provides a CSV with video_id, label, split etc.
    # Or a class list file
    candidates = [
        data_root / "kinetics" / "kinetics_700_labels.csv",
        data_root / "kinetics" / "labels.csv",
        data_root / "kinetics" / "kinetics700.json",
    ]
    for path in candidates:
        if path.exists() and path.suffix == ".csv":
            labels = set()
            with open(path, "r") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                label_col = 0
                if header:
                    for i, h in enumerate(header):
                        if "label" in h.lower():
                            label_col = i
                            break
                for row in reader:
                    if len(row) > label_col:
                        labels.add(row[label_col])
            for label in labels:
                pool.add("actions", label, source)
            log.info(f"[{source}] Extracted {len(labels)} action classes")
            return
        elif path.exists() and path.suffix == ".json":
            data = load_json(path)
            if isinstance(data, dict):
                labels = set()
                for v in data.values():
                    if isinstance(v, dict) and "label" in v:
                        labels.add(v["label"])
                for label in labels:
                    pool.add("actions", label, source)
                log.info(f"[{source}] Extracted {len(labels)} action classes")
                return

    log.warning(f"[{source}] Not found. Expected: {candidates[0]}")


# =========================================================================
# AXIS 5: SCENES / CONTEXT
# =========================================================================

def extract_places365(data_root: Path, pool: ConceptPool):
    """Extract scene categories from Places365."""
    source = "Places365"
    candidates = [
        data_root / "places365" / "categories_places365.txt",
        data_root / "places365" / "IO_places365.txt",
    ]
    for path in candidates:
        if not path.exists():
            continue
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Format: /a/airfield 0  or  /a/airfield
                parts = line.split()
                category = parts[0]
                # Strip path prefix: /a/airfield -> airfield
                name = category.split("/")[-1]
                name = name.replace("_", " ")
                pool.add("scenes", name, source)
        count = len(pool.data["scenes"])
        log.info(f"[{source}] Extracted {count} scene categories")
        return

    log.warning(f"[{source}] Not found. Expected: {candidates[0]}")


def extract_sun397(data_root: Path, pool: ConceptPool):
    """Extract scene categories and attributes from SUN397."""
    source = "SUN397"
    # Scene attributes
    attr_path = data_root / "sun397" / "SUNAttributeDB" / "attributes.mat"
    cat_path = data_root / "sun397" / "ClassName.txt"

    if cat_path.exists():
        with open(cat_path, "r") as f:
            for line in f:
                name = line.strip().split("/")[-1].replace("_", " ")
                if name:
                    pool.add("scenes", name, source)
        log.info(f"[{source}] Extracted scene categories from ClassName.txt")
    else:
        log.warning(f"[{source}] Not found at {cat_path}")

    # SUN attributes (102 scene attributes) — try text file
    attr_list = data_root / "sun397" / "SUNAttributeDB" / "attributeLabels_continuous.txt"
    attr_names = data_root / "sun397" / "SUNAttributeDB" / "attributes.txt"
    if attr_names.exists():
        with open(attr_names, "r") as f:
            for line in f:
                attr = line.strip()
                if attr:
                    pool.add("attributes", attr, source + "-scene-attr")
        log.info(f"[{source}] Extracted scene attributes")


def extract_ade20k_scenes(data_root: Path, pool: ConceptPool):
    """Extract scene-level labels from ADE20K."""
    source = "ADE20K"
    path = data_root / "ade20k" / "sceneCategories.txt"
    if not path.exists():
        path = data_root / "ade20k" / "objectInfo150.txt"

    if path.exists() and "scene" in path.name.lower():
        with open(path, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    scene = parts[1].replace("_", " ")
                    pool.add("scenes", scene, source)
        log.info(f"[{source}] Extracted scene categories")
    elif path.exists():
        # objectInfo150: extract object categories from ADE20K
        with open(path, "r") as f:
            header = next(f, "")
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 5:
                    name = parts[4].replace(",", " ").strip()  # Name field
                    pool.add("objects", name, source)
        log.info(f"[{source}] Extracted object categories")
    else:
        log.warning(f"[{source}] Not found")


# =========================================================================
# AXIS 6: MATERIALS / TEXTURES
# =========================================================================

def extract_dtd(data_root: Path, pool: ConceptPool):
    """Extract 47 texture attributes from DTD."""
    source = "DTD"
    base = data_root / "dtd" / "images"
    if not base.exists():
        base = data_root / "dtd"

    if base.exists():
        textures = set()
        for item in base.iterdir():
            if item.is_dir() and not item.name.startswith("."):
                textures.add(item.name)
        if textures:
            for t in textures:
                pool.add("materials", t, source + "-texture")
            log.info(f"[{source}] Extracted {len(textures)} texture classes")
            return

    # Fallback: canonical 47 DTD texture attributes
    dtd_textures = [
        "banded", "blotchy", "braided", "bubbly", "bumpy", "chequered",
        "cobwebbed", "cracked", "crosshatched", "crystalline", "dotted",
        "fibrous", "flecked", "freckled", "frilly", "gauzy", "grid",
        "grooved", "honeycombed", "interlaced", "knitted", "lacelike",
        "lined", "marbled", "matted", "meshed", "paisley", "perforated",
        "pitted", "pleated", "polka-dotted", "porous", "potholed",
        "scaly", "smeared", "spiralled", "sprinkled", "stained",
        "stratified", "striped", "studded", "swirly", "veined", "waffled",
        "woven", "wrinkled", "zigzagged",
    ]
    for t in dtd_textures:
        pool.add("materials", t, source + "-texture")
    log.info(f"[{source}] Injected {len(dtd_textures)} canonical texture attributes (fallback)")


def extract_fmd(data_root: Path, pool: ConceptPool):
    """Extract 10 material categories from FMD."""
    source = "FMD"
    base = data_root / "fmd" / "image"
    if base.exists():
        materials = set()
        for item in base.iterdir():
            if item.is_dir():
                materials.add(item.name)
        for m in materials:
            pool.add("materials", m, source)
        log.info(f"[{source}] Extracted {len(materials)} material classes")
        return

    # Fallback: the 10 canonical FMD materials
    fmd_materials = [
        "fabric", "foliage", "glass", "leather", "metal",
        "paper", "plastic", "stone", "water", "wood",
    ]
    for m in fmd_materials:
        pool.add("materials", m, source)
    log.info(f"[{source}] Injected {len(fmd_materials)} canonical materials (fallback)")


def extract_opensurfaces(data_root: Path, pool: ConceptPool):
    """Extract material labels from OpenSurfaces."""
    source = "OpenSurfaces"
    path = data_root / "opensurfaces" / "materials.json"
    data = load_json(path)
    if not data:
        path = data_root / "opensurfaces" / "photos.json"
        data = load_json(path)
    if not data:
        log.warning(f"[{source}] Not found")
        return

    materials = set()
    if isinstance(data, list):
        for entry in data:
            mat = entry.get("substance_name", entry.get("material", ""))
            if mat:
                materials.add(mat)
    for m in materials:
        pool.add("materials", m, source)
    log.info(f"[{source}] Extracted {len(materials)} materials")


# =========================================================================
# AXIS 7: PARTS
# =========================================================================

def extract_partimagenet(data_root: Path, pool: ConceptPool):
    """Extract part labels from PartImageNet."""
    source = "PartImageNet"
    candidates = [
        data_root / "partimagenet" / "train.json",
        data_root / "partimagenet" / "val.json",
    ]
    for path in candidates:
        data = load_json(path)
        if not data:
            continue
        categories = data.get("categories", [])
        for cat in categories:
            name = cat.get("name", "")
            if name:
                pool.add("parts", name.replace("_", " "), source)
        log.info(f"[{source}] Extracted {len(categories)} part categories")
        return

    log.warning(f"[{source}] Not found. Expected: {candidates[0]}")


def extract_pascal_parts(data_root: Path, pool: ConceptPool):
    """Extract part labels from PASCAL-Part."""
    source = "PASCAL-Part"
    # PASCAL-Part uses .mat annotations or converted JSON
    path = data_root / "pascal_part" / "classes.txt"
    if path.exists():
        with open(path, "r") as f:
            for line in f:
                name = line.strip()
                if name:
                    pool.add("parts", name, source)
        log.info(f"[{source}] Extracted part classes from classes.txt")
        return

    # Try JSON annotation
    path = data_root / "pascal_part" / "pascal_part.json"
    data = load_json(path)
    if data:
        parts = set()
        categories = data.get("categories", [])
        for cat in categories:
            name = cat.get("name", "")
            if name:
                parts.add(name)
        for p in parts:
            pool.add("parts", p, source)
        log.info(f"[{source}] Extracted {len(parts)} part categories")
    else:
        log.warning(f"[{source}] Not found at {data_root / 'pascal_part'}")


def extract_ade20k_parts(data_root: Path, pool: ConceptPool):
    """Extract part annotations from ADE20K."""
    source = "ADE20K-Parts"
    # ADE20K parts are in the index file
    path = data_root / "ade20k" / "index_ade20k.pkl"
    if path.exists():
        import pickle
        with open(path, "rb") as f:
            index = pickle.load(f)
        # The index contains objectnames, which include parts
        parts = set()
        for name in index.get("objectnames", []):
            # ADE20K marks parts with a hierarchy indicator
            pool.add("parts", name.replace(",", "").strip(), source)
            parts.add(name)
        log.info(f"[{source}] Extracted {len(parts)} object/part names")
        return

    path = data_root / "ade20k" / "objectInfo150.txt"
    if path.exists():
        with open(path, "r") as f:
            next(f, "")  # skip header
            for line in f:
                parts_f = line.strip().split("\t")
                if len(parts_f) >= 5:
                    name = parts_f[4].strip()
                    pool.add("parts", name, source)
        log.info(f"[{source}] Extracted from objectInfo150.txt")
    else:
        log.warning(f"[{source}] Not found")


# =========================================================================
# CROSS-AXIS: CAPTION MINING
# =========================================================================

def extract_from_captions(data_root: Path, pool: ConceptPool):
    """
    Mine nouns, adjectives, verbs, and prepositions from caption datasets
    to discover concepts that structured annotations missed.
    Requires spaCy for POS tagging.
    """
    if not HAS_SPACY:
        log.warning("[Captions] spaCy not installed — skipping caption mining")
        return

    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    nlp.add_pipe("sentencizer")

    caption_files = {
        "COCO-Captions": [
            data_root / "coco" / "annotations" / "captions_val2017.json",
            data_root / "coco" / "annotations" / "captions_train2017.json",
        ],
    }

    pos_axis_map = {
        "NOUN": "objects",
        "PROPN": "objects",
        "ADJ": "attributes",
        "VERB": "actions",
        "ADP": "relations",
    }

    for source, paths in caption_files.items():
        for path in paths:
            data = load_json(path)
            if not data:
                continue

            annotations = data.get("annotations", [])
            log.info(f"[{source}] Processing {len(annotations)} captions from {path.name}...")

            counts = collections.Counter()
            for ann in tqdm(annotations, desc=source):
                caption = ann.get("caption", "")
                doc = nlp(caption)
                for token in doc:
                    if token.is_stop or token.is_punct or len(token.text) < 3:
                        continue
                    pos = token.pos_
                    if pos in pos_axis_map:
                        lemma = token.lemma_.lower()
                        axis = pos_axis_map[pos]
                        counts[(axis, lemma)] += 1

            # Only add high-frequency terms not already in the pool
            for (axis, lemma), count in counts.items():
                if count >= 50:  # Robust frequency threshold
                    pool.add(axis, lemma, source + "-mined", count=count)

            log.info(f"[{source}] Mined {len(counts)} unique (axis, term) pairs")

    # Conceptual Captions — TSV format
    for cc_name, cc_file in [
        ("CC-3M", data_root / "conceptual_captions" / "Train_GCC-training.tsv"),
        ("CC-3M-val", data_root / "conceptual_captions" / "Validation_GCC-1.1.0-Validation.tsv"),
    ]:
        if not cc_file.exists():
            continue
        log.info(f"[{cc_name}] Processing...")
        counts = collections.Counter()
        with open(cc_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            for i, row in enumerate(tqdm(reader, desc=cc_name)):
                if i > 500_000:  # Cap for memory
                    break
                if len(row) < 1:
                    continue
                caption = row[0]
                doc = nlp(caption)
                for token in doc:
                    if token.is_stop or token.is_punct or len(token.text) < 3:
                        continue
                    pos = token.pos_
                    if pos in pos_axis_map:
                        lemma = token.lemma_.lower()
                        axis = pos_axis_map[pos]
                        counts[(axis, lemma)] += 1

        for (axis, lemma), count in counts.items():
            if count >= 100:
                pool.add(axis, lemma, cc_name + "-mined", count=count)
        log.info(f"[{cc_name}] Mined {len(counts)} unique (axis, term) pairs")


# =========================================================================
# POST-PROCESSING: WORDNET SYNONYM MERGING
# =========================================================================

def merge_synonyms_wordnet(pool: ConceptPool) -> Dict[str, Dict[str, str]]:
    """
    Use WordNet to merge synonyms within each axis.
    Returns a mapping: axis -> {term -> canonical_term}.
    """
    if not HAS_WORDNET:
        log.warning("[WordNet] nltk not installed — skipping synonym merging")
        return {}

    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)

    merge_map = {}
    for axis in pool.AXES:
        terms = list(pool.data[axis].keys())
        if not terms:
            continue

        # Build synset clusters
        synset_to_terms: Dict[str, list] = collections.defaultdict(list)
        term_to_synsets: Dict[str, list] = {}

        for term in terms:
            # Look up in WordNet
            query = term.replace(" ", "_")
            synsets = wn.synsets(query)
            if synsets:
                # Take the most common synset
                best = synsets[0]
                synset_to_terms[best.name()].append(term)
                term_to_synsets[term] = [s.name() for s in synsets[:3]]

        # For each cluster of terms sharing a synset, pick the canonical one
        # (highest count)
        axis_merges = {}
        for synset_name, cluster_terms in synset_to_terms.items():
            if len(cluster_terms) <= 1:
                continue
            # Pick the term with highest count as canonical
            canonical = max(
                cluster_terms,
                key=lambda t: pool.data[axis][t]["count"]
            )
            for t in cluster_terms:
                if t != canonical:
                    axis_merges[t] = canonical
                    # Transfer counts and sources
                    pool.data[axis][canonical]["count"] += pool.data[axis][t]["count"]
                    pool.data[axis][canonical]["sources"].update(pool.data[axis][t]["sources"])
                    pool.data[axis][canonical]["raw_forms"].update(pool.data[axis][t]["raw_forms"])

        # Remove merged terms
        for t in axis_merges:
            del pool.data[axis][t]

        merge_map[axis] = axis_merges
        if axis_merges:
            log.info(f"[WordNet] {axis}: merged {len(axis_merges)} synonyms")

    return merge_map


# =========================================================================
# POST-PROCESSING: CONCEPT CLEANING FILTERS
# =========================================================================

_ARTICLE_RE = re.compile(r"^(a|an|the)\s+", re.IGNORECASE)


def filter_numeric(pool: ConceptPool):
    """Remove concepts that are purely numeric (e.g. '10', '2009', '1:50')."""
    for axis in pool.AXES:
        to_remove = [
            t for t in pool.data[axis]
            if re.fullmatch(r"[\d\s:.,/\-]+", t)
        ]
        for t in to_remove:
            del pool.data[axis][t]
        if to_remove:
            log.info(f"[filter_numeric] {axis}: removed {len(to_remove)} numeric-only entries")


def filter_article_prefixed(pool: ConceptPool):
    """Remove concepts that start with an article (a/an/the) in objects and attributes."""
    for axis in ["objects", "attributes"]:
        to_remove = [
            t for t in pool.data[axis]
            if _ARTICLE_RE.match(t)
        ]
        for t in to_remove:
            del pool.data[axis][t]
        if to_remove:
            log.info(
                f"[filter_article_prefixed] {axis}: removed {len(to_remove)} "
                f"article-prefixed entries"
            )


def filter_nouns_in_actions(pool: ConceptPool):
    """Remove actions that are actually nouns (not verbs) according to WordNet."""
    if not HAS_WORDNET:
        log.warning("[filter_nouns_in_actions] nltk not installed — skipping")
        return

    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)

    to_remove = []
    for term in pool.data["actions"]:
        query = term.replace(" ", "_")
        synsets = wn.synsets(query)
        if not synsets:
            continue
        # If all synsets are nouns (no verb sense at all), it's misclassified
        pos_set = {s.pos() for s in synsets}
        if "v" not in pos_set:
            to_remove.append(term)

    for t in to_remove:
        del pool.data["actions"][t]
    if to_remove:
        log.info(
            f"[filter_nouns_in_actions] actions: removed {len(to_remove)} "
            f"noun-only entries"
        )


# =========================================================================
# POST-PROCESSING: SUBSUMPTION PRUNING
# =========================================================================

def prune_subsumption(pool: ConceptPool, min_count: int = 50):
    """
    Remove concepts that are strict hypernyms of more specific concepts,
    unless the hypernym is very frequent.
    """
    if not HAS_WORDNET:
        log.warning("[Subsumption] nltk not installed — skipping")
        return

    for axis in ["objects"]:  # Most relevant for objects
        terms = list(pool.data[axis].keys())
        to_remove = set()

        for term in terms:
            query = term.replace(" ", "_")
            synsets = wn.synsets(query)
            if not synsets:
                continue
            # Check if any hypernym of this term is also in the pool
            for ss in synsets[:2]:
                for hyper in ss.hypernyms():
                    hyper_lemmas = [l.name().replace("_", " ") for l in hyper.lemmas()]
                    for hl in hyper_lemmas:
                        hl_norm = normalize(hl)
                        if hl_norm in pool.data[axis] and hl_norm != normalize(term):
                            hyper_count = pool.data[axis][hl_norm]["count"]
                            child_count = pool.data[axis][normalize(term)]["count"]
                            # Remove the hypernym if the child is frequent enough
                            if child_count >= min_count and hyper_count < child_count * 5:
                                to_remove.add(hl_norm)

        for t in to_remove:
            if t in pool.data[axis]:
                del pool.data[axis][t]

        if to_remove:
            log.info(f"[Subsumption] {axis}: pruned {len(to_remove)} hypernyms")


# =========================================================================
# MAIN
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Extract visual concepts from major CV datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Expected data directory structure:
  data/
    lvis/            lvis_v1_val.json
    paco_lvis/       paco_lvis_v1_val.json
    coco/annotations/  panoptic_val2017.json, captions_val2017.json
    visual_genome/   objects.json, attributes.json, relationships.json
    open_images/     class-descriptions-boxable.csv
    vaw/             attribute_index.json or train_part1.json
    mit_states/      images/ or metadata.json
    gqa/sceneGraphs/ train_sceneGraphs.json
    hico_det/        list_action.csv or anno.json
    vcoco/           vcoco_test.json
    imsitu/          imsitu_space.json
    kinetics/        kinetics_700_labels.csv
    ava/             ava_action_list_v2.2.pbtxt
    places365/       categories_places365.txt
    sun397/          ClassName.txt, SUNAttributeDB/
    ade20k/          objectInfo150.txt, sceneCategories.txt
    dtd/images/      (47 subdirectories)
    fmd/image/       (10 subdirectories)
    opensurfaces/    materials.json
    partimagenet/    train.json
    pascal_part/     classes.txt or pascal_part.json
    conceptual_captions/ Train_GCC-training.tsv
        """,
    )
    parser.add_argument(
        "--data-root", type=Path, default=Path("./data"),
        help="Root directory containing all dataset folders",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("./output"),
        help="Directory for output files",
    )
    parser.add_argument(
        "--skip-captions", action="store_true",
        help="Skip caption mining (saves time if spaCy is slow)",
    )
    parser.add_argument(
        "--skip-wordnet", action="store_true",
        help="Skip WordNet synonym merging",
    )
    parser.add_argument(
        "--min-count", type=int, default=5,
        help="Minimum occurrence count to keep a concept (default: 5)",
    )
    parser.add_argument(
        "--filter-numeric", action="store_true",
        help="Remove concepts that are purely numeric",
    )
    parser.add_argument(
        "--filter-article-prefixed", action="store_true",
        help="Remove object/attribute concepts starting with a/an/the",
    )
    parser.add_argument(
        "--filter-nouns-in-actions", action="store_true",
        help="Remove actions that have no verb sense in WordNet",
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    pool = ConceptPool()
    data = args.data_root

    log.info("=" * 60)
    log.info("PHASE 1: Extract raw concepts from each source")
    log.info("=" * 60)

    # Objects (+ parts/attributes from PACO-LVIS)
    extract_lvis(data, pool)
    extract_coco_panoptic(data, pool)
    extract_visual_genome_objects(data, pool)
    extract_open_images(data, pool)
    extract_paco_lvis(data, pool)

    # Attributes
    extract_vaw(data, pool)
    extract_mit_states(data, pool)
    extract_vg_attributes(data, pool)

    # Relations
    extract_vg_relationships(data, pool)
    extract_gqa_relations(data, pool)

    # Actions
    extract_hico_det(data, pool)
    extract_vcoco(data, pool)
    extract_imsitu(data, pool)
    extract_ava_actions(data, pool)
    extract_kinetics_actions(data, pool)

    # Scenes
    extract_places365(data, pool)
    extract_sun397(data, pool)
    extract_ade20k_scenes(data, pool)

    # Materials / Textures
    extract_dtd(data, pool)
    extract_fmd(data, pool)
    extract_opensurfaces(data, pool)

    # Parts
    extract_partimagenet(data, pool)
    extract_pascal_parts(data, pool)
    extract_ade20k_parts(data, pool)

    # Caption mining
    if not args.skip_captions:
        extract_from_captions(data, pool)

    log.info("")
    log.info("=" * 60)
    log.info("PHASE 1 COMPLETE — Raw concept pool")
    log.info("=" * 60)
    for axis, count in pool.summary().items():
        log.info(f"  {axis:12s}: {count:>6,d} unique terms")

    # Save raw pool before merging
    raw_path = args.output / "01_raw_concept_pool.json"
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(pool.to_serializable(), f, indent=2, ensure_ascii=False)
    log.info(f"\nSaved raw pool → {raw_path}")

    # ----- Phase 2: Synonym Merging -----
    log.info("")
    log.info("=" * 60)
    log.info("PHASE 2: WordNet synonym merging")
    log.info("=" * 60)

    if not args.skip_wordnet:
        merge_map = merge_synonyms_wordnet(pool)
        if merge_map:
            merge_path = args.output / "02_synonym_merges.json"
            with open(merge_path, "w", encoding="utf-8") as f:
                json.dump(merge_map, f, indent=2, ensure_ascii=False)
            log.info(f"Saved merge map → {merge_path}")

    # ----- Phase 2b: Concept Cleaning Filters -----
    if args.filter_numeric:
        filter_numeric(pool)
    if args.filter_article_prefixed:
        filter_article_prefixed(pool)
    if args.filter_nouns_in_actions:
        filter_nouns_in_actions(pool)

    # # ----- Phase 3: Subsumption Pruning -----
    # log.info("")
    # log.info("=" * 60)
    # log.info("PHASE 3: Subsumption pruning")
    # log.info("=" * 60)

    # if not args.skip_wordnet:
    #     prune_subsumption(pool, min_count=args.min_count)

    # # ----- Phase 4: Frequency Filtering -----
    # log.info("")
    # log.info("=" * 60)
    # log.info("PHASE 4: Frequency filtering (min_count={})".format(args.min_count))
    # log.info("=" * 60)

    # for axis in pool.AXES:
    #     before = len(pool.data[axis])
    #     pool.data[axis] = {
    #         k: v for k, v in pool.data[axis].items()
    #         if v["count"] >= args.min_count
    #     }
    #     after = len(pool.data[axis])
    #     if before != after:
    #         log.info(f"  {axis}: {before} → {after} (removed {before - after})")

    # ----- Final Output -----
    log.info("")
    log.info("=" * 60)
    log.info("FINAL CONCEPT POOL")
    log.info("=" * 60)
    total = 0
    for axis, count in pool.summary().items():
        log.info(f"  {axis:12s}: {count:>6,d} concepts")
        total += count
    log.info(f"  {'TOTAL':12s}: {total:>6,d}")

    final_path = args.output / "03_final_concept_pool.json"
    with open(final_path, "w", encoding="utf-8") as f:
        json.dump(pool.to_serializable(), f, indent=2, ensure_ascii=False)
    log.info(f"\nSaved final pool → {final_path}")

    # Also write a flat TSV for easy inspection
    tsv_path = args.output / "04_concepts_flat.tsv"
    with open(tsv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["axis", "concept", "count", "sources", "raw_forms"])
        for axis in pool.AXES:
            for term, info in sorted(
                pool.data[axis].items(), key=lambda x: -x[1]["count"]
            ):
                writer.writerow([
                    axis,
                    term,
                    info["count"],
                    "|".join(sorted(info["sources"])),
                    "|".join(sorted(info["raw_forms"])),
                ])
    log.info(f"Saved flat TSV  → {tsv_path}")

    # Per-axis files for convenience
    for axis in pool.AXES:
        axis_path = args.output / f"05_{axis}.txt"
        with open(axis_path, "w", encoding="utf-8") as f:
            for term in sorted(pool.data[axis].keys()):
                f.write(term + "\n")
        log.info(f"Saved {axis:12s} → {axis_path}")

    log.info("\nDone.")


if __name__ == "__main__":
    main()