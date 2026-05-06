"""
Merge annotations from COCO, LVIS, PACO-LVIS, and Visual Genome
into a unified vision rationale dataset.

Concept levels covered:
  - COCO:      object (instance & stuff segmentation)
  - LVIS:      object (large-vocabulary instance segmentation)
  - PACO-LVIS: part, color, texture, pattern, material (attributes)
  - VG:        object, relation, attribute (color, shape, etc.)

Usage:
    python merge_annotations.py \
        --coco_instances   path/to/coco/instances_val2017.json \
        --coco_stuff       path/to/coco/stuff_val2017.json \
        --lvis             path/to/lvis/lvis_v1_val.json \
        --paco             path/to/paco/paco_lvis_v1_val.json \
        --vg_objects       path/to/vg/objects.json \
        --vg_relationships path/to/vg/relationships.json \
        --vg_attributes    path/to/vg/attributes.json \
        --vg_image_data    path/to/vg/image_data.json \
        --output           path/to/merged_annotations.json

All flags are optional — the script merges whichever datasets you provide.
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_json(path: str) -> Any:
    """Load a JSON file with progress logging."""
    logger.info(f"Loading {path} ...")
    with open(path) as f:
        data = json.load(f)
    logger.info(f"  Done. Type={type(data).__name__}, "
                f"Top keys={list(data.keys()) if isinstance(data, dict) else f'list[{len(data)}]'}")
    return data


def build_category_map(categories: list[dict]) -> dict[int, str]:
    """Map category_id -> category_name from a COCO-style category list."""
    return {c["id"]: c["name"] for c in categories}


def rle_to_summary(segmentation) -> dict:
    """
    Keep segmentation in a compact form.
    - polygon  → store as-is (list of lists of floats)
    - RLE      → store as-is (dict with 'counts' and 'size')
    - None     → store bbox only (handled by caller)
    """
    if segmentation is None:
        return None
    return segmentation


# ---------------------------------------------------------------------------
# Per-image unified record
# ---------------------------------------------------------------------------

def make_empty_record(coco_image_id: int) -> dict:
    """Create an empty merged record for one image."""
    return {
        "coco_image_id": coco_image_id,
        "image_info": {},          # filled from whichever source first provides it
        "concepts": {
            "object": [],          # from COCO instances, LVIS, VG
            # "stuff": [],           # from COCO-Stuff (background / uncountable)
            "part": [],            # from PACO-LVIS
            "attribute": [],       # from PACO-LVIS attrs + VG attributes
            "relation": [],        # from VG relationships
        },
        "sources": set(),          # track which datasets contributed
    }


# ---------------------------------------------------------------------------
# 1. COCO instances
# ---------------------------------------------------------------------------

def ingest_coco_instances(data: dict, records: dict[int, dict]):
    """Add COCO instance annotations (object-level)."""
    cat_map = build_category_map(data["categories"])

    # Store image metadata
    for img in data["images"]:
        iid = img["id"]
        if iid not in records:
            records[iid] = make_empty_record(iid)
        records[iid]["image_info"] = {
            "file_name": img["file_name"],
            "width": img["width"],
            "height": img["height"],
            "coco_url": img.get("coco_url", ""),
        }

    for ann in data["annotations"]:
        iid = ann["image_id"]
        if iid not in records:
            records[iid] = make_empty_record(iid)
        records[iid]["concepts"]["object"].append({
            "source": "coco_instances",
            "concept_level": "object",
            "category": cat_map.get(ann["category_id"], "unknown"),
            "category_id": ann["category_id"],
            "annotation_id": ann["id"],
            "bbox": ann.get("bbox"),               # [x, y, w, h]
            "segmentation": rle_to_summary(ann.get("segmentation")),
            "area": ann.get("area"),
            "iscrowd": ann.get("iscrowd", 0),
        })
        records[iid]["sources"].add("coco_instances")

    logger.info(f"  COCO instances: {len(data['annotations'])} annotations, "
                f"{len(data['images'])} images")


# ---------------------------------------------------------------------------
# 2. COCO-Stuff
# ---------------------------------------------------------------------------

def ingest_coco_stuff(data: dict, records: dict[int, dict]):
    """Add COCO-Stuff annotations (stuff / texture-region level)."""
    cat_map = build_category_map(data["categories"])

    for ann in data["annotations"]:
        iid = ann["image_id"]
        if iid not in records:
            records[iid] = make_empty_record(iid)
        records[iid]["concepts"]["stuff"].append({
            "source": "coco_stuff",
            "concept_level": "stuff",
            "category": cat_map.get(ann["category_id"], "unknown"),
            "category_id": ann["category_id"],
            "annotation_id": ann["id"],
            "segmentation": rle_to_summary(ann.get("segmentation")),
            "area": ann.get("area"),
            "iscrowd": ann.get("iscrowd", 0),
        })
        records[iid]["sources"].add("coco_stuff")

    logger.info(f"  COCO-Stuff: {len(data['annotations'])} annotations")


# ---------------------------------------------------------------------------
# 3. LVIS
# ---------------------------------------------------------------------------

def ingest_lvis(data: dict, records: dict[int, dict]):
    """Add LVIS annotations (large-vocabulary object-level)."""
    cat_map = build_category_map(data["categories"])

    # LVIS images are COCO images — same IDs
    for img in data.get("images", []):
        iid = img["id"]
        if iid not in records:
            records[iid] = make_empty_record(iid)
        # Fill image_info if not already present
        if not records[iid]["image_info"]:
            records[iid]["image_info"] = {
                "file_name": img.get("file_name", img.get("coco_url", "")),
                "width": img["width"],
                "height": img["height"],
                "coco_url": img.get("coco_url", ""),
            }

    for ann in data["annotations"]:
        iid = ann["image_id"]
        if iid not in records:
            records[iid] = make_empty_record(iid)
        records[iid]["concepts"]["object"].append({
            "source": "lvis",
            "concept_level": "object",
            "category": cat_map.get(ann["category_id"], "unknown"),
            "category_id": ann["category_id"],
            "annotation_id": ann["id"],
            "bbox": ann.get("bbox"),
            "segmentation": rle_to_summary(ann.get("segmentation")),
            "area": ann.get("area"),
        })
        records[iid]["sources"].add("lvis")

    logger.info(f"  LVIS: {len(data['annotations'])} annotations, "
                f"{len(data.get('images', []))} images")


# ---------------------------------------------------------------------------
# 4. PACO-LVIS
# ---------------------------------------------------------------------------

# PACO attribute taxonomy — maps attr IDs to (concept_level, value).
# This mirrors the PACO attribute ontology; adapt if your version differs.
PACO_ATTR_CONCEPT_MAP = {
    "color":    "color",
    "pattern":  "texture",      # map pattern → texture concept level
    "material": "texture",      # map material → texture concept level
}


def ingest_paco(data: dict, records: dict[int, dict]):
    """
    Add PACO-LVIS annotations (part-level + attributes).

    PACO annotations contain:
      - Object and part segmentation masks (category hierarchy)
      - Per-instance attributes: color, pattern_marking, material
    """
    cat_map = build_category_map(data["categories"])

    # Build attribute name lookup  (PACO stores attrs in a separate key)
    attr_map = {}
    for attr in data.get("attributes", []):
        attr_map[attr["id"]] = attr["name"]   # e.g. 1 -> "black"

    # Build a category hierarchy to distinguish objects vs parts
    cat_info = {}
    for c in data["categories"]:
        cat_info[c["id"]] = {
            "name": c["name"],
            "supercategory": c.get("supercategory", ""),
        }

    for ann in data["annotations"]:
        iid = ann["image_id"]
        if iid not in records:
            records[iid] = make_empty_record(iid)

        cat = cat_info.get(ann["category_id"], {})
        cat_name = cat.get("name", "unknown")
        supercategory = cat.get("supercategory", "")

        # Determine if this is an object or part annotation
        # PACO parts have a supercategory that is the parent object
        is_part = ":" in cat_name  # PACO uses "object:part" naming
        cat_name = cat_name.split(':')[-1] if is_part else cat_name

        concept_level = "part" if is_part else "object"
        target_list = "part" if is_part else "object"

        entry = {
            "source": "paco_lvis",
            "concept_level": concept_level,
            "category": cat_name,
            "category_id": ann["category_id"],
            "annotation_id": ann["id"],
            "bbox": ann.get("bbox"),
            "segmentation": rle_to_summary(ann.get("segmentation")),
            "area": ann.get("area"),
        }
        records[iid]["concepts"][target_list].append(entry)

        # ---- Extract attributes as separate concept entries ----
        # PACO stores attributes per annotation in various formats;
        # common keys: "color", "pattern_marking", "material"
        for attr_key in ("color", "pattern_marking", "material"):
            attr_val = ann.get(attr_key) or ann.get(f"{attr_key}_id")
            if attr_val is None:
                continue

            # attr_val might be an ID (int) or a list of IDs
            if isinstance(attr_val, int):
                attr_val = [attr_val]
            elif isinstance(attr_val, str):
                attr_val = [attr_val]

            for av in attr_val:
                attr_name = attr_map.get(av, av) if isinstance(av, int) else av
                # Skip negative / absent labels
                if attr_name in ("", "not applicable", "n/a", None):
                    continue

                mapped_level = PACO_ATTR_CONCEPT_MAP.get(attr_key, "attribute")
                records[iid]["concepts"]["attribute"].append({
                    "source": "paco_lvis",
                    "concept_level": mapped_level,
                    "attribute_type": attr_key,
                    "value": attr_name,
                    "parent_annotation_id": ann["id"],
                    "parent_category": cat_name,
                    "bbox": ann.get("bbox"),  # inherit localization from parent
                })

        records[iid]["sources"].add("paco_lvis")

    logger.info(f"  PACO-LVIS: {len(data['annotations'])} annotations")


# ---------------------------------------------------------------------------
# 5. Visual Genome
# ---------------------------------------------------------------------------

def build_vg_to_coco_map(vg_image_data: list[dict]) -> dict[int, dict[str, Any]]:
    """
    Build VG image_id -> metadata mapping.
    Visual Genome provides a 'coco_id' field in its image metadata.
    Returns {vg_id: {"coco_id": ..., "width": ..., "height": ...}}.
    """
    mapping = {}
    for img in vg_image_data:
        vg_id = img["image_id"]
        mapping[vg_id] = {
            "coco_id": img.get("coco_id"),  # may be None / null
            "width": img.get("width"),
            "height": img.get("height"),
        }
    return mapping


def rescale_vg_bbox(
    bbox: list[float] | None,
    *,
    vg_width: float | None,
    vg_height: float | None,
    coco_width: float | None,
    coco_height: float | None,
) -> list[float] | None:
    """Rescale a VG [x, y, w, h] bbox into the matched COCO image size."""
    if bbox is None or len(bbox) != 4:
        return bbox

    if not vg_width or not vg_height or not coco_width or not coco_height:
        return bbox

    scale_x = coco_width / vg_width
    scale_y = coco_height / vg_height
    x, y, w, h = bbox
    return [x * scale_x, y * scale_y, w * scale_x, h * scale_y]


def ingest_vg_objects(vg_objects: list[dict],
                      vg2coco: dict[int, dict[str, Any]],
                      records: dict[int, dict]):
    """Add Visual Genome object annotations mapped to COCO image IDs."""
    n_mapped, n_skipped = 0, 0
    for img_entry in vg_objects:
        vg_id = img_entry["image_id"]
        vg_meta = vg2coco.get(vg_id, {})
        coco_id = vg_meta.get("coco_id")
        if coco_id is None:
            n_skipped += 1
            continue

        if coco_id not in records:
            records[coco_id] = make_empty_record(coco_id)

        coco_info = records[coco_id].get("image_info", {})
        vg_width = vg_meta.get("width")
        vg_height = vg_meta.get("height")
        coco_width = coco_info.get("width")
        coco_height = coco_info.get("height")

        for obj in img_entry.get("objects", []):
            records[coco_id]["concepts"]["object"].append({
                "source": "visual_genome",
                "concept_level": "object",
                "category": ", ".join(obj.get("names", [obj.get("name", "unknown")])),
                "synsets": obj.get("synsets", []),
                "vg_object_id": obj.get("object_id"),
                "bbox": rescale_vg_bbox(
                    [obj.get("x", 0), obj.get("y", 0), obj.get("w", 0), obj.get("h", 0)],
                    vg_width=vg_width,
                    vg_height=vg_height,
                    coco_width=coco_width,
                    coco_height=coco_height,
                ),
                "segmentation": None,  # VG doesn't provide masks
            })
            n_mapped += 1

        records[coco_id]["sources"].add("visual_genome_objects")

    logger.info(f"  VG objects: {n_mapped} mapped, {n_skipped} images skipped (no COCO match)")


def ingest_vg_attributes(vg_attributes: list[dict],
                         vg2coco: dict[int, dict[str, Any]],
                         records: dict[int, dict]):
    """
    Add Visual Genome attribute annotations.
    VG attributes include color, shape, material, state, etc.
    """
    # Simple heuristic to map VG attributes to concept levels
    COLOR_KEYWORDS = {
        "red", "blue", "green", "yellow", "black", "white", "brown", "gray",
        "grey", "orange", "pink", "purple", "beige", "tan", "golden", "silver",
        "dark", "light", "bright", "colorful", "multicolored",
    }
    SHAPE_KEYWORDS = {
        "round", "square", "rectangular", "circular", "oval", "triangular",
        "curved", "flat", "long", "short", "tall", "thin", "thick", "narrow",
        "wide", "small", "large", "big", "little", "huge", "tiny",
    }
    TEXTURE_KEYWORDS = {
        "wooden", "metal", "metallic", "glass", "plastic", "brick", "stone",
        "concrete", "fabric", "leather", "rubber", "smooth", "rough", "soft",
        "hard", "shiny", "matte", "glossy", "furry", "fluffy", "striped",
        "spotted", "plaid", "checkered",
    }

    def classify_attribute(attr_name: str) -> str:
        attr_lower = attr_name.lower().strip()
        if attr_lower in COLOR_KEYWORDS:
            return "color"
        if attr_lower in SHAPE_KEYWORDS:
            return "shape"
        if attr_lower in TEXTURE_KEYWORDS:
            return "texture"
        return "attribute"  # generic fallback

    n_mapped, n_skipped = 0, 0
    for img_entry in vg_attributes:
        vg_id = img_entry["image_id"]
        vg_meta = vg2coco.get(vg_id, {})
        coco_id = vg_meta.get("coco_id")
        if coco_id is None:
            n_skipped += 1
            continue

        if coco_id not in records:
            records[coco_id] = make_empty_record(coco_id)

        coco_info = records[coco_id].get("image_info", {})
        vg_width = vg_meta.get("width")
        vg_height = vg_meta.get("height")
        coco_width = coco_info.get("width")
        coco_height = coco_info.get("height")

        for obj in img_entry.get("attributes", []):
            obj_name = ", ".join(obj.get("names", [obj.get("name", "unknown")]))
            bbox = rescale_vg_bbox(
                [obj.get("x", 0), obj.get("y", 0), obj.get("w", 0), obj.get("h", 0)],
                vg_width=vg_width,
                vg_height=vg_height,
                coco_width=coco_width,
                coco_height=coco_height,
            )

            for attr in obj.get("attributes", []):
                concept_level = classify_attribute(attr)
                records[coco_id]["concepts"]["attribute"].append({
                    "source": "visual_genome",
                    "concept_level": concept_level,
                    "attribute_type": concept_level,
                    "category": attr,
                    "parent_object": obj_name,
                    "vg_object_id": obj.get("object_id"),
                    "bbox": bbox,
                })
                n_mapped += 1

        records[coco_id]["sources"].add("visual_genome_attributes")

    logger.info(f"  VG attributes: {n_mapped} mapped, {n_skipped} images skipped")


def ingest_vg_relationships(vg_rels: list[dict],
                            vg2coco: dict[int, dict[str, Any]],
                            records: dict[int, dict]):
    """Add Visual Genome relationship annotations (relation-level concepts)."""
    n_mapped, n_skipped = 0, 0
    for img_entry in vg_rels:
        vg_id = img_entry["image_id"]
        vg_meta = vg2coco.get(vg_id, {})
        coco_id = vg_meta.get("coco_id")
        if coco_id is None:
            n_skipped += 1
            continue

        if coco_id not in records:
            records[coco_id] = make_empty_record(coco_id)

        coco_info = records[coco_id].get("image_info", {})
        vg_width = vg_meta.get("width")
        vg_height = vg_meta.get("height")
        coco_width = coco_info.get("width")
        coco_height = coco_info.get("height")

        for rel in img_entry.get("relationships", []):
            subj = rel.get("subject", {})
            obj = rel.get("object", {})
            predicate = rel.get("predicate", "unknown")

            subj_name = ", ".join(subj.get("names", [subj.get("name", "unknown")]))
            obj_name = ", ".join(obj.get("names", [obj.get("name", "unknown")]))

            records[coco_id]["concepts"]["relation"].append({
                "source": "visual_genome",
                "concept_level": "relation",
                "category": predicate,
                "synsets": rel.get("synsets", []),
                "subject": {
                    "name": subj_name,
                    "vg_object_id": subj.get("object_id"),
                    "bbox": rescale_vg_bbox(
                        [subj.get("x", 0), subj.get("y", 0), subj.get("w", 0), subj.get("h", 0)],
                        vg_width=vg_width,
                        vg_height=vg_height,
                        coco_width=coco_width,
                        coco_height=coco_height,
                    ),
                },
                "object": {
                    "name": obj_name,
                    "vg_object_id": obj.get("object_id"),
                    "bbox": rescale_vg_bbox(
                        [obj.get("x", 0), obj.get("y", 0), obj.get("w", 0), obj.get("h", 0)],
                        vg_width=vg_width,
                        vg_height=vg_height,
                        coco_width=coco_width,
                        coco_height=coco_height,
                    ),
                },
                "vg_relationship_id": rel.get("relationship_id"),
            })
            n_mapped += 1

        records[coco_id]["sources"].add("visual_genome_relationships")

    logger.info(f"  VG relationships: {n_mapped} mapped, {n_skipped} images skipped")


# ---------------------------------------------------------------------------
# Post-processing: deduplication & statistics
# ---------------------------------------------------------------------------

def deduplicate_objects(records: dict[int, dict]):
    """
    Remove duplicate objects across COCO / LVIS / VG for the same image.
    Strategy: within each image, group by overlapping bbox (IoU > threshold)
    and keep the richest annotation (prefer mask over bbox-only).
    """
    IOU_THRESH = 0.5

    def iou_bbox(a, b):
        """Compute IoU between two [x, y, w, h] bboxes."""
        if a is None or b is None:
            return 0.0
        ax1, ay1, aw, ah = a
        bx1, by1, bw, bh = b
        ax2, ay2 = ax1 + aw, ay1 + ah
        bx2, by2 = bx1 + bw, by1 + bh

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = aw * ah + bw * bh - inter
        return inter / union if union > 0 else 0.0

    n_before = 0
    n_after = 0

    for iid, rec in records.items():
        objs = rec["concepts"]["object"]
        n_before += len(objs)
        if len(objs) <= 1:
            n_after += len(objs)
            continue

        # Greedy merge: mark duplicates
        keep = [True] * len(objs)
        for i in range(len(objs)):
            if not keep[i]:
                continue
            for j in range(i + 1, len(objs)):
                if not keep[j]:
                    continue
                if iou_bbox(objs[i].get("bbox"), objs[j].get("bbox")) > IOU_THRESH:
                    # Keep the one with segmentation mask; prefer COCO/LVIS over VG
                    has_seg_i = objs[i].get("segmentation") is not None
                    has_seg_j = objs[j].get("segmentation") is not None
                    if has_seg_j and not has_seg_i:
                        keep[i] = False
                    else:
                        keep[j] = False

        rec["concepts"]["object"] = [o for o, k in zip(objs, keep) if k]
        n_after += len(rec["concepts"]["object"])

    logger.info(f"  Object dedup: {n_before} → {n_after} "
                f"({n_before - n_after} duplicates removed)")


def compute_statistics(records: dict[int, dict]) -> dict:
    """Compute summary statistics for the merged dataset."""
    stats = {
        "total_images": len(records),
        "images_by_source_count": defaultdict(int),
        "concept_counts": defaultdict(int),
        "concept_counts_by_source": defaultdict(int),
        "images_with_all_levels": 0,
    }

    all_concept_keys = {"object", "part", "attribute", "relation"}

    for iid, rec in records.items():
        n_sources = len(rec["sources"])
        stats["images_by_source_count"][n_sources] += 1

        present_levels = set()
        for level, entries in rec["concepts"].items():
            stats["concept_counts"][level] += len(entries)
            for e in entries:
                stats["concept_counts_by_source"][(level, e["source"])] += 1
            if entries:
                present_levels.add(level)

        if present_levels >= all_concept_keys:
            stats["images_with_all_levels"] += 1

    # Convert defaultdicts for JSON serialization
    stats["images_by_source_count"] = dict(stats["images_by_source_count"])
    stats["concept_counts"] = dict(stats["concept_counts"])
    stats["concept_counts_by_source"] = {
        f"{k[0]}|{k[1]}": v
        for k, v in stats["concept_counts_by_source"].items()
    }
    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Merge COCO, LVIS, PACO-LVIS, and Visual Genome annotations "
                    "into a unified vision rationale dataset."
    )
    parser.add_argument("--coco_instances", type=str, default='data/coco/annotations/instances_train2017.json',
                        help="Path to COCO instances JSON (e.g. instances_train2017.json)")
    parser.add_argument("--coco_stuff", type=str, default=None,
                        help="Path to COCO-Stuff JSON (e.g. stuff_val2017.json)")
    parser.add_argument("--lvis", type=str, default='data/lvis/lvis_v1_train.json',
                        help="Path to LVIS JSON (e.g. lvis_v1_val.json)")
    parser.add_argument("--paco", type=str, default='data/paco_lvis/paco_lvis_v1_train.json',
                        help="Path to PACO-LVIS JSON")
    parser.add_argument("--vg_objects", type=str, default='data/visual_genome/objects.json',
                        help="Path to Visual Genome objects.json")
    parser.add_argument("--vg_relationships", type=str, default='data/visual_genome/relationships.json',
                        help="Path to Visual Genome relationships.json")
    parser.add_argument("--vg_attributes", type=str, default='data/visual_genome/attributes.json',
                        help="Path to Visual Genome attributes.json")
    parser.add_argument("--vg_image_data", type=str, default='data/visual_genome/image_data.json',
                        help="Path to Visual Genome image_data.json (required for VG→COCO mapping)")
    parser.add_argument("--output", type=str, default="merged_annotations.json",
                        help="Output path for merged annotations")
    parser.add_argument("--output_stats", type=str, default='stats.json',
                        help="Optional path to write statistics JSON")
    args = parser.parse_args()

    records: dict[int, dict] = {}  # coco_image_id → merged record

    # --- Ingest COCO instances ---
    if args.coco_instances:
        data = load_json(args.coco_instances)
        ingest_coco_instances(data, records)
        del data

    # --- Ingest COCO-Stuff ---
    if args.coco_stuff:
        data = load_json(args.coco_stuff)
        ingest_coco_stuff(data, records)
        del data

    # --- Ingest LVIS ---
    if args.lvis:
        data = load_json(args.lvis)
        ingest_lvis(data, records)
        del data

    # --- Ingest PACO-LVIS ---
    if args.paco:
        data = load_json(args.paco)
        ingest_paco(data, records)
        del data

    # --- Ingest Visual Genome (requires image_data for COCO mapping) ---
    vg_requested = any([args.vg_objects, args.vg_relationships, args.vg_attributes])
    if vg_requested:
        if not args.vg_image_data:
            logger.warning("Visual Genome data requested but --vg_image_data not provided. "
                           "Cannot map VG images to COCO. Skipping VG.")
        else:
            vg_img = load_json(args.vg_image_data)
            vg2coco = build_vg_to_coco_map(vg_img)
            n_with_coco = sum(1 for v in vg2coco.values() if v is not None)
            logger.info(f"  VG→COCO mapping: {n_with_coco}/{len(vg2coco)} images have COCO IDs")
            del vg_img

            if args.vg_objects:
                vg_data = load_json(args.vg_objects)
                ingest_vg_objects(vg_data, vg2coco, records)
                del vg_data

            if args.vg_attributes:
                vg_data = load_json(args.vg_attributes)
                ingest_vg_attributes(vg_data, vg2coco, records)
                del vg_data

            if args.vg_relationships:
                vg_data = load_json(args.vg_relationships)
                ingest_vg_relationships(vg_data, vg2coco, records)
                del vg_data

    if not records:
        logger.error("No datasets loaded! Provide at least one dataset path.")
        return

    # --- Filter: keep only images shared by ALL provided sources ---
    # Map granular source tags → top-level dataset groups
    SOURCE_TO_GROUP = {
        "coco_instances":               "coco",
        # "coco_stuff":                   "coco",
        "lvis":                         "lvis",
        "paco_lvis":                    "paco_lvis",
        "visual_genome_objects":        "visual_genome",
        "visual_genome_attributes":     "visual_genome",
        "visual_genome_relationships":  "visual_genome",
    }

    # Determine which top-level groups were actually loaded
    provided_groups = set()
    if args.coco_instances or args.coco_stuff:
        provided_groups.add("coco")
    if args.lvis:
        provided_groups.add("lvis")
    if args.paco:
        provided_groups.add("paco_lvis")
    if vg_requested and args.vg_image_data:
        provided_groups.add("visual_genome")

    n_before_filter = len(records)
    if len(provided_groups) > 1:
        to_remove = []
        for iid, rec in records.items():
            image_groups = {SOURCE_TO_GROUP[s] for s in rec["sources"]
                           if s in SOURCE_TO_GROUP}
            if not image_groups >= provided_groups:
                to_remove.append(iid)
        for iid in to_remove:
            del records[iid]
        logger.info(f"  Shared-image filter: {n_before_filter} → {len(records)} images "
                    f"(kept only images present in all {len(provided_groups)} sources: "
                    f"{sorted(provided_groups)})")
    else:
        logger.info(f"  Only 1 source group provided — no cross-source filtering needed.")

    # --- Post-processing ---
    logger.info("Post-processing ...")
    deduplicate_objects(records)

    # --- Statistics ---
    stats = compute_statistics(records)
    logger.info("=" * 60)
    logger.info(f"  Total images:         {stats['total_images']}")
    logger.info(f"  Images w/ all levels: {stats['images_with_all_levels']}")
    for level, count in sorted(stats["concept_counts"].items()):
        logger.info(f"  {level:>12s}: {count:>10,} annotations")
    logger.info("=" * 60)

    # --- Serialize ---
    # Convert sets to lists for JSON
    output_records = []
    for iid in sorted(records):
        rec = records[iid]
        rec["sources"] = sorted(rec["sources"])
        output_records.append(rec)

    output = {
        "info": {
            "description": "Merged vision rationale annotations — "
                           "filtered to images shared by ALL provided datasets",
            "datasets_used": sorted(provided_groups),
            "concept_levels": ["object", "part", "attribute", "relation"],
            "total_images_before_filter": n_before_filter,
            "total_images_after_filter": len(records),
        },
        "statistics": stats,
        "images": output_records,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Writing merged annotations to {out_path} ...")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info(f"  Done. File size: {out_path.stat().st_size / 1e6:.1f} MB")

    if args.output_stats:
        with open(args.output_stats, "w") as f:
            json.dump(stats, f, indent=2, default=str)
        logger.info(f"  Stats written to {args.output_stats}")


if __name__ == "__main__":
    main()
