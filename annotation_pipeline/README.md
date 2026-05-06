# Dataset Construction Workflow

This document summarizes the current workflow used in this repository to construct the dataset.

## 1. Merge Raw Annotations

The pipeline starts by merging annotations from multiple sources into one unified per-image record:

- COCO instances
- COCO-Stuff
- LVIS
- PACO-LVIS
- Visual Genome objects, relationships, and attributes

The merged output is written to:

- `merged_annotations.json`

Script:

- `merge_annotations.py`

## 2. Clean to the Curated Concept Set

The merged annotations are filtered against the curated concept vocabulary in `all_concepts.txt`.
Any concept not in the concept set is removed, and the remaining concept names are normalized.

The cleaned output is written to:

- `cleaned_annotations.json`

Script:

- `clean_annotations.py`

## 3. Identify Missing Concepts with a Vision-Language Model

For each image, the pipeline compares:

- the concepts already present in `cleaned_annotations.json`
- the full curated concept set

A vision-language model then predicts additional concepts that are clearly visible in the image but missing from the existing annotations.

Outputs:

- `identified_concepts_openai_full.jsonl`
- optionally merged JSON summaries such as `identified_concepts_openai_full.json`

Script:

- `identify_concepts.py`

## 4. Postprocess Concepts into Annotation Buckets and Masks

The identified concepts are combined with the cleaned annotations and divided into three buckets per image:

- `concepts_with_mask`: concepts that already have usable segmentation masks
- `concepts_with_bbox`: concepts that only have bounding boxes
- `concepts_without_annotation`: concepts predicted by the model that have no source annotation

During this step:

- existing segmentations are converted into binary mask PNGs
- bbox-only concepts are tracked for later mask generation
- relation concepts may use SAM-based mask construction from subject/object boxes

Outputs:

- `postprocessed_annotations/`
- `postprocessed_annotations_summary.json`

Script:

- `postprocess_annotations.py`

## 5. Generate Candidate Masks for Missing Concepts

Concepts that do not already have final masks are passed through a set of mask-generation tools.
The current pipeline supports multiple annotators, including:

- `chefer`
- `attention`
- `grounded_sam`
- `sam3`
- `clipseg`

For each concept, the pipeline saves:

- candidate masks
- overlays
- supporting intermediate outputs

Outputs:

- `annotated_masks/`

Script:

- `annotate.py`

## 6. Verify Candidate Masks

A multimodal verifier reviews the candidate overlays for each concept and selects the best tool output, or rejects all candidates if none are reliable.

This stage produces concept-level verification records, typically in JSONL form.

Outputs include files such as:

- `verified_annotations_glm8b.jsonl`
- `verified_annotations_it8b.jsonl`
- `verified_annotations_qw9b.jsonl`
- `verified_annotations_voting.jsonl`

Script:

- `verify.py`

## 7. Assemble Final Annotations

The final annotation set is built by combining:

- masks copied directly from `postprocessed_annotations/` for concepts that already had masks
- verifier-selected masks from `annotated_masks/` for bbox-only or previously unannotated concepts

At this point, concepts are also grouped into cleaned concept categories using the concept-set directory.

Final outputs:

- `final_annotations/`
- `final_annotation.jsonl`

Script:

- `finalize_annotations.py`

## 8. Generate Captions

After the final masks and concept annotations are assembled, captions are generated from:

- the source image
- the finalized concept annotations

This uses the OpenAI Batch API and writes one caption record per image.

Outputs:

- `generated_captions.jsonl`
- batch artifacts in `captions_openai_batch/`

Script:

- `generate_captions.py`

## End-to-End Summary

In short, the current workflow is:

1. Merge raw annotations from multiple datasets.
2. Filter and normalize them to the curated concept set.
3. Use a vision-language model to identify additional visible concepts.
4. Convert available annotations into masks and bucket concepts by annotation quality.
5. Generate candidate masks for concepts that still need localization.
6. Verify and select the best candidate masks.
7. Assemble the final per-image annotation dataset.
8. Generate grounded image captions from the finalized annotations.
