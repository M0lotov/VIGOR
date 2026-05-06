# VIGOR: Benchmarking Visual Rationale Correctness in Multimodal Large Language Models

This repository contains the collection and evaluation code for the VIGOR benchmark.

## Run Evaluation

Install the main dependencies:

```bash
pip install torch transformers pillow numpy pycocotools tqdm
```

Run the default model on the benchmark:

```bash
python evaluate.py
```

Run a specific Hugging Face model:

```bash
python evaluate.py --model Qwen/Qwen2.5-VL-7B-Instruct
```

The script prints a table with `mean_auprc` and `pointing_accuracy` for the
overall benchmark and each concept level.

## Repository Contents

- `evaluate.py`: main evaluation script.
- `train.json`: COCO-style benchmark annotations.
- `concept_set/`: scripts and notebooks for building the visual concept set.
- `annotation_pipeline/`: scripts for constructing image-level annotations,
  masks, verification records, and captions.
