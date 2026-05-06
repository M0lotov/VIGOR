# Concept Set Construction Workflow

This repository builds a visual concept set in five stages:

1. Extract raw candidate concepts from multiple vision datasets.
2. Merge and normalize the extracted pool.
3. Ask one or more LLM providers to filter and recategorize the candidates.
4. Keep concepts that are stable across providers.
5. Run one final pool-level cleaning pass to remove residual duplicates and category errors.

## 1. Extract the raw concept pool

Download or place the annotation files under `data/` using [download_datasets.sh](download_datasets.sh), then run [extract_visual_concepts.py](extract_visual_concepts.py).

```bash
python extract_visual_concepts.py --data-root ./data --output ./output
```

What this step does:

- Pulls concepts from many dataset-specific sources across the axes `objects`, `attributes`, `relations`, `actions`, `scenes`, `materials`, and `parts`.
- Normalizes strings and tracks provenance and counts.
- Optionally applies WordNet synonym merging and a few cleanup filters.

Main outputs:

- `output/01_raw_concept_pool.json`: raw extracted concepts before merging.
- `output/02_synonym_merges.json`: WordNet-based merge map when enabled.
- `output/03_final_concept_pool.json`: normalized pool after merging/filtering.
- `output/04_concepts_flat.tsv`: flat inspection table.
- `output/05_objects.txt`, `05_attributes.txt`, ..., `05_parts.txt`: one text file per extraction axis.

## 2. Build one deduplicated candidate list

The LLM filtering stage operates on a single flat list. The repo currently builds that file in [util.ipynb](util.ipynb) by unioning all `output/05_*.txt` files and removing duplicates:

- Input: `output/05_*.txt`
- Output: `output/05_all_concepts.txt`

This file is the shared input for all provider-specific filtering runs.

## 3. Filter concepts with LLM batches

The filtering prompt is defined in [llm_filter_concepts.py](llm_filter_concepts.py). Each concept is judged for:

- lexical well-formedness
- visual groundability
- atomicity
- semantic level
- positive assertion

The same step also assigns one visual category:

- `color`
- `edge`
- `texture`
- `shape`
- `part`
- `object`
- `motion`
- `relation`

There are two ways to run this stage.

### Option A: direct API calls

Run [llm_filter_concepts.py](llm_filter_concepts.py) directly against one provider.

```bash
python llm_filter_concepts.py \
  --input output/05_all_concepts.txt \
  --output output \
  --provider openai
```

This writes provider outputs under `output/<provider>/`:

- `05_raw_responses.json`
- `06_llm_judgments_full.json`
- `07_filtered_concept_pool.json`
- `08_by_visual_category.json`
- `08_filtered_by_category_txt/*.txt`
- `09_rejected_concepts.txt`

### Option B: provider batch workflows

For large runs, the repo prepares requests and parses results separately:

- OpenAI: [prepare_openai_batch_requests.py](prepare_openai_batch_requests.py) and [parse_openai_batch_results.py](parse_openai_batch_results.py)
- Anthropic: [prepare_anthropic_batch_requests.py](prepare_anthropic_batch_requests.py) and [parse_anthropic_batch_results.py](parse_anthropic_batch_results.py)
- Google: [prepare_google_batch_requests.py](prepare_google_batch_requests.py) and [parse_google_batch_results.py](parse_google_batch_results.py)

The provider notebooks document submission details:

- [openai_batch_workflow.ipynb](openai_batch_workflow.ipynb)
- [anthropic_batch_workflow.ipynb](anthropic_batch_workflow.ipynb)
- [google_batch_workflow.ipynb](google_batch_workflow.ipynb)

All parse scripts call [batch_result_utils.py](batch_result_utils.py) so they end up producing the same standardized artifacts as the direct run.

## 4. Reconcile providers into a consensus concept pool

The repo does not currently have a standalone Python script for this reconciliation step. The logic lives in [util.ipynb](util.ipynb).

For each visual category, it scans:

- `output/openai/08_filtered_by_category_txt/<category>.txt`
- `output/anthropic/08_filtered_by_category_txt/<category>.txt`
- `output/google/08_filtered_by_category_txt/<category>.txt`

Then it keeps only concepts that appear in more than one provider output. In practice, this is a simple consensus filter:

- keep a concept if at least 2 providers kept it in the same category
- discard concepts supported by only 1 provider

Outputs:

- `output/final_concepts/color.txt`
- `output/final_concepts/edge.txt`
- `output/final_concepts/texture.txt`
- `output/final_concepts/shape.txt`
- `output/final_concepts/part.txt`
- `output/final_concepts/object.txt`
- `output/final_concepts/motion.txt`
- `output/final_concepts/relation.txt`
- `output/final_concepts/all_concepts.txt`

## 5. Run final pool-level cleaning

The last step is [final_cleaning.py](final_cleaning.py):

```bash
python final_cleaning.py \
  --input-dir output/final_concepts \
  --output-dir output/final_concepts_cleaned \
  --provider openai
```

This stage operates on the full pool at once and fixes issues that only become visible globally:

- spelling and hyphenation variants
- singular/plural duplicates
- morphological variants
- near-synonyms
- concepts composable from other concepts already in the pool
- category misassignments

Outputs:

- `output/final_cleaning_raw_response.txt`: raw LLM response.
- `output/final_cleaning_actions.json`: structured merge/remove/recategorize actions.
- `output/final_concepts_cleaned/*.txt`: cleaned final category files.
- `output/final_concepts_cleaned/all_concepts.txt`: final deduplicated concept inventory.

## Practical summary

If you want the shortest accurate mental model, it is:

1. Mine candidate concepts from dataset annotations.
2. Flatten them into one master candidate list.
3. Let multiple LLMs reject bad concepts and assign visual categories.
4. Keep only concepts that multiple providers agree on.
5. Do one final LLM cleaning pass over the consensus pool.

That is the workflow that produced the current contents of `output/`.
