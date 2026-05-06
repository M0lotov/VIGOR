import json


INPUT_PATH = "merged_annotations.json"
OUTPUT_PATH = "cleaned_annotations.json"
CONCEPT_SET_PATH = "../concept_set/output/final_concepts_cleaned/all_concepts.txt"


def normalize_concept(text):
    return text.strip().lower().replace("_", " ")


with open(INPUT_PATH) as f:
    annotations = json.load(f)

with open(CONCEPT_SET_PATH) as f:
    concept_set = {line.strip() for line in f if line.strip()}

original_count = 0
filtered_count = 0

for image in annotations["images"]:
    for level, concepts in image["concepts"].items():
        original_count += len(concepts)
        filtered_concepts = []
        for concept in concepts:
            normalized_category = normalize_concept(concept["category"])
            if normalized_category in concept_set:
                concept["category"] = normalized_category
                filtered_concepts.append(concept)
        image["concepts"][level] = filtered_concepts
        filtered_count += len(filtered_concepts)

if "statistics" in annotations:
    concept_counts = {}
    concept_counts_by_source = {}

    for image in annotations["images"]:
        for level, concepts in image["concepts"].items():
            concept_counts[level] = concept_counts.get(level, 0) + len(concepts)
            for concept in concepts:
                key = f"{level}|{concept['source']}"
                concept_counts_by_source[key] = concept_counts_by_source.get(key, 0) + 1

    annotations["statistics"]["concept_counts"] = concept_counts
    annotations["statistics"]["concept_counts_by_source"] = concept_counts_by_source

with open(OUTPUT_PATH, "w") as f:
    json.dump(annotations, f)

print(f"Original annotations: {original_count}")
print(f"Filtered annotations: {filtered_count}")
print(f"Wrote cleaned annotations to {OUTPUT_PATH}")
