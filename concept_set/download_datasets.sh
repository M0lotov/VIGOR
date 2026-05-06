#!/usr/bin/env bash
# ============================================================================
# download_datasets.sh
# Downloads annotation files (NOT full images) for all sources.
# Most downloads are small JSON/CSV metadata files (a few hundred MB total).
# Run: bash download_datasets.sh ./data
# ============================================================================
set -euo pipefail

DATA_ROOT="${1:-./data}"
echo "=== Downloading dataset annotations to: $DATA_ROOT ==="
echo ""

# ---- Helper ----
download() {
    local url="$1" dest="$2"
    mkdir -p "$(dirname "$dest")"
    if [ -f "$dest" ]; then
        echo "  [skip] $dest already exists"
    else
        echo "  [get]  $dest"
        curl -fsSL --retry 3 -o "$dest" "$url" || wget -q -O "$dest" "$url" || {
            echo "  [FAIL] Could not download: $url"
            return 1
        }
    fi
}

# ============================================================================
# 1. LVIS
# ============================================================================
echo "--- LVIS v1 (object categories) ---"
download \
    "https://dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json.zip" \
    "$DATA_ROOT/lvis/lvis_v1_val.json.zip"
if [ -f "$DATA_ROOT/lvis/lvis_v1_val.json.zip" ] && [ ! -f "$DATA_ROOT/lvis/lvis_v1_val.json" ]; then
    cd "$DATA_ROOT/lvis" && unzip -qo lvis_v1_val.json.zip && cd -
fi
echo ""

# ============================================================================
# 1b. PACO-LVIS (objects, parts, attributes)
# ============================================================================
echo "--- PACO-LVIS (objects, parts, attributes) ---"
download \
    "https://dl.fbaipublicfiles.com/paco/annotations/paco_lvis_v1.zip" \
    "$DATA_ROOT/paco_lvis/paco_lvis_v1.zip"
if [ -f "$DATA_ROOT/paco_lvis/paco_lvis_v1.zip" ] && [ ! -d "$DATA_ROOT/paco_lvis/annotations" ]; then
    cd "$DATA_ROOT/paco_lvis" && unzip -qo paco_lvis_v1.zip && cd -
fi
echo ""

# ============================================================================
# 2. COCO Panoptic + Captions
# ============================================================================
echo "--- COCO (panoptic + captions) ---"
download \
    "http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip" \
    "$DATA_ROOT/coco/panoptic_annotations_trainval2017.zip"
download \
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip" \
    "$DATA_ROOT/coco/annotations_trainval2017.zip"
if [ -f "$DATA_ROOT/coco/annotations_trainval2017.zip" ] && [ ! -d "$DATA_ROOT/coco/annotations" ]; then
    cd "$DATA_ROOT/coco" && unzip -qo annotations_trainval2017.zip && cd -
fi
echo ""

# ============================================================================
# 3. Visual Genome
# ============================================================================
echo "--- Visual Genome (objects, attributes, relationships) ---"
VG="$DATA_ROOT/visual_genome"
download "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/objects.json.zip" \
    "$VG/objects.json.zip"
download "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/attributes.json.zip" \
    "$VG/attributes.json.zip"
download "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/relationships.json.zip" \
    "$VG/relationships.json.zip"
for f in "$VG"/*.json.zip; do
    [ -f "$f" ] && unzip -qon "$f" -d "$VG/" 2>/dev/null || true
done
echo ""

# ============================================================================
# 4. Open Images (class descriptions)
# ============================================================================
echo "--- Open Images V7 (boxable classes) ---"
download \
    "https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv" \
    "$DATA_ROOT/open_images/class-descriptions-boxable.csv"
echo ""

# ============================================================================
# 5. VAW (Visual Attributes in the Wild)
# ============================================================================
echo "--- VAW (attributes) ---"
echo "  Clone from: https://github.com/adobe-research/vaw_dataset"
echo "  Place JSON files (train_part1.json, ...) in $DATA_ROOT/vaw/"
echo ""

# ============================================================================
# 6. MIT-States
# ============================================================================
echo "--- MIT-States ---"
echo "  Download from: http://web.mit.edu/phillipi/Public/states_and_transformations/"
echo "  Extract to: $DATA_ROOT/mit_states/"
echo ""

# ============================================================================
# 7. GQA Scene Graphs
# ============================================================================
echo "--- GQA (scene graphs) ---"
download \
    "https://downloads.cs.stanford.edu/nlp/data/gqa/sceneGraphs.zip" \
    "$DATA_ROOT/gqa/sceneGraphs.zip"
if [ -f "$DATA_ROOT/gqa/sceneGraphs.zip" ] && [ ! -d "$DATA_ROOT/gqa/sceneGraphs" ]; then
    cd "$DATA_ROOT/gqa" && unzip -qo sceneGraphs.zip && cd -
fi
echo ""

# ============================================================================
# 8. HICO-DET
# ============================================================================
echo "--- HICO-DET (action verbs) ---"
echo "  The original site is offline. Download from Hugging Face:"
echo "  https://huggingface.co/datasets/zhimeng/hico_det"
echo "  Place list_action.csv in $DATA_ROOT/hico_det/"
echo "  (The script has a built-in fallback with 117 canonical verbs)"
echo ""

# ============================================================================
# 8b. V-COCO
# ============================================================================
echo "--- V-COCO (actions) ---"
echo "  Clone from: https://github.com/s-gupta/v-coco"
echo "  Place vcoco_test.json / vcoco_train.json in $DATA_ROOT/vcoco/"
echo "  (The script has a built-in fallback with 26 canonical actions)"
echo ""

# ============================================================================
# 9. imSitu
# ============================================================================
echo "--- imSitu (situation recognition) ---"
download \
    "https://s3.amazonaws.com/my89-frame-annotation/imsitu_space.json" \
    "$DATA_ROOT/imsitu/imsitu_space.json" || \
    echo "  Note: imSitu URL may have changed. Check https://imsitu.org/"
echo ""

# ============================================================================
# 9b. AVA (atomic visual actions)
# ============================================================================
echo "--- AVA (atomic visual actions) ---"
download \
    "https://raw.githubusercontent.com/cvdfoundation/ava-dataset/main/annotations/ava_action_list_v2.2.pbtxt" \
    "$DATA_ROOT/ava/ava_action_list_v2.2.pbtxt" || \
    echo "  Note: AVA URL may have changed. Check https://research.google.com/ava/"
echo ""

# ============================================================================
# 9c. Kinetics-700 (action classes)
# ============================================================================
echo "--- Kinetics-700 (action labels) ---"
download \
    "https://raw.githubusercontent.com/deepmind/kinetics-i3d/master/data/label_map_600.txt" \
    "$DATA_ROOT/kinetics/kinetics_700_labels.csv" || \
    echo "  Note: Kinetics label URL may have changed. Check https://github.com/google-deepmind/kinetics-i3d"
echo ""

# ============================================================================
# 10. Places365 (categories)
# ============================================================================
echo "--- Places365 (scene categories) ---"
download \
    "https://raw.githubusercontent.com/CSAILVision/places365/master/categories_places365.txt" \
    "$DATA_ROOT/places365/categories_places365.txt"
echo ""

# ============================================================================
# 10b. SUN397 (scene categories & attributes)
# ============================================================================
echo "--- SUN397 (scenes + scene attributes) ---"
echo "  Download from: https://vision.princeton.edu/projects/2010/SUN/"
echo "  Extract to: $DATA_ROOT/sun397/ (should contain ClassName.txt)"
echo "  For scene attributes: https://cs.brown.edu/~gen/sunattributes.html"
echo "  Place SUNAttributeDB/ in $DATA_ROOT/sun397/"
echo ""

# ============================================================================
# 11. DTD (Describable Textures)
# ============================================================================
echo "--- DTD (textures) ---"
echo "  Download from: https://www.robots.ox.ac.uk/~vgg/data/dtd/"
echo "  Extract to: $DATA_ROOT/dtd/  (should contain images/ with 47 subdirs)"
echo "  (The script has a built-in fallback with 47 canonical textures)"
echo ""

# ============================================================================
# 12. PartImageNet
# ============================================================================
echo "--- PartImageNet (parts) ---"
echo "  Download from: https://github.com/TACJu/PartImageNet"
echo "  Place train.json/val.json in $DATA_ROOT/partimagenet/"
echo ""

# ============================================================================
# 13. ADE20K
# ============================================================================
echo "--- ADE20K (objects, parts, scenes) ---"
download \
    "https://raw.githubusercontent.com/CSAILVision/placeschallenge/master/sceneparsing/objectInfo150.txt" \
    "$DATA_ROOT/ade20k/objectInfo150.txt" || true
download \
    "https://raw.githubusercontent.com/CSAILVision/sceneparsing/master/sceneCategories.txt" \
    "$DATA_ROOT/ade20k/sceneCategories.txt" || true
echo ""

# ============================================================================
# 14. FMD (Flickr Material Database)
# ============================================================================
echo "--- FMD (materials) ---"
echo "  Download from: https://people.csail.mit.edu/celiu/CVPR2010/FMD/"
echo "  Extract to: $DATA_ROOT/fmd/ (should contain image/ with 10 subdirs)"
echo "  (The script has a built-in fallback with 10 canonical materials)"
echo ""

# ============================================================================
# 15. OpenSurfaces
# ============================================================================
echo "--- OpenSurfaces (materials) ---"
echo "  Download from: http://opensurfaces.cs.cornell.edu/"
echo "  Place materials.json in $DATA_ROOT/opensurfaces/"
echo ""

# ============================================================================
# 16. PASCAL-Part
# ============================================================================
echo "--- PASCAL-Part (parts) ---"
echo "  Download from: https://www.cs.stanford.edu/~roozbeh/pascal-parts/pascal-parts.html"
echo "  Place classes.txt or pascal_part.json in $DATA_ROOT/pascal_part/"
echo ""

# ============================================================================
# 17. Conceptual Captions (for caption mining)
# ============================================================================
echo "--- Conceptual Captions (caption mining) ---"
echo "  Download from: https://ai.google.com/research/ConceptualCaptions/"
echo "  Place Train_GCC-training.tsv in $DATA_ROOT/conceptual_captions/"
echo "  (Optional — only used if --skip-captions is not set)"
echo ""

# ============================================================================
echo ""
echo "=== Download complete ==="
echo "Datasets that require manual download are noted above."
echo "Run:  python extract_visual_concepts.py --data-root $DATA_ROOT"