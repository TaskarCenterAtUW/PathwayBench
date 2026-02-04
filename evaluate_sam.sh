#!/bin/bash

# Usage: ./evaluate_sam.sh sam_road

# Root directory of your project
ROOT_DIR="/home/yz325/PathwayBench"

# Input argument for dataset name (default to sam_road if not provided)
DATASET_NAME=${1:-sam_road}

# Directories
ANNOT_DIR="$ROOT_DIR/tests/$DATASET_NAME/annotations"
REVERSE_DIR="$ROOT_DIR/tests/$DATASET_NAME/reverse_edges_geojson"
OUTPUT_CSV="$ROOT_DIR/tests/$DATASET_NAME/results.csv"

# Write header to CSV
echo "file,TraversabilitySimilarity,Precision,Recall,F1" > "$OUTPUT_CSV"

# Loop over all *_edges.geojson in reverse dir
for reverse_file in "$REVERSE_DIR"/*.edges.geojson; do
# for reverse_file in "$REVERSE_DIR"/*_edges.geojson; do
    # Extract base filename without suffix
    base_name=$(basename "$reverse_file")
    base_name_no_ext="${base_name%.geojson}"

    # Matching file in annotations (must match exact filename)
    annot_file="$ANNOT_DIR/$base_name"

    if [[ -f "$annot_file" ]]; then
        echo "Processing $base_name_no_ext..."

        # Step 1: tessellate
        python "$ROOT_DIR/scripts/tessellate_area.py" "$annot_file"

        # Step 2: compute scores
        output=$(python "$ROOT_DIR/scripts/compute_scores.py" \
            "$ANNOT_DIR/${base_name_no_ext}_tip.geojson" \
            "$reverse_file" \
            "$annot_file")

        # Extract metrics
        ts=$(echo "$output" | grep "TraversabilitySimilarity" | awk -F: '{print $2}' | xargs)
        precision=$(echo "$output" | grep "Precision" | awk -F: '{print $2}' | xargs)
        recall=$(echo "$output" | grep "Recall" | awk -F: '{print $2}' | xargs)
        f1=$(echo "$output" | grep "F1" | awk -F: '{print $2}' | xargs)

        # Append to CSV
        echo "$base_name_no_ext,$ts,$precision,$recall,$f1" >> "$OUTPUT_CSV"
    else
        echo "Annotation file not found for $base_name_no_ext, skipping."
    fi
done

echo "Done! Results saved to $OUTPUT_CSV"
