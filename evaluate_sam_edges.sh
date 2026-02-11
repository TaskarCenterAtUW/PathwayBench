#!/bin/bash

# Usage: ./evaluate_sam_edges.sh <dataset_name>

ROOT_DIR="/home/yz325/PathwayBench"
DATASET_NAME=${1:-sam_road}

ANNOT_DIR="$ROOT_DIR/tests/$DATASET_NAME/annotations"
REVERSE_DIR="$ROOT_DIR/tests/$DATASET_NAME/reverse_edges_geojson"
OUTPUT_EDGES_CSV="$ROOT_DIR/tests/$DATASET_NAME/edges_results.csv"

echo "file,TraversabilitySimilarity,Precision,Recall,F1" > "$OUTPUT_EDGES_CSV"

for reverse_file in "$REVERSE_DIR"/*.edges.geojson; do
    base_name=$(basename "$reverse_file")
    base_name_no_ext="${base_name%.geojson}"
    annot_file="$ANNOT_DIR/$base_name"

    if [[ -f "$annot_file" ]]; then
        echo "Processing $base_name_no_ext..."
        tile_path="$ANNOT_DIR/${base_name_no_ext}_tip.geojson"

        python "$ROOT_DIR/scripts/tessellate_area.py" "$annot_file"

        output_edges=$(python "$ROOT_DIR/scripts/compute_scores.py" \
            "$tile_path" \
            --edges-path "$reverse_file" \
            --gt-edges-path "$annot_file")

        ts=$(echo "$output_edges" | grep "TraversabilitySimilarity" | awk -F: '{print $2}' | xargs)
        precision=$(echo "$output_edges" | grep "Precision" | head -n1 | awk -F: '{print $2}' | xargs)
        recall=$(echo "$output_edges" | grep "Recall" | head -n1 | awk -F: '{print $2}' | xargs)
        f1=$(echo "$output_edges" | grep "F1" | head -n1 | awk -F: '{print $2}' | xargs)

        echo "$base_name_no_ext,$ts,$precision,$recall,$f1" >> "$OUTPUT_EDGES_CSV"
    else
        echo "Annotation file not found for $base_name_no_ext, skipping."
    fi
done

echo "Done! Edge results saved to $OUTPUT_EDGES_CSV"
