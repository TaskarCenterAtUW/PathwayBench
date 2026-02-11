#!/bin/bash

# Usage: ./evaluate_sam.sh sam_road

# Root directory of your project
ROOT_DIR="/home/yz325/PathwayBench"

# Input argument for dataset name (default to sam_road if not provided)
DATASET_NAME=${1:-sam_road}

# Directories
ANNOT_DIR="$ROOT_DIR/tests/$DATASET_NAME/annotations"
REVERSE_DIR="$ROOT_DIR/tests/$DATASET_NAME/reverse_edges_geojson"
OUTPUT_EDGES_CSV="$ROOT_DIR/tests/$DATASET_NAME/edges_results.csv"
OUTPUT_NODES_CSV="$ROOT_DIR/tests/$DATASET_NAME/nodes_results.csv"

# Write headers
echo "file,TraversabilitySimilarity,Precision,Recall,F1" > "$OUTPUT_EDGES_CSV"
echo "file,Precision,Recall,F1,KerbError" > "$OUTPUT_NODES_CSV"

# Loop over all *_edges.geojson in reverse dir
for reverse_file in "$REVERSE_DIR"/*.edges.geojson; do
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
        tile_path="$ANNOT_DIR/${base_name_no_ext}_tip.geojson"

        # ---------- Edges ----------
        output_edges=$(python "$ROOT_DIR/scripts/compute_scores.py" \
            "$tile_path" \
            --edges-path "$reverse_file" \
            --gt-edges-path "$annot_file")

        # Extract metrics
        ts=$(echo "$output_edges" | grep "TraversabilitySimilarity" | awk -F: '{print $2}' | xargs)
        precision=$(echo "$output_edges" | grep "Precision" | head -n1 | awk -F: '{print $2}' | xargs)
        recall=$(echo "$output_edges" | grep "Recall" | head -n1 | awk -F: '{print $2}' | xargs)
        f1=$(echo "$output_edges" | grep "F1" | head -n1 | awk -F: '{print $2}' | xargs)

        echo "$base_name_no_ext,$ts,$precision,$recall,$f1" >> "$OUTPUT_EDGES_CSV"

        # ---------- Nodes (optional) ----------
        nodes_pred="$REVERSE_DIR/${base_name_no_ext/edges/nodes}.geojson"
        nodes_gt="$ANNOT_DIR/${base_name_no_ext/edges/nodes}.geojson"

        if [[ -f "$nodes_pred" && -f "$nodes_gt" ]]; then
            output_nodes=$(python "$ROOT_DIR/scripts/compute_scores.py" \
                "$tile_path" \
                --nodes-path "$nodes_pred" \
                --gt-nodes-path "$nodes_gt")

            n_precision=$(echo "$output_nodes" | grep "Precision" | head -n1 | awk -F: '{print $2}' | xargs)
            n_recall=$(echo "$output_nodes" | grep "Recall" | head -n1 | awk -F: '{print $2}' | xargs)
            n_f1=$(echo "$output_nodes" | grep "F1" | head -n1 | awk -F: '{print $2}' | xargs)
            kerb_error=$(echo "$output_nodes" | grep "Kerb Error" | head -n1 | awk -F: '{print $2}' | xargs)

            echo "$base_name_no_ext,$n_precision,$n_recall,$n_f1,$kerb_error" >> "$OUTPUT_NODES_CSV"
        else
            echo "Node files not found for $base_name_no_ext, skipping node metrics."
        fi
    else
        echo "Annotation file not found for $base_name_no_ext, skipping."
    fi
done

echo "Done! Edge results saved to $OUTPUT_EDGES_CSV"
echo "Done! Node results saved to $OUTPUT_NODES_CSV"
