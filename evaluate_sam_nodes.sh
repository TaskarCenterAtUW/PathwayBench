#!/bin/bash

# Usage: ./evaluate_sam_nodes.sh <dataset_name>

ROOT_DIR="/home/yz325/PathwayBench"
DATASET_NAME=${1:-sam_road}

ANNOT_DIR="$ROOT_DIR/tests/$DATASET_NAME/annotations"
REVERSE_DIR="$ROOT_DIR/tests/$DATASET_NAME/reverse_edges_geojson"
OUTPUT_NODES_CSV="$ROOT_DIR/tests/$DATASET_NAME/nodes_results.csv"

echo "file,Precision,Recall,F1,KerbError" > "$OUTPUT_NODES_CSV"

for pred_nodes in "$REVERSE_DIR"/*.nodes.geojson; do
    base_name=$(basename "$pred_nodes")
    base_name_no_ext="${base_name%.geojson}"

    gt_nodes="$ANNOT_DIR/$base_name"
    if [[ ! -f "$gt_nodes" ]]; then
        echo "GT node file not found for $base_name_no_ext, skipping."
        continue
    fi

    # tile derived from matching edges name
    edge_base="${base_name_no_ext/nodes/edges}"
    annot_edges="$ANNOT_DIR/$edge_base.geojson"
    tile_path="$ANNOT_DIR/${edge_base}_tip.geojson"

    if [[ ! -f "$tile_path" ]]; then
        if [[ -f "$annot_edges" ]]; then
            echo "Tessellating $annot_edges for tile..."
            python "$ROOT_DIR/scripts/tessellate_area.py" "$annot_edges"
        else
            echo "Annotation edge file not found for $edge_base, skipping."
            continue
        fi
    fi

    echo "Processing $base_name_no_ext..."
    output_nodes=$(python "$ROOT_DIR/scripts/compute_scores.py" \
        "$tile_path" \
        --nodes-path "$pred_nodes" \
        --gt-nodes-path "$gt_nodes")

    n_precision=$(echo "$output_nodes" | grep "Precision" | head -n1 | awk -F: '{print $2}' | xargs)
    n_recall=$(echo "$output_nodes" | grep "Recall" | head -n1 | awk -F: '{print $2}' | xargs)
    n_f1=$(echo "$output_nodes" | grep "F1" | head -n1 | awk -F: '{print $2}' | xargs)
    kerb_error=$(echo "$output_nodes" | grep "Kerb Error" | head -n1 | awk -F: '{print $2}' | xargs)

    echo "$base_name_no_ext,$n_precision,$n_recall,$n_f1,$kerb_error" >> "$OUTPUT_NODES_CSV"
done

echo "Done! Node results saved to $OUTPUT_NODES_CSV"
