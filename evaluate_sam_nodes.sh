#!/bin/bash

# Usage: ./evaluate_sam_nodes.sh [dataset_name] [--node-matching standard|strict|by_id] [--node-strict|--node_strict] [--node-type|--node_type All|Kerb]

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_NAME="sam_road"
DATASET_NAME_SET=false
NODE_STRICT=false
NODE_MATCHING="standard"
NODE_TYPE="Kerb"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --node-strict|--node_strict)
            NODE_STRICT=true
            NODE_MATCHING="strict"
            shift
            ;;
        --node-matching|--node_matching)
            if [[ $# -lt 2 ]]; then
                echo "Missing value for --node-matching. Use standard, strict, or by_id."
                exit 1
            fi
            NODE_MATCHING="$2"
            shift 2
            ;;
        --node-matching=*|--node_matching=*)
            NODE_MATCHING="${1#*=}"
            shift
            ;;
        --node-type|--node_type)
            if [[ $# -lt 2 ]]; then
                echo "Missing value for --node-type. Use All or Kerb."
                exit 1
            fi
            NODE_TYPE="$2"
            shift 2
            ;;
        --node-type=*|--node_type=*)
            NODE_TYPE="${1#*=}"
            shift
            ;;
        -h|--help)
            echo "Usage: ./evaluate_sam_nodes.sh [dataset_name] [--node-matching standard|strict|by_id] [--node-strict|--node_strict] [--node-type|--node_type All|Kerb]"
            exit 0
            ;;
        *)
            if [[ "$DATASET_NAME_SET" == false ]]; then
                DATASET_NAME="$1"
                DATASET_NAME_SET=true
                shift
            else
                echo "Unknown argument: $1"
                exit 1
            fi
            ;;
    esac
done

NODE_ARGS=(--node-type "$NODE_TYPE" --node-matching "$NODE_MATCHING")

ANNOT_DIR="$ROOT_DIR/tests/$DATASET_NAME/annotations"
REVERSE_DIR="$ROOT_DIR/tests/$DATASET_NAME/reverse_edges_geojson"
OUTPUT_NODES_CSV="$ROOT_DIR/tests/$DATASET_NAME/nodes_results.csv"

echo "file,Precision,Recall,F1,NodeError" > "$OUTPUT_NODES_CSV"

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
        --gt-nodes-path "$gt_nodes" \
        --output-dir "$ROOT_DIR/tests/$DATASET_NAME" \
        "${NODE_ARGS[@]}")

    n_precision=$(echo "$output_nodes" | grep "Precision" | head -n1 | awk -F: '{print $2}' | xargs)
    n_recall=$(echo "$output_nodes" | grep "Recall" | head -n1 | awk -F: '{print $2}' | xargs)
    n_f1=$(echo "$output_nodes" | grep "F1" | head -n1 | awk -F: '{print $2}' | xargs)
    node_error=$(echo "$output_nodes" | grep "Node Error" | head -n1 | awk -F: '{print $2}' | xargs)

    echo "$base_name_no_ext,$n_precision,$n_recall,$n_f1,$node_error" >> "$OUTPUT_NODES_CSV"
done

echo "Done! Node results saved to $OUTPUT_NODES_CSV"
