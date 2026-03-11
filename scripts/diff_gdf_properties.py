#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path


def load_feature_collection(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if data.get("type") != "FeatureCollection":
        raise ValueError(f"{path} is not a GeoJSON FeatureCollection")
    return data


def is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def to_float(value):
    if value is None:
        return None
    if is_number(value):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def json_safe(value):
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    return value


def geometry_key(feature):
    return json.dumps(feature.get("geometry"), sort_keys=True, separators=(",", ":"))


def build_diff_features(left_features, right_features, allow_geometry_mismatch):
    if len(left_features) != len(right_features):
        raise ValueError(
            f"Feature count mismatch: left={len(left_features)} right={len(right_features)}"
        )

    diff_features = []
    for idx, (left_f, right_f) in enumerate(zip(left_features, right_features)):
        left_geom = left_f.get("geometry")
        right_geom = right_f.get("geometry")
        if not allow_geometry_mismatch and geometry_key(left_f) != geometry_key(right_f):
            raise ValueError(f"Geometry mismatch at feature index {idx}")

        left_props = left_f.get("properties", {}) or {}
        right_props = right_f.get("properties", {}) or {}
        all_cols = sorted(set(left_props.keys()) | set(right_props.keys()))

        diff_props = {}
        for col in all_cols:
            left_val = left_props.get(col)
            right_val = right_props.get(col)

            left_num = to_float(left_val)
            right_num = to_float(right_val)

            # Numeric values: element-wise diff (left - right).
            if left_num is not None and right_num is not None:
                diff_props[col] = left_num - right_num
            else:
                # Non-numeric values: 0 if equal, 1 if changed.
                diff_props[f"{col}__changed"] = 0 if left_val == right_val else 1

        diff_features.append(
            {
                "type": "Feature",
                "geometry": left_geom,
                "properties": diff_props,
            }
        )

    return diff_features


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute per-feature, per-property diffs between two GeoJSON FeatureCollections. "
            "Numeric diff is left - right."
        )
    )
    parser.add_argument("left_geojson", help="Left input GeoJSON (minuend)")
    parser.add_argument("right_geojson", help="Right input GeoJSON (subtrahend)")
    parser.add_argument("out_geojson", help="Output diff GeoJSON path")
    parser.add_argument(
        "--allow-geometry-mismatch",
        action="store_true",
        help="Allow geometry mismatch at same feature index (geometry from left is preserved).",
    )
    args = parser.parse_args()

    left = load_feature_collection(args.left_geojson)
    right = load_feature_collection(args.right_geojson)

    diff_features = build_diff_features(
        left.get("features", []),
        right.get("features", []),
        args.allow_geometry_mismatch,
    )

    out_fc = {
        "type": "FeatureCollection",
        "features": diff_features,
    }

    # Preserve GeoJSON metadata for better compatibility with GIS tools.
    if "name" in left:
        out_fc["name"] = f"{left['name']}_diff"
    if "crs" in left:
        out_fc["crs"] = left["crs"]

    out_path = Path(args.out_geojson)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(json_safe(out_fc), f, indent=2, allow_nan=False)

    print(f"Saved per-feature diff GeoJSON: {out_path}")


if __name__ == "__main__":
    main()
