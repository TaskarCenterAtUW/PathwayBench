#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path


def load_geojson(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if data.get("type") != "FeatureCollection":
        raise ValueError(f"{path} is not a FeatureCollection")
    return data


def is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def safe_float(value):
    if value is None:
        return None
    if is_number(value):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def to_json_safe(value):
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {k: to_json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_json_safe(v) for v in value]
    return value


def build_feature_map(features, match_by):
    mapping = {}
    if match_by == "index":
        for i, feat in enumerate(features):
            mapping[str(i)] = feat
        return mapping

    # geometry
    for feat in features:
        geom = feat.get("geometry")
        key = json.dumps(geom, sort_keys=True, separators=(",", ":"))
        mapping[key] = feat
    return mapping


def compute_diff(left_data, right_data, match_by):
    left_features = left_data.get("features", [])
    right_features = right_data.get("features", [])

    left_map = build_feature_map(left_features, match_by)
    right_map = build_feature_map(right_features, match_by)

    keys = sorted(set(left_map.keys()) | set(right_map.keys()))
    paired = []
    left_only = 0
    right_only = 0

    for key in keys:
        left_feat = left_map.get(key)
        right_feat = right_map.get(key)
        if left_feat is None:
            right_only += 1
            continue
        if right_feat is None:
            left_only += 1
            continue
        paired.append((key, left_feat, right_feat))

    left_cols = set()
    right_cols = set()
    for _, l, r in paired:
        left_cols.update(l.get("properties", {}).keys())
        right_cols.update(r.get("properties", {}).keys())

    cols = sorted(left_cols & right_cols)
    col_only_left = sorted(left_cols - right_cols)
    col_only_right = sorted(right_cols - left_cols)

    numeric_stats = {}
    non_numeric_stats = {}
    diff_features = []

    for col in cols:
        numeric_stats[col] = {
            "is_numeric": True,
            "n_compared": 0,
            "n_changed": 0,
            "sum_left": 0.0,
            "sum_right": 0.0,
            "sum_diff": 0.0,
            "sum_abs_diff": 0.0,
            "max_abs_diff": 0.0,
        }
        non_numeric_stats[col] = {"n_compared": 0, "n_changed": 0}

    for key, left_feat, right_feat in paired:
        lp = left_feat.get("properties", {})
        rp = right_feat.get("properties", {})

        out_props = {"match_key": key}
        for col in cols:
            lv = lp.get(col)
            rv = rp.get(col)

            lf = safe_float(lv)
            rf = safe_float(rv)

            if lf is not None and rf is not None:
                stats = numeric_stats[col]
                diff = rf - lf
                abs_diff = abs(diff)
                stats["n_compared"] += 1
                stats["sum_left"] += lf
                stats["sum_right"] += rf
                stats["sum_diff"] += diff
                stats["sum_abs_diff"] += abs_diff
                stats["max_abs_diff"] = max(stats["max_abs_diff"], abs_diff)
                if abs_diff > 0:
                    stats["n_changed"] += 1
                out_props[f"{col}_left"] = lv
                out_props[f"{col}_right"] = rv
                out_props[f"{col}_diff"] = diff
            else:
                numeric_stats[col]["is_numeric"] = False
                s = non_numeric_stats[col]
                s["n_compared"] += 1
                changed = lv != rv
                if changed:
                    s["n_changed"] += 1
                out_props[f"{col}_left"] = lv
                out_props[f"{col}_right"] = rv
                out_props[f"{col}_changed"] = changed

        diff_features.append(
            {
                "type": "Feature",
                "geometry": left_feat.get("geometry"),
                "properties": out_props,
            }
        )

    summary_numeric = {}
    summary_non_numeric = {}
    for col in cols:
        if numeric_stats[col]["is_numeric"]:
            s = numeric_stats[col]
            n = s["n_compared"] if s["n_compared"] > 0 else 1
            summary_numeric[col] = {
                "n_compared": s["n_compared"],
                "n_changed": s["n_changed"],
                "sum_left": s["sum_left"],
                "sum_right": s["sum_right"],
                "sum_diff": s["sum_diff"],
                "mean_abs_diff": s["sum_abs_diff"] / n,
                "max_abs_diff": s["max_abs_diff"],
            }
        else:
            summary_non_numeric[col] = non_numeric_stats[col]

    diff_geojson = {
        "type": "FeatureCollection",
        "features": diff_features,
    }
    # Preserve optional metadata fields commonly present in GeoJSON exports.
    if "name" in left_data:
        diff_geojson["name"] = left_data["name"]
    if "crs" in left_data:
        diff_geojson["crs"] = left_data["crs"]

    return {
        "n_left_features": len(left_features),
        "n_right_features": len(right_features),
        "n_paired_features": len(paired),
        "n_left_only_features": left_only,
        "n_right_only_features": right_only,
        "columns_in_both": cols,
        "columns_left_only": col_only_left,
        "columns_right_only": col_only_right,
        "numeric_summary": summary_numeric,
        "non_numeric_summary": summary_non_numeric,
        "diff_geojson": diff_geojson,
    }


def print_summary(summary):
    print(f"Left features: {summary['n_left_features']}")
    print(f"Right features: {summary['n_right_features']}")
    print(f"Paired features: {summary['n_paired_features']}")
    print(f"Left-only features: {summary['n_left_only_features']}")
    print(f"Right-only features: {summary['n_right_only_features']}")
    print(f"Common columns ({len(summary['columns_in_both'])}): {summary['columns_in_both']}")
    if summary["columns_left_only"]:
        print(f"Columns only in left: {summary['columns_left_only']}")
    if summary["columns_right_only"]:
        print(f"Columns only in right: {summary['columns_right_only']}")

    print("\nNumeric columns summary:")
    if not summary["numeric_summary"]:
        print("  (none)")
    else:
        for col, s in summary["numeric_summary"].items():
            print(
                f"  {col}: n_changed={s['n_changed']}/{s['n_compared']}, "
                f"sum_diff={s['sum_diff']:.6f}, mean_abs_diff={s['mean_abs_diff']:.6f}, "
                f"max_abs_diff={s['max_abs_diff']:.6f}"
            )

    print("\nNon-numeric columns summary:")
    if not summary["non_numeric_summary"]:
        print("  (none)")
    else:
        for col, s in summary["non_numeric_summary"].items():
            print(f"  {col}: n_changed={s['n_changed']}/{s['n_compared']}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute column-wise diff between two GeoJSON FeatureCollections."
    )
    parser.add_argument("left_geojson", help="Path to left GeoJSON")
    parser.add_argument("right_geojson", help="Path to right GeoJSON")
    parser.add_argument(
        "--match-by",
        choices=["index", "geometry"],
        default="index",
        help="How to pair features between files (default: index).",
    )
    parser.add_argument(
        "--out-geojson",
        help="Optional path to write per-feature diff GeoJSON.",
    )
    parser.add_argument(
        "--out-json",
        help="Optional path to write summary JSON.",
    )
    args = parser.parse_args()

    left = load_geojson(args.left_geojson)
    right = load_geojson(args.right_geojson)
    summary = compute_diff(left, right, args.match_by)

    print_summary(summary)

    if args.out_geojson:
        out_path = Path(args.out_geojson)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(to_json_safe(summary["diff_geojson"]), f, indent=2, allow_nan=False)
        print(f"\nSaved per-feature diff GeoJSON to: {out_path}")

    if args.out_json:
        out_summary = dict(summary)
        out_summary.pop("diff_geojson", None)
        out_path = Path(args.out_json)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(to_json_safe(out_summary), f, indent=2, allow_nan=False)
        print(f"Saved summary JSON to: {out_path}")


if __name__ == "__main__":
    main()
