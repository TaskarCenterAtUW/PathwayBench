import os
os.environ['USE_PYGEOS'] = '0'
import networkx as nx
import argparse
import sys
import copy
import json
import traceback
import geopandas as gpd
import osmnx as ox
import dask_geopandas
from statistics import stdev, mean
#from osmapi import OsmApi
import geonetworkx as gnx
from shapely import Point, LineString, MultiLineString, Polygon, MultiPolygon
from shapely.ops import voronoi_diagram, substring, nearest_points
from scipy.spatial import ConvexHull
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator
from tqdm import tqdm
import itertools
import numpy as np
import pandas as pd

from tessellate_area import create_tip
from summarize_stats import compute_aggregate_f1, compute_aggregate_avg_d, compute_tra_jaccard, create_score_json

import warnings
warnings.filterwarnings("ignore")


PROJ = 'epsg:26910'
PRES = 1e-5

BUFFER_SIZE = 5
E_THRESHOLD = 5
KERB_BUFFER_SIZE = 10
NODE_ERROR_OUTLIER_MEDIAN_MULTIPLIER = 2.0

OSM_ROAD_EDGES = None
OSM_ROAD_SINDEX = None
OSM_INTERSECTION_NODES = None

Distance_to_intersection = {
    "living_street": 5.9,
    "residential": 6.9,
    "tertiary": 7.1,
    "trunk": 7.7,
    "secondary": 7.7,
    "primary": 8.9,
    "unclassified": 10.0,
    "trunk_link": 10.3,
    "secondary_link": 13.3,
    "motorway_link": 15.8,
    "tertiary_link": 17.4,
    "primary_link": 19.8
}


def _normalize_highway_type(highway_value):
    if isinstance(highway_value, list):
        highway_values = highway_value
    elif isinstance(highway_value, str) and ";" in highway_value:
        highway_values = [v.strip() for v in highway_value.split(";")]
    else:
        highway_values = [highway_value]

    for hv in highway_values:
        if hv in Distance_to_intersection:
            return hv
    return None


def _get_intersection_point(crossing_geom, road_geom):
    inter = crossing_geom.intersection(road_geom)
    if inter.is_empty:
        # Fallback to closest point on crossing to the road
        pt_cross, _ = nearest_points(crossing_geom, road_geom)
        return pt_cross

    if inter.geom_type == "Point":
        return inter
    if inter.geom_type == "MultiPoint":
        return list(inter.geoms)[0]

    # For non-point intersections (line overlap, collection), pick a representative point
    return inter.representative_point()


def init_osm_road_context(tile_gdf):
    global OSM_ROAD_EDGES, OSM_ROAD_SINDEX, OSM_INTERSECTION_NODES

    if tile_gdf is None or tile_gdf.empty:
        return

    try:
        bounds = tile_gdf.total_bounds  # minx, miny, maxx, maxy in PROJ
        minx, miny, maxx, maxy = bounds

        # Expand fetch area slightly to cover boundary effects.
        expand_m = 50
        area_poly = Polygon([
            (minx - expand_m, miny - expand_m),
            (maxx + expand_m, miny - expand_m),
            (maxx + expand_m, maxy + expand_m),
            (minx - expand_m, maxy + expand_m),
        ])

        area_gdf = gpd.GeoDataFrame({"geometry": [area_poly]}, crs=PROJ).to_crs("epsg:4326")
        fetch_poly = area_gdf.iloc[0].geometry

        G_osm = ox.graph_from_polygon(fetch_poly, network_type="all", simplify=True, retain_all=True)
        road_edges = ox.graph_to_gdfs(G_osm, nodes=False, edges=True).to_crs(PROJ)
        road_edges = road_edges[road_edges["highway"].notna()].copy()

        node_gdf = ox.graph_to_gdfs(G_osm, nodes=True, edges=False).to_crs(PROJ)
        node_degree = dict(G_osm.degree())
        node_gdf["degree"] = node_gdf.index.map(lambda n: node_degree.get(n, 0))
        # Degree >= 3 is a practical proxy for intersections.
        intersection_nodes = node_gdf[node_gdf["degree"] >= 3].copy()

        OSM_ROAD_EDGES = road_edges
        OSM_ROAD_SINDEX = road_edges.sindex
        OSM_INTERSECTION_NODES = intersection_nodes
        print(
            f"Loaded OSM road context with {len(road_edges)} edges "
            f"and {len(intersection_nodes)} intersection nodes"
        )
    except Exception as e:
        # Fallback to default buffering if OSM fetch fails.
        OSM_ROAD_EDGES = None
        OSM_ROAD_SINDEX = None
        OSM_INTERSECTION_NODES = None
        print(f"Warning: OSM road context unavailable ({e}). Using default buffer for all edges.")


def save_osm_debug_geojson(edges_path):
    if OSM_ROAD_EDGES is None:
        return

    def sanitize_for_geojson_export(gdf):
        out = gdf.copy()
        for col in out.columns:
            if col == "geometry":
                continue
            # Fiona cannot serialize list/dict/tuple field types directly.
            out[col] = out[col].apply(
                lambda v: json.dumps(v) if isinstance(v, (list, dict, tuple)) else v
            )
        return out

    roads_path = edges_path.replace('.geojson', '_osm_roads.geojson')
    sanitize_for_geojson_export(OSM_ROAD_EDGES).to_file(roads_path, driver='GeoJSON')
    print(f'{roads_path} saved')

    if OSM_INTERSECTION_NODES is not None:
        intersections_path = edges_path.replace('.geojson', '_osm_intersections.geojson')
        sanitize_for_geojson_export(OSM_INTERSECTION_NODES).to_file(intersections_path, driver='GeoJSON')
        print(f'{intersections_path} saved')


def build_crossing_buffer_geometry(crossing_geom, default_buffer):
    # Requires global OSM context; fallback to default buffer if unavailable.
    if OSM_ROAD_EDGES is None or OSM_ROAD_SINDEX is None:
        return crossing_geom.buffer(default_buffer)

    try:
        candidate_idx = list(OSM_ROAD_SINDEX.intersection(crossing_geom.bounds))
        if len(candidate_idx) == 0:
            return crossing_geom.buffer(default_buffer)

        candidates = OSM_ROAD_EDGES.iloc[candidate_idx].copy()
        candidates["dist_to_crossing"] = candidates.geometry.distance(crossing_geom)
        candidates = candidates.sort_values("dist_to_crossing")

        best_row = None
        best_highway = None

        # Prefer truly intersecting roads first.
        intersecting = candidates[candidates.geometry.intersects(crossing_geom)]
        source_rows = intersecting if not intersecting.empty else candidates

        for _, row in source_rows.iterrows():
            highway = _normalize_highway_type(row.get("highway"))
            if highway is not None:
                best_row = row
                best_highway = highway
                break

        if best_row is None or best_highway is None:
            return crossing_geom.buffer(default_buffer)

        road_distance = Distance_to_intersection[best_highway]
        half_window = 0.8 * road_distance

        intersection_pt = _get_intersection_point(crossing_geom, best_row.geometry)
        proj_d = crossing_geom.project(intersection_pt)
        start_d = max(0.0, proj_d - half_window)
        end_d = min(crossing_geom.length, proj_d + half_window)

        if end_d <= start_d:
            return crossing_geom.buffer(default_buffer)

        crossing_segment = substring(crossing_geom, start_d, end_d)
        if crossing_segment.is_empty:
            return crossing_geom.buffer(default_buffer)

        return crossing_segment.buffer(default_buffer)
    except Exception:
        return crossing_geom.buffer(default_buffer)


def get_unmarked_crossing_gdf(gt_reference):
    if gt_reference is None or gt_reference.empty:
        return None
    if "footway" not in gt_reference.columns:
        return None

    marking_col = None
    for col_name in ["crossing:markings", "crossing_markings", "crossing_marking"]:
        if col_name in gt_reference.columns:
            marking_col = col_name
            break
    if marking_col is None:
        return None

    footway_series = gt_reference["footway"].astype(str).str.lower()
    marking_series = gt_reference[marking_col].astype(str).str.lower()
    mask = (footway_series == "crossing") & (marking_series == "no")
    if mask.sum() == 0:
        return None

    return gt_reference.loc[mask, ["geometry"]].copy()


def is_unmarked_crossing_edge(shape_geo, unmarked_crossings):
    if unmarked_crossings is None or unmarked_crossings.empty:
        return False

    try:
        candidate_idx = list(unmarked_crossings.sindex.intersection(shape_geo.bounds))
        if len(candidate_idx) == 0:
            return False
        candidates = unmarked_crossings.iloc[candidate_idx]
        # Small tolerance for slight geometric misalignments.
        return candidates.geometry.intersects(shape_geo.buffer(1.0)).any()
    except Exception:
        return False


def add_edges_from_linestring(graph, linestring, edge_attrs):
    """ Add edges to the NetworkX graph from a LineString. """
    points = list(linestring.coords)
    #points = [(round(x, 0), round(y, 0)) for x, y in points] #round to meter
    for start, end in zip(points[:-1], points[1:]):
        graph.add_edge(start, end, **edge_attrs)


def graph_from_gdf(gdf):
    # Initialize an empty graph
    G = nx.Graph()

    # Iterate through each row in the GeoDataFrame
    for index, row in gdf.iterrows():
        geom = row.geometry
        if isinstance(geom, LineString):
            add_edges_from_linestring(G, geom, row.to_dict())
        elif isinstance(geom, MultiLineString):
            for linestring in geom.geoms:
                add_edges_from_linestring(G, linestring, row.to_dict())
    return G


def group_G_pts(G, poly):
    P = poly
    
    node_pts = [Point(x) for x in G.nodes()]
    # Get polygon boundary as a list of line segments
    boundary = list(P.boundary.coords)
    segments = [LineString([boundary[i], boundary[i + 1]]) for i in range(len(boundary) - 1)]

    # Dictionary to hold points grouped by line segments
    segment_point_map = {index: [] for index in range(len(segments))}

    # Group points by which line segment they fall on
    for point in node_pts:
        for idx, segment in enumerate(segments):
            if segment.distance(point) < PRES:  # Small threshold for precision issues
                segment_point_map[idx].append((point.x, point.y))
                break
    return segment_point_map


def edges_are_connected(G, e1_pts, e2_pts):
    for pt1 in e1_pts:
        for pt2 in e2_pts: 
            # node self to self should be count at connected
            if pt1 != pt2:
                if nx.has_path(G, pt1, pt2):
                    return True
    return False 


def tile_tra_score(G, polygon):
    # compute number of connected edge pairs

    # assign each point to an polygon line
    pts_line_map = group_G_pts(G, polygon) 
    boundary_nodes = [item for sublist in pts_line_map.values() for item in sublist]

    # find all pair of edges
    #edge_pairs = list(itertools.combinations(pts_line_map.keys(), 2))
    edge_pairs = list(itertools.combinations_with_replacement(pts_line_map.keys(), 2))

    n_total = len(edge_pairs)
    n_connected = 0
    connected_pairs = list()
    for pair in edge_pairs:
        is_connected = edges_are_connected(G, pts_line_map[pair[0]], pts_line_map[pair[1]])
        if is_connected:
            #print(f'{pair} is connected')
            connected_pairs.append(pair)
            n_connected += 1

    #print(f'number of boundary pts {len(boundary_nodes)}')
    #print(f'{len(G.edges())}: {n_total} pair of edges, {n_connected} are connected')

    """
    # Plot the graph
    pos = {tuple(node): node for node in G.nodes()}

    plt.figure(figsize=(20, 15))  # Optional: Adjust figure size

    # first plot the polygon
    x, y = polygon.exterior.xy
    plt.plot(x, y, color='green', alpha=0.7, linewidth=1, solid_capstyle='round', zorder=2)
    # Annotate each edge with its index
    for i, (start, end) in enumerate(zip(polygon.exterior.coords[:-1], polygon.exterior.coords[1:])):
        # Find the midpoint of each edge
        midpoint = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
        # Draw the index number at the midpoint
        plt.text(midpoint[0], midpoint[1], str(i), color='green', fontsize=12, ha='center')

    nx.draw_networkx_nodes(G, pos, node_color='blue', node_size=5)  # Draw nodes
    nx.draw_networkx_edges(G, pos, alpha=0.5)  # Draw edges

    nx.draw_networkx_nodes(G, pos={node: pos[node] for node in boundary_nodes}, nodelist = boundary_nodes, node_color='red', node_size=20)

    """

    return n_total, n_connected, connected_pairs 


def compute_angle(line):
    def angle_for_linestring(ls):
        coords = list(ls.coords)
        if len(coords) < 2:
            return None
        start, end = coords[0], coords[-1]
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        return np.degrees(np.arctan2(dy, dx)) % 180  # Normalize to [0, 180)

    # If it's a LineString, compute directly
    if isinstance(line, LineString):
        angle = angle_for_linestring(line)
        if angle is None:
            raise ValueError("LineString must have at least two coordinates")
        return angle

    # If it's a MultiLineString, average the angles of all parts
    elif isinstance(line, MultiLineString):
        angles = [angle_for_linestring(part) for part in line.geoms]
        angles = [a for a in angles if a is not None]
        if not angles:
            raise ValueError("MultiLineString has no valid LineStrings with at least two coordinates")
        return np.mean(angles)

    else:
        raise TypeError(f"Unsupported geometry type: {type(line)}")


def compute_f1(pred, gt, buff_dis=5, e_thres=5, unmarked_crossings=None):
    angle_thres = 30
    match_thres = 10

    num_splits = 5

    tp = 0
    fp = 0
    avg_d_list = []

    pred_sw = pred
    gt_sw = gt
    for it, pred_it in pred_sw.iterrows():
        try:
            shape_geo = pred_it['geometry']
            pred_angle = compute_angle(shape_geo)

            use_unmarked_crossing_logic = isinstance(shape_geo, LineString) and is_unmarked_crossing_edge(
                shape_geo, unmarked_crossings
            )
            if use_unmarked_crossing_logic:
                shape_geo_dia = build_crossing_buffer_geometry(shape_geo, buff_dis)
            else:
                shape_geo_dia = shape_geo.buffer(buff_dis)

            pred_copy = copy.deepcopy(pred_it)
            pred_copy = pred_copy.to_frame().T.reset_index()
            pred_copy['geometry'] = shape_geo_dia
            pred_copy = gpd.GeoDataFrame(pred_copy, geometry=pred_copy['geometry'], crs=PROJ)

            inter = gt_sw.overlay(pred_copy, keep_geom_type=True, how='intersection')

            # Compute and filter by angle
            inter['angle'] = inter['geometry'].apply(compute_angle)
            inter = inter[inter['angle'].apply(lambda a: abs(a - pred_angle) < angle_thres)]

            pred_it_pts = [pred_it['geometry'].interpolate((i/num_splits), normalized=True) for i in range(1, num_splits)]
            # pred_it_pts_gdf = gpd.GeoDataFrame({'geometry': pred_it_pts}, crs=pred_copy.crs)

            avg_d = None
            if not inter.empty:
                # distance_matched = pred_it_pts_gdf.sjoin_nearest(inter, distance_col="distances", how="inner")
                # distance_lst = distance_matched['distances'].tolist()

                # union
                inter_union = inter.unary_union
                distance_lst = [pt.distance(inter_union) for pt in pred_it_pts]

                d_filter = [x for x in distance_lst if x <= match_thres]

                if len(d_filter) > 0:
                    avg_d = np.average(d_filter)

                if avg_d is not None and avg_d < e_thres:
                    tp += 1
                else:
                    fp += 1
                if avg_d is not None:
                    avg_d_list.append(avg_d)
            else:
                fp += 1

        except Exception as e:
            traceback.print_exc()
            #exit()
            continue

    agg_avg_d = -99.99
    if len(avg_d_list) > 0:
        agg_avg_d = float(np.average(avg_d_list))

    return tp, fp, agg_avg_d


def compute_f1_point_distance(pred, gt, dist_thres=4):
    tp = 0
    fp = 0

    if gt is None or gt.empty:
        return 0, len(pred)

    for it, pred_it in pred.iterrows():
        try:
            pred_pt = pred_it['geometry']
            nearest_dist = gt['geometry'].distance(pred_pt).min()

            # print(f"Nearest distance for prediction {it}: {nearest_dist:.2f}")

            if nearest_dist <= dist_thres:
                tp += 1
            else:
                fp += 1
        except Exception as e:
            # print(f"Error in processing prediction {it}: {e}")
            fp += 1
    return tp, fp


def compute_one_to_one_point_matches(pred, gt, dist_thres=4):
    if pred is None or gt is None or pred.empty or gt.empty:
        return [], set(), set()

    candidate_pairs = []
    gt_sindex = gt.sindex

    for pred_idx, pred_row in pred.iterrows():
        pred_geom = pred_row['geometry']
        if pred_geom is None or pred_geom.is_empty:
            continue

        candidate_positions = list(gt_sindex.intersection(pred_geom.buffer(dist_thres).bounds))
        for _, gt_row in gt.iloc[candidate_positions].iterrows():
            gt_geom = gt_row['geometry']
            if gt_geom is None or gt_geom.is_empty:
                continue

            distance = pred_geom.distance(gt_geom)
            if distance <= dist_thres:
                candidate_pairs.append((float(distance), pred_idx, gt_row.name))

    candidate_pairs.sort(key=lambda item: item[0])

    matches = []
    matched_pred = set()
    matched_gt = set()
    for distance, pred_idx, gt_idx in candidate_pairs:
        if pred_idx in matched_pred or gt_idx in matched_gt:
            continue
        matched_pred.add(pred_idx)
        matched_gt.add(gt_idx)
        matches.append((pred_idx, gt_idx, distance))

    return matches, matched_pred, matched_gt


def compute_by_id_node_matches(pred, gt):
    if pred is None or gt is None or pred.empty or gt.empty:
        return [], set(), set()
    if "_id" not in pred.columns or "_id" not in gt.columns:
        return [], set(), set()

    pred_with_id = pred[pred["_id"].notna()].copy()
    gt_with_id = gt[gt["_id"].notna()].copy()
    if pred_with_id.empty or gt_with_id.empty:
        return [], set(), set()

    pred_with_id["_match_id"] = pred_with_id["_id"].astype(str)
    gt_with_id["_match_id"] = gt_with_id["_id"].astype(str)

    shared_ids = set(pred_with_id["_match_id"]) & set(gt_with_id["_match_id"])
    candidate_pairs = []
    for match_id in shared_ids:
        pred_group = pred_with_id[pred_with_id["_match_id"] == match_id]
        gt_group = gt_with_id[gt_with_id["_match_id"] == match_id]
        for pred_idx, pred_row in pred_group.iterrows():
            pred_geom = pred_row["geometry"]
            if pred_geom is None or pred_geom.is_empty:
                continue
            for gt_idx, gt_row in gt_group.iterrows():
                gt_geom = gt_row["geometry"]
                if gt_geom is None or gt_geom.is_empty:
                    continue
                candidate_pairs.append((float(pred_geom.distance(gt_geom)), pred_idx, gt_idx))

    candidate_pairs.sort(key=lambda item: item[0])

    matches = []
    matched_pred = set()
    matched_gt = set()
    for distance, pred_idx, gt_idx in candidate_pairs:
        if pred_idx in matched_pred or gt_idx in matched_gt:
            continue
        matched_pred.add(pred_idx)
        matched_gt.add(gt_idx)
        matches.append((pred_idx, gt_idx, distance))

    return matches, matched_pred, matched_gt


def compute_global_node_stats(pred, gt, matching_mode, dist_thres=4, error_buffer_size=10):
    pred_count = 0 if pred is None else len(pred)
    gt_count = 0 if gt is None else len(gt)

    if matching_mode == "by_id":
        f1_matches, _, _ = compute_by_id_node_matches(pred, gt)
    else:
        f1_matches, _, _ = compute_one_to_one_point_matches(pred, gt, dist_thres=dist_thres)
    error_distribution = compute_node_error_distribution(
        pred,
        gt,
        buffer_size=error_buffer_size,
        matching_mode=matching_mode,
    )

    tp = len(f1_matches)
    return {
        "tp": tp,
        "fp": pred_count - tp,
        "fn": gt_count - tp,
        "node_error": summarize_node_error(error_distribution),
    }


def compute_node_error_distribution(pred, gt, buffer_size=10, matching_mode="standard"):
    columns = ["match_id", "pred_index", "gt_index", "distance_m", "matching_mode"]
    if gt is None or pred is None or gt.empty or pred.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    if matching_mode == "strict":
        matches, _, _ = compute_one_to_one_point_matches(pred, gt, dist_thres=buffer_size)
        for match_id, (pred_idx, gt_idx, distance) in enumerate(matches):
            rows.append({
                "match_id": match_id,
                "pred_index": pred_idx,
                "gt_index": gt_idx,
                "distance_m": float(distance),
                "matching_mode": "strict",
            })
        return pd.DataFrame(rows, columns=columns)

    if matching_mode == "by_id":
        matches, _, _ = compute_by_id_node_matches(pred, gt)
        for match_id, (pred_idx, gt_idx, distance) in enumerate(matches):
            rows.append({
                "match_id": match_id,
                "pred_index": pred_idx,
                "gt_index": gt_idx,
                "distance_m": float(distance),
                "matching_mode": "by_id",
            })
        return pd.DataFrame(rows, columns=columns)

    match_id = 0
    for gt_idx, gt_row in gt.iterrows():
        try:
            gt_geom = gt_row['geometry']
            if gt_geom is None:
                continue

            gt_buffer = gt_geom.buffer(buffer_size)
            candidates = pred[pred['geometry'].within(gt_buffer)]
            if candidates.empty:
                continue

            distances = candidates['geometry'].distance(gt_geom)
            pred_idx = distances.idxmin()
            rows.append({
                "match_id": match_id,
                "pred_index": pred_idx,
                "gt_index": gt_idx,
                "distance_m": float(distances.loc[pred_idx]),
                "matching_mode": "nearest-within-buffer",
            })
            match_id += 1
        except Exception:
            continue

    return pd.DataFrame(rows, columns=columns)


def summarize_node_error(distribution_df):
    distribution_df = filter_node_error_outliers(distribution_df)
    if distribution_df is None or distribution_df.empty:
        return 0.0
    return float(pd.to_numeric(distribution_df["distance_m"]).mean())


def node_error_outlier_threshold(distribution_df):
    if distribution_df is None or distribution_df.empty:
        return None

    distances = pd.to_numeric(distribution_df["distance_m"], errors="coerce")
    valid_distances = distances.dropna()
    if valid_distances.empty:
        return None

    median_distance = float(valid_distances.median())
    threshold = median_distance * NODE_ERROR_OUTLIER_MEDIAN_MULTIPLIER
    if threshold <= 0:
        threshold = float(valid_distances.max())
    return threshold


def filter_node_error_outliers(distribution_df):
    if distribution_df is None or distribution_df.empty:
        return distribution_df

    distances = pd.to_numeric(distribution_df["distance_m"], errors="coerce")
    threshold = node_error_outlier_threshold(distribution_df)
    if threshold is None:
        return distribution_df.iloc[0:0].copy()

    return distribution_df.loc[distances <= threshold].copy()


def compute_node_error(pred, gt, buffer_size=10):
    """Mean distance from each GT node to its nearest predicted node within a buffer."""
    distribution = compute_node_error_distribution(pred, gt, buffer_size=buffer_size, matching_mode="standard")
    return summarize_node_error(distribution)


def compute_id_summary(gdf, prefix):
    total = 0 if gdf is None else len(gdf)
    summary = {
        f"{prefix}_nodes_total": total,
        f"{prefix}_nodes_with_id": 0,
        f"{prefix}_nodes_missing_id": total,
        f"{prefix}_unique_id_count": 0,
        f"{prefix}_duplicate_id_count": 0,
        f"{prefix}_duplicate_node_count": 0,
    }
    if gdf is None or gdf.empty or "_id" not in gdf.columns:
        return summary

    ids = gdf["_id"].dropna().astype(str)
    counts = ids.value_counts()
    duplicate_counts = counts[counts > 1]

    summary[f"{prefix}_nodes_with_id"] = int(len(ids))
    summary[f"{prefix}_nodes_missing_id"] = int(total - len(ids))
    summary[f"{prefix}_unique_id_count"] = int(len(counts))
    summary[f"{prefix}_duplicate_id_count"] = int(len(duplicate_counts))
    summary[f"{prefix}_duplicate_node_count"] = int(duplicate_counts.sum() - len(duplicate_counts))
    return summary


def summarize_distance_distribution(distribution_df, prefix):
    summary = {
        f"{prefix}_match_count": 0,
        f"{prefix}_unique_pred_match_count": 0,
        f"{prefix}_unique_gt_match_count": 0,
        f"{prefix}_node_error_mean_m": 0.0,
        f"{prefix}_node_error_median_m": 0.0,
        f"{prefix}_node_error_stddev_m": 0.0,
        f"{prefix}_node_error_min_m": 0.0,
        f"{prefix}_node_error_max_m": 0.0,
    }
    if distribution_df is None or distribution_df.empty:
        return summary

    distances = pd.to_numeric(distribution_df["distance_m"], errors="coerce").dropna()
    summary[f"{prefix}_match_count"] = int(len(distribution_df))
    summary[f"{prefix}_unique_pred_match_count"] = int(distribution_df["pred_index"].nunique())
    summary[f"{prefix}_unique_gt_match_count"] = int(distribution_df["gt_index"].nunique())
    if distances.empty:
        return summary

    summary[f"{prefix}_node_error_mean_m"] = float(distances.mean())
    summary[f"{prefix}_node_error_median_m"] = float(distances.median())
    summary[f"{prefix}_node_error_stddev_m"] = float(distances.std(ddof=0))
    summary[f"{prefix}_node_error_min_m"] = float(distances.min())
    summary[f"{prefix}_node_error_max_m"] = float(distances.max())
    return summary


def compute_node_match_summary(pred, gt, raw_distribution, filtered_distribution, matching_mode, node_type, stats):
    pred_total = 0 if pred is None else len(pred)
    gt_total = 0 if gt is None else len(gt)
    raw_unique_pred = 0 if raw_distribution is None or raw_distribution.empty else int(raw_distribution["pred_index"].nunique())
    raw_unique_gt = 0 if raw_distribution is None or raw_distribution.empty else int(raw_distribution["gt_index"].nunique())
    filtered_unique_pred = 0 if filtered_distribution is None or filtered_distribution.empty else int(filtered_distribution["pred_index"].nunique())
    filtered_unique_gt = 0 if filtered_distribution is None or filtered_distribution.empty else int(filtered_distribution["gt_index"].nunique())

    threshold = node_error_outlier_threshold(raw_distribution)
    summary = {
        "node_type": node_type,
        "matching_mode": matching_mode,
        "f1_tp": int(stats.get("tp", 0)),
        "f1_fp": int(stats.get("fp", 0)),
        "f1_fn": int(stats.get("fn", 0)),
        "pred_nodes_skipped_raw": int(pred_total - raw_unique_pred),
        "gt_nodes_skipped_raw": int(gt_total - raw_unique_gt),
        "pred_nodes_skipped_after_outlier_filter": int(pred_total - filtered_unique_pred),
        "gt_nodes_skipped_after_outlier_filter": int(gt_total - filtered_unique_gt),
        "outlier_threshold_m": -99.99 if threshold is None else float(threshold),
        "outlier_removed_count": int(0 if raw_distribution is None else len(raw_distribution) - len(filtered_distribution)),
        "outlier_policy": f"distance_m <= {NODE_ERROR_OUTLIER_MEDIAN_MULTIPLIER} * median(distance_m)",
    }
    summary.update(compute_id_summary(pred, "pred"))
    summary.update(compute_id_summary(gt, "gt"))
    summary.update(summarize_distance_distribution(raw_distribution, "raw"))
    summary.update(summarize_distance_distribution(filtered_distribution, "filtered"))

    if pred is not None and gt is not None and "_id" in pred.columns and "_id" in gt.columns:
        pred_ids = set(pred["_id"].dropna().astype(str))
        gt_ids = set(gt["_id"].dropna().astype(str))
        shared_ids = pred_ids & gt_ids
        summary["shared_id_count"] = int(len(shared_ids))
        summary["pred_id_not_in_gt_count"] = int(len(pred_ids - gt_ids))
        summary["gt_id_not_in_pred_count"] = int(len(gt_ids - pred_ids))
    else:
        summary["shared_id_count"] = 0
        summary["pred_id_not_in_gt_count"] = 0
        summary["gt_id_not_in_pred_count"] = 0

    return summary


def save_node_match_summary(summary, summary_path):
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    print(f'{summary_path} saved')


def get_node_error_title_context(nodes_path, output_dir=None):
    if output_dir:
        return os.path.basename(os.path.normpath(output_dir))

    node_dir = os.path.dirname(nodes_path)
    node_dir_name = os.path.basename(os.path.normpath(node_dir))
    if node_dir_name == "reverse_edges_geojson":
        return os.path.basename(os.path.dirname(os.path.normpath(node_dir)))
    return node_dir_name


def save_node_error_outputs(distribution_df, distribution_path, histogram_path, title_context=None):
    distribution_df = filter_node_error_outliers(distribution_df)
    distribution_df.to_csv(distribution_path, index=False)
    print(f'{distribution_path} saved')

    title = "Node Error Distribution"
    if title_context:
        title = f"{title} ({title_context})"

    fig, ax = plt.subplots(figsize=(8, 5))
    distances = pd.to_numeric(distribution_df["distance_m"]) if not distribution_df.empty else pd.Series(dtype=float)
    distances = distances.dropna()
    if len(distances) > 0:
        mean_distance = float(distances.mean())
        std_distance = float(distances.std(ddof=0))
        median_distance = float(distances.median())
        center_distance = max(median_distance, 0.001)
        x_min = 0.0
        x_max = max(float(distances.max()), center_distance * NODE_ERROR_OUTLIER_MEDIAN_MULTIPLIER)

        bins = min(30, max(10, int(np.sqrt(len(distances)))))
        counts, bin_edges = np.histogram(distances, bins=bins, range=(x_min, x_max))
        bar_widths = np.diff(bin_edges)
        use_log_scale = len(counts) > 0 and counts.max() > 100
        if use_log_scale:
            log_floor = 0.8
            nonzero_counts = counts[counts > 0]
            ax.bar(
                bin_edges[:-1][counts > 0],
                nonzero_counts - log_floor,
                width=bar_widths[counts > 0],
                bottom=log_floor,
                align="edge",
                edgecolor="black",
                color="#4c78a8",
            )
            ax.set_yscale("log")
            ax.set_ylim(bottom=log_floor)
            ax.yaxis.set_major_locator(LogLocator(base=10))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
            ax.set_ylabel("Count (log scale)")
        else:
            ax.bar(
                bin_edges[:-1],
                counts,
                width=bar_widths,
                align="edge",
                edgecolor="black",
                color="#4c78a8",
            )
            ax.set_ylabel("Count")

        ax.axvline(mean_distance, color="#f58518", linestyle="--", linewidth=2, label=f"mean {mean_distance:.3f} m")
        ax.axvline(median_distance, color="#54a24b", linestyle=":", linewidth=2, label=f"median {median_distance:.3f} m")
        ax.plot([], [], " ", label=f"stddev {std_distance:.3f} m")
        ax.legend(loc="upper left")
        ax.set_xlim(x_min, x_max)
        ax.set_title(title)
    else:
        ax.text(0.5, 0.5, "No node error matches", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.set_ylabel("Count")

    ax.set_xlabel("Distance (m)")
    fig.tight_layout()
    fig.savefig(histogram_path, dpi=150)
    plt.close(fig)
    print(f'{histogram_path} saved')
    return distribution_df


def get_stats(polygon, G, gdf, gdf_gt):
    stats = {}
    undirected_g = nx.Graph(G)
    unmarked_crossings = get_unmarked_crossing_gdf(gdf_gt)

    # edge-to-edge connected paths
    try:
        # print('computing traversability !!!!!!!!!!!!!!!!!!!!')
        n_total, n_connected, connected_pairs = tile_tra_score(undirected_g, polygon)
        # print(n_total, n_connected, connected_pairs)
        stats["n_total_edges"] = n_total
        stats["n_connect_edges"] = n_connected
        connected_pairs_str = ' '.join([f"({t[0]},{t[1]})" for t in connected_pairs])
        stats['connected_pairs'] = connected_pairs_str
    except Exception as e:
        print(f"Unexpected {e}, {type(e)} with polygon {polygon} when getting number of connected edge pairs")
        traceback.print_exc()
        stats["n_total_edges"] = -99.99
        stats["n_connect_edges"] = -99.99
        stats['connected_pairs'] = "-99.99"
        exit()

    # f1 score
    try:
        tp, fp, avg_d = compute_f1(
            gdf, gdf_gt, buff_dis=BUFFER_SIZE, e_thres=E_THRESHOLD, unmarked_crossings=unmarked_crossings
        )
        tp, fn, _ = compute_f1(
            gdf_gt, gdf, buff_dis=BUFFER_SIZE, e_thres=E_THRESHOLD, unmarked_crossings=unmarked_crossings
        )
        # precision = tp/(tp+fp)
        # recall = tp/(tp+fn)
        # f1 = 2*(precision*recall)/(precision + recall)
        stats["tp"] = tp
        stats["fp"] = fp
        stats["fn"] = fn
        stats["avg_d"] = avg_d
    except Exception as e:
        #print(f"Unexpected {e}, {type(e)} with polygon {polygon} when getting f1 score")
        #traceback.print_exc()
        stats["tp"] = -99.99
        stats["fp"] = -99.99
        stats["fn"] = -99.99
        stats["avg_d"] = -99.99
    return stats


def get_node_stats(polygon, G, gdf, gdf_gt):
    stats = {}

    # f1 score
    try:
        tp, fp = compute_f1_point_distance(gdf, gdf_gt, dist_thres = E_THRESHOLD)
        tp, fn = compute_f1_point_distance(gdf_gt, gdf, dist_thres = E_THRESHOLD)
        # precision = tp/(tp+fp)
        # recall = tp/(tp+fn)
        # f1 = 2*(precision*recall)/(precision + recall)
        stats["tp"] = tp
        stats["fp"] = fp
        stats["fn"] = fn
    except Exception as e:
        #print(f"Unexpected {e}, {type(e)} with polygon {polygon} when getting f1 score")
        #traceback.print_exc()
        stats["tp"] = -99.99
        stats["fp"] = -99.99
        stats["fn"] = -99.99

    # node error
    try:
        node_error = compute_node_error(gdf, gdf_gt, buffer_size=KERB_BUFFER_SIZE)
        stats["node_error"] = node_error
    except Exception:
        stats["node_error"] = -99.99
    return stats


def get_measures_from_polygon(polygon, gdf, gdf_gt):
    if isinstance(polygon, MultiPolygon) and len(polygon.geoms)==1:
        polygon = polygon.geoms[0]

    # crop gdf to the polygon
    cropped_gdf = gpd.clip(gdf, polygon)
    cropped_gdf_gt = gpd.clip(gdf_gt, polygon)

    G = graph_from_gdf(cropped_gdf) 

    stats = get_stats(polygon, G, cropped_gdf, cropped_gdf_gt)
    return stats


def get_node_measures_from_polygon(polygon, gdf, gdf_gt):
    if isinstance(polygon, MultiPolygon) and len(polygon.geoms)==1:
        polygon = polygon.geoms[0]

    # crop gdf to the polygon
    cropped_gdf = gpd.clip(gdf, polygon)
    cropped_gdf_gt = gpd.clip(gdf_gt, polygon)

    G = graph_from_gdf(cropped_gdf) 

    stats = get_node_stats(polygon, G, cropped_gdf, cropped_gdf_gt)
    return stats


def compute_edge_score(feature, gdf, gdf_gt):
    poly = feature.geometry
    if (poly.geom_type == "Polygon" or poly.geom_type == "MultiPolygon"):
        measures = get_measures_from_polygon(poly, gdf, gdf_gt)
        feature.loc['total_edges'] = measures["n_total_edges"]
        feature.loc['connect_edges'] = measures["n_connect_edges"]
        feature.loc['connected_pairs'] = measures["connected_pairs"]
        feature.loc['tp'] = measures["tp"]
        feature.loc['fp'] = measures["fp"]
        feature.loc['fn'] = measures["fn"]
        feature.loc['avg_d'] = measures["avg_d"]
        return feature
    

def compute_node_score(feature, gdf, gdf_gt):
    poly = feature.geometry
    if (poly.geom_type == "Polygon" or poly.geom_type == "MultiPolygon"):
        measures = get_node_measures_from_polygon(poly, gdf, gdf_gt)
        feature.loc['tp'] = measures["tp"]
        feature.loc['fp'] = measures["fp"]
        feature.loc['fn'] = measures["fn"]
        feature.loc['node_error'] = measures.get("node_error", -99.99)
        return feature


def read_gdf(p):
    return gpd.read_file(p)


def normalize_node_type(node_type):
    node_type = node_type.lower()
    if node_type == "all":
        return "all"
    if node_type == "kerb":
        return "kerb"
    raise ValueError(f"Unsupported node type: {node_type}")


def filter_nodes_by_type(nodes_gdf, node_type):
    if node_type == "all":
        return nodes_gdf.copy()

    if "barrier" not in nodes_gdf.columns:
        return nodes_gdf.iloc[0:0].copy()

    return nodes_gdf[nodes_gdf["barrier"].astype(str).str.lower() == "kerb"].copy()


def compute_precision_recall_f1_from_counts(tp, fp, fn):
    precision_denom = tp + fp
    recall_denom = tp + fn

    precision = tp / precision_denom if precision_denom > 0 else 0.0
    recall = tp / recall_denom if recall_denom > 0 else 0.0
    f1_denom = precision + recall
    f1 = 2 * (precision * recall) / f1_denom if f1_denom > 0 else 0.0

    return np.round(precision, 3), np.round(recall, 3), np.round(f1, 3)


def node_stats_basename(node_type, matching_mode="standard"):
    label = "kerb" if node_type == "kerb" else "all_nodes"
    return f"{label}_{matching_mode}_stats" if matching_mode != "standard" else f"{label}_stats"


def node_error_artifact_base(nodes_path, node_type, matching_mode="standard", output_dir=None):
    label = "kerb" if node_type == "kerb" else "all_nodes"
    artifact_name = os.path.basename(nodes_path).replace(
        '.geojson',
        f'_{label}_{matching_mode}_node_error',
    )
    if output_dir is None:
        output_dir = os.path.dirname(nodes_path)
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, artifact_name)


def resolve_node_matching_mode(args):
    if args.node_strict:
        return "strict"
    return args.node_matching or "standard"


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Compute metrics for edges and/or nodes within tiles."
    )
    parser.add_argument("tile_path", help="Path to tile polygon GeoData (e.g., .geojson/.shp)")
    parser.add_argument("--edges-path", help="Path to predicted edges")
    parser.add_argument("--gt-edges-path", help="Path to ground-truth edges")
    parser.add_argument("--nodes-path", help="Path to predicted nodes")
    parser.add_argument("--gt-nodes-path", help="Path to ground-truth nodes")
    parser.add_argument("--e-threshold", type=float, default=5,
                        help="Edge threshold (float), default = 5")
    parser.add_argument("--node-strict", "--node_strict", action="store_true",
                        help="Use one-to-one closest-node matching across the whole node dataset.")
    parser.add_argument("--node-matching", "--node_matching",
                        choices=["standard", "strict", "by_id"],
                        help="Node matching mode. Defaults to standard, or strict when --node-strict is set.")
    parser.add_argument("--node-type", "--node_type", type=str.lower, choices=["all", "kerb"], default="kerb",
                        help="Which nodes to evaluate: All or Kerb. Default = Kerb")
    parser.add_argument("--output-dir", "--output_dir",
                        help="Directory for node error distribution CSV and histogram PNG artifacts.")

    args = parser.parse_args()

    if not args.edges_path and not args.nodes_path:
        parser.error("Provide at least edges (--edges-path & --gt-edges-path) or nodes (--nodes-path & --gt-nodes-path).")

    if (args.edges_path and not args.gt_edges_path) or (args.gt_edges_path and not args.edges_path):
        parser.error("If providing edges, you must pass BOTH --edges-path and --gt-edges-path.")

    if (args.nodes_path and not args.gt_nodes_path) or (args.gt_nodes_path and not args.nodes_path):
        parser.error("If providing nodes, you must pass BOTH --nodes-path and --gt-nodes-path.")

    tile_gdf = read_gdf(args.tile_path)

    edges_gdf = edges_gdf_gt = None
    if args.edges_path:
        edges_gdf = read_gdf(args.edges_path)
        edges_gdf_gt = read_gdf(args.gt_edges_path)

    nodes_gdf = nodes_gdf_gt = None
    if args.nodes_path:
        nodes_gdf = read_gdf(args.nodes_path)
        nodes_gdf_gt = read_gdf(args.gt_nodes_path)

    tile_gdf = tile_gdf.to_crs(PROJ)
    if edges_gdf is not None:
        edges_gdf = edges_gdf.to_crs(PROJ)
        edges_gdf_gt = edges_gdf_gt.to_crs(PROJ)
    if nodes_gdf is not None:
        nodes_gdf = nodes_gdf.to_crs(PROJ)
        nodes_gdf_gt = nodes_gdf_gt.to_crs(PROJ)

    E_THRESHOLD = args.e_threshold
    node_type = normalize_node_type(args.node_type)
    node_matching_mode = resolve_node_matching_mode(args)

    df_dask = dask_geopandas.from_geopandas(tile_gdf, npartitions=32)

    if edges_gdf is not None:
        init_osm_road_context(tile_gdf)
        save_osm_debug_geojson(args.edges_path)
        print('computing stats for edges...')
        output = df_dask.apply(compute_edge_score, axis=1, meta=[
            ('geometry', 'geometry'),
            ('total_edges', 'object'),
            ('connect_edges', 'object'),
            ('connected_pairs', 'object'),
            ('tp', 'object'),
            ('fp', 'object'),
            ('fn', 'object'),
            ('avg_d', 'object'),
            ], gdf=edges_gdf, gdf_gt=edges_gdf_gt).compute(scheduler='multiprocessing')
        
        edge_save_path = args.edges_path.replace('.geojson','_stats.geojson')
        output.to_file(edge_save_path, driver='GeoJSON')
        print(f'{edge_save_path} saved')

        output_gt = df_dask.apply(compute_edge_score, axis=1, meta=[
        ('geometry', 'geometry'),
        ('total_edges', 'object'),
        ('connect_edges', 'object'),
        ('connected_pairs', 'object'),
        ('tp', 'object'),
        ('fp', 'object'),
        ('fn', 'object'),
        ('avg_d', 'object'),
        ], gdf=edges_gdf_gt, gdf_gt=edges_gdf_gt).compute(scheduler='multiprocessing')
        
        gt_edge_save_path = args.gt_edges_path.replace('.geojson','_stats.geojson')
        output_gt.to_file(gt_edge_save_path, driver='GeoJSON')
        print(f'{gt_edge_save_path} saved')
        
        print('edge stats: ')
        pred_stats = gpd.read_file(edge_save_path)
        gt_stats = gpd.read_file(gt_edge_save_path)
        print(f"TraversabilitySimilarity: {compute_tra_jaccard(pred_stats, gt_stats)}")
        precision, recall, f1 = compute_aggregate_f1(pred_stats)
        avg_d = compute_aggregate_avg_d(pred_stats)
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1: {f1}")
        print(f"AvgD: {avg_d}")

        create_score_json(pred_stats, gt_stats)
        pred_stats.to_file(edge_save_path.replace('stats.geojson', 'scores.geojson'), driver="GeoJSON")

    if nodes_gdf is None:
        exit()

    pred_node_gdf = filter_nodes_by_type(nodes_gdf, node_type)
    gt_node_gdf = filter_nodes_by_type(nodes_gdf_gt, node_type)
    node_label = "kerb" if node_type == "kerb" else "all"

    if node_matching_mode != "standard":
        print(f'computing {node_matching_mode} stats for {node_label} nodes...')
        global_distribution = compute_node_error_distribution(
            pred_node_gdf,
            gt_node_gdf,
            buffer_size=KERB_BUFFER_SIZE,
            matching_mode=node_matching_mode,
        )
        global_stats = compute_global_node_stats(
            pred_node_gdf,
            gt_node_gdf,
            matching_mode=node_matching_mode,
            dist_thres=E_THRESHOLD,
            error_buffer_size=KERB_BUFFER_SIZE,
        )
        global_stats["node_error"] = summarize_node_error(global_distribution)
        global_output = gpd.GeoDataFrame(
            [global_stats],
            geometry=[tile_gdf.unary_union],
            crs=tile_gdf.crs,
        )
        global_save_path = args.nodes_path.replace(
            '.geojson',
            f'_{node_stats_basename(node_type, matching_mode=node_matching_mode)}.geojson',
        )
        global_output.to_file(global_save_path, driver='GeoJSON')
        print(f'{global_save_path} saved')
        global_error_base = node_error_artifact_base(
            args.nodes_path,
            node_type,
            matching_mode=node_matching_mode,
            output_dir=args.output_dir,
        )
        filtered_global_distribution = save_node_error_outputs(
            global_distribution,
            f'{global_error_base}_distribution.csv',
            f'{global_error_base}_histogram.png',
            title_context=get_node_error_title_context(args.nodes_path, args.output_dir),
        )
        global_summary = compute_node_match_summary(
            pred_node_gdf,
            gt_node_gdf,
            global_distribution,
            filtered_global_distribution,
            node_matching_mode,
            node_type,
            global_stats,
        )
        save_node_match_summary(global_summary, f'{global_error_base}_summary.csv')

        print(f'stats for {args.nodes_path} at threshold {E_THRESHOLD} meter')
        print(f'{node_label} node stats: ')
        precision, recall, f1 = compute_precision_recall_f1_from_counts(
            global_stats["tp"],
            global_stats["fp"],
            global_stats["fn"],
        )
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1: {f1}")
        print(f"Node Error: {global_stats['node_error']}")
        exit()

    print(f'computing stats for {node_label} nodes...')

    node_output = df_dask.apply(compute_node_score, axis=1, meta=[
    ('geometry', 'geometry'),
    ('tp', 'object'),
    ('fp', 'object'),
    ('fn', 'object'),
    ('node_error', 'object'),
    ], gdf=pred_node_gdf, gdf_gt=gt_node_gdf).compute(scheduler='multiprocessing')

    node_save_path = args.nodes_path.replace(
        '.geojson',
        f'_{node_stats_basename(node_type, matching_mode=node_matching_mode)}.geojson',
    )
    node_output.to_file(node_save_path, driver='GeoJSON')
    print(f'{node_save_path} saved')
    node_error_distribution = compute_node_error_distribution(
        pred_node_gdf,
        gt_node_gdf,
        buffer_size=KERB_BUFFER_SIZE,
        matching_mode=node_matching_mode,
    )
    node_error_base = node_error_artifact_base(
        args.nodes_path,
        node_type,
        matching_mode=node_matching_mode,
        output_dir=args.output_dir,
    )
    filtered_node_error_distribution = save_node_error_outputs(
        node_error_distribution,
        f'{node_error_base}_distribution.csv',
        f'{node_error_base}_histogram.png',
        title_context=get_node_error_title_context(args.nodes_path, args.output_dir),
    )
    node_summary = compute_node_match_summary(
        pred_node_gdf,
        gt_node_gdf,
        node_error_distribution,
        filtered_node_error_distribution,
        node_matching_mode,
        node_type,
        {
            "tp": node_output["tp"].sum(),
            "fp": node_output["fp"].sum(),
            "fn": node_output["fn"].sum(),
        },
    )
    save_node_match_summary(node_summary, f'{node_error_base}_summary.csv')

    curb_link_save_path = None
    if node_type == "kerb" and edges_gdf is not None:
        print('computing stats for curb and link nodes...')

        merge_forward = pd.merge(pred_node_gdf, edges_gdf, left_on='_id', right_on='_u_id')
        merge_forward = pd.merge(merge_forward, nodes_gdf, left_on='_v_id', right_on='_id', suffixes=('', '_matched'))

        merge_reverse = pd.merge(pred_node_gdf, edges_gdf, left_on='_id', right_on='_v_id')
        merge_reverse = pd.merge(merge_reverse, nodes_gdf, left_on='_u_id', right_on='_id', suffixes=('', '_matched'))

        pred_curb_link = pd.concat([merge_forward, merge_reverse], ignore_index=True)
        pred_curb_link = pred_curb_link.drop(['geometry_x', 'geometry_y'], axis=1)

        gt_merge_forward = pd.merge(gt_node_gdf, edges_gdf_gt, left_on='_id', right_on='_u_id')
        gt_merge_forward = pd.merge(gt_merge_forward, nodes_gdf_gt, left_on='_v_id', right_on='_id', suffixes=('', '_matched'))

        gt_merge_reverse = pd.merge(gt_node_gdf, edges_gdf_gt, left_on='_id', right_on='_v_id')
        gt_merge_reverse = pd.merge(gt_merge_reverse, nodes_gdf_gt, left_on='_u_id', right_on='_id', suffixes=('', '_matched'))

        gt_curb_link = pd.concat([gt_merge_forward, gt_merge_reverse], ignore_index=True)
        gt_curb_link = gt_curb_link.drop(['geometry_x', 'geometry_y'], axis=1)

        curb_link_output = df_dask.apply(compute_node_score, axis=1, meta=[
        ('geometry', 'geometry'),
        ('tp', 'object'),
        ('fp', 'object'),
        ('fn', 'object'),
        ('node_error', 'object'),
        ], gdf=pred_curb_link, gdf_gt=gt_curb_link).compute(scheduler='multiprocessing')

        curb_link_save_path = args.nodes_path.replace('.geojson','_curb_link_stats.geojson')
        curb_link_output.to_file(curb_link_save_path, driver='GeoJSON')
        print(f'{curb_link_save_path} saved')

    print(f'stats for {args.nodes_path if args.nodes_path else args.edges_path} at threshold {E_THRESHOLD} meter')

    node_stats = gpd.read_file(node_save_path)
    print(f'{node_label} node stats: ')
    precision, recall, f1 = compute_aggregate_f1(node_stats)
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1: {f1}")
    node_error_total = summarize_node_error(node_error_distribution)
    print(f"Node Error: {node_error_total}")

    if curb_link_save_path is not None:
        curb_link_node_stats = gpd.read_file(curb_link_save_path)
        print('curb and link node stats: ')
        precision, recall, f1 = compute_aggregate_f1(curb_link_node_stats)
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1: {f1}")
