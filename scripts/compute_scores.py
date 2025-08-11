import os
os.environ['USE_PYGEOS'] = '0'
import networkx as nx
import argparse
import sys
import copy
import traceback
import geopandas as gpd
import osmnx as ox
import dask_geopandas
from statistics import stdev, mean
#from osmapi import OsmApi
import geonetworkx as gnx
from shapely import Point, LineString, MultiLineString, Polygon, MultiPolygon
from shapely.ops import voronoi_diagram
from scipy.spatial import ConvexHull
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
import itertools
import numpy as np
import pandas as pd

from tessellate_area import create_tip
from summarize_stats import compute_aggregate_f1, compute_tra_jaccard

import warnings
warnings.filterwarnings("ignore")


PROJ = 'epsg:26910'
PRES = 1e-5

BUFFER_SIZE = 5
E_THRESHOLD = 5


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


def compute_f1(pred, gt, buff_dis=5, e_thres=5):
    angle_thres = 30
    match_thres = 10

    num_splits = 5

    tp = 0
    fp = 0

    pred_sw = pred
    gt_sw = gt
    for it, pred_it in pred_sw.iterrows():
        try:
            shape_geo = pred_it['geometry']
            pred_angle = compute_angle(shape_geo)
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

            if not inter.empty:
                # distance_matched = pred_it_pts_gdf.sjoin_nearest(inter, distance_col="distances", how="inner")
                # distance_lst = distance_matched['distances'].tolist()

                # union
                inter_union = inter.unary_union
                distance_lst = [pt.distance(inter_union) for pt in pred_it_pts]

                d_filter = [x for x in distance_lst if x <= match_thres]

                if len(d_filter) > 0:
                    avg_d = np.average(d_filter)
                else:
                    avg_d = 99999

                if avg_d < e_thres:
                    tp += 1
                else:
                    fp += 1

        except Exception as e:
            traceback.print_exc()
            #exit()
            continue

    return tp, fp


def compute_f1_point_distance(pred, gt, dist_thres=4):
    tp = 0
    fp = 0

    for it, pred_it in pred.iterrows():
        try:
            pred_pt = pred_it['geometry']
            gt['dist'] = gt['geometry'].distance(pred_pt)
            nearest_dist = gt['dist'].min()

            # print(f"Nearest distance for prediction {it}: {nearest_dist:.2f}")

            if nearest_dist <= dist_thres:
                tp += 1
            else:
                fp += 1
        except Exception as e:
            # print(f"Error in processing prediction {it}: {e}")
            fp += 1
    return tp, fp


def get_stats(polygon, G, gdf, gdf_gt):
    stats = {}
    undirected_g = nx.Graph(G)

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
        tp, fp = compute_f1(gdf, gdf_gt, buff_dis=BUFFER_SIZE, e_thres = E_THRESHOLD)
        tp, fn = compute_f1(gdf_gt, gdf, buff_dis=BUFFER_SIZE, e_thres = E_THRESHOLD)
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
        return feature
    

def compute_node_score(feature, gdf, gdf_gt):
    poly = feature.geometry
    if (poly.geom_type == "Polygon" or poly.geom_type == "MultiPolygon"):
        measures = get_node_measures_from_polygon(poly, gdf, gdf_gt)
        feature.loc['tp'] = measures["tp"]
        feature.loc['fp'] = measures["fp"]
        feature.loc['fn'] = measures["fn"]
        return feature


def read_gdf(p):
    return gpd.read_file(p)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Load tile + edges (+ optional nodes) and project to CRS."
    )
    # required
    parser.add_argument("tile_path", help="Path to tile polygon GeoData (e.g., .geojson/.shp)")
    parser.add_argument("edges_path", help="Path to predicted edges")
    parser.add_argument("gt_edges_path", help="Path to ground-truth edges")

    # optional positional pair
    parser.add_argument("nodes_path", nargs="?", help="Path to predicted nodes (optional)")
    parser.add_argument("gt_nodes_path", nargs="?", help="Path to ground-truth nodes (optional)")

    # optional flag with default
    parser.add_argument("--e-threshold", type=float, default=5,
                        help="Edge threshold (float), default = 0.5")

    args = parser.parse_args()

    # Read required
    tile_gdf = read_gdf(args.tile_path)
    edges_gdf = read_gdf(args.edges_path)
    edges_gdf_gt = read_gdf(args.gt_edges_path)

    # Optional nodes (all-or-nothing)
    nodes_gdf = nodes_gdf_gt = None
    if args.nodes_path or args.gt_nodes_path:
        if not (args.nodes_path and args.gt_nodes_path):
            parser.error("If providing nodes, you must pass BOTH NODES_PATH and GT_NODES_PATH.")
        nodes_gdf = read_gdf(args.nodes_path)
        nodes_gdf_gt = read_gdf(args.gt_nodes_path)

    # Reproject everything provided
    tile_gdf = tile_gdf.to_crs(PROJ)
    edges_gdf = edges_gdf.to_crs(PROJ)
    edges_gdf_gt = edges_gdf_gt.to_crs(PROJ)
    if nodes_gdf is not None:
        nodes_gdf = nodes_gdf.to_crs(PROJ)
        nodes_gdf_gt = nodes_gdf_gt.to_crs(PROJ)

    # Threshold is always set now
    E_THRESHOLD = args.e_threshold


    # compute local stats
    df_dask = dask_geopandas.from_geopandas(tile_gdf, npartitions=32)

    print('computing stats for edges...')
    # Pred vs GT
    output = df_dask.apply(compute_edge_score, axis=1, meta=[
        ('geometry', 'geometry'),
        ('total_edges', 'object'),
        ('connect_edges', 'object'),
        ('connected_pairs', 'object'),
        ('tp', 'object'),
        ('fp', 'object'),
        ('fn', 'object'),
        ], gdf=edges_gdf, gdf_gt=edges_gdf_gt).compute(scheduler='multiprocessing')
    
    edge_save_path = args.edges_path.replace('.geojson','_stats.geojson')
    output.to_file(edge_save_path, driver='GeoJSON')
    print(f'{edge_save_path} saved')

    # GT vs GT, for traversability
    output_gt = df_dask.apply(compute_edge_score, axis=1, meta=[
    ('geometry', 'geometry'),
    ('total_edges', 'object'),
    ('connect_edges', 'object'),
    ('connected_pairs', 'object'),
    ('tp', 'object'),
    ('fp', 'object'),
    ('fn', 'object'),
    ], gdf=edges_gdf_gt, gdf_gt=edges_gdf_gt).compute(scheduler='multiprocessing')
    
    gt_edge_save_path = args.gt_edges_path.replace('.geojson','_stats.geojson')
    output_gt.to_file(gt_edge_save_path, driver='GeoJSON')
    print(f'{gt_edge_save_path} saved')
    
    # Run sequentially using .apply instead of Dask
    # output_gt = tile_gdf.apply(compute_edge_score, axis=1, args=(edges_gdf_gt, edges_gdf_gt))
    # output_gt.to_file(gt_edges_path.split('/')[-1].replace('.geojson','_stats.geojson'), driver='GeoJSON')

    # Compute and print summary stats for edges
    print('edge stats: ')
    pred_stats = gpd.read_file(edge_save_path)
    gt_stats = gpd.read_file(gt_edge_save_path)
    print(f"TraversabilitySimilarity: {compute_tra_jaccard(pred_stats, gt_stats)}")
    precision, recall, f1 = compute_aggregate_f1(pred_stats)
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1: {f1}")

    if not args.nodes_path:
        exit()

    print('computing stats for curb nodes...')

    pred_curb_gdf = nodes_gdf[nodes_gdf['ext:node_type'] == 'curb']
    gt_curb_gdf = nodes_gdf_gt[nodes_gdf_gt['barrier'] == 'kerb']

    curb_output = df_dask.apply(compute_node_score, axis=1, meta=[
    ('geometry', 'geometry'),
    ('tp', 'object'),
    ('fp', 'object'),
    ('fn', 'object'),
    ], gdf=pred_curb_gdf, gdf_gt=gt_curb_gdf).compute(scheduler='multiprocessing')

    curb_node_save_path = args.nodes_path.replace('.geojson','_curb_stats.geojson')
    curb_output.to_file(curb_node_save_path, driver='GeoJSON')
    print(f'{curb_node_save_path} saved')


    print('computing stats for curb and link nodes...')
    ## find link nodes and join with curb nodes
    # filter prediction
    # First join: node_df1['_id'] == edge_df['_u_id']
    merge_forward = pd.merge(pred_curb_gdf, edges_gdf, left_on='_id', right_on='_u_id')
    merge_forward = pd.merge(merge_forward, nodes_gdf, left_on='_v_id', right_on='_id', suffixes=('', '_matched'))

    # Second join: node_df1['_id'] == edge_df['_v_id']
    merge_reverse = pd.merge(pred_curb_gdf, edges_gdf, left_on='_id', right_on='_v_id')
    merge_reverse = pd.merge(merge_reverse, nodes_gdf, left_on='_u_id', right_on='_id', suffixes=('', '_matched'))

    # Combine both sets of matches
    pred_curb_link = pd.concat([merge_forward, merge_reverse], ignore_index=True)

    pred_curb_link = pred_curb_link.drop(['geometry_x', 'geometry_y'], axis=1)

    # pred_curb_link = pred_curb_link.to_crs('epsg:26910')
    # pred_curb_link.to_file(nodes_path.split('/')[-1].replace('.geojson','_curbs_links.geojson'), driver='GeoJSON')
    

    # filter gt
    # First join: node_df1['_id'] == edge_df['_u_id']
    gt_merge_forward = pd.merge(gt_curb_gdf, edges_gdf_gt, left_on='_id', right_on='_u_id')
    gt_merge_forward = pd.merge(gt_merge_forward, nodes_gdf_gt, left_on='_v_id', right_on='_id', suffixes=('', '_matched'))

    # Second join: node_df1['_id'] == edge_df['_v_id']
    gt_merge_reverse = pd.merge(gt_curb_gdf, edges_gdf_gt, left_on='_id', right_on='_v_id')
    gt_merge_reverse = pd.merge(gt_merge_reverse, nodes_gdf_gt, left_on='_u_id', right_on='_id', suffixes=('', '_matched'))

    # Combine both sets of matches
    gt_curb_link = pd.concat([gt_merge_forward, gt_merge_reverse], ignore_index=True)

    gt_curb_link = gt_curb_link.drop(['geometry_x', 'geometry_y'], axis=1)

    # gt_curb_link = gt_curb_link.to_crs('epsg:26910')
    # gt_curb_link.to_file(gt_nodes_path.split('/')[-1].replace('.geojson','_curbs_links.geojson'), driver='GeoJSON')

    curb_link_output = df_dask.apply(compute_node_score, axis=1, meta=[
    ('geometry', 'geometry'),
    ('tp', 'object'),
    ('fp', 'object'),
    ('fn', 'object'),
    ], gdf=pred_curb_link, gdf_gt=gt_curb_link).compute(scheduler='multiprocessing')

    curb_link_save_path = args.nodes_path.replace('.geojson','_curb_link_stats.geojson')
    curb_link_output.to_file(curb_link_save_path, driver='GeoJSON')
    print(f'{curb_link_save_path} saved')

    print(f'stats for {args.edges_path} at threshold {E_THRESHOLD} meter')

    # Compute and print summary stats for nodes
    curb_node_stats = gpd.read_file(curb_node_save_path)
    print('curb node stats: ')
    precision, recall, f1 = compute_aggregate_f1(curb_node_stats)
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1: {f1}")

    curb_link_node_stats = gpd.read_file(curb_link_save_path)
    print('curb and link node stats: ')
    precision, recall, f1 = compute_aggregate_f1(curb_link_node_stats)
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1: {f1}")





