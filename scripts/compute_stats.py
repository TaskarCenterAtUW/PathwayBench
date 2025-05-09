import os
os.environ['USE_PYGEOS'] = '0'
import networkx as nx
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

import warnings
warnings.filterwarnings("ignore")


PROJ = 'epsg:26910'
PRES = 1e-5

def bounding_box_from_gdf(gdf):
    # Get the bounding box coordinates directly from the GeoDataFrame
    min_x, min_y, max_x, max_y = gdf.total_bounds

    # Create a Polygon from the bounding box coordinates
    bounding_box_polygon = Polygon([(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)])

    return bounding_box_polygon


def create_voronoi_diagram(G_roads_simplified, bounds):
    # first thin the nodes 
    gdf_roads_simplified = gnx.graph_edges_to_gdf(G_roads_simplified)
    voronoi = voronoi_diagram(gdf_roads_simplified.boundary.unary_union, envelope = bounds)
    voronoi_gdf = gpd.GeoDataFrame({"geometry": voronoi.geoms})
    voronoi_gdf = voronoi_gdf.set_crs(gdf_roads_simplified.crs)
    voronoi_gdf_clipped = gpd.clip(voronoi_gdf, bounds)
    voronoi_gdf_clipped = voronoi_gdf_clipped.to_crs(PROJ)
    
    return voronoi_gdf_clipped


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


def hull_connected_paths(G):
    # G is a NetworkX graph, with nodes being tuples of (longitude, latitude)

    pos = {node: node for node in G.nodes()}

    points = np.array([node for node in G.nodes()])

    # Calculate the convex hull
    hull = ConvexHull(points)

    # Get the vertices of the convex hull
    hull_vertices = points[hull.vertices]
    hull_nodes = [tuple(point) for point in hull_vertices] 

    # find all parirs
    hull_vertices_pairs = list(itertools.combinations(hull_nodes, 2))

    n_total = len(hull_nodes)
    n_connected = 0
    c_pairs = list()
    for pair in hull_vertices_pairs:
        is_connected = nx.has_path(G, pair[0], pair[1])
        if is_connected:
            n_connected += 1
            if pair[0] not in c_pairs:
                c_pairs.append(pair[0])
            if pair[1] not in c_pairs:
                c_pairs.append(pair[1])
    #print(f'total hull nodes: {len(hull_nodes)}, connected pair hull nodes: {n_connected}')

    """
    # Plot the graph
    pos = {tuple(node): node for node in G.nodes()}
    plt.figure(figsize=(8, 6))  # Optional: Adjust figure size
    nx.draw_networkx_nodes(G, pos, node_color='blue', node_size=5)  # Draw nodes
    nx.draw_networkx_edges(G, pos, alpha=0.5)  # Draw edges

    nx.draw_networkx_nodes(G, pos={node: pos[node] for node in hull_nodes}, nodelist = hull_nodes, node_color='red', node_size=10)

    # Optionally, draw the convex hull as a polygon if desired
    #for simplex in hull.simplices:
    #    plt.plot(points[simplex, 0], points[simplex, 1], 'k-', linewidth=2)

    nx.draw_networkx_nodes(G, pos={node: pos[node] for node in c_pairs}, nodelist = c_pairs, node_color='green', node_size=10)

    plt.savefig('tile2net_test_graph_3.png')
    exit()
    """

    return n_total, n_connected


def clip_gdf(gdf, poly):
    P = poly

    # Clip the GeoDataFrame
    gdf_clipped = gpd.clip(gdf, P)

    # Prepare to store intersection points
    intersection_points = []

    # Loop through the clipped geometries to find intersection points with the polygon boundary
    for geometry in gdf_clipped.geometry:
        intersection = geometry.intersection(P.boundary)
        if isinstance(intersection, Point):
            intersection_points.append(intersection)
        elif isinstance(intersection, LineString):
            # No intersection points if the intersection is a LineString
            continue
        else:
            # If multiple points (or other geometries), handle appropriately
            for geom in intersection.geoms:
                if isinstance(geom, Point):
                    intersection_points.append(geom)

    return gdf_clipped, intersection_points


def group_pts(pts, poly):
    P = poly
    intersection_points = pts
    # Get polygon boundary as a list of line segments
    boundary = list(P.boundary.coords)
    segments = [LineString([boundary[i], boundary[i + 1]]) for i in range(len(boundary) - 1)]

    # Dictionary to hold points grouped by line segments
    segment_point_map = {index: [] for index in range(len(segments))}

    # Group points by which line segment they fall on
    for point in intersection_points:
        for idx, segment in enumerate(segments):
            if segment.distance(point) < PRES:  # Small threshold for precision issues
                segment_point_map[idx].append((point.x, point.y))
                break
    return segment_point_map


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


def compute_f1(pred, gt, e_thres=5, buff_dis=5):
    angle_thres = 30
    match_thres = 10
    e_thres = 5
    buff_dis = 5

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


def compute_angle(line):
    start, end = line.coords[0], line.coords[-1]
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    return np.degrees(np.arctan2(dy, dx)) % 180  # Normalize to 0–180°


def compute_angle_from_centerline(rect: Polygon):
    coords = list(rect.exterior.coords)[:-1]  # remove duplicate closing point

    # Find all edges
    edges = [(coords[i], coords[i+1]) for i in range(4)]  # rectangle assumed

    # Compute lengths and identify longer edges
    lengths = [np.hypot(x2 - x1, y2 - y1) for (x1, y1), (x2, y2) in edges]
    edge_pairs = [(edges[i], edges[(i+2)%4]) for i in range(2)]  # 0-2 and 1-3 are opposite pairs

    # Find the pair with the longer edge
    if lengths[0] > lengths[1]:
        (e1, e2) = edge_pairs[0]
    else:
        (e1, e2) = edge_pairs[1]

    # Get midpoints of the longer opposing edges
    def midpoint(a, b):
        return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)

    mid1 = midpoint(*e1)
    mid2 = midpoint(*e2)

    # Angle of centerline
    dx = mid2[0] - mid1[0]
    dy = mid2[1] - mid1[1]
    return np.degrees(np.arctan2(dy, dx)) % 180


# for dev purpose, pred is now a polygon insetead of centerline
def compute_f1_iou_polygon(pred, gt, buff_dis=4, iou_thres=0.1, angle_thres=30):
    tp = 0
    fp = 0

    # print(pred)
    # print(gt)

    if pred.empty:
        if gt.empty:
            tp += 1
        return tp, fp 

    if gt.empty:
        if pred.empty:
            tp += 1
        else: 
            fp += 1
        return tp, fp 


    for it, pred_it in pred.iterrows():
        try:
            # add buffer to pred
            shape_geo = pred_it['geometry']
            if isinstance(shape_geo, Polygon):
                pred_buffer = shape_geo
                # pred_angle = compute_angle_from_centerline(shape_geo)
            elif isinstance(shape_geo, LineString):
                pred_buffer = shape_geo.buffer(buff_dis, cap_style=2)
                # pred_angle = compute_angle(shape_geo)

            area_thres = 0.1

            # add buffer to gt
            if isinstance(gt.iloc[0]['geometry'], Polygon):
                gt_buffered = gt
            elif isinstance(gt.iloc[0]['geometry'], LineString):
                gt_buffered = gt.copy()
                gt_buffered['geometry'] = gt_buffered['geometry'].buffer(buff_dis, cap_style=2)

            # filter by overlap
            gt_buffered['intersect_area'] = gt_buffered['geometry'].intersection(pred_buffer).area

            # print(gt_buffered)

            # Filter: only keep rows where intersection area exceeds a threshold
            gt_filtered = gt[gt_buffered['intersect_area'] > area_thres].copy()

            if gt_filtered.empty:
                print(f"No GT segments intersect prediction {pred_it}")
                fp += 1
                continue

            # # Compute and filter by angle
            # if isinstance(gt.iloc[0]['geometry'], Polygon):
            #     gt_filtered['angle'] = gt_filtered['geometry'].apply(compute_angle_from_centerline)
            #     gt_filtered = gt_filtered[gt_filtered['angle'].apply(lambda a: abs(a - pred_angle) < angle_thres)]
            # elif isinstance(gt.iloc[0]['geometry'], LineString):
            #     gt_filtered['angle'] = gt_filtered['geometry'].apply(compute_angle)
            #     gt_filtered = gt_filtered[gt_filtered['angle'].apply(lambda a: abs(a - pred_angle) < angle_thres)]

            if gt_filtered.empty:
                print(f"No directionally aligned GT segments for prediction {pred_it}")
                fp += 1
                continue

            # Union the nearby GT segments and buffer them
            if isinstance(gt.iloc[0]['geometry'], Polygon):
                gt_union_geom = gt_filtered.unary_union
            elif isinstance(gt.iloc[0]['geometry'], LineString):
                gt_union_geom = gt_filtered.unary_union.buffer(buff_dis, cap_style=2)

            # Compute IOU
            intersection_area = pred_buffer.intersection(gt_union_geom).area
            union_area = pred_buffer.union(gt_union_geom).area
            iou = intersection_area / union_area if union_area != 0 else 0

            # print(f"IOU for prediction {it}: {iou:.3f}")

            if iou > iou_thres:
                tp += 1
            else:
                fp += 1

        except Exception as e:
            print(f"Error in processing prediction {it}: {e}")
            traceback.print_exc()
            fp += 1

    return tp, fp


def compute_f1_iou(pred, gt, buff_dis=4, iou_thres=0.1, angle_thres=30):
    # buff_dis=0.5
    # angle_thres= 10 
    # area_thres = 0.01 # just to exclue lines
    tp = 0
    fp = 0

    # print("Equal:", pred.equals(gt))                  # Strict: everything must match
    # print("CRS Equal:", pred.crs == gt.crs)           # Check CRS
    # print("Geometry Equal:", pred.geometry.equals(gt.geometry))  # Just geometry
    # print("IDs Equal:", all(pred['_id'] == gt['_id']))  # ID columns

    for it, pred_it in pred.iterrows():
        try:
            shape_geo = pred_it['geometry']
            pred_buffer = shape_geo.buffer(buff_dis, cap_style=2)
            pred_angle = compute_angle(shape_geo)

            # Filter GT lines that intersect the prediction buffer
            # gt_filtered = gt[gt['geometry'].buffer(buff_dis, cap_style=2).intersects(pred_buffer)].copy()

            # Set dynamic area threshold based on segment length
            area_thres = shape_geo.length * buff_dis * 0.25

            # Buffer ground truth geometries
            gt_buffered = gt.copy()
            gt_buffered['geometry'] = gt_buffered['geometry'].buffer(buff_dis, cap_style=2)

            # Compute intersection area with pred_buffer
            gt_buffered['intersect_area'] = gt_buffered['geometry'].intersection(pred_buffer).area

            # print(gt_buffered)

            # Filter: only keep rows where intersection area exceeds a threshold
            gt_filtered = gt[gt_buffered['intersect_area'] > area_thres].copy()

            if gt_filtered.empty:
                # print(f"No GT segments intersect prediction with sufficient area {pred_it} ")
                fp += 1
                continue

            # Compute and filter by angle
            gt_filtered['angle'] = gt_filtered['geometry'].apply(compute_angle)
            gt_filtered = gt_filtered[gt_filtered['angle'].apply(lambda a: abs(a - pred_angle) < angle_thres)]

            if gt_filtered.empty:
                # print(f"No directionally aligned GT segments for prediction {pred_it}")
                fp += 1
                continue
            
            # Union the nearby GT segments and buffer them
            gt_union_geom = gt_filtered.unary_union.buffer(buff_dis, cap_style=2)


            # Compute IOU

            intersection_area = pred_buffer.intersection(gt_union_geom).area
            union_area = pred_buffer.union(gt_union_geom).area
            iou = intersection_area / union_area if union_area != 0 else 0


            # print(f'pred {pred_buffer.area}')
            # print(f'gt {gt_union_geom.area}')
            # print(f'intersection {intersection_area}')
            # print(f'union {union_area}')
            # print(f'iou {iou}')


            # print(f"IOU for prediction {it}: {iou:.3f}")

            if iou > iou_thres:
                tp += 1
            else:
                print(iou)
                print(pred_it['_id'])
                fp += 1

                ## debug #################################
                ## Save for debugging
                # debug_pred_buffer = gpd.GeoDataFrame({'geometry': [pred_buffer]}, crs=gt.crs)
                # gt_buffered = gt.copy()
                # gt_buffered['geometry'] = gt_buffered['geometry'].buffer(buff_dis, cap_style=2)
                # debug_gt_buffered = gt_buffered

                # # Optionally save to disk
                # debug_pred_buffer.to_file("debug_pred_buffer.geojson", driver="GeoJSON")
                # debug_gt_buffered.to_file("debug_gt_buffered.geojson", driver="GeoJSON")

                # ## Wrap into GeoDataFrame for export or inspection
                # debug_gt_union = gpd.GeoDataFrame({'geometry': [gt_union_geom]}, crs=gt_filtered.crs)
                # # Optionally save to GeoJSON for debugging
                # debug_gt_union.to_file("debug_gt_union_buffer.geojson", driver="GeoJSON")
                
                # sidewalk_id = pred_it['_id']
                # gt_filtered.to_file(f'{sidewalk_id}_gt_union.geojson', driver='GeoJSON')
                ############################################

        except Exception as e:
            print(f"Error in processing prediction {pred_it}: {e}")
            fp += 1

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


def get_stats_polygon(polygon, G, gdf, gdf_gt):
    stats = {}

    # f1 score
    try:
        tp, fp = compute_f1_iou_polygon(gdf, gdf_gt,buff_dis=2.5, iou_thres=0.1, angle_thres=30)
        tp, fn = compute_f1_iou_polygon(gdf_gt, gdf,buff_dis=2.5, iou_thres=0.1, angle_thres=30)

        stats["tp"] = tp
        stats["fp"] = fp
        # stats["fn"] = fn
        stats["fn"] = 0
    except Exception as e:
        #print(f"Unexpected {e}, {type(e)} with polygon {polygon} when getting f1 score")
        #traceback.print_exc()
        stats["tp"] = -99.99
        stats["fp"] = -99.99
        stats["fn"] = -99.99

    return stats



def get_stats(polygon, G, gdf, gdf_gt):
    stats = {}
    undirected_g = nx.Graph(G)

    
    # if undirected_g.number_of_nodes() > 0 and undirected_g.number_of_edges() > 0:
    #     # # betweenness
    #     # try:
    #     #     bet = nx.betweenness_centrality(undirected_g, normalized = True, endpoints=False)
    #     #     stats["bet_centrality_avg"] = mean(bet.values())
    #     #     stats["bet_stdev"] = stdev(bet.values())
    #     # except Exception as e:
    #     #     #print(f"Unexpected {e}, {type(e)} with polygon {polygon} when getting betweenness value")
    #     #     stats["bet_centrality_avg"] = -99.99
    #     #     stats["bet_stdev"] = -99.99
    #     #     traceback.print_exc()

    #     # # eigen
    #     # try:
    #     #     eigen = nx.eigenvector_centrality(undirected_g, max_iter=1000)
    #     #     stats["eig_centrality_avg"] = mean(eigen.values())
    #     # except Exception as e:
    #     #     #print(f"Unexpected {e}, {type(e)} with polygon {polygon} when getting eigen value")
    #     #     stats["eig_centrality_avg"] = -99.99
    #     #     traceback.print_exc()

    #     # # degree
    #     # try:
    #     #     deg = nx.degree_centrality(undirected_g)
    #     #     stats["deg_centrality_avg"] = mean(deg.values())
    #     # except Exception as e:
    #     #     #print(f"Unexpected {e}, {type(e)} with polygon {polygon} when getting degree cventrality")
    #     #     stats["deg_centrality_avg"] = -99.99

    #     # # number of connected components
    #     # try:
    #     #     noc = nx.number_connected_components(undirected_g)
    #     #     stats["num_connect_comp_avg"] = noc
    #     # except Exception as e:
    #     #     #print(f"Unexpected {e}, {type(e)} with polygon {polygon} when getting number of connected components")
    #     #     #traceback.print_exc()
    #     #     stats["num_connect_comp_avg"] = -99.99

    #     # # node connectivity 
    #     # try:
    #     #     conn = nx.average_node_connectivity(undirected_g)
    #     #     stats["node_connect_avg"] = conn
    #     # except Exception as e:
    #     #     #print(f"Unexpected {e}, {type(e)} with polygon {polygon} when getting node connectivity")
    #     #     #traceback.print_exc()
    #     #     stats["node_connect_avg"] = -99.99

    #     # # node-to-node connected paths
    #     # try:
    #     #     _, n_pahts = hull_connected_paths(undirected_g)
    #     #     stats["n_connect_paths"] = n_pahts
    #     # except Exception as e:
    #     #     #print(f"Unexpected {e}, {type(e)} with polygon {polygon} when getting number of connected paths")
    #     #     #traceback.print_exc()
    #     #     stats["n_connect_paths"] = -99.99

    #     # edge-to-edge connected paths
    #     # try:
    #     #     n_total, n_connected, connected_pairs = tile_tra_score(undirected_g, polygon)
    #     #     stats["n_total_edges"] = n_total
    #     #     stats["n_connect_edges"] = n_connected
    #     #     connected_pairs_str = ' '.join([f"({t[0]},{t[1]})" for t in connected_pairs])
    #     #     stats['connected_pairs'] = connected_pairs_str
    #     # except Exception as e:
    #     #     print(f"Unexpected {e}, {type(e)} with polygon {polygon} when getting number of connected edge pairs")
    #     #     traceback.print_exc()
    #     #     stats["n_total_edges"] = -99.99
    #     #     stats["n_connect_edges"] = -99.99
    #     #     stats['connected_pairs'] = "-99.99"
    # else:
    #     # stats["bet_centrality_avg"] = -99.99
    #     # stats["bet_stdev"] = -99.99
    #     # stats["eig_centrality_avg"] = -99.99
    #     # stats["deg_centrality_avg"] = -99.99
    #     # stats["num_connect_comp_avg"] = -99.99
    #     # stats["node_connect_avg"] = -99.99
    #     # stats["n_connect_paths"] = -99.99
    #     stats["n_total_edges"] = -99.99
    #     stats["n_connect_edges"] = -99.99
    #     stats['connected_pairs'] = "-99.99"

    # f1 score
    try:
        # print("Equal:", gdf.equals(gdf_gt))                  # Strict: everything must match
        # print("CRS Equal:", gdf.crs == gdf_gt.crs)           # Check CRS
        # print("Geometry Equal:", gdf.geometry.equals(gdf_gt.geometry))  # Just geometry
        # print("IDs Equal:", all(gdf['_id'] == gdf_gt['_id']))  # ID columns

        tp, fp = compute_f1_iou(gdf, gdf_gt, buff_dis=2.5, iou_thres=0.1, angle_thres=10)
        tp, fn = compute_f1_iou(gdf_gt, gdf, buff_dis=2.5, iou_thres=0.1, angle_thres=10)

        # tp, fp = compute_f1(gdf, gdf_gt)
        # tp, fn = compute_f1(gdf_gt, gdf)
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

    distance = 2.5
    # f1 score
    try:
        tp, fp = compute_f1_point_distance(gdf, gdf_gt, distance)
        tp, fn = compute_f1_point_distance(gdf_gt, gdf, distance)
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
    # stats = get_stats_polygon(polygon, G, cropped_gdf, cropped_gdf_gt)

    #direct_trust_score, time_trust_score = analyze_sidewalk_data(G)
    #stats["direct_trust_score"] = direct_trust_score
    #stats["time_trust_score"] = time_trust_score

    #stats['indirect_values'] = get_indirect_trust_score_from_polygon(polygon)
    
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


def compute_global_stats(filepath):
    gdf = gpd.read_file(filepath)
    G = graph_from_gdf(gdf)

    # (1) Total number of nodes
    total_nodes = G.number_of_nodes()

    # (2) Total number of edges
    total_edges = G.number_of_edges()

    # (3) Average degree of nodes
    average_degree = sum(dict(G.degree()).values()) / total_nodes

    print(f'computing global stats for {filepath}')
    print(f"Total number of nodes: {total_nodes}")
    print(f"Total number of edges: {total_edges}")
    print(f"Average degree of nodes: {average_degree}")


def compute_edge_score(feature, gdf, gdf_gt):

    poly = feature.geometry
    if (poly.geom_type == "Polygon" or poly.geom_type == "MultiPolygon"):
        measures = get_measures_from_polygon(poly, gdf, gdf_gt)

        # feature.loc['degree'] = measures["deg_centrality_avg"]
        # feature.loc['eigen'] = measures["eig_centrality_avg"]
        # feature.loc['betweenness'] = measures["bet_centrality_avg"]
        # feature.loc['bet_stdev'] = measures["bet_stdev"]
        # feature.loc['noc'] = measures["num_connect_comp_avg"]
        # feature.loc['conn'] = measures["node_connect_avg"]
        # feature.loc['n_path'] = measures["n_connect_paths"]

        # feature.loc['total_edges'] = measures["n_total_edges"]
        # feature.loc['connect_edges'] = measures["n_connect_edges"]
        # feature.loc['connected_pairs'] = measures["connected_pairs"]
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


if __name__ == '__main__':
    edges_path = sys.argv[1]
    nodes_path = sys.argv[2]
    gt_edges_path = sys.argv[3]
    gt_nodes_path = sys.argv[4]
    tile_gdf = gpd.read_file(sys.argv[5])

    edges_gdf = gpd.read_file(edges_path)
    edges_gdf_gt = gpd.read_file(gt_edges_path)

    nodes_gdf = gpd.read_file(nodes_path)
    nodes_gdf_gt = gpd.read_file(gt_nodes_path)

    edges_gdf = edges_gdf.to_crs(PROJ)
    edges_gdf_gt = edges_gdf_gt.to_crs(PROJ)
    nodes_gdf = nodes_gdf.to_crs(PROJ)
    nodes_gdf_gt = nodes_gdf_gt.to_crs(PROJ)
    tile_gdf = tile_gdf.to_crs(PROJ)

    # compute local stats
    df_dask = dask_geopandas.from_geopandas(tile_gdf, npartitions=1)

    print('computing stats for edges...')
    # output = df_dask.apply(compute_edge_score, axis=1, meta=[
    #     ('geometry', 'geometry'),
    #     # ('total_edges', 'object'),
    #     # ('connect_edges', 'object'),
    #     # ('connected_pairs', 'object'),
    #     ('tp', 'object'),
    #     ('fp', 'object'),
    #     ('fn', 'object'),
    #     ], gdf=edges_gdf, gdf_gt=edges_gdf_gt).compute(scheduler='multiprocessing')
    
    # output.to_file(edges_path.split('/')[-1].replace('.geojson','_stats.geojson'), driver='GeoJSON')

    # edges_gdf_gt = edges_gdf_gt[edges_gdf_gt['_id'] == '846896']
    # edges_gdf_gt = edges_gdf_gt[edges_gdf_gt['_id'] == '959833']

    output_gt = df_dask.apply(compute_edge_score, axis=1, meta=[
    ('geometry', 'geometry'),
    # ('total_edges', 'object'),
    # ('connect_edges', 'object'),
    # ('connected_pairs', 'object'),
    ('tp', 'object'),
    ('fp', 'object'),
    ('fn', 'object'),
    ], gdf=edges_gdf_gt, gdf_gt=edges_gdf_gt).compute(scheduler='multiprocessing')
    
    output_gt.to_file(gt_edges_path.split('/')[-1].replace('.geojson','_stats.geojson'), driver='GeoJSON')
    
    # Run sequentially using .apply instead of Dask
    # output_gt = tile_gdf.apply(compute_edge_score, axis=1, args=(edges_gdf_gt, edges_gdf_gt))
    # output_gt.to_file(gt_edges_path.split('/')[-1].replace('.geojson','_stats.geojson'), driver='GeoJSON')
    exit()

    """

    print('computing stats for curb nodes...')

    pred_curb_gdf = nodes_gdf[nodes_gdf['ext:node_type'] == 'curb']
    gt_curb_gdf = nodes_gdf_gt[nodes_gdf_gt['barrier'] == 'kerb']

    curb_output = df_dask.apply(compute_node_score, axis=1, meta=[
    ('geometry', 'geometry'),
    ('tp', 'object'),
    ('fp', 'object'),
    ('fn', 'object'),
    ], gdf=pred_curb_gdf, gdf_gt=gt_curb_gdf).compute(scheduler='multiprocessing')

    curb_output.to_file(nodes_path.split('/')[-1].replace('.geojson','_curb_stats.geojson'), driver='GeoJSON')


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

    pred_curb_link = pred_curb_link.to_crs('epsg:26910')
    pred_curb_link.to_file(nodes_path.split('/')[-1].replace('.geojson','_curbs_links.geojson'), driver='GeoJSON')
    

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

    curb_link_output.to_file(nodes_path.split('/')[-1].replace('.geojson','_curb_link_stats.geojson'), driver='GeoJSON')
    """






