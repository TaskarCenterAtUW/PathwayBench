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
from shapely import Point, LineString, MultiLineString, Polygon
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
    for pair in edge_pairs:
        is_connected = edges_are_connected(G, pts_line_map[pair[0]], pts_line_map[pair[1]])
        if is_connected:
            #print(f'{pair} is connected')
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

    plt.savefig(f'new_{len(G.edges())}.png')
    print(f'new_{len(G.edges())}.png saved')
    """

    return n_total, n_connected 


def compute_f1(pred, gt):
    E_THRES = 5
    BUFF_DIS = 10
    pred_buff_gdf = None
    d_sum = 0
    sw_cnt = 0
    sw_gt_cnt = 0

    num_splits = 5

    d_lst = list()

    tp = 0
    fp = 0

    tp_d = 0
    fp_d = 0

    pred_sw = pred
    gt_sw = gt
    for it, pred_it in pred_sw.iterrows():
        try:
            shape_geo = pred_it['geometry']
            shape_geo_dia = shape_geo.buffer(BUFF_DIS)

            pred_copy = copy.deepcopy(pred_it)
            pred_copy = pred_copy.to_frame().T.reset_index()
            pred_copy['geometry'] = shape_geo_dia
            pred_copy = gpd.GeoDataFrame(pred_copy, geometry=pred_copy['geometry'], crs="EPSG:4326")

            inter = gt_sw.overlay(pred_copy, keep_geom_type=True, how='intersection')
            pred_it_pts = [pred_it['geometry'].interpolate((i/num_splits), normalized=True) for i in range(1, num_splits)]
            pred_it_pts_gdf = gpd.GeoDataFrame({'geometry': pred_it_pts}, crs=pred_copy.crs)

            if not inter.empty:
                distance_matched = pred_it_pts_gdf.sjoin_nearest(inter, distance_col="distances", how="inner")
                distance_lst = distance_matched['distances'].tolist()
                d_filter = [x for x in distance_lst if x <= E_THRES]
                avg_d = np.average(np.array(d_filter))
                d_lst.append(avg_d)

                if len(d_filter) != 0:

                    tp += 1

                    sw_cnt += 1
                    sw_gt_cnt += len(d_filter)
                    d_sum += avg_d
                else:
                    fp += 1

        except Exception as e:
            traceback.print_exc()
            #exit()
            continue

    return tp, fp


def get_stats(polygon, G, gdf, gdf_gt):
    stats = {}
    undirected_g = nx.Graph(G)
    
    # betweenness
    try:
        bet = nx.betweenness_centrality(undirected_g, normalized = True, endpoints=False)
        stats["bet_centrality_avg"] = mean(bet.values())
        stats["bet_stdev"] = stdev(bet.values())
    except Exception as e:
        print(f"Unexpected {e}, {type(e)} with polygon {polygon} when getting betweenness value")
        stats["bet_centrality_avg"] = -99.99
        stats["bet_stdev"] = -99.99

    # eigen
    try:
        eigen = nx.eigenvector_centrality(undirected_g, max_iter=1000)
        stats["eig_centrality_avg"] = mean(eigen.values())
    except Exception as e:
        print(f"Unexpected {e}, {type(e)} with polygon {polygon} when getting eigen value")
        stats["eig_centrality_avg"] = -99.99

    # degree
    try:
        deg = nx.degree_centrality(undirected_g)
        stats["deg_centrality_avg"] = mean(deg.values())
    except Exception as e:
        print(f"Unexpected {e}, {type(e)} with polygon {polygon} when getting degree cventrality")
        stats["deg_centrality_avg"] = -99.99

    # number of connected components
    try:
        noc = nx.number_connected_components(undirected_g)
        stats["num_connect_comp_avg"] = noc
    except Exception as e:
        print(f"Unexpected {e}, {type(e)} with polygon {polygon} when getting number of connected components")
        #traceback.print_exc()
        stats["num_connect_comp_avg"] = -99.99

    # node connectivity 
    try:
        conn = nx.average_node_connectivity(undirected_g)
        stats["node_connect_avg"] = conn
    except Exception as e:
        print(f"Unexpected {e}, {type(e)} with polygon {polygon} when getting node connectivity")
        #traceback.print_exc()
        stats["node_connect_avg"] = -99.99

    # node-to-node connected paths
    try:
        _, n_pahts = hull_connected_paths(undirected_g)
        stats["n_connect_paths"] = n_pahts
    except Exception as e:
        print(f"Unexpected {e}, {type(e)} with polygon {polygon} when getting number of connected paths")
        #traceback.print_exc()
        stats["n_connect_paths"] = -99.99

    # edge-to-edge connected paths
    try:
        n_total, n_connected = tile_tra_score(undirected_g, polygon)
        stats["n_total_edges"] = n_total
        stats["n_connect_edges"] = n_connected
    except Exception as e:
        print(f"Unexpected {e}, {type(e)} with polygon {polygon} when getting number of connected edge pairs")
        #traceback.print_exc()
        stats["n_total_edges"] = -99.99
        stats["n_connect_edges"] = -99.99

    # f1 score
    try:
        tp, fp = compute_f1(gdf, gdf_gt)
        tp, fn = compute_f1(gdf_gt, gdf)
        precision = tp/(tp+fp)
        recall = tp/(tp+fn)
        f1 = 2*(precision*recall)/(precision + recall)
        stats["precision"] = precision
        stats["recall"] = recall
        stats["f1"] = f1
    except Exception as e:
        print(f"Unexpected {e}, {type(e)} with polygon {polygon} when getting f1 score")
        #traceback.print_exc()
        stats["precision"] = -99.99
        stats["recall"] = -99.99
        stats["f1"] = -99.99


    return stats


def get_measures_from_polygon(polygon, gdf, gdf_gt):
    gdf = gdf.to_crs(PROJ)
    gdf_gt = gdf_gt.to_crs(PROJ)

    # crop gdf to the polygon
    #cropped_gdf = gdf
    cropped_gdf = gpd.clip(gdf, polygon)
    cropped_gdf_gt = gpd.clip(gdf_gt, polygon)

    G = graph_from_gdf(cropped_gdf) 

    stats = get_stats(polygon, G, cropped_gdf, cropped_gdf_gt)
    
    #direct_trust_score, time_trust_score = analyze_sidewalk_data(G)
    #stats["direct_trust_score"] = direct_trust_score
    #stats["time_trust_score"] = time_trust_score

    #stats['indirect_values'] = get_indirect_trust_score_from_polygon(polygon)
    
    return stats


def compute_global_stats(gdf, filepath):
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



def func(feature, gdf, gdf_gt):
    poly = feature.geometry
    if (poly.geom_type == "Polygon" or poly.geom_type == "MultiPolygon"):
        measures = get_measures_from_polygon(poly, gdf, gdf_gt)
        feature.loc['degree'] = measures["deg_centrality_avg"]
        feature.loc['eigen'] = measures["eig_centrality_avg"]
        feature.loc['betweenness'] = measures["bet_centrality_avg"]
        feature.loc['bet_stdev'] = measures["bet_stdev"]
        #feature.direct_trust_score = measures["direct_trust_score"]
        #feature.time_trust_score = measures["time_trust_score"]
        #feature.indirect_values = measures["indirect_values"]
        feature.loc['noc'] = measures["num_connect_comp_avg"]
        feature.loc['conn'] = measures["node_connect_avg"]
        feature.loc['n_path'] = measures["n_connect_paths"]
        feature.loc['total_edges'] = measures["n_total_edges"]
        feature.loc['connect_edges'] = measures["n_connect_edges"]
        feature.loc['precision'] = measures["precision"]
        feature.loc['recall'] = measures["recall"]
        feature.loc['f1'] = measures["f1"]
        return feature


if __name__ == '__main__':

    filepath = sys.argv[1]
    gdf = gpd.read_file(filepath)

    gt_filepath = sys.argv[2]
    gdf_gt = gpd.read_file(gt_filepath)

    tile_gdf = gpd.read_file(sys.argv[3])

    # compute local stats
    df_dask = dask_geopandas.from_geopandas(tile_gdf, npartitions=64)

    print('computing stats...')
    output = df_dask.apply(func, axis=1, meta=[
        ('geometry', 'geometry'),
        ('degree','object'),
        ('eigen', 'object'),
        ('betweenness', 'object'), 
        ('bet_stdev', 'object'),
        ('noc', 'object'),
        ('conn', 'object'),
        ('n_path', 'object'),
        ('total_edges', 'object'),
        ('connect_edges', 'object'),
        ('precision', 'object'),
        ('recall', 'object'),
        ('f1', 'object'),
        ], gdf=gdf, gdf_gt=gdf_gt).compute(scheduler='multiprocessing')

    output.to_file(filepath.split('/')[-1].replace('.geojson','_eval.geojson'), driver='GeoJSON')

