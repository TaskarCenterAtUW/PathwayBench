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

PROJ = 'epsg:26910'

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


def create_tip(filepath):
    print('creating TIPs ...')

    # --- Step 1: Read input ---
    gdf = gpd.read_file(filepath)

    # --- Step 2: Construct the convex hull boundary ---
    multi_line = MultiLineString(gdf.geometry.values)
    outer_boundary = multi_line.convex_hull

    # --- Step 3: Extract road network within boundary ---
    g_roads_simplified = ox.graph.graph_from_polygon(
        outer_boundary,
        network_type='drive',
        simplify=True,
        retain_all=True
    )

    # --- Step 4: Create Voronoi tiles ---
    tile_gdf = create_voronoi_diagram(g_roads_simplified, outer_boundary)

    # --- Step 5: Save to same directory with _tip.geojson suffix ---
    input_dir = os.path.dirname(filepath)
    input_filename = os.path.basename(filepath)
    output_filename = input_filename.replace('.geojson', '_tip.geojson')
    output_path = os.path.join(input_dir, output_filename)

    tile_gdf.to_file(output_path, driver='GeoJSON')
    print(f'TIP GeoJSON saved to: {output_path}')

if __name__ == '__main__':
    create_tip(sys.argv[1])



