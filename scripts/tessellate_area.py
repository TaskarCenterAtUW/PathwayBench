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


if __name__ == '__main__':
    filepath = sys.argv[1]
    gdf = gpd.read_file(filepath)

    print('creating TIPs ...')
    # find bbox of the prediction graph
    bbox = bounding_box_from_gdf(gdf)
    # find tileing within the bbox
    g_roads_simplified = ox.graph.graph_from_polygon(bbox, network_type = 'drive', simplify=True, retain_all=True)
    tile_gdf = create_voronoi_diagram(g_roads_simplified, bbox)
    tile_gdf.to_file(filepath.split('/')[-1].replace('.geojson','_tip.geojson'), driver='GeoJSON')


