import os
os.environ['USE_PYGEOS'] = '0'
import networkx as nx
import sys
import traceback
import geopandas as gpd
import numpy as np
import osmnx as ox
import dask_geopandas
import math
from statistics import stdev, mean
import geonetworkx as gnx
from shapely import Point, LineString, MultiLineString, Polygon
from shapely.ops import voronoi_diagram
from datetime import datetime
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt


def remove_na_data(data_set_1):
    data_set_1_new = list()

    for i in range(len(data_set_1)):
        if data_set_1[i] == -99.99:
            pass
        else:
            data_set_1_new.append(data_set_1[i])
    return np.array(data_set_1_new)


def compute_avg(gdf, metric):
    data = gdf[metric]
    data = remove_na_data(data)
    return np.average(np.array(data))


def compute_tra_avg(gdf):
    t = gdf['total_edges']
    c = gdf['connect_edges']

    t = remove_na_data(t)
    c = remove_na_data(c)
    return np.average(np.array(c/t))


if __name__ == "__main__":
    gdf = gpd.read_file(sys.argv[1]) # pred1

    g_name = sys.argv[1].split('/')[-1]

    print(f"Avg degree for {g_name}: {compute_avg(gdf, 'degree')}")
    print(f"Avg f1  score R for {g_name}: {compute_avg(gdf, 'f1')}")
    print(f"Avg betweenness R for {g_name}: {compute_avg(gdf, 'betweenness')}")
    print(f"Avg noc R for {g_name}: {compute_avg(gdf, 'noc')}")
    print(f"tra score R for {g_name}: {compute_tra_avg(gdf)}")
