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


def str_2_set(data_str):
    # Remove parentheses and split by space
    tuple_strings = data_str.replace('(', '').replace(')', '').split()

    # Convert each string to a tuple
    data = [tuple(map(int, t.split(','))) for t in tuple_strings]

    return set(data)


def remove_na_str(data_set_1, data_set_2):
    assert len(data_set_1) == len(data_set_2)

    data_set_1_new = list()
    data_set_2_new = list()

    for i in range(len(data_set_1)):
        if data_set_1[i] == '-99.99' or data_set_2[i] == '-99.99':
            pass
        else:
            data_set_1_new.append(data_set_1[i])
            data_set_2_new.append(data_set_2[i])

    return np.array(data_set_1_new), np.array(data_set_2_new)


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


def compute_tra_jaccard(gdf1, gdf2):
    t1 = gdf1['connected_pairs'] # pred
    t2 = gdf2['connected_pairs'] # gt

    t1 = t1.tolist()
    t2 = t2.tolist()

    t1, t2 = remove_na_str(t1, t2)

    iou_l = list()

    for i in range(len(t1)):
        set1 = str_2_set(t1[i])
        set2 = str_2_set(t2[i])

        inter = set1 & set2
        union = set1 | set2
        if len(union) > 0:
            iou = len(inter)/len(union)
            iou_l.append(iou)

    r = np.average(np.array(iou_l))

    return r



if __name__ == "__main__":
    gdf1 = gpd.read_file(sys.argv[1]) # pred
    gdf2 = gpd.read_file(sys.argv[2]) # gt

    g_name = sys.argv[1].split('/')[-1]

    print(f"Avg degree for {g_name}: {compute_avg(gdf1, 'degree')}")
    print(f"Avg f1  score R for {g_name}: {compute_avg(gdf1, 'f1')}")
    print(f"Avg betweenness R for {g_name}: {compute_avg(gdf1, 'betweenness')}")
    print(f"Avg number of connected components for {g_name}: {compute_avg(gdf1, 'noc')}")
    print(f"Traversability R for {g_name}: {compute_tra_avg(gdf1)}")
    print(f"TraversabilitySimilarity for {g_name}: {compute_tra_jaccard(gdf1, gdf2)}")
