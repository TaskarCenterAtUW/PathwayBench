import os
os.environ['USE_PYGEOS'] = '0'
import networkx as nx
import sys
import traceback
import geopandas as gpd
import numpy as np
import osmnx as ox
import math
from statistics import stdev, mean
from shapely import Point, LineString, MultiLineString, Polygon
from shapely.ops import voronoi_diagram
from datetime import datetime
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


def compute_aggregate_f1(gdf):
    tp = np.sum(np.array(gdf['tp']))
    fp = np.sum(np.array(gdf['fp']))
    fn = np.sum(np.array(gdf['fn']))

    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1 = 2*(precision*recall)/(precision + recall)

    return np.round(precision,3), np.round(recall,3), np.round(f1,3)


def tra_jaccard(row, gdf2):
    index = row.name

    row2 = gdf2.loc[index]

    t1 = row['connected_pairs'] # pred
    t2 = row2['connected_pairs'] # gt

    set1 = str_2_set(t1)
    set2 = str_2_set(t2)

    inter = set1 & set2
    union = set1 | set2

    iou = 0
    if len(union) > 0:
        iou = len(inter)/len(union)

    return iou


def create_score_json(gdf1, gdf2):
    gdf1['ts'] = gdf1.apply(tra_jaccard, axis=1, args=(gdf2,))


if __name__ == "__main__":
    gdf1 = gpd.read_file(sys.argv[1]) # pred

    if 'edge' in sys.argv[1]:
        gdf2 = gpd.read_file(sys.argv[2]) # gt

    g_name = sys.argv[1].split('/')[-1]

    if 'edge' in g_name:
        # print(f"Avg degree for {g_name}: {compute_avg(gdf1, 'degree')}")
        # print(f"Avg f1  score R for {g_name}: {compute_avg(gdf1, 'f1')}")
        # print(f"Avg betweenness R for {g_name}: {compute_avg(gdf1, 'betweenness')}")
        # print(f"Avg number of connected components for {g_name}: {compute_avg(gdf1, 'noc')}")

        # print(f"Traversability R for {g_name}: {compute_tra_avg(gdf1)}")
        print(f"TraversabilitySimilarity for {g_name}: {compute_tra_jaccard(gdf1, gdf2)}")

        precision, recall, f1 = compute_aggregate_f1(gdf1)
        print(f"Precision for {g_name}: {precision}")
        print(f"Recall for {g_name}: {recall}")
        print(f"F1 for {g_name}: {f1}")
    elif 'node' in g_name:
        precision, recall, f1 = compute_aggregate_f1(gdf1)
        print(f"Precision for {g_name}: {precision}")
        print(f"Recall for {g_name}: {recall}")
        print(f"F1 for {g_name}: {f1}")
    else:
        precision, recall, f1 = compute_aggregate_f1(gdf1)
        print(f"Precision for {g_name}: {precision}")
        print(f"Recall for {g_name}: {recall}")
        print(f"F1 for {g_name}: {f1}")

    # Optional: Also save the per-TIP Traversability Similarity score geojson 
    # create_score_json(gdf1, gdf2)
    # gdf1.to_file(sys.argv[1].replace('stats.geojson', 'scores.geojson'), driver="GeoJSON")
