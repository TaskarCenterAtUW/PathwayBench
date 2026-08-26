# PathwayBench: A Benchmark for Extracting Routable Pedestrian Path Network Graphs

<p align="center"><img width="600" src="./img/teaser.png"></p>

This repository contains the PathwayBench dataset and benchmark for extracting routable pedestrian pathway graphs. The dataset includes aerial images, road graphs, road rasters, and ground truth data for multiple cities.

- **Metadata**: [PathwayBench Croissant metadata](./metadata/PathwayBench_metadata.jsonld)
- **License**: [ODbL](https://opendatacommons.org/licenses/odbl/)


## Installation

```shell
# create and activate the conda environment
conda create -n pathwaybench python=3.9
conda activate pathwaybench

# install the necessary packages with `requirements.txt`:
pip install -r requirements.txt
```
This code has been tested with Python 3.9 on Ubuntu 20.04. 

## Datasets
Each set of samples in the PathwayBench dataset includes five co-registered features. The filename of each set of samples and the corresponding features are listed below:

| Filename | Feature Type
|--|--|
| xxxx_aerial.png | The aerial satellite imagery.
| xxxx_road.geojson | The street (road) graph.
| xxxx_road.png | The rasterized street map (with additional features).
| xxxx_gt_graph.geojson | The human-validated pedestrian pathway graph.
| xxxx_gt_mask.png | The rasterized human-validated pedestrian pathway graph to support semantic segmentation tasks.
| xxxx_gt_color.png | The color-coded version of xxxx_gt_mask.png for visualization purposes.

Below are the links to the dataset that are currently supported by PathwayBench
| City | Data |
|--|--|
| Seattle, WA| [Link to dataset](https://drive.google.com/drive/folders/1Rbpah5J-9xtw3UM1SpZQPk0Fc_p6f6dB?usp=drive_link)
| Washington, D.C. | [Link to dataset](https://drive.google.com/drive/folders/1GGPCVdJKPaZ_JFfVRjcWabY1hZQ2N7XY?usp=drive_link)
| Portland, OR | [Link to dataset](https://drive.google.com/drive/folders/1xyyU3fRlNwsKARdaAhBXzSzXMwWqM0Y0?usp=drive_link)
| Bellevue, WA | [Will be released soon]
| Quito, Ecuador | [Will be released soon]
| Sao Paulo, Brazil | [Will be released soon]
| Santiago, Chile | [Will be released soon]  
| Valparaiso, Chile | [Will be released soon]  
## Benchmark

PathwayBench provides utilities for evaluating graphs by the extent to which their structural characteristics align with ground truth, as described below.

Partition test area: This step partitions the entire test area into Tessellating Intersection Polygons (TIP). Each TIP is created by assigning a point location to a road intersection, then computing the associated Voronoi polygons to tessellate the entire test area. `Ground Truth GeoJSON` is provided for each of the support city in PathwayBench dataset.

  ```shell
  python scripts/tessellate_area.py <Ground Truth GeoJSON>
  ```  

Compute scores: This step computes the scroes (edge-retrieval F1 score, edge TraversabilitySimilarity, node F1 score) for the test area. `TIP GeoJSON` is the area partition generated in the previous step. 

  ```shell
  python scripts/compute_scores.py <TIP GeoJSON> --edges-path <Prediction Edge GeoJSON> --gt-edges-path <Ground Truth Edge GeoJSON> --nodes-path <Prediction Node GeoJSON> --gt-nodes-path <Ground Truth Node GeoJSON>
  ```  

Node evaluation supports `--node-type all|kerb`, `--node-matching standard|strict|by_id`, and the legacy `--node-strict` shortcut for strict matching. `by_id` matches predicted and ground-truth nodes by exact `_id`. It reports `NodeError`, the mean node distance in meters, and writes the node-error distance distribution CSV, histogram PNG, and matching summary CSV in the same directory as the result CSVs. High-end outliers above `2 * median` are removed from the NodeError summary, distribution CSV, and histogram.
