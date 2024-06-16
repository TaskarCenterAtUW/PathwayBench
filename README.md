# PathwayBench: A Benchmark for Extracting Routable Pedestrian Path Network Graphs

<p align="center"><img width="600" src="./img/teaser.png"></p>

This repository contains the PathwayBench dataset and benchmark for extracting routable pedestrian pathway graphs. The dataset includes aerial images, road graphs, road rasters, and ground truth data for multiple cities.

- **Metadata**: [PathwayBench Croissant metadata](./metadata/PathwayBench_metadata.jsonld)
- **License**: [ODbL](https://opendatacommons.org/licenses/odbl/)


## Installation

```shell
# create and activate the conda environment
conda create -n pathwaybench python=3.8
conda activate pathwaybench

# install the necessary packages with `requirements.txt`:
pip install -r requirements.txt
```
This code has been tested with Python 3.8 on Ubuntu 20.04. 

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
| Seattle, WA| [Link to dataset](https://drive.google.com/drive/folders/1CnTVuARwv7j-9WXXJpAb3l6NC3n0nhO9?usp=sharing)
| Washington, D.C. | [Link to dataset](https://drive.google.com/drive/folders/1anMEeDbUZPquwEMGA8V3YPeWQJxFHDnu?usp=sharing)
| Portland, OR | [Link to dataset](https://drive.google.com/drive/folders/1yFViA6PaDxqQWvS_iqDay65pEMijiS05?usp=sharing)
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

Compute statistics per TIP: This step computes the statistics (edge-retrieval F1 score, betweenness centrality, number of connected components, TraversabilitySimilarity) for each TIP in the test area. The statistics computed in this step are useful for analyzing local graph routability. `Prediction GeoJSON` is the prediction graph GeoJSON, `TIP GeoJSON` is the area partition generated in the previous step. 

  ```shell
  python scripts/compute_stats.py <Prediction GeoJSON> <Ground Truth GeoJSON> <TIP GeoJSON>
  ```  

Summarize statistics: This step aggregates the computed statistics from the previous step and provides summarizing statistics for the entire test area. The statistics computed in this step are useful for analyzing global graph routability. `Prediction Stats GeoJSON` and `GT Stats GeoJSON` are generated in the previous step.
\end{enumerate}

  ```shell
  python scripts/summarize_stats.py <Prediction Stats GeoJSON> <GT Stats GeoJSON>
  ```  


