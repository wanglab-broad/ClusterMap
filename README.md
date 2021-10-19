# ClusterMap

This repository contains an available tool for ClusterMap for multi-scale clustering analysis of spatial gene expression, and ClusterMap examples of the 3D STARmap human cardiac organoid dataset, 2D STARmap mouse brain V1 dataset, and 3D STARmap mouse brain V1 dataset.

**Original scripts for generating data at ClusterMap paper are at: https://codeocean.com/capsule/9820099/tree/v1.**

<img src="./datasets/FeaturedImage.jpg" alt="FeaturedImage" style="zoom:10%;" />

### Install

> pip install git+https://github.com/LiuLab-Bioelectronics-Harvard/ClusterMap.git



### Tutorial

We are currently uploading more files for packaging and testing and will finish update soon.

##### Basics

- Input data format

  | Index | spot_location_1 | spot_location_2 | spot_location_3 | gene | Optional other info: gene_name |
  | ----- | :-------------: | :-------------: | :-------------: | :--: | :----------------------------: |
  | 0     |       105       |       239       |        1        |  1   |            Syndig1l            |
  | 1     |       110       |       243       |        1        |  1   |            Syndig1l            |
  | 2     |       115       |       178       |        1        |  2   |             Acot13             |

- Input Parameters

`xy_radius`: estimation of radius of cells in x-y plane

`z_radius`: estimation of radius of cells in z axis; 0 if data is 2D.

`cell_num_threshold`:  a threshold for deciding the number of cells; larger value gives more cells; defaults: 0.1.

- Functions

- Output parameters

`model.cellid_unique`: unique cell id values

`model.cellcenter_unique`:  cell centers in order of `model.cellid_unique`

##### Analysis on STARmap 2D V1 1020-gene sample

- Example file at *ClusterMap_STARmap_human_cardiac_organoid.ipynb*

##### Analysis on STARmap human cardiac sample

- Example file at *ClusterMap_STARmap_V1_1020.ipynb*

##### Analysis on STARmap 3D V1 28-gene sample



### Other Info

#### Citation

If you find ClusterMap useful for your work, please cite our paper: 

```
He, Y., Tang, X., Huang, J. et al. ClusterMap for multi-scale clustering analysis of spatial gene expression. Nat Commun 12, 5909 (2021). https://doi.org/10.1038/s41467-021-26044-x
```

#### Contact

Contact us at clustermap.issue@gmail.com if you have any issues.

#### Contributor

Yichun He, Emma Bou Hanna, Jiahao Huang, Xin Tang

