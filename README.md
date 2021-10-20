# ClusterMap

This repository contains an available tool for ClusterMap for multi-scale clustering analysis of spatial gene expression, and ClusterMap examples of the 3D STARmap human cardiac organoid dataset, 2D STARmap mouse brain V1 dataset, and 3D STARmap mouse brain V1 dataset.

- **Original scripts for generating data at ClusterMap paper are at: https://codeocean.com/capsule/9820099/tree/v1.**

<img src="./datasets/FeaturedImage.jpg" alt="FeaturedImage" style="zoom:10%;" />

***

<br >

### Install

> pip install git+https://github.com/LiuLab-Bioelectronics-Harvard/ClusterMap.git

***

<br >

### Tutorial

- We are currently uploading more files for packaging and testing and will finish update soon.

#### Basics

**Input data**

`spots`: data matrix of mRNA spots with 2D/3D physical location and gene identity information (pandas dataframe)

* Example

| Index | spot_location_1 | spot_location_2 | spot_location_3 | gene | Optional other info: gene_name |
| ----- | :-------------: | :-------------: | :-------------: | :--: | :----------------------------: |
| 0     |       105       |       239       |        1        |  1   |            Syndig1l            |
| 1     |       110       |       243       |        1        |  1   |            Syndig1l            |
| 2     |       115       |       178       |        1        |  2   |             Acot13             |

`dapi`: a 2D/3D image corronsponding to `spots`



**Input Parameters**

`xy_radius`: estimation of radius of cells in x-y plane

`z_radius`: estimation of radius of cells in z axis; 0 if data is 2D.

`cell_num_threshold`:  a threshold for deciding the number of cells. A larger value gives more cells; Default: 0.1.

`dapi_grid_interval`: sample interval in DAPI image. A large value will consume more computation resources and give more accurate results (most of the time). Default: 3.



**Basic functions**

- [x] Build a ClusterMap model

```
model = ClusterMap(spots=spots, dapi=dapi, gene_list=gene_list, num_dims=num_dims,
                   xy_radius=xy_radius,z_radius=0,fast_preprocess=True)
```

Parameters:

- [x] Preprocess data

```
model_tile.preprocess(dapi_grid_interval=3,pct_filter=pct_filter)
```

Paramters

- [x] Cell segmentation

```
model_tile.segmentation(cell_num_threshold=0.1,dapi_grid_interval=3,add_dapi=True,use_genedis=True)
```

Paramters

- [ ] Find cell types
- [ ] Identify tissue layers

**Other functions**

1. **Plot functions**

- [x] Show spatial distribution of interested gene markers

```
model.plot_gene(marker_genes,genes_list,figsize=(4,4),s=0.6)
```

- [x] Plot cell segmentation results

```
model.plot_segmentation(figsize=(8,8),s=0.005,plot_with_dapi=True,plot_dapi=True)
```

- [x] Plot cell segmentation results in 3D

```
model.plot_segmentation_3D(figsize=(8,8),elev=45, azim=-65)
```

- [ ] Construct and plot convex hull of cells

2. **Save functions**

- [ ] Save cell segmentation results
- [ ] Save cell typing results

3. **Large input data**

If input data is large (>100,000 spots), processing over whole data at one time may be time-consuming, we implemented trimming and stitching functions to process over each trimmed tile to save computational resources. Note that there won't be any cracks in results as we consider a 10% overlap when trimming and stitching.

**Reletive functions:**  

- [x] Trim

```
img = dapi
window_size=2000
label_img = get_img(img, spots, window_size=window_size, margin=math.ceil(window_size*0.1))
out = split(img, label_img, spots, window_size=window_size, margin=math.ceil(window_size*0.1))
```

Parameters:



- [x] Stitch after cell segmentation over the tile

```
cell_info=model.stitch(model_tile,out,tile_num, cell_info)
```

Parameters:



**Output parameters**

`model.cellid_unique`: unique cell id values

`model.cellcenter_unique`:  cell centers in order of `model.cellid_unique`



#### Analysis on STARmap 2D V1 1020-gene sample

- Example file at `ClusterMap_STARmap_human_cardiac_organoid.ipynb`



#### Analysis on STARmap human cardiac sample

- Example file at `ClusterMap_STARmap_V1_1020.ipynb`



#### Analysis on STARmap 3D V1 28-gene sample



#### Time estimation

Time is dependent on the number of input spots, and potentially the area the DAPI foreground. Currently testing on several samples: 

- 1mins 42s for 49,712 input spots (all 273,242 spots) without GPU, single thread
- 34mins 53s for 471,295 input spots without GPU, single thread

***

<br >

### Other Info

#### Citation

If you find ClusterMap useful for your work, please cite our paper: 

> He, Y., Tang, X., Huang, J. et al. ClusterMap for multi-scale clustering analysis of spatial gene expression. Nat Commun 12, 5909 (2021). https://doi.org/10.1038/s41467-021-26044-x

#### Contact

Contact us at clustermap.issue@gmail.com if you have any issues.

#### Contributor

Yichun He, Emma Bou Hanna, Jiahao Huang, Xin Tang

