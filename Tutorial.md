# ClusterMap

### Tutorial

- We are currently uploading more files for packaging and testing and will finish update soon.

#### Basics

**Input data**

- `spots`: data matrix of mRNA spots with 2D/3D physical location and gene identity information (pandas dataframe)
  - Example

| Index | spot_location_1 | spot_location_2 | spot_location_3 | gene | Optional other info: gene_name |
| ----- | :-------------: | :-------------: | :-------------: | :--: | :----------------------------: |
| 0     |       105       |       239       |        1        |  1   |            Syndig1l            |
| 1     |       110       |       243       |        1        |  1   |            Syndig1l            |
| 2     |       115       |       178       |        1        |  2   |             Acot13             |

- `dapi`: a 2D/3D image corronsponding to `spots`



**Input Parameters**

- `xy_radius`: estimation of radius of cells in x-y plane

- `z_radius`: estimation of radius of cells in z axis; 0 if data is 2D.

- `cell_num_threshold`:  a threshold for deciding the number of cells. A larger value gives more cells; Default: 0.1.

- `dapi_grid_interval`: sample interval in DAPI image. A large value will consume more computation resources and give more accurate results (most of the time). Default: 3.



**Basic functions**

- [x] Build a ClusterMap model

```
model = ClusterMap(spots=spots, dapi=dapi, gene_list=gene_list, num_dims=num_dims,
                   xy_radius=xy_radius,z_radius=0,
                   fast_preprocess=False,gauss_blur=False,sigma=1)
```

> `spots`: pandas DataFrame.
>
> `dapi`：numpy array.
>
> `gene_list`: numpy array. An array of gene id values.
>
> `num_dims`: int. Number of data dimensions, 2 or 3.
>
> `xy_radius`: float. Estimation of radius of cells in x-y plane.
>
> `z_radius`: float. Estimation of radius of cells in z axis; 0 if data is 2D.
>
> `gauss_blur`: bool. Choose whether to apply Gaussian blur with sigma =`sigma`.
>
> `sigma`: float. Sigma value for Gaussian blur.
>
> `fast_preprocess`: bool. Binarize DAPI images with erosion and morphological reconstruction before OTSU thresholding when `True`. Binarize DAPI images with only OTSU thresholding when `False`.

Output: binarized DAPI results are saved in model.dapi_binary (2D or 3D) and model.dapi_stacked (2D). 

Note: A clean binarized DAPI image is essential for later processing. We seggest two binarization settings: (1) `fast_preprocess` = `False`; (2) `gauss_blur` =`True` and  `fast_preprocess` = `True`. If your DAPI images have special background noise, we suggesting additional denoising processing for DAPI images.

- [x] Preprocess data

```
model_tile.preprocess(dapi_grid_interval=5, LOF=False, contamination=0.1, pct_filter=0.1)
```

> `dapi_grid_interval`: int (default: 5).	The size of sampling interval on DAPI image.
>
> `pct_filter`: float (between 0 and 1, default: 0.1).	The percentage of filtered noise reads.
>
> `LOF`: bool (default: False).	Choose if to apply [local noise rejction](https://scikit-learn.org/stable/auto_examples/neighbors/plot_lof_outlier_detection.html).



- [x] Cell segmentation

```
model_tile.segmentation(self,cell_num_threshold=0.01, dapi_grid_interval=5, add_dapi=True,use_genedis=True)
```

​	*Paramters*

- [x] create adata, saved in model.cell_adata (Cell typing processing is mostly based on [Scanpy](https://scanpy.readthedocs.io/en/stable/index.html) and [anndata](https://anndata.readthedocs.io/en/latest/index.html))

```
model.create_cell_adata(cellid,geneid,gene_list,genes,num_dims)
```

- [x] Find cell types

```
model.cell_typing(cluster_method='leiden',resol=1.5)
```

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

- [x] Construct and plot convex hull of cells

```
model.create_convex_hulls()
```

- [ ] Plot cell typing results

```
cluster_pl=model.plot_cell_typing(umap=True,heatmap=False, celltypemap=True)
```



2. **Save functions**

- [ ] Save cell segmentation results
- [ ] Save cell typing results

3. **Large input data**

   If input data is large (>100,000 spots), processing over whole data at one time may be time-consuming, we implemented trimming and stitching functions to process over each trimmed tile to save computational resources. Note that there won't be any cracks in results as we consider a 10% overlap when trimming and stitching.

   **Relative functions:**  

   - [x] Trim

   ```
   img = dapi
   window_size=2000
   label_img = get_img(img, spots, window_size=window_size, margin=math.ceil(window_size*0.1))
   out = split(img, label_img, spots, window_size=window_size, margin=math.ceil(window_size*0.1))
   ```

   *Parameters:*

   - [x] Stitch after cell segmentation over the tile

   ```
   cell_info=model.stitch(model_tile,out,tile_num, cell_info)
   ```

   *Parameters:*

4. **Cell typing relative functions**

   - [x] Merge cell types

   ```
   merge_list = [[0,2,3,8,9],[1,4,5,6,10]]
   model.merge_multiple_clusters(merge_list)
   ```



**Output parameters**

- `model.cellid_unique`: unique cell id values

- `model.cellcenter_unique`:  cell centers in order of `model.cellid_unique`



#### Analysis on STARmap 2D V1 1020-gene sample

- Example file at `ClusterMap_STARmap_human_cardiac_organoid.ipynb`



#### Analysis on STARmap human cardiac sample

- Example file at `ClusterMap_STARmap_V1_1020.ipynb`



#### Analysis on STARmap 3D V1 28-gene sample



#### Time estimation

- Time is dependent on the number of input spots, and potentially the area the DAPI foreground. Currently testing on several samples: 
  - 1mins 42s for 49,712 input spots (all 273,242 spots) without GPU, single thread
  - 34mins 53s for 471,295 input spots without GPU, single thread
