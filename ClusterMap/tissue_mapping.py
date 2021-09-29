import numpy as np
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors

def neighboring_cell_types(data, radius):
    cell_types = np.unique(data['cell_type'])
    cell_types = cell_types[cell_types>=0]
    
    X_data = data[['cell_center_1', 'cell_center_2', 'cell_center_3']]
    
    knn = NearestNeighbors(radius=radius)
    knn.fit(X_data)
    cell_number = data.shape[0]
    res_neighbors = knn.radius_neighbors(X_data, return_distance=False)

    res_ncc = np.zeros((cell_number, len(cell_types)))
    for i in range(cell_number):
        neighbors_i = res_neighbors[i]
        type_neighbors_i = data.loc[neighbors_i, :].groupby('cell_type').size()
        res_ncc[i, type_neighbors_i.index.to_numpy() - np.min(cell_types)] = np.array(type_neighbors_i)
        res_ncc[i] /= len(neighbors_i)
    return(res_ncc)

def tissue2spot(adata, spots, method):
    
    '''
    Assign a cell type label to each spot

    params :    - adata (AnnData) = annotated cell type expression matrix
                - spots (dataframe) = spatial locations + gene identities + cell labels
                - method (str) = name of the column where the results are stored
    '''

    spots['tissue'] = -1
    cells_unique = np.unique(spots[method])
    cells_unique = cells_unique[cells_unique>=0]

    bad_cells = np.setdiff1d(cells_unique, np.unique(adata.obs['index'].astype(int)), assume_unique=False)
    dico_cells2tissues = dict([(cell, int(tissue)) for cell, tissue in zip(adata.obs['index'].astype(int), adata.obs['tissue'])] + [(-1,-1)] + [(-2,-1)] + [(bad, -1) for bad in bad_cells])
    spots['tissue'] = np.array(list(map(dico_cells2tissues.get, spots[method])), dtype=int)