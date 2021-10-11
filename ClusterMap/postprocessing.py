import numpy as np
import pandas as pd
from skimage.morphology import ball,erosion
from sklearn.neighbors import NearestNeighbors

def res_over_dapi_erosion(spots, dapi_binary, method='clustermap', minus1 = False):
    
    '''
    Erase cells that do not overlap with DAPI signals

    params :    - spots (dataframe) = spatial locations and gene identities
                - dapi_binary (ndarray) = binarized DAPI image
                - method (str) = name of column containing the results
                                 of segmentation
                - minus1 (bool) = if spots' locations start at 1 instead of 0
    
    returns :   None
    '''

    dapi_binary_eroded = dapi_binary
    cell_list = np.unique(spots[method])[1:]
    if minus1:
        for cell in cell_list:
            spots_cell = spots.loc[spots[method]==cell, ['spot_location_2', 'spot_location_1', 'spot_location_3']].to_numpy()
            number_overlap = np.sum(dapi_binary_eroded[spots_cell[:,0] - 1,spots_cell[:,1] - 1, spots_cell[:,2] - 1])
            if number_overlap ==0:
                spots.loc[spots[method]==cell, method] = -1      
    else:
        for cell in cell_list:
            spots_cell = spots.loc[spots[method]==cell, ['spot_location_2', 'spot_location_1', 'spot_location_3']].to_numpy()
            number_overlap = np.sum(dapi_binary_eroded[spots_cell[:,0]-1,spots_cell[:,1]-1, spots_cell[:,2] - 1])
            if number_overlap ==0:
                spots.loc[spots[method]==cell, method] = -1   


       
def erase_small_clusters(spots, N=10,method='clustermap'):

    '''
    Erase small cells

    params :    - spots (dataframe) = dataset
                - method (str) = name of column containing the results
                                 of segmentation
                - N (int) = minimal number of spots within a cell

    returns :   None
    '''

    spots_per_cluster = spots.groupby(method).size()
    dico_corres = dict([(leid, -1) if spots_per_cluster.loc[leid]<=N else (leid, leid) for leid in spots_per_cluster.index.to_numpy()])
    spots[method] = list(map(dico_corres.get, spots[method]))
    
    
    
    
    