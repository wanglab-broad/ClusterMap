import numpy as np
import random
import pandas as pd
import cv2
import seaborn as sns

def create_mask(data, kernel, num_iter, median_size):
    
    '''
    Dilate and Blur the image
    params :    - data (ndarray) = 2D projection of results
                - kernel (ndarray) = convolution matrix to perform the dilation
                - num_iter (int) = number of iterations for dilation
                - median_size (int) = size of the kernel used for MedianBlur

    returns :   Blured image
    '''

    data_dil = cv2.dilate(data, kernel=kernel, iterations=num_iter).astype(np.uint16)
    data_blured = cv2.medianBlur(data_dil, median_size)
    return(data_blured)

def df_to_array(spots, method='leiden', spot_columns=['spot_location_1', 'spot_location_2']):

    '''
    Transform a dataframe with spatial locations and values to an array
    
    params :    - spots (dataframe) = dataset
                - method (str) = name of the column where results are stored
    
    returns :   Resulting numpy array
    '''

    arr = np.zeros((np.max(spots[spot_columns[1]]) +1 , np.max(spots[spot_columns[0]] + 1)))
    arr[spots[spot_columns[1]], spots[spot_columns[0]]] = spots[method] + 1
    return(arr)

def label2rgb(data):
    
    '''
    Transform labeled image to a RGB image

    params :    - data (ndarray) = labeled image

    returns :   RGB image
    '''

    labels = np.unique(data[data>=1])
    palette = np.round(255*np.array(sns.color_palette('gist_ncar', len(labels))),0)
    data_rgb = np.zeros((data.shape[0], data.shape[1], 3))    
    for i,label in enumerate(labels):
        data_rgb[data==label] = palette[i]
    return(data_rgb.astype(np.uint8))