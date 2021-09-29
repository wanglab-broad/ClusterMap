import numpy as np
import pandas as pd

def metric_window(data, method, real_res, var1_min, var2_min, height, width):
    
    '''
    slide a small window over the results to compute the oversegmentation score and the undersegmentation score
    
    params :    - data (dataframe) = spots on which we want to evaluate the performance of our cell segmentation
                - method (str) = name of the column in which we placed our results
                - real_res = name of the column where the results from the ground truth are stored
                - var1_min (float), var_2_min (float) = anchors of the window
                - height (float), width (float) = params of the window
    
    returns :   - ratio_cells (float) = #cells identified by ClusterMap / # cells identified by Ground Truth in the window
    '''

    x_cond = (data['spot_location_1']>=var1_min) & (data['spot_location_1']<var1_min + width)
    y_cond = (data['spot_location_2']>=var2_min) & (data['spot_location_2']<var2_min + height)
    data_portion = data.loc[x_cond & y_cond, :].copy()
    if data_portion[real_res].unique().shape[0]>0:
        ratio_cells = data_portion[method].unique().shape[0]/data_portion[real_res].unique().shape[0]
    else:
        ratio_cells = 0
        
    return(ratio_cells)


def compute_metrics_over_under(spots, method, real_res):

    '''
    Compute OverSegmentation Metric and UnderSegmentation Metric

    params :    - spots (dataframe) = current dataset with ClusterMap's results and Ground Truth's results
                - method (str) = name of column where ClusterMap's results are stored
                - real_res (str) = name of column where Ground Truth's results are stored
    
    returns :   - oversegmentation score (float), undersegmentation score (float)
    '''

    over_total = 0
    under_total = 0
    y_max, y_min = np.max(spots['spot_location_2']), np.min(spots['spot_location_2'])
    x_max, x_min = np.max(spots['spot_location_1']), np.min(spots['spot_location_1'])
    height = (y_max - y_min)//10
    width = (x_max - x_min)//10
    x_var = x_min
    y_var = y_min
    i = 0
    while x_var + width<x_max:
        while y_var + height<y_max:
            ratio_cells = metric_window(spots, method, real_res, x_var, y_var, height, width)
            if ratio_cells>1:
                over_total += 1 - 1/ratio_cells 
            else:
                under_total += 1 - ratio_cells
            y_var += height
            i += 1
        x_var += width
        y_var = y_min
    return(under_total/i, over_total/i)