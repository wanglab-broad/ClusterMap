import numpy as np
import pandas as pd
import os
from tqdm import tqdm

# def load_tile_config(path):
#         tile_config = np.loadtxt(path, dtype='str', delimiter=';')
#         df_config = pd.DataFrame(columns=['Tile', 'X_init', 'Y_init'])
#         df_config['Tile'] = list(map(lambda x: x[:-4],tile_config[:,0]))
#         df_config['Tile'] = list(map(lambda x: int(x.split('tile_')[-1]), df_config['Tile']))
#         coord_list = list(map(lambda x: x.split(', '), tile_config[:,-1]))
#         x_list = list(map(lambda x: float(x[0].split(' (')[-1]), coord_list))
#         y_list = list(map(lambda x: float(x[1].split(')')[0]), coord_list))

#         df_config['X_init'] = list(map(lambda x: int(x),x_list))
#         df_config['Y_init'] = list(map(lambda x: int(x), y_list))
#     return(df_config)

# def gather_all_tiles(path, res_name):
#     tiles = os.listdir(path)
#     testspots=pd.read_csv(path+tiles[0])
#     spots_results = pd.DataFrame(columns=testspots.columns)
#     for tile in tqdm(tiles):
#         spots = pd.read_csv(path + tile)
#         spots.loc[spots[res_name]==-1, res_name] = -2
#         spots_results = spots_results.append(spots)
#     return(spots_results)

def create_img_label(self ):
#     num_col = np.sum(df_config['X_init']==0)
#     num_row = np.sum(df_config['Y_init']==0)
    df_config=self.config
    max_col = max(df_config['Y_init'])+self.size_single_img
    max_row = max(df_config['X_init'])+self.size_single_img
    img = np.zeros((max_row, max_col))
    for i, row in df_config.iterrows():
        tx = 0
        ty = 0
        
        if row[1] > 0: # if x not zero
            tx = int(self.size_single_img*0.05)
        if row[-1] > 0: # if y not zero
            ty = int(self.size_single_img*0.05)
        img[row[1]+tx:row[1]+self.size_single_img, row[-1]+ty:row[-1]+self.size_single_img] = row[0]
    return(img)

def stitch_all_tiles(self):
    img=self.img
    df_config=self.config
    res_name=self.res_name
    path_res=self.path_res
    
    unique_tiles = list(df_config.Tile) #np.unique(np.array(spots_results['spot_image_position'], dtype='str'))
    rotation_degree = -90
    spot_merged = np.zeros((0,3))
    cellid = []
    geneid = []
    existtiles = os.listdir(path_res)
    for i in tqdm(unique_tiles): #range(1, num_row*num_col + 1):
        if i<1000:
            i_str="{:03}".format(i)+'.csv'
        else:
            i_str=str(i)+'.csv'
        if i_str not in existtiles:
            continue
        else:
            spots_portion = pd.read_csv(path_res+i_str)
            #spots_results.loc[spots_results['spot_image_position']=='tile_'+str(i),:].copy()
            spatial = spots_portion[['spot_location_1', 'spot_location_2','spot_location_3']].copy()
            
            ### rotate img
            if rotation_degree == -90:
                spatial['spot_location_2'] = self.size_single_img +1 - np.array(spatial['spot_location_2'])

            ### Give absolute coordinates
            spatial['spot_location_1'] = np.array(spatial['spot_location_1']) + np.array(df_config.loc[df_config['Tile']==i, 'Y_init'])
            spatial['spot_location_2'] = np.array(spatial['spot_location_2']) + np.array(df_config.loc[df_config['Tile']==i, 'X_init'])
            
            ### keep spots
            good_spots_idx = np.ones(spatial.shape[0])
            unique_cell_ids_portion = np.unique(spots_portion[res_name])
            for cell in unique_cell_ids_portion:
                test = spots_portion.loc[spots_portion[res_name]==cell,:].index.to_list()
                cell_in_block = img[int(np.mean(spatial.iloc[test,1])), int(np.mean(spatial.iloc[test,0]))]
                if cell_in_block != i:
                    good_spots_idx[test] = 0
            spatial = np.array(spatial.loc[good_spots_idx == 1., :])
            good_cellid = np.array(spots_portion.loc[good_spots_idx == 1, res_name])
            good_geneid = np.array(spots_portion.loc[good_spots_idx == 1, 'gene'])
            spot_merged = np.concatenate((spot_merged, spatial), axis=0) 
            
            if len(cellid)>0:
                if sum(good_cellid>=0)>0:
                    good_cellid[good_cellid>=0] += np.max(cellid) + 1
            cellid += list(good_cellid)
            geneid += list(good_geneid)
    spots_all = pd.DataFrame(zip(spot_merged[:,0], spot_merged[:,1], spot_merged[:,2], geneid, cellid), columns=['spot_merged_1', 'spot_merged_2', 'spot_merged_3', 'geneid', 'cellid'])
    return(spots_all)
 
            

