from .utils import *
from .preprocessing import *
from .postprocessing import *
from .metrics import *
from .stitch import *
from .tissue_mapping import *
from .Points2Cell import *
import random
import tifffile
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.morphology import convex_hull_image
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial import Delaunay, ConvexHull
from sklearn.metrics import adjusted_rand_score
from skimage.color import label2rgb
from tqdm import tqdm
from sklearn.neighbors import LocalOutlierFactor
from matplotlib.colors import ListedColormap
from anndata import AnnData
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

class ClusterMap():
    """
    Variables:
        *.spots: spatial transcriptomics data including spots location and gene identities.
        *.dapi:  DAPI image corresponding to spots
        *.dapi_binary: binarized DAPI image
        *.dapi_stacked: 2D DAPI image
        *.gene_list
        *.num_dims: number of data dimensions
        *.xy_radius 
        *.z_radius 
        *.cellcenter
        *.all_points: coordinates of all points (RNA spots + DAPI sampled spots)
        *.all_points_cellid: cell ID corresponding to *.all_points
        *.cellid_unique: values corrsponding to *.spots['clustermap']
        *.cellcenter_unique: coordinates of cell centers, each row corresponding to *.cellcenter_unique (Only RNA spots)
        
    
    Functions:
        *.preprocess - This function is designed for ....
        *.segmentation - This function is designed for ....
        *.create_convex_hulls
        *.plot_gene
        *.plot_segmentation
        *.plot_segmentation_3D
        *.calculate_metrics
        *.save_segmentation
        *.create_cell_adata
        *.cell_preprocess
        *.merge_multiple_clusters
        *.cell_typing
        *.plot_cell_typing
        *.combine_cell_type
        *.map_cell_type_to_spots
        *.stitch
    """
    
    def __init__(self, spots, dapi, gene_list, num_dims, xy_radius,z_radius,fast_preprocess=False,gauss_blur=False,sigma=1):
        
        '''
        params :    - spots (dataframe) = columns should be 'spot_location_1', 'spot_location_2',
                     ('spot_location_3'), 'gene'
                    - dapi (ndarray) = original dapi image
                    - gene_list (1Darray) = list of gene identities (encoded as ints)
                    - num_dims (int) = number of dimensions for cell segmentation (2 or 3)
        '''

        self.spots = spots
        self.dapi = dapi
        self.dapi_binary, self.dapi_stacked = binarize_dapi(self.dapi,fast_preprocess,gauss_blur, sigma)
        self.gene_list = gene_list
        self.num_dims = num_dims
        self.xy_radius = xy_radius
        self.z_radius = z_radius
        self.cellcenter = None
    
    def preprocess(self,dapi_grid_interval=5, LOF=False, contamination=0.1, pct_filter=0.1):
        preprocessing_data(self.spots, dapi_grid_interval, self.dapi_binary, LOF,contamination, self.xy_radius,pct_filter)

    def segmentation(self,cell_num_threshold=0.01, dapi_grid_interval=5, add_dapi=True,use_genedis=True):
        
        '''
        params :    - R (float) = rough radius of cells
                    - d_max (float) = maximum distance to use (often chosen as R)
                    - add_dapi (bool) = whether or not to add Dapi points for DPC
        '''
        
        spots_denoised = self.spots.loc[self.spots['is_noise']==0,:].copy()
        if 'level_0' in self.spots.columns:
            spots_denoised=spots_denoised.drop('level_0',axis=1)
        spots_denoised.reset_index(inplace=True)
        print(f'After denoising, mRNA spots: {spots_denoised.shape[0]}')
        
        print('Computing NGC coordinates')
        ngc = NGC(self, spots_denoised)
        if add_dapi:
            all_coord, all_ngc = add_dapi_points(self.dapi_binary, dapi_grid_interval,spots_denoised, ngc, self.num_dims)
            self.num_spots_with_dapi=all_coord.shape[0]
            print(f'After adding DAPI points, all spots:{self.num_spots_with_dapi}')
            print('DPC')
            cell_ids = DPC(self,all_coord, all_ngc,cell_num_threshold,use_genedis)
        else:
            self.num_spots_with_dapi=0
            spatial = np.array(spots_denoised[['spot_location_1', 'spot_location_2', 'spot_location_3']]).astype(np.float32)
            print('DPC')
            cell_ids = DPC(self,spatial,ngc, cell_num_threshold, use_genedis)
        
        self.spots['clustermap'] = -1      
        
        # Let's keep only the spots' labels
        if 'level_0' in spots_denoised:
            self.spots.loc[spots_denoised.loc[:, 'level_0'], 'clustermap'] = cell_ids[:len(ngc)]
        else:
            self.spots.loc[spots_denoised.loc[:, 'index'], 'clustermap'] = cell_ids[:len(ngc)]

        
        print('Postprocessing')
        erase_small_clusters(self.spots,self.min_spot_per_cell)
        res_over_dapi_erosion(self.num_dims, self.spots, self.dapi_binary)
        
        # Keep all spots id for plotting
        is_remain=np.in1d(cell_ids, self.spots['clustermap'].unique())
        self.all_points=all_coord[is_remain]
        self.all_points_cellid=cell_ids[is_remain]        
        
        
        # range cell centers in order of self.cellid_unique
        cellid_unique=self.spots['clustermap'].unique()
        cellid_unique=cellid_unique[cellid_unique>=0]
        self.cellid_unique=cellid_unique
        self.cellcenter_unique=self.cellcenter[self.cellid_unique.astype('int'),:]
        
        
    def create_convex_hulls(self, plot_with_dapi=True, figscale = 50,
                            bg_color=[1,1,1], good_cells=None, width=5,
                            height=5,alpha=1,vmin=None,vmax=None,k=0.9,
                            rescale_colors=False, figsize=(10,10)):
        
        '''
        Plot the results of segmentation with convex hull instead of customized cell shapes
        '''
        if plot_with_dapi:
            cell_ids = self.all_points_cellid
            cells_unique = np.unique(cell_ids)
            spots_repr = self.all_points
            
        else:
            cell_ids = self.spots['clustermap']
            cells_unique = np.unique(cell_ids)
            spots_repr = np.array(self.spots[['spot_location_2', 'spot_location_1']])
            
            
        cells_unique = cells_unique[cells_unique>=0]
        img_res = np.zeros(self.dapi_stacked.shape)
        spots_repr[:, [0,1]] = spots_repr[:, [1,0]]
        Nlabels = cells_unique.shape[0]
        hulls = []
        coords = []
        num_cells = 0
        print('Creat cell convex hull')
        for i in cells_unique:   
            curr_coords = spots_repr[cell_ids==i,0:2]
            curr_coords=reject_outliers(curr_coords)
            if curr_coords.shape[0] < 100000 and curr_coords.shape[0] > 50:
                num_cells += 1
                hulls.append(ConvexHull(curr_coords))
                coords.append(curr_coords)
        print("Used %d / %d" % (num_cells, Nlabels))        
        
        if self.num_dims==2:
            dapi_2D=self.dapi
        else:
            dapi_2D=np.sum(self.dapi,axis=2)
            
        
        plt.figure(figsize=(figscale*width/float(height),figscale))
        polys = [hull_to_polygon(h,k) for h in hulls]
        if good_cells is not None:
            polys = [p for i,p in enumerate(polys) if i in good_cells]
        p = PatchCollection(polys,alpha=alpha, cmap='tab20',edgecolor='k', linewidth=0.5)
        colors=cell_ids
        if vmin or vmax is not None:
            p.set_array(colors)
            p.set_clim(vmin=vmin,vmax=vmax)
        else:
            if rescale_colors:
                p.set_array(colors+1)
                p.set_clim(vmin=0, vmax=max(colors+1))
            else:
                
                p.set_array(colors)
                p.set_clim(vmin=0, vmax=max(colors))        
        # dapi_2D = (dapi_2D > 0).astype(np.int)
        plt.imshow(dapi_2D,cmap=plt.cm.gray_r,alpha=0.35,origin='lower')
        plt.gca().add_collection(p)      
    
    def plot_gene(self,marker_genes, genes_list, figsize=(5,5),c='r',s=1):
        for i in range(len(marker_genes)):
            plt.figure(figsize=figsize)
            if self.num_dims==3:
                plt.imshow(np.sum(self.dapi_binary,axis=2),origin='lower', cmap='binary_r')
            else:
                plt.imshow(self.dapi_binary,origin='lower', cmap='binary_r')
            plt.scatter(self.spots.loc[self.spots['gene']==1+genes_list.index(marker_genes[i]),'spot_location_1'],
                        self.spots.loc[self.spots['gene']==1+genes_list.index(marker_genes[i]),'spot_location_2'],
                        c=c,s=s)
            plt.title(marker_genes[i])
            plt.show()
        
        
    def plot_segmentation(self,figsize=(10,10),plot_with_dapi=True, plot_dapi=False,method='clustermap',s=5,cmap=None, show=True,save=False,savepath=None):
        
        cell_ids = self.spots[method]
        cells_unique = np.unique(cell_ids)
        try:
            spots_repr = np.array(self.spots[['spot_location_2', 'spot_location_1']])[cell_ids>=0]
        except:
            print('No cell is identified!')
            return
        cell_ids=cell_ids[cell_ids>=0]
        
        if method == 'clustermap':
            if plot_with_dapi:
                cell_ids = self.all_points_cellid
                cells_unique = np.unique(cell_ids)
                spots_repr = self.all_points[cell_ids>=0]
                cell_ids=cell_ids[cell_ids>=0]
        if len(cell_ids)==0:
            print('Error:cell id is empty!')
            return       
        if not show:
            plt.ioff()
        plt.figure(figsize=figsize)
        if cmap is None:
            myList = [];

            cmap=np.random.rand(int(max(cell_ids)+1),3)
        
        if plot_dapi:
            if self.num_dims==3:
                plt.imshow(np.sum(self.dapi_binary,axis=2),origin='lower', cmap='binary_r')
            elif self.num_dims==2:
                plt.imshow(self.dapi_binary,origin='lower', cmap='binary_r')
            plt.scatter(spots_repr[:,1],spots_repr[:,0],
            c=cmap[[int(x) for x in cell_ids]],s=s)
        else:
            plt.scatter(spots_repr[:,1],spots_repr[:,0],
            c=cmap[[int(x) for x in cell_ids]],s=s)

        plt.title('Segmentation')
        if save:
            plt.savefig(savepath)
        if show:
            plt.show()
        
    def plot_segmentation_3D(self,figsize=(10,10),elev=45,azim=-45, method='clustermap',s=10,cmap=None):
        
        cell_ids = self.spots[method]
        cells_unique = np.unique(cell_ids)
        spots_repr = np.array(self.spots[['spot_location_2', 'spot_location_1','spot_location_3']])[cell_ids>=0]
        cell_ids=cell_ids[cell_ids>=0]
        
        if cmap is None:
            cmap=np.random.rand(int(max(cell_ids)+1),3)
        
        fig = plt.figure(figsize=figsize)
        ax = plt.axes(projection='3d')
        ax.view_init(elev=elev, azim=azim)
        ax.scatter3D(spots_repr[:,1], 
                     spots_repr[:,0],  
                     spots_repr[:,2], 
                     c=cmap[[int(x) for x in cell_ids]],
                     s=s)

        X=spots_repr[:,1]
        Y=spots_repr[:,0]
        Z=spots_repr[:,2]
        #Create cubic bounding box to simulate equal aspect ratio
        max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
        Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(X.max()+X.min())
        Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Y.max()+Y.min())
        Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Z.max()+Z.min())
        # Comment or uncomment following both lines to test the fake bounding box:
        for xb, yb, zb in zip(Xb, Yb, Zb):
           ax.plot([xb], [yb], [zb], 'w')
        plt.title(f'3D cell map in {method}')
            
    def calculate_metrics(self, gt_column):

        '''
        params :    - gt_column (str) : name of the column where ground truth's results are stored
        '''

        self.underseg, self.overseg = compute_metrics_over_under(self.spots, method='clustermap', real_res=gt_column)
        print(f'OverSegmentation Score = {self.overseg} \nUnderSegmentation Score = {self.underseg}')
        return(self.underseg, self.overseg)
    
        
    def save_segmentation(self, path_save):
        
        self.spots.to_csv(path_save, index=False)
    
    def create_cell_adata(self,cellid, geneid, gene_list, genes, num_dims):
        
#         
            
        ### find unique cell id
        cellid_unique=self.spots[cellid].unique()
        cellid_unique=cellid_unique[cellid_unique>=0]
        self.cellid_unique=cellid_unique
        
        ### compute cell x gene matrix and obs
        gene_expr_vector = np.zeros((len(cellid_unique), len(gene_list)))
        obs=np.zeros((len(cellid_unique),num_dims))
        gene_expr=self.spots.groupby([cellid, geneid]).size()
        
        for ind,i in enumerate(cellid_unique):
            gene_expr_vector[ind, gene_expr[i].index-np.min(gene_list)] = gene_expr[i].to_numpy()
            if self.cellcenter is None:
                if num_dims==3:
                    obs[ind,:]=self.spots.loc[self.spots[cellid]==i,['spot_location_1','spot_location_2','spot_location_3']].mean(axis=0)
                else:
                    obs[ind,:]=self.spots.loc[self.spots[cellid]==i,['spot_location_1','spot_location_2']].mean(axis=0)
            else:
                if num_dims==3:
                    obs[ind,:]=self.cellcenter[int(i),0:3]
                else:
                    obs[ind,:]=self.cellcenter[int(i),0:2]
            
#             obs[ind,:]=self.cellcenter[int(i),:]
        if num_dims==3:
            obs=pd.DataFrame(data=obs,columns=['col','row','z'])
        else:
            obs=pd.DataFrame(data=obs,columns=['col','row'])
        var = pd.DataFrame(index=genes[0])
        self.cell_adata = AnnData(X=gene_expr_vector, var=var, obs=obs)
        
    def cell_preprocess(self, min_genes=3, min_cells=3, min_counts=5, 
                    n_neighbors=30, n_pcs=15):
        '''
        Performs preprocess for cells using scanpy
        Params:
        - n_neighbors (20) = number of neighbors to use for scanpy pp.neighbors
        '''
        # Calculate QC metrics
        sc.pp.calculate_qc_metrics(self.cell_adata, percent_top=None, inplace=True)
        
        # Plot top 20 most expressed genes 
        sc.pl.highest_expr_genes(self.cell_adata, n_top=20)
        sns.jointplot(x="total_counts", y="n_genes_by_counts", data=self.cell_adata.obs, kind="hex")
        plt.xlabel("# Spots per cell")
        plt.ylabel("# Genes per cell")
        plt.show()
        
        # Filtration 
        sc.pp.filter_cells(self.cell_adata, min_genes=min_genes)
        sc.pp.filter_genes(self.cell_adata, min_cells=min_cells)

        sc.pp.filter_cells(self.cell_adata, min_counts=min_counts)
        
        # Normalization scaling
        sc.pp.normalize_total(self.cell_adata)
        sc.pp.log1p(self.cell_adata)

        # Save raw data
        self.cell_adata.raw = self.cell_adata
        # Scale data to unit variance and zero mean
        sc.pp.regress_out(self.cell_adata, ['total_counts'])
        sc.pp.scale(self.cell_adata)

        # Run PCA
        sc.tl.pca(self.cell_adata, svd_solver='arpack')     
        
        # Computing the neighborhood graph
        sc.pp.neighbors(self.cell_adata, n_neighbors=n_neighbors, n_pcs=n_pcs)

        # Run UMAP
        sc.tl.umap(self.cell_adata)
    
    def merge_multiple_clusters(self, merge_list):
        self.cell_adata.obs['cell_type']=self.cell_adata.obs['cell_type'].astype('int')
        for sublist in merge_list:   
            self.cell_adata.obs.loc[self.cell_adata.obs['cell_type'].isin(sublist),'cell_type']=sublist[0]
        a1=list(np.unique(self.cell_adata.obs['cell_type']))
        self.cell_adata.obs['cell_type']=list(map(lambda x: a1.index(x), self.cell_adata.obs['cell_type']))
        self.cell_adata.obs['cell_type']=self.cell_adata.obs['cell_type'].astype('category')
        
    def cell_typing(self, use_adata=None, random_state=42, resol=1, n_clusters=3, cluster_method=None):
        
        '''
        Performs cell typing.

        params :    
                    - resol (float) = resolution of Leiden of Louvain clustering
                    - n_clusters (int) = number of clusters to determine (in case we are using agglomerative clustering)
                    - cluster_method (str) = type of clustering for cell typing. Can be 'leiden', 'louvain', or 'hierarchical'
        '''
        if use_adata is None:
            adata=self.cell_adata
        else:
            adata=use_adata
            
        print(f'clustering method: {cluster_method}')
        if cluster_method == 'leiden':
            print('Leiden clustering')
            sc.tl.leiden(adata, resolution=resol, random_state=random_state, key_added='cell_type')
        elif cluster_method == 'louvain':
            sc.tl.louvain(adata, resolution=resol, random_state=random_state, key_added='cell_type')
        elif cluster_method == 'aggre':
            cluster = AgglomerativeClustering(n_clusters=n_clusters,
                                              affinity='euclidean', linkage='ward')
            adata.obs['cell_type'] = cluster.fit_predict(adata.obsm['X_pca']).astype(str)
        num_cluster=len(adata.obs['cell_type'].unique() )
        print(f'Get {num_cluster} clusters')
            
            
    def plot_cell_typing(self, use_adata=None,umap=True,heatmap=True,
                         print_markers=True,celltypemap=True):
        '''
        Plot cell typing results in adata.obs['cell_type'].

        params :    
                    - use_adata: if none then use self.cell_adata, else use use_adata.
        '''
        if use_adata is None:
            adata=self.cell_adata
        else:
            adata=use_adata
            
        # Get colormap
        cluster_pl = sns.color_palette('tab20',len(adata.obs['cell_type'].unique()))
        cluster_cmap = ListedColormap(cluster_pl)
#         sns.palplot(cluster_pl)
        
        if umap:
            # Plot UMAP with cluster labels w/ new color
            sc.pl.umap(adata, color='cell_type', legend_loc='on data',
                       legend_fontsize=12, legend_fontoutline=2, frameon=False, 
                       title='Clustering of cells', palette=cluster_pl)

        sc.tl.rank_genes_groups(adata, 'cell_type', method='t-test')

        # Get markers for each cluster
        sc.tl.rank_genes_groups(adata, 'cell_type', method='t-test')
        sc.tl.filter_rank_genes_groups(adata, min_fold_change=0.01)
        
        if heatmap:
            sc.pl.rank_genes_groups_heatmap(adata, n_genes=5, min_logfoldchange=1, use_raw=False, swap_axes=True,
                                            vmin=-3, vmax=3, cmap='bwr', show_gene_labels=True,
                                            dendrogram=False, figsize=(20, 10))

        if print_markers:
            markers = []
            temp = pd.DataFrame(adata.uns['rank_genes_groups']['names']).head(5)
            for i in range(temp.shape[1]):
                curr_col = temp.iloc[:, i].to_list()
                markers = markers + curr_col
                print(i, curr_col)
        
        #plot cells in centers
        col=adata.obs['col'].tolist()
        row=adata.obs['row'].tolist()
#             z=self.cell_adata.obs['z'].tolist()
        cell_type=adata.obs['cell_type'].tolist()
        cell_type = [int(item) for item in cell_type]
        if celltypemap:
            plt.figure(figsize=((3*max(col)/max(row),3)))
            if max(col)>max(row):
                plt.scatter(col, row, s=50,edgecolors='none', c=np.array(cluster_pl)[cell_type])
            else:
                plt.scatter(row, col, s=50,edgecolors='none', c=np.array(cluster_pl)[cell_type])
            plt.title('cell type map')
            plt.axis('equal')
#             plt.axis('off')
            plt.tight_layout()
            
        return(cluster_pl)
    
    def combine_cell_type(self, list_adata):
        self.cell_adata.obs['cell_type']=-1
        for sub_adata in list_adata:
            sub_adata.obs['cell_type']=sub_adata.obs['cell_type'].astype('int')
            self.cell_adata.obs.loc[sub_adata.obs.index,'cell_type'] = sub_adata.obs['cell_type']+max(self.cell_adata.obs['cell_type'])+1
    
        
    def map_cell_type_to_spots(self, cellid):
        self.spots['cell_type']=-1
        for ind,i in enumerate(self.cellid_unique):
            try:
                test=list(self.cell_adata.obs.index).index(str(ind))
                self.spots.loc[self.spots[cellid]==i,'cell_type']=int(self.cell_adata.obs.loc[str(ind),'cell_type'])
            except ValueError:
                test=0
#         self.spots['cell_type']=-1
#         for ind,i in enumerate(self.cellid_unique):
#             self.spots.loc[self.spots[cellid]==i,'cell_type']=int(self.cell_adata.obs['cell_type'][ind])

    def stitch(self,model_tile,out,tile_num ):
        ### stitch based on label_img
        label_img=out.loc[tile_num,'label_img']       
        
        ### remove cells outside label img
        outlabel_id=[]
        for i in range(model_tile.cellcenter_unique.shape[0]):
            if label_img[model_tile.cellcenter_unique[i,0],model_tile.cellcenter_unique[i,1],]!=tile_num:
                outlabel_id.append(model_tile.cellid_unique[i])
        model_tile.spots.loc[model_tile.spots['clustermap'].isin(outlabel_id),'clustermap']=-1
        model_tile.spots.loc[model_tile.spots['clustermap'].isin(outlabel_id),'is_noise']=-1        
        
        ### reassign cell center
        cellid_unique=model_tile.spots['clustermap'].unique()
        cellid_unique=cellid_unique[cellid_unique>=0]
        model_tile.cellid_unique=cellid_unique
        model_tile.cellcenter_unique=model_tile.cellcenter[model_tile.cellid_unique.astype('int'),:]

        ### get current max cellid
        model_cellid_max=max(self.spots['clustermap'])+1

        ### stitch spots
        model_tilespots=model_tile.spots.loc[model_tile.spots['clustermap']>=0,:]

        self.spots.loc[model_tilespots['index'],'clustermap']=list(model_cellid_max+ model_tilespots['clustermap'])

        
        ### stitch cell centers        
        updated_all_points=model_tile.cellcenter_unique.copy()
        updated_all_points[:,1]=updated_all_points[:,1]-model_tilespots.iloc[0]['spot_location_1']+ self.spots.loc[model_tilespots.iloc[0]['index'],'spot_location_1']
        updated_all_points[:,0]=updated_all_points[:,0]-model_tilespots.iloc[0]['spot_location_2']+ self.spots.loc[model_tilespots.iloc[0]['index'],'spot_location_2']
        
        try:
            self.cellid_unique
            self.cellid_unique=np.concatenate((self.cellid_unique,
                                              model_tile.cellid_unique+model_cellid_max),axis=0)
            self.cellcenter_unique=np.concatenate((self.cellcenter_unique,
                                              updated_all_points),axis=0)
        except:
            self.cellid_unique=model_tile.cellid_unique+model_cellid_max
            self.cellcenter_unique=updated_all_points
    
    
        if model_tilespots.shape[0]>0:
            ### add all_points and all_points_cellid
            updated_all_points=model_tile.all_points.copy()
            updated_all_points[:,1]=updated_all_points[:,1]-model_tilespots.iloc[0]['spot_location_1']+ self.spots.loc[model_tilespots.iloc[0]['index'],'spot_location_1']
            updated_all_points[:,0]=updated_all_points[:,0]-model_tilespots.iloc[0]['spot_location_2']+ self.spots.loc[model_tilespots.iloc[0]['index'],'spot_location_2']
        
            updated_all_points_cellid=model_tile.all_points_cellid.copy()
            final_spots=[x in model_tilespots['clustermap'].unique() for x in updated_all_points_cellid]
            updated_all_points_cellid=model_tile.all_points_cellid[final_spots]
            updated_all_points = updated_all_points[final_spots,:]

            try:
                self.all_points_cellid
                self.all_points_cellid=np.concatenate((self.all_points_cellid,updated_all_points_cellid),axis=0)
                self.all_points=np.concatenate((self.all_points,updated_all_points),axis=0)
            except:
                self.all_points_cellid=updated_all_points_cellid
                self.all_points=updated_all_points
    

    
    
class StitchSpots():
    def __init__(self, path_res, path_config, res_name):

        '''
        params :    - path_res (str) = root path of the results of AutoSeg's segmentation
                    - path_config (str) = path of tile configuration
                    - res_name (str) = name of the column where AutoSeg's results are stored in each dataset
        '''
        
        self.path_res = path_res
        self.path_config = path_config
        self.res_name = res_name
       
    def gather_tiles(self):
        print('Gathering tiles')
        self.spots_gathered = gather_all_tiles(self.path_res, self.res_name)

    def stitch_tiles(self):
        print('Loading config')
        self.config = load_tile_config(self.path_config)
        print('Stitching tiles')
        self.img, self.num_col, self.num_row = create_img_label(self.config)
        self.spots_all = stitch_all_tiles(self.spots_gathered, self.img, self.num_col, self.num_row, self.config, self.res_name)
    
    def plot_stitched_data(self, figsize=(16,10), s=0.5):
        spots_all_repr = self.spots_all.loc[self.spots_all['cellid']>=0,:]
        plt.figure(figsize=figsize)
        palette = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                for k in range(spots_all_repr['cellid'].unique().shape[0])]
        sns.scatterplot(x='spot_merged_1', y='spot_merged_2', data=spots_all_repr, hue='cellid', legend=False, s=s, palette=palette)
        plt.title('Stitched dataset')
        plt.show()
    
    def save_stitched_data(self, path_save):
        self.spots_all.to_csv(path_save, index=False)

  
        

class TissueMapping():
    def __init__(self, path_spots, path_all_spots):
        self.path_spots = path_spots
        self.path_all_spots = path_all_spots
        self.spots = pd.read_csv(path_spots)
        self.all_spots = pd.read_csv(path_all_spots)
        try:
            a = self.spots['cell_type']
        except KeyError as err:
            print('You need to do the cell typing first. See CellTyping() class')
        
    def compute_ncc(self, radius):
        print('Computing cell neighborhoods')
        self.ncc = neighboring_cell_types(self.spots, radius)
    
    def identify_tissues(self, n_neighbors=30, resol=0.1, figsize=(30,20), s=0.5):
        adata = sc.AnnData(self.ncc)
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep='X', random_state=42)
        sc.tl.umap(adata, random_state=42)

        print('Leiden clustering')
        sc.tl.leiden(adata, resolution=resol, random_state=42, key_added='tissue')
        sc.pl.umap(adata, color='tissue', palette='gist_ncar')
        adata.obs['index'] = np.array(self.spots['index'].astype(int))
        self.adata = adata
        
        print('Assign tissue label to each spot')
        tissue2spot(adata, self.all_spots, 'cellid')

        repres = self.all_spots.loc[self.all_spots['tissue']>=0,:]
        plt.figure(figsize=figsize)
        sns.scatterplot(x='spot_merged_1', y='spot_merged_2', data=repres, hue='tissue', s=s, palette=sns.color_palette('Paired', len(repres['tissue'].unique())))
        plt.title('Tissue regions')
        plt.show()
    
    def save_tissues(self, save_path):
        self.spots.to_csv(save_path, index=False)

class CellNiches():
    def __init__(self, centroid_path, gene_expr_path):
        self.centroid_path = centroid_path
        self.centroids = pd.read_csv(centroid_path)
        try:
            self.gene_expr = np.load(gene_expr_path)
        except KeyError as err:
            print('You need to apply cell typing first')
        self.tri = Delaunay(self.centroids[['cell_center_1', 'cell_center_2', 'cell_center_3']]).simplices
    
    def compute_counts(self):
        cell_types = np.unique(self.centroids['cell_type'])
        mean_cell_types = []
        counts_per_cell_type = []
        for cell_type in cell_types:
            print(f'Processing cell type : {cell_type}')
            cells = np.unique(self.centroids.loc[self.centroids['cell_type']==cell_type,:].index.to_list())
            counts = np.zeros((len(cells),len(cell_types)))
            for i,cell in enumerate(cells):
                connected_idx = [cell in self.tri[i] for i in range(len(self.tri))]
                connected_cells = np.unique(self.tri[connected_idx])
                connected_types = self.centroids.loc[connected_cells,:].groupby('cell_type').size()
                counts[i, connected_types.index.to_list()] = np.array(connected_types)
            counts_per_cell_type.append(counts)
            mean = np.mean(counts, axis=0)
            mean_cell_types.append(mean)
        self.mean_cell_types = mean_cell_types
        self.counts_per_cell_type = counts_per_cell_type
    
    def plot_stats(self, figsize=(20,25), lw=2, size=1):
        fig, axes = plt.subplots(nrows=len(self.centroids['cell_type'].unique()), ncols=1, figsize=figsize)
        plt.rc('font', size=15)
        plt.rc('axes', labelsize=10)
        plt.rc('xtick', labelsize=10)

        for i in range(len(self.mean_cell_types)):
            dg = pd.DataFrame(zip(np.round(self.mean_cell_types[i],2), np.arange(len(self.mean_cell_types[0]))), columns=['value', 'cell_type'])
            pl = sns.barplot(x='cell_type', y='value', data=dg, ax=axes[i],fc='white', ec='black',lw=lw)
            pl.bar_label(pl.containers[0])
        
            df = pd.DataFrame(zip(np.round(np.ravel(self.counts_per_cell_type[i]),2), len(self.counts_per_cell_type[i])*list(np.arange(len(self.mean_cell_types[0])))), columns=['value', 'cell_type'])
            sns.stripplot(x='cell_type', y='value', data=df, palette='pastel',ax=axes[i], size=size)
            axes[i].set_frame_on(False)
            axes[i].set_xticks([])
            axes[i].set_yticks([])
            axes[i].set_xlabel('')
            axes[i].set_ylabel('Cell Type '+str(i))
            axes[i].axhline()
        axes[-1].set_xlabel('Cell types')
        plt.xticks(np.arange(len(self.mean_cell_types)), ['Cell Type '+str(i) for i in range(len(self.mean_cell_types))])
        plt.show()
                
    def discover_subclusters(self, resol_gene, resol_niche, n_neighbors):
        '''
        Compare subcelltyping from (1) gene expression subclustering and (2) cell niche expression clustering

        params :    - resol_gene (float) = Leiden resolution for gene expression clustering
                    - resol_niche (float) = Leiden resolution for niche clustering
                    - n_neighbors (int) = number of neighbors to use to ompute Leiden's graph
        '''
        ari_list = []
        cell_types = np.unique(self.centroids['cell_type'])
        for ct in cell_types:
            print(f'Processing cell type : {ct}')
            ### Clustering with gene expression
            idx = np.array(self.centroids.loc[self.centroids['cell_type']==ct, :].index, dtype=int)
            expr_ct = self.gene_expr[idx]
            adata_g = sc.AnnData(expr_ct)
            sc.pp.neighbors(adata_g, n_neighbors=n_neighbors, random_state=42, use_rep='X')
            sc.tl.umap(adata_g, random_state=42)
            sc.tl.leiden(adata_g, resolution=resol_gene, random_state=42, key_added='subclusters')
            sc.pl.umap(adata_g, color='subclusters')
            
            ## Clustering with niche expressions
            adata_n = sc.AnnData(self.counts_per_cell_type[int(ct)])
            sc.pp.neighbors(adata_n, n_neighbors=30, random_state=42, use_rep='X')
            sc.tl.leiden(adata_n, resolution=resol_niche, random_state=42, key_added='niche_subclusters')
            adata_g.obs['niche_subclusters'] = adata_n.obs['niche_subclusters']
            sc.pl.umap(adata_g, color='niche_subclusters')
            
            ari = adjusted_rand_score(adata_g.obs['subclusters'], adata_n.obs['niche_subclusters'])
            print(f'Adjusted Rand index for cell type {ct} = {ari}')
            
            ari_list.append(ari)