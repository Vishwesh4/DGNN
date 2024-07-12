"""
Builds a directed graph for the sampled patches from a WSI. The graph is built using KNN with random walk positional embedding. Additional graph preprocessing 
is done such as removing repeated tissue sections or small disconnected tissue islands
"""

import os

import torch
import numpy as np
from pathlib import Path
import torch_geometric
from torch_geometric.data import Data as geomData
from torch_geometric.utils import to_scipy_sparse_matrix
import scipy.sparse as sp
import nmslib

from utils_graph.helper_functions.read_slide import read_slide 

class Hnsw:
    def __init__(self, space='cosinesimil', index_params=None,
                 query_params=None, print_progress=True):
        self.space = space
        self.index_params = index_params
        self.query_params = query_params
        self.print_progress = print_progress

    def fit(self, X):
        index_params = self.index_params
        if index_params is None:
            index_params = {'M': 16, 'post': 0, 'efConstruction': 400}

        query_params = self.query_params
        if query_params is None:
            query_params = {'ef': 90}

        # this is the actual nmslib part, hopefully the syntax should
        # be pretty readable, the documentation also has a more verbiage
        # introduction: https://nmslib.github.io/nmslib/quickstart.html
        index = nmslib.init(space=self.space, method='hnsw')
        index.addDataPointBatch(X)
        index.createIndex(index_params, print_progress=self.print_progress)
        index.setQueryTimeParams(query_params)

        self.index_ = index
        self.index_params_ = index_params
        self.query_params_ = query_params
        return self

    def query(self, vector, topn, dist_norm):
        # the knnQuery returns indices and corresponding distance
        # we will return the normalized distance based on dist_norm
        indices, dist = self.index_.knnQuery(vector, k=topn)
        return indices, np.sqrt(dist)/dist_norm

def pt2graph(wsi_h5, radius=9, dist_threshold=3, patch_size=224*4):
    """
    Main function to form graph based on KNN calculated based on eucledian distance.
    We throw away nodes which are above dist_threshold*patch_size
    """
    coords, features, density = wsi_h5['coords'], wsi_h5['features'], wsi_h5['density']
    assert coords.shape[0] == features.shape[0]
    num_patches = coords.shape[0]
    
    model = Hnsw(space='l2')
    model.fit(coords)
    a = np.repeat(range(num_patches), radius-1)
    b = []
    for v_idx in range(num_patches):
        out = np.stack(model.query(coords[v_idx], topn=radius, dist_norm=patch_size),axis=1)[1:,]
        b.append(out)
    b = np.concatenate(b)
    b = np.concatenate((a[:,np.newaxis],b),axis=1)
    #Set a distance threshold at 2.8 patch distance
    edge_spatial = torch.Tensor(b[np.where(b[:,2]<=dist_threshold)[0],:2].T).type(torch.LongTensor)

    if density is not None:
        density = torch.Tensor(density)

    G = geomData(x = features,
                 edge_index = edge_spatial,
                 density = density,
                 centroid = torch.Tensor(coords))
    return G

def graph_preprocessing(data, throw_number=10,same_tissue_percent=0.15,add_pe=False):
    """
    Preprocess the obtained graph by
    1. Removing components with less than certain number of nodes (throw_percent*max component size), small island of tissue
    2. Remove duplicate tissues (same tissue section duplicated in a slide)
        2.1 Identify duplicate tissues as tissues with component size within same_tissue_percent difference
        2.2 Calculate the centroid and compare the distance between the largest component
        2.3 The distance should be more than 1/number of detected such similar tissue components * max slide dimension
            This is based on assumption that similar tissues would be arranged length or breadth wise
    3. Adding positional embedding to the new graph
    """
    adj = to_scipy_sparse_matrix(data.edge_index, num_nodes=data.num_nodes)
    num_components, component = sp.csgraph.connected_components(
        adj, connection="weak")
    comps, comps_count = np.unique(component,return_counts=True)
    take_comps = []
    for i in range(len(comps)):
        if throw_number<1:
            throw_number = throw_number*np.max(comps_count)
        if comps_count[i]>=throw_number:
            take_comps.append(comps[i])

    #Remove similar tissue patches
    largest_comp = np.argmax(comps_count)
    diff_comps = comps_count[largest_comp] - comps_count
    sim_comp_index = np.where((diff_comps<same_tissue_percent*comps_count[largest_comp]) & (diff_comps>0))[0]
    w,h = slide.dimensions
    max_slide = max(h,w)
    #take random points
    lar_cent = data.centroid[np.isin(component,largest_comp)].mean(dim=0)
    dist_compare = max_slide
    for comp_idx in sim_comp_index:
        dist = torch.norm(data.centroid[np.isin(component,comp_idx)].mean(dim=0)-lar_cent)
        if dist/dist_compare > (1/(len(sim_comp_index)+1)):
            print("Detected same tissue sections in a slide, Removing...")
            take_comps.remove(comp_idx)
    subset = torch.tensor(np.isin(component,np.array(take_comps))).to(torch.bool)
    data_preprocessed = data.subgraph(subset)
    print("Before preprocessing: {}".format((list(comps), list(comps_count))))
    print("After preprocessing: {}".format((take_comps, comps_count[take_comps])))
    #Add PE
    if add_pe:
        pe_re = torch_geometric.transforms.AddRandomWalkPE(walk_length=RANDOM_WALK_LENGTH)
        rw = pe_re(data_preprocessed)
        data_preprocessed.random_walk_pe = torch.cat([rw.x,rw.random_walk_pe],dim=1)
    return data_preprocessed

if __name__=="__main__":
    TILE_SIZE=224
    TILE_STRIDE_SIZE=1
    #graph building hyperparameters
    DIST_THRESH = 20
    #Preprocessing hyperparameters
    THROW_PERC = 0.05
    SAME_TISSUE_PERC=0.15
    RANDOM_WALK_LENGTH=24
    PREPROCESS = True
    ADD_PE = True

    ONCO_CODE = "UCEC"
    INPUT_DIR = list(Path(f"/aippmdata/public/TCGA/TCGA-{ONCO_CODE}/images/").rglob("*.svs"))
    DENSITY_DIR = Path(f"/localdisk3/ramanav/TCGA_processed/DGNN_graphs/no_sample_{ONCO_CODE}/")
    OUTPUT_DIR = Path(f"/localdisk3/ramanav/TCGA_processed/DGNN_graphs/knn_no_sample_{ONCO_CODE}")
    if not OUTPUT_DIR.is_dir():
        os.mkdir(OUTPUT_DIR)

    processed_files = [files.stem.split("_")[0] for files in OUTPUT_DIR.glob("*.pt")]

    for paths in INPUT_DIR:
        slide_name = paths.stem.split(".")[0]
        slide, spacing = read_slide(paths)
        ih, iw = slide.dimensions
        print(f"Processing {slide_name}")
        patch_size = (1/spacing)*TILE_SIZE
        if slide_name in processed_files:
            print("Already processed...")
            continue
        if not (DENSITY_DIR/f"{slide_name}_featvec.pt").exists():
            continue
        data = torch.load(DENSITY_DIR/f"{slide_name}_featvec.pt",map_location="cpu")
        coords = data["coord"]
        feats = data["feat"]
        density = data["density"]
        if len(feats)<=15:
            continue
        G = pt2graph({"coords":coords,"features":feats,"density":density},dist_threshold=DIST_THRESH,patch_size=patch_size)
        if PREPROCESS:
            g_processed = graph_preprocessing(G, throw_number=THROW_PERC,same_tissue_percent=SAME_TISSUE_PERC,add_pe=ADD_PE)
        else:
            g_processed = G
        g_processed["slide_dim"] = torch.tensor([iw, ih])
        torch.save(g_processed, str(OUTPUT_DIR/f"{slide_name}_graph.pt"))