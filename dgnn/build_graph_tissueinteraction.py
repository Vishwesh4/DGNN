"""
Based on constructed processed complete graph, the script constructs different tissue interaction graph by sampling 
different tissue type patches and constructing directed graph with the positional embedding 
"""
import os
import argparse

import torch
import numpy as np
from pathlib import Path
import torch_geometric
from torch_geometric.data import Data as geomData
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

def pt2graph(wsi_h5, radius=9, dist_threshold=3, patch_size=224*4, add_pe=False):
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
    
    #Add PE
    if add_pe:
        pe_re = torch_geometric.transforms.AddRandomWalkPE(walk_length=24)
        rw = pe_re(G)
        G.random_walk_pe = torch.cat([rw.x,rw.random_walk_pe],dim=1)
    return G

if __name__=="__main__":
    TILE_SIZE=224
    TILE_STRIDE_SIZE=1
    DIST_THRESH = 20
    DENSITY_THRESH=0.25
    ADD_PE = True

    parser = argparse.ArgumentParser(description='Configurations for extraction of ')
    parser.add_argument('--graph', type=str, default='tumor_stroma', help='Choose out of [tumor_stroma,tumor_rest,stroma_rest]')
    parser.add_argument('--onco_code', type=str, default="BRCA", help='type code of TCGA dataset in all caps')
    args = parser.parse_args()

    SETTING = args.graph
    ONCO_CODE = args.onco_code

    print(args)

    INPUT_DIR = list(Path(f"/aippmdata/public/TCGA/TCGA-{ONCO_CODE}/images/").rglob("*.svs"))
    DENSITY_DIR = Path(f"/localdisk3/ramanav/TCGA_processed/DGNN_graphs/knn_no_sample_{ONCO_CODE}")
    OUTPUT_DIR = Path(f"/localdisk3/ramanav/TCGA_processed/DGNN_graphs/knn_no_sample_{ONCO_CODE}_{SETTING}_v2_t")

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
        if (DENSITY_DIR/f"{slide_name}_graph.pt").exists():
            data = torch.load(DENSITY_DIR/f"{slide_name}_graph.pt",map_location="cpu")
        else:
            continue
        coords = data.centroid
        feats = data.x
        try:
            density = data.density
        except:
            continue

        #Select nodes which qualify certain conditions
        if SETTING=="tumor_stroma":
            tissue_type1 = 0
            tissue_type2 = 1
        elif SETTING=="tumor_rest":
            tissue_type1 = 0
            tissue_type2 = 2
        elif SETTING=="stroma_rest":
            tissue_type1 = 1
            tissue_type2 = 2
        else:
            raise ValueError
        idx_type1 = torch.where(density[:,tissue_type1]>=DENSITY_THRESH)[0]
        idx_type2 = torch.where(density[:,tissue_type2]>=DENSITY_THRESH)[0]
        idx_interest = torch.unique(torch.cat([idx_type1,idx_type2]))
        coords_interest = coords[idx_interest,:].numpy()
        feats_interest = feats[idx_interest,:]
        density_interest = density[idx_interest,:].numpy()

        G = pt2graph({"coords":coords_interest,"features":feats_interest,"density":density_interest},dist_threshold=DIST_THRESH,patch_size=patch_size,add_pe=ADD_PE)
        G["slide_dim"] = torch.tensor([iw, ih])
        torch.save(G, str(OUTPUT_DIR/f"{slide_name}_graph.pt"))