"""
This script is about performing segmentation and sampling patches from different tissue regions accordingly
"""

import sys
from collections import OrderedDict
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import pandas as pd
import openslide
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import torchvision
import matplotlib
from matplotlib import pyplot as plt
from torch import nn, optim
import torch.nn.functional as F
from sklearn.neighbors import KDTree

from utils_graph.get_samples import Get_Samples
from dplabtools.slides.processing import WSITissueMask
from dplabtools.slides.patches import WholeImageGridPatches
from dplabtools.slides import GenericSlide
from utils.extract_patches import ExtractPatches
from utils_graph.helper_functions.read_slide import read_slide

#Set for TCGA, should be changed for other dataset
GenericSlide.set_external_mpp(0.25)

def load_model_weights(model, weights):
    model_dict = model.state_dict()
    weights = {k: v for k, v in weights.items() if k in model_dict}
    if weights == {}:
        print('No weight could be loaded..')
    model_dict.update(weights)
    model.load_state_dict(model_dict)

    return model

def get_coordinates(slide_path):
    #Ensure all the parameters are same as how the patch features were extracted
    scan = openslide.OpenSlide(slide_path)
    size = 1000
    level_dimensions = scan.level_dimensions
    for i,levels in enumerate(level_dimensions[::-1]):
        #Select level with dimension around 1000 otherwise it becomes too big
        if levels[0]>size or levels[1]>size:
            break
    level = len(level_dimensions)-1-i   
    mask = WSITissueMask(wsi_file=slide_path,
                                  level_or_minsize=level,
                                  mode="lab",
                                  color_threshold=0.1,
                                  close_fill_kernel_size=9,
                                  remove_small_holes_ratio=0,
                                  remove_small_objects_ratio=0,
                                  remove_all_holes=True)

    grid_patches = WholeImageGridPatches(wsi_file = slide_path,
                                    mask_data = mask.array,
                                    patch_size = TILE_SIZE,
                                    level_or_mpp = TILE_MPP,
                                    foreground_ratio=0.95,
                                    patch_stride = TILE_STRIDE_SIZE,
                                    overlap_ratio = 1.0,
                                    weak_label = "label")
    coordinates = np.array([coords[0] for coords in grid_patches.patch_data])
    return coordinates

def perform_segmentation(wsi_path, til_scorer, all_portions=False):
    """
    Performs segmentation and returns the tumorbed map as well as tissue and TILs segmentation
    We assume TCGA slides are around 0.25 MPP, hence we take double of the spacing. If less than 0.2 then we assume its an error
    """
    slide, orig_spacing = read_slide(str(wsi_path))
    if orig_spacing<0.3:
        if orig_spacing<0.2:
            #Slide has incorrect spacing information
            orig_spacing = 0.25
        #For dealing with 40x, convert to 20x
        spacing = 2*orig_spacing
    else:
        #assuming the spacing to be around 0.5
        spacing = orig_spacing
    try:
        patches_dataset = ExtractPatches(wsi_file=str(wsi_path),
                                patch_size=SEG_TIL_SIZE,
                                foreground_threshold=0.95,
                                patch_stride=SEG_STRIDE,
                                level_or_mpp = spacing,
                                mask_threshold=0.1,
                                mask_kernelsize=9,
                                num_workers=10)        
    except Exception as e:
        print(f"Error: {e}")
        return None
    til_scorer.reset()
    til_scorer.set_spacing(spacing)
    all_densities, tumorbedmap = til_scorer.construct_density(patches_dataset.dataset,patches_dataset.coordinates,patches_dataset.template, patches_dataset.scan)
    with open(str(DENSITY_DIR/f"{wsi_path.stem.split('.')[0]}.npy"),"wb") as f:
        np.savez(f,all_densities=all_densities,tumorbedmap=tumorbedmap)
    all_densities = til_scorer.prepare_density(all_densities,tumorbedmap,wsi_path,all_portions)
    return all_densities

DEVICE = torch.device("cuda:0")

#Risk prediction happens at 1MPP for more encompassing resolution
TILE_SIZE=224
TILE_STRIDE_SIZE=1
TILE_MPP=1.0
#Segmentation happens at 0.5MPP. Model trained with bigger patch window
SEG_TIL_SIZE=512
SEG_STRIDE=0.5

ONCO_CODE = "UCEC"
INPUT_DIR = list(Path(f"../dataset/TCGA/TCGA-{ONCO_CODE}/images/").rglob("*.svs"))
DENSITY_DIR = Path(f"../dataset/TCGA_processed/density_maps/density_maps_{ONCO_CODE}")
OUTPUT_DIR = Path(f"../dataset/TCGA_processed/patch_samples/no_sample_{ONCO_CODE}")
FEATURE_DIR = Path(f"../dataset/TCGA_processed/features/TCGA_MIL_Patches_Ctrans_1MPP_{ONCO_CODE}")
Path.mkdir(DENSITY_DIR, parents=True, exist_ok=True)
Path.mkdir(OUTPUT_DIR, parents=True, exist_ok=True)

TIL_PATH="./utils_graph/segmentation_model_weights/segmentation_model_weights.pt"
TUM_PATH = "/utils_graph/segmentation_model_weights/tumorbed_network_weights.pt"

log_text = ""
processed_files = [files.stem.split("_")[0] for files in OUTPUT_DIR.glob("*.pt")]

til_scorer =  Get_Samples(til_model_path=TIL_PATH,
                            tumor_model_path=TUM_PATH,
                            transform=torchvision.transforms.ToTensor(),
                            threshold_tumor=0.5,
                            threshold_cell=0.5,
                            device=DEVICE,
                            force_compute=True,
                            plot_maps=False,
                            peritumor_boundary=None,
                            graph_node_resolution=TILE_MPP, #1MPP
                            graph_patch=TILE_SIZE,
                            num_stroma=None,
                            init_stroma_prob=None,
                            init_rest_prob=None,
                            nontils_prob=None,
                            percentile_tils=None,
                        )

for paths in tqdm(INPUT_DIR):  
    slide_name = paths.stem.split(".")[0]
    print(f"Processing {slide_name}...")
    if slide_name in processed_files:
        print("Already processed...")
        continue
    #Get samples by processing density file
    til_scorer.reset()
    patches_coords = []
    if (DENSITY_DIR/f"{slide_name}.npy").exists():
        all_densities = til_scorer.load_density(density_path=str(DENSITY_DIR/f"{slide_name}.npy"),
                                slide_path=str(paths),all_portions=False
                                )
    else:
        #Perform segmentation
        print("Density file does not exist, performing segmentation")
        all_densities = perform_segmentation(paths, til_scorer, all_portions=False)
        
    if (all_densities is not None) and (all_densities.size>0):
        invasive_coords, stroma_coords, rest_coords, patches_coords, patch_densities = til_scorer.sample_regions(all_densities,no_sampling=True)       

    #Get feature map
    if (FEATURE_DIR/f"{slide_name}_featvec.pt").exists():
        feature_vec = torch.load(FEATURE_DIR/f"{slide_name}_featvec.pt")
        coordinates = get_coordinates(paths)
        assert len(feature_vec)==len(coordinates)
    else: #no patches were detected, faulty slide
        print(f"Warning...Feature vector file not present for {slide_name}")
        continue
    if len(patches_coords)>=50:
        tree = KDTree(coordinates)
        dist, indices = tree.query(patches_coords[:,::-1],return_distance=True)
        desired_idx = np.where(dist<=5)[0] 
        idx, uniq_desired_idx = np.unique(indices.ravel()[desired_idx],return_index=True)
        feature_vec_new = feature_vec[idx]
        coords_new = coordinates[idx]
        assert np.isclose(desired_idx,desired_idx[uniq_desired_idx]).all()
        density_new = patch_densities[desired_idx[uniq_desired_idx]]
        if (len(np.where(dist>=5)[0])/len(dist)>=0.1) and (len(np.where(dist>=5)[0])>50):
            print("Too many features missing, copying original features and coordinates")
            feature_vec_new = feature_vec
            coords_new = coordinates
            density_new = None
        else:
            print("Extracted {} features, ommitted {} features".format(len(feature_vec_new),len(patches_coords)-len(coords_new)))
    else:
        feature_vec_new = feature_vec
        coords_new = coordinates
        density_new = None
        print("No samples were detected")
    # torch.save({"feat":feature_vec_new,"coord":coords_new,"density":density_new},str(OUTPUT_DIR/f"{slide_name}_featvec.pt"))