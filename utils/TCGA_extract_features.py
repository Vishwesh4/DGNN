'''
Script for generating bag of feature vectors per slide for MIL training on TCGA dataset
'''
import sys
import os
from collections import OrderedDict
from pathlib import Path
sys.path.append("/aippmdata/trained_models/Martel_lab/pathology/SSL_CTransPath/")
import argparse

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

from extract_patches import ExtractPatches
# Pretrained model from https://github.com/Xiyue-Wang/TransPath
from get_features_CTransPath import model, trnsfrms_val

DEVICE = torch.device("cuda:0")

model_fv = model.to(DEVICE)
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
TRANSFORM_FV = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean = mean, std = std)
    ]
)

#############################################################################################################################################################

TILE_SIZE=224
TILE_STRIDE_SIZE=1

parser = argparse.ArgumentParser(description='Configurations for extraction of ')
parser.add_argument('--onco_code', type=str, default='BRCA', help='type code of TCGA dataset in all caps')
args = parser.parse_args()
ONCO_CODE = args.onco_code


INPUT_DIR = list(Path(f"/aippmdata/public/TCGA/TCGA-{ONCO_CODE}/images/").rglob("*.svs"))
OUTPUT_DIR = Path(f"/localdisk3/ramanav/TCGA_processed/TCGA_MIL_Patches_Ctrans_1MPP_{ONCO_CODE}")
Path.mkdir(OUTPUT_DIR,parents=False,exist_ok=True)
processed_files = [files.stem.split("_")[0] for files in OUTPUT_DIR.glob("*.pt")]

for paths in INPUT_DIR:  
    slide_name = paths.stem.split(".")[0]
    print(f"Processing {slide_name}...")
    if slide_name in processed_files:
        print("Already processed...")
        continue
    try:
        patch_dataset = ExtractPatches(wsi_file = str(paths),
                                        patch_size = TILE_SIZE,
                                        level_or_mpp=1.0,
                                        foreground_threshold = 0.95,
                                        patch_stride = TILE_STRIDE_SIZE,
                                        mask_threshold = 0.1,
                                        mask_kernelsize = 9,
                                        num_workers = 4,
                                        save_preview=False,
                                        save_mask=False,
                                        transform=TRANSFORM_FV)
    except:
        print("Might be empty slide file")
        continue

    dataloader = torch.utils.data.DataLoader(patch_dataset, batch_size=512, num_workers=4)
    all_feats = []
    with torch.no_grad():
        for data in tqdm(dataloader,desc="Extracting and saving feature vectors"):
            img = data[0].to(DEVICE)
            feats = model_fv(img)
            all_feats.extend(feats.cpu())
        all_feats = torch.stack(all_feats,dim=0)
    print("Extracted {} features".format(len(all_feats)))
    torch.save(all_feats,str(OUTPUT_DIR/f"{slide_name}_featvec.pt"))