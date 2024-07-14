"""
Module for loading and performing tissue and TIL segmentation at 0.5MPP
"""
import os
import warnings
from pathlib import Path

import openslide
from tqdm import tqdm
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import torch
import torchvision
import torch.nn as nn
from time import time

import trainer
from .helper_functions.segmentation_writer import SegmentationWriter
from .helper_functions.dataset_wrapper import TILData
from .helper_functions.postprocess import postprocess_tumorbed
from .helper_functions.segmentation_model import *

class TIL_Segmentation_Score:
    TILE_WRITE_SIZE=256
    BATCH_SIZE=128
    def __init__(self,
                 til_model_path: str,
                 tumor_model_path: str,
                 transform:torchvision.transforms,
                 spacing:float=0.5,
                 threshold_tumor:float=0.5,
                 threshold_cell:float=0.3,
                 device=torch.device("cpu"),
                 force_compute=False,
                 plot_maps=False)->None:
        
        self.tilpath = til_model_path
        self.tumpath = tumor_model_path
        self.device = device
        self.transform = transform
        self.spacing = spacing
        self.threshold_tumor = threshold_tumor
        self.threshold_cell = threshold_cell
        self.plot_maps = plot_maps
        self.force_compute = force_compute
        
        try:
            self._load_til_model()
            self._load_tumor_model()
        except Exception as e:
            print(f"Model not loaded because of {e}")
            pass

        self.reset()

    def set_spacing(self,spacing:float):
        self.spacing = spacing
    
    def get_dataset(self,patches,coordinates,template):
        self.patches = patches
        self.coordinates = coordinates
        self.template = template
    
    def compute_tumorbed(self):
        """
        Computes tumorbed based on classification model. Postprocessing is also performed
        """
        self.start = time()
        tumor_results = []
        tum_dataset = TILData(patches=self.patches,coordinates=self.coordinates,transform=self.transform,mode="tumorbed")
        tum_dataloader = torch.utils.data.DataLoader(tum_dataset, batch_size=self.BATCH_SIZE,shuffle=False,num_workers=4)
        with torch.no_grad():
            for data in tqdm(tum_dataloader,desc="Getting the tumorbed"):
                images = data[0].to(self.device)
                is_relevant = self.tumourNet(images)[:, -1]
                tumor_results.extend(is_relevant.cpu().numpy())
            tumor_results = np.stack(tumor_results,axis=0)
        self.processed_tumorbed = postprocess_tumorbed(self.template,self.spacing,tumor_results)
        self.tumorbed_idx = np.where(self.processed_tumorbed>= self.threshold_tumor)[0]
    
    def compute_til(self):
        """
        Performs tissue and TIL segmentation
        """
        if len(self.tumorbed_idx)==0:
            print("No tumorbed found")
            self.tilscore = 0
            self.tilarea=0
            self.tissuearea=0
            self.end = time()
            return None
        if self.force_compute:
            #Compute at all locations
            patches_temp = self.patches
            coordinates_temp = self.coordinates
        else:
            #Only performs at tumorbed
            patches_temp = [self.patches[idx] for idx in self.tumorbed_idx]
            coordinates_temp = [self.coordinates[idx] for idx in self.tumorbed_idx]
        til_dataset = TILData(patches=patches_temp,coordinates=coordinates_temp,transform=self.transform,mode="til")
        til_dataloader = torch.utils.data.DataLoader(til_dataset, batch_size=self.BATCH_SIZE,shuffle=False,num_workers=4)
        til_outputs = []
        tilcoordinates = []
        with torch.no_grad():
            for data in tqdm(til_dataloader,desc="Evaluating TILS"):
                img = data[0].to(self.device)
                tilcoordinates.extend(torch.stack(data[1],dim=1).numpy())
                pred_tissue, pred_mask = self.tilmodel(img,mode="all")
                output_mask = np.asarray((torch.sigmoid(pred_mask)[:,0,:,:]>=self.threshold_cell).cpu().numpy(),np.uint8)
                output_tissue = np.asarray(torch.argmax(pred_tissue,dim=1).cpu().numpy(),np.uint8)+1
                til_outputs.extend(np.stack((output_mask,output_tissue),axis=1))
        self.til_outputs = np.stack(til_outputs)
        self.tilcoordinates = np.stack(tilcoordinates)
        self._calc_tilscore()
        self.end = time()

    def log_results(self,slide_path=None,output_loc=None):
        """
        Write cell and tissue masks
        """
        self._read_slide(slide_path)
        if self.plot_maps:
            start = time()
            output_loc = output_loc / self.wsiname
            if not output_loc.exists():
                os.mkdir(output_loc)
            print(f"Setting up writers")
            self.tissue_segmentation_writer = SegmentationWriter(
                Path(output_loc) / f"{self.wsiname}_tissuesegmentation.tif",
                tile_size=self.TILE_WRITE_SIZE,
                dimensions=self.dimensions,
                spacing=(self.spacing,self.spacing),
                software="sedeen",
                colormap={0:np.array([255,255,255]),1:np.array([255,0,0]),2:np.array([0,255,255]),3:np.array([0,255,0])}
            )
            self.cell_segmentation_writer = SegmentationWriter(
                Path(output_loc) / f"{self.wsiname}_cellsegmentation.tif",
                tile_size=self.TILE_WRITE_SIZE,
                dimensions=self.dimensions,
                spacing=(self.spacing,self.spacing),
                software="sedeen",
                colormap={0:np.array([255,255,255]),1:np.array([255,0,0])}
            )

            for idx,coords in enumerate(self.tilcoordinates):
                x_write,y_write = coords
                tissue = self.til_outputs[idx,1,:,:]
                cell = self.til_outputs[idx,0,:,:]
                if self.space_flag:
                    x_write = x_write/2
                    y_write = y_write/2
                self.tissue_segmentation_writer.write_segmentation(tile=tissue, x=x_write, y=y_write)
                self.cell_segmentation_writer.write_segmentation(tile=cell, x=x_write, y=y_write)
            self.tissue_segmentation_writer.save()
            self.cell_segmentation_writer.save()
            
            #Write tumorbedmap
            template = self.template.astype(np.float64)
            template[template==0] = np.nan
            fill_tumour = template.flatten().copy()
            fill_tumour[np.where(fill_tumour >= 1)[0]] = self.processed_tumorbed
            tumour_heatmap = np.reshape(fill_tumour, np.shape(template))

            cmap = matplotlib.cm.jet
            cmap.set_bad('white',1.)
            plt.figure()
            plt.imshow(tumour_heatmap, interpolation="nearest")
            plt.axis('off')
            plt.savefig(str(Path(output_loc) / f"{self.wsiname}_tumorbedviewer.png"))

            end = time()
            print(f"Total writing time for {len(self.tilcoordinates)} patches: {end-start}")
        # Save TIL score
        with open(str(Path(output_loc) / f"{self.wsiname}_TIL_Score.txt"), "w") as file:
            file.write(
                "Tumorbed Model Path: {}\nTIL Model Path: {}\nCalculated TIL Score: {}\nCalculated TIL Area Density: {}\nCalculated Tissue Area Density: {}\nTime Taken: {}\n".format(
                    self.tumpath, self.tilpath, self.tilscore, self.tilarea/(256*256), self.tissuearea/(256*256), self.end-self.start
                )
            )
            if self.plot_maps:
                file.write(f"Total writing time for {len(self.tilcoordinates)} patches: {end-start}")
    
    def _calc_tilscore(self):
        tumour_associated_stroma = np.asarray((self.til_outputs[:,1]==2),np.uint8)
        tils_in_stroma = tumour_associated_stroma*self.til_outputs[:,0]
        #The constant is ratio between 18*18 and circle of radius 6 which is the label, it is 2.86 but rounding it off
        self.tilarea = np.sum(tils_in_stroma)*3
        self.tissuearea = np.sum(tumour_associated_stroma)
        # 1.5mm2 is the allowed tumorbed area, roughly that amounts to 9x9 patch area, hence 
        # the 91*0.25(min til area) ~ 20-5 ~ 15. We need minimum of this sum. That amounts to 15/91 ~ 0.24mm2 which is pretty small
        if self.tissuearea<=15*256*256:
            print("Very small tissue area(0.24mm2), hence evaluating til score as 0")
            calc = 0
        else:
            calc = 100 * self.tilarea / (self.tissuearea + 0.00001)
        calc = np.clip(calc,a_min=0,a_max=100)
        self.tilscore = calc

    def _read_slide(self,path):
        #Slide properties
        self.space_flag = 0 #incase of forgetting to reset
        self.slide = openslide.OpenSlide(path)
        self.dimensions = self.slide.dimensions
        if "tiff.XResolution" in self.slide.properties.keys():
            self.spacing = 1 / (float(self.slide.properties["tiff.XResolution"]) / 10000)
        elif "openslide.mpp-x" in self.slide.properties.keys():
            self.spacing = np.asarray((self.slide.properties["openslide.mpp-x"]),dtype=np.float32)
        else:
            print("using default spacing value")
            self.spacing = 0.25
            # raise ValueError("Not able to find spacing")
        print(f"Slide spacing: {self.spacing}")
        if self.spacing<0.3:
            #For dealing with 40x
            self.spacing = 2*self.spacing
            self.dimensions = (self.dimensions[0]//2, self.dimensions[1]//2)
            self.space_flag = 1
        self.wsiname = Path(path).stem.split(".")[0]
    
    def _load_til_model(self):
        self.tilmodel = trainer.Model.create("resmanet_multi_512_v2",
                                            encoder_name= "resnet34",
                                            encoder_depth= 5,
                                            encoder_weights=None,
                                            decoder_use_batchnorm= True,
                                            decoder_channels= [256, 128, 64, 32, 16],
                                            decoder_pab_channels= 64,
                                            in_channels= 3,
                                            tissue_classes= 3,
                                            cell_classes= 1)
        self.tilmodel.load_model_weights(model_path=self.tilpath,device=self.device)
        self.tilmodel.eval()
        self.tilmodel.to(self.device)

    def _load_tumor_model(self):
        """
        Loads model weight
        """
        # Tumour model
        self.tumourNet = torchvision.models.__dict__["resnet18"](pretrained=False)
        self.tumourNet.fc = nn.Sequential(
            torch.nn.Linear(self.tumourNet.fc.in_features, 300),
            torch.nn.Linear(300, 2),
            torch.nn.Softmax(-1),
        )
        state = torch.load(self.tumpath, map_location=self.device)
        state_dict = state["model_state_dict"]
        for key in list(state_dict.keys()):
            state_dict[
                key.replace("resnet.", "").replace("module.", "")
            ] = state_dict.pop(key)
        model_dict = self.tumourNet.state_dict()
        weights = {k: v for k, v in state_dict.items() if k in model_dict}
        if len(state_dict.keys()) != len(model_dict.keys()):
            warnings.warn("Warning... Some Weights could not be loaded")
        if weights == {}:
            warnings.warn("Warning... No weight could be loaded..")
        model_dict.update(weights)
        self.tumourNet.load_state_dict(model_dict)
        self.tumourNet.eval()
        self.tumourNet.to(self.device)

    def reset(self):
        self.space_flag = 0
        self.tilcoordinates = []
        self.tilcoordinate = []
        self.processed_tumorbed = []
        self.count = 0
        self.tumorbed_idx = []
        self.til_outputs = []
        try:
            del self.spacing
        except:
            pass