"""
Module for loading/calculating the segmentation maps and sampling representative coordinates from different regions of tissues. Converts segmentation
at 0.5MPP to density values at 1.0MPP where survival prediction is performed
By default assumed no sampling
"""
import os
from pathlib import Path
import random

import cv2
import openslide
from matplotlib import colors
from PIL import ImageDraw
import numpy as np
import torch
import torchvision
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import KDTree

from dplabtools.slides.processing import WSITissueMask
from dplabtools.slides.patches import WholeImageGridPatches

from .segmentation_module import TIL_Segmentation_Score

class Get_Samples(TIL_Segmentation_Score):
    def __init__(self,
                 peritumor_boundary:int=18000,
                 graph_node_resolution:float=1.0, #1MPP
                 graph_patch:int=224,
                 num_stroma:int=1500,
                 init_stroma_prob:tuple=(0.6,1,1),
                 init_rest_prob:tuple=(0.1,0.3,0.7),
                 nontils_prob:tuple=(0.3,0.6,0.95),
                 percentile_tils:int=50,
                 seed:int=2023,
                 **kwargs):
        """
        Class to perform sampling of patches for downstream risk modelling 
        Parameters:
            peritumor_boundary: Dense sampling performed within this region, distance given at 40x
            graph_node_resolution: Resolution at which graph processing will happen
            graph_patch: Size of patches at which graph processing will happen
            num_stroma: Number of stromal patches beyond which you perform more rigourous sampling
            init_stroma_prob: (outside_peritumor prob, inside_peritumor_min prob, inside_peritumor_max prob) for stroma
            init_rest_prob: (outside_peritumor prob, inside_peritumor_min prob, inside_peritumor_max prob) for rest
            nontils_prob: (outside_peritumor prob, inside_peritumor_min prob, inside_peritumor_max prob) for non til(less than percentile_tils) stroma
            percentile_tils: Gets triggered when stromal patches are greater than num_stroma. Percentile tils above which the stromal patches get priority sampling
        """
        self.peritumor_boundary_orig = peritumor_boundary
        self.graph_node_resolution = graph_node_resolution
        self.graph_patch = graph_patch
        self.num_stroma = num_stroma
        self.init_stroma_prob = init_stroma_prob
        self.init_rest_prob = init_rest_prob
        self.nontils_prob = nontils_prob
        self.perc_tils = percentile_tils
        self.seed = seed

        #Set seed for reproducibility
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        super().__init__(**kwargs)
    
    def construct_density(self, patches, coordinates, template, scan)->None:
        """
        Performs computation of segmentation maps and finds the densities of various tissue and cell classes
        Returns:
            heatmap where each pixel is a patch at self.graph_node_resolution. Contains densities of all the classes and
            of shape [h,w,4]
        """
        # self.downsample_ratio = np.floor(self.graph_node_resolution/self.spacing)
        self.slide = scan
        self.get_dataset(patches,coordinates,template)
        #Get tumorbed
        self.compute_tumorbed()
        #Get computation of pixel wise til and tissue classification at 0.5 MPP
        self.compute_til()
        self.template_shape = self.template.shape

        if len(self.til_outputs)==0:
            return np.array([]),self._get_heatmap(self.processed_tumorbed,True)
        ## Get highlevel patchinformation by aggregation
        tumor_associated_stroma = np.asarray((self.til_outputs[:,1]==2),np.uint8)
        #get density by normalize with different values 
        tumor_density = np.sum(np.asarray((self.til_outputs[:,1]==1),np.uint8), axis=(1,2))/(self.TILE_WRITE_SIZE*self.TILE_WRITE_SIZE)
        rest_density = np.sum(np.asarray((self.til_outputs[:,1]==3),np.uint8), axis=(1,2))/(self.TILE_WRITE_SIZE*self.TILE_WRITE_SIZE)
        stroma_density = np.sum(tumor_associated_stroma, axis=(1,2))/(self.TILE_WRITE_SIZE*self.TILE_WRITE_SIZE)
        stils_density =  np.sum(tumor_associated_stroma*self.til_outputs[:,0],axis=(1,2))/(0.25*self.TILE_WRITE_SIZE*self.TILE_WRITE_SIZE)
        #get heatmaps
        tumorbed_heatmap = self._get_heatmap(self.processed_tumorbed,True)
        tumor_heatmap = self._get_heatmap(tumor_density,self.force_compute)
        stroma_heatmap = self._get_heatmap(stroma_density,self.force_compute)
        rest_heatmap = self._get_heatmap(rest_density,self.force_compute)
        stils_heatmap = self._get_heatmap(stils_density,self.force_compute)
        all_densities = np.stack((tumor_heatmap,stroma_heatmap,rest_heatmap,stils_heatmap),axis=2)
        return all_densities, tumorbed_heatmap

    def load_density(self, density_path, slide_path, all_portions=False):
        with open(density_path,"rb") as f:
            data = np.load(density_path,allow_pickle=True)
        all_densities = self.prepare_density(data["all_densities"],
                                             data["tumorbedmap"],
                                             slide_path,
                                             all_portions)
        return all_densities
    
    def prepare_density(self, all_densities, tumorbedmap, slide_path, all_portions):
        if not all_portions:
            all_densities = all_densities*tumorbedmap[:,:,np.newaxis]
        else:
            all_densities = all_densities
        self.template_shape = all_densities.shape[:-1]
        self._read_slide(slide_path)
        self.slide_path = slide_path
        self.orig_coords, self.bb_cords = self._get_coordinates(mpp=self.graph_node_resolution,patch_size=self.graph_patch)
        if self.space_flag==0:
            #indicates the 0 level is already at 0.5mpp, multiply with 2 to compensate
            #This is done to accomodate for change in mpp used for segmentation and risk model
            self.spacing = 2*self.spacing
        return all_densities

    def sample_regions(self, all_densities, no_sampling = True):
        """
        Tumor regions along with stroma and high tils regions, which we will use for building graphs
        Also sample randomly regions from negative region
        Parameters:
            all_densities: Calculated values of per patch tumor/stroma/rest/tils density
            no_sampling: If True, does not perform any sampling and simply returns patches found in the tumorbed region
        """
        if self.space_flag==0:
            #This would matter for case of sampling
            factor = 2 #slide at 20x
        else:
            factor = 1 #slide at 40x
        all_densities = self._downsample_density(all_densities)

        invasive = np.array((np.where(all_densities[:,:,0]>=0.33))).T
        stroma = np.array((np.where(all_densities[:,:,1]>=0.33))).T
        rest_area = np.array((np.where(all_densities[:,:,2]>=0.33))).T
             
        if len(stroma)==0:
            return [],[],[],[],[]
        
        if no_sampling:
            patches = np.unique(np.concatenate((invasive,stroma,rest_area),axis=0),axis=0)
            invasive_coords = self._get_coords_from_template(invasive)
            stroma_coords = self._get_coords_from_template(stroma)
            rest_coords = self._get_coords_from_template(rest_area)
            patches_coords = self._get_coords_from_template(patches)
            patch_densities = all_densities[patches[:,0],patches[:,1],:]
            return invasive_coords, stroma_coords, rest_coords, patches_coords, patch_densities
        
        self.peritumor_boundary = self.peritumor_boundary_orig/factor
        patch_size = (2*self.graph_node_resolution/self.spacing)*self.graph_patch
        distance_threshold = self.peritumor_boundary/patch_size #at 10x, chosen by intuition
        
        if len(invasive)>=5:
            distance_matrix_stroma = pairwise_distances(X=stroma,Y=invasive)
            #Get sampling prob of stroma 
            stroma_dist = distance_matrix_stroma[np.arange(len(stroma)),np.argmin(distance_matrix_stroma,axis=1)]
            sampling_probs = self._sampling_prob(stroma_dist, distance_threshold, self.init_stroma_prob)
            #sample stroma points based on independent coin tosses
            selection_stroma = sampling_probs >= np.random.rand(len(stroma))
            sampled_stroma = stroma[selection_stroma]
            #Rest area sampling
            if len(rest_area)>0:
                distance_matrix = pairwise_distances(X=rest_area,Y=invasive)
                #Get sampling prob of rest tissue 
                rest_dist = distance_matrix[np.arange(len(rest_area)),np.argmin(distance_matrix,axis=1)]
                sampling_probs = self._sampling_prob(rest_dist, distance_threshold, self.init_rest_prob)
                #sample stroma points based on independent coin tosses
                selection = sampling_probs >= np.random.rand(len(rest_area))
                sampled_rest = rest_area[selection]
            else:
                sampled_rest = rest_area
        else:
            sampled_stroma = stroma
            sampled_rest = rest_area
        if (len(sampled_stroma)>=self.num_stroma) and (len(invasive)>=5):
            # Keep samples with tils, beyond that flip a coin to decide to keep samples or not beyond peritumoral region
            perc_threshold = np.percentile(all_densities[sampled_stroma[:,0],sampled_stroma[:,1],3].flatten()[(all_densities[sampled_stroma[:,0],sampled_stroma[:,1],3].flatten()>0).nonzero()[0]],self.perc_tils)
            stils_coords = all_densities[sampled_stroma[:,0],sampled_stroma[:,1],3]>=perc_threshold
            stils_samples = sampled_stroma[stils_coords]
            less_stils = sampled_stroma[~stils_coords]
            idx_dist = selection_stroma.nonzero()[0][~stils_coords]
            dist_less = distance_matrix_stroma[idx_dist,np.argmin(distance_matrix_stroma[idx_dist],axis=1)]
            sampling_probs = self._sampling_prob(dist_less,distance_threshold,self.nontils_prob)
            selection_stroma = sampling_probs >= np.random.rand(len(less_stils))
            sampled_less = less_stils[selection_stroma]
            sampled_stroma = np.concatenate((stils_samples,sampled_less),axis=0)
        patches = np.unique(np.concatenate((invasive,sampled_stroma,sampled_rest),axis=0),axis=0)
        invasive_coords = self._get_coords_from_template(invasive)
        stroma_coords = self._get_coords_from_template(sampled_stroma)
        rest_coords = self._get_coords_from_template(sampled_rest)
        patches_coords = self._get_coords_from_template(patches)
        patch_densities = all_densities[patches[:,0],patches[:,1],:]
        return invasive_coords, stroma_coords, rest_coords, patches_coords, patch_densities        
    
    @staticmethod
    def _sampling_prob(X:np.array, distance_threshold:float, sample_probs:tuple)->np.array:
        """
        Map distance to sampling probability weightage
        """
        outside_prob, inside_prob_min, inside_prob_max = sample_probs
        thresh_status = (X<=distance_threshold)*1.0
        prob = thresh_status*(inside_prob_min + (1-X/distance_threshold)*(inside_prob_max-inside_prob_min)) + (1-thresh_status)*outside_prob
        return prob
        
    def visualize_patches(self, coordinates, output_loc, color):
        count = 0
        tree = KDTree(self.orig_coords)
        assert isinstance(color,list)
        color_val = [tuple(np.asarray((np.array(colors.to_rgba(c=color_ind,alpha=60/255))*255),int)) for color_ind in color]
        thumbnail = self.slide.get_thumbnail((2000,2000))
        ih,iw = self.slide.dimensions
        th,tw = thumbnail.size
        thumbnail_draw = ImageDraw.Draw(thumbnail,"RGBA")
        patch_size = (2*self.graph_node_resolution/self.spacing)*self.graph_patch
        for class_color, coords_class in enumerate(coordinates):
            dist, indices = tree.query(coords_class[:,::-1],return_distance=True)
            coords_class = coords_class[np.where(dist<=5)[0]]
            count+=(len(dist) - len(coords_class))
            for coords in coords_class:
                thumb_coords = coords * np.array([tw/iw, th/ih])
                box_dim = np.floor(np.array([(tw/iw)*patch_size, (th/ih)*patch_size])) #224*4 at 0.25 is 224 at 1.0
                thumbnail_draw.rectangle([(thumb_coords[1],thumb_coords[0]),(thumb_coords[1]+box_dim[1],thumb_coords[0]+box_dim[0])], fill=color_val[class_color])
                thumbnail_draw.rectangle([(thumb_coords[1],thumb_coords[0]),(thumb_coords[1]+box_dim[1],thumb_coords[0]+box_dim[0])],outline=(0,0,0,60))
        thumbnail.save(output_loc)
        print(f"Ommited {count} patches due to distance")
    
    def _get_heatmap(self, heatmap_array, force_compute=False):
        tumor_template = self.template.copy()
        #assuming heatmap array calculation is only on tumor found regions
        if not force_compute:
            tumorbed_idx = np.where(self.processed_tumorbed>0)[0]
        else:
            #all coordinates
            tumorbed_idx = np.where(self.processed_tumorbed>=0)[0]
        #Form template
        fill_tumour = tumor_template.flatten().copy()
        proc_temp = np.zeros_like(self.processed_tumorbed, dtype=np.float32)
        proc_temp[tumorbed_idx] = heatmap_array
        fill_tumour[np.where(fill_tumour >= 1)[0]] = proc_temp
        heatmap = np.reshape(fill_tumour, np.shape(tumor_template))
        return heatmap

    def _get_array(self,heatmap):
        tumorbed_idx = np.where(self.processed_tumorbed>0)[0]
        orig_template = self.template.copy()
        orig_fill = np.where(orig_template.flatten() >= 1)[0]
        all_vals = heatmap.flatten().copy()
        heatmap_array = all_vals[orig_fill[tumorbed_idx]]
        return heatmap_array

    def _downsample_density(self, all_densities):
        #Downsample to obtain at desired sampling
        if self.space_flag==0:
            factor = 1 #slide at 20x
        else:
            factor = 2 #slide at 40x
        downsample = (256*factor)/((2*self.graph_node_resolution/self.spacing)*self.graph_patch)
        tumor_down = cv2.resize(all_densities[:,:,0], None, fx=downsample, fy=downsample, interpolation=cv2.INTER_AREA)
        stroma_down = cv2.resize(all_densities[:,:,1], None, fx=downsample, fy=downsample, interpolation=cv2.INTER_AREA)
        rest_down = cv2.resize(all_densities[:,:,2], None, fx=downsample, fy=downsample, interpolation=cv2.INTER_AREA)
        stils_down = cv2.resize(all_densities[:,:,3], None, fx=downsample, fy=downsample, interpolation=cv2.INTER_AREA)
        # return np.stack((tumor_down,stroma_down,rest_down,stils_down),axis=2)
        all_densities = np.stack((tumor_down,stroma_down,rest_down,stils_down),axis=2)
        return all_densities

    def _get_coords_from_template(self, template_coords):
        # Convert template point coordinates to avoid floating point errors
        patch_size = (2*self.graph_node_resolution/self.spacing)*self.graph_patch
        offset = (self.bb_cords%patch_size)[::-1]
        slide_coords = np.floor(template_coords*patch_size + offset)
        return slide_coords

    def _get_coordinates(self, mpp, patch_size):
        scan = openslide.OpenSlide(self.slide_path)
        size = 1000
        level_dimensions = scan.level_dimensions
        for i,levels in enumerate(level_dimensions[::-1]):
            #Select level with dimension around 1000 otherwise it becomes too big
            if levels[0]>size or levels[1]>size:
                break
        level = len(level_dimensions)-1-i        
        mask = WSITissueMask(wsi_file=self.slide_path,
                                  level_or_minsize=level,
                                  mode="lab",
                                  color_threshold=0.1,
                                  close_fill_kernel_size=9,
                                  remove_small_holes_ratio=0,
                                  remove_small_objects_ratio=0,
                                  remove_all_holes=True)

        grid_patches = WholeImageGridPatches(wsi_file = self.slide_path,
                                        mask_data = mask.array,
                                        patch_size = patch_size,
                                        level_or_mpp = mpp,
                                        foreground_ratio=0.95,
                                        patch_stride = 1,
                                        overlap_ratio = 1.0,
                                        weak_label = "label")
        coordinates = np.array([coords[0] for coords in grid_patches.patch_data])
        bb_coords = np.array(grid_patches._bounding_boxes).ravel()[:2]*(np.array(level_dimensions[0]) / np.array(level_dimensions[level]))
        return coordinates, bb_coords
    
    @staticmethod
    def _remove_close_patches(input_array,radius=1):
        """
        Filter out adjacent patches by finding nearest neighbours using KD tree
        """
        tree = KDTree(input_array)
        throw_points = []
        for i,points in enumerate(input_array):
            if i not in throw_points:
                ind =  tree.query_radius(np.reshape(points,(1,-1)),radius)
                ind = list(ind.item())
                ind.remove(i)
                throw_points.extend(ind)
        throw_points = np.unique(np.array(throw_points))
        temp_array = np.ones(len(input_array),dtype=bool)
        temp_array[throw_points] = False
        return input_array[temp_array]
    
    def reset(self):
        super().reset()
        #Rest seed for reproducibility
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        try:
            del self.spacing
        except:
            pass

            
if __name__=="__main__":
    INPUT_FILE = list(Path("../../dataset/TCGA/TCGA-BRCA/images/").rglob("*.svs"))
    SAVED_LOC = Path("../../dataset/TCGA_processed/density_maps")
    OUTPUT_LOC = Path("../../results")
    processed_files = [files.stem.split("_")[0] for files in OUTPUT_LOC.glob("*.txt")]
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    TIL_PATH="./segmentation_model_weights/segmentation_model_weights.pt"
    TUM_PATH = "./segmentation_model_weights/tumorbed_network_weights.pt"

    #example slides
    slides = [
        "TCGA-AO-A03P-01Z-00-DX1",
        "TCGA-A2-A0D4-01Z-00-DX1",
        "TCGA-A2-A04T-01Z-00-DX1",
        "TCGA-A1-A0SK-01Z-00-DX1",
        "TCGA-A1-A0SH-01Z-00-DX1",
        "TCGA-S3-AA15-01Z-00-DX1",
    ]
    
    til_scorer =  Get_Samples(til_model_path=TIL_PATH,
                                tumor_model_path=TUM_PATH,
                                transform=torchvision.transforms.ToTensor(),
                                threshold_tumor=0.5,
                                threshold_cell=0.5,
                                device=DEVICE,
                                plot_maps=False,
                                peritumor_boundary=8000,
                                graph_node_resolution=1.0, #1MPP
                                graph_patch=224,
                                num_stroma=12500,
                                init_stroma_prob=(1,1,1),
                                init_rest_prob=(0.25,0.25,0.95),
                                nontils_prob=(1,1,1),
                                percentile_tils=30,
                            )

    for keyword in slides:
        input_wsis =  next((x for x in INPUT_FILE if keyword in x.stem), None)
        output_loc = OUTPUT_LOC / Path(input_wsis).stem.split(".")[0]
        print(f"Processing {input_wsis.stem}")
        if not output_loc.is_dir():
            os.mkdir(output_loc)
        til_scorer.reset()
        all_densities = til_scorer.load_density(density_path=str(SAVED_LOC/f"{input_wsis.stem.split('.')[0]}.npy"),
                                slide_path=str(input_wsis)
                                )
        invasive_coords, stroma_coords, rest_coords, patches_coords, _ = til_scorer.sample_regions(all_densities,no_sampling=False)
        til_scorer.visualize_patches([invasive_coords, stroma_coords, rest_coords],output_loc/"all.png",color=["red","blue","green"])
