# Ensemble of Prior-guided Expert Graph Models for Survival Prediction in Digital Pathology
Code and data for the manuscript: **Ensemble of Prior-guided Expert Graph Models for Survival Prediction in Digital Pathology**  
Link for OpenReview: [https://papers.miccai.org/miccai-2024/291-Paper2280.html]  
Link for full paper: [https://papers.miccai.org/miccai-2024/paper/2280_paper.pdf]  
Link for supplementary: [https://papers.miccai.org/miccai-2024/supp/2280_supp.pdf]  
## Description

This repository contains the code and data of the paper "Ensemble of Prior-guided Expert Graph Models for Survival Prediction in Digital Pathology", which is accepted at MICCAI 2024 for publication.

There are several limitations associated with survival prediction using graph based algorithms. GNNs operate with homophilly assumption failing to account for the inherent heterogeneity present among different entities within tissue slides. Additionally, the convoluted downstream task relevant information is not effectively exploited by graph based methods, especially when working with large slide graphs.
We propose a novel prior-guided, edge-attributed tissue-graph construction to address these challenges, followed by an ensemble of expert graph-attention survival models. The algorithm improves overall survival prediction against other baselines. Due to the nature of the prior guided graph construction, the algorithm can lead to better exploration of model attended prognostic features.  

## Methodology
The methodlogy consists of multiple steps:-
1. Segmentation Module: Performing segmentation to identify different entities present in the tissue slide
2. WSI graph construction: Using the identified entites, construct different directed subset graphs (tumor-stroma, stroma-others, tumor-others, complete (tumor-stroma-others))
3. Density based GNN (D-GNN): Using different graphs as input, train individual expert models for survival prediction
4. Ensemble of D-GNNs (ED-GNNs): Using simple weighted linear ensemble between different experts for final survival prediction
An overview of the methodology and its results are shown below
<img src="https://github.com/Vishwesh4/DGNN/blob/master/images/fig1.png" align="center" width="880" ><figcaption>Fig.1 - Overall Methodology. Different entities shown in different colors (Tumor(red), Tumor-associated stroma (blue) and others (green))</figcaption></a>

<img src="https://github.com/Vishwesh4/DGNN/blob/master/images/fig2.png" align="center" width="880" ><figcaption>Fig.2 - Visualization of attended regions statistics across multiple patients</figcaption></a> 

<img src="https://github.com/Vishwesh4/DGNN/blob/master/images/heatmap.png" align="center" width="880" ><figcaption>Fig.2 - Visualization of attended region in a tissue slide by Complete-graph D-GNN</figcaption></a> 

## Getting Started

### Dependencies

```
opencv
pytorch-gpu
torchvision
wandb
timm
einops
openslide
scikit-learn
lifelines
dplabtools
nmslib
scipy
scikit-image
scikit-survival
torch-geometric
nystrom-attention
segmentation_models_pytorch
trainer - https://github.com/Vishwesh4/TrainerCode/tree/master
```
### Datasets

The project is applied to 4 public datasets from TCGA (BRCA, STAD, UCEC, COADREAD) downloaded from [link](https://portal.gdc.cancer.gov/)


### Implementation
In order to apply our model on tissue slides, the following steps need to be followed in order.
#### Feature Maps
First, feature embeddings of individual patches are to be extracted from tissue slides. One can perform this step by running ```python ./utils/TCGA_extract_features.py --onco_code [str: BRCA/COADREAD/STAD/UCEC]```. For our experiments we used [CtransPath](https://github.com/Xiyue-Wang/TransPath), however other pretrained feature extractors can also be used. Note that we follow file naming convention based on the TCGA slide naming convention. Please follow a consistent naming convention in all the scripts. Extracted features will be saved in ```./dataset/TCGA_processed/features```.
#### Segmentation Module
Tissue and TILs segmentation are to be performed next which will be used in building graphs in the subsequent steps. The weights for the segmentation module can be downloaded [here](https://drive.google.com/drive/folders/1pfhOttn4JRxIyG422Ejbvz2-67LIDDPo?usp=sharing) and should be saved in location```./dgnn/utils_graph/segmentation_model_weights```. For performing segmentation and performing patch sampling based on the segmentation prediction, please run ```python ./dgnn/deploy_get_samples.py```. Segmentation predictions will be saved as density maps in ```./dataset/TCGA_processed/density_maps``` and sampled patches(in paper we don't sample patches) and their features will be saved in ```./dataset/TCGA_processed/patch_samples```.
#### Graph Construction
Before running the model on the dataset, complete graphs are to be constructed. This can be done using ```./dgnn/build_graph_complete.py```. The scripts constructs complete graphs(tumor-stroma-others) based on sampled patches. Subset graphs can be constructed using the complete graphs and density maps using ```python ./dgnn/build_graph_tissueinteraction.py --onco_code [str: BRCA/COADREAD/STAD/UCEC] --graph [str: tumor_stroma/tumor_rest/stroma_rest]```. All the graphs will be saved in ```./dataset/TCGA_processed/graphs```.
#### Model Training
The model can be trained using ```main.py```. Examples are given in ```scripts/run_interactionmodels.sh```. For baselines, examples are given in ```scripts/run_baselines.sh```. Once the individual expert graphs are trained, the weights can be used in ED-GNN (linear ensemble) using ```ensemble_interaction_eval.py```


## Acknowledgements
We would like to express our gratitude to the creators of Patch-GCN [link](https://github.com/mahmoodlab/Patch-GCN) repository, from which most of the training code has been adapted. We would also like to thank the authors of CtransPath and the authors of various baseline models for making their code public and easy to use.
## Contact
You can reach the authors by raising an issue in this repo or
 email them at vishwesh.ramanathan@mail.utoronto.ca/a.martel@utoronto.ca

## Cite
```

```
To be updated with new citations after camera ready submission and publication.
