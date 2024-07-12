from typing import Optional, Union, List
import warnings

import torch
import torch.nn as nn

import trainer
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base import (
    SegmentationModel,
    SegmentationHead,
)
from segmentation_models_pytorch.decoders.manet.decoder import MAnetDecoder
from segmentation_models_pytorch.base.initialization import initialize_decoder, initialize_head

class MAnet(SegmentationModel, trainer.Model):
    """
    Code adapted from MAnet defined in segmentation_models_pytorch repository
    MAnet_ :  Multi-scale Attention Net. The MA-Net can capture rich contextual dependencies based on
    the attention mechanism, using two blocks:
    - Position-wise Attention Block (PAB), which captures the spatial dependencies between pixels in a global view
    - Multi-scale Fusion Attention Block (MFAB), which  captures the channel dependencies between any feature map by
    multi-scale semantic feature fusion

    Args:
        encoder_name: Name of the classification model that will be used as an encoder (a.k.a backbone)
            to extract features of different spatial resolution
        encoder_depth: A number of stages used in encoder in range [3, 5]. Each stage generate features
            two times smaller in spatial dimensions than previous one (e.g. for depth 0 we will have features
            with shapes [(N, C, H, W),], for depth 1 - [(N, C, H, W), (N, C, H // 2, W // 2)] and so on).
            Default is 5
        encoder_weights: One of **None** (random initialization), **"imagenet"** (pre-training on ImageNet) and
            other pretrained weights (see table with available weights for each encoder_name)
        decoder_channels: List of integers which specify **in_channels** parameter for convolutions used in decoder.
            Length of the list should be the same as **encoder_depth**
        decoder_use_batchnorm: If **True**, BatchNorm2d layer between Conv2D and Activation layers
            is used. If **"inplace"** InplaceABN will be used, allows to decrease memory consumption.
            Available options are **True, False, "inplace"**
        decoder_pab_channels: A number of channels for PAB module in decoder.
            Default is 64.
        in_channels: A number of input channels for the model, default is 3 (RGB images)
        classes: A number of classes for output mask (or you can think as a number of channels of output mask)
        activation: An activation function to apply after the final convolution layer.
            Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**,
                **callable** and **None**.
            Default is **None**
        aux_params: Dictionary with parameters of the auxiliary output (classification head). Auxiliary output is build
            on top of encoder if **aux_params** is not **None** (default). Supported params:
                - classes (int): A number of classes
                - pooling (str): One of "max", "avg". Default is "avg"
                - dropout (float): Dropout factor in [0, 1)
                - activation (str): An activation function to apply "sigmoid"/"softmax"
                    (could be **None** to return logits)

    Returns:
        ``torch.nn.Module``: **MAnet**

    .. _MAnet:
        https://ieeexplore.ieee.org/abstract/document/9201310

    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = None,
        encoder_transfer: Optional[str] = None,
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_pab_channels: int = 64,
        in_channels: int = 3,
        tissue_classes: int = 3,
        cell_classes:int = 1,
        activation: Optional[Union[str, callable]] = None,
        **kwargs
    ):
        SegmentationModel.__init__(self)

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        if (encoder_weights is None) and (encoder_transfer is not None):
            self.load_encoder(path=encoder_transfer)

        self.tissue_decoder = MAnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            pab_channels=decoder_pab_channels,
        )

        self.cell_decoder = MAnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            pab_channels=decoder_pab_channels,
        )

        self.tissue_segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=tissue_classes,
            activation=activation,
            kernel_size=3,
        )

        self.cell_segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=cell_classes,
            activation=activation,
            kernel_size=3,
        )

        # if aux_params is not None:
        #     self.classification_head = ClassificationHead(in_channels=self.encoder.out_channels[-1], **aux_params)
        # else:
        #     self.classification_head = None

        self.name = "manet-{}".format(encoder_name)
        self.initialize()

    def initialize(self):
        initialize_decoder(self.tissue_decoder)
        initialize_decoder(self.cell_decoder)
        initialize_head(self.tissue_segmentation_head)
        initialize_head(self.cell_segmentation_head)
    
    def load_encoder(self, path):
        #Load model weights
        state = torch.load(path,map_location="cpu")
        state_dict = state['model']
        for key in list(state_dict.keys()):
            state_dict[key.replace('model.', '').replace('resnet.', '').replace('module.','')] = state_dict.pop(key)
        model_dict = self.encoder.state_dict()
        weights = {k:v for k, v in state_dict.items() if k in model_dict}
        if len(state_dict.keys()) != len(model_dict.keys()):
            not_loaded = [x for x in model_dict.keys() if not x in list(state_dict.keys())]
            if weights == {}:
                warnings.warn(f"Warning... No weight could be loaded..\n{not_loaded}")
            else:
                warnings.warn(f"Warning... Some Weights could not be loaded\n{not_loaded}")
        else:
            print("All weights successfully loaded")

        model_dict.update(weights)
        self.encoder.load_state_dict(model_dict)

    def forward(self, x, mode:str="all"):
        """Sequentially pass `x` through model`s encoder, decoder and heads"""

        self.check_input_shape(x)
        features = self.encoder(x)
        tissue_masks = None
        cell_masks = None
        if (mode=="tissue") or (mode=="all"):
            tissue_decoder_output = self.tissue_decoder(*features)
            tissue_masks = self.tissue_segmentation_head(tissue_decoder_output)
        if (mode=="cell") or (mode=="all"):
            cell_decoder_output = self.cell_decoder(*features)
            cell_masks = self.cell_segmentation_head(cell_decoder_output)    


        return tissue_masks, cell_masks

@trainer.Model.register("resmanet_multi_512_v2")
class MAnet_512(MAnet):
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = None,
        encoder_transfer: Optional[str] = None,
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_pab_channels: int = 64,
        in_channels: int = 3,
        tissue_classes: int = 3,
        cell_classes:int = 1,
        activation: Optional[Union[str, callable]] = None,
        **kwargs
    ):
        super().__init__(encoder_name,
        encoder_depth,
        encoder_weights,
        encoder_transfer,
        decoder_use_batchnorm,
        decoder_channels,
        decoder_pab_channels,
        in_channels,
        tissue_classes,
        cell_classes,
        activation)

    def forward(self, x, mode:str="all"):
        """Sequentially pass `x` through model`s encoder, decoder and heads"""

        self.check_input_shape(x)
        features = self.encoder(x)
        tissue_masks = None
        cell_masks = None
        #Take the central frame
        if (mode=="tissue") or (mode=="all"):
            tissue_decoder_output = self.tissue_decoder(*features)
            tissue_masks = self.tissue_segmentation_head(tissue_decoder_output)[:,:,128:128+256,128:128+256]
        if (mode=="cell") or (mode=="all"):
            cell_decoder_output = self.cell_decoder(*features)
            cell_masks = self.cell_segmentation_head(cell_decoder_output)[:,:,128:128+256,128:128+256]    


        return tissue_masks, cell_masks