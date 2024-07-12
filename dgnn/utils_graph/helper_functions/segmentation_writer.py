"""
Module for saving segmentation maps on the whole slide level
"""
import json
from pathlib import Path
from typing import List, Protocol, Union
import numpy as np
from matplotlib import pyplot as plt

READING_LEVEL = 0
WRITING_TILE_SIZE = 256


class SegmentationWriter:
    def __init__(
        self, output_path: Path, tile_size: int, dimensions: tuple, spacing: tuple, software="asap", colormap=None
    ):
        """Writer for writing and saving multiresolution mask/prediction images     
        Args:
            output_path (Path): path to output file 
            tile_size (int): tile size used for writing image tiles
            dimensions (tuple): dimensions of the output image
            spacing (tuple): base spacing of the output image
        """
        #external package, need to install if want to output segmentation maps in tiff format
        #can be installed from https://github.com/computationalpathologygroup/ASAP
        import multiresolutionimageinterface as mir

        if output_path.suffix != '.tif':
            output_path = output_path / '.tif' 

        self._writer = mir.MultiResolutionImageWriter()
        self._writer.openFile(str(output_path))
        self._writer.setTileSize(tile_size)
        self._writer.setCompression(mir.Compression_LZW)
        self._writer.setDataType(mir.DataType_UChar)
        self._writer.setInterpolation(mir.Interpolation_NearestNeighbor)
        if software=="sedeen":
            self._writer.setColorType(mir.ColorType_RGB)
        else:
            self._writer.setColorType(mir.ColorType_Monochrome)
        self._writer.writeImageInformation(dimensions[0], dimensions[1])
        pixel_size_vec = mir.vector_double()
        pixel_size_vec.push_back(spacing[0])
        pixel_size_vec.push_back(spacing[1])
        self._writer.setSpacing(pixel_size_vec)
        self.software = software
        # self.norm = plt.Normalize()
        self.colormap = colormap

    def write_segmentation(
        self, tile: np.ndarray, x: Union[int, float], y: Union[int, float]
    ):
        if tile.shape[0] != WRITING_TILE_SIZE or tile.shape[1] != WRITING_TILE_SIZE:
            raise ValueError(f"Dimensions of tile {tile.shape} is incompatible with writing tile size {WRITING_TILE_SIZE}.") 
        tile = self._preprocess(tile) 
        self._writer.writeBaseImagePartToLocation(tile.flatten().astype('uint8'), x=int(x), y=int(y))

    def _preprocess(self, tile: np.ndarray):
        if self.software=="asap":
            return tile
        elif self.software=="sedeen":
            # return np.asarray(plt.cm.BuGn(self.norm(tile))[:,:,:-1]*255,np.uint8)
            u,inv = np.unique(tile,return_inverse=True)
            return np.array([self.colormap[x] for x in u])[inv].reshape(WRITING_TILE_SIZE,WRITING_TILE_SIZE,3)
        else:
            raise ValueError(f"Unknown software {self.software}")
            
    def save(self):
        self._writer.finishImage()