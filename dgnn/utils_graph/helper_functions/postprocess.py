"""
This script is preprocess tumorbed region based on raw prediciton values, performs opening, closing, filling and removing operations
"""
from typing import Union
import numpy as np
from skimage import morphology
import cv2

def construct_image(template, X):
    """
    Based on the given tissuemasks template, puts the given array in the correct grid
    """
    fill_tumour = template.flatten().copy()
    fill_tumour[np.where(fill_tumour >= 1)[0]] = 255 * X
    tumour_heatmap = np.reshape(fill_tumour, np.shape(template))
    return tumour_heatmap, template


def resize(img, scale_percent=500, dim=None):
    if dim is None:
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
    else:
        dim = dim
    # resize image
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized

def dist2px(dist, spacing):
    dist_px = int(round(dist / (256 * spacing)))
    return dist_px


def area2px(areainmm, spacing):
    # 64 pixels is 1mm^2 , 1*10^6(um to mm)/(0.5*0.5) px= 4*10^6px, 1 px^2 is 256*256 => 4*10^6/(256*256) = 61
    px_area = int(round(areainmm * 1000000 / ((256 * spacing) ** 2)))
    return px_area


def convert2list_tumor(tumorbed, template):
    """
    Converts the image into list for computation purposes
    """
    processed_map = template * tumorbed
    # Subtracting 1 because indices start from 0
    indices = processed_map[np.where(processed_map > 0)].astype(int) - 1
    tem_flat = template.flatten()
    # To get the template in flattened form
    tem_flat = 0 * tem_flat[np.where(tem_flat > 0)]
    tem_flat[indices] = 1
    return tem_flat.astype(int)
def postprocess_tumorbed(
    template: np.array, spacing: float, Tumorbed_calculation: np.array
) -> np.array:
    """
    Performs post processing on a given slide, by resizing, closing/opening, filling and removing isolated pixels,
    to give a processed tumorbed
    """
    tissue_shape = np.shape(template)
    # Small buffer added, if ratio<0.5 then we dont perform any operation
    temp_prop = 0.02 + np.sum(template > 0) / (tissue_shape[0] * tissue_shape[1])

    resize_dim = (500, 500)
    kernel_size = (3, 3)
    fill_area_thresh_mm = 0.05
    remove_area_thresh_mm = 1.5

    if temp_prop < 0.5:
        # For biopsies with very less tissue area
        resize_dim = (700, 700)
        kernel_size = (2, 2)
        fill_area_thresh_mm = 0.05
        remove_area_thresh_mm = 0.5

    # Read the mask at a reasonable level for fast processing.

    tumourbed, template = construct_image(template, Tumorbed_calculation)
    tumourbed = tumourbed.astype(np.uint8)

    original_shape = np.shape(tumourbed.T)
    # Processing
    tumourbed = resize(tumourbed, dim=resize_dim)
    ret, thresh1 = cv2.threshold(tumourbed, int(0.5 * 255), 255, cv2.THRESH_BINARY)
    # Opening and closing operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    closing = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    # Reshape
    tumourbed_reshape = resize(opening, dim=original_shape)

    # Flood Fill
    filled = morphology.remove_small_holes(
        tumourbed_reshape,
        area_threshold=area2px(fill_area_thresh_mm, spacing),
        connectivity=2,
    )
    removesmallitems = morphology.remove_small_objects(
        filled, min_size=area2px(remove_area_thresh_mm, spacing), connectivity=2
    )
    processed_tumorbed = convert2list_tumor(removesmallitems, template)

    return processed_tumorbed