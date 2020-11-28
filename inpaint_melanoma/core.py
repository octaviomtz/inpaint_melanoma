# AUTOGENERATED! DO NOT EDIT! File to edit: 00_core.ipynb (unless otherwise specified).

__all__ = ['rgb2gray', 'otsu_thresh_4largest_component', 'component_closest_center', 'get_center', 'denormalizePatches',
           'figs_horizontal2', 'figs_comparison', 'figs_horizontal3', 'plot_inpaints_pairs']

# Cell
import argparse
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
import torch.optim
from copy import copy, deepcopy
import time
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pandas as pd
from skimage import measure, morphology
from itertools import groupby, count
import matplotlib.patches as patches
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from torch.autograd import Variable
from scipy.spatial import distance
import sys
from PIL import Image
from matplotlib.gridspec import GridSpec

# Cell
# from models.skip import skip
from .skip import *
from .inpainting_utils import *

# Cell
from inpainting_nodules_functions import *
import warnings
from torch.autograd import Variable
# from google.colab import drive
from scipy import ndimage
from skimage import filters

warnings.filterwarnings("ignore", category=UserWarning)

# Cell
def rgb2gray(rgb):
    '''https://stackoverflow.com/questions/12201577/how-can-i-convert-an-rgb-image-into-grayscale-in-python'''
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def otsu_thresh_4largest_component(img2):
    val = filters.threshold_otsu(img2)
    mask_otsu_orig = img2<val
    mask_otsu = ndimage.morphology.binary_erosion(mask_otsu_orig, iterations=20)
    mask_otsu = ndimage.morphology.binary_dilation(mask_otsu, iterations=80)
    mask_otsu = ndimage.morphology.binary_fill_holes(mask_otsu)
    labeled_mask, cc_num = ndimage.label(mask_otsu)
    sorted_comp = np.bincount(labeled_mask.flat)
    sorted_comp = np.sort(sorted_comp)[::-1]
    mask_lesions = []
    for i in np.arange(1, np.min([len(sorted_comp), 4])):
        mask_lesions.append((labeled_mask == np.where(np.bincount(labeled_mask.flat) == sorted_comp[i])[0][0]))
    return mask_lesions

def component_closest_center(img2, masks_lesions):
    y_half, x_half = [i//2 for i in np.shape(img2)]
    y_half_x_half = np.asarray([y_half, x_half])
    ml_closest = masks_lesions[0] # default
    dist_min = 10000
    for i in masks_lesions:
        yy,xx = np.where(i==1)
        ymed_xmed = np.asarray([np.median(yy), np.median(xx)])
        dist_new = distance.cdist(np.expand_dims(y_half_x_half,0), np.expand_dims(ymed_xmed,0))
        if dist_new < dist_min:
            dist_min = dist_new
            ml_closest = i
    return ml_closest

def get_center(img, part=.25):
    factor = 32
    y_half, x_half, _ = [i//2 for i in np.shape(img)]
    y_include, x_include = np.asarray([y_half, x_half])* part
    y_include = y_include + (factor - y_include % factor)
    x_include = x_include + (factor - x_include % factor)
    y_part1, x_part1 = int(y_half - y_include), int(x_half - x_include)
    y_part2, x_part2 = int(y_half + y_include), int(x_half + x_include)
    y_part1, y_part2, x_part1, x_part2
    return img[y_part1: y_part2, x_part1: x_part2,:], y_part1, x_part1

def denormalizePatches(img):
    img_min = np.min(img)
    img_max = np.max(img)
    img = img * img_max + img_min
    img = img.astype('int16')
    return img

# Cell
def figs_horizontal2(ff, names_selected, suffix_available, path_source):
    f1 = [names_selected+j for j in suffix_available if names_selected+j in ff]
    f1 = np.unique(f1)
    f1 = np.sort(f1)
    n_img = len(f1)
    fig, ax = plt.subplots(1,n_img,figsize=(24,5))
    for idx,i in enumerate(f1):
        # name_display = '_'.join(names_selected.split('_')[1:])
        name_display = i.split('_ISIC')[0].split('_')[-1]
        a = np.fromfile(f'{path_source}{i}',dtype='int16')
        a = a/255
        a = np.clip(a,0,1)
        a = np.reshape(a, (3,512,512))
        a = np.swapaxes(a,0,1)
        a = np.swapaxes(a,1,2)
        ax[idx].imshow(a)
        ax[idx].text(10,50,name_display)
    for axx in ax.ravel(): axx.axis('off')
    plt.tight_layout()
    print(names_selected)
    return f1

# Cell
def figs_comparison(ff, names_selected, suffix_available, gen_idx, folder='/mnt/90cf2a10-3cf8-48a6-9234-9973231cadc6/Kaggle/melanoma/datasets_preprocessed/size_512/'):
    f1 = [names_selected+j for j in suffix_available if names_selected+j in ff]
    f1 = np.unique(f1)
    f1 = np.sort(f1)
    n_img = len(f1)
    i = f1[gen_idx]

    key = 'ISIC'+suffix_available[0].split('.raw')[0].split('ISIC')[-1]
    orig = plt.imread(f'{folder}{key}.jpg')
    mask = np.load(f'{folder}mask_{key}.npz')
    mask = mask.f.arr_0

    fig, ax = plt.subplots(1,4,figsize=(12,5))
    name_display = i.split('_ISIC')[0].split('_')[-1]
    inpain = np.fromfile(f'{path_source}{i}',dtype='int16')
    inpain = inpain/255
    inpain = np.clip(inpain,0,1)
    inpain = np.reshape(inpain, (3,512,512))
    inpain = np.swapaxes(inpain,0,1)
    inpain = np.swapaxes(inpain,1,2)
    ax[1].imshow(orig)
    ax[0].imshow(orig)
    ax[0].imshow(mask, alpha=.3)
    ax[2].imshow(inpain)
    ax[3].imshow(inpain)
    ax[3].imshow(mask, alpha=.3)
    for axx in ax.ravel(): axx.axis('off')
    plt.tight_layout()
    return key, inpain

# Cell
def figs_horizontal3(ff, names_selected, suffix_available, path_results):
    f1 = [names_selected+j for j in suffix_available if names_selected+j in ff]
    f1 = np.unique(f1)
    f1 = np.sort(f1)
    n_img = len(f1)
    fig, ax = plt.subplots(1,n_img,figsize=(24,5))
    for idx,i in enumerate(f1):
      name_display = i.split('_ISIC')[0].split('_')[-1]
      a = Image.open(f'{path_results}{i}')
      ax[idx].imshow(a)
      ax[idx].text(10,50,name_display)
    for axx in ax.ravel(): axx.axis('off')
    plt.tight_layout()
    print(names_selected)
    return f1

# Cell
def plot_inpaints_pairs(mse_error, images_raw, images_combined, epochs_saved, filename, archi, params, path_save=''):
    fontsize = 20
    color1 = "#3F5D7D"
    color2 = "#990F02"
    color3 = "#ffe84f"
    widths = [1,2,2,2,2]
    fig=plt.figure(figsize=(18,8));
    gs=GridSpec(2,5, width_ratios=widths)
    ax1=fig.add_subplot(gs[:,0]) # First row, first column
    ax2=fig.add_subplot(gs[0,1]) # First row, second column
    ax3=fig.add_subplot(gs[0,2]) # First row, third column
    ax4=fig.add_subplot(gs[0,3])
    ax5=fig.add_subplot(gs[0,4])
    ax6=fig.add_subplot(gs[1,1])
    ax7=fig.add_subplot(gs[1,2])
    ax8=fig.add_subplot(gs[1,3])
    ax9=fig.add_subplot(gs[1,4])

    count=0
    for i, ax_ in zip(images_raw,      [ax2, ax4, ax6, ax8]):
        ax_.imshow(i)
        ax_.text(10, 50, str(epochs_saved[-4+count]*10), fontsize=fontsize)
        count+=1
    for i, ax_ in zip(images_combined, [ax3, ax5, ax7, ax9]): ax_.imshow(i)
    for i in [ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]: i.axis('off')

    name = f'{archi}\n{params}'
    name = name.replace('_LR','\nLR')
    ax9.text(10,140,name, fontsize=fontsize)
    ax1.semilogy(mse_error, color=color1)
    epochs_saved = np.asarray(epochs_saved)*10
    ax1.semilogy(np.asarray(epochs_saved)[-4:],np.asarray(mse_error)[np.asarray(epochs_saved)][-4:], marker='.', linestyle='None', markersize=20, color=color1)
    fig.tight_layout()

    if len(path_save)>0:
        fig.savefig(f'{path_save}ov_{filename}_{name}.png' )
        plt.close()