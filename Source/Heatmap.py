from scipy.stats import norm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import scipy.io
import cv2
import glob
import numpy as np
import pandas as pd
#from sklearn.metrics import confusion_matrixmatrix
import matplotlib.pyplot as plt
import torch.utils.data

New_width=None
New_height=None
BATCH_SIZE=None
def create_gaussian_map(center, output_shape, sigma=4):
   
    height, width = output_shape
    x = torch.arange(0, width, 1, dtype=torch.float32, device=center.device)
    y = torch.arange(0, height, 1, dtype=torch.float32, device=center.device)
    x, y = torch.meshgrid(x, y)
    x0, y0 = center[0].item(), center[1].item()
    heatmap=torch.zeros((New_width, New_height), dtype=torch.float32)


    if(x0==-1 or y0==-1 ):
        return torch.zeros((New_width,New_height), dtype=torch.float32)
    map=torch.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    mask = (torch.abs(x - x0) <= 5) & (torch.abs(y - y0) <= 5)
    heatmap[mask] = torch.exp(-((x[mask] - x0) ** 2 + (y[mask] - y0) ** 2) / (2 * sigma ** 2))
    heatmap=heatmap*60
    
    return heatmap


def create_weights(map):
    mapped=map.clone()
    #mapped=mapped*80

    return mapped