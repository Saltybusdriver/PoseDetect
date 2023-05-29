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
import matplotlib.pyplot as plt
import torch.utils.data
import argparse

parser = argparse.ArgumentParser(description='Description of your script.')
parser.add_argument('-b', '--batch', type=int, default=1, help='Sets the batch size.')
parser.add_argument('-s', '--stack', type=int, default=8, help='Sets the amount of hourglass modules stacked.')
parser.add_argument('-d', '--depth', type=int, default=4, help='Sets the amount of residual and downscale pairs on a single side in the network.')
parser.add_argument('-S', '--size', type=int, default=96, help='Sets the resolution of the modified image, only accepts a single integer since the height and width dimensions are the same.')
parser.add_argument('-mb', '--maxbatch', type=int, default=None, help='Sets the maximum number of batches to use during training.')
parser.add_argument('-e', '--epoch', type=int, default=300, help='Sets the amount of epochs during training.')


New_width=None
New_height=None
BATCH_SIZE=None

args = parser.parse_args()
arg_batch=args.batch
arg_stack=args.stack
arg_depth=args.depth
arg_size=args.size
arg_epoch=args.epoch
arg_maxbatch=args.maxbatch
arg_newdata=False

import Model
import Heatmap
import Loss
import Loader
import sys
import getopt




Heatmap.BATCH_SIZE=arg_batch
Heatmap.New_height=arg_size
Heatmap.New_width=arg_size

Loader.BATCH_SIZE=arg_batch
Loader.New_height=arg_size
Loader.New_width=arg_size
Loader.newdata=arg_newdata
Loader.PreEdit()
Loader.datasetinit()
Loader.EvalLoad()
Loader.evalinit()
_test_loader=Loader.test_loader
criterion = Loss.LossCalculation()

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using device: ",dev)
print("Model params:")
print("Stack: ",arg_stack,"Depth: ",arg_depth)
num_joints=16
model=Model.StackedHourglassFCN(arg_stack,num_joints,arg_depth)
model=torch.load('Model/model.pth')
model.to(dtype=torch.bfloat16, device=dev)


print("loaded model")
def predict_image(model, _test_loader, device):
    #model.eval()
    
    with torch.no_grad():
        predictions_denorm=[]
        
        for i, (data, target) in enumerate(_test_loader):
            if i== 1:
                break
            data=data.unsqueeze(1)
            target=target.unsqueeze(2)
            data=data.to(dtype=torch.bfloat16)
            target=target.to(dtype=torch.bfloat16)
            data=data.to(dev)
            predictions = model(data)
            data=data.type(torch.float32)
            for i in range(0,len(data)):
                img_data = np.transpose(data[i].cpu().numpy(), (1, 2, 0))
                plt.imshow(img_data)
                plt.show()
          
                predictions_tensor = torch.from_numpy(predictions)
                summed_map = torch.sum(predictions_tensor[i,:], dim=0)
                plt.imshow(summed_map, cmap='viridis')
                plt.colorbar()
                plt.show()
predict_image(model,_test_loader,dev)