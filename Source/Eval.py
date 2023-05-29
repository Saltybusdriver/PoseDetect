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



def evaluate(model, dataloader):
    #model.eval() # Turned off due to BatchNorm layer anomaly in pytorch.
    
    total_loss = 0
    total_samples = 0
    x=0
    height, width = arg_size, arg_size
    num_joints=16
    batch_size=BATCH_SIZE
    with torch.no_grad():
        print("Evaluating performance...")
        for data, target in dataloader:
            data=data.unsqueeze(1)
            joint_mapss = torch.zeros(arg_batch, 16, arg_size, arg_size)
            
            target=target.reshape(arg_batch,16,2)
           
            
            joint_mapss = torch.zeros(arg_batch, num_joints, height, width)
            weight_ten = torch.zeros(arg_batch, num_joints, height, width)
            
            torch.manual_seed(123)
            
            for j in range(arg_batch):
                for i in range(16):
                    joint_center = target[j, i, :]
                    joint_map = Heatmap.create_gaussian_map(joint_center, (arg_size, arg_size), sigma=1)
                    weight=Heatmap.create_weights(joint_map)
                    weight_ten[j,i,:,:]=weight
                    joint_mapss[j, i, :, :] = joint_map
            joint_mapss=joint_mapss.transpose(2,3)
            weight_ten=weight_ten.transpose(2,3)
            
            # Forward pass
            data=data.to(dtype=torch.bfloat16)
            target=target.to(dtype=torch.bfloat16)
            joint_mapss=joint_mapss.to(dtype=torch.bfloat16)
            weight_ten=weight_ten.to(dtype=torch.bfloat16)
            data, target, joint_mapss, weight_ten = data.to(dev), target.to(dev), joint_mapss.to(dev), weight_ten.to(dev)
            output = model(data)
            loss=criterion(output,joint_mapss,weight_ten,data,arg_depth)

            # Update statistics
            total_loss += loss.item() * data.size(0)
            total_samples += data.size(0)
            if(x==arg_maxbatch):
                x=0
                break
            x+=1
    avg_loss = total_loss / total_samples
    print('Eval loss: {:.4f}'.format(avg_loss))
    return avg_loss

evaluate(model,_test_loader)

    