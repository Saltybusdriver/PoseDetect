#from sklearn.preprocessing import LabelEncoder
#from IPython.display import clear_output
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
import os
import timeit



import Model
import Heatmap
import Loss
import Loader
import sys
import getopt
import argparse

parser = argparse.ArgumentParser(description='Description of your script.')

parser.add_argument('-b', '--batch', type=int, default=1, help='Sets the batch size.')
parser.add_argument('-s', '--stack', type=int, default=8, help='Sets the amount of hourglass modules stacked.')
parser.add_argument('-d', '--depth', type=int, default=4, help='Sets the amount of residual and downscale pairs on a single side in the network.')
parser.add_argument('-S', '--size', type=int, default=96, help='Sets the resolution of the modified image, only accepts a single integer since the height and width dimensions are the same.')
parser.add_argument('-mb', '--maxbatch', type=int, default=None, help='Sets the maximum number of batches to use during training.')
parser.add_argument('-e', '--epoch', type=int, default=300, help='Sets the amount of epochs during training.')
parser.add_argument('-nd', '--newdataset', type=bool, default=False, help='Tells the dataloader to do additional checks on the dataset since its new.')
parser.add_argument('-l', '--load', type=bool, default=False, help='Loads a model from the Model folder.')

args = parser.parse_args()
arg_batch=args.batch
arg_stack=args.stack
arg_depth=args.depth
arg_size=args.size
arg_epoch=args.epoch
arg_maxbatch=args.maxbatch
arg_newdata=args.newdataset 
_loadmodel=args.load

print("running train with current parameters: batch: ",arg_batch," stacK: ",arg_stack," depth: ",arg_depth," size: ",arg_size," epochs: ",arg_epoch, " maxbatch:",arg_maxbatch, " Load model: ",_loadmodel)
num_joints = 16
num_body_parts = 32


Loader.BATCH_SIZE=arg_batch
Loader.New_height=arg_size
Loader.New_width=arg_size
Loader.newdata=arg_newdata
Loader.PreEdit()
Loader.datasetinit()
Loader.preloading()
Loader.loaderinit()

Heatmap.BATCH_SIZE=arg_batch
Heatmap.New_height=arg_size
Heatmap.New_width=arg_size




dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using device: ",dev)
print("Model params:")
print("Stack: ",arg_stack,"Depth: ",arg_depth)
model=Model.StackedHourglassFCN(arg_stack,num_joints,arg_depth)
model.to(dtype=torch.bfloat16, device=dev)
criterion = Loss.LossCalculation()
weight_dec = 1e-5
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

_train_loader=Loader.train_loader



def saveModel(Smodel):
    torch.save(Smodel, "Model/model.pth")
def loadModel(model):
    model=torch.load("Model/model.pth")


if(_loadmodel==True):
    loadModel(model)
def train2(model, dev, _train_loader, optimizer, num_epochs=arg_epoch, batch_size=arg_batch):
    model.train()
    id=0

    height, width = arg_size, arg_size
    num_joints=16
    for epoch in range(num_epochs):
        print("Loading Epoch...")
        start = timeit.default_timer()
        for data, target in _train_loader:
            #####unsquueze so i have channel dimension
            data = data.unsqueeze(1)
            target=target.reshape(batch_size,num_joints,2)  ## reshape target so that pairs of xy are next to each other..
            joint_mapss = torch.zeros(batch_size, num_joints, height, width)
            weight_ten = torch.zeros(batch_size, num_joints, height, width)
         
            regression_target = target
            torch.manual_seed(123)
            
            for j in range(batch_size):
                for i in range(16):
                    joint_center = regression_target[j, i, :]
                
                    joint_map = Heatmap.create_gaussian_map(joint_center, (height, width), sigma=1)
                    weight=Heatmap.create_weights(joint_map)
                    weight_ten[j,i,:,:]=weight
                    joint_mapss[j, i, :, :] = joint_map
            joint_mapss=joint_mapss.transpose(2,3)
            weight_ten=weight_ten.transpose(2,3)
            
            data=data.to(dtype=torch.bfloat16)
            regression_target=regression_target.to(dtype=torch.bfloat16)
            joint_mapss=joint_mapss.to(dtype=torch.bfloat16)
            weight_ten=weight_ten.to(dtype=torch.bfloat16)
            data, regression_target, joint_mapss, weight_ten =  data.to(dev), regression_target.to(dev), joint_mapss.to(dev), weight_ten.to(dev)
            output = model(data)
            
           
            for param in model.parameters():
                param.grad = None
            loss=criterion(output,joint_mapss,weight_ten,data,arg_depth)
            loss.backward()
            optimizer.step()
            print(f'Train Epoch: {epoch} [{id}/{len(_train_loader)} ({100.*id/len(_train_loader):.0f}%)]\tLoss: {loss.item()}')
            del loss
            if(arg_maxbatch is not None):
                if(id==arg_maxbatch):
                    id=0
                    break
            id=id+1
            if(id==(len(_train_loader)-20)):
                id=0
                break
        stop = timeit.default_timer()
        print('Time: ', stop - start) 
train2(model, dev, _train_loader, optimizer, num_epochs=arg_epoch, batch_size=arg_batch)
saveModel(model)