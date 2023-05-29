import sys 
from numpy import copy
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
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
import os


New_width=None
New_height=None
BATCH_SIZE=None
newdata=False
img_loc=(r"Images/images/")

F_trainimglabels=None
F_trainfilenames=None
F_testimglabels=None
F_testfilenames=None

train_loader=None
test_loader=None
train_dataset=None
test_dataset=None

def dataframe_editing():
    global F_trainimglabels
    global F_trainfilenames
    global F_testimglabels
    global F_testfilenames
    
    df=pd.read_csv('csv/mpii_human_pose.csv')
    
    df = df.sample(n = len(df)).reset_index(drop=True)
    transferdata_df = df.iloc[- 1400:, 0:]

    test_df = pd.DataFrame()
    test_df= test_df.append(transferdata_df)
    train_df =df.drop(transferdata_df.index)
    test_df = test_df.reset_index(drop=True)
    
   
    trainimglabels=train_df.loc[:,'r ankle_X':'l wrist_Y']
    trainfilenames=train_df['NAME']

    testimglabels=test_df.loc[:,'r ankle_X':'l wrist_Y']
    testfilenames=test_df['NAME']
    print("Checking and removing incorrect data for train dataset..")
    to_remove=[]
    for i, row in trainimglabels.iterrows():
        img=cv2.imread(img_loc+trainfilenames[i])
        
        width,height, ch=img.shape
        for j in range(0,len(row),2):
            x=row[j]
            y=row[j+1]
            if x > width or y > height:
                to_remove.append(i)
                break
    filtered_trainimglabels=trainimglabels.drop(to_remove).reset_index(drop=True)
    filtered_trainfilenames=trainfilenames.drop(to_remove).reset_index(drop=True)
    print("Checking and removing complete")
    print("Checking and removing incorrect data for test dataset..")
    to_remove=[]
    for i, row in testimglabels.iterrows():
        img=cv2.imread(img_loc+testfilenames[i])
        width,height, ch=img.shape
        for j in range(0,len(row),2):
            x=row[j]
            y=row[j+1]
            if x > width or y > height:
                to_remove.append(i)
                break
    filtered_testimglabels=testimglabels.drop(to_remove).reset_index(drop=True)
    filtered_testfilenames=testfilenames.drop(to_remove).reset_index(drop=True)
    print("Checking and removing complete")
    filtered_trainimglabels.to_csv('csv/filtered_trainimg')
    filtered_trainfilenames.to_csv('csv/filtered_names')
    filtered_testimglabels.to_csv('csv/filtered_testimg')
    filtered_testfilenames.to_csv('csv/filtered_testnames')
    print("Saved csv's to /csv/")
    F_trainimglabels=filtered_trainimglabels
    F_trainfilenames=filtered_trainfilenames
    F_testimglabels=filtered_testimglabels
    F_testfilenames=filtered_testfilenames
    F_trainfilenames=F_trainfilenames.squeeze()
    F_testfilenames=F_testfilenames.squeeze()

def LoadCleanedCSV():
    global F_trainimglabels
    global F_trainfilenames
    global F_testimglabels
    global F_testfilenames
    print("Loading Cleaned csv")
    filtered_trainimglabels=pd.read_csv('csv/filtered_trainimg')
    filtered_trainfilenames=pd.read_csv('csv/filtered_names')

    filtered_testimglabels=pd.read_csv('csv/filtered_testimg')
    filtered_testfilenames=pd.read_csv('csv/filtered_testnames')
    F_trainimglabels=filtered_trainimglabels.drop(columns=filtered_trainimglabels.columns[0], axis=1, inplace=False)
    F_trainfilenames=filtered_trainfilenames.drop(columns=filtered_trainfilenames.columns[0], axis=1, inplace=False)
    F_testimglabels=filtered_testimglabels.drop(columns=filtered_testimglabels.columns[0], axis=1, inplace=False)
    F_testfilenames=filtered_testfilenames.drop(columns=filtered_testfilenames.columns[0], axis=1, inplace=False)
    filtered_trainfilenames.reset_index()
    filtered_testfilenames.reset_index()
    F_trainfilenames=F_trainfilenames.squeeze()
    F_testfilenames=F_testfilenames.squeeze()
    print("csv load complete")

class Imagedataset(Dataset):
    def __init__(self,fnames,labels,transform=None):
        self.fnames=fnames
        self.labels=labels
        self.transform=transform
        self.BackupLabels=labels.copy()
        self.target=[]
        self.imgdataarr=[]
    def __len__(self):
        return len(self.fnames)
    def SaveArr(self,name):
        torch.save(self.target,name+'_Targetval.pt')
        torch.save(self.imgdataarr,name+'_Imgdata.pt')
    def LoadArr(self,name):
        self.target=torch.load("../working/Train_Imgdata.pt")
        self.imgdataarr=torch.load("../working/Train_Imgdata.pt")
        print("Successfull data load")
    def CropBox(self, img, idx):
        min_x = float('inf')
        max_x = float('-inf')
        min_y = float('inf')
        max_y = float('-inf')
        for i in range(0,32,2):
            if (self.labels.iloc[idx,i]<min_x and self.labels.iloc[idx,i]!=-1):
                min_x=self.labels.iloc[idx,i] 
            if (self.labels.iloc[idx,i]>max_x and self.labels.iloc[idx,i]!=-1):
                max_x=self.labels.iloc[idx,i]
            if (self.labels.iloc[idx,i+1]<min_y and self.labels.iloc[idx,i+1]!=-1):
                min_y=self.labels.iloc[idx,i+1]
            if (self.labels.iloc[idx,i+1]>max_y and self.labels.iloc[idx,i+1]!=-1):
                max_y=self.labels.iloc[idx,i+1]
        h, w = img.shape[:2]
        padding=100
        paddedmin_y=max(0,round(min_y)-padding)
        paddedmax_y=min(h,round(max_y)+padding)
        paddedmin_x=max(0,round(min_x)-padding)
        paddedmax_x=min(w,round(max_x)+padding)
        crop_img=img[paddedmin_y:paddedmax_y, paddedmin_x:paddedmax_x]
        
        for i in range(0,32,2):
            if(self.labels.iloc[idx,i]!=-1):
                self.labels.iloc[idx,i]=self.labels.iloc[idx,i]-paddedmin_x
            if(self.labels.iloc[idx,i+1]!=-1):
                self.labels.iloc[idx,i+1]=self.labels.iloc[idx,i+1]-paddedmin_y
        return crop_img
    def NormaliseData(self,img,idx):
        crop_img=self.CropBox(img,idx)
        height, width= crop_img.shape
        for i in range(0,32,2):
            if(self.labels.iloc[idx,i]!=-1):
                self.labels.iloc[idx,i]=(self.labels.iloc[idx,i]*(New_width/width))
            if(self.labels.iloc[idx,i+1]!=-1):
                self.labels.iloc[idx,i+1]=(self.labels.iloc[idx,i+1]*(New_height/height))
        normImg=cv2.resize(crop_img,(New_width,New_height))  
        return normImg
    def preload(self,idx):
        Rimg=cv2.imread(img_loc+self.fnames[idx])
        Gimg=cv2.cvtColor(Rimg,cv2.COLOR_BGR2GRAY)
        Gimg=self.NormaliseData(Gimg,idx)
        Gimg = Gimg.astype(np.float16) / 255.
        if self.transform:
            Gimg = self.transform(Gimg)
        target=self.labels.iloc[idx].values
        target.astype(np.float16)
       
        self.target.append(target)
        self.imgdataarr.append(Gimg)

    def __getitem__ (self, idx):
        return (self.imgdataarr[idx],self.target[idx])


def PreEdit():
    if(newdata==True):
        dataframe_editing()
    else:
        LoadCleanedCSV()

def datasetinit():
    global train_dataset
    global test_dataset
    train_dataset = Imagedataset(F_trainfilenames,F_trainimglabels)
    test_dataset = Imagedataset(F_testfilenames,F_testimglabels)
    


def loaderinit():
    global train_loader
    global test_loader
    train_loader=DataLoader(train_dataset,batch_size=BATCH_SIZE, shuffle=False,num_workers=2)
    test_loader=DataLoader(test_dataset,batch_size=BATCH_SIZE, shuffle=True)
def evalinit():
    
    global test_loader
    test_loader=DataLoader(test_dataset,batch_size=BATCH_SIZE, shuffle=True)

def preloading():
    print("loading train dataset to memory...")
    for i in range (len(train_dataset)):
        train_dataset.preload(i)
        if(i%100==0):
            print(i)
    print("loading complete")
    print("loading test dataset to memory...")    
    for i in range (len(test_dataset)):
        test_dataset.preload(i)
        if(i%100==0):
            print(i)
    print("loading complete")

def EvalLoad():
    print("loading test dataset to memory...")    
    for i in range (len(test_dataset)):
        test_dataset.preload(i)
        if(i%100==0):
            print(i)
    print("loading complete")
