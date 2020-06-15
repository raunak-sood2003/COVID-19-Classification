import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from monai.transforms import Compose, LoadPNG, AddChannel, ScaleIntensity, ToTensor, RandRotate, RandFlip, RandZoom, Resize, RandGaussianNoise
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix
import os
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import math
from tqdm import tqdm
import cv2
import pandas as pd

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dirCOVID = 'C:/Users/rrsoo/AdvancedStuff/Medical_AI/Images-processed-new/CT_COVID'
dirNonCOVID = 'C:/Users/rrsoo/AdvancedStuff/Medical_AI/Images-processed-new/CT_NonCOVID'

class CTCOVIDDataset(Dataset):
    '''
    Dataset class for preprocessing CT COVID images in PNG format.
    
    Args: 
        dirCOVID: COVID positive image directory
        dirNonCOVID: COVID negative image directory
        transforms: list of transforms using torchvision.transforms
        data: either 'train', 'test', or 'val' (split 80: 10: 10)
        orig_res: option of leaving images in original CT (512, 512, 3) format or standard image resolution (224, 224, 3) for state of the art models
    
    Output:
        Images and labels wrapped in Dataset class prepared for DataLoader creation
    '''
    
    def __init__(self, dirCOVID, dirNonCOVID, transforms = None, data = None, orig_res = False):
        if orig_res:
            self.IMG_SIZE = 512
        else:
            self.IMG_SIZE = 224
        self.valSplit = 0.1
        self.transforms = transforms
        self.data = data
        self.orig_res = orig_res
        self.COVID = dirCOVID
        self.NonCOVID = dirNonCOVID
        self.LABELS = {self.NonCOVID:0, self.COVID:1}
        self.training_data = []
        
        for label in self.LABELS:
            for f in tqdm(os.listdir(label)):
                path = os.path.join(label, f)
                img = cv2.imread(path, cv2.IMREAD_COLOR)
                img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
                self.training_data.append([np.array(img), self.LABELS[label]])
                
        np.random.shuffle(self.training_data)
        
        self.valSize = int(self.valSplit*len(self.training_data))

        self.X_val = torch.Tensor([i[0] for i in self.training_data[0:self.valSize]]).view(-1, 3, self.IMG_SIZE, self.IMG_SIZE)
        self.X_val/=255
        self.y_val = torch.Tensor([i[1] for i in self.training_data[0:self.valSize]])

        self.X_test = torch.Tensor([i[0] for i in self.training_data[self.valSize: self.valSize+self.valSize]]).view(-1, 3, self.IMG_SIZE, self.IMG_SIZE)
        self.X_test/=255
        self.y_test = torch.Tensor([i[1] for i in self.training_data[self.valSize: self.valSize+self.valSize]])

        self.X_train = torch.Tensor([i[0] for i in self.training_data[self.valSize+self.valSize:]]).view(-1, 3, self.IMG_SIZE, self.IMG_SIZE)
        self.X_train/=255
        self.y_train = torch.Tensor([i[1] for i in self.training_data[self.valSize+self.valSize:]])

    def __len__(self):
        
        if self.data == 'train':
            return self.y_train.shape[0]
        elif self.data == 'val':
            return self.y_val.shape[0]
        elif self.data == 'test':
            return self.y_test.shape[0]

    def __getitem__(self, idx):
        
        if self.data == 'train':
            return self.X_train[idx], self.y_train[idx]
        elif self.data == 'val':
            return self.X_val[idx], self.y_val[idx]
        elif self.data == 'test':
            return self.X_test[idx], self.y_test[idx]