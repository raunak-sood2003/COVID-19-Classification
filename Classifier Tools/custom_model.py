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

class CNN_COVIDCT(nn.Module):
    '''
    Custom COVID CT classifier using. 3 convolutional layers and 3 linear layers.
    
    Input:
        X: Batch of training images
        
    Output:
        Class probabilities for COVID positive and negative
    '''
    
    def __init__(self):
        super(CNN_COVIDCT, self).__init__()
        
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 16, (5, 5)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)))
        
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, (5, 5)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)))
        
        self.block3 = nn.Sequential(
            nn.Conv2d(32, 64, (5, 5)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)))
        
        self.linear = nn.Sequential(
            nn.Linear(64*60*60, 1000),
            nn.ReLU(),
            nn.Linear(1000, 2),
            nn.Sigmoid())
          
    def forward(self, X):
        X = self.block1(X)
        X = self.block2(X)
        X = self.block3(X)
        X = self.linear(X)
        
        return X
