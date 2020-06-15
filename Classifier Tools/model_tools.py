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

def validation(model, val_loader, criterion, device):
    '''
    Helper validation loss and accuracy function for use during model training

    Args:
        model: model object (ex: vgg16, resnet or custom model using nn.Module)
        val_loader: data loader for validation set
        criterion: loss function used for training
        device: cuda or cpu device

    Returns:
        test_loss: the loss during validation testing
        accuracy: the accuracy during validation testing
    '''

    test_loss = 0
    total = 0
    correct = 0
    for data in tqdm(val_loader):
        X, y = data[0].to(device), data[1].to(device)
        output = model(X)
        test_loss += criterion(output, y.long()).item()

        for index, i in enumerate(output):
            if torch.argmax(i) == y[index]:
                correct+=1
            total+=1

    accuracy = correct/total

    return test_loss, accuracy

def train(model, train_loader, val_loader, optimizer, criterion, epochs, device, scheduler = None):
    '''
    Training loop function

    Args:
        model: model: model object (ex: vgg16, resnet or custom model using nn.Module)
        train_loader: data loader object with the training data
        val_loader: data loader object with the validation images
        optimizer: optimier used from torchvision.optim
        criterion: loss function used
        epochs: number of epochs that training loop runs
        device: cuda or cpu device
        scheduler: (optional) learning rate scheduler

    Returns:
        plot_train_loss: numpy array of training loss every epoch (length of epochs)
        plot_val_loss: numpy array of validation loss every epoch (length of epochs)
        plot_train_acc: numpy array of training accuracy every epoch (length of epochs)
        plot_val_acc: numpy array of validation accuracy every epoch (length of epochs)
    '''

    plot_train_loss = []
    plot_val_loss = []
    plot_train_acc = []
    plot_val_acc = []
    for epoch in range(epochs):
        model.train()
        total = 0
        correct = 0
        train_loss = 0
        for data in tqdm(train_loader):
            X, y = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y.long())
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

            for index, i in enumerate(output):
                if torch.argmax(i) == y[index]:
                    correct+=1
                total+=1

        if scheduler is not None:
            scheduler.step()
        
        model.eval()

        with torch.no_grad():
            val_loss, val_acc = validation(model, val_loader, criterion, device = device)

        train_acc = correct/total

        print("Training Loss:", train_loss)
        plot_train_loss.append(train_loss)

        print("Training Accuracy:", train_acc)
        plot_train_acc.append(train_acc)

        print("Validation Loss:", val_loss)
        plot_val_loss.append(val_loss)

        print("Validation Accuracy:", val_acc)
        plot_val_acc.append(val_acc)

        model.train()

    plot_train_loss = np.array(plot_train_loss) 
    plot_val_loss = np.array(plot_val_loss)
    plot_train_acc = np.array(plot_train_acc)
    plot_val_acc = np.array(plot_val_acc)

    return plot_train_loss, plot_val_loss, plot_train_acc, plot_val_acc

def load_model_checkpoint(filepath):
    '''
    Helper function to load saved models in local directory

    Args:
        filepath: directory for model file

    Returns:
        model: the saved model object
    '''

    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    return model
    
    
