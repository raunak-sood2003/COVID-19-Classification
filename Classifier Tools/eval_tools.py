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

class VGG19_CAM(nn.Module):
    '''
    Wrapper class for VGG19 to prepare activation hooks for class activation mapping.
    
    '''
    
    def __init__(self, device):
        super(VGG19_CAM, self).__init__()
        
        # get the pretrained VGG19 network
        self.vgg = model.to(device)
        
        for param in self.vgg.parameters():
            param.requires_grad = True
        
        # disect the network to access its last convolutional layer
        self.features_conv = self.vgg.features[:36]
        
        # get the max pool of the features stem
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        
        # get the classifier of the vgg19
        self.classifier = self.vgg.classifier
        
        # placeholder for the gradients
        self.gradients = None
    
    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad
        
    def forward(self, x):
        x = self.features_conv(x)
        
        # register the hook
        h = x.register_hook(self.activations_hook)
        
        # apply the remaining pooling
        x = self.max_pool(x)
        x = x.view((1, -1))
        x = self.classifier(x)
        return x
    
    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients
    
    # method for the activation exctraction
    def get_activations(self, x):
        return self.features_conv(x)


def plot_train_val(history, num_epochs, fig_size):
    '''
    Plots training and validation loss and accuracy using pyplot.

    Args:
        num_epochs: number of epochs that model is trained for
        fig_size: size of output graph

    Returns: void
    '''

    plot_epochs = np.array([i for i in range(1, num_epochs+1)])
    fig, axs = plt.subplots(2, 2)

    axs[0, 0].plot(plot_epochs, history[0])
    axs[0, 0].set_title("Training Loss")

    axs[1, 0].plot(plot_epochs, history[1])
    axs[1, 0].set_title("Validation Loss")

    axs[0, 1].plot(plot_epochs, history[2])
    axs[0, 1].set_title("Training Accuracy")

    axs[1, 1].plot(plot_epochs, history[3])
    axs[1, 1].set_title("Validation Accuracy")

    fig.set_figheight(fig_size[0])
    fig.set_figwidth(fig_size[1])


def test(model, test_loader, criterion, device):
    '''
    Model testing 

    Args:
        model: model used during training and validation
        test_loader: data loader object containing testing data
        criterion: loss function used
        device: cuda or cpu device

    Returns:
        test_loss: calculated loss during testing
        accuracy: calculated accuracy during testing
        y_proba: predicted class probabilities
        y_truth: ground truth of testing data
    '''

    y_proba = []
    y_truth = []
    test_loss = 0
    total = 0
    correct = 0
    for data in tqdm(test_loader):
        X, y = data[0].to(device), data[1].to(device)
        output = model(X)
        test_loss += criterion(output, y.long()).item()
        for index, i in enumerate(output):
            y_proba.append(i[1])
            y_truth.append(y[index])
            if torch.argmax(i) == y[index]:
                correct+=1
            total+=1

    accuracy = correct/total

    y_proba_out = np.array([float(y_proba[i]) for i in range(len(y_proba))])
    y_truth_out = np.array([float(y_truth[i]) for i in range(len(y_truth))])

    return test_loss, accuracy, y_proba_out, y_truth_out

def display_FP_FN(model, test_loader, criterion, device, display = 'fp'):
    '''
    Displaying false positive or false negative images.

    Args:
        model: model used during training, testing and validation
        test_loader: data loader object for testing set
        criterion: loss function used
        device: cuda or cpu device
        display: either 'fp' for displaying false positives or 'fn' for false negatives

    Returns: void
    '''

    fp = []
    fn = []
    for data in tqdm(test_loader):
        X, y = data[0].to(device), data[1].to(device)
        output = model(X)
        for index, i in enumerate(output):
            if torch.argmax(i) == torch.Tensor([1]) and y[index] == torch.Tensor([0]):
                fp.append(X[index])
            elif torch.argmax(i) == torch.Tensor([0]) and y[index] == torch.Tensor([1]):
                fn.append(X[index])

    fig = plt.figure()

    if display == 'fp':
        n_img = len(fp)
        cols = int(math.sqrt(n_img))
        for idx, img in enumerate(fp):
            a = fig.add_subplot(cols, np.ceil(n_img/float(cols)), idx + 1)
            plt.imshow(img.view(224, 224, 3).cpu())
            plt.axis('off')


    elif display == 'fn':
        n_img = len(fn)
        cols = int(math.sqrt(n_img))
        for idx, img in enumerate(fn):
            a = fig.add_subplot(cols, np.ceil(n_img/float(cols)), idx + 1)
            plt.imshow(img.view(224, 224, 3).cpu())
            plt.axis('off')


def get_confusion_matrix(y_truth, y_proba, labels):
    '''
    Displays confusion matrix given output and ground truth data.

    Args:
        y_truth: ground truth for testing data output
        y_proba: class probabilties predicted from model
        labels: a list of labels for each cell of confusion matrix

    Returns:
        cm: returns a numpy array representing the confusion matrix

    '''

    y_in = np.array([round(i) for i in y_proba])
    cm = confusion_matrix(y_truth, y_in, labels)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    plt.title('COVID Confusion Matrix')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    return cm

def plot_ROCAUC_curve(y_truth, y_proba, fig_size):
    '''
    Plots the Receiver Operating Characteristic Curve (ROC) and displays Area Under the Curve (AUC) score.

    Args:
        y_truth: ground truth for testing data output
        y_proba: class probabilties predicted from model
        fig_size: size of the output pyplot figure

    Returns: void
    '''

    fpr, tpr, threshold = roc_curve(y_truth, y_proba)
    auc_score = roc_auc_score(y_truth, y_proba)
    txt_box = "AUC Score: " + str(round(auc_score, 4))
    plt.figure(figsize=fig_size)
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1],'--')
    plt.annotate(txt_box, xy=(0.65, 0.05), xycoords='axes fraction')
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")


def showCAM(model, dataloader):
    '''
    Displays class activation mapping for the final convolutional layer (GradCAM implementation from 
    https://arxiv.org/pdf/1610.02391.pdf and https://medium.com/@stepanulyanin/implementing-grad-cam-in-pytorch-ea0937c31e82)

    Args:
        model: model used during testing
        dataloader: dataloader object for test set

    Returns: void
    '''

    model.eval()
    img, _ = next(iter(dataloader))
    pred = model(img)
    idx = torch.argmax(pred)

    pred[:, 1].backward()

    gradients = model.get_activations_gradient()

    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

    activations = model.get_activations(img).detach()

    for i in range(512):
        activations[:, i, :, :] *= pooled_gradients[i]

    heatmap = torch.mean(activations, dim=1).squeeze()

    heatmap = np.maximum(heatmap, 0)

    heatmap /= torch.max(heatmap)

    npmap = heatmap.numpy()
    heat_mp = cv2.resize(npmap, (224, 224))
    heat_map = np.uint8(255*heat_mp)

    colored_map = cv2.cvtColor(heat_map, cv2.COLOR_GRAY2RGB)
    final_heatmap = cv2.applyColorMap(colored_map, cv2.COLORMAP_JET)

    #Alpha blending
    superimposed_img = cv2.addWeighted(img.view(224, 224, 3).numpy(), 0.5, final_heatmap, 0.5, 0, dtype=cv2.CV_64F)


    f = plt.figure(figsize = (10, 10))
    f.add_subplot(1,3, 1)
    plt.imshow(img.view(224,224,3))
    f.add_subplot(1,3, 2)
    plt.imshow(final_heatmap)
    f.add_subplot(1,3, 3)
    plt.imshow(superimposed_img)
    plt.show(block=True)

    
    
