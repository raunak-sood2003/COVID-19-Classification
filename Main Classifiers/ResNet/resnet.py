# ResNet Model Development + Training

from CT_COVID_Preprocess import CTCOVIDDataset
from model_tools import validation, train, load_model_checkpoint
from eval_tools import VGG19_CAM
from eval_tools import plot_train_val, test, display_FP_FN, get_confusion_matrix, plot_ROCAUC_curve, showCAM
from custom_model import CNN_COVIDCT
import torch
import torchvision
import torch.optim as optim
from monai.transforms import Compose, LoadPNG, AddChannel, ScaleIntensity, ToTensor, RandRotate, RandFlip, RandZoom, Resize, RandGaussianNoise
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import OrderedDict
from torchvision import transforms

# CHANGEABLE PARAMS
PRETRAINED = True 
verbose = True 
epochs = 40 
BATCH_SIZE = 32

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
dirCOVID = 'C:/Users/rrsoo/AdvancedStuff/Medical_AI/Classifier/Images-processed-new/CT_COVID'
dirNonCOVID = 'C:/Users/rrsoo/AdvancedStuff/Medical_AI/Classifier/Images-processed-new/CT_NonCOVID'

train_transforms = transforms.Compose([
        LoadPNG(),
        AddChannel(),
        ScaleIntensity(),
        RandRotate(degrees=15, prob=0.5),
        RandFlip(spatial_axis=0, prob=0.5),
        RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),
        RandGaussianNoise(prob = 0.5),
        Resize(spatial_size=(224, 224)),
        ToTensor()
    ])
   

val_transforms = transforms.Compose([
    LoadPNG(),
    AddChannel(),
    ScaleIntensity(),
    ToTensor()
])

ORIG_RES = False

train_ds = CTCOVIDDataset(dirCOVID, dirNonCOVID, transforms = train_transforms, data = 'train', orig_res = ORIG_RES)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

val_ds = CTCOVIDDataset(dirCOVID, dirNonCOVID, transforms = val_transforms, data = 'val', orig_res = ORIG_RES)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=True)

test_ds = CTCOVIDDataset(dirCOVID, dirNonCOVID, transforms = val_transforms, data = 'test', orig_res = ORIG_RES)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True)

model = torchvision.models.resnet34(pretrained = PRETRAINED).to(device)

if PRETRAINED:
    for param in model.parameters():
        param.requires_grad = False
        
layers_resnet = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(512, 256)),
            ('activation1', nn.ReLU()),
            ('fc2', nn.Linear(256, 128)),
            ('activation2', nn.ReLU()),
            ('fc3', nn.Linear(128, 2)),
            ('out', nn.Sigmoid())
        ])).to(device)

model.fc = layers_resnet

optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

if __name__ == '__main__':
    history = train(model, train_loader, val_loader, optimizer, criterion, epochs, device = device, scheduler = None)
    
    if verbose:
        plot_train_val(history, epochs, (10, 10))

    model_dir = 'resnetmodel.pth'

    checkpoints = {
        'model_name' : 'ResNet',
        'model' : model,
        'input_shape' : (224, 224, 3),
        'batch_size' : BATCH_SIZE,
        'epochs' : epochs,
        'state_dict' : model.state_dict()
    }

    torch.save(checkpoints, model_dir)
