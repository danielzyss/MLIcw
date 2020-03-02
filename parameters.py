import pandas as pd
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from ipywidgets import interact, fixed
import torchvision.transforms as T
from torch.nn import Conv3d, MaxPool3d, AvgPool3d, Linear, ReLU, MSELoss
import torch.optim as optim
data_dir = 'data/brain_age/'
import PIL
import random
from scipy.ndimage import zoom, shift

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torchvision

USE_GPU = True

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print("PYTORCH RUNNING ON DEVICE : ", device)