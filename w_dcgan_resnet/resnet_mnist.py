# %% 
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

import os 
print(os.listdir("input"))
# %%
import torch
import torch.nn as nn
import torchvision.transforms as transforms 
from torch.autograd import Variable
from sklearn.model_selection import train_test_split

# %%
# prepare dataset 
train = pd.read_csv(r"input/train.csv", dtype=np.float32)

targets_numpy = train.label.values
features_numpy = train.loc[:,train.columns != "label"].values/255

