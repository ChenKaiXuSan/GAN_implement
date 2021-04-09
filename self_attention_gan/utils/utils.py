# %% 
import os 
import torch
from torch.autograd import Variable

import shutil

# %%
def del_folder(path, version):
    if os.path.exists(os.path.join(path, version)):
        shutil.rmtree(os.path.join(path, version))
    
def make_folder(path, version):
    if not os.path.exists(os.path.join(path, version)):
        os.makedirs(os.path.join(path, version))

def tensor2var(x, grad=False):
    if torch.cuda.is_available():
        x = x.cuda()
        x.requires_grad_(grad)
    return x

def var2tensor(x):
    return x.data.cpu()

def var2numpy(x):
    return x.data.cpu().numpy()

def denorm(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)

def str2bool(v):
    return v.lower() in ('true')

def to_LongTensor(labels):
    LongTensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor
    return LongTensor(labels)