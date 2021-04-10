# %% 
import os 
import torch
from torch.autograd import Variable

import shutil

# %%
def del_folder(path, version):
    '''
    delete the folder which path/version

    Args:
        path (str): path
        version (str): version
    '''    
    if os.path.exists(os.path.join(path, version)):
        shutil.rmtree(os.path.join(path, version))
    
def make_folder(path, version):
    '''
    make folder which path/version

    Args:
        path (str): path
        version (str): version
    '''    
    if not os.path.exists(os.path.join(path, version)):
        os.makedirs(os.path.join(path, version))

def tensor2var(x, grad=False):
    '''
    put tensor to gpu, and compute grad

    Args:
        x (tensor): input tensor
        grad (bool, optional): if need grad. Defaults to False.

    Returns:
        tensor: tensor in gpu and need grad
    '''    
    if torch.cuda.is_available():
        x = x.cuda()
        x.requires_grad_(grad)
    return x

def var2tensor(x):
    '''
    put date to cpu

    Args:
        x (tensor): input tensor 

    Returns:
        tensor: put data to cpu
    '''    
    return x.data.cpu()

def var2numpy(x):
    return x.data.cpu().numpy()

def denorm(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)

def str2bool(v):
    return v.lower() in ('true')

def to_LongTensor(labels):
    '''
    put input labels to LongTensor

    Args:
        labels (numpy): labels

    Returns:
        LongTensor: return LongTensor labels
    '''    
    LongTensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor
    return LongTensor(labels)

def to_Tensor(x, *arg):
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.LongTensor
    return Tensor(x, *arg)