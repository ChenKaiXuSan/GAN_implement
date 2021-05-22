import torch
import torchvision.utils as utils

import os 
import shutil

from options import args

def get_cuda(tensor):
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def generate_samples(img_name, opt, E, D, G):
    z_p = torch.randn(36, opt.latent_dim)
    z_p = get_cuda(z_p)

    E.eval()
    G.eval()
    D.eval()

    with torch.autograd.no_grad():
        x_p = G(z_p)
    
    utils.save_image(x_p.cpu(), img_name, normalize=True, nrow=6)

def build_tensorboard():
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter()
    return writer

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
        os.makedirs(os.path.join(path, version, args.real_image))
        os.makedirs(os.path.join(path, version, args.generate_image))
        os.makedirs(os.path.join(path, version, args.recon_image))

def tensor2var(x, grad=False):
    '''
    put tensor to gpu, and set grad to false

    Args:
        x (tensor): input tensor
        grad (bool, optional):  Defaults to False.

    Returns:
        tensor: tensor in gpu and set grad to false 
    '''    
    if torch.cuda.is_available():
        x = x.cuda()
        x.requires_grad_(grad)
    return x