# %%
from operator import truediv
import numpy as np
from numpy.lib.arraypad import pad

import torchvision.transforms as transforms
from torchvision.utils import save_image

import torch.nn as nn 
import torch.nn.functional as F
import torch
from torch.autograd import Variable

from options import args
# %%
def reparameterization(mu, logvar):
    std = torch.exp(logvar / 2)
    sampled_z = Variable(torch.randn_like(std))
    z = sampled_z * std + mu
    return z 
    
# %%
cuda = True if torch.cuda.is_available() else False
img_shape = (args.channels, args.img_size, args.img_size)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.fc_mean = nn.Linear(512, 128)
        self.fc_logvar = nn.Linear(512, 128)

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)

        img = self.model(img_flat)

        mean = self.fc_mean(img)
        logvar = self.fc_logvar(img)
        z = reparameterization(mean, logvar)

        return z

# %%
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.label_emb = nn.Embedding(args.n_classes, args.latent_dim)
        
        self.model = nn.Sequential(
            nn.Linear(args.latent_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        gen_input = torch.mul(self.label_emb(labels), noise)

        img_flat = self.model(gen_input)
        img = img_flat.view(img_flat.shape[0], *img_shape)

        return img
# %% 
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.fc = nn.Linear(64 * 64, 128)

        self.model = nn.Sequential(
            nn.Linear(args.latent_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Linear(256, 1),
            # nn.Sigmoid(),
        )

        # output layers 
        self.adv_layer = nn.Sequential(nn.Linear(256, 1), nn.Sigmoid())
        self.aux_layer = nn.Sequential(nn.Linear(256, args.n_classes), nn.Softmax())

    def forward(self, img):
        if len(img.shape) == 4:
            img = img.view(img.size(0), -1)
            img = self.fc(img)

        out = self.model(img)

        validity = self.adv_layer(out)
        label = self.aux_layer(out)

        return validity, label

# %% 
class Self_Attn(nn.Module):
    '''
    self attention layer

    Args:
        nn (father): father class
    '''

    def __init__(self, in_dim, activation):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        '''
        input:
            x: input feature maps(b, x, w, h)

        Returns:
            out: self attention value + input feature
            attention: b, n, n (n is width*height)
        '''
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(
            m_batchsize, -1, width*height).permute(0, 2, 1)  # b, (w*h), c
        proj_key = self.key_conv(x).view(
            m_batchsize, -1, width*height)  # b, c, (w*h)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # x, n, n
        proj_value = self.value_conv(x).view(
            m_batchsize, -1, width*height)  # b, c, n

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out, attention
