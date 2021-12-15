# %%
'''
code similar sample from the pytorch code, and with the spectral normalization.
https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

SAGAN implement.
with full attention in every layer in the generator.
And with the instance normalization instead of the batch normal.
'''
import torch.nn as nn
from torch.nn.utils import spectral_norm
import torch.nn.functional as F

from models.attention import Attention

import numpy as np
# %%
class Generator(nn.Module):
    '''
    pure Generator structure

    '''    
    def __init__(self, image_size=64, z_dim=100, conv_dim=64, channels = 1):
        
        super(Generator, self).__init__()
        self.imsize = image_size
        self.channels = channels
        self.z_dim = z_dim

        repeat_num = int(np.log2(self.imsize)) - 3  # 3
        mult = 2 ** repeat_num  # 8

        self.l1 = nn.Sequential(
            # input is Z, going into a convolution.
            nn.ConvTranspose2d(self.z_dim, conv_dim * mult, 4, 1, 0, bias=False),
            nn.InstanceNorm2d(conv_dim * mult),
            nn.ReLU(True)
        )

        curr_dim = conv_dim * mult

        self.l2 = nn.Sequential(
            nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1, bias=False),
            nn.InstanceNorm2d(conv_dim * mult),
            nn.ReLU(True)
        )

        curr_dim = int(curr_dim / 2)

        self.l3 = nn.Sequential(
            nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1, bias=False),
            nn.InstanceNorm2d(conv_dim * mult),
            nn.ReLU(True),
        )

        curr_dim = int(curr_dim / 2)

        self.l4 = nn.Sequential(
            nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1, bias=False),
            nn.InstanceNorm2d(conv_dim * mult),
            nn.ReLU(True)
        )
        
        curr_dim = int(curr_dim / 2)
        
        self.last = nn.Sequential(
            nn.ConvTranspose2d(curr_dim, self.channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

        self.attn1 = Attention(512)
        self.attn2 = Attention(256)
        self.attn3 = Attention(128)
        self.attn4 = Attention(64)

    def forward(self, z):
        z = z.view(z.size(0), -1, 1, 1) # (*, 100, 1, 1)

        out = self.l1(z) # (*, 512, 4, 4)
        attn1 = self.attn1(out)
        out = self.l2(attn1) # (*, 256, 8, 8)
        attn2 = self.attn2(out)
        out = self.l3(attn2) # (*, 128, 16, 16)
        attn3 = self.attn3(out)
        out = self.l4(attn3) # (*, 64, 32, 32)
        attn4 = self.attn4(out)

        out = self.last(attn4) # (*, c, 64, 64)

        return out

# %%
class Discriminator(nn.Module):
    '''
    pure discriminator structure

    '''
    def __init__(self, image_size = 64, conv_dim = 64, channels = 1):
        super(Discriminator, self).__init__()
        self.imsize = image_size
        self.channels = channels

        # (*, 1, 64, 64)
        self.l1 = nn.Sequential(
            nn.Conv2d(self.channels, conv_dim, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(conv_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )

        curr_dim = conv_dim
        # (*, 64, 32, 32)
        self.l2 = nn.Sequential(
            nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(curr_dim * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )

        curr_dim = curr_dim * 2
        # (*, 128, 16, 16)
        self.l3 = nn.Sequential(
            nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(curr_dim * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        curr_dim = curr_dim * 2
        # (*, 256, 8, 8)
        self.l4 = nn.Sequential(
            nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(curr_dim * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )

        curr_dim = curr_dim * 2
        
        # self.attn1 = Attention(64)
        # self.attn2 = Attention(128)
        # self.attn3 = Attention(256)
        # self.attn4 = Attention(512)

        # output layers
        # (*, 512, 4, 4)
        self.last_adv = nn.Sequential(
            nn.Conv2d(curr_dim, 1, 4, 1, 0, bias=False),
            # without sigmoid, used in the loss funciton
            )

    def forward(self, x):
        out = self.l1(x) # (*, 64, 32, 32)
        out = self.l2(out) # (*, 128, 16, 16)
        out = self.l3(out) # (*, 256, 8, 8)
        out = self.l4(out) # (*, 512, 4, 4)

        validity = self.last_adv(out) # (*, 1, 1, 1)

        return validity.squeeze()