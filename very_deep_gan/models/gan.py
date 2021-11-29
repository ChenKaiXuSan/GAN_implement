# %%
'''
pure gan structure.
'''
import torch
from torch.functional import norm
import torch.nn as nn

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

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(self.z_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, self.channels),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)


        return img

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
            nn.LeakyReLU(0.2, inplace=True)
        )

        curr_dim = conv_dim
        # (*, 64, 32, 32)
        self.l2 = nn.Sequential(
            nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(curr_dim * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )

        curr_dim = curr_dim * 2
        # (*, 128, 16, 16)
        self.l3 = nn.Sequential(
            nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1),
            nn.BatchNorm2d(curr_dim * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        curr_dim = curr_dim * 2
        # (*, 256, 8, 8)
        self.l4 = nn.Sequential(
            nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1),
            nn.BatchNorm2d(curr_dim * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )

        curr_dim = curr_dim * 2
        
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