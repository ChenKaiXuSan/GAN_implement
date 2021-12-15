# %%
'''
pure dcgan structure.
code similar sample from the pytorch code.
https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
'''
import torch
import torch.nn as nn
from torch.nn.modules.conv import Conv2d
import torch.nn.utils.spectral_norm as spectral_norm

import numpy as np

# %% 
def conv1x1(in_channels, out_channels): # not change resolusion
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)

def conv3x3(in_channels, out_channels): # not chagne resolusion
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, bias=False)
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
            nn.ConvTranspose2d(self.z_dim, conv_dim * mult * 2, 2, 1, 0, bias=False),
            nn.BatchNorm2d(conv_dim * mult * 2),
            nn.ReLU(True)
        )

        curr_dim = conv_dim * mult * 2

        self.l2 = nn.Sequential(
            nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1, bias=False),
            nn.BatchNorm2d(int(curr_dim / 2)),
            nn.ReLU(True)
        )

        curr_dim = int(curr_dim / 2)

        self.l3 = nn.Sequential(
            nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1, bias=False),
            nn.BatchNorm2d(int(curr_dim / 2)),
            nn.ReLU(True),
        )

        curr_dim = int(curr_dim / 2)

        self.l4 = nn.Sequential(
            nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1, bias=False),
            nn.BatchNorm2d(int(curr_dim / 2)),
            nn.ReLU(True)
        )
        
        curr_dim = int(curr_dim / 2)

        self.l5 = nn.Sequential(
            nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1, bias=False),
            nn.BatchNorm2d(int(curr_dim / 2)),
            nn.ReLU(True)
        )
        
        curr_dim = int(curr_dim / 2)

        self.last = nn.Sequential(
            nn.ConvTranspose2d(curr_dim, self.channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z, feat):
        mix_z = torch.matmul(z, feat[5])
        out = self.l1(mix_z) # (*, 1024, 2, 2)
        # mix_out_1 = torch.matmul(out, feat[4])
        out = self.l2(out) # (*, 512, 4, 4)
        # mix_out_2 = torch.matmul(out, feat[3])
        out = self.l3(out)
        # mix_out_3 = torch.matmul(out, feat[2])
        out = self.l4(out) # (*, 64, 32, 32)
        # mix_out_4 = torch.matmul(out, feat[1])
        out = self.l5(out)
        # mix_out_5 = torch.matmul(out, feat[0])

        out = self.last(out) # (*, 1, 64, 64)

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
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.f1 = nn.Sequential(
            spectral_norm(
            conv1x1(conv_dim, conv_dim)
            ),
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

        self.f2 = nn.Sequential(
            spectral_norm(
                conv1x1(curr_dim, curr_dim)
            ), 
            nn.LeakyReLU(0.2, inplace=True)
        )

        # (*, 128, 16, 16)
        self.l3 = nn.Sequential(
            nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1),
            nn.BatchNorm2d(curr_dim * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        curr_dim = curr_dim * 2

        self.f3 = nn.Sequential(
            spectral_norm(
                conv1x1(curr_dim, curr_dim)
            ),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # (*, 256, 8, 8)
        self.l4 = nn.Sequential(
            nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1),
            nn.BatchNorm2d(curr_dim * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )

        curr_dim = curr_dim * 2

        self.f4 = nn.Sequential(
            spectral_norm(
                conv1x1(curr_dim, curr_dim)
            ),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.l5 = nn.Sequential(
            nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1),
            nn.BatchNorm2d(curr_dim * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )

        curr_dim = curr_dim * 2 

        self.f5 = nn.Sequential(
            spectral_norm(
                conv1x1(curr_dim, curr_dim)
            ),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # output layers
        # (*, 512, 4, 4)
        self.last_adv = nn.Sequential(
            nn.Conv2d(curr_dim, 1, 2, 1, 0, bias=False),
            # without sigmoid, used in the loss funciton
            )

    def forward(self, x):
        feat = [] 

        out = self.l1(x) # (*, 64, 32, 32)
        feat1 = self.f1(out) # (*, 64, 32, 32)
        feat.append(feat1) 

        out = self.l2(out) # (*, 128, 16, 16)
        feat2 = self.f2(out) # (*, 128, 16, 16)
        feat.append(feat2)

        out = self.l3(out) # (*, 256, 8, 8)
        feat3 = self.f3(out) # (*, 256, 8, 8)
        feat.append(feat3)

        out = self.l4(out) # (*, 512, 4, 4)
        feat4 = self.f4(out) # (*, 512, 4, 4)
        feat.append(feat4)
        
        out = self.l5(out) # (*, 1024, 2, 2)
        feat5 = self.f5(out) # (*, 1024, 2, 2)
        feat.append(feat5)

        validity = self.last_adv(out) # (*, 1, 1, 1)
        feat.append(validity)

        return validity.squeeze(), feat