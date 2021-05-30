# %% 
import sys
from utils.utils import tensor2var, weights_init 
sys.path.append('..')
sys.path.append('.')

import torch
import torch.nn as nn
from torch.autograd import Variable

from options import args

# %%
class EncoderBlock(nn.Module):
    '''
    encoder block used in encoder
    '''
    def __init__(self, channel_in, channel_out):
        super(EncoderBlock, self).__init__()
        # convolution to halve the dimensions
        self.conv = nn.Conv2d(in_channels=channel_in, out_channels=channel_out, kernel_size=5, padding=2, stride=2)
        self.bn = nn.BatchNorm2d(num_features=channel_out, momentum=0.9)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, ten):

        ten = self.conv(ten)
        ten = self.bn(ten)
        ten = self.relu(ten)

        return ten

class DecoderBlock(nn.Module):
    '''
    decoder block used in decoder

    '''    
    def __init__(self, channel_in, channel_out):
        super(DecoderBlock, self).__init__()
        # transpose convolution to double the dimensions
        self.conv = nn.ConvTranspose2d(channel_in, channel_out, kernel_size=6, padding=2, stride=2)
        self.bn = nn.BatchNorm2d(channel_out, momentum=0.9)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, ten):
        ten = self.conv(ten)
        ten = self.bn(ten)
        ten = self.relu(ten)
        return ten

# %%
class Encoder(nn.Module):
    def __init__(self, channel_in = 3, z_size = 128):
        super(Encoder, self).__init__()

        self.size = channel_in
        layers_list = []

        layers_list.append(EncoderBlock(channel_in=self.size, channel_out=64))
        self.size = 64
        layers_list.append(EncoderBlock(channel_in=self.size, channel_out=self.size * 2))
        self.size = self.size * 2 # 128
        layers_list.append(EncoderBlock(channel_in=self.size, channel_out=self.size * 2))
        self.size = self.size * 2 # 256

        self.conv = nn.Sequential(*layers_list)

        # fully-connected, final shape, 8, 256, 8, 8
        self.fc = nn.Sequential(
            nn.Linear(in_features=8 * 8 * self.size, out_features=2048, bias=False),
            nn.BatchNorm1d(num_features=2048, momentum=0.9),
            nn.LeakyReLU(negative_slope=0.2)
        )

        # two linear to get the mu ans log_variance
        self.fc_mean = nn.Linear(in_features=2048, out_features=z_size)
        self.fc_logvar = nn.Linear(in_features=2048, out_features=z_size)

    def forward(self, ten):
        batch_size = ten.size()[0]

        ten = self.conv(ten)
        ten = ten.view(batch_size, -1)

        ten = self.fc(ten)

        # get the mean and logvar for loss 
        mean = self.fc_mean(ten)
        logvar = self.fc_logvar(ten)

        return mean, logvar

    def __call__(self, *args, **kwargs):
        return super(Encoder, self).__call__(*args, **kwargs)

# %%
class Decoder(nn.Module):
    def __init__(self, z_size, size, channel_in = 3):
        super(Decoder, self).__init__()

        # start from B, z_size 
        self.fc = nn.Sequential(
            nn.Linear(in_features=z_size, out_features=8 * 8 * size),
            nn.BatchNorm1d(num_features=8 * 8 * size, momentum=0.9),
            nn.LeakyReLU(0.2),
        )

        self.size = size # 256
        self.channels = channel_in

        layers_list = []
        layers_list.append(DecoderBlock(channel_in=self.size, channel_out=self.size))
        layers_list.append(DecoderBlock(channel_in=self.size, channel_out=self.size // 2))

        self.size = self.size // 2 # 128

        layers_list.append(DecoderBlock(channel_in=self.size, channel_out=self.size // 4))

        self.size = self.size // 4 # 32

        # final conv to get channels and tanh layer
        layers_list.append(nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.size, out_channels=self.channels, kernel_size=5, stride=1, padding=2),
            nn.Tanh()
        ))

        self.conv = nn.Sequential(*layers_list)

    def forward(self, ten):
        batch_size = ten.size()[0]

        ten = self.fc(ten)
        ten = ten.view(batch_size, -1, 8, 8)
        ten = self.conv(ten)

        return ten
    
    def __call__(self, *args, **kwargs):
        return super(Decoder, self).__call__(*args, **kwargs)

        
# %%
class Discriminator(nn.Module):
    def __init__(self, channel_in = 3):
        super(Discriminator, self).__init__()

        self.size = channel_in

        layers_list = []
        layers_list.append(nn.Sequential(
            nn.Conv2d(in_channels=self.size, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(negative_slope=0.2),
        ))
        self.size = 32

        layers_list.append(EncoderBlock(channel_in=self.size, channel_out=128))
        self.size = 128
        layers_list.append(EncoderBlock(channel_in=self.size, channel_out=256))
        self.size = 256
        layers_list.append(EncoderBlock(channel_in=self.size, channel_out=256))

        self.conv = nn.Sequential(*layers_list)

        # final fc to get the score (real or fake)
        self.fc = nn.Sequential(
            nn.Linear(in_features=8 * 8 * self.size, out_features=512),
            nn.BatchNorm1d(num_features=512, momentum=0.9),
            nn.LeakyReLU(0.2)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, ten):
        batch_size = ten.size()[0]

        ten = self.conv(ten)

        ten = ten.view(batch_size, -1)
        ten1 = ten;

        ten = self.fc(ten)
        ten = self.fc2(ten)

        return ten, ten1

    def __call__(self, *args, **kwargs):
        return super(Discriminator, self).__call__(*args, **kwargs)

# %%
class VaeGan(nn.Module):
    def __init__(self, z_size = 128, channels_in = 3):
        super(VaeGan, self).__init__()
        # latent space size 
        self.z_size = z_size
        self.channels = channels_in

        # init network
        self.encoder = Encoder(channel_in=self.channels, z_size=self.z_size)
        self.decoder = Decoder(z_size=self.z_size, size=self.encoder.size, channel_in=self.channels)
        self.discriminator = Discriminator(channel_in=self.channels)

        # init the parameters
        self.encoder.apply(weights_init)
        self.decoder.apply(weights_init)
        self.discriminator.apply(weights_init)

    def reparameterize(self, mean:torch.Tensor, logvar: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:

        std = logvar.mul(0.5).exp_()

        eps = eps

        out = eps * std + mean
        return out

    def forward(self, x):
        batch_size = x.size()[0]

        z_mean, z_logvar = self.encoder(x)

        # sampling epsilon from normal distribution
        epsilon = tensor2var(torch.randn(batch_size, self.z_size))

        z = self.reparameterize(mean=z_mean, logvar=z_logvar, eps=epsilon)

        x_tilda = self.decoder(z)

        return z_mean, z_logvar, x_tilda

    def __call__(self, *args, **kwargs):
        return super(VaeGan, self).__call__(*args, **kwargs)