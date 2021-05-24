# %%
import sys
sys.path.append('..')
sys.path.append('.')

from utils.utils import tensor2var, weights_init
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

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
        self.conv = nn.ConvTranspose2d(channel_in, channel_out, kernel_size=5, padding=2, stride=2)
        self.bn = nn.BatchNorm2d(channel_out, momentum=0.9)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, ten):
        ten = self.conv(ten)
        ten = self.bn(ten)
        ten = self.relu(ten)
        return ten

# %%
class Encoder(nn.Module):
    '''
    encoder, for real image x 

    Args:
        nn ([type]): [description]
    '''    
    def __init__(self, channel_in=3, z_size=128):
        super(Encoder, self).__init__()
        
        self.size = channel_in
        layers_list = []

        # the first time 3 -> 64, for every other double the channel size
        for i in range(3):
            if i == 0:
                layers_list.append(EncoderBlock(channel_in=self.size, channel_out=64))
                self.size = 64
            else:
                layers_list.append(EncoderBlock(channel_in=self.size, channel_out=self.size * 2))
                self.size *= 2
        
        # fully-connected, final shape B, 256, 8, 8
        self.conv = nn.Sequential(*layers_list)
        self.fc = nn.Sequential(nn.Linear(in_features=8 * 8 * self.size, out_features=2048, bias=False),
                                nn.BatchNorm1d(num_features=2048, momentum=0.9),
                                nn.LeakyReLU(0.2)
                                )
        
        # two linear to get the mu vector and the diagonal of the log_variance
        self.fc_mean = nn.Linear(in_features=2048, out_features=z_size)
        self.fc_logvar = nn.Linear(in_features=2048, out_features=z_size)

    def forward(self, ten):
        ten = self.conv(ten)
        batch_size = ten.size()[0]
        ten = ten.view(batch_size, -1)
        ten = self.fc(ten)

        # get the mean and logvar for loss
        mean = self.fc_mean(ten)
        logvar = self.fc_logvar(ten)

        return mean, logvar

    def __call__(self, *args, **kwargs):
        return super(Encoder, self).__call__(*args, **kwargs)

class Decoder(nn.Module):
    '''
    decoder and generator

    Args:
        nn ([type]): [description]
    '''    
    def __init__(self, z_size, size, channel_in=3):
        super(Decoder, self).__init__()
        # start from B, z_size
        self.fc = nn.Sequential(
            nn.Linear(in_features=z_size, out_features=8 * 8 * size),
            nn.BatchNorm1d(num_features=8 * 8 * size, momentum=0.9),
            nn.LeakyReLU(0.2),
        )
        self.size = size
        self.channels = channel_in

        layers_list = []
        layers_list.append(DecoderBlock(channel_in=self.size, channel_out=self.size))
        layers_list.append(DecoderBlock(channel_in=self.size, channel_out=self.size // 2))

        self.size = self.size // 2

        layers_list.append(DecoderBlock(channel_in=self.size, channel_out=self.size // 4))

        self.size = self.size // 4

        # final conv to get channels and tanh layer
        layers_list.append(nn.Sequential(
            nn.Conv2d(in_channels=self.size, out_channels=self.channels, kernel_size=5, stride=1, padding=2),
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
    def __init__(self, channel_in=3):
        super(Discriminator, self).__init__()

        self.size = channel_in

        # module list because we need to extract an intermediate output 
        self.conv = nn.ModuleList()
        self.conv.append(nn.Sequential(
            nn.Conv2d(in_channels=self.size, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(0.2)
        ))
        self.size = 32
        self.conv.append(EncoderBlock(channel_in=self.size, channel_out=128))
        self.size = 128
        self.conv.append(EncoderBlock(channel_in=self.size, channel_out=256))
        self.size = 256
        self.conv.append(EncoderBlock(channel_in=self.size, channel_out=256))

        # final fc to get the score (real or fake)
        self.fc = nn.Sequential(
            nn.Linear(in_features=8 * 8 * self.size, out_features=512),
            nn.BatchNorm1d(num_features=512, momentum=0.9),
            nn.LeakyReLU(0.2)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(512, 1), # todo
            nn.Sigmoid()
        )

    def forward(self, ten):
        batch_size = ten.size()[0]
        
        for i, lay in enumerate(self.conv):
            ten = lay(ten)

        ten = ten.view(batch_size, -1)
        ten1 = ten;

        ten = self.fc(ten)
        ten = self.fc2(ten)

        return ten, ten1

    def __call__(self, *args, **kwargs):
        return super(Discriminator, self).__call__(*args, **kwargs)

# %%
class VaeGan(nn.Module):
    def __init__(self, z_size=128, channels_in=3):
        super(VaeGan, self).__init__()
        # latent space size
        self.z_size = z_size
        self.channels = channels_in

        self.encoder = Encoder(z_size=self.z_size, channel_in=self.channels)
        self.decoder = Decoder(z_size=self.z_size, size=self.encoder.size, channel_in=self.channels)
        self.discriminator = Discriminator(channel_in=self.channels)

        # init the parameters, like next function
        self.encoder.apply(weights_init)
        self.decoder.apply(weights_init)
        self.discriminator.apply(weights_init)

        # self-defined function to init the parameters
        # self.init_parameters()

    def init_parameters(self):
        # just explore the network, find every weight and bias matrix and fill it 
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                if hasattr(m, "weight") and m.weight is not None and m.weight.requires_grad:
                    # init as original implementation
                    scale = 1.0 / np.sqrt(np.prod(m.weight.shape[1:]))
                    scale /= np.sqrt(3)
                    
                    nn.init.uniform_(m.weight, -scale, scale)
                if hasattr(m, "bias") and m.bias is not None and m.bias.requires_grad:
                    nn.init.constant_(m.bias, 0.0)

    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        '''
        reparameterization trick to sample from N(mu, var) from N(0,1)

        Args:
            mu (Tensor): mean of the latent Gaussian [B x D]
            logvar (Tensor): standard deviation of the latent Gaussian [B x D]

        Returns:
            [Tensor]: B x D
        '''        
        std = logvar.mul(0.5).exp_()

        # epsilon, for z_noise
        # eps = Variable(std.data.new(std.size()).normal_())
        # eps = torch.randn_like(std)
        eps = eps
        
        out = eps * std + mean
        return out

    def forward(self, x, gen_size=10):
        batch_size = x.size()[0]

        z_mean, z_logvar = self.encoder(x)

        # sampling epsilon from normal distribution
        epsilon = tensor2var(torch.randn(batch_size, self.z_size)) # just sample and decode

        z = self.reparameterize(mean= z_mean, logvar=z_logvar, eps = epsilon)

        x_tilda = self.decoder(z)

        return z_mean, z_logvar, x_tilda
    
    def __call__(self, *args, **kwargs):
        return super(VaeGan, self).__call__(*args, **kwargs)

# %%
if __name__ == "__main__":
    # init net work
    generator = VaeGan(z_size=args.z_size, channels_in=args.channels).cuda()
    discriminator = Discriminator(channel_in=args.channels).cuda()

    print(generator)
    print(discriminator)

    datav = tensor2var(torch.randn(64, 1, 64, 64))
    mean, logvar, rec_enc = generator(datav)
    print(mean, logvar, rec_enc)
