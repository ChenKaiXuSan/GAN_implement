# %%
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
    sampled_z = Variable(Tensor(np.random.normal(0, 1, (mu.size(0), args.latent_dim))))
    z = sampled_z * std + mu
    return z 
    
# %%
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
img_shape = (args.channels, args.img_size, args.img_size)
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        def encoder_block(in_filters, out_filters):
            block = [nn.Conv2d(in_filters, out_filters, 4, 2, 1),
                    nn.BatchNorm2d(out_filters, 0.9),
                    nn.LeakyReLU(0.2)]
            return block

        self.model = nn.Sequential(
            *encoder_block(args.channels, 64),
            *encoder_block(64, 128),
            *encoder_block(128, 256),
            *encoder_block(256, 512),
            # *encoder_block(512, 1024)
        )

        self.relu = nn.LeakyReLU(0.2)

        self.fc1 = nn.Linear(512 * 4 * 4, 1024)
        self.bn_fc = nn.BatchNorm1d(1024, momentum=0.9)
        self.fc3 = nn.Linear(1024, 512)
        self.bn_fc2= nn.BatchNorm1d(512, momentum=0.9)
        
        self.fc_mean = nn.Linear(512, 128)
        self.fc_logvar = nn.Linear(512, 128)

    def forward(self, img):
        batch_size = int(img.size()[0])

        out = self.model(img)

        out_flat = out.view(batch_size, -1)

        out = self.relu(self.bn_fc(self.fc1(out_flat)))
        out = self.relu(self.bn_fc2(self.fc3(out))) # 512

        mean = self.fc_mean(out)
        logvar = self.fc_logvar(out)
        z = reparameterization(mean, logvar)
        return z

# %%
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.fc1 = nn.Linear(128, 4 * 4 * 512)
        self.bn1 = nn.BatchNorm1d(4 * 4 * 512, momentum=0.9)

        self.relu = nn.LeakyReLU(0.2)

    
        def decoder_block(in_filters, out_filters):
            block = [nn.ConvTranspose2d(in_filters, out_filters, 4, 2, 1),
                    nn.BatchNorm2d(out_filters, 0.9),
                    nn.LeakyReLU(0.2)
                    ]
            return block

        self.model = nn.Sequential(
            *decoder_block(512, 256),
            *decoder_block(256, 128),
            *decoder_block(128, 64),
            *decoder_block(64, 32),
        )
        
        self.deconv4 = nn.ConvTranspose2d(32, args.channels, kernel_size=5, stride=1, padding=2)
        self.tanh = nn.Tanh()

        self.fc_last = nn.Linear(args.channels * 64 * 64 , 128 )
        self.bn_last = nn.BatchNorm1d(128, momentum=0.9)

    def forward(self, z):
        img = self.relu(self.bn1(self.fc1(z)))
        img = img.view(z.size()[0], 512, 4, 4)

        img = self.model(img)

        img = self.tanh(self.deconv4(img))

        return img
# %% 
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.fc = nn.Linear(128, 64 * 64 * args.channels)
        self.bn = nn.BatchNorm1d(64 * 64 * args.channels, momentum=0.9)

        def discriminator_block(in_filters, out_filters):
            block = [nn.Conv2d(in_filters, out_filters, 4, 2, 1),
                    nn.BatchNorm2d(out_filters, 0.9),
                    nn.LeakyReLU(0.2)]
            return block

        self.model = nn.Sequential(
            *discriminator_block(args.channels, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
        )

        self.fc = nn.Linear(2 * 2 * 512, 1024)
        self.bn4 = nn.BatchNorm1d(1024, momentum=0.9)

        self.fc2 = nn.Linear(1024, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, img):
        # img = self.relu(self.bn(self.fc(z)))
        # img = img.view(z.size()[0], args.channels, 64, 64)

        img = self.model(img)

        img = img.view(img.size(0), 2 * 2 * 512)
        img = self.bn4(self.fc(img))
        img = self.sigmoid(self.fc2(img))

        return img

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
