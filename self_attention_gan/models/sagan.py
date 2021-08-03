# %%
from numpy.lib import utils
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append('../..')
sys.path.append('./')

from torch.autograd import Variable
import numpy as np

from .spectral import SpectralNorm
from .attention import *

# %%
def fill_labels(img_size):
    '''
    for D fill labels

    Args:
        img_size (int): image size

    Returns:
        tensor: filled type
    '''    
    fill = torch.zeros([10, 10, img_size, img_size])
    for i in range(10):
        fill[i, i, :, :] = 1
    return fill.cuda()

# %%
class Generator(nn.Module):
    '''
    Generator

    '''
    def __init__(self, batch_size, image_size=64, z_dim=100, conv_dim=64, channels = 1, n_classes = 10):
        
        super(Generator, self).__init__()
        self.imsize = image_size
        self.channels = channels
        self.n_classes = n_classes
        self.label_emb = nn.Embedding(n_classes, z_dim).cuda()

        layer1 = []
        layer2 = []
        layer3 = []
        last = []

        repeat_num = int(np.log2(self.imsize)) - 3  # 3
        mult = 2 ** repeat_num  # 8
        layer1.append(SpectralNorm(
            nn.ConvTranspose2d(z_dim, conv_dim * mult, 4, 1, 0, bias=False)))
        layer1.append(nn.BatchNorm2d(conv_dim * mult))
        layer1.append(nn.ReLU())

        curr_dim = conv_dim * mult

        layer2.append(SpectralNorm(
            nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1, bias = False)))
        layer2.append(nn.BatchNorm2d(int(curr_dim / 2)))
        layer2.append(nn.ReLU())

        curr_dim = int(curr_dim / 2)

        layer3.append(SpectralNorm(
            nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1, bias=False)))
        layer3.append(nn.BatchNorm2d(int(curr_dim / 2)))
        layer3.append(nn.ReLU())

        if self.imsize == 64:
            layer4 = []
            curr_dim = int(curr_dim / 2)
            layer4.append(SpectralNorm(
                nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1, bias=False)))
            layer4.append(nn.BatchNorm2d(int(curr_dim / 2)))
            layer4.append(nn.ReLU())
            self.l4 = nn.Sequential(*layer4)
        curr_dim = int(curr_dim / 2)

        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)

        last.append(nn.ConvTranspose2d(curr_dim, self.channels, 4, 2, 1, bias=False))
        last.append(nn.Tanh())
        self.last = nn.Sequential(*last)

        self.attn1 = Self_Attn(128, 'relu')
        self.attn2 = Self_Attn(64, 'relu')

    def forward(self, z, labels):

        labels_emb = self.label_emb(labels).cuda()

        input = torch.mul(labels_emb, z)
        # input = torch.cat((labels_emb, z), 1)

        input = input.view(input.size(0), input.size(1), 1, 1)
        out = self.l1(input)
        out = self.l2(out)
        out = self.l3(out)

        # out, p = self.attn1(out)
        out = self.l4(out)
        # out, p = self.attn2(out)

        out = self.last(out)

        return out, out


# %%
class Discriminator(nn.Module):
    '''
    discriminator, Auxiliary classifier

    '''
    def __init__(self, batch_size, n_classes = 10, image_size = 64, conv_dim = 64, channels = 1):
        super(Discriminator, self).__init__()
        self.imsize = image_size
        self.channels = channels
        self.n_classes = n_classes
        layer1 = []
        layer2 = []
        layer3 = []
        last_adv_layer = []
        last_aux_layer = []

        layer1.append(SpectralNorm(
            nn.Conv2d(self.n_classes, conv_dim, 4, 2, 1, bias=False)))
        layer1.append(nn.LeakyReLU(0.1))

        curr_dim = conv_dim

        layer2.append(SpectralNorm(
            nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1, bias=False)))
        # layer2.append(nn.BatchNorm2d(curr_dim * 2, ))
        layer2.append(nn.LeakyReLU(0.1))

        curr_dim = curr_dim * 2
        
        layer3.append(SpectralNorm(
            nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1, bias=False)))
        # layer3.append(nn.BatchNorm2d(curr_dim * 2))
        layer3.append(nn.LeakyReLU(0.1))
        
        curr_dim = curr_dim * 2

        if self.imsize == 64:
            layer4 = []
            layer4.append(SpectralNorm(
                nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1, bias=False)))
            # layer4.append(nn.BatchNorm2d(curr_dim * 2))
            layer4.append(nn.LeakyReLU(0.1))
            self.l4 = nn.Sequential(*layer4)
            curr_dim = curr_dim * 2
        
        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)

        # output layers
        last_adv_layer.append(nn.Conv2d(curr_dim, 1, 4, 1, 0, bias=False))
        last_aux_layer.append(nn.Conv2d(curr_dim, self.n_classes, 4, 1, 0, bias=False))

        self.last_adv = nn.Sequential(*last_adv_layer)
        self.last_aux = nn.Sequential(*last_aux_layer)

        self.attn1 = Self_Attn(256, 'relu')
        self.attn2 = Self_Attn(512, 'relu')

    def forward(self, x, labels):
        labels_fill = fill_labels(self.imsize)[labels] # 10, 10, img_size, img_size torch.cuda.floattensor
        # input = torch.cat((x, labels_fill), 1) # torch.cuda.floattensor
        input = torch.mul(x, labels_fill)

        out = self.l1(input)
        out = self.l2(out)
        out = self.l3(out)
        # out, p = self.attn1(out)

        out = self.l4(out)
        # out, p = self.attn2(out)

        valiidity = self.last_adv(out)
        label = self.last_aux(out)

        return valiidity.squeeze(), label.squeeze()

# %% 
if __name__ == '__main__':
    genertor = Generator(64, image_size=64)
    discriminator = Discriminator(64)

    # print(genertor, discriminator)

    image = torch.randn(64, 1, 64, 64)
    image = image.cuda()

    z = torch.randn(64, 128)
    z = z.cuda()

    label = torch.randn(64)
    label = label.type(torch.LongTensor)

    y, p1, p2 = discriminator(image, label)

    x, att1, att2 = genertor(z, label)
    print(x.shape, att1.shape, att2.shape)

# %%
