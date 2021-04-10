# %%
from numpy.lib import utils
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append('../..')
sys.path.append('./')

from torch.autograd import Variable
from self_attention_gan.models.spectral import SpectralNorm
import numpy as np

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
# %%
def fill_labels(img_size):
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
        self.label_emb = nn.Embedding(n_classes, n_classes, 1, 1).cuda()

        layer1 = []
        layer2 = []
        layer3 = []
        last = []

        repeat_num = int(np.log2(self.imsize)) - 3  # 3
        mult = 2 ** repeat_num  # 8
        layer1.append(SpectralNorm(
            nn.ConvTranspose2d(z_dim + self.n_classes, conv_dim * mult, 4)))
        layer1.append(nn.BatchNorm2d(conv_dim * mult))
        layer1.append(nn.ReLU())

        curr_dim = conv_dim * mult

        layer2.append(SpectralNorm(nn.ConvTranspose2d(
            curr_dim, int(curr_dim / 2), 4, 2, 1)))
        layer2.append(nn.BatchNorm2d(int(curr_dim / 2)))
        layer2.append(nn.ReLU())

        curr_dim = int(curr_dim / 2)

        layer3.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1)))
        layer3.append(nn.BatchNorm2d(int(curr_dim / 2)))
        layer3.append(nn.ReLU())

        if self.imsize == 64:
            layer4 = []
            curr_dim = int(curr_dim / 2)
            layer4.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1)))
            layer4.append(nn.BatchNorm2d(int(curr_dim / 2)))
            layer4.append(nn.ReLU())
            self.l4 = nn.Sequential(*layer4)
            curr_dim = int(curr_dim / 2)

        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)

        last.append(nn.ConvTranspose2d(curr_dim, self.channels, 4, 2, 1))
        last.append(nn.Tanh())
        self.last = nn.Sequential(*last)

        self.attn1 = Self_Attn(128, 'relu')
        self.attn2 = Self_Attn(64, 'relu')

    def forward(self, z, labels):

        labels_emb = self.label_emb(labels).cuda()

        z = z.view(z.size(0), z.size(1))
        
        input = torch.cat((labels_emb, z), 1)
        input = input.view(input.size(0), input.size(1), 1, 1)
        out = self.l1(input)
        out = self.l2(out)
        out = self.l3(out)
        out, p1 = self.attn1(out)
        out = self.l4(out)
        out, p2 = self.attn2(out)
        out = self.last(out)

        return out, p1, p2


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
        last = []

        layer1.append(SpectralNorm(nn.Conv2d(self.channels + self.n_classes , conv_dim, 4, 2, 1)))
        layer1.append(nn.LeakyReLU(0.1))

        curr_dim = conv_dim

        layer2.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer2.append(nn.LeakyReLU(0.1))

        curr_dim = curr_dim * 2
        
        layer3.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer3.append(nn.LeakyReLU(0.1))
        
        curr_dim = curr_dim * 2

        if self.imsize == 64:
            layer4 = []
            layer4.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
            layer4.append(nn.LeakyReLU(0.1))
            self.l4 = nn.Sequential(*layer4)
            curr_dim = curr_dim * 2
        
        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)

        last.append(nn.Conv2d(curr_dim, 1, 4))
        self.last = nn.Sequential(*last)

        self.attn1 = Self_Attn(256, 'relu')
        self.attn2 = Self_Attn(512, 'relu')

    def forward(self, x, labels):
        labels_fill = fill_labels(self.imsize)[labels] # 10, 10, img_size, img_size torch.cuda.floattensor
        input = torch.cat((x, labels_fill), 1) # torch.cuda.floattensor
        out = self.l1(input)
        out = self.l2(out)
        out = self.l3(out)
        out, p1 = self.attn1(out)
        out = self.l4(out)
        out, p2 = self.attn2(out)
        out = self.last(out)

        return out.squeeze(), p1, p2

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