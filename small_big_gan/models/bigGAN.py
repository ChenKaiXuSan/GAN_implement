import torch
import torch.nn as nn 
import torch.nn.functional as F

import numpy as np 

def conv3x3(in_channels, out_channels): # not change resolusion
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)

def conv1x1(in_channels, out_channels): # not change resolusion
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)

def init_weight(m):
    classname = m.__class__.__name__

    if classname.find('Conv') != -1:
        nn.init.orthogonal_(m.weight, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    
    elif classname.find('Batch') != -1:
        m.weight.data.normal_(1, 0.02)
        m.bias.data.zero_()

    elif classname.find('Linear') != -1:
        nn.init.orthogonal_(m.weight, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()

    elif classname.find('Embedding') != -1:
        nn.init.orthogonal_(m.weight, gain=1)

class Attention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.theta = nn.utils.spectral_norm(conv1x1(channels, channels // 8)).apply(init_weight)
        self.phi = nn.utils.spectral_norm(conv1x1(channels, channels // 8)).apply(init_weight)

        self.g = nn.utils.spectral_norm(conv1x1(channels, channels // 2)).apply(init_weight)
        self.o = nn.utils.spectral_norm(conv1x1(channels // 2, channels)).apply(init_weight)

        self.gamma = nn.Parameter(torch.tensor(0.), requires_grad=True)

    def forward(self, inputs):
        batch, c, h, w = inputs.size()

        theta = self.theta(inputs) # (*, c/8, h, w)
        phi = F.max_pool2d(self.phi(inputs), [2, 2]) # (*, c/8, h/2, w/2)
        g = F.max_pool2d(self.g(inputs), [2, 2]) # (*, c/2, h/2, w/2)

        theta = theta.view(batch, self.channels // 8, -1) # (*, c/8, h*w)
        phi = phi.view(batch, self.channels // 8, -1) # (*, c/8, h*w/4)
        g = g.view(batch, self.channels // 2, -1) # (*, c/2, h*w/4)

        beta = F.softmax(torch.bmm(theta.transpose(1, 2), phi), -1) # (*, h*w, h*w/4)
        o = self.o(torch.bmm(g, beta.transpose(1, 2)).view(batch, self.channels//2, h, w)) # (*, c, h, w)

        return self.gamma * o + inputs

class ConditionalNorm(nn.Module):
    def __init__(self, in_channel, n_condition):
        super().__init__()

        self.bn = nn.BatchNorm2d(in_channel, affine=False) # no learning parameters
        self.embed = nn.Linear(n_condition, in_channel * 2)

        nn.init.orthogonal_(self.embed.weight.data[:, :in_channel], gain=1)

        self.embed.weight.data[:, in_channel:].zero_()

    def forward(self, inputs, label):
        out = self.bn(inputs)
        embed = self.embed(label.float())
        gamma, beta = embed.chunk(2, dim=1)

        gamma = gamma.unsqueeze(2).unsqueeze(3)
        beta = beta.unsqueeze(2).unsqueeze(3)

        out = gamma * out + beta

        return out

# BigGAN
class ResBlock_G(nn.Module):
    def __init__(self, in_channel, out_channel, condition_dim, upsample=True):
        super().__init__()

        self.cbn1 = ConditionalNorm(in_channel, condition_dim)

        self.upsample = nn.Sequential()
        if upsample:
            self.upsample.add_module('upsample', nn.Upsample(scale_factor=2, mode='nearest'))

        self.conv3x3_1 = nn.utils.spectral_norm(conv3x3(in_channel, out_channel)).apply(init_weight)
        self.cbn2 = ConditionalNorm(out_channel, condition_dim)
        self.conv3x3_2 = nn.utils.spectral_norm(conv3x3(out_channel, out_channel)).apply(init_weight)
        self.conv1x1 = nn.utils.spectral_norm(conv1x1(in_channel, out_channel)).apply(init_weight)

    def forward(self, inputs, condition):
        x = F.leaky_relu(self.cbn1(inputs, condition))
        x = self.upsample(x)
        x = self.conv3x3_1(x)
        x = self.conv3x3_2(F.leaky_relu(self.cbn2(x, condition)))
        x += self.conv1x1(self.upsample(inputs)) # shortcut
        return x 

class Generator(nn.Module):
    def __init__(self, n_feat, codes_dim=24, n_classes=10):
        super().__init__()
        self.fc = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(codes_dim, 16*n_feat*4*4)).apply(init_weight)
        )
        self.res1 = ResBlock_G(16*n_feat, 16*n_feat, codes_dim+n_classes, upsample=True)
        self.res2 = ResBlock_G(16*n_feat, 8*n_feat, codes_dim+n_classes, upsample=True)
        self.res3 = ResBlock_G(8*n_feat, 4*n_feat, codes_dim+n_classes, upsample=True)
        
        self.attn = Attention(4*n_feat)

        self.res4 = ResBlock_G(4*n_feat, 2*n_feat, codes_dim+n_classes, upsample=True)

        self.conv = nn.Sequential(
            nn.LeakyReLU(),
            nn.utils.spectral_norm(conv3x3(2*n_feat, 3)).apply(init_weight)
        )

    def forward(self, z, label_ohe, codes_dim=24):
        '''
        z.shape = (*, 120)
        label_ohe.shape = (*, n_classes)

        Args:
            z ([type]): [description]
            label_ohe ([type]): [description]
            codes_dim (int, optional): [description]. Defaults to 24.
        '''
        batch = z.size(0)
        z = z.squeeze()
        label_ohe = label_ohe.squeeze()
        codes = torch.split(z, codes_dim, dim=1)

        x = self.fc(codes[0]) # (*, 16ch*4*4)
        x = x.view(batch, -1, 4, 4) # (*, 16ch, 4, 4)

        condition = torch.cat([codes[1], label_ohe], dim=1) # (codes_dim+n_classes)
        x = self.res1(x, condition) # (*, 16ch, 8, 8)

        condition = torch.cat([codes[2], label_ohe], dim=1)
        x = self.res2(x, condition) # (*, 8ch, 16, 16)

        condition = torch.cat([codes[3], label_ohe], dim=1)
        x = self.res3(x, condition) # (*, 4ch, 32, 32)
        x = self.attn(x) # not change shape

        condition = torch.cat([codes[4], label_ohe], dim=1)
        x = self.res4(x, condition) # (*, 2ch, 64, 64)
        
        x = self.conv(x) # (*, 3, 64, 64)
        x = torch.tanh(x)
        return x

class ResBlock_D(nn.Module):
    def __init__(self, in_channel, out_channel, downsample=True):
        super().__init__()
        self.layer = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(conv3x3(in_channel, out_channel)).apply(init_weight),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(conv3x3(out_channel, out_channel)).apply(init_weight)
        )

        self.shortcut = nn.Sequential(
            nn.utils.spectral_norm(conv1x1(in_channel, out_channel)).apply(init_weight)
        )
        
        if downsample:
            self.layer.add_module('avgpool', nn.AvgPool2d(kernel_size=2, stride=2))
            self.shortcut.add_module('avgpool', nn.AvgPool2d(kernel_size=2, stride=2))

    def forward(self, inputs):
        x = self.layer(inputs)
        x += self.shortcut(inputs)
        return x 

class Discriminator(nn.Module):
    def __init__(self, n_feat, n_classes=10):
        super().__init__()
        self.res1 = ResBlock_D(3, n_feat, downsample=True)
        self.attn = Attention(n_feat)
        self.res2 = ResBlock_D(n_feat, 2*n_feat, downsample=True)
        self.res3 = ResBlock_D(2*n_feat, 4*n_feat, downsample=True)
        self.res4 = ResBlock_D(4*n_feat, 8*n_feat, downsample=True)
        self.res5 = ResBlock_D(8*n_feat, 16*n_feat, downsample=False)

        self.fc = nn.utils.spectral_norm(nn.Linear(16*n_feat, 1)).apply(init_weight)
        self.embedding = nn.Embedding(num_embeddings=n_classes, embedding_dim=16*n_feat).apply(init_weight)

    def forward(self, inputs, label):
        batch = inputs.size(0) # (*, 3, 64, 64)
        h = self.res1(inputs) # (*, ch, 32, 32)
        h = self.attn(h) # not change shape
        h = self.res2(h) # (*, 2ch, 16, 16)
        h = self.res3(h) # (*, 4ch, 8, 8)
        h = self.res4(h) # (*, 8ch, 4, 4)
        h = self.res5(h) # (*, 16ch, 4, 4)
        h = torch.sum((F.leaky_relu(h, 0.2)).view(batch, -1, 4*4), dim=2) # GlobalSumPool (*, 16ch)

        outputs = self.fc(h) # (*, 1)

        if label is not None:
            embed = self.embedding(label) # (*, 16ch)
            outputs += torch.sum(embed*h, dim=1, keepdim=True) # (*, 1)

        outputs = torch.sigmoid(outputs)
        return outputs

