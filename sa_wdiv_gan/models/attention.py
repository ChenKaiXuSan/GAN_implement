import torch
import torch.nn as nn 
import torch.nn.functional as F

from torch.nn.utils import spectral_norm 

def conv1x1(in_channels, out_channels): # not change resolusion
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)

class Attention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.theta = spectral_norm(conv1x1(channels, channels // 8))
        self.phi = spectral_norm(conv1x1(channels, channels // 8))

        self.o = spectral_norm(conv1x1(channels // 2, channels))
        self.g = spectral_norm(conv1x1(channels, channels // 2))

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

