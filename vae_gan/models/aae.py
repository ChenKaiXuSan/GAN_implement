# %%
import numpy as np

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

        self.conv1 = nn.Conv2d(args.channels, 64, kernel_size=5, padding=2, stride=2)
        self.bn1 = nn.BatchNorm2d(64, momentum=0.9)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, padding=2, stride=2)
        self.bn2 = nn.BatchNorm2d(128, momentum=0.9)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=5, padding=2, stride=2)
        self.bn3 = nn.BatchNorm2d(256, momentum=0.9)

        self.relu = nn.LeakyReLU(0.2)

        self.fc1 = nn.Linear(256 * 8 * 8, 2048)
        self.bn4 = nn.BatchNorm1d(2048, momentum=0.9)

        self.fc_mean = nn.Linear(2048, 128)
        self.fc_logvar = nn.Linear(2048, 128)

    def forward(self, img):
        batch_size = int(img.size()[0])
        out = self.relu(self.bn1(self.conv1(img)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.relu(self.bn3(self.conv3(out)))

        out_flat = out.view(batch_size, -1)

        out = self.relu(self.bn4(self.fc1(out_flat)))

        mean = self.fc_mean(out)
        logvar = self.fc_logvar(out)
        z = reparameterization(mean, logvar)
        return z

# %%
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(args.latent_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, int(np.prod(img_shape))),
            nn.Tanh(),
        )
    
    def forward(self, z):
        img_flat = self.model(z)
        img = img_flat.view(img_flat.shape[0], *img_shape)
        return img 

# %% 
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(args.latent_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        validity = self.model(z)
        return validity