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

        self.fc1 = nn.Linear(128, 8 * 8 * 256)
        self.bn1 = nn.BatchNorm1d(8 * 8 * 256, momentum=0.9)

        self.relu = nn.LeakyReLU(0.2)

        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=6, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(128, momentum=0.9)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=6, stride=2, padding=2)
        self.bn3 = nn.BatchNorm2d(64, momentum=0.9)
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2, padding=2)
        self.bn4 = nn.BatchNorm2d(32, momentum=0.9)

        self.deconv4 = nn.ConvTranspose2d(32, args.channels, kernel_size=5, stride=1, padding=2)
        self.tanh = nn.Tanh()
    
    def forward(self, z):
        img = self.relu(self.bn1(self.fc1(z)))
        img = img.view(z.size()[0], 256, 8, 8)
        img = self.relu(self.bn2(self.deconv1(img)))
        img = self.relu(self.bn3(self.deconv2(img)))
        img = self.relu(self.bn4(self.deconv3(img)))

        img = self.tanh(self.deconv4(img))
        return img 

# %% 
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.fc = nn.Linear(128, 64 * 64 * args.channels)
        self.bn = nn.BatchNorm1d(64 * 64 * args.channels, momentum=0.9)

        self.conv1 = nn.Conv2d(args.channels, 32, kernel_size=5, padding=2, stride=1)
        self.relu = nn.LeakyReLU(0.2)        

        self.conv2 = nn.Conv2d(32, 128, kernel_size=5, padding=2, stride=2)
        self.bn1 = nn.BatchNorm2d(128, momentum=0.9)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=5, padding=2, stride=2)
        self.bn2 = nn.BatchNorm2d(256, momentum=0.9)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=5, padding=2, stride=2)
        self.bn3 = nn.BatchNorm2d(512, momentum=0.9)

        self.fc1 = nn.Linear(8 * 8 * 512, 1024)
        self.bn4 = nn.BatchNorm1d(1024, momentum=0.9)

        self.fc2 = nn.Linear(1024, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        img = self.relu(self.bn(self.fc(z)))
        img = img.view(z.size()[0], args.channels, 64, 64)

        img = self.relu(self.conv1(img))
        img = self.relu(self.bn1(self.conv2(img)))
        img = self.relu(self.bn2(self.conv3(img)))
        img = self.relu(self.bn3(self.conv4(img)))

        img = img.view(-1, 512 * 8 * 8)

        img = self.relu(self.bn4(self.fc1(img)))
        img = self.sigmoid(self.fc2(img))

        return img