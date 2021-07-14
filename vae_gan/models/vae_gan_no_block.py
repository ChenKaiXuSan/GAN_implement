import torch 
import torch.nn as nn 
from torch.autograd import Variable

from options import args

from utils.utils import tensor2var, weights_init
from models.spectral import Self_Attn, SpectralNorm

class Encoder(nn.Module):
    def __init__(self, channels_in, z_size) -> None:
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size()[0]
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.relu(self.bn3(self.conv3(out)))

        out = out.view(batch_size, -1)

        out = self.relu(self.bn4(self.fc1(out)))

        mean = self.fc_mean(out)
        logvar = self.fc_logvar(out)

        return mean, logvar

class Decoder(nn.Module):
    def __init__(self, z_size, size, channel_in) -> None:
        super(Decoder, self).__init__()

        self.fc1 = nn.Linear(128, 8 * 8 * 256)
        self.bn1 = nn.BatchNorm1d(8 * 8 * 256, momentum=0.9)

        self.relu = nn.LeakyReLU(0.2)

        self.deconv1 = SpectralNorm(nn.ConvTranspose2d(256, 128, kernel_size=6, stride=2, padding=2))
        self.bn2 = nn.BatchNorm2d(128, momentum=0.9)
        self.deconv2 = SpectralNorm(nn.ConvTranspose2d(128, 64, kernel_size=6, stride=2, padding=2))
        self.bn3 = nn.BatchNorm2d(64, momentum=0.9)
        self.deconv3 = SpectralNorm(nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2, padding=2))
        self.bn4 = nn.BatchNorm2d(32, momentum=0.9)

        self.deconv4 = nn.ConvTranspose2d(32, args.channels, kernel_size=5, stride=1, padding=2)
        self.tanh = nn.Tanh()

        self.attn1 = Self_Attn(32, 'relu')

    def forward(self, x) -> torch.Tensor:
        batch_size = x.size()[0]
        x = self.relu(self.bn1(self.fc1(x)))
        x = x.view(batch_size, 256, 8, 8)
        x = self.relu(self.bn2(self.deconv1(x)))
        x = self.relu(self.bn3(self.deconv2(x)))
        x = self.relu(self.bn4(self.deconv3(x)))

        # x, _ = self.attn1(x)

        x = self.tanh(self.deconv4(x))

        return x

class Discriminator(nn.Module):
    def __init__(self, channel_in) -> torch.Tensor:
        super(Discriminator, self).__init__()
        
        self.fc = nn.Linear(128, 64 * 64 * channel_in)
        self.bn = nn.BatchNorm1d(64 * 64 * channel_in, momentum=0.9)

        self.conv1 = SpectralNorm(nn.Conv2d(args.channels, 32, kernel_size=5, padding=2, stride=1))
        self.relu = nn.LeakyReLU(0.2)

        self.conv2 = SpectralNorm(nn.Conv2d(32, 128, kernel_size=5, padding=2, stride=2))
        self.bn1 = nn.BatchNorm2d(128, momentum=0.9)
        self.conv3 = SpectralNorm(nn.Conv2d(128, 256, kernel_size=5, padding=2, stride=2))
        self.bn2 = nn.BatchNorm2d(256, momentum=0.9)
        self.conv4 = SpectralNorm(nn.Conv2d(256, 512, kernel_size=5, padding=2, stride=2))
        self.bn3 = nn.BatchNorm2d(512, momentum=0.9)

        self.fc1 = nn.Linear(8 * 8 * 512, 1024)
        self.bn4 = nn.BatchNorm1d(1024, momentum=0.9)

        self.attn1 = Self_Attn(512, 'relu')

        self.fc2 = nn.Linear(1024, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x) -> torch.Tensor:
        batch_size = x.size()[0]

        # x = self.relu(self.bn(self.fc(x)))
        # x = x.view(batch_size, -1, 64, 64)

        x = self.relu(self.conv1(x))
        x = self.relu(self.bn1(self.conv2(x)))
        x = self.relu(self.bn2(self.conv3(x)))
        x = self.relu(self.bn3(self.conv4(x)))

        # x, _ = self.attn1(x)

        x = x.view(-1, 512 * 8 * 8)
        x1 = x;

        x = self.relu(self.bn4(self.fc1(x)))
        x = self.sigmoid(self.fc2(x))

        return x, x1

class VaeGan(nn.Module):
    def __init__(self, z_size, channels_in) -> None:
        super(VaeGan, self).__init__()

        self.encoder = Encoder(channels_in=channels_in, z_size=z_size)
        self.decoder = Decoder(z_size=z_size, size=z_size, channel_in=channels_in)
        self.discriminator = Discriminator(channel_in=channels_in)

        # self.encoder.apply(weights_init)
        # self.decoder.apply(weights_init)
        # self.discriminator.apply(weights_init)

    def forward(self, x) -> torch.Tensor:
        batch_size = x.size()[0]
        
        z_mean, z_logvar = self.encoder(x)
        
        std = z_logvar.mul(0.5).exp_()

        epsilon = tensor2var(torch.randn(batch_size, 128))
        z = z_mean + std * epsilon

        x_tilda = self.decoder(z)

        return z_mean, z_logvar, x_tilda
