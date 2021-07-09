# %% 
import itertools
import os

import sys

from torch.nn.modules.loss import BCELoss

sys.path.append('..')
sys.path.append('.')

import numpy as np
import torch
import torch.nn as nn
import random

np.random.seed(8)
torch.manual_seed(8)
torch.cuda.manual_seed(8)

# from models.vae_gan_no_block import Discriminator, Encoder, Decoder
# from models.vae_gan import Discriminator, Encoder, VaeGan
from models.aae import Discriminator, Encoder, Decoder
from dataset.dataset import get_Dataset
from utils.utils import *

from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR
from torch.optim import RMSprop, Adam, SGD
import torchvision.utils as vutils
from torchvision.utils import save_image

from options import args
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# %%
# set random seed for reproducibility
manualSeed = 999
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# %%
# delete the exists path
del_folder(args.sample_path, args.version)
del_folder(args.sample_path, 'sample')
# del_folder('runs', '')

# create dir if not exist
make_folder(args.sample_path, args.version)
make_folder(args.sample_path, 'sample')

# ----------- tensorboard ------------
# writer = build_tensorboard()

# ------------ dataloader ------------
train_loader = get_Dataset(args, train=True)
test_loader = get_Dataset(args, train=False)

# %%
# ------------ network ------------

# generator = VaeGan(z_size=args.z_size, channels_in=args.channels).cuda()
encoder = Encoder().cuda()
decoder = Decoder().cuda()
discriminator = Discriminator().cuda()

# print(generator)
# print(generator.decoder)
print(encoder)
print(decoder)
print(discriminator)

# %%
# ------------ margin and equilibirum ------------
margin = 0.35
equilibrium = 0.68
gamma = 15

# %%
# --------------------- optimizers --------------------
# an optimizer for each of the sub-networks, so we can selectively backprop

optimizer_generator = Adam(
    itertools.chain(encoder.parameters(), decoder.parameters()), lr=args.lr, betas=(args.b1, args.b2)
)
optimizer_discriminator = Adam(params=discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))

# %%
real_batch = next(iter(train_loader))

# ------------ loss function ------------
bce_loss = nn.BCELoss().cuda()
mse_loss = nn.MSELoss().cuda()
pixelwise_loss = nn.L1Loss().cuda()

# %%
# ------------ training loop ------------
if __name__ == "__main__":
    
    z_size = args.z_size
    n_epochs = args.n_epochs

    print('Start training!')
    for i in range(n_epochs + 1):
        print('Epoch: %s' % (i))
        for j, (img, label) in enumerate (train_loader):
            encoder.train()
            decoder.train()
            discriminator.train()

            batch_size = img.size()[0]

            ones_label = tensor2var(torch.ones(batch_size, 1))
            zeros_label = tensor2var(torch.zeros(batch_size, 1))

            datav = tensor2var(img)

            # train generator
            optimizer_generator.zero_grad()

            encoded_imgs = encoder(datav)
            decoded_imgs = decoder(encoded_imgs)

            # Loss 
            g_loss = 0.001 * bce_loss(discriminator(encoded_imgs), ones_label) + \
                    0.999 * pixelwise_loss(decoded_imgs, datav)

            g_loss.backward()
            optimizer_generator.step()

            # train discriminator 
            optimizer_discriminator.zero_grad()

            # sample noise as discriminator ground truth 
            z_p = tensor2var(torch.randn(img.shape[0], args.latent_dim))

            # Measure discriminator's ability to classify real from generated samples 
            real_loss = bce_loss(discriminator(z_p), ones_label)
            fake_loss = bce_loss(discriminator(encoded_imgs.detach()), zeros_label)
            d_loss = 0.5 * (real_loss + fake_loss)

            d_loss.backward()
            optimizer_discriminator.step()

            print('total epoch: [%02d] step: [%02d] | encoder_decoder loss: %.5f | discriminator loss: %.5f' % (i, j, g_loss.item(), d_loss.item()))

        # save sample, use train image
        if (i + 1) % 10 == 0:
            decoder.eval()
            encoder.eval()

            path = os.path.join(args.sample_path, 'sample')

            # save real image 
            save_image(denorm(datav.data.cpu()), path +'/real_image/original%s.png' % (i), nrow=8, normalize=True)

            # save x_fixed image
            # x_fixed = tensor2var(real_batch[0])
            # z_mean, z_logvar = encoder(x_fixed)
            # std = z_logvar.mul(0.5).exp_()
            # epsilon = tensor2var(torch.randn(batch_size, 128))
            # z = z_mean + std * epsilon
            z = tensor2var(torch.randn(100, args.latent_dim))
            with torch.no_grad():
                out = decoder(z)
            save_image(denorm(out.data.cpu()), path + '/recon_image/reconstructed%s.png' % (i), nrow=8, normalize=True)
        
            # # save z_fixed image
            # z_fixed = tensor2var(torch.randn((8, args.z_size)))
            # with torch.no_grad():
            #     out = decoder(z_fixed)
            # save_image(denorm(out.data.cpu()), path + '/generate_image/generated%s.png' % (j), nrow=8, normalize=True)

    exit(0)

