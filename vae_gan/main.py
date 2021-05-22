# %% 
import os

from torch.nn.modules.loss import BCELoss

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
sys.path.append('..')
sys.path.append('.')

import numpy as np
import torch
import torch.nn as nn
import random
import torchvision.utils as utils
from torch.autograd import Variable, variable

np.random.seed(8)
torch.manual_seed(8)
torch.cuda.manual_seed(8)

from models.vae_gan import Discriminator, Encoder, VaeGan
from dataset.dataset import get_Dataset
from utils.utils import *

from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR
from torch.optim import RMSprop, Adam, SGD
from torchvision.utils import make_grid
import torchvision.utils as vutils
from torchvision.utils import save_image

from options import args

import argparse
# %%
# set random seed for reproducibility
manualSeed = 999
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# %%
# delete the exists path
del_folder(args.sample_path, '')
del_folder('runs', '')

# create dir if not exist
make_folder(args.sample_path, args.real_image)
make_folder(args.sample_path, args.generate_image)
make_folder(args.sample_path, args.recon_image)

# ----------- tensorboard ------------
writer = build_tensorboard()

# ------------ dataloader ------------

train_loader = get_Dataset(args, train=True)
test_loader = get_Dataset(args, train=False)
# print(len(train_loader))

# %%
# ------------ network ------------

generator = VaeGan(z_size=args.z_size, channels_in=args.channels).cuda()
discriminator = Discriminator(channel_in=args.channels).cuda()

print(generator)
# print(generator.decoder)
print(discriminator)

# %%
# ------------ margin and equilibirum ------------
margin = 0.35
equilibrium = 0.68
gamma = 15

# %%
# --------------------- optimizers --------------------
# an optimizer for each of the sub-networks, so we can selectively backprop

optimizer_encoder = RMSprop(params=generator.encoder.parameters(), lr=args.lr, alpha=0.9, eps=1e-8, weight_decay=0, momentum=0, centered=False)
lr_encoder = ExponentialLR(optimizer=optimizer_encoder, gamma=args.decay_lr)

optimizer_decoder = RMSprop(params=generator.decoder.parameters(), lr=args.lr, alpha=0.9, eps=1e-8, weight_decay=0, momentum=0, centered=False)
lr_decoder = ExponentialLR(optimizer=optimizer_decoder, gamma=args.decay_lr)

optimizer_discriminator = RMSprop(params=generator.discriminator.parameters(), lr=args.lr, alpha=0.9, eps=1e-8, weight_decay=0, momentum=0, centered=False)
lr_discriminator = ExponentialLR(optimizer=optimizer_discriminator, gamma=args.decay_lr)

# %%
real_batch = next(iter(train_loader))
z_fixed = Variable(torch.randn((64, 128)).cuda())
x_fixed = Variable(real_batch[0]).cuda()

# loss function 
bce_loss = nn.BCELoss().cuda()
# %%
# ------------ training loop ------------

if __name__ == "__main__":
    
    z_size = args.z_size
    recon_level = args.recon_level
    decay_mse = args.decay_mse
    decay_margin = args.decay_margin
    n_epochs = args.n_epochs
    lambda_mse = args.lambda_mse
    lr = args.lr
    decay_lr = args.decay_lr
    decay_equilibrium = args.decay_equilibrium

    print('Start training!')
    for i in range(n_epochs + 1):
        print('Epoch: %s' % (i))
        for j, (img, label) in enumerate (train_loader):
            generator.train()
            batch_size = img.size()[0]

            ones_label = tensor2var(torch.ones(batch_size, 1))
            zeros_label = tensor2var(torch.zeros(batch_size, 1))
            zeros_label_1 = tensor2var(torch.zeros(64, 1))

            datav = tensor2var(img)

            mean, logvar, rec_enc = generator(datav)

            # from random noise 
            z_p = tensor2var(torch.randn(64, 128))
            x_p_tilda = generator.decoder(z_p)

            # * ----- train discriminator -----
            # real data, to 1
            output = discriminator(datav)[0]
            loss_discriminator_real = bce_loss(output, ones_label)

            # encoder data, to 0
            output = discriminator(rec_enc)[0]
            loss_discriminator_rec_enc = bce_loss(output, zeros_label)

            # from random noise, to 0
            output = discriminator(x_p_tilda)[0]
            loss_discriminator_noise = bce_loss(output, zeros_label_1)

            # gan loss, like paper
            gan_loss = loss_discriminator_real + loss_discriminator_rec_enc + loss_discriminator_noise

            optimizer_discriminator.zero_grad()
            gan_loss.backward(retain_graph = True)
            optimizer_discriminator.step()

            writer.add_scalar('gan_loss', gan_loss, i)

            # * ----- train decoder -----
            # real data, to 1.
            output = discriminator(datav)[0]
            loss_discriminator_real = bce_loss(output, ones_label)

            writer.add_scalar('loss_discriminator_real', loss_discriminator_real, i)

            # encoder decoder data, to 0
            output = discriminator(rec_enc)[0]
            loss_discriminator_rec_enc = bce_loss(output, zeros_label)

            writer.add_scalar('loss_discriminator_rec_enc', loss_discriminator_rec_enc, i)

            # from random noise, to 0
            output = discriminator(x_p_tilda)[0]
            loss_discriminator_noise = bce_loss(output, zeros_label_1)

            writer.add_scalar('loss_discriminator_noise', loss_discriminator_noise, i)

            gan_loss = loss_discriminator_real + loss_discriminator_rec_enc + loss_discriminator_noise

            # logvar
            x_l_tilda = discriminator(rec_enc)[1]
            x_l = discriminator(datav)[1]

            loss_rec = ((x_l_tilda - x_l) ** 2).mean()

            loss_decoder = gamma * loss_rec - gan_loss

            optimizer_decoder.zero_grad()
            loss_decoder.backward(retain_graph=True)
            optimizer_decoder.step()

            writer.add_scalar('loss_decoder', loss_decoder, i)

            # * ----- train encoder -----
            mean, logvar, rec_enc = generator(datav)

            # logvar 
            x_l_tilda = discriminator(rec_enc)[1]
            x_l = discriminator(datav)[1]

            loss_rec = ((x_l_tilda - x_l) ** 2).mean() # mse

            writer.add_scalar('loss_rec', loss_rec, i)

            prior_loss = 1 + logvar - mean.pow(2) - logvar.exp()
            prior_loss = (-0.5 * torch.sum(prior_loss)) / torch.numel(mean.data) # kl

            writer.add_scalar('prior_loss', prior_loss, i)

            loss_encoder = prior_loss + 5 * loss_rec

            optimizer_encoder.zero_grad()
            loss_encoder.backward(retain_graph=True)
            optimizer_encoder.step()

            writer.add_scalar('loss_encoder', loss_encoder, i)

            print('total epoch: [%02d] step: [%02d] | encoder loss: %.5f | decoder loss: %.5f | discriminator loss: %.5f' % (i, j, loss_encoder, loss_decoder, gan_loss))

        lr_encoder.step()
        lr_decoder.step()
        lr_discriminator.step()

        # margin *= decay_margin
        # equilibrium *= decay_equilibrium
        # if margin > equilibrium:
        #     equilibrium = margin
        # lambda_mse *= decay_mse
        # if lambda_mse > 1:
        #     lambda_mse = 1
        
        # save results
        for j, (x, label) in enumerate(test_loader):
            generator.eval()

            datav = tensor2var(x)

            # save real image
            out = (datav + 1) / 2
            save_image(vutils.make_grid(out[:64], padding=5, normalize=True).cpu(), './images/real_image/original%s.png' % (i), nrow=8)

            # save x_fixed image, from encoder > decoder
            out = generator(x_fixed)[2] # out = x_tilde
            out = (out + 1) / 2
            save_image(vutils.make_grid(out[:64], padding=5, normalize=True).cpu(), './images/recon_image/reconstructed%s.png' % (i), nrow=8)

            # save z_fixed image, from noise z > decoer
            out = generator.decoder(z_fixed) # out = x_p
            out = (out + 1) / 2
            save_image(vutils.make_grid(out[:64], padding=5, normalize=True).cpu(), './images/generate_image/generated%s.png' % (i), nrow=8)

            break

    exit(0)

