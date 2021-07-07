# %% 
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

from models.vae_gan_no_block import Discriminator, VaeGan
# from models.vae_gan import Discriminator, Encoder, VaeGan
from dataset.dataset import get_Dataset
from utils.utils import *

from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR
from torch.optim import RMSprop, Adam, SGD, adam
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
del_folder('runs', '')

# create dir if not exist
make_folder(args.sample_path, args.version)
make_folder(args.sample_path, 'sample')

# ----------- tensorboard ------------
writer = build_tensorboard()

# ------------ dataloader ------------
train_loader = get_Dataset(args, train=True)
test_loader = get_Dataset(args, train=False)

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

optimizer_encoder = Adam(params=generator.encoder.parameters(), lr=args.lr)
# lr_encoder = ExponentialLR(optimizer=optimizer_encoder, gamma=args.decay_lr)

optimizer_decoder = Adam(params=generator.decoder.parameters(), lr=args.lr)
# lr_decoder = ExponentialLR(optimizer=optimizer_decoder, gamma=args.decay_lr)


# optimizer_discriminator = RMSprop(params=discriminator.parameters(), lr=args.lr * 0.1, alpha=0.9, eps=1e-8, weight_decay=0, momentum=0, centered=False)
optimizer_discriminator = Adam(discriminator.parameters(), lr=args.lr)
# lr_discriminator = ExponentialLR(optimizer=optimizer_discriminator, gamma=args.decay_lr)

# %%
real_batch = next(iter(train_loader))

# ------------ loss function ------------
bce_loss = nn.BCELoss().cuda()
mse_loss = nn.MSELoss().cuda()

# %%
# ------------ training loop ------------
if __name__ == "__main__":
    
    z_size = args.z_size
    n_epochs = args.n_epochs

    print('Start training!')
    for i in range(n_epochs + 1):
        print('Epoch: %s' % (i))
        for j, (img, label) in enumerate (train_loader):
            generator.train()
            discriminator.train()

            batch_size = img.size()[0]

            real_label = tensor2var(torch.ones(batch_size, 1))
            fake_label = tensor2var(torch.zeros(batch_size, 1))

            datav = tensor2var(img)

            # 
            #   D
            #
            mean, logvar, rec_enc = generator(datav)

            z_p = tensor2var(torch.randn(batch_size, z_size))
            x_p_tilda = generator.decoder(z_p)

            output = discriminator(datav)[0]
            errD_real = bce_loss(output, real_label)

            output = discriminator(rec_enc)[0]
            errD_rec_enc = bce_loss(output, fake_label)

            output = discriminator(x_p_tilda)[0]
            errD_rec_noise = bce_loss(output, fake_label)

            gan_loss = errD_real + errD_rec_enc + errD_rec_noise

            optimizer_discriminator.zero_grad()
            gan_loss.backward(retain_graph=True)
            optimizer_discriminator.step()

            #
            #   Decoder
            #
            output = discriminator(datav)[0]
            errD_real = bce_loss(output, real_label)
            output = discriminator(rec_enc)[0]
            errD_rec_enc = bce_loss(output, fake_label)
            output = discriminator(x_p_tilda)[0]
            errD_rec_noise = bce_loss(output, fake_label)
            gan_loss = errD_real + errD_rec_enc + errD_rec_noise
            
            x_l_tilda = discriminator(rec_enc)[1]
            x_l = discriminator(datav)[1]
            rec_loss = ((x_l_tilda - x_l) ** 2).mean()
            err_dec = gamma * rec_loss - gan_loss 

            optimizer_decoder.zero_grad()
            err_dec.backward(retain_graph=True)
            optimizer_decoder.step()

            #
            #   Encoder
            #
            mean, logvar, rec_enc = generator(datav)
            x_l_tilda = discriminator(rec_enc)[1]
            x_l = discriminator(datav)[1]
            rec_loss = ((x_l_tilda - x_l) ** 2).mean()

            prior_loss = 1 + logvar - mean.pow(2) - logvar.exp()
            prior_loss = (-0.5 * torch.sum(prior_loss))/torch.numel(mean.data)
            
            err_enc = prior_loss + 5*rec_loss

            optimizer_encoder.zero_grad()
            err_enc.backward(retain_graph=True)
            optimizer_encoder.step()

            print('total epoch: [%02d] step: [%02d] | encoder loss: %.5f | decoder loss: %.5f | discriminator loss: %.5f' % (i, j, err_enc, err_dec, gan_loss.item()))

            # save sample, use train image
            if (j + 1) % 10 == 0:
                generator.eval()

                datav = tensor2var(img)

                path = os.path.join(args.sample_path, 'sample')

                # save real image 
                save_image(denorm(datav[:64].data), path +'/real_image/original%s.png' % (j), nrow=8, normalize=True)

                # save x_fixed image
                x_fixed = tensor2var(real_batch[0])
                out = generator(x_fixed)[2]
                save_image(denorm(out[:64].data), path + '/recon_image/reconstructed%s.png' % (j), nrow=8, normalize=True)
            
                # save z_fixed image
                z_fixed = tensor2var(torch.randn((args.batch_size, args.z_size)))
                out = generator.decoder(z_fixed)
                save_image(denorm(out[:64].data), path + '/generate_image/generated%s.png' % (j), nrow=8, normalize=True)


    exit(0)
