# %% 
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
sys.path.append('..')
sys.path.append('.')

import numpy as np
import torch
import torch.nn.functional as F
import random
import torchvision.utils as utils
from torch.autograd import Variable

np.random.seed(8)
torch.manual_seed(8)
torch.cuda.manual_seed(8)

from vae_gan.models.vae_gan import Discriminator, Encoder, VaeGan
from vae_gan.dataset.dataset import get_Dataset
from vae_gan.utils.utils import *

from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR
from torch.optim import RMSprop, Adam, SGD
from torchvision.utils import make_grid
import torchvision.utils as vutils
from torchvision.utils import save_image

from vae_gan.options import args

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
# %%
# manual_seed = random.randint(1, 10000)
# random.seed(manual_seed)
# torch.manual_seed(manual_seed)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(manual_seed)

# ------------ dataloader ------------

train_loader = get_Dataset(args, train=True)
test_loader = get_Dataset(args, train=False)
# print(len(train_loader))

# %%
net = VaeGan(z_size=args.z_size, recon_level=args.recon_level).cuda()

print(net)

# %%
# ------------ margin and equilibirum ------------
margin = 0.35
equilibrium = 0.68
# %%
# --------------------- optimizers --------------------
# an optimizer for each of the sub-networks, so we can selectively backprop

optimizer_encoder = RMSprop(params=net.encoder.parameters(), lr=args.lr, alpha=0.9, eps=1e-8, weight_decay=0, momentum=0, centered=False)
lr_encoder = ExponentialLR(optimizer=optimizer_encoder, gamma=args.decay_lr)

optimizer_decoder = RMSprop(params=net.decoder.parameters(), lr=args.lr, alpha=0.9, eps=1e-8, weight_decay=0, momentum=0, centered=False)
lr_decoder = ExponentialLR(optimizer=optimizer_decoder, gamma=args.decay_lr)

optimizer_discriminator = RMSprop(params=net.discriminator.parameters(), lr=args.lr, alpha=0.9, eps=1e-8, weight_decay=0, momentum=0, centered=False)
lr_discriminator = ExponentialLR(optimizer=optimizer_discriminator, gamma=args.decay_lr)

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
        for j, (x, label) in enumerate (train_loader):
            net.train()
            batch_size = len(x)

            x = Variable(x, requires_grad=False).float().cuda()

            x_tilde, disc_class, disc_layer, mus, log_variances = net(x) 
            # * disc_layer 192, 16384
            # * disc_class 192, 512

            # split so we can get the different parts
            # * recon
            disc_layer_original = disc_layer[:batch_size]
            disc_layer_predicted = disc_layer[batch_size:-batch_size]
            disc_layer_sampled = disc_layer[-batch_size:]

            # * gan
            disc_class_original = disc_class[:batch_size]
            disc_class_predicted = disc_class[batch_size:-batch_size]
            disc_class_sampled = disc_class[-batch_size:]

            nle, kl, mse, bce_dis_original, bce_dis_predicted, bce_dis_sampled = VaeGan.loss(
                x, x_tilde=x_tilde, disc_layer_original=disc_layer_original, \
                disc_layer_predicted=disc_layer_predicted, disc_layer_sampled=disc_layer_sampled, \
                disc_class_original=disc_class_original, disc_class_predicted=disc_class_predicted, \
                disc_class_sampled=disc_class_sampled,
                mus=mus, variances=log_variances,
                disc_layer = disc_layer, disc_class = disc_class
            )
            
            # this is the most important part of the code 
            # todo
            loss_encoder = torch.mean(torch.sum(kl) +torch.sum(mse))
            loss_discriminator = torch.sum(bce_dis_original) + torch.sum(bce_dis_predicted) + torch.sum(bce_dis_sampled)
            loss_decoder = torch.sum(lambda_mse * mse) - (1.0 - lambda_mse) * loss_discriminator

            # selectively disable the decoder of the discriminator if they are unbalanced
            train_dis = True
            train_dec = True

            if torch.mean(bce_dis_original).item() < equilibrium-margin or torch.mean(bce_dis_predicted).item() < equilibrium-margin:
                train_dis = False
            if torch.mean(bce_dis_original).item() > equilibrium+margin or torch.mean(bce_dis_predicted).item() > equilibrium+margin:
                train_dec = False
            if train_dec is False and train_dis is False:
                train_dis = True
                train_dec = True

            net.zero_grad()

            # encoder 
            loss_encoder.backward(retain_graph=True)
            # optimizer_encoder.step()
            net.zero_grad() # cleanothers, so they are not afflicted by encoder loss 

            writer.add_scalar('loss_encoder', loss_encoder, i)

            # decoder 
            if train_dec:
                loss_decoder.backward(retain_graph=True)
                # optimizer_decoder.step()
                net.discriminator.zero_grad() # clean the discriminator

            # writer tensorboard
            writer.add_scalar('loss_decoder', loss_decoder, i)

            # discriminator 
            if train_dis:
                loss_discriminator.backward()
                # optimizer_discriminator.step()
            
            # writer tensorboard 
            writer.add_scalar('loss_discriminator', loss_discriminator, i)
            
            # todo 这个地方存在问题，有可能
            optimizer_encoder.step()
            optimizer_decoder.step()
            optimizer_discriminator.step()

            print('total epoch: [%02d] step: [%02d] | encoder loss: %.5f | decoder loss: %.5f | discriminator loss: %.5f' % (i, j, loss_encoder, loss_decoder, loss_discriminator))

        lr_encoder.step()
        lr_decoder.step()
        lr_discriminator.step()

        margin *= decay_margin
        equilibrium *= decay_equilibrium
        if margin > equilibrium:
            equilibrium = margin
        lambda_mse *= decay_mse
        if lambda_mse > 1:
            lambda_mse = 1
        
        # save results
        for j, (x, label) in enumerate(test_loader):
            net.eval()

            x = Variable(x, requires_grad=False).float().cuda()

            out = x.data.cpu()
            out = (out + 1) / 2
            save_image(vutils.make_grid(out[:64], padding=5, normalize=True).cpu(), './images/real_image/original%s.png' % (i), nrow=8)

            out = net(x) # out = x_tilde
            out = out.data.cpu()
            out = (out + 1) / 2
            save_image(vutils.make_grid(out[:64], padding=5, normalize=True).cpu(), './images/recon_image/reconstructed%s.png' % (i), nrow=8)

            out = net(None, 100) # out = x_p
            out = out.data.cpu()
            out = (out + 1) / 2
            save_image(vutils.make_grid(out[:64], padding=5, normalize=True).cpu(), './images/generate_image/generated%s.png' % (i), nrow=8)

            break

    exit(0)
