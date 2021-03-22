# %%
import argparse
import os
import numpy as np
import math
import sys

import torch
from torch import tensor
from torch import random
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

import random
import torchvision.transforms as transform
from torchvision.utils import save_image
# from torch.utils.data import DataLoader, dataloader
from torchvision import datasets

import models.mlp as mlp
import dataset.dataset as dst
import models.dcgan as dcgan

from torch.utils.tensorboard import SummaryWriter
# %%
os.makedirs('images/', exist_ok=True)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import shutil
shutil.rmtree('runs/')
writer = SummaryWriter('runs/c_wgan')
# %%
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=150, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.00005, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
parser.add_argument("--dcgan", action='store_false', help='use MLP')
parser.add_argument("--dataset", type=str, choices=['mnist', 'fashion'],
                    default='mnist', help="dataset to use")
opt = parser.parse_args([])
print(opt)
# %%
opt.img_shape = (opt.channels, opt.img_size, opt.img_size)

opt.manualSeed = random.randint(1, 10000) # fix seed 
print("Random seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cuda = True if torch.cuda.is_available() else False

opt.n_classes = 10
# %%
# loss weight for gradient penalty
lambda_gp = 10

# init g and d
if not opt.dcgan:
    generator = mlp.Generator(opt)
    discriminator = mlp.Discriminator(opt)
else:
    generator = dcgan.DCGAN_G(opt)
    discriminator = dcgan.DCGAN_D(opt)
    generator.apply(dcgan.weights_init)
    discriminator.apply(dcgan.weights_init)

if cuda:
    generator.cuda()
    discriminator.cuda()

print(generator)
print(discriminator)
# %%
# configure data loader 
dataloader = dst.getdDataset(opt)

# optimizers 
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

# %%

def sample_image(n_row, batches_done):
    '''
    用训练好的G来生成图片，保存用

    Args:
        n_row (int): 生成图片的列数
        batches_done (int): 图片批次
    '''    

    # sample noise
    z = Tensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim)))
    # get labels ranging from 0 to n_classes for n rows 
    labels = np.array([num for _ in range(n_row) for num in range(n_row)])
    with torch.no_grad():
        labels = LongTensor(labels)
        gen_imgs = generator(z, labels)
    save_image(gen_imgs.data, 'images/c_wgan/%d.png' % batches_done, nrow=n_row, normalize=True)

# %%
def compute_gradient_penalty(D, real_samples, fake_samples, labels):
    '''
    计算gp惩罚

    Args:
        D (D): 传入的D
        real_samples (tensor): 真实样本
        fake_samples (tensor): 假的样本
        labels (tensor): 标签
    '''

     # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    labels = LongTensor(labels)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates, labels)
    fake = Tensor(real_samples.shape[0], 1, 1, 1).fill_(1.0)
    fake.requires_grad = False
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )
    gradients = gradients[0].view(gradients[0].size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

# %%
# ----------
#  Training
# ----------

batches_done = 0
for epoch in range(opt.n_epochs):
    for i, (imgs, labels) in enumerate(dataloader):
        batch_size = imgs.shape[0]

        # to GPU
        real_imgs = imgs.type(Tensor)
        labels = labels.type(LongTensor)
        # save real img
        save_image(real_imgs.data, 'images/c_wgan/real_image.png', nrow=opt.n_classes, normalize=True)
        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # sample noise and labels as generator input
        z = Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim)))

        # generate a batch of images 
        fake_imgs = generator(z, labels)

        # real images 
        real_validity = discriminator(real_imgs, labels)
        # fake images 
        fake_validity = discriminator(fake_imgs, labels)

        # gradient penalty
        gradient_penalty = compute_gradient_penalty(
            discriminator, real_imgs.data, fake_imgs.data,
            labels.data
        )
        
        # adversarial loss
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty

        d_loss.backward()
        optimizer_D.step()

        writer.add_scalar('epoch/d_loss', d_loss, epoch)

        optimizer_G.zero_grad()

        # train the generator every n_critic steps
        if i % opt.n_critic == 0:

            # -----------------
            #  Train Generator
            # -----------------

            # generate a batch of images 
            fake_imgs = generator(z, labels)

            # loss measures generator's ability to fool the discriminator
            # train on fake images 
            fake_validity = discriminator(fake_imgs, labels)
            g_loss = -torch.mean(fake_validity)

            g_loss.backward()
            optimizer_G.step()

            writer.add_scalar('epoch/g_loss', g_loss, epoch)

            writer.add_scalars('epoch', {'g_loss':g_loss, 'd_loss':d_loss}, epoch)

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            )

            if batches_done % opt.sample_interval == 0:
                sample_image(opt.n_classes, batches_done)

            batches_done += opt.n_critic
# %%
