# %% 
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import torch
import torch.nn.functional as F
import random
import torchvision.utils as utils

from models.vae_gan import Discriminator, Encoder, Generator
from dataset.dataset import get_Dataset
from utils.utils import get_cuda, weights_init, generate_samples

import argparse
# %%
os.makedirs("images/", exist_ok=True)

# %%
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=150, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.00005, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=128, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
parser.add_argument("--dcgan", action='store_false', help='use MLP')
parser.add_argument("--dataset", type=str, choices=['mnist', 'fashion', 'cifar10'],
                    default='cifar10', help="dataset to use")
parser.add_argument("--dataroot", type=str, default='../data/', help='path to dataset')
parser.add_argument('--w_kld', type=float, default=1)
parser.add_argument('--w_loss_g', type=float, default=0.01)
parser.add_argument('--w_loss_gd', type=float, default=1)

opt = parser.parse_args([])
print(opt)
# %%
manual_seed = random.randint(1, 10000)
random.seed(manual_seed)
torch.manual_seed(manual_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(manual_seed)

train_loader = get_Dataset(opt)
# %%
encoder = get_cuda(Encoder(opt))
generator = get_cuda(Generator(opt)).apply(weights_init)
discriminator = get_cuda(Discriminator(opt)).apply(weights_init)

print(encoder)
print(generator)
print(discriminator)
# %%
optimizer_E = torch.optim.Adam(encoder.parameters(), lr=opt.lr)
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(0.5, 0.999))

# %%
def train_batch(x_r):
    batch_size = x_r.size(0)
    y_real = get_cuda(torch.ones(batch_size))
    y_fake = get_cuda(torch.zeros(batch_size))

    z, kld = encoder(x_r)
    kld = kld.mean()

    x_f = generator(z)

    z_p = torch.randn(batch_size, opt.latent_dim)
    z_p = get_cuda(z_p)

    x_p = generator(z_p)

    # compute d(x) for real and fake images along with their features
    ld_r, fd_r = discriminator(x_r)
    ld_f, fd_f = discriminator(x_f)
    ld_p, fd_p = discriminator(x_p)

    # ---------------------
    #  Train Discriminator
    # ---------------------
    loss_D = F.binary_cross_entropy(ld_r, y_real) + 0.5 * (F.binary_cross_entropy(ld_f, y_fake) + F.binary_cross_entropy(ld_p, y_fake))
    optimizer_D.zero_grad()
    loss_D.backward(retain_graph=True)
    optimizer_D.step()

    # ---------------------
    #  Train G and E
    # ---------------------
    with torch.autograd.set_detect_anomaly(True):
        # loss to -log(D(G(z_p)))
        y_real.requires_grad_(True)
        loss_GD = F.binary_cross_entropy(ld_p, y_real)
        # pixel wise matching loss and discriminator's feature matching loss
        loss_G = 0.5 * (0.01 * (x_f - x_r).pow(2).sum() + (fd_f - fd_r.detach()).pow(2).sum()) / batch_size
        loss_G.requires_grad_(True)
        loss_G.backward()

    optimizer_E.zero_grad()
    optimizer_G.zero_grad()

    loss = opt.w_kld * kld + opt.w_loss_g * loss_G + opt.w_loss_gd * loss_GD
    loss.requires_grad_(True)
    loss.backward()
    optimizer_E.step()
    optimizer_G.step()

    return loss_D.item(), loss_G.item(), loss_GD.item(), kld.item()

# %%
def training():
    print("start training!")
    start_epoch = 0
    
    for epoch in range(opt.n_epochs):
        encoder.train()
        generator.train()
        discriminator.train()

        for imgs, _ in train_loader:
            imgs = get_cuda(imgs)
            loss_D, loss_G, loss_GD, loss_kld = train_batch(imgs) #64, 3, 128, 128

        print("epoch:", epoch, "loss_D:", "%.4f"% loss_D, "loss_G:", "%.4f"%loss_G, "loss_GD:", "%.4f"%loss_GD, "loss_kld:", "%.4f"%loss_kld)

        generate_samples("images/%d.png" % epoch, opt, encoder, discriminator, generator)

# %%
if __name__ == "__main__":
    # training()
    for img, _ in train_loader:
        img = get_cuda(img)
        loss_D, loss_G, loss_GD, loss_kld = train_batch(img)
# %%
