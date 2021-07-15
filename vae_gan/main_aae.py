# %% 
import itertools
import os
from socket import AF_UNSPEC

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

# from models.aae import Discriminator, Encoder, Decoder
from models.aae_mlp import Discriminator, Encoder, Decoder
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

cuda = True if torch.cuda.is_available() else False
# %%
# delete the exists path
del_folder(args.sample_path, args.version)
del_folder('runs', '')

# create dir if not exist
make_folder(args.sample_path, args.version)
make_folder('runs', '')

# ----------- tensorboard ------------
writer = build_tensorboard()

# ------------ dataloader ------------
train_loader = get_Dataset(args, train=True)
test_loader = get_Dataset(args, train=False)

# %%
# ------------ network ------------
# init the network
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
        
if cuda:
    encoder = Encoder().cuda()
    decoder = Decoder().cuda()
    discriminator = Discriminator().cuda()

# Initialize weights
# encoder.apply(weights_init)
# decoder.apply(weights_init)
# discriminator.apply(weights_init)

print(encoder)
print(decoder)
print(discriminator)

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
bce_sigmoid_loss = nn.BCEWithLogitsLoss().cuda()
mse_loss = nn.MSELoss().cuda()
pixelwise_loss = nn.L1Loss().cuda()
auxiliary_loss = nn.CrossEntropyLoss().cuda()

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

            # adversarial ground truths
            ones_label = tensor2var(torch.ones(batch_size, 1), grad=False)
            zeros_label = tensor2var(torch.zeros(batch_size, 1), grad=False)

            # configure input 
            datav = tensor2var(img)
            labelv = tensor2var(label)

            # sample noise and labels 
            z = tensor2var(torch.randn(batch_size, args.latent_dim))
            gen_labels = tensor2var(torch.randint(0, args.n_classes, (batch_size,)))

            encoded_imgs = encoder(datav) # 128, 128
            decoded_imgs= decoder(encoded_imgs, gen_labels) # 128, 1, 64, 64

            # train generator

            optimizer_generator.zero_grad()

            validity, pred_label = discriminator(decoded_imgs.detach())
            
            # Loss 
            aae_loss = 0.001 * bce_loss(discriminator(encoded_imgs)[0], ones_label) + \
                    0.999 * pixelwise_loss(decoded_imgs, datav)

            acgan_loss = 0.5 * (bce_loss(validity, ones_label) + auxiliary_loss(pred_label, gen_labels))

            g_loss = 0.5 * (aae_loss + acgan_loss)

            g_loss.backward()
            optimizer_generator.step()

            writer.add_scalar('g_loss', g_loss, i)

            # train discriminator 
            optimizer_discriminator.zero_grad()

            # sample noise as discriminator ground truth 
            z_p = tensor2var(torch.randn(img.shape[0], args.latent_dim))

            # Measure discriminator's ability to classify real from generated samples 
            # loss for real image 
            real_pred, real_aux = discriminator(datav)
            d_real_loss = bce_loss(real_pred, ones_label) + auxiliary_loss(real_aux, labelv)

            # loss for fake image 
            fake_pred, fake_aux = discriminator(decoded_imgs.detach())
            d_fake_loss = bce_loss(fake_pred, zeros_label) + auxiliary_loss(fake_aux, gen_labels)


            real_loss = bce_loss(discriminator(z_p)[0], ones_label)
            fake_loss = bce_loss(discriminator(encoded_imgs.detach())[0], zeros_label)

            d_acgan_loss = 0.5 * (d_fake_loss + d_real_loss)
            d_aae_loss = 0.5 * (real_loss + fake_loss)

            d_loss = (d_acgan_loss + d_aae_loss) * 0.5

            # calcuate discriminator accuracy 
            pred = np.concatenate([real_aux.data.cpu().numpy(), fake_aux.data.cpu().numpy()], axis=0)
            gt = np.concatenate([labelv.data.cpu().numpy(), gen_labels.data.cpu().numpy()], axis=0)
            d_acc = np.mean(np.argmax(pred, axis=1) ==  gt)

            d_loss.backward()
            optimizer_discriminator.step()

            writer.add_scalar('d_loss', d_loss, i)

            print('total epoch: [%02d] step: [%02d] | encoder_decoder loss: %.5f | discriminator loss: %.5f, d_acc: %d%%' % (i, j, g_loss.item(), d_loss.item(), 100 * d_acc))

        # save sample, use train image
        for j, (img, label) in enumerate(train_loader):
            decoder.eval()
            encoder.eval()

            path = os.path.join(args.sample_path, args.version)

            # save real image 
            save_image(denorm(img[:100]), path +'/real_image/original%s.png' % (i), nrow=10, normalize=True)

            # save x_fixed image
            x_fixed = tensor2var(img)
            labels = np.array([num for _ in range(12) for num in range(10)]) # 120个标签
            labels = np.append(labels, [num for num in range(8)]) # 在后面添加8个标签
            labels = tensor2var(torch.cuda.LongTensor(labels))

            with torch.no_grad():
                encoder_imgs = encoder(x_fixed)
                out = decoder(encoder_imgs, labels)
            save_image(denorm(out[:100].data), path + '/recon_image/reconstructed%s.png' % (i), nrow=10, normalize=True)
        
            # save z_fixed image
            z_fixed = tensor2var(torch.randn(img.size(0), args.latent_dim))

            with torch.no_grad():
                out = decoder(z_fixed, labels)
            save_image(denorm(out[:100].data), path + '/generate_image/generated%s.png' % (i), nrow=10, normalize=True)

            break;

    exit(0)

