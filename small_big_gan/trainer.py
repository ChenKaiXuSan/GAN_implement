# %% 
import os 
import time
import torch
import datetime

import torch.nn as nn 
from torch.autograd import Variable
import torchvision
from torchvision.utils import save_image
import torch.autograd as autograd

import numpy as np

import sys 
sys.path.append('.')
sys.path.append('..')

from models.sagan import Generator, Discriminator
from utils.utils import *
import models.FID as fid

# %%
class Trainer(object):
    def __init__(self, data_loader, config):
        super(Trainer, self).__init__()

        # data loader 
        self.data_loader = data_loader

        # exact model and loss 
        self.model = config.model
        self.adv_loss = config.adv_loss

        # model hyper-parameters
        self.imsize = config.img_size 
        self.g_num = config.g_num
        self.z_dim = config.z_dim
        self.channels = config.channels
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.parallel = config.parallel

        self.lambda_gp = config.lambda_gp
        self.total_step = config.total_step
        self.d_iters = config.d_iters
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers 
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr 
        self.lr_decay = config.lr_decay
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.pretrained_model = config.pretrained_model
        self.n_classes = config.n_classes
        self.lambda_aux = config.lambda_aux

        self.dataset = config.dataset 
        self.use_tensorboard = config.use_tensorboard
        self.image_path = config.dataroot 
        self.log_path = config.log_path
        self.sample_path = config.sample_path
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.version = config.version

        # path
        self.log_path = os.path.join(config.log_path, self.version)
        self.sample_path = os.path.join(config.sample_path, self.version)


        if self.use_tensorboard:
            self.build_tensorboard()

        self.build_model()

        self.build_FID()

    def train(self):

        # data iterator 
        data_iter = iter(self.data_loader)
        step_per_epoch = len(self.data_loader)

        # start time
        start_time = time.time()
        for step in range(self.total_step):

            # ==================== Train D ==================
            self.D.train()
            self.G.train()

            try:
                real_images, labels = next(data_iter)
            except:
                data_iter = iter(self.data_loader)
                real_images, labels = next(data_iter)

            # compute loss with real images 
            # dr1, dr2, df1, df2, gf1, gf2 are attention scores 
            real_images = tensor2var(real_images)
            labels = tensor2var(labels)

            d_out_real, real_aux = self.D(real_images, labels)

            if self.adv_loss == 'wgan-gp':
                d_loss_real = -torch.mean(d_out_real) + self.lambda_aux * self.c_loss(real_aux, labels)
            elif self.adv_loss == 'hinge':
                d_loss_real = torch.nn.ReLU()(1.0 - d_out_real).mean()
            elif self.adv_loss == 'wgan-div':
                d_loss_real = -torch.mean(d_out_real)

            # apply Gumbel Softmax
            z = tensor2var(torch.randn(real_images.size(0), self.z_dim)) # 64, 100
            fake_images, gf = self.G(z, labels)
            d_out_fake, fake_aux = self.D(fake_images, labels)
            # todo

            self.save_image_tensorboard(fake_images, 'D/fake_images', step)

            if self.adv_loss == 'wgan-gp':
                d_loss_fake = torch.mean(d_out_fake) + self.lambda_aux * self.c_loss(fake_aux, labels)
            elif self.adv_loss == 'hinge':
                d_loss_fake = torch.nn.ReLU()(1.0 + d_out_fake).mean()
            elif self.adv_loss == 'wgan-div':
                d_loss_fake = d_out_fake.mean()

            
            # backward + optimize 
            d_loss = d_loss_real + d_loss_fake
            # self.reset_grad()
            # d_loss.backward()
            # self.d_optimizer.step()

            self.logger.add_scalar('d_loss_real + d_loss_fake', d_loss, step)

            if self.adv_loss == 'wgan-gp':
                # compute gradient penalty
                alpha = torch.rand(real_images.size(0), 1, 1, 1).cuda().expand_as(real_images)
                # 64, 1, 64, 64
                interpolated = Variable(alpha * real_images.data + (1 - alpha) * fake_images.data, requires_grad = True)
                # 64
                out, _ = self.D(interpolated, labels)

                grad = autograd.grad(
                    outputs=out,
                    inputs = interpolated,
                    grad_outputs = torch.ones(out.size()).cuda(),
                    retain_graph = True,
                    create_graph = True,
                    only_inputs = True
                )[0]

                grad = grad.view(grad.size(0), -1)
                grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
                d_loss_gp = torch.mean((grad_l2norm - 1) ** 2)

                # backward + optimize 
                d_loss_g = self.lambda_gp * d_loss_gp + d_loss

                self.reset_grad()
                d_loss_g.backward()
                self.d_optimizer.step()

                self.logger.add_scalar('d_loss_gp', d_loss_g, step)
            
            # todo
            elif self.adv_loss == 'wgan-div':
                p = 6
                k = 2

                # compute w-div gradient penalty

                # real grad
                interpolated = Variable(real_images, requires_grad = True)
                out, _ = self.D(interpolated, labels)

                real_grad_out = torch.ones(real_images.size(0)).cuda()

                real_grad = autograd.grad(
                    outputs = out,
                    inputs = interpolated,
                    grad_outputs = real_grad_out,
                    retain_graph = True,
                    create_graph = True,
                    only_inputs = True,
                )[0]
                real_grad_norm = real_grad.view(real_grad.size(0), -1).pow(2).sum(1) ** (p / 2)

                # fake grad
                interpolated = Variable(fake_images, requires_grad = True)
                out, _ = self.D(interpolated, labels)

                fake_grad_out = torch.ones(fake_images.size(0)).cuda()

                fake_grad = autograd.grad(
                    outputs = out,
                    inputs = interpolated,
                    grad_outputs = fake_grad_out,
                    retain_graph = True,
                    create_graph = True,
                    only_inputs = True,
                )[0]
                fake_grad_norm = fake_grad.view(fake_grad.size(0), -1).pow(2).sum(1) ** (p / 2)

                # compute div_gp loss 
                div_gp = torch.mean(real_grad_norm + fake_grad_norm) * k / 2

                d_loss = div_gp

                self.reset_grad()
                d_loss.backward()
                self.d_optimizer.step()

                self.logger.add_scalar('d_loss_div', d_loss, step)

            # train the generator every 5 steps
            if step % self.g_num == 0:

                # =================== Train G and gumbel =====================
                # create random noise 
                z = tensor2var(torch.randn(real_images.size(0), self.z_dim))
                gen_labels = tensor2var(torch.LongTensor(np.random.randint(0, self.n_classes, self.batch_size)))
                fake_images, attention = self.G(z, labels)

                # save intermediate images
                self.save_image_tensorboard(fake_images, 'G/fake_images', step)

                # compute loss with fake images 
                g_out_fake, pred_labels = self.D(fake_images, labels) # batch x n
                if self.adv_loss == 'wgan-gp':
                    g_loss_fake = -g_out_fake.mean() * self.lambda_aux + self.c_loss(pred_labels, labels)
                if self.adv_loss == 'wgan-div':
                    g_loss_fake = -g_out_fake.mean()

                self.reset_grad()
                g_loss_fake.backward()
                self.g_optimizer.step()

                self.logger.add_scalar('g_loss_fake', g_loss_fake, step)

            # calculate FID
            # fretchet_dist = fid.calculate_fretchet(real_images, fake_images, self.fid_model)
            # print out log info
            if (step + 1) % self.log_step == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))
                print("Elapsed [{}], G_step [{}/{}], D_step[{}/{}], d_out_gp: {:.4f}, g_loss: {:.4f}, "
                      " ave_gamma_l3: {:.4f}, ave_gamma_l4: {:.4f}, Fretchet_Distance:".
                      format(elapsed, step + 1, self.total_step, (step + 1),
                             self.total_step , d_loss_g.item(), g_loss_fake.item(),
                             self.G.attn1.gamma.mean().item(), self.G.attn2.gamma.mean().item() ))

            # sample images 
            if (step + 1) % self.sample_step == 0:
                self.save_sample(real_images, step)
                # make the fake labels by classes 
                labels = np.array([num for _ in range(self.n_classes) for num in range(self.n_classes)])
                
                # fixed input for debugging
                fixed_z = tensor2var(torch.randn(self.n_classes ** 2, self.z_dim)) # 100, 100

                with torch.no_grad():
                    labels = to_LongTensor(labels)
                    fake_images, _= self.G(fixed_z, labels)
                    self.save_image_tensorboard(fake_images[:64], 'G/from_noise', step+1)
                    # save fake image 
                    save_image(fake_images.data, 
                                os.path.join(self.sample_path + '/fake_images/', '{}_fake.png'.format(step + 1)), nrow=self.n_classes, normalize=True)


    def build_model(self):

        self.G = Generator(batch_size = self.batch_size, image_size = self.imsize, z_dim = self.z_dim, conv_dim = self.g_conv_dim, channels = self.channels, n_classes=self.n_classes).cuda()
        self.D = Discriminator(batch_size = self.batch_size, image_size = self.imsize, conv_dim = self.d_conv_dim, channels = self.channels, n_classes=self.n_classes).cuda()

        # apply the weights_init to randomly initialize all weights
        # to mean=0, stdev=0.2
        # self.G.apply(weights_init)
        # self.D.apply(weights_init)
        
        # loss and optimizer 
        self.g_optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.G.parameters()), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.D.parameters()), self.d_lr, [self.beta1, self.beta2])

        self.c_loss = torch.nn.CrossEntropyLoss().cuda()

        data_iter = iter(self.data_loader)
        real_images, labels = next(data_iter)

        self.logger.add_graph(self.D, (real_images.cuda(), labels.cuda()))
        self.logger.close()
        
        # print networks
        print(self.G)
        print(self.D)

    def build_tensorboard(self):
        from torch.utils.tensorboard import SummaryWriter
        self.logger = SummaryWriter(self.log_path)

    def reset_grad(self):
        self.d_optimizer.zero_grad()
        self.g_optimizer.zero_grad()

    def save_sample(self, real_images, step):
        path = self.sample_path + '/real_images/'
        save_image(real_images.data, os.path.join(path, '{}_real.png'.format(step + 1)), normalize=True)
    
    def build_FID(self):
        block_idx = fid.InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        model = fid.InceptionV3([block_idx])
        self.fid_model=model.cuda()

    def save_image_tensorboard(self, images, text, step):
        if step % 100 == 0:
            img_grid = torchvision.utils.make_grid(images, nrow=8)

            self.logger.add_image(text + str(step), img_grid, step)
            self.logger.close()

# %%
import self_attention_gan.main  as main
from self_attention_gan.dataset.dataset import getdDataset

if __name__ == '__main__':
    config = main.get_parameters()
    config.total_step = 1000
    config.img_size = 32
    print(config)
    # main(config)
    data_loader = getdDataset(config)

    trainer = Trainer(data_loader, config)
    trainer.train()
# %%
