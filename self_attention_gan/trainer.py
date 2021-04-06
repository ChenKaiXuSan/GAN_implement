# %% 
import os 
import time
import torch
import datetime

import torch.nn as nn 
from torch.autograd import Variable
from torch.utils.data import dataloader
from torchvision.utils import save_image

import sys 
sys.path.append('.')
sys.path.append('..')

from self_attention_gan.models.sagan import Generator, Discriminator
from self_attention_gan.utils.utils import *
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

        self.build_model()

        if self.use_tensorboard:
            self.build_tensorboard()

    def train(self):

        # data iterator 
        data_iter = iter(self.data_loader)
        step_per_epoch = len(self.data_loader)

        # fixed input for debugging
        fixed_z = tensor2var(torch.randn(self.batch_size, self.z_dim))

        # start time
        start_time = time.time()
        for step in range(self.total_step):

            # ==================== Train D ==================
            self.D.train()
            self.G.train()

            try:
                real_images, _ = next(data_iter)
            except:
                data_iter = iter(self.data_loader)
                real_images, _ = next(data_iter)

            # compute loss with real images 
            # dr1, dr2, df1, df2, gf1, gf2 are attention scores 
            real_images = tensor2var(real_images)
            d_out_real, dr1, dr2 = self.D(real_images)
            if self.adv_loss == 'wgan-gp':
                d_loss_real = -torch.mean(d_out_real)
            elif self.adv_loss == 'hinge':
                d_loss_real = torch.nn.ReLU()(1.0 - d_out_real).mean()

            # apply Gumbel Softmax
            z = tensor2var(torch.randn(real_images.size(0), self.z_dim))
            fake_images, gf1, gf2 = self.G(z)
            d_out_fake, df1, df2 = self.D(fake_images)

            if self.adv_loss == 'wgan-gp':
                d_loss_fake = d_out_fake.mean()
            elif self.adv_loss == 'hinge':
                d_loss_fake = torch.nn.ReLU()(1.0 + d_out_fake).mean()

            
            # backward + optimize 
            d_loss = d_loss_real + d_loss_fake
            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()

            if self.adv_loss == 'wgan-gp':
                # compute gradient penalty
                alpha = torch.rand(real_images.size(0), 1, 1, 1).cuda().expand_as(real_images)
                interpolated = Variable(alpha * real_images.data + (1 - alpha) * fake_images.data, requires_grad = True)
                out, _, _ = self.D(interpolated)

                grad = torch.autograd.grad(
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
                d_loss = self.lambda_gp * d_loss_gp

                self.reset_grad()
                d_loss.backward()
                self.d_optimizer.step()

            # =================== Train G and gumbel =====================
            # create random noise 
            z = tensor2var(torch.randn(real_images.size(0), self.z_dim))
            fake_images, _, _ = self.G(z)

            # compute loss with fake images 
            g_out_fake, _, _ = self.D(fake_images) # batch x n
            if self.adv_loss == 'wgan-gp':
                g_loss_fake = - g_out_fake.mean()

            self.reset_grad()
            g_loss_fake.backward()
            self.g_optimizer.step()

            # print out log info
            if (step + 1) % self.log_step == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))
                print("Elapsed [{}], G_step [{}/{}], D_step[{}/{}], d_out_real: {:.4f}, "
                      " ave_gamma_l3: {:.4f}, ave_gamma_l4: {:.4f}".
                      format(elapsed, step + 1, self.total_step, (step + 1),
                             self.total_step , d_loss_real.item(),
                             self.G.attn1.gamma.mean().item(), self.G.attn2.gamma.mean().item() ))

            # sample images 
            if (step + 1) % self.sample_step == 0:
                fake_images, _, _ = self.G(fixed_z)
                save_image(denorm(fake_images.data),
                            os.path.join(self.sample_path, '{}_fake.png'.format(step + 1)))



    def build_model(self):

        self.G = Generator(self.batch_size, self.imsize, self.z_dim, self.g_conv_dim).cuda()
        self.D = Discriminator(self.batch_size, self.imsize, self.d_conv_dim).cuda()
        
        # loss and optimizer 
        self.g_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.G.parameters()), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.D.parameters()), self.d_lr, [self.beta1, self.beta2])

        self.c_loss = torch.nn.CrossEntropyLoss()

        # print networks
        print(self.G)
        print(self.D)

    def build_tensorboard(self):
        from torch.utils.tensorboard import SummaryWriter
        self.logger = SummaryWriter(self.log_path)

    def reset_grad(self):
        self.d_optimizer.zero_grad()
        self.g_optimizer.zero_grad()

    def save_sample(self, data_iter):
        real_images, _ = next(data_iter)
        save_image(denorm(real_images). os.path.join(self.sample_path, 'real.png'))