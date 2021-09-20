# %% 
import os
import time
import torch
import datetime

import torch.nn as nn 
import torchvision
from torchvision.utils import save_image

import numpy as np
import random

# from models.bigGAN import Generator, Discriminator
from models.bigGAN_noattn import Generator, Discriminator
from utils.utils import *

import models.FID as fid

# for draw network
from torchviz import make_dot
from torchvision import models

# %% 
# random seeds 
seed = 2021
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

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
        self.z_dim = config.z_dim
        self.channels = config.channels
        self.g_n_feat = config.g_n_feat
        self.d_n_feat = config.d_n_feat
        self.d_ite_num = config.d_ite_num

        self.epochs = config.epochs
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers 
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr 
        self.lr_decay = config.lr_decay
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.pretrained_model = config.pretrained_model
        self.n_classes = config.n_classes

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

        # self.build_FID()

    def train(self):

        # data iterator 
        # data_iter = iter(self.data_loader)
        # step_per_epoch = len(self.data_loader)

        real_label = 0.9
        fake_label = 0

        D_loss_list = []
        G_loss_list = []

        self.G.train()
        self.D.train()

        # training here 
        for epoch in range(1, self.epochs + 1):
            D_running_loss = 0
            G_running_loss = 0

            # for after to record the one image.
            self.number_fake = 1
            self.number_real = 1

            start_time = time.time()
            for i, (imgs, labels) in enumerate(self.data_loader):
                ##################
                # udpate D network
                ##################
                for _ in range(self.d_ite_num):

                    # because the batch size is not same as the dataloader.
                    self.batch_size = imgs.size(0)
                    # train with real 
                    self.D.zero_grad() # the next set the optimizer zero grad, so there not use it.
                    real_images = imgs.cuda()
                    dis_labels = torch.full((self.batch_size, 1), real_label).cuda() # (*, 1)
                    aux_labels = labels.long().cuda() # (*, )
                    dis_output, preds_real_labels = self.D(real_images, aux_labels) # dis shape (*, 1)

                    errD_real = self.bce_loss(dis_output, dis_labels)
                    errD_real += self.c_loss(preds_real_labels, aux_labels) * 0.1 # coefficient gamma is quite sensitive.
                    errD_real.backward(retain_graph=True)

                    # train with fake 
                    noise = torch.randn(self.batch_size, self.z_dim, 1, 1).cuda()

                    aux_labels = np.random.randint(0, self.n_classes, self.batch_size)
                    aux_labels_ohe = np.eye(self.n_classes)[aux_labels]
                    aux_labels_ohe = torch.from_numpy(aux_labels_ohe[:, :, np.newaxis, np.newaxis])
                    aux_labels_ohe = aux_labels_ohe.float().cuda()

                    aux_labels = torch.from_numpy(aux_labels).long().cuda()

                    fake = self.G(noise, aux_labels_ohe) # (*, 3, 64, 64)

                    dis_labels.fill_(fake_label)
                    dis_output, preds_fake_labels = self.D(fake.detach(), aux_labels)

                    errD_fake = self.bce_loss(dis_output, dis_labels)
                    errD_fake += self.c_loss(preds_fake_labels, aux_labels) * 0.1 # coefficient gamma is quite sensitive.
                    errD_fake.backward(retain_graph=True)

                    # self.reset_grad()
                    
                    errD_total = errD_real.item() + errD_fake.item() # for tensorboard
                    D_running_loss += errD_total / len(self.data_loader)
                    self.d_optimizer.step()

                ##################
                # update G network
                ##################
                self.G.zero_grad() # the next set the optimizer zero grad, so there not use it.
                dis_labels.fill_(real_label) # fake labels are real for generator cost 

                # like up, but refrom because the random noise.
                noise = torch.randn(self.batch_size, self.z_dim, 1, 1).cuda()

                aux_labels = np.random.randint(0, self.n_classes, self.batch_size)
                aux_labels_ohe = np.eye(self.n_classes)[aux_labels]
                aux_labels_ohe = torch.from_numpy(aux_labels_ohe[:, :, np.newaxis, np.newaxis])
                aux_labels_ohe = aux_labels_ohe.float().cuda()

                aux_labels = torch.from_numpy(aux_labels).long().cuda()

                fake = self.G(noise, aux_labels_ohe)

                dis_output, preds_fake_labels = self.D(fake, aux_labels)

                errG = self.bce_loss(dis_output, dis_labels)
                errG += self.c_loss(preds_fake_labels, aux_labels) * 0.1 # coefficient gamma is quite sensitive.
                # self.reset_grad()
                errG.backward(retain_graph = True)

                G_running_loss += errG.item() / len(self.data_loader)
                self.g_optimizer.step()

                # sample one image for FID
                self.save_sample_one_image(real_images, epoch)
            # end one epoch

            # output for display, every 10 epoch.
            if epoch % 10 == 0:
                elapsed = time.time() - start_time
                # elapsed = str(datetime.timedelta(seconds=elapsed) * 60)
                print('[{:d}/{:d}] D_loss = {:.3f}, G_loss = {:.3f}, elapsed_time = {:.1f} min'.format(
                        epoch,self.epochs,D_running_loss,G_running_loss, elapsed / 60))

            # log 
            D_loss_list.append(D_running_loss)
            G_loss_list.append(G_running_loss)

            # for tensorboard, every epoch
            self.logger.add_scalar('D_loss', errD_total, epoch)
            self.logger.add_scalar('G_loss', errG.item(), epoch)

            # sample images 
            self.save_sample(real_images, epoch)
            # # sample one image for FID
            # self.save_sample_one_image(real_images, epoch)

            # save point, every 100 epoch
            if epoch % 100 == 0:
                torch.save(self.G.state_dict(), f'generator_epoch{epoch}.pth')
                torch.save(self.D.state_dict(), 'discriminator.pth')
        # end total epoch

    def build_model(self):

        self.G = Generator(n_feat=self.g_n_feat, codes_dim=24, n_classes=self.n_classes).cuda()
        self.D = Discriminator(n_feat=self.d_n_feat, n_classes=self.n_classes).cuda()

        # loss and optimizer 
        # self.g_optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.G.parameters()), self.g_lr, [self.beta1, self.beta2])
        # self.d_optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.D.parameters()), self.d_lr, [self.beta1, self.beta2])
        
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), lr=self.g_lr, betas=(self.beta1, self.beta2))
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), lr=self.d_lr, betas=(self.beta1, self.beta2))

        # loss function 
        # self.c_loss = nn.CrossEntropyLoss().cuda()
        self.bce_loss = nn.BCELoss().cuda()
        self.c_loss = nn.NLLLoss().cuda()

        # print networks
        print(self.G)
        print(self.D)
        print('G parameters:', count_parameters(self.G))
        print('D parameters:', count_parameters(self.D))

    def build_tensorboard(self):
        from torch.utils.tensorboard import SummaryWriter
        self.logger = SummaryWriter(self.log_path)

    def reset_grad(self):
        self.d_optimizer.zero_grad()
        self.g_optimizer.zero_grad()

    def save_sample(self, real_images, epoch):
        if epoch % 10 == 0:

            # save real image 
            real_image_path = self.sample_path + '/real_images/'
            save_image(real_images[:100].data, os.path.join(real_image_path, '{}_real.png'.format(epoch + 1)),
                         nrow=self.n_classes, normalize=True)

            # save fake image
            fake_image_path = self.sample_path + '/fake_images/'
            
            # for generate sample
            fixed_noise = torch.randn(self.batch_size, self.z_dim, 1, 1).cuda()

            fixed_aux_labels = np.random.randint(0, self.n_classes, self.batch_size)
            fixed_aux_labels_ohe = np.eye(self.n_classes)[fixed_aux_labels]
            fixed_aux_labels_ohe = torch.from_numpy(fixed_aux_labels_ohe[:, :, np.newaxis, np.newaxis])
            fixed_aux_labels_ohe = fixed_aux_labels_ohe.float().cuda()

            with torch.no_grad():
                gen_image = self.G(fixed_noise, fixed_aux_labels_ohe).to('cpu').clone().detach().squeeze(0)
                save_image(gen_image[:100].data,
                        os.path.join(fake_image_path, '{}_fake.png'.format(epoch + 1)), 
                        nrow=self.n_classes, normalize=True)

    # todo 保存的是最后要给mini batch 的图片，有可能是16张图片，不够
    def save_sample_one_image(self, real_images, epoch):
        if epoch % 10 == 0:
            
            make_folder(self.sample_path, str(epoch) + '/real_images')
            make_folder(self.sample_path, str(epoch) + '/fake_images')
            real_images_path = os.path.join(self.sample_path, str(epoch), 'real_images')
            fake_image_path = os.path.join(self.sample_path, str(epoch), 'fake_images')

            if len(os.listdir(real_images_path)) <= 1000:
                # save real image 
                # real_images_path = self.sample_path + '/' + str(epoch) + '/real_images/'
                for i in range(real_images.size(0)):
                    one_real_image = real_images[i]
                    save_image(one_real_image.data,
                            os.path.join(real_images_path, '{}_real.png'.format(self.number_real)), normalize=True
                            )
                    self.number_real += 1

                # save fake image
                # fake_image_path = self.sample_path + '/'+ str(epoch) + '/fake_images/'
                
                # for generate sample
                fixed_noise = torch.randn(real_images.size(0), self.z_dim, 1, 1).cuda()

                fixed_aux_labels = np.random.randint(0, self.n_classes, real_images.size(0))
                fixed_aux_labels_ohe = np.eye(self.n_classes)[fixed_aux_labels]
                fixed_aux_labels_ohe = torch.from_numpy(fixed_aux_labels_ohe[:, :, np.newaxis, np.newaxis])
                fixed_aux_labels_ohe = fixed_aux_labels_ohe.float().cuda()

                with torch.no_grad():
                    gen_image = self.G(fixed_noise, fixed_aux_labels_ohe).to('cpu').clone().detach().squeeze(0)
                    
                    for i in range(gen_image.size(0)):
                        one_fake_image = gen_image[i]
                        save_image(
                            one_fake_image.data,
                            os.path.join(fake_image_path, '{}_fake.png'.format(self.number_fake)), normalize = True
                        )
                        self.number_fake += 1 
                

    
    def build_FID(self):
        block_idx = fid.InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        model = fid.InceptionV3([block_idx])
        self.fid_model=model.cuda()

    def save_image_tensorboard(self, images, text, step):
        if step % 100 == 0:
            img_grid = torchvision.utils.make_grid(images, nrow=8)

            self.logger.add_image(text + str(step), img_grid, step)
            self.logger.close()
    
    def draw_network(self):
        noise = torch.randn(self.batch_size, self.z_dim, 1, 1).cuda()

        generate_img = torch.randn(self.batch_size, self.channels, self.imsize, self.imsize).cuda()

        aux_labels = np.random.randint(0, self.n_classes, self.batch_size)
        aux_labels_ohe = np.eye(self.n_classes)[aux_labels]
        aux_labels_ohe = torch.from_numpy(aux_labels_ohe[:, :, np.newaxis, np.newaxis])
        aux_labels_ohe = aux_labels_ohe.float().cuda()

        aux_labels = torch.from_numpy(aux_labels).long().cuda()
        
        # generator = self.G(noise, aux_labels_ohe)
        # discriminator = self.D(generator, aux_labels)

        # g = make_dot(generator)
        # d = make_dot(discriminator)

        self.logger.add_graph(self.G, [noise, aux_labels_ohe])
        self.logger.add_graph(self.D, [generate_img, aux_labels])
        # g = make_dot(y.mean(), params=dict(model.named_parameters()))
        # g.view()
        # d.view()



# %%
from dataset.dataset import getdDataset
import main

if __name__ == '__main__':
    config = main.get_parameters()
    config.total_step = 1000
    config.img_size = 64
    config.batch_size = 64
    print(config)
    data_loader = getdDataset(config)

    trainer = Trainer(data_loader, config)
    trainer.train()
    # trainer.draw_network()
# %%
