from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.parallel
import numpy as np

class DCGAN_D(nn.Module):
    def __init__(self, opt):
        super(DCGAN_D, self).__init__()

        self.label_embedding = nn.Embedding(opt.n_classes, opt.n_classes)
        self.opt = opt

        main = nn.Sequential()
        # input is nc x isize x isize
        main.add_module('initial:{0}-{1}:conv'.format(opt.n_classes + 1, 64),
                        nn.Conv2d(opt.n_classes + 1, 64, 4, 2, 1, bias=False))
        main.add_module('initial:{0}:relu'.format(64),
                        nn.LeakyReLU(0.2, inplace=True))
        csize, cndf = opt.img_size / 2, 64

        # Extra layers
        # for t in range(n_extra_layers):
        #     main.add_module('extra-layers-{0}:{1}:conv'.format(t, cndf),
        #                     nn.Conv2d(cndf, cndf, 3, 1, 1, bias=False))
        #     main.add_module('extra-layers-{0}:{1}:batchnorm'.format(t, cndf),
        #                     nn.BatchNorm2d(cndf))
        #     main.add_module('extra-layers-{0}:{1}:relu'.format(t, cndf),
        #                     nn.LeakyReLU(0.2, inplace=True))

        while csize > 4:
            in_feat = cndf
            out_feat = cndf * 2
            main.add_module('pyramid:{0}-{1}:conv'.format(in_feat, out_feat),
                            nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
            main.add_module('pyramid:{0}:batchnorm'.format(out_feat),
                            nn.BatchNorm2d(out_feat))
            main.add_module('pyramid:{0}:relu'.format(out_feat),
                            nn.LeakyReLU(0.2, inplace=True))
            cndf = cndf * 2
            csize = csize / 2

        # state size. K x 4 x 4
        main.add_module('final:{0}-{1}:conv'.format(cndf, 1),
                        nn.Conv2d(cndf, 1, 4, 1, 0, bias=False))
        self.main = main

        self.fill = torch.zeros([10, 10, opt.img_size, opt.img_size])
        for i in range(10):
            self.fill[i, i, :, :] = 1
        self.fill.cuda()

    def forward(self, img, labels):
        labels_fill = self.fill[labels].cuda()
        d_in = torch.cat((img, labels_fill), 1)
        output = self.main(d_in)
        return output

class DCGAN_G(nn.Module):
    def __init__(self, opt):
        super(DCGAN_G, self).__init__()
        
        isize = opt.img_size
        assert isize % 16 == 0, "isize has to be a multiple of 16"
        nz = opt.latent_dim
        nc = opt.channels
        cngf, tisize = 64 // 2, 4
        self.opt = opt
        while tisize != isize:
            cngf = cngf * 2
            tisize = tisize * 2

        self.label_emb = nn.Embedding(opt.n_classes, opt.n_classes, 1, 1)

        main = nn.Sequential()
        # input is Z, going into a convolution
        main.add_module('initial:{0}-{1}:convt'.format(nz + opt.n_classes, cngf),
                        nn.ConvTranspose2d(nz + opt.n_classes, cngf, 4, 1, 0, bias=False))
        main.add_module('initial:{0}:batchnorm'.format(cngf),
                        nn.BatchNorm2d(cngf))
        main.add_module('initial:{0}:relu'.format(cngf),
                        nn.ReLU(True))

        csize, cndf = 4, cngf
        while csize < isize//2:
            main.add_module('pyramid:{0}-{1}:convt'.format(cngf, cngf//2),
                            nn.ConvTranspose2d(cngf, cngf//2, 4, 2, 1, bias=False))
            main.add_module('pyramid:{0}:batchnorm'.format(cngf//2),
                            nn.BatchNorm2d(cngf//2))
            main.add_module('pyramid:{0}:relu'.format(cngf//2),
                            nn.ReLU(True))
            cngf = cngf // 2
            csize = csize * 2

        # Extra layers
        # for t in range(n_extra_layers):
        #     main.add_module('extra-layers-{0}:{1}:conv'.format(t, cngf),
        #                     nn.Conv2d(cngf, cngf, 3, 1, 1, bias=False))
        #     main.add_module('extra-layers-{0}:{1}:batchnorm'.format(t, cngf),
        #                     nn.BatchNorm2d(cngf))
        #     main.add_module('extra-layers-{0}:{1}:relu'.format(t, cngf),
        #                     nn.ReLU(True))

        main.add_module('final:{0}-{1}:convt'.format(cngf, nc),
                        nn.ConvTranspose2d(cngf, nc, 4, 2, 1, bias=False))
        main.add_module('final:{0}:tanh'.format(nc),
                        nn.Tanh())
        self.main = main

    def forward(self, z, labels):
        labels_emb = self.label_emb(labels)
        gen_input = torch.cat((labels_emb, z), -1).view(z.size(0), -1, 1, 1)
        # labels = self.label_emb(labels).view(opt.batch_size, 110, 1, 1)
        # gen_input = torch.cat((z, self.label_emb(labels)), -1 )

        output = self.main(gen_input)
        output = output.view(output.shape[0], * self.opt.img_shape)
        return output 


# custom weights initialization called on netG and netD
def weights_init(m):
    '''
    初始化conv参数

    Args:
        m (layer): layer名字
    '''    
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
