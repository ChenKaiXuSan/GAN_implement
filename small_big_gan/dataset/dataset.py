# %%
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import torch
import torchvision.transforms as transform
from torchvision.transforms.transforms import RandomResizedCrop, Resize
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
# %%
def getdDataset(opt):

    if opt.dataset == 'mnist':
        dst = datasets.MNIST(
            # 相对路径，以调用的文件位置为准
            root=opt.dataroot,
            train=True,
            download=True,
            transform=transform.Compose(
                [transform.Resize(opt.img_size), transform.ToTensor(), transform.Normalize([0.5], [0.5])]
            )
        )
    elif opt.dataset == 'fashion':
        dst = datasets.FashionMNIST(
            root=opt.dataroot,
            train=True,
            download=True,
            # split='mnist',
            transform=transform.Compose(
                [transform.Resize(opt.img_size), transform.ToTensor(), transform.Normalize([0.5], [0.5])]
            )
        )
    elif opt.dataset == 'cifar10':
        dst = datasets.CIFAR10(
            root=opt.dataroot,
            train=True,
            download=True,
            transform=transform.Compose(
                [transform.Resize(opt.img_size), transform.ToTensor(), transform.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
            )
        )
    elif opt.dataset == 'lsun':
        dst = datasets.LSUN(
            root=opt.dataroot, 
            classes=['bedroom_train'],
            transform=transform.Compose(
                [transform.Resize([opt.img_size, opt.img_size]),
                transform.ToTensor(), 
                # transform.RandomCrop(size=([opt.img_size, opt.img_size])),
                transform.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
            )
        )

    dataloader = DataLoader(
        dst,
        batch_size=opt.batch_size, 
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True
    )

    return dataloader

# %%
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil

if __name__ == "__main__":
    class opt:
        # dataroot = '/home/xchen/data/'
        dataroot = '../../data/'
        dataset = 'cifar10'
        img_size = 64
        batch_size = 64
        num_workers = 4
        epoch = 10

    dataloader = getdDataset(opt)

    for i, (imgs, labels) in enumerate(dataloader):
        print(i, imgs.shape, labels.shape)
        print(labels)

        save_sample_one_image(imgs, i)

        img = imgs[0]
        img = img.numpy()
        img = make_grid(imgs, normalize=True).numpy()
        img = np.transpose(img, (1, 2, 0))

        plt.imshow(img)
        plt.show()
        plt.close()
        break
# %%
def save_sample_one_image(real_images, epoch):
    shutil.rmtree('epoch/real_images')
    os.makedirs('epoch/real_images')
    # save real image 
    real_images_path = './epoch/real_images/'
    for i in range(real_images.size(0)):
        one_image = real_images[i]
        save_image(one_image.data,
                os.path.join(real_images_path, '{}_real.png'.format(i+1)), normalize=False
                    )
# %%
