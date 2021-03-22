# %%
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import torch
import torchvision.transforms as transform
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
# %%
def getdDataset(opt):

    if opt.dataset == 'mnist':
        dst = datasets.MNIST(
            '../../data/',
            train=True,
            download=True,
            transform=transform.Compose(
                [transform.Resize(opt.img_size), transform.ToTensor(), transform.Normalize([0.5], [0.5])]
            )
        )
    elif opt.dataset == 'fashion':
        dst = datasets.FashionMNIST(
            '../../data/',
            train=True,
            download=True,
            # split='mnist',
            transform=transform.Compose(
                [transform.Resize(opt.img_size), transform.ToTensor(), transform.Normalize([0.5], [0.5])]
            )
        )

    dataloader = DataLoader(
        dst,
        batch_size=opt.batch_size, 
        shuffle=True,
    )

    return dataloader

# %%
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    class opt:
        dataset = 'mnist'
        img_size = 32
        batch_size = 10

    dataloader = getdDataset(opt)
    for i, (imgs, labels) in enumerate(dataloader):
        print(i, imgs.shape, labels.shape)
        print(labels)

        img = imgs[0]
        img = img.numpy()
        img = make_grid(imgs, normalize=True).numpy()
        img = np.transpose(img, (1, 2, 0))

        plt.imshow(img)
        plt.show()
        plt.close()
        break
# %%
