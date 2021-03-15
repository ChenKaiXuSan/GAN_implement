# %%
import torch
import torchvision.transforms as transform
from torchvision.utils import save_image
from torch.utils.data import DataLoader, dataloader
from torchvision import datasets
# %%
def getdDataset(opt):

    dst = datasets.MNIST(
        '../data/mnist',
        train=True,
        download=False,
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