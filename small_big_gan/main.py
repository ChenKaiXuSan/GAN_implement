# %%
from utils.utils import make_folder

from trainer import Trainer
from utils.utils import *
from dataset.dataset import getdDataset

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse

# %%
def get_parameters():

    parser = argparse.ArgumentParser()

    # Model hyper-parameters
    parser.add_argument('--model', type=str, default='sagan', choices=['sagan', 'qgan'])
    parser.add_argument('--adv_loss', type=str, default='wgan-gp', choices=['wgan-gp', 'hinge', 'wgan-div'])
    parser.add_argument('--img_size', type=int, default=64)
    parser.add_argument('--channels', type=int, default=3, help='number of image channels')
    parser.add_argument('--d_ite_num', type=int, default=1)
    parser.add_argument('--z_dim', type=int, default=120)
    parser.add_argument('--g_n_feat', type=int, default=36)
    parser.add_argument('--d_n_feat', type=int, default=42)

    # Training setting
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--g_lr', type=float, default=3e-4)
    parser.add_argument('--d_lr', type=float, default=3e-4)
    parser.add_argument('--lr_decay', type=float, default=0.95)
    parser.add_argument('--beta1', type=float, default=0.0)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--n_classes', type=int, default=10, help='how many labels in dataset')

    # using pretrained
    parser.add_argument('--pretrained_model', type=int, default=None)

    # Misc
    parser.add_argument('--train', type=str2bool, default=True)
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['mnist', 'cifar10', 'fashion', 'lsun'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=True)

    # Path
    parser.add_argument('--dataroot', type=str, default='../data')
    parser.add_argument('--log_path', type=str, default='./logs')
    parser.add_argument('--model_save_path', type=str, default='./models')
    parser.add_argument('--sample_path', type=str, default='./samples')
    parser.add_argument('--attn_path', type=str, default='./attn')
    parser.add_argument('--version', type=str, default='debug')

    # Step size
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=100)
    parser.add_argument('--model_save_step', type=float, default=1.0)


    return parser.parse_args()

# %%
def main(config):
    # data loader 
    data_loader = getdDataset(config)

    # delete the exists path
    del_folder(config.sample_path, config.version)
    del_folder(config.log_path, config.version)
    del_folder(config.sample_path, config.version + '/real_images')

    # create directories if not exist
    make_folder(config.sample_path, config.version)
    make_folder(config.log_path, config.version)
    # make_folder(config.attn_path, config.version)
    make_folder(config.sample_path, config.version + '/real_images')
    make_folder(config.sample_path, config.version + '/fake_images')

    if config.train:
        if config.model == 'sagan':
            trainer = Trainer(data_loader, config)
        trainer.train()
    
# %% 
if __name__ == '__main__':
    config = get_parameters()
    print(config)
    main(config)
# %%
