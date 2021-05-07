import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=128, help='input batch size for training')
parser.add_argument('--input_size', default=[3, 64, 64])
parser.add_argument('--beta1', default=0.5, help='Beta1 hyperparam for Adam optimizers')

parser.add_argument('--train_img_dir', type=str, default='../dataset/celeba/train')
parser.add_argument('--train_attr_path', type=str, default='../dataset/celeba/list_attr_celeba_train.txt')
parser.add_argument('--test_img_dir', type=str, default='../dataset/celeba/test')
parser.add_argument('--test_attr_path', type=str, default='../dataset/celeba/list_attr_celeba_test.txt')
parser.add_argument('--selected_attrs', '--list', nargs='+', help='selected attributes for the CelebA dataset',
                        default=['Bald', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Bushy_Eyebrows', 'Eyeglasses', 'Gray_Hair', 'Heavy_Makeup', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'Pale_Skin', 'Receding_Hairline', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Hat'])
parser.add_argument('--celeba_crop_size', type=int, default=178, help='crop size for the CelebA dataset')
parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])

parser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')
parser.add_argument('--num_workers', type=int, default=16)
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed')

parser.add_argument("--z_size",default=128,action="store",type=int,dest="z_size")
parser.add_argument("--recon_level",default=3,action="store",type=int,dest="recon_level")
parser.add_argument("--lambda_mse",default=1e-6,action="store",type=float,dest="lambda_mse")
parser.add_argument("--lr",default=3e-4,action="store",type=float,dest="lr")
parser.add_argument("--decay_lr",default=0.75,action="store",type=float,dest="decay_lr")
parser.add_argument("--decay_mse",default=1,action="store",type=float,dest="decay_mse")
parser.add_argument("--decay_margin",default=1,action="store",type=float,dest="decay_margin")
parser.add_argument("--decay_equilibrium",default=1,action="store",type=float,dest="decay_equilibrium")

parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
# parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
# parser.add_argument("--lr", type=float, default=0.00005, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
# parser.add_argument("--dcgan", action='store_false', help='use MLP')
parser.add_argument("--dataset", type=str, choices=['mnist', 'fashion', 'cifar10'],
                    default='cifar10', help="dataset to use")
parser.add_argument("--dataroot", type=str, default='../data/', help='path to dataset')
parser.add_argument('--w_kld', type=float, default=1)
parser.add_argument('--w_loss_g', type=float, default=0.01)
parser.add_argument('--w_loss_gd', type=float, default=1)

args = parser.parse_args([])
args.cuda = not args.no_cuda and torch.cuda.is_available()

print(args)

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if __name__ == "__main__":
    args