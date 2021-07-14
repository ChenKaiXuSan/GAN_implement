import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=128, help='input batch size for training')
parser.add_argument('--input_size', default=[3, 64, 64])

parser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')
parser.add_argument('--num_workers', type=int, default=16)
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed')

parser.add_argument("--z_size",default=128,action="store",type=int,dest="z_size")
parser.add_argument("--lambda_mse",default=1e-6,action="store",type=float,dest="lambda_mse")
parser.add_argument("--lr",default=0.0002, action="store",type=float,dest="lr")
parser.add_argument("--decay_lr",default=0.75,action="store",type=float,dest="decay_lr")
parser.add_argument("--decay_mse",default=1,action="store",type=float,dest="decay_mse")
parser.add_argument("--decay_margin",default=1,action="store",type=float,dest="decay_margin")
parser.add_argument("--decay_equilibrium",default=1,action="store",type=float,dest="decay_equilibrium")

parser.add_argument("--n_epochs", type=int, default=50, help="number of epochs of training")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--latent_dim", type=int, default=128, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--dataset", type=str, choices=['mnist', 'fashion', 'cifar10'],
                    default='mnist', help="dataset to use")
parser.add_argument('--gpu', type=str, default='1')
parser.add_argument("--n_classes", type=int, default=10, help="number of classes for dataset")

# Path
parser.add_argument('--dataroot', type=str, default='../data', help='path to dataset')
parser.add_argument('--log_path', type=str, default='./logs')
parser.add_argument('--sample_path', type=str, default='./images_local')

parser.add_argument('--version', type=str, default='debug')

parser.add_argument('--real_image', type=str, default='real_image')
parser.add_argument('--generate_image', type=str, default='generate_image')
parser.add_argument('--recon_image', type=str, default='recon_image')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

print(args)

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if __name__ == "__main__":
    args