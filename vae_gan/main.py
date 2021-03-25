# %%
import torch
import numpy
import argparse
import sys

numpy.random.seed(8)
torch.manual_seed(8)
torch.cuda.manual_seed(8)

from network import VaeGan
from torch.autograd import Variable
from torch.utils.data import Dataset, dataset
from tensorboardX import SummaryWriter
from torch.optim import RMSprop, Adam, SGD
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR
import progressbar
from torchvision.utils import make_grid
from generator import CELEBA, CELEBA_SLURM
# 引入指定路径文件
sys.path.append(r"./utils.py")
from utils import RollingMeasure

# %%
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="VAEGAN")
    parser.add_argument("--train_folder", default="H:\data\img_align_celeba", action="store",dest="train_folder")
    parser.add_argument("--test_folder", default="H:\data\img_align_celeba", action="store",dest="test_folder")
    parser.add_argument("--n_epochs",default=12,action="store",type=int,dest="n_epochs")
    parser.add_argument("--z_size",default=128,action="store",type=int,dest="z_size")
    parser.add_argument("--recon_level",default=3,action="store",type=int,dest="recon_level")
    parser.add_argument("--lambda_mse",default=1e-6,action="store",type=float,dest="lambda_mse")
    parser.add_argument("--lr",default=3e-4,action="store",type=float,dest="lr")
    parser.add_argument("--decay_lr",default=0.75,action="store",type=float,dest="decay_lr")
    parser.add_argument("--decay_mse",default=1,action="store",type=float,dest="decay_mse")
    parser.add_argument("--decay_margin",default=1,action="store",type=float,dest="decay_margin")
    parser.add_argument("--decay_equilibrium",default=1,action="store",type=float,dest="decay_equilibrium")
    parser.add_argument("--slurm",default=False,action="store",type=bool,dest="slurm")

    args = parser.parse_args([])

    train_folder = args.train_folder
    test_folder = args.test_folder
    z_size = args.z_size
    recon_level = args.recon_level
    decay_mse = args.decay_mse
    decay_margin = args.decay_margin
    n_epochs = args.n_epochs
    lambda_mse = args.lambda_mse
    lr = args.lr
    decay_lr = args.decay_lr
    decay_equilibrium = args.decay_equilibrium
    slurm = args.slurm
    
    writer = SummaryWriter(comment="_CELEBA_NEW_DATA_STOCK_GAN")
    net = VaeGan(z_size=z_size, recon_level=recon_level).cuda()

    # dataset
    if not slurm:
        dataloader = torch.utils.data.DataLoader(CELEBA(train_folder), batch_size=64,
                                                shuffle=True, num_workers=4)
        # dataset for test
        # if you want to split train from test just move some files in another dir
        dataloader_test = torch.utils.data.DataLoader(CELEBA(test_folder), batch_size=100,
                                                shuffle=False, num_workers=1)
    else:
        dataloader = torch.utils.data.DataLoader(CELEBA_SLURM(train_folder), bath_size=64,
                                                shuffle=True, num_workers=1)
        # dataset for test 
        # if you want to split train from test just move some files in another dir
        dataloader_test = torch.utils.data.DataLoader(CELEBA_SLURM(test_folder), batch_size=100,
                                                shuffle=False, num_workers=1)

    # margin and equilibirum 
    margin = 0.35
    equilibirum = 0.68

    # optim-loss
    # an optimizer for each of the sub-networks, so we can selectively backprop
    optimizer_encoder = RMSprop(params=net.encoder.parameters(), lr=lr, alpha=0.9, eps=1e-8, weight_decay=0, momentum=0, centered=False)
    lr_encoder = ExponentialLR(optimizer_encoder, gamma=decay_lr)

    optimizer_decoder = RMSprop(params=net.decoder.parameters(), lr=lr, alpha=0.9, eps=1e-8, weight_decay=0, momentum=0, centered=False)
    lr_decoder = ExponentialLR(optimizer_decoder, gamma=decay_lr)

    optimizer_discriminator = RMSprop(params=net.discriminator.parameters(), lr=lr, alpha=0.9, eps=1e-8, weight_decay=0, momentum=0, centered=False)
    lr_discriminator = ExponentialLR(optimizer_discriminator, gamma=decay_lr)

    batch_number = len(dataloader)
    step_index = 0
    widgets = [
        'Batch: ', progressbar.Counter(),
        '/', progressbar.FormatCustomText('%(total)s', {"total": batch_number}),
        ' ', progressbar.Bar(marker="-", left='[', right=']'),
        ' ', progressbar.ETA(),
        ' ',
        progressbar.DynamicMessage('loss_nle'),
        ' ',
        progressbar.DynamicMessage('loss_encoder'),
        ' ',
        progressbar.DynamicMessage('loss_decoder'),
        ' ',
        progressbar.DynamicMessage('loss_discriminator'),
        ' ',
        progressbar.DynamicMessage('loss_mse_layer'),
        ' ',
        progressbar.DynamicMessage('loss_kld'),
        ' ',
        progressbar.DynamicMessage("epoch")
    ]

    # for each epoch 
    if slurm:
        print(args)
    for i in range(n_epochs):

        progress = progressbar.ProgressBar(min_value=0, maxval=batch_number, initial_value=0, widgets=widgets).start()

        # reset rolling average 
        loss_nle_mean = RollingMeasure()
        loss_encoder_mean = RollingMeasure()
        loss_decoder_mean = RollingMeasure()
        loss_discriminator_mean = RollingMeasure()
        loss_reconstruction_layer_mean = RollingMeasure()
        loss_kld_mean = RollingMeasure()
        gan_gen_eq_mean = RollingMeasure()
        gan_dis_eq_mean = RollingMeasure()

        # for each batch 
        for j, (data_batch, target_batch) in enumerate(dataloader):

            # set to train mode 
            # train_batch = len(data_batch)
            net.train()
            # target and input are the same images 

            data_target = Variable(target_batch, requires_grad=False).float().cuda()
            data_in = Variable(data_batch, requires_grad=False).float().cuda()

            # get output
            out, out_labels, out_layer, mus, variances = net(data_in)
            # split so we can get the different parts 
            out_layer_predicted = out_layer[:len(out_layer) // 2]
            out_layer_original = out_layer[len(out_layer) // 2:]
            # out_layer_sampled = out_layer[-train_batch]
            # todo: set a batch_len variable to get a clean code here
            out_labels_original = out_labels[:len(out_labels) // 2]
            out_labels_sampled = out_labels[-len(out_labels) // 2:]

            # loss, nothing special here
            nel_value, kl_value, mse_value, bce_dis_original_value, bce_dis_sampled_value, \
                bce_gen_original_value, bce_gen_sampled_value = VaeGan.loss(data_target, out, out_layer_original,
                                                                            out_layer_predicted, 
                                                                            # out_layer_sampled, 
                                                                            out_labels_original, 
                                                                            # out_labels_predicted,
                                                                            out_labels_sampled, mus, variances)

            # this is the most important part of the code 
            loss_encoder = torch.sum(kl_value) + torch.sum(mse_value)
            loss_discriminator = torch.sum(bce_dis_original_value) + torch.sum(bce_dis_sampled_value)
            loss_decoder = torch.sum(lambda_mse * mse_value) - (1.0 - lambda_mse) * loss_discriminator

            # register mean values of the losses for logging 
            loss_nle_mean(torch.mean(nel_value).data.cpu().numpy()[0])
            loss_discriminator_mean((torch.mean(bce_dis_original_value) + torch.mean(bce_dis_sampled_value)).data.cpu().numpy()[0])
            loss_decoder_mean((torch.mean(lambda_mse * mse_value) - (1 - lambda_mse) * (torch.mean(bce_dis_original_value) 
                                + torch.mean(bce_dis_sampled_value))).data.cpu().numpy()[0])    

            loss_encoder_mean((torch.mean(kl_value) + torch.mean(mse_value)).data.cpu().numpy()[0])        
            loss_reconstruction_layer_mean(torch.mean(mse_value).data.cpu().numpy()[0])
            loss_kld_mean(torch.mean(kl_value).data.cpu().numpy()[0])

            # selectively disable the decoder of the discriminator if they are unbalanced 
            train_dis = True
            train_dec = True
            if torch.mean(bce_dis_original_value).data[0] < equilibirum-margin or \
                torch.mean(bce_dis_sampled_value).data[0] < equilibirum-margin:
                train_dis = False
            if torch.mean(bce_dis_original_value).data[0] > equilibirum+margin or \
                torch.mean(bce_dis_sampled_value).data[0] > equilibirum+margin:
                train_dec = False
            if train_dec is False and train_dis is False:
                train_dis = True
                train_dec = True

            # aggiungo log 
            if train_dis:
                gan_dis_eq_mean(1.0)
            else:
                gan_dis_eq_mean(0.0)

            if train_dec:
                gan_gen_eq_mean(1.0)
            else:
                gan_gen_eq_mean(0.0)

            # backprop
            # clean grads 
            net.zero_grad()
            # decoder 
            if train_dec:
                loss_decoder.backward(retain_graph = True)

                optimizer_decoder.step()
                # clean the discriminator 
                net.discriminator.zero_grad()

            # discriminator
            if train_dis:
                loss_discriminator.backward()

                optimizer_discriminator.step()

            # logging
            if slurm:
                progress.update(
                progress.value + 1, loss_nle=loss_nle_mean.measure,
                loss_encoder=loss_encoder_mean.measure,
                loss_decoder=loss_decoder_mean.measure,
                loss_discriminator=loss_discriminator_mean.measure,
                loss_mse_layer=loss_reconstruction_layer_mean.measure,
                loss_kld=loss_kld_mean.measure,
                epoch=i + 1    
                )

        # epoch end 
        if slurm:
            progress.update(
                progress.value + 1, loss_nle=loss_nle_mean.measure,
                loss_encoder=loss_encoder_mean.measure,
                loss_decoder=loss_decoder_mean.measure,
                loss_discriminator=loss_discriminator_mean.measure,
                loss_mse_layer=loss_reconstruction_layer_mean.measure,
                loss_kld=loss_kld_mean.measure,
                epoch=i + 1
            )

        lr_encoder.step()
        lr_decoder.step()
        lr_discriminator.step()

        margin *= decay_margin
        equilibirum *= decay_equilibrium

        # margin non puo essere piu alto di equilibrium
        if margin > equilibirum:
            equilibirum = margin
        lambda_mse *= decay_mse
        if lambda_mse > 1:
            lambda_mse = 1
        progress.finish()

        writer.add_scalar('loss_encoder', loss_encoder_mean.measure, step_index)
        writer.add_scalar('loss_decoder', loss_decoder_mean.measure, step_index)
        writer.add_scalar('loss_discriminator', loss_discriminator_mean.measure, step_index)
        writer.add_scalar('loss_reconstruction', loss_nle_mean.measure, step_index)
        writer.add_scalar('loss_kld',loss_kld_mean.measure,step_index)
        writer.add_scalar('gan_gen',gan_gen_eq_mean.measure,step_index)
        writer.add_scalar('gan_dis',gan_dis_eq_mean.measure,step_index)

        for j, (data_batch, target_batch) in enumerate(dataloader_test):
            net.eval()

            data_in = Variable(data_batch, requires_grad=False).float().cuda()
            data_target = Variable(target_batch, requires_grad=False).float().cuda()
            out = net(data_in)
            out = out.data.cpu()
            out = (out + 1) / 2
            out = make_grid(out, nrow=8)
            writer.add_image("reconstructed", out, step_index)

            out = net(None, 100)
            out = out.data.cpu()
            out = (out + 1) / 2
            out = make_grid(out, nrow=8)
            writer.add_image("generated", out, step_index)

            out = data_target.data.cpu()
            out = (out + 1) / 2
            out = make_grid(out, nrow=8)
            writer.add_image("original", out, step_index)
            break

        step_index += 1
    exit(0)