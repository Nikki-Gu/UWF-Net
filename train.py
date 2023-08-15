#!/usr/bin/python3
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import itertools
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
from UWFNet.models.models import Generator
from UWFNet.models.models import Discriminator, diseaseLoss
from utils import ReplayBuffer, Logger
from utils import LambdaLR
from utils import weights_init_normal
from datasets import ImageDataset

seed = 3407
torch.manual_seed(seed)            
torch.cuda.manual_seed(seed)      
torch.cuda.manual_seed_all(seed) 

parser = argparse.ArgumentParser()
parser.add_argument('--pretrained', type=bool, default=False, help='fine tune Model')
parser.add_argument('--ckpt', type=str, default="/CKPT/PATH",help='path of pretrained model')
parser.add_argument('--save_dir', type=str, default="/SAVEDIR/",help='save dir of trained model')
parser.add_argument('--model_name', type=str, default="My_model_1",help='save name of your model')
parser.add_argument('--epoch', type=int, default=1, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=2000, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=2, help='size of the batches')
parser.add_argument('--dataroot', type=str, default="HERE/IS/YOUR/DATADIR/", help='root directory of the dataset')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=300, help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--size', type=int, default=256, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--cuda', default=True, help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--lambda_DCLoss', type=float, default=40, help='weight for DC Loss')
opt = parser.parse_args()
print(opt)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    ###### Definition of variables ######
    # Networks
    netG_A2B = Generator(opt.input_nc, opt.output_nc)
    netG_B2A = Generator(opt.output_nc, opt.input_nc)
    netD_A = Discriminator(opt.input_nc)
    netD_B = Discriminator(opt.output_nc)

    # Optimizers
    optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                                   lr=opt.lr, betas=(0.5, 0.999))
    optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    # put on GPU
    if opt.cuda:
        netG_A2B.cuda()
        netG_B2A.cuda()
        netD_A.cuda()
        netD_B.cuda()

    if opt.pretrained:
        checkpoint = torch.load(opt.ckpt)
        netG_A2B.load_state_dict(checkpoint['netG_A2B_state_dict'])
        netG_B2A.load_state_dict(checkpoint['netG_B2A_state_dict'])
        netD_A.load_state_dict(checkpoint['netD_A_state_dict'])
        netD_B.load_state_dict(checkpoint['netD_B_state_dict'])
        optimizer_G.load_state_dict(checkpoint['optimizer_G'])
        optimizer_D_A.load_state_dict(checkpoint['optimizer_D_A'])
        optimizer_D_B.load_state_dict(checkpoint['optimizer_D_B'])
    else:
        # init weights
        netG_A2B.apply(weights_init_normal)
        netG_B2A.apply(weights_init_normal)
        netD_A.apply(weights_init_normal)
        netD_B.apply(weights_init_normal)

    # Losses
    criterion_GAN = torch.nn.MSELoss()
    criterion_identity = torch.nn.L1Loss()
    criterion_DC = diseaseLoss().cuda()

    # LR schedulers
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch,
                                                                                       opt.decay_epoch).step)
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch,
                                                                                           opt.decay_epoch).step)
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch,
                                                                                           opt.decay_epoch).step)

    # Inputs & targets memory allocation on GPU
    Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
    input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
    enhanced_input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
    input_B = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)
    target_real = Variable(Tensor(opt.batchSize).fill_(1.0), requires_grad=False)
    target_fake = Variable(Tensor(opt.batchSize).fill_(0.0), requires_grad=False)

    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    # Dataset loader
    transforms_ = [transforms.RandomCrop(opt.size),
                   transforms.RandomHorizontalFlip(),
                   transforms.ToTensor(),
                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                   ]
    dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, unaligned=True, mode=''),
                            batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu, drop_last=True,
                            pin_memory=True)
    train_num_batches = len(dataloader)

    # Loss plot
    logger = Logger(opt.epoch, opt.n_epochs, len(dataloader))

    ###### Training ######
    for epoch in range(opt.epoch, opt.n_epochs):
        netG_A2B.train()
        netG_B2A.train()
        netD_A.train()
        netD_B.train()
        for n, batch in enumerate(dataloader):
            # Set model input
            real_A = Variable(input_A.copy_(batch['A']))
            real_B = Variable(input_B.copy_(batch['B']))

            ###### Generators A2B and B2A ######
            optimizer_G.zero_grad()

            # Identity loss
            # G_A2B(B) should equal B if real B is fed
            same_B = netG_A2B(real_B)
            loss_identity_B = criterion_identity(same_B, real_B) * 5.0
            # G_B2A(A) should equal A if real A is fed
            same_A = netG_B2A(real_A)
            loss_identity_A = criterion_identity(same_A, real_A) * 5.0

            # GAN loss
            fake_B = netG_A2B(real_A)
            pred_fake = netD_B(fake_B)
            loss_GAN_A2B = criterion_GAN(pred_fake, target_real)

            fake_A = netG_B2A(real_B)
            pred_fake = netD_A(fake_A)
            loss_GAN_B2A = criterion_GAN(pred_fake, target_real)

            # Cycle loss DC loss
            recovered_A = netG_B2A(fake_B)
            loss_cycle_ABA = criterion_DC(recovered_A, real_A) * opt.lambda_DCLoss

            recovered_B = netG_A2B(fake_A)
            loss_cycle_BAB = criterion_DC(recovered_B, real_B) * opt.lambda_DCLoss
            loss_cycle = loss_cycle_ABA + loss_cycle_BAB

            # Total loss
            loss_G = loss_identity_A + loss_identity_B \
                     + loss_GAN_A2B + loss_GAN_B2A \
                     + loss_cycle
            loss_G.backward()

            optimizer_G.step()
            ###################################

            ###### Discriminator A ######
            optimizer_D_A.zero_grad()

            # Real loss
            pred_real = netD_A(real_A)
            loss_D_real = criterion_GAN(pred_real, target_real)

            # Fake loss
            fake_A = fake_A_buffer.push_and_pop(fake_A)
            pred_fake = netD_A(fake_A.detach())
            loss_D_fake = criterion_GAN(pred_fake, target_fake)

            # Total loss
            loss_D_A = (loss_D_real + loss_D_fake) * 0.5
            loss_D_A.backward()

            optimizer_D_A.step()
            ###################################

            ###### Discriminator B ######
            optimizer_D_B.zero_grad()

            # Real loss
            pred_real = netD_B(real_B)
            loss_D_real = criterion_GAN(pred_real, target_real)

            # Fake loss
            fake_B = fake_B_buffer.push_and_pop(fake_B)
            pred_fake = netD_B(fake_B.detach())
            loss_D_fake = criterion_GAN(pred_fake, target_fake)

            # Total loss
            loss_D_B = (loss_D_real + loss_D_fake) * 0.5
            loss_D_B.backward()

            optimizer_D_B.step()
            ###################################

            loss = loss_G + loss_D_A + loss_D_B
            logger.log(losses={'loss_G': loss_G, 'loss_G_identity': (loss_identity_A + loss_identity_B),
                               'loss_G_GAN': (loss_GAN_A2B + loss_GAN_B2A),
                               'loss_G_cycle': (loss_cycle),
                               'loss_D': (loss_D_A + loss_D_B),
                               'loss': loss},
                       images={'real_A': real_A, 'real_B': real_B,
                               'fake_A': fake_A, 'fake_B': fake_B})

        ###################################
        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()

        # Save models checkpoints
        #
        if epoch > 0:
            torch.save({'netG_A2B_state_dict': netG_A2B.state_dict(),
                        'netG_B2A_state_dict': netG_B2A.state_dict(),
                        'netD_A_state_dict': netD_A.state_dict(),
                        'netD_B_state_dict': netD_B.state_dict(),
                        'optimizer_G': optimizer_G.state_dict(),
                        'optimizer_D_A': optimizer_D_A.state_dict(),
                        'optimizer_D_B': optimizer_D_B.state_dict()},
                       opt.save_dir + opt.model_name + '-' + str(epoch) + '.tar')
