#!/usr/bin/python3
import os
import argparse
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import torch
from UWFNet.models.models import Generator
from datasets import ImageDatasetfortest
from tqdm import tqdm 
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
from UWFNet.models.models import DenseNet121
from collections import Counter


parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--dataroot', type=str, default=r'', help='root directory of the dataset')
parser.add_argument('--model_dir', type=str, default=r'', help='trained enhancing model_dir of model checkpoints')
parser.add_argument('--model_name', type=str, default=r'', help='model checkpoints name')
parser.add_argument('--save_dir', type=str, default=r'', help='images results save dir')
parser.add_argument('--output_exist', action="store_true", help='If the enhenced images already stored')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--cuda', default=True, help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=1, help='number of cpu threads to use during batch generation')
opt = parser.parse_args()

def get_myFIQA(gen_img):
    QA_model = DenseNet121(n_class=3)
    state = torch.load("YOURDIR/UWFQA.tar") 
    QA_model.load_state_dict(state['state_dict'])
    QA_model.cuda().eval()
    output2 = QA_model(gen_img)
    result2 = output2.argmax(dim=1).cpu().numpy()[0]
    return result2

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

if __name__ == '__main__':
    bestFIQA = 0.0
    bestmodel = ''
    torch.multiprocessing.set_start_method('spawn')
    netG_A2B = Generator(opt.input_nc, opt.output_nc)
    netG_A2B.cuda()
    netG_B2A = Generator(opt.input_nc, opt.output_nc)
    netG_B2A.cuda()
    
    # 多卡训练设置    
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        netG_A2B = torch.nn.DataParallel(netG_A2B)
        netG_B2A = torch.nn.DataParallel(netG_B2A)
        
    if not opt.output_exist:
        checkpoint = torch.load(os.path.join(opt.model_dir, opt.model_name))
        netG_A2B.load_state_dict(checkpoint['netG_A2B_state_dict'])
        netG_A2B.eval()
        netG_B2A.load_state_dict(checkpoint['netG_B2A_state_dict'])
        netG_B2A.eval()

    transforms_ = [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    test_data = ImageDatasetfortest(opt.dataroot, transforms_=transforms_, mode='')
    dataloader = DataLoader(test_data, batch_size=opt.batchSize, shuffle=False, num_workers=opt.n_cpu)

    myqa_all = []
                    
    with torch.no_grad():
        with tqdm(total=int(len(test_data) / opt.batchSize)) as pbar:
            for i, batch in enumerate(dataloader):
                real_A = batch['A']
                real_A = real_A.cuda()
                
                if opt.output_exist:
                    output = real_A
                else:
                    output = netG_A2B(real_A)
                    ## 保存图片 ## 
                    if not os.path.exists(opt.save_dir):
                        os.makedirs(opt.save_dir)
                    save_path1 = os.path.join(opt.save_dir, os.path.split(dataloader.dataset.files_A[i])[-1])
                    save_image(output*0.5+0.5, save_path1)

                myqa = get_myFIQA(output)
                myqa_all.append(myqa)
                pbar.update()

        test_img_nums = len(test_data)
        myqa = Counter(myqa_all)
        print("UWFQA:", myqa[0], myqa[1], myqa[2])
        MyQA = round(myqa[2]/test_img_nums, 3)
        print("UWFQA rate", MyQA) # generated image FIQA rate of good imgs