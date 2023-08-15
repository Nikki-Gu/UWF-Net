#!/usr/bin/python3
import os
import argparse
import torchvision.transforms as transforms
import torch
from tqdm import tqdm 
import torch
from tqdm import tqdm
from collections import Counter
from models.MCF_Net import dense121_mcs
from models.MCF_Net import DatasetGenerator


parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--save_dir', type=str, default='/YOUR/TEST/IMAGE/DIR/', help='test images results save dir')
opt = parser.parse_args()
print(opt)


QA_model = dense121_mcs(n_class=3)
state = torch.load('/PATH/OF/MCF_NET/MODEL/') # See https://github.com/hzfu/EyeQ for trained model downloading.
QA_model.load_state_dict(state['state_dict'])
QA_model.cuda().eval()
result_all = []

transformList2 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
])

transform_list_val1 = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
])

data_test = DatasetGenerator(data_dir=opt.save_dir, transform1=transform_list_val1,
                                transform2=transformList2, n_class=3, set_name='test')
test_loader = torch.utils.data.DataLoader(dataset=data_test, batch_size=opt.batchSize, shuffle=False, num_workers=0,
                                            pin_memory=True)

with torch.no_grad():
    with tqdm(total=int(len(data_test))) as pbar:
        for epochID, (imagesA, imagesB, imagesC) in enumerate(test_loader):
            # uncomment the following 3 lines if CUDA is enabled (GPU)
            imagesA = imagesA.cuda()
            imagesB = imagesB.cuda()
            imagesC = imagesC.cuda()
            _, _, _, _, result_mcs = QA_model(imagesA, imagesB, imagesC)
            result = result_mcs.argmax(dim=1).cpu().numpy()[0]
            result_all.append(result)
            pbar.update()
qa = Counter(result_all)
print("fiqa: bad, medium, good", qa[2], qa[1], qa[0])
score = round(qa[0]/len(data_test), 3)
print("fiqa", score) # generated image FIQA rate of good imgs