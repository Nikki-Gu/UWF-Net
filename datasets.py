import glob
import random
import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, '%s/A' % mode) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, '%s/B' % mode) + '/*.*'))

    def __getitem__(self, index):
        item_A = self.transform(pil_loader(self.files_A[index % len(self.files_A)]))
        item_B = self.transform(pil_loader(self.files_B[random.randint(0, len(self.files_B) - 1)]))
        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return len(self.files_B)


class ImageDatasetfortest(Dataset):
    def __init__(self, root, transforms_=None):
        self.transform = transforms.Compose(transforms_)
        self.files_A = sorted(glob.glob(root + '/*.*'))
        
    def __getitem__(self, index):
        item_A = self.transform(pil_loader(self.files_A[index % len(self.files_A)]))

        return {'A': item_A}

    def __len__(self):
        return len(self.files_A)