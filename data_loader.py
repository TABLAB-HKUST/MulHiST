from http import server
from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
import torch
import os
import random
import numpy as np
import cv2
import math
import torchvision.transforms as transforms


# for test
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]


class DataSampling(data.Dataset):
    def __init__(self, opt,transform):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.opt = opt
        self.root = opt.dataroot
        self.train_data = opt.train_data  # the num of training image patches per epoch
       
        
        if self.opt.mode=='train':
            self.slides_A, self.slides_B, self.slides_C, self.slides_D = [], [], [], []
            A_list = [f for f in os.listdir(opt.dataroot) if opt.AF in f]   #AF
            B_list = [f for f in os.listdir(opt.dataroot) if opt.HE in f]   #'HE'
            C_list = [f for f in os.listdir(opt.dataroot) if opt.PAS in f]   #'PAS'
            D_list =  [f for f in os.listdir(opt.dataroot) if opt.MT in f]   #'MT'

            print(A_list, B_list, C_list,D_list)
            A_list.sort()
            B_list.sort()
            C_list.sort()
            D_list.sort()
            for file_a, file_b, file_c, file_d in zip(A_list, B_list, C_list, D_list):
                slide_A = cv2.imread(os.path.join(opt.dataroot, file_a))
                slide_B = cv2.imread(os.path.join(opt.dataroot, file_b))
                slide_C = cv2.imread(os.path.join(opt.dataroot, file_c))
                slide_D = cv2.imread(os.path.join(opt.dataroot, file_d))
                ha, wa, _ = slide_A.shape
                hb, wb, _ = slide_B.shape
                hc, wc, _ = slide_C.shape
                hd, wd, _ = slide_D.shape
                # in case of different magnification between unstained WSI and stained WSI.
                slide_A = cv2.resize(slide_A, (int(wa*opt.scale_A), int(ha*opt.scale_A)))
                slide_B = cv2.resize(slide_B, (int(wb*opt.scale_B), int(hb*opt.scale_B)))
                slide_C = cv2.resize(slide_C, (int(wc*opt.scale_B), int(hc*opt.scale_B)))
                slide_D = cv2.resize(slide_D, (int(wd*opt.scale_B), int(hd*opt.scale_B)))
    
                self.slides_A.append(slide_A)
                self.slides_B.append(slide_B)
                self.slides_C.append(slide_C)
                self.slides_D.append(slide_D)

        else:
            self.dir_A = os.path.join(opt.test_dataroot, 'testA-256')  # create a path '/path/to/data/trainA'     
            self.A_paths = sorted(make_dataset(self.dir_A, 50000))   # load images from '/path/to/data/trainA'
            self.A_size = len(self.A_paths)  # get the size of dataset A
        self.transform = transform

    

    def __getitem__(self, index):

        if self.opt.mode=='train':
            r = random.randint(0,self.opt.c_dim +1) 
            index = random.randint(0, len(self.slides_A)-1)
            label = torch.zeros([self.opt.c_dim], dtype=torch.float32)

            if r == 0:
                slide = self.slides_A[index]
                label[0] = 1.
            elif r == 1:
                slide = self.slides_B[index]
                label[1] = 1.
            elif r == 2:
                slide = self.slides_C[index]
                label[2] = 1.
            else:
                slide = self.slides_D[index]
                label[3] = 1.

            h,w,_ = slide.shape
            # random extract image patch from WSIs.
            cor_x, cor_y = random.randint(0, w - self.opt.crop_size), random.randint(0, h - self.opt.crop_size)
            img = slide[cor_y:cor_y+self.opt.crop_size, cor_x:cor_x+self.opt.crop_size]
            img = self.transform(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
            
            return img, torch.FloatTensor(label)

        else:
            ### for testing
            A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
            A_img = Image.open(A_path).convert('RGB')
            A_name = A_path.split('/')[-1].split('.')[0]

            # apply image transformation
            A = self.transform(A_img)
            label = np.zeros([self.opt.c_dim, 1])
            label[0] = 1.     
            return A, torch.FloatTensor(label), A_name      


  
    def __len__(self):
        if self.opt.mode=='train':
            return self.train_data
        else:
            return self.A_size


def get_loader(config):
            # image_dir, attr_path, selected_attrs, crop_size=178, image_size=128, 
            #    batch_size=16, dataset='CelebA', mode='train', num_workers=1):
    """Build and return a data loader."""
    transform = []
    if config.mode == 'train':
        transform.append(T.RandomHorizontalFlip())
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)


    dataset = DataSampling(config, transform)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=config.batch_size,
                                  shuffle=(config.mode=='train'),
                                  num_workers=config.num_workers)
    return data_loader