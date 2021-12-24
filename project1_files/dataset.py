import os

import torch
from torch.utils.data import Dataset
from torchvision import transforms

import image


class RGBDataset(Dataset):
    def __init__(self, img_dir, has_gt):
        """
        In:
            img_dir: string, path of train, val or test folder.
            has_gt: bool, indicating if the dataset has ground truth masks.
        Out:
            None.
        Purpose:
            Initialize instance variables.
        Hint:
            Check __getitem__() and add more instance variables to initialize what you need in this method.
        """
        # Input normalization info to be used in transforms.Normalize()
        mean_rgb = [0.722, 0.751, 0.807]
        std_rgb = [0.171, 0.179, 0.197]

        self.img_dir = img_dir
        self.has_gt = has_gt
        
        # TODO: transform to be applied on a sample.
        #  For this homework, compose transforms.ToTensor() and transforms.Normalize() for RGB image should be enough.
        self.transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=mean_rgb, std=std_rgb)])
        #self.transform_gt = transforms.Compose([torch.LongTensor()])
        
        # TODO: number of samples in the dataset.
        #  You'd better not hard code the number,
        #  because this class is used to create train, validation and test dataset.
        path = self.img_dir+'rgb'
        dir_path = os.listdir(path)

        len_dataset= 0
        for file in dir_path:
            if '.png' in file:
                len_dataset += 1
        
        self.dataset_length = len_dataset

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):
        """
        In:
            idx: int, index of each sample, in range(0, dataset_length).
        Out:
            sample: a dictionary that stores paired rgb image and corresponding ground truth mask (if available).
                    rgb_img: Tensor [3, height, width]
                    target: Tensor [height, width], use torch.LongTensor() to convert.
        Purpose:
            Given an index, return paired rgb image and ground truth mask as a sample.
        Hint:
            Use image.read_rgb() and image.read_mask() to read the images.
            Think about how to associate idx with the file name of images.
        """
        # TODO: read RGB image and ground truth mask, apply the transformation, and pair them as a sample.
        
        # a. reading in rgb_img and gt_mask        
        img_name= str(idx)+'_rgb.png'
        #print("filepath: ", self.img_dir+'rgb/'+img_name)
        rgb_img= image.read_rgb(self.img_dir+'rgb/'+img_name)

        img_name= str(idx)+'_gt.png'
        gt_mask= image.read_mask(self.img_dir+'gt/'+img_name)
        
        # applying transformation
        rgb_img= self.transform(rgb_img)
        
        # b. pairing rgb and corr gt image (if exists)
        if self.has_gt is False:
            sample = {'input': rgb_img}
        else:
            #gt_mask= self.transform_gt(gt_mask)
            gt_mask= torch.LongTensor(gt_mask)
            sample = {'input': rgb_img, 'target': gt_mask}
        
        # c. applying transformation
        #sample= self.transform(sample)
                
        return sample
