# https://github.com/hq-deng/RD4AD/blob/main/dataset.py

from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image

import torch
import numpy as np

MVTEC_CLASS_NAMES = ['bottle', 'cable', 'capsule', 'carpet', 'grid',
               'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
               'tile', 'toothbrush', 'transistor', 'wood', 'zipper']


class MVTecADDataset(Dataset):
    def __init__(self, data_dir, gt_dir=None, transform=None, gt_transform=None):
        # gt_dir == None --> train
        # gt_dir != None -->test
        self.transform = transform
        self.gt_transform = gt_transform
        self.data_info = self.get_data_info(data_dir, gt_dir)

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, index):
        img_path, gt, ad_label, ad_type = self.data_info[index]
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if gt == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            if self.gt_transform is not None:
                gt = self.gt_transform(gt)

        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return img, gt, ad_label, ad_type

    def get_data_info(self, data_dir, gt_dir):
        data_info = list()

        for root, dirs, _ in os.walk(data_dir):
            for sub_dir in dirs:
                img_names = os.listdir(os.path.join(root, sub_dir))
                img_names = list(filter(lambda x: x.endswith('.png'), img_names))
                for img_name in img_names:
                    img_path = os.path.join(root, sub_dir, img_name)
                    if sub_dir == 'good':
                        data_info.append((img_path, 0, 0, sub_dir))
                    else:
                        gt_name = img_name.replace(".png", "_mask.png")
                        # gt_name = img_name
                        gt_path = os.path.join(gt_dir, sub_dir, gt_name)
                        data_info.append((img_path, gt_path, 1, sub_dir))

        np.random.shuffle(data_info)

        return data_info


class MVTecADTestDataset(Dataset):
    def __init__(self, data_dir, gt_dir=None, transform=None, gt_transform=None):
        # gt_dir == None --> train
        # gt_dir != None -->test
        self.transform = transform
        self.gt_transform = gt_transform
        self.data_info = self.get_data_info(data_dir, gt_dir)

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, index):
        img_path, gt, ad_label, ad_type = self.data_info[index]
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if gt == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            if self.gt_transform is not None:
                gt = self.gt_transform(gt)

        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return img, gt, ad_label, ad_type, img_path

    def get_data_info(self, data_dir, gt_dir):
        data_info = list()

        for root, dirs, _ in os.walk(data_dir):
            for sub_dir in dirs:
                img_names = os.listdir(os.path.join(root, sub_dir))
                img_names = list(filter(lambda x: x.endswith('.png'), img_names))
                for img_name in img_names:
                    img_path = os.path.join(root, sub_dir, img_name)
                    if sub_dir == 'good':
                        data_info.append((img_path, 0, 0, sub_dir))
                    else:
                        gt_name = img_name.replace(".png", "_mask.png")
                        # gt_name = img_name
                        gt_path = os.path.join(gt_dir, sub_dir, gt_name)
                        data_info.append((img_path, gt_path, 1, sub_dir))

        np.random.shuffle(data_info)

        return data_info
