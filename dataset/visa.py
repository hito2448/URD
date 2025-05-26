# https://github.com/hq-deng/RD4AD/blob/main/dataset.py

from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image

from torchvision import transforms
import torch
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST, ImageFolder
import numpy as np

import cv2
import glob
import imgaug.augmenters as iaa
from utils.perlin import perlin_noise, get_forehead_mask


class VisANoiseDataset(Dataset):   # for train with anomaly images(add noise)
    def __init__(self, data_dir, anomaly_source_path=None, transform=None, gt_transform=None, rotate_90=False, random_rotate=0, mask=False, classname=None):
        # gt_dir == None --> train
        # gt_dir != None -->test
        self.transform = transform
        self.gt_transform = gt_transform

        self.resize_shape = [256, 256]

        self.mask = mask

        self.classname = classname

        self.data_info = self.get_data_info(data_dir)

        self.anomaly_source_paths = sorted(glob.glob(anomaly_source_path + "/*/*.jpg"))

        self.rotate_90 = rotate_90
        self.random_rotate = random_rotate


    def __len__(self):
        return len(self.data_info)

    def get_data_info(self, data_dir):
        data_info = list()

        for root, dirs, _ in os.walk(data_dir):
            for sub_dir in dirs:
                img_names = os.listdir(os.path.join(root, sub_dir))
                img_names = list(filter(lambda x: x.endswith('.JPG'), img_names))
                for img_name in img_names:
                    # print(img_name)
                    img_path = os.path.join(root, sub_dir, img_name)

                    if self.mask is False:
                        forehead_mask = None
                    else:
                        if self.classname in ['capsule', 'screw']:
                            # print(img_path)
                            image = cv2.imread(img_path)
                            image = cv2.resize(image, self.resize_shape)
                            forehead_mask = get_forehead_mask(image, classname=self.classname)
                        else:
                            image = cv2.imread(img_path)
                            image = cv2.resize(image, self.resize_shape)
                            forehead_mask = get_forehead_mask(image, classname=None)

                    data_info.append((img_path, 0, 0, sub_dir, forehead_mask))

        np.random.shuffle(data_info)

        return data_info

    def __getitem__(self, index):
        img_path, gt, ad_label, ad_type, forehead_mask = self.data_info[index]
        image = Image.open(img_path).convert("RGB")
        image = image.resize(self.resize_shape, Image.BILINEAR)

        anomaly_source_idx = torch.randint(0, len(self.anomaly_source_paths), (1,)).item()
        dtd_image = Image.open(self.anomaly_source_paths[anomaly_source_idx]).convert("RGB")
        dtd_image = dtd_image.resize(self.resize_shape, Image.BILINEAR)
        noise_image = dtd_image

        # perlin_noise implementation
        aug_image, aug_mask = perlin_noise(image, noise_image, aug_prob=1.0, mask=forehead_mask)

        if self.transform is not None:
            image = self.transform(image)
        if self.transform is not None:
            aug_image = self.transform(aug_image)
        if self.gt_transform is not None:
            aug_mask = self.gt_transform(aug_mask)

        return image, aug_image, aug_mask


class VisADataset(Dataset):
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
                img_names = list(filter(lambda x: x.endswith('.JPG'), img_names))
                for img_name in img_names:
                    img_path = os.path.join(root, sub_dir, img_name)
                    if sub_dir == 'good':
                        data_info.append((img_path, 0, 0, sub_dir))
                    else:
                        gt_name = img_name.replace(".JPG", ".png")
                        gt_path = os.path.join(gt_dir, sub_dir, gt_name)
                        data_info.append((img_path, gt_path, 1, sub_dir))

        np.random.shuffle(data_info)

        return data_info


class VisATestDataset(Dataset):
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
                img_names = list(filter(lambda x: x.endswith('.JPG'), img_names))
                for img_name in img_names:
                    img_path = os.path.join(root, sub_dir, img_name)
                    if sub_dir == 'good':
                        data_info.append((img_path, 0, 0, sub_dir))
                    else:
                        gt_name = img_name.replace(".JPG", ".png")
                        # gt_name = img_name
                        gt_path = os.path.join(gt_dir, sub_dir, gt_name)
                        data_info.append((img_path, gt_path, 1, sub_dir))

        np.random.shuffle(data_info)

        return data_info
