"""
file - dataset.py
Customized dataset class to loop through the AVA dataset and apply needed image augmentations for training.

Copyright (C) Yunxiao Shi 2017 - 2021
NIMA is released under the MIT license. See LICENSE for the fill license text.
"""

import os

import pandas as pd
from PIL import Image

import torch
from torch.utils import data
import torchvision.transforms as transforms

from fastsam import FastSAM, FastSAMPrompt

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    # transforms.RandomCrop(448),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.RandomCrop(448),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


class AVADataset(data.Dataset):
    """AVA dataset

    Args:
        csv_file: a 11-column csv_file, column one contains the names of image files, column 2-11 contains the empiricial distributions of ratings
        root_dir: directory to the images
        transform: preprocessing and augmentation of the training images
    """

    def __init__(self, csv_file, root_dir, transform=None, imgsz=512, mask_num=30, mask=True,device='cpu'):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.mask = mask
        if self.mask:
            self.FASTSAM = FastSAM('./FastSAM-x.pt')
        self.device = device
        self.imgsz = imgsz
        self.mask_num = mask_num

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, str(self.annotations.iloc[idx, 0]) + '.jpg')
        image = Image.open(img_name).convert('RGB')
        # resize img
        image = image.resize((self.imgsz, self.imgsz))
        if self.mask:
            try:
                mask = self.FASTSAM(image, device=self.device, retina_masks=True,
                                    imgsz=self.imgsz, conf=0.2, iou=0.9, verbose=False)[0].masks.data.cpu()
            except Exception as e:
                print(e)
                mask = torch.ones((self.mask_num, self.imgsz, self.imgsz))


            mask_sum = mask.view(mask.shape[0], -1).sum(dim=1)

            # 获取排序后的索引
            sorted_indices = torch.argsort(mask_sum, descending=True)

            # 根据排序的索引重新排列掩码
            sorted_mask = mask[sorted_indices]
            masks = torch.zeros((self.mask_num, self.imgsz, self.imgsz))
            if sorted_mask.shape[0] > self.mask_num:
                masks = sorted_mask[:self.mask_num]
            else:
                masks[:sorted_mask.shape[0]] = sorted_mask
        annotations = self.annotations.iloc[idx, 1:].to_numpy()
        annotations = annotations.astype('float').reshape(-1,)
        sample = {'img_id': img_name, 'image': image, 'annotations': annotations}
        if self.mask:
            sample['masks'] = masks
        else:
            sample['masks'] = torch.zeros((self.mask_num, self.imgsz, self.imgsz))

        if self.transform:
            sample['image'] = self.transform(sample['image'])
            # print(sample['image'].shape)

        return sample


if __name__ == '__main__':
    # sanity check
    root = './dataset/images'
    csv_file = './dataset/labels/train_labels.csv'
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    dset = AVADataset(csv_file=csv_file, root_dir=root, transform=train_transform)
    train_loader = data.DataLoader(dset, batch_size=4, shuffle=True, num_workers=4)
    for i, data in enumerate(train_loader):
        images = data['image']
        print(images.size())
        labels = data['annotations']
        print(labels.size())
        masks = data['masks']
        print(masks.size())
        break
