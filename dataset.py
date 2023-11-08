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
from tqdm import tqdm
import cv2
import numpy as np

compare_fastsam = False

mean = [0.485, 0.456, 0.406]  # RGB
std = [0.229, 0.224, 0.225]

if compare_fastsam:
    from fastsam import FastSAM, FastSAMPrompt

train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    # transforms.RandomCrop(448),
    transforms.RandomHorizontalFlip(),

    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((512, 512)),
    transforms.RandomCrop(448),

    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


class AVADatasetFastSAM(data.Dataset):
    """AVA dataset

    Args:
        csv_file: a 11-column csv_file, column one contains the names of image files, column 2-11 contains the empiricial distributions of ratings
        root_dir: directory to the images
        transform: preprocessing and augmentation of the training images
    """

    def __init__(self, csv_file, root_dir, transform=None, imgsz=512, mask_num=30, mask=True, device='cpu'):
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
        annotations = annotations.astype('float').reshape(-1, )
        sample = {'img_id': img_name, 'image': image, 'annotations': annotations}
        if self.mask:
            sample['masks'] = masks
        else:
            sample['masks'] = torch.zeros((self.mask_num, self.imgsz, self.imgsz))

        if self.transform:
            sample['image'] = self.transform(sample['image'])
            # print(sample['image'].shape)

        return sample


class BBDataset(data.Dataset):
    def __init__(self, file_dir='dataset', img_dir="images", type='train', test=False, mask_num=30):
        self.if_test = test
        self.mask_num = mask_num
        self.train_transformer = transforms.Compose(
            [
                transforms.Resize((299, 299)), # TODO: 299 -> 224, 299 is for inception_v3
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

        self.test_transformer = transforms.Compose(
            [
                transforms.Resize((299, 299)), # TODO: 299 -> 224
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

        self.images = []
        self.pic_paths = []
        self.labels = []

        if type == 'train':
            DATA = pd.read_csv(os.path.join(file_dir, 'train_set.csv'))
        elif type == 'validation':
            DATA = pd.read_csv(os.path.join(file_dir, 'val_set.csv'))
        elif type == 'test':
            DATA = pd.read_csv(os.path.join(file_dir, 'test_set.csv'))

        labels = DATA['score'].values.tolist()
        pic_paths = DATA['image'].values.tolist()
        for i in tqdm(range(len(pic_paths))):
            pic_path = os.path.join(img_dir, pic_paths[i])
            label = float(labels[i])
            self.pic_paths.append(pic_path)
            self.labels.append(label)

    def __len__(self):
        return len(self.pic_paths)

    def __getitem__(self, index):
        pic_path = self.pic_paths[index]
        img = Image.open(pic_path).convert('RGB')
        if self.if_test:
            img = self.test_transformer(img)
        else:
            img = self.train_transformer(img)
        #
        mask_name = pic_path.replace('images', 'masks').replace('.jpg', '.npz')

        masks = np.load(mask_name)['masks']

        # add random horizontal flip augmentation to masks and img
        if not self.if_test:
            if np.random.rand() > 0.5:
                img = torch.flip(img, dims=[2])
                masks = np.flip(masks, axis=2).copy()

        mask_loc = np.zeros((self.mask_num, 2))
        resized_masks = []
        for i, mask in enumerate(masks):
            if i >= self.mask_num:
                break
            mask = np.array(mask, dtype=np.uint8)
            M = cv2.moments(mask)
            centroid_x = int(M["m10"] / M["m00"])
            centroid_y = int(M["m01"] / M["m00"])
            mask_loc[i] = [centroid_x, centroid_y]
            mask = cv2.resize(mask, (224, 224), interpolation=cv2.INTER_NEAREST)
            mask = np.array(mask, dtype=np.float32)
            resized_masks.append(mask)

        if len(resized_masks) < self.mask_num:
            resized_masks = resized_masks + [np.zeros((224, 224), dtype=np.float32)] * (
                    self.mask_num - len(resized_masks))
            mask_loc = np.concatenate((mask_loc, np.zeros((self.mask_num - len(mask_loc), 2))))
        else:
            resized_masks = resized_masks[:self.mask_num]
            mask_loc = mask_loc[:self.mask_num]

        resized_masks = torch.from_numpy(np.array(resized_masks))
        mask_loc = torch.from_numpy(np.array(mask_loc).astype(np.float32))

        return img, self.labels[index], resized_masks, mask_loc


if __name__ == '__main__':
    # sanity check
    val_Dataset = BBDataset(file_dir='F:\\Dataset\\BAID\\dataset', img_dir="F:\\Dataset\\BAID\\images",
                            type='validation')
    val_loader = torch.utils.data.DataLoader(val_Dataset, batch_size=8, shuffle=False, num_workers=0)
    for i, data in enumerate(val_loader):
        imgs, labels, masks, loc = data
        print(imgs.dtype)
        print(labels)
        print(masks.dtype)
        print(loc.dtype)
        break
