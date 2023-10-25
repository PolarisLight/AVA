from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
import json
import torch
import glob
import os
import h5py
import sys

img_files = glob.glob("dataset/images/*.jpg")

sam_h = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
sam_b = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth")
sam_l = sam_model_registry["vit_l"](checkpoint="sam_vit_l_0b3195.pth")

device = "cuda" if torch.cuda.is_available() else "cpu"
sam_h.to(device)
sam_b.to(device)
sam_l.to(device)

mask_generator_h = SamAutomaticMaskGenerator(sam_h, min_mask_region_area=100, points_per_batch=64)
mask_generator_b = SamAutomaticMaskGenerator(sam_b, min_mask_region_area=100, points_per_batch=64)
mask_generator_l = SamAutomaticMaskGenerator(sam_l, min_mask_region_area=100, points_per_batch=64)

def get_img_masks(img_name):
    img = cv2.imread("dataset/images/" + img_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = cv2.resize(img, (224, 224))

    tic = time.time()
    masks = mask_generator_h.generate(img)
    h_time = time.time() - tic
    print(f'huge model generate time: {h_time}')
    print(len(masks))


    tic = time.time()
    masks = mask_generator_l.generate(img)
    l_time = time.time() - tic
    print(f'large model generate time: {l_time}')
    print(len(masks))

    tic = time.time()
    masks = mask_generator_b.generate(img)
    b_time = time.time() - tic
    print(f'big model generate time: {b_time}')
    print(len(masks))

    return h_time, l_time, b_time

if __name__ == "__main__":
    h_times = []
    l_times = []
    b_times = []
    for i, img_file in enumerate(img_files):
        if sys.platform.startswith('win'):
            img_name = img_file.split("\\")[-1]
        elif sys.platform.startswith('linux'):
            img_name = img_file.split("/")[-1]
        else:
            print("Unsupport platform")
            exit(1)
        h_time, l_time, b_time = get_img_masks(img_name)
        h_times.append(h_time)
        l_times.append(l_time)
        b_times.append(b_time)
        if i == 10:
            break
    print(f'huge model average time: {np.mean(h_times)}')
    print(f'large model average time: {np.mean(l_times)}')
    print(f'big model average time: {np.mean(b_times)}')


