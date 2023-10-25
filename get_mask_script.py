import tqdm
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

img_files = glob.glob(" ")
save_root = "dataset/masks/"

# default sam_vit_h_4b8939
# vit_b sam_vit_b_01ec64
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")

device = "cuda" if torch.cuda.is_available() else "cpu"
sam.to(device)
mask_generator = SamAutomaticMaskGenerator(sam, min_mask_region_area=100, points_per_batch=64)


def get_img_masks(img_name):
    img = cv2.imread(img_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if sys.platform.startswith('win'):
        name = img_name.split("\\")[-1]
    elif sys.platform.startswith('linux'):
        name = img_name.split("/")[-1]
    else:
        print("Unsupport platform")
        exit(1)
    masks = mask_generator.generate(img)
    try:
        array_masks = []
        for mask in masks:
            mask["segmentation"] = [[1 if i else 0 for i in j] for j in mask["segmentation"]]
            array_masks.append(mask["segmentation"])
    except Exception as e:
        print(e)
        array_masks = np.zeros((1, img.shape[0], img.shape[1]))
    finally:
        pass
    array_masks = np.array(array_masks)
    with h5py.File(save_root + "masks.h5", "a") as f:
        # 将img_name和对应的array_masks存入f
        f.create_dataset(name, data=array_masks)


if __name__ == "__main__":
    with tqdm.tqdm(total=len(img_files)) as pbar:
        for i, img_file in enumerate(img_files):
            if sys.platform.startswith('win'):
                img_name = img_file.split("\\")[-1]
            elif sys.platform.startswith('linux'):
                img_name = img_file.split("/")[-1]
            else:
                print("Unsupport platform")
                exit(1)
            pbar.set_description(f"Processing {img_name}")
            get_img_masks(img_file)

            pbar.update(1)
            if i > 10:
                break
