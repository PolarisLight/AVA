from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
import json
import torch
import glob

img_files = glob.glob("dataset/images/*.jpg")


sam = sam_model_registry["default"](checkpoint="sam_vit_h_4b8939.pth")
device = "cuda" if torch.cuda.is_available() else "cpu"
sam.to(device)
def get_img_masks(img_name):
    img = cv2.imread("dataset/images/" + img_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


    mask_generator = SamAutomaticMaskGenerator(sam, min_mask_region_area=int(img.shape[0] * img.shape[1] / 100))
    masks = mask_generator.generate(img)


    for mask in masks:
        mask["segmentation"] = mask["segmentation"].tolist()

    json_str = json.dumps(masks)
    with open("dataset/masks/" + img_name[:-4] + ".json", "w") as f:
        f.write(json_str)

    return

from multiprocessing import Pool