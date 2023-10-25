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

img_files = glob.glob("dataset/images/*.jpg")

# default sam_vit_h_4b8939
# vit_b sam_vit_b_01ec64
sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth")

device = "cuda" if torch.cuda.is_available() else "cpu"
sam.to(device)
mask_generator = SamAutomaticMaskGenerator(sam, min_mask_region_area=100, points_per_batch=64)


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


def get_img_masks(img_name):
    img = cv2.imread("dataset/images/" + img_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = cv2.resize(img, (224, 224))

    tic = time.time()
    masks = mask_generator.generate(img)
    print(f'generate time: {time.time() - tic}')
    # plt.figure(figsize=(20, 20))
    # plt.imshow(img)
    # show_anns(masks)
    # plt.axis('off')
    # os.makedirs("dataset/masks", exist_ok=True)
    # plt.savefig("dataset/masks/lm" + img_name[:-4] + ".png", bbox_inches='tight', pad_inches=0, dpi=300)
    # plt.show()
    array_masks = []
    for mask in masks:
        mask["segmentation"] = [[1 if i else 0 for i in j] for j in mask["segmentation"]]
        array_masks.append(mask["segmentation"])
    array_masks = np.array(array_masks)
    with h5py.File("dataset/masks/mask.h5", "a") as f:
        # 将img_name和对应的array_masks存入f
        f.create_dataset(img_name,data=array_masks)

    # json_str = json.dumps(masks)
    # with open("dataset/masks/" + img_name[:-4] + ".json", "w") as f:
    #     f.write(json_str)
    #
    # return


if __name__ == "__main__":
    for i, img_file in enumerate(img_files):
        img_name = img_file.split("\\")[-1]
        get_img_masks(img_name)

        if i == 10:
            break
