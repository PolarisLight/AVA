from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import numpy as np
import cv2
import json
import torch
import glob
import tqdm
import os

os.makedirs("dataset/masks", exist_ok=True)

def get_img_masks(img_name):
    img = cv2.imread(img_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    mask_generator = SamAutomaticMaskGenerator(sam, min_mask_region_area=int(img.shape[0] * img.shape[1] / 100))
    masks = mask_generator.generate(img)


    for mask in masks:
        mask["segmentation"] = mask["segmentation"].tolist()

    json_str = json.dumps(masks)
    with open("dataset/masks/" + os.path.basename(img_name)[:-4] + ".json", "w") as f:
        f.write(json_str)

    return

def process_images_with_threadpool(img_files):
    with ThreadPoolExecutor(max_workers=4) as executor:  # 根据需要设置最大线程数
        futures = {executor.submit(get_img_masks, img_name): img_name for img_name in img_files}
        pbar = tqdm.tqdm(total=len(futures))
        for future in concurrent.futures.as_completed(futures):
            img_name = futures[future]
            try:
                future.result()
                print(f"Processed {img_name}")
                pbar.update(1)
            except Exception as e:
                print(f"Error processing {img_name}: {str(e)}")

sam = sam_model_registry["default"](checkpoint="sam_vit_h_4b8939.pth")
device = "cuda" if torch.cuda.is_available() else "cpu"
sam.to(device)

if __name__ == "__main__":
    img_files = glob.glob("dataset/images/*.jpg")
    print(img_files)

    process_images_with_threadpool(img_files)
