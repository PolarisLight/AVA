from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
import json
import torch
import glob

img_files = glob.glob("dataset/images/*.jpg")

# default sam_vit_h_4b8939
# vit_b sam_vit_b_01ec64
sam = sam_model_registry["default"](checkpoint="sam_vit_h_4b8939.pth")

device = "cuda" if torch.cuda.is_available() else "cpu"
sam.to(device)
for layer in sam.image_encoder.children():
    print(layer)
for img_name in img_files:
    img = cv2.imread(img_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (1024 , 1024))
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(device)
    print(img.shape)

    tic = time.time()
    embeddings = sam.image_encoder(img)
    print(f'generate time: {time.time() - tic}')
    print(embeddings.shape)