from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
import json
import torch
import glob
from dataset import AVADataset, train_transform, val_transform
from torch.utils.data import DataLoader
import tqdm
import argparse

img_files = glob.glob("dataset/images/*.jpg")

# default sam_vit_h_4b8939
# vit_b sam_vit_b_01ec64
sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth")

device = "cuda" if torch.cuda.is_available() else "cpu"
sam.to(device)
for layer in sam.image_encoder.children():
    print(layer)

argparser = argparse.ArgumentParser()
argparser.add_argument("-bs", "--batch_size", required=False, default=16, type=int, help="batch size")

opt = vars(argparser.parse_args())


image_dir = "dataset/images"
train_csv = "dataset/labels/train_labels.csv"



dataset = AVADataset(csv_file=train_csv, root_dir=image_dir, transform=train_transform)
train_loader = DataLoader(dataset, batch_size=opt["batch_size"], shuffle=True)

for i, data in tqdm.tqdm(enumerate(train_loader)):
    img, label = data["image"].to(device), data["annotations"].to(device)
    print(img.shape)

    tic = time.time()
    with torch.no_grad():
        embeddings = sam.image_encoder(img)
    print(f'generate time: {time.time() - tic}')
    print(embeddings.shape)