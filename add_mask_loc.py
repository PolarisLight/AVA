import numpy as np
import torch
import os
from utils import calculate_centroids
import tqdm
from concurrent.futures import ThreadPoolExecutor

def process_file(file):
    source_file_path = os.path.join(source_folder, file)
    target_file_path = os.path.join(target_folder, file)

    # 从npz文件中读取masks
    masks = np.load(source_file_path)['masks']

    # 将numpy数组转换为torch张量并计算质心
    masks_tensor = torch.from_numpy(masks).float().to('cuda')  # 假设您使用CUDA
    mask_loc = calculate_centroids(masks_tensor)

    # 将torch张量转换回numpy数组
    mask_loc_numpy = mask_loc.cpu().numpy()

    # 将原始的masks和新计算的mask_loc保存到新的npz文件中
    np.savez_compressed(target_file_path, masks=masks, mask_loc=mask_loc_numpy)




# 读取masks/文件夹下的所有npz文件
source_folder = 'D:\\Dataset\\AVA\\masks\\'
target_folder = 'D:\\Dataset\\AVA\\masks_with_loc'

process_file('128391.npz')

# # 创建目标文件夹，如果它不存在
# if not os.path.exists(target_folder):
#     os.makedirs(target_folder)
#
# npz_files = [f for f in os.listdir(source_folder) if f.endswith('.npz')]
#
# # 使用ThreadPoolExecutor来处理文件
# with ThreadPoolExecutor(max_workers=32) as executor:  # 可以调整max_workers的数量
#     list(tqdm.tqdm(executor.map(process_file, npz_files), total=len(npz_files)))
#
# print("处理完成！")
