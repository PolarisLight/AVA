import os
import shutil
import tqdm

# 定义文件夹A和文件夹B的路径
folderA = 'F:\\Dataset\\BAID\\images1'
folderB = 'F:\\Dataset\\BAID\\images3'
os.makedirs(folderB, exist_ok=True)
# 获取文件夹A中的所有文件
files = os.listdir(folderA)

# 计算要移动的文件数量（一半）
half = len(files) // 2

# 获取要移动的文件列表
files_to_move = files[:half]

# 移动文件到文件夹B
for file in tqdm.tqdm(files_to_move):
    source_path = os.path.join(folderA, file)
    dest_path = os.path.join(folderB, file)
    shutil.move(source_path, dest_path)

print(f'已将{half}个文件移动到文件夹B中。')
