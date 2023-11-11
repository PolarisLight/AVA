import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np

from torch.utils.data import DataLoader
from dataset import BBDataset, train_transform, val_transform
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score

import tqdm
import os


def main():
    dataset = BBDataset(file_dir='F:\\Dataset\\BAID\\dataset', img_dir="F:\\Dataset\\BAID\\images",
                        type='test', test=True, mask_num=30)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=16)

    model = torchvision.models.inception_v3(
        pretrained=False)
    model.fc = nn.Sequential(
        nn.Dropout(0.75),
        nn.Linear(model.fc.in_features, 1),
        nn.Sigmoid()
    )
    model.load_state_dict(torch.load("saved_models\\nima-lr3e-5-inception\\model_0.pth"))
    model.cuda()
    model.eval()
    val_loss = []
    pred_list = []
    target_list = []

    with torch.no_grad():
        for i, datas in enumerate(dataloader):
            with tqdm.tqdm(total=len(dataloader), desc=f"test {i}") as pbar:
                data, target, mask, loc = datas
                data, target, mask, loc = data.cuda(), target.cuda(), mask.cuda(), loc.cuda()

                output = model(data).squeeze(-1) * 10.0


                val_loss.append(F.mse_loss(output, target).item())
                pred_list.extend(output.tolist())
                target_list.extend(target.tolist())
                pbar.update(1)


    val_loss = sum(val_loss) / len(val_loss)

    # 计算皮尔逊相关系数
    pearson = pearsonr(pred_list, target_list)[0]
    # 计算斯皮尔曼相关系数
    spearman = spearmanr(pred_list, target_list)[0]

    pred_score_list = np.array(pred_list)
    target_score_list = np.array(target_list)

    pred_label = np.where(pred_score_list <= 5.0, 0, 1)
    target_label = np.where(target_score_list <= 5.0, 0, 1)

    acc = accuracy_score(target_label, pred_label)

    print(f"val_loss:{val_loss}, val_pearson:{pearson}, val_spearman:{spearman}, val_acc:{acc}")


if __name__ == '__main__':
    main()

# val_loss:0.9532391710748307, val_pearson:-0.0875440345020417, val_spearman:0.17274120359240422, val_acc:0.76203125
# 1 val_loss:0.39264276027176187, val_pearson:0.23722869293553933, val_spearman:0.2526819406123867, val_acc:0.7615625
# 2(测试最优） val_loss:0.5248460377552718, val_pearson:0.13010489325185187, val_spearman:0.13780586009610749, val_acc:0.75296875
# 20 val_loss:1.3258017045767991, val_pearson:0.047521662979604196, val_spearman:0.06079522103347232, val_acc:0.71265625
# 3 val_loss:0.8495508854338252, val_pearson:0.10130378727709335, val_spearman:0.09147038223993671, val_acc:0.73484375
