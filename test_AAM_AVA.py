import os
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score

import tqdm
import wandb  # wandb is a tool for visualizing the training process, please refer to https://wandb.ai/site

from dataset import AVADatasetSAM, train_transform, val_transform
from utils import EMD_loss, dis_2_score
from AAM import AAM3, AAM4

# this is for solving the problem of "OMP: Error #15: Initializing libiomp5.dylib,
# but found libiomp5.dylib already initialized."
# when using scipy, you might face this problem
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# if you have multiple GPUs, you can use CUDA_VISIBLE_DEVICES to choose which GPU to use
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

arg = argparse.ArgumentParser()
arg.add_argument("-n", "--task_name", required=False, default="AAM3-M40-F1024-lr3e-6", type=str, help="task name")
arg.add_argument("-b", "--batch_size", required=False, default=64, type=int, help="batch size")
arg.add_argument("-e", "--epochs", required=False, default=30, help="epochs")
arg.add_argument("-lr", "--learning_rate", required=False, type=float, default=3e-6, help="learning rate")
arg.add_argument("-m", "--model_saved_path", required=False, default="saved_models", help="model saved path")
arg.add_argument("-d", "--image_dir", required=False, default="D:\\Dataset\\AVA\\images", help="image dir")
arg.add_argument("-c", "--csv_dir", required=False, default="D:\\Dataset\\AVA\\labels", help="csv dir")
arg.add_argument("-s", "--image_size", required=False, default=(224, 224), help="image size")
arg.add_argument("-w", "--use_wandb", required=False, type=int, default=1, help="use wandb or not")
arg.add_argument("-nw", "--num_workers", required=False, type=int, default=8, help="num_workers")
arg.add_argument("-mn", "--mask_num", required=False, type=int, default=40, help="mask num")
arg.add_argument("-fn", "--feat_num", required=False, type=int, default=1024, help="feature num")
arg.add_argument("-sn", "--use_subnet", required=False, type=str, default="both", help="use subnet:cnn, gcn, both")

opt = vars(arg.parse_args())

# CUDA, MPS(Mac) and CPU support
if torch.cuda.is_available():
    device = torch.device('cuda')
    print('Using device:cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
    opt["use_wandb"] = 0
    print('Using device:mps')
else:
    device = torch.device('cpu')
    opt["use_wandb"] = 0


def main():
    for k, v in opt.items():
        print("\t{}: {}".format(k, v))

    image_dir = opt["image_dir"]
    csv_dir = opt["csv_dir"]
    test_csv = os.path.join(csv_dir, "test_labels.csv")

    test_dataset = AVADatasetSAM(csv_file=test_csv, root_dir=image_dir, mask_num=opt["mask_num"],
                                 imgsz=224, if_test=True, transform=True)

    test_loader = DataLoader(test_dataset, batch_size=opt["batch_size"], shuffle=False, num_workers=opt["num_workers"])

    model = AAM3(feat_num=opt["feat_num"], mask_num=opt["mask_num"], use_subnet=opt["use_subnet"])
    model.to(device)

    emd_loss = EMD_loss()
    mse_loss = nn.MSELoss()

    model_saved_path = os.path.join(opt["model_saved_path"], opt["task_name"])
    model_saved_path = os.path.join(model_saved_path, "model_7.pth")

    model.load_state_dict(torch.load(model_saved_path))

    model.eval()

    with torch.no_grad():
        emd = []
        pred_score_list = []
        target_score_list = []
        for i, datas in enumerate(tqdm.tqdm(test_loader)):
            data, target, mask = datas["image"].to(device), datas["annotations"].to(device), datas["masks"].to(
                device)
            mask_loc = datas["mask_loc"].to(device)
            output = model(data, mask, mask_loc)
            emd.append(emd_loss(output, target).item())
            pred_score_list += dis_2_score(output).tolist()
            target_score_list += dis_2_score(target).tolist()

        emd = sum(emd) / len(emd)
        mse = mse_loss(torch.tensor(pred_score_list), torch.tensor(target_score_list)).item()
        # 计算皮尔逊相关系数
        pearson = pearsonr(pred_score_list, target_score_list)[0]
        # 计算斯皮尔曼相关系数
        spearman = spearmanr(pred_score_list, target_score_list)[0]

        pred_score_list = np.array(pred_score_list)
        target_score_list = np.array(target_score_list)

        pred_label = np.where(pred_score_list <= 5.00, 0, 1)
        target_label = np.where(target_score_list <= 5.00, 0, 1)

        acc = accuracy_score(target_label, pred_label)

        print(f"EMD:{emd}, MSE:{mse}, Pearson:{pearson}, Spearman:{spearman}, Acc:{acc}")


if __name__ == "__main__":
    main()

# AAM3-M40-F1024-lr3e-6
# 7  EMD:0.08868434140003172, MSE:0.2965905964374542, Pearson:0.6699390524473332, Spearman:0.6558976986613683, Acc:0.7890301942732308
# 9  EMD:0.10049738267409734, MSE:0.2979836165904999, Pearson:0.6680584671751162, Spearman:0.6531161283548503, Acc:0.7907466645860965
# 10 EMD:0.08878615393307511, MSE:0.29508569836616516, Pearson:0.6698902588140313, Spearman:0.655897292149705, Acc:0.7896543652960911
# 11 EMD:0.08900960259418239, MSE:0.29667016863822937, Pearson:0.6698113618172505, Spearman:0.6556159665585526, Acc:0.7884060232503706
