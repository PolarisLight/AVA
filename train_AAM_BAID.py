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

from dataset import BBDataset, train_transform, val_transform
from utils import EMD_loss, dis_2_score
from AAM import *

# this is for solving the problem of "OMP: Error #15: Initializing libiomp5.dylib,
# but found libiomp5.dylib already initialized."
# when using scipy, you might face this problem
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# if you have multiple GPUs, you can use CUDA_VISIBLE_DEVICES to choose which GPU to use
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

arg = argparse.ArgumentParser()
arg.add_argument("-n", "--task_name", required=False, default="AAM3-M60-F512-LR3e-5", type=str, help="task name")
arg.add_argument("-b", "--batch_size", required=False, default=64, type=int, help="batch size")
arg.add_argument("-e", "--epochs", required=False, default=30, help="epochs")
arg.add_argument("-lr", "--learning_rate", required=False, type=float, default=3e-5, help="learning rate")
arg.add_argument("-m", "--model_saved_path", required=False, default="saved_models", help="model saved path")
arg.add_argument("-d", "--image_dir", required=False, default="D:\\Dataset\\BAID\\images", help="image dir")
arg.add_argument("-c", "--csv_dir", required=False, default="D:\\Dataset\\BAID\\dataset", help="csv dir")
arg.add_argument("-s", "--image_size", required=False, default=(224, 224), help="image size")
arg.add_argument("-w", "--use_wandb", required=False, type=int, default=1, help="use wandb or not")
arg.add_argument("-nw", "--num_workers", required=False, type=int, default=8, help="num_workers")
arg.add_argument("-mn", "--mask_num", required=False, type=int, default=60, help="mask num")
arg.add_argument("-fn", "--feat_num", required=False, type=int, default=512, help="feature num")

opt = vars(arg.parse_args())

TASK_NAME = opt["task_name"]

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
    opt["use_wandb"] = 0
else:
    device = torch.device('cpu')
    opt["use_wandb"] = 0

model_saved_path = os.path.join(opt["model_saved_path"], TASK_NAME)
if not os.path.exists(model_saved_path):
    os.makedirs(model_saved_path, exist_ok=True)


def learning_rate_decay(optimizer, epoch, decay_rate=0.5, decay_epoch=5):
    """
    simple linear learning rate decay
    :param optimizer: instance of optimizer
    :param epoch: current epoch
    :param decay_rate: learning rate decay rate
    :param decay_epoch: step of learning rate decay
    :return:
    """
    if epoch % decay_epoch or epoch == 0:
        return
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay_rate
    print(f"learning rate decay to {optimizer.param_groups[0]['lr']}")


def train(model, train_loader, val_loader, criterion, optimizer, epochs=10,
          model_saved_path=None):
    """
    training function
    train the model and only save the better model on validation set in order to avoid storge problem
    :param model: model to be trained
    :param train_loader: data loader of training set
    :param val_loader: data loader of validation set
    :param criterion: loss function
    :param optimizer: optimizer
    :param epochs: total epochs
    :param model_saved_path: path to save the model, if None, the model will not be saved. default: None
    :return:
    """
    previous_val_loss = 1e10
    for epoch in range(epochs):
        learning_rate_decay(optimizer, epoch)
        if opt["use_wandb"]:
            wandb.log({"lr": optimizer.param_groups[0]['lr']})
        model.train()
        with tqdm.tqdm(train_loader, unit='batch') as pbar:
            for batch_idx, datas in enumerate(train_loader):
                data, target, mask, loc = datas
                data, target, mask, loc = data.to(device), target.to(device).float(), mask.to(device), loc.to(
                    device).float()
                optimizer.zero_grad()

                output = model(data, mask, loc).squeeze(-1) * 10.0
                # output = dis_2_score(output)
                loss = criterion(output, target)

                if opt["use_wandb"]:
                    print()
                    print(f"0:output:{output[0].detach().cpu().numpy()}, "
                          f"\ntarget:{target[0].detach().cpu().numpy()}")
                loss.backward()

                optimizer.step()
                pbar.update(1)

                pbar.set_description('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss:{:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()
                ))
                if opt["use_wandb"] and batch_idx % 10 == 0:
                    wandb.log({"loss": loss, "epoch": epoch})

        val_loss = validate(model, val_loader, criterion)

        if model_saved_path is not None:
            if val_loss < previous_val_loss:
                torch.save(model.state_dict(), os.path.join(model_saved_path, f"model_{epoch}.pth"))
                previous_val_loss = val_loss
                print(f"Model saved at {model_saved_path}/model_{epoch}.pth")
            else:
                print("Model not saved")


def validate(model, val_loader, criterion):
    """
    validation function
    :param model: model to be validated
    :param val_loader: data loader of validation set
    :param criterion: loss function
    :return: validation loss
    """
    model.eval()
    val_loss = []
    pred_list = []
    target_list = []

    with torch.no_grad():
        for i, datas in tqdm.tqdm(enumerate(val_loader)):
            data, target, mask, loc = datas
            data, target, mask, loc = data.to(device), target.to(device), mask.to(device), loc.to(device)

            output = model(data, mask, loc).squeeze(-1) * 10.0

            val_loss.append(criterion(output, target).item())
            pred_list.extend(output.tolist())
            target_list.extend(target.tolist())

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

        if opt["use_wandb"]:
            wandb.log({"val_loss": val_loss, "val_pearson": pearson, "val_spearman": spearman,
                       "val_acc": acc})

        print(f"val_loss:{val_loss}, val_pearson:{pearson}, val_spearman:{spearman}, val_acc:{acc}")

    return val_loss


def main():
    """
    main function
    :return:
    """

    # show options formally
    print("Options:")
    for k, v in opt.items():
        print("\t{}: {}".format(k, v))
    # print device
    print(f"Device: {device}")

    image_dir = opt["image_dir"]
    csv_dir = opt["csv_dir"]

    train_dataset = BBDataset(file_dir=csv_dir, img_dir=image_dir, type='train', test=False, mask_num=opt["mask_num"])
    val_dataset = BBDataset(file_dir=csv_dir, img_dir=image_dir, type='test', test=True,
                            mask_num=opt["mask_num"])

    train_loader = DataLoader(train_dataset, batch_size=opt["batch_size"], shuffle=True, num_workers=opt["num_workers"])
    val_loader = DataLoader(val_dataset, batch_size=opt["batch_size"], shuffle=False, num_workers=opt["num_workers"])

    model = AAM3(mask_num=opt['mask_num'], feat_num=opt['feat_num'], out_class=1)
    model.to(device)

    criterion = nn.MSELoss()  # it can be replaced by other loss function
    optimizer = optim.Adam(model.parameters(), lr=opt["learning_rate"], betas=(0.9, 0.9))

    # you can use wandb to log your training process
    # if not, just set use_wandb to False
    if opt['use_wandb']:
        wandb.init(
            project="BAID",
            name=TASK_NAME,
            config={
                "learning_rate": opt["learning_rate"],
                "batch_size": opt["batch_size"],
                "epochs": opt["epochs"],
            }
        )

    train(model, train_loader, val_loader, criterion, optimizer, epochs=opt["epochs"],
          model_saved_path=model_saved_path)


if __name__ == '__main__':
    main()
