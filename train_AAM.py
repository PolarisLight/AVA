import os
import argparse
import random

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

from dataset import AVADatasetSAM, train_transform, val_transform, AVADatasetSAM_New
from utils import EMD_loss, dis_2_score,emd_loss
from AAM import AAM3, AAM4, AAM5

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # 如果使用多个GPU

# this is for solving the problem of "OMP: Error #15: Initializing libiomp5.dylib,
# but found libiomp5.dylib already initialized."
# when using scipy, you might face this problem
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# if you have multiple GPUs, you can use CUDA_VISIBLE_DEVICES to choose which GPU to use
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

arg = argparse.ArgumentParser()
arg.add_argument("-n", "--task_name", required=False, default="AAM4-FM40-F2048-", type=str, help="task name")
arg.add_argument("-b", "--batch_size", required=False, default=64, type=int, help="batch size")
arg.add_argument("-e", "--epochs", required=False, default=10, help="epochs")
arg.add_argument("-lr", "--learning_rate", required=False, type=float, default=3e-6, help="learning rate")
arg.add_argument("-m", "--model_saved_path", required=False, default="saved_models", help="model saved path")
arg.add_argument("-d", "--image_dir", required=False, default="D:\\Dataset\\AVA\\images", help="image dir")
arg.add_argument("-c", "--csv_dir", required=False, default="D:\\Dataset\\AVA\\labels", help="csv dir")
arg.add_argument("-s", "--image_size", required=False, default=224, type=int, help="image size")
arg.add_argument("-w", "--use_wandb", required=False, type=int, default=1, help="use wandb or not")
arg.add_argument("-nw", "--num_workers", required=False, type=int, default=8, help="num_workers")
arg.add_argument("-mn", "--mask_num", required=False, type=int, default=40, help="mask num")
arg.add_argument("-fn", "--feat_num", required=False, type=int, default=1024, help="feature num")
arg.add_argument("-sn", "--use_subnet", required=False, type=str, default="both", help="use subnet:cnn, gcn, both")
arg.add_argument("-fs", "--feature_scale", required=False, type=int, default=4, help="which layer feature to use")
arg.add_argument("-ff", "--freeze_feat", required=False, type=int, default=0,
                 help="whether detach the gradient from gcn to feature extractor or not")
arg.add_argument("-gn", "--gcn_num", required=False, type=int, default=2, help="gcn layer num")
arg.add_argument("-r", "--resnet", required=False, type=int, default=0, help="use resnet gcn or not")
arg.add_argument("-sf", "--shuffle", required=False, type=int, default=0, help="shuffle mask channel or not")
arg.add_argument("-l2", "--use_L2", required=False, type=int, default=1, help="use mask center space distance or not")
arg.add_argument("-bn", "--use_BN", required=False, type=int, default=0, help="use bn in gcn or not")

opt = vars(arg.parse_args())

TASK_NAME = opt["task_name"]

model_saved_path = os.path.join(opt["model_saved_path"], TASK_NAME)
if not os.path.exists(model_saved_path):
    os.makedirs(model_saved_path, exist_ok=True)

# CUDA, MPS(Mac) and CPU support
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
    opt["use_wandb"] = 0
else:
    device = torch.device('cpu')
    opt["use_wandb"] = 0


def learning_rate_decay(optimizer, epoch, decay_rate=0.1, decay_epoch=3):
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


def train(model, train_loader, val_loader, criterion_train, criterion_test, optimizer, epochs=10,
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
                data, target, mask = datas["image"].to(device), datas["annotations"].to(device), datas["masks"].to(
                    device)
                mask_loc = datas["mask_loc"].to(device)
                optimizer.zero_grad()
                output = model(data, mask, mask_loc)
                loss = criterion_train(output, target)
                if not opt["use_wandb"]:
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
                # print(f"gcn layer1 grad{torch.mean(model.GCN[0].GCN_W.grad)},"
                #       f"layer2 grad{torch.mean(model.GCN[1].GCN_W.grad)},"
                #       f"layer3 grad{torch.mean(model.GCN[2].GCN_W.grad)},"
                #       f"gcn fc grad{torch.mean(model.gcn_projector[2].weight.grad)}")
                if opt["use_wandb"]:
                    wandb.log({"loss": loss,
                               "epoch": epoch})
                    # wandb.log({"gcn grad": torch.mean(model.gcn_projector[2].weight.grad),
                    #            "cnn grad": torch.mean(model.cnn_projector[3].weight.grad)})

        val_loss = validate(model, val_loader, criterion_train, criterion_test)

        if model_saved_path is not None:
            if val_loss < previous_val_loss:
                torch.save(model.state_dict(), os.path.join(model_saved_path, f"model_{epoch}.pth"))
                previous_val_loss = val_loss
                print(f"Model saved at {model_saved_path}/model_{epoch}.pth")
            else:
                print("Model not saved")


def validate(model, val_loader, criterion_train, criterion_test):
    """
    validation function
    :param model: model to be validated
    :param val_loader: data loader of validation set
    :param criterion: loss function
    :return: validation loss
    """
    model.eval()
    val_loss_r2 = []
    val_loss_r1 = []
    pred_list = []
    target_list = []
    pred_score_list = []
    target_score_list = []

    with torch.no_grad():
        for datas in tqdm.tqdm(val_loader):
            data, target, mask = datas["image"].to(device), datas["annotations"].to(device), datas["masks"].to(device)
            mask_loc = datas["mask_loc"].to(device)
            output = model(data, mask, mask_loc)
            val_loss_r2.append(criterion_train(output, target).item())
            val_loss_r1.append(criterion_test(output, target).item())
            pred_list.append(output)
            target_list.append(target)
            pred_score_list += dis_2_score(output).tolist()
            target_score_list += dis_2_score(target).tolist()

        val_loss_r2 = sum(val_loss_r2) / len(val_loss_r2)
        val_loss_r1 = sum(val_loss_r1) / len(val_loss_r1)

        mse_loss = torch.nn.functional.mse_loss(torch.tensor(pred_score_list), torch.tensor(target_score_list)).item()

        # 计算皮尔逊相关系数
        pearson = pearsonr(pred_score_list, target_score_list)[0]
        # 计算斯皮尔曼相关系数
        spearman = spearmanr(pred_score_list, target_score_list)[0]

        pred_score_list = np.array(pred_score_list)
        target_score_list = np.array(target_score_list)

        pred_label = np.where(pred_score_list <= 5.00, 0, 1)
        target_label = np.where(target_score_list <= 5.00, 0, 1)

        acc = accuracy_score(target_label, pred_label)

        if opt["use_wandb"]:
            wandb.log({"val_loss": val_loss_r1, "val_pearson": pearson, "val_spearman": spearman,
                       "val_acc": acc, "val_mse": mse_loss, "emd_r2": val_loss_r2})

        print(
            f"val_loss:{val_loss_r1}, val_pearson:{pearson}, val_spearman:{spearman}, val_acc:{acc}, val_mse:{mse_loss}")

    return val_loss_r1


def main():
    print(f"using device {device}")
    # show options formally
    print("Options:")
    for k, v in opt.items():
        print("\t{}: {}".format(k, v))
    """
    main function
    :return:
    """
    image_dir = opt["image_dir"]
    csv_dir = opt["csv_dir"]
    train_csv = os.path.join(csv_dir, "train_labels.csv")
    val_csv = os.path.join(csv_dir, "test_labels.csv")

    train_dataset = AVADatasetSAM_New(csv_file=train_csv, root_dir=image_dir, mask_num=opt["mask_num"],
                                      imgsz=(opt['image_size'], opt['image_size']), if_test=False, transform=True,
                                      shuffle=opt["shuffle"])
    val_dataset = AVADatasetSAM_New(csv_file=val_csv, root_dir=image_dir, mask_num=opt["mask_num"],
                                    imgsz=(opt['image_size'], opt['image_size']), if_test=True, transform=True,
                                    shuffle=opt["shuffle"])

    train_loader = DataLoader(train_dataset, batch_size=opt["batch_size"], shuffle=True, num_workers=opt["num_workers"])
    val_loader = DataLoader(val_dataset, batch_size=opt["batch_size"], shuffle=False, num_workers=opt["num_workers"])

    model = AAM5(mask_num=opt["mask_num"], feat_num=opt["feat_num"], use_subnet=opt["use_subnet"],
                 feat_scale=opt["feature_scale"], freeze_feat=opt['freeze_feat'], gcn_layer_num=opt['gcn_num'],
                 resnet=opt['resnet'], use_L2=opt['use_L2'],use_BN=opt['use_BN'])
    model.to(device)

    criterion_r2 = emd_loss(dist_r=2)  # it can be replaced by other loss function
    criterion_r1 = emd_loss(dist_r=1)
    optimizer = optim.SGD(model.parameters(), lr=opt["learning_rate"], momentum=0.9, weight_decay=1e-3)

    # feature_extractor_params = model.feature_extractor.parameters()
    #
    # # 获取除feature_extractor外模型的其余部分的参数
    # remaining_params = [param for name, param in model.named_parameters() if "feature_extractor" not in name]
    #
    # # 创建优化器，并为不同部分的参数设置不同的学习率
    # optimizer = optim.Adam([
    #     {'params': feature_extractor_params, 'lr': 3e-7},  # 对于feature_extractor使用1e-7的学习率
    #     {'params': remaining_params, 'lr': 3e-6}  # 对于模型的其余部分使用1e-6的学习率
    # ],betas=(0.9, 0.9))

    # you can use wandb to log your training process
    # if not, just set use_wandb to False
    if opt['use_wandb']:
        wandb.init(
            project="AVA",
            name=TASK_NAME,
            config={
                "learning_rate": opt["learning_rate"],
                "batch_size": opt["batch_size"],
                "epochs": opt["epochs"],
            }
        )

    train(model, train_loader, val_loader, criterion_r2, criterion_r1, optimizer, epochs=opt["epochs"],
          model_saved_path=model_saved_path)


if __name__ == '__main__':
    main()
