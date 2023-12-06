import os
import argparse
import random
import datetime

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
from utils import EMD_loss, dis_2_score
from AAM import AAM3, AAM4

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

# CUDA, MPS(Mac) and CPU support
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

arg = argparse.ArgumentParser()
arg.add_argument("-e", "--epochs", required=False, default=5, help="epochs")
arg.add_argument("-d", "--image_dir", required=False, default="D:\\Dataset\\AVA\\images", help="image dir")
arg.add_argument("-c", "--csv_dir", required=False, default="D:\\Dataset\\AVA\\labels", help="csv dir")
arg.add_argument("-nw", "--num_workers", required=False, type=int, default=8, help="num_workers")

opt = vars(arg.parse_args())

sweep_config = {
    'method': 'random'
}

metric = {
    'name': 'val_acc',
    'goal': 'maximize'
}

sweep_config['metric'] = metric

sweep_config['parameters'] = {}
sweep_config['parameters'].update({
    'project_name': {'value': 'AAM-sweep'},
    'epochs': {'value': opt['epochs']},
    'ckpt_path': {'value': 'saved_models/swept_best_AAM.pth'},
    'image_dir': {'value': opt['image_dir']},
    'csv_dir': {'value': opt['csv_dir']},
    'num_workers': {'value': opt['num_workers']},
})

# 离散型分布超参
sweep_config['parameters'].update({
    'optim_type': {
        'values': ['Adam', 'SGD', 'RMSprop']
    },
    'use_subnet': {
        'values': ['cnn', 'gcn', 'both']
    }
})
# 连续型分布超参
sweep_config['parameters'].update({

    'lr': {
        'distribution': 'log_uniform_values',
        'min': 1e-8,
        'max': 1e-2,
    },
    'batch_size': {
        'distribution': 'q_uniform',
        'q': 8,
        'min': 16,
        'max': 64,
    },
    'dropout_p': {
        'distribution': 'uniform',
        'min': 0,
        'max': 0.75,
    },
    'mask_num': {
        'distribution': 'q_uniform',
        'q': 5,
        'min': 10,
        'max': 80,
    },
    'image_size': {
        'distribution': 'q_uniform',
        'q': 16,
        'min': 224,
        'max': 400,
    },
    'gcn_layer': {
        'distribution': 'q_uniform',
        'q': 1,
        'min': 1,
        'max': 10,
    },
})

sweep_config['early_terminate'] = {
    'type': 'hyperband',
    'min_iter': 4,
    'eta': 2,
    's': 4
}

from pprint import pprint
pprint(sweep_config)

sweed_id = wandb.sweep(sweep_config, project='AAM-sweep')



def learning_rate_decay(optimizer, epoch, decay_rate=0.1, decay_epoch=5):
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


def train(model, train_loader, val_loader, criterion, optimizer,
          config):
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
    model.best_acc = -1.0
    for epoch in range(config.epoches):
        learning_rate_decay(optimizer, epoch)
        print(f"learning rate:{optimizer.param_groups[0]['lr']}")
        model.train()
        with tqdm.tqdm(train_loader, unit='batch') as pbar:
            for batch_idx, datas in enumerate(train_loader):
                data, target, mask = datas["image"].to(device), datas["annotations"].to(device), datas["masks"].to(
                    device)
                mask_loc = datas["mask_loc"].to(device)
                optimizer.zero_grad()
                output = model(data, mask, mask_loc)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                pbar.update(1)
                pbar.set_description('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss:{:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()
                ))
                wandb.log({"train_loss": loss})

        val_acc = validate(model, val_loader, criterion)

        if config.ckpt_path is not None:
            if val_acc > model.best_acc:
                torch.save(model.state_dict(), os.path.join(config.ckpt_path, config.ckpt_path))
                model.best_acc = val_acc
                print(f"Model saved at {config.ckpt_path}")
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
    pred_score_list = []
    target_score_list = []

    with torch.no_grad():
        for datas in tqdm.tqdm(val_loader):
            data, target, mask = datas["image"].to(device), datas["annotations"].to(device), datas["masks"].to(device)
            mask_loc = datas["mask_loc"].to(device)
            output = model(data, mask, mask_loc)
            val_loss.append(criterion(output, target).item())
            pred_list.append(output)
            target_list.append(target)
            pred_score_list += dis_2_score(output).tolist()
            target_score_list += dis_2_score(target).tolist()

        val_loss = sum(val_loss) / len(val_loss)

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

        wandb.log({"val_loss": val_loss, "val_pearson": pearson, "val_spearman": spearman,
                   "val_acc": acc, "val_mse": mse_loss})

        print(f"val_loss:{val_loss}, val_pearson:{pearson}, val_spearman:{spearman}, val_acc:{acc}, val_mse:{mse_loss}")

    return acc


def main(config=None):
    print(f"using device {device}")
    """
    main function
    :return:
    """
    image_dir = config.image_dir
    csv_dir = config.csv_dir
    train_csv = os.path.join(csv_dir, "train_labels.csv")
    val_csv = os.path.join(csv_dir, "test_labels.csv")

    train_dataset = AVADatasetSAM_New(csv_file=train_csv, root_dir=image_dir, mask_num=config.mask_num,
                                      imgsz=(config.image_size, config.image_size), if_test=False, transform=True)
    val_dataset = AVADatasetSAM_New(csv_file=val_csv, root_dir=image_dir, mask_num=config.mask_num,
                                    imgsz=(config.image_size, config.image_size), if_test=True, transform=True)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    model = AAM4(mask_num=config.mask_num, feat_num=config.feat_num, use_subnet=config.use_subnet,
                 feat_scale=3, freeze_feat=0, gcn_layer_num=config.gcn_num,dropout=config.dropout_p,
                 resnet=1)
    model.to(device)

    criterion = EMD_loss()  # it can be replaced by other loss function
    optimizer = torch.optim.__dict__[config.optim_type](model.parameters(), lr=config.lr)

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
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    wandb.init(project=config.project_name, config=config.__dict__, name=nowtime, save_code=True)

    train(model, train_loader, val_loader, criterion, optimizer, config=config)
    model.run_id = wandb.run.id

    wandb.finish()

    return model

if __name__ == '__main__':
    wandb.agent(sweep_config, function=main, count=100)
    # main(config=opt)
