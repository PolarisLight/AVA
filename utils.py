import torch
import numpy as np


def EMD(pred, gt):
    """
    Earth Mover's Distance
    :param pred: prediction
    :param gt: ground truth
    :return: EMD
    """
    assert pred.shape == gt.shape
    pred = pred.flatten()
    gt = gt.flatten()
    pred = pred / np.sum(pred)
    gt = gt / np.sum(gt)
    pred = np.cumsum(pred)
    gt = np.cumsum(gt)
    return np.sum(np.abs(pred - gt))

class EMD_loss(torch.nn.Module):
    """
    EMD loss
    """
    def __init__(self):
        super(EMD_loss, self).__init__()

    def forward(self, pred, gt):
        """
        forward
        :param pred: prediction
        :param gt: ground truth
        :return: EMD loss
        """
        assert pred.shape == gt.shape
        pred = pred.flatten()
        gt = gt.flatten()
        pred = pred / torch.sum(pred)
        gt = gt / torch.sum(gt)
        pred = torch.cumsum(pred, dim=0)
        gt = torch.cumsum(gt, dim=0)
        return torch.sum(torch.abs(pred - gt))

def dis_2_score(dis):
    """
    convert distance to score
    :param dis: distance
    :return: score
    """
    dis = dis.detach().cpu().numpy()
    w = np.linspace(1, 10, 10)

    w_batch = np.tile(w, (dis.shape[0], 1))

    score = (dis * w_batch).sum(axis=1)
    return score

class MAELoss(torch.nn.Module):
    """
    MAE loss
    """
    def __init__(self):
        super(MAELoss, self).__init__()

    def forward(self, pred, gt):
        """
        forward
        :param pred: prediction
        :param gt: ground truth
        :return: MAE loss
        """
        assert pred.shape == gt.shape
        return torch.mean(torch.abs(pred - gt))
