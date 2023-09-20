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

def EMD_loss(pred, gt):
    """
    Earth Mover's Distance
    :param pred: prediction
    :param gt: ground truth
    :return: EMD loss
    """
    pred = pred.cpu().numpy()
    gt = gt.cpu().numpy()
    assert pred.shape == gt.shape
    batch_size = pred.shape[0]
    loss = 0
    for i in range(batch_size):
        loss += EMD(pred[i], gt[i])
    return loss / batch_size

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