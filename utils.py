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

    def forward(self, p_estimate, p_target):
        assert p_target.shape == p_estimate.shape
        cdf_target = torch.cumsum(p_target, dim=1)
        cdf_estimate = torch.cumsum(p_estimate, dim=1)

        cdf_diff = cdf_estimate - cdf_target
        # samplewise_emd = torch.sqrt(torch.mean(torch.pow(torch.abs(cdf_diff), 2)))  # train
        samplewise_emd = torch.mean(torch.pow(torch.abs(cdf_diff), 1)) # test

        return samplewise_emd.mean()

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
