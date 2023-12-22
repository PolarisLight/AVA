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

def calculate_centroids(masks):
    if masks.dim() != 3:
        raise ValueError("Masks must be a 3D tensor of shape [N, H, W]")

    N, height, width = masks.size()

    # 生成坐标网格
    x_coords = torch.arange(width, device=masks.device).view(1, 1, -1).expand(N, height, -1)
    y_coords = torch.arange(height, device=masks.device).view(1, -1, 1).expand(N, -1, width)

    # 应用mask
    masked_x = masks * x_coords
    masked_y = masks * y_coords

    # 计算质心
    sum_masks = masks.sum(dim=[1, 2])
    sum_masks[sum_masks == 0] = 1e-6  # 避免除以零

    centroid_x = (masked_x.sum(dim=[1, 2]) / sum_masks).type(torch.int)
    centroid_y = (masked_y.sum(dim=[1, 2]) / sum_masks).type(torch.int)

    # 将质心坐标组合成一个[N, 2]的张量
    centroids = torch.stack((centroid_x, centroid_y), dim=1)

    return centroids


class emd_loss(torch.nn.Module):
    """
    Earth Mover Distance loss
    """

    def __init__(self, dist_r=2,
                 use_l1loss=True, l1loss_coef=0.0):
        super(emd_loss, self).__init__()
        self.dist_r = dist_r
        self.use_l1loss = use_l1loss
        self.l1loss_coef = l1loss_coef

    def check_type_forward(self, in_types):
        assert len(in_types) == 2

        x_type, y_type = in_types
        assert x_type.size()[0] == y_type.shape[0]
        assert x_type.size()[0] > 0
        assert x_type.ndim == 2
        assert y_type.ndim == 2

    def forward(self, x, y_true):
        self.check_type_forward((x, y_true))

        cdf_x = torch.cumsum(x, dim=-1)
        cdf_ytrue = torch.cumsum(y_true, dim=-1)
        if self.dist_r == 2:
            samplewise_emd = torch.sqrt(torch.mean(torch.pow(cdf_ytrue - cdf_x, 2), dim=-1))
        else:
            samplewise_emd = torch.mean(torch.abs(cdf_ytrue - cdf_x), dim=-1)
        loss = torch.mean(samplewise_emd)
        if self.use_l1loss:
            rate_scale = torch.tensor([float(i + 1) for i in range(x.size()[1])], dtype=x.dtype, device=x.device)
            x_mean = torch.mean(x * rate_scale, dim=-1)
            y_true_mean = torch.mean(y_true * rate_scale, dim=-1)
            l1loss = torch.mean(torch.abs(x_mean - y_true_mean))
            loss += l1loss * self.l1loss_coef
        return loss
