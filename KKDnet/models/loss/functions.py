import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import gaussian_filter
import random
import cv2
from torchvision.transforms import Resize


# TODO 添加余弦距离作为距离系数
def cal_similarity(fs_list: list, ft_list: list):
    N = len(fs_list)
    cos = torch.nn.CosineSimilarity()  # [0, 1]
    mse = torch.nn.MSELoss()
    sim_list = []
    for i in range(N):
        fs = fs_list[i]
        ft = ft_list[i]
        fs_norm = F.normalize(fs, p=2)
        ft_norm = F.normalize(ft, p=2)
        cc = 1 - cos(ft_norm.view(ft_norm.shape[0], -1),
                     fs_norm.view(fs_norm.shape[0], -1))
        cc = cc / torch.sqrt(torch.sum(cc**2))
        # cos_loss = torch.sum(1 - cos(ft_norm.view(ft_norm.shape[0], -1),
        #                              fs_norm.view(fs_norm.shape[0], -1)))
        cos_loss = torch.sum(cc)
        _, _, h, w = fs.shape
        mm = (0.5 * (ft_norm - fs_norm) ** 2) / (h * w)
        mm = mm.sum(dim=[1, 2, 3])
        mm = mm / torch.sqrt(torch.sum(mm**2))
        # mse_loss = mse(ft_norm, fs_norm)
        mse_loss = torch.sum(mm)
        sim = (cos_loss * mse_loss) / (cos_loss + mse_loss)
        sim_list.append(sim)
    sim_sum = sum(sim_list)
    sim_list = [x / sim_sum for x in sim_list]
    return sim_list


def cal_loss(fs_list, ft_list, train_mask):
    #train_mask = train_mask.unsqueeze(1).float()
    train_mask = train_mask.float()
    train_mask = torch.nn.functional.interpolate(train_mask, scale_factor=1 / 4, mode='bilinear',
                                                    align_corners=False)
    # train_mask = train_mask.squeeze(1)
    t_loss = 0
    N = len(fs_list)
    # sim_list = cal_similarity(fs_list, ft_list)
    # kl_distance = torch.nn.KLDivLoss()
    # cos = torch.nn.CosineSimilarity()
    for i in range(4,6):
        fs = fs_list[i]
        ft = ft_list[i]
        fs_norm = F.normalize(fs, p=2)
        #cv2.imwrite("fs_norm.jpg", np.array((fs_norm[0, 0, :, :]*255).data.cpu()).astype(np.uint8))
        fs_norm1 = fs_norm * train_mask
        #cv2.imwrite("fs_norm_mask.jpg", np.array((fs_norm1[0, 0, :, :] * 255).data.cpu()).astype(np.uint8))
        ft_norm = F.normalize(ft, p=2)
        ft_norm1 = ft_norm * train_mask
        f_loss = cal_feature_loss(fs_norm1, ft_norm1)
        # l1_loss = cal_l1_loss(fs_norm, ft_norm)
        # kl_loss = kl_distance(fs_norm, ft_norm)
        # g_loss = cal_gram_loss(fs_norm, ft_norm)
        # cos_loss = torch.sum(1 - cos(ft_norm.view(ft_norm.shape[0], -1),
        #                              fs_norm.view(fs_norm.shape[0], -1)))
        t_loss += f_loss
    return t_loss / N


def cal_feature_loss(fs, ft):
    _, c, h, w = fs.shape
    f_loss = 0.5 * (ft - fs) ** 2
    f_loss = f_loss.sum() / (h * w * c)
    return f_loss


def cal_l1_loss(fs, ft):
    _, _, h, w = fs.shape
    f_loss = torch.abs(ft - fs)
    f_loss = f_loss.sum() / (h * w)
    return f_loss


# log cosh 损失
def logcosh(true, pred):
    loss = torch.log(torch.cosh(pred - true))
    return torch.sum(loss)


def cal_gram_loss(fs, ft):
    # gram: [b, c, c]
    def _gram_matrix(y):
        (b, c, h, w) = y.size()
        features = y.view(b, c, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram

    gram_fs = _gram_matrix(fs)
    gram_ft = _gram_matrix(ft)

    loss = torch.pairwise_distance(gram_fs, gram_ft) ** 2
    _, c = loss.size()  # loss.size: [b, c]
    loss = loss.sum() / c
    return loss


def cal_adain_loss(fs, ft):
    # avgpool: [b, c, 1, 1]
    mean_t = torch.mean(ft, dim=0)
    mean_s = torch.mean(fs, dim=0)
    variance_t = torch.var(ft, dim=0)
    variance_s = torch.var(fs, dim=0)
    new_fs = variance_t * (fs - mean_s) / variance_s + mean_t
    # loss = 0.5 * (mean_t - mean_s)**2 + 0.5 * (variance_t - variance_s)**2
    return cal_feature_loss(new_fs, ft)


# TODO: 添加特征图相乘
def cal_anomaly_maps(fs_list, ft_list, out_size):
    anomaly_map = 0
    # sim_list = cal_similarity(fs_list, ft_list)
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]

        # fs, ft = random_feature_select(fs, ft, 0.6)

        fs_norm = F.normalize(fs, p=2)
        ft_norm = F.normalize(ft, p=2)
        _, _, h, w = fs.shape

        a_map = (0.5 * (ft_norm - fs_norm) ** 2) / (h * w)
        # a_map = torch.abs(ft_norm - fs_norm) / (h * w)
        a_map = a_map.sum(1, keepdim=True)
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=False)

        if i == 0:
            inter_map = a_map
        else:
            inter_map = torch.cat([inter_map, a_map], dim=1)
        anomaly_map += a_map

    # inter_map = feature_maps_mul(inter_map)
    # anomaly_map = sim_list[0] * inter_map[:, 0, :, :] + \
    #               sim_list[1] * inter_map[:, 1, :, :] + \
    #               sim_list[2] * inter_map[:, 2, :, :] + anomaly_map.squeeze()

    anomaly_map = anomaly_map.squeeze().cpu().numpy()
    inter_map = inter_map.squeeze().cpu().numpy()
    for i in range(anomaly_map.shape[0]):
        anomaly_map[i] = gaussian_filter(anomaly_map[i], sigma=4)
        # inter_map[i] = gaussian_filter(inter_map[i], sigma=4)
    return anomaly_map, inter_map


def feature_maps_mul(inter_map: torch.Tensor):
    # input: [b, 3, 224, 224]
    [x1, x2, x3] = inter_map.chunk(3, dim=1)
    x12 = x1 * x2
    x13 = x1 * x3
    x23 = x2 * x3
    ret = torch.cat([x12, x13], dim=1)
    ret = torch.cat([ret, x23], dim=1)
    return ret


def random_feature_select(fs, ft, select_num):
    _, c, _, _ = fs.shape
    if type(select_num) == float:
        select_num = int(c * select_num)
    assert select_num <= c
    idx = torch.tensor(random.sample(range(0, c), select_num)).to(fs.device)
    fs = torch.index_select(fs, 1, idx)
    ft = torch.index_select(ft, 1, idx)
    return fs, ft


if __name__ == '__main__':
    x1 = torch.rand([32, 512, 48, 48]).cuda()
    x2 = torch.rand([32, 512, 48, 48]).cuda()
    # y = x1.view(x1.shape[0], -1)

    # fs = cal_adain_loss(x1, x2)
    # criterion = torch.nn.KLDivLoss()
    # loss = torch.nn.KLDivLoss(x1, x2)
    # loss1 = cal_feature_loss(x1, x2)
    # map = cal_loss([x1], [x2])
    # gram_loss = cal_l1_loss(x1, x2)
    # y = x1.sum()
    # torch.nn.MSELoss()
    y = cal_anomaly_maps([x1, x2], [x2, x1], 224)
    # x = torch.rand([32, 3, 224, 224]).cuda()
    # [x1, x2, x3] = x.chunk(3, dim=1)
    # y = feature_maps_mul(x)
    # sum = y.sum(dim=1)
