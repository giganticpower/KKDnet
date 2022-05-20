import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .backbone.resnet import resnet18
from .neck.fpem_v2 import FPEM_v2
from .neck.fpem_v1 import FPEM_v1
from .utils import Conv_BN_ReLU
from models.neck.visual_transformer_noxin import FilterBasedTokenizer, Transformer, Projector
import cv2
import numpy as np
from models.loss.dice_loss import DiceLoss
from models.loss.emb_loss_v1 import EmbLoss_v1
from models.loss.iou import iou
from models.loss.ohem import ohem_batch



class det_Head(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_classes):
        super(det_Head, self).__init__()
        self.conv1 = nn.Conv2d(in_channels,
                               hidden_dim,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(hidden_dim,
                               num_classes,
                               kernel_size=1,
                               stride=1,
                               padding=0)

        self.text_loss = DiceLoss(loss_weight=1.0)
        self.kernel_loss = DiceLoss(loss_weight=0.5)
        self.emb_loss = EmbLoss_v1(feature_dim=4, loss_weight=0.25)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, f):
        out = self.conv1(f)
        out = self.relu1(self.bn1(out))
        out = self.conv2(out)

        return out

    def loss(self, out, gt_texts, gt_kernels, training_masks, gt_instances,
             gt_bboxes):
        # output
        texts = out[:, 0, :, :]
        kernels = out[:, 1:2, :, :]
        embs = out[:, 2:, :, :]

        # text loss
        selected_masks = ohem_batch(texts, gt_texts, training_masks)
        loss_text = self.text_loss(texts,
                                   gt_texts,
                                   selected_masks,
                                   reduce=False)
        iou_text = iou((texts > 0).long(),
                       gt_texts,
                       training_masks,
                       reduce=False)
        losses = dict(loss_text=loss_text, iou_text=iou_text)

        # kernel loss
        loss_kernels = []
        selected_masks = gt_texts * training_masks
        for i in range(kernels.size(1)):
            kernel_i = kernels[:, i, :, :]
            gt_kernel_i = gt_kernels[:, i, :, :]
            loss_kernel_i = self.kernel_loss(kernel_i,
                                             gt_kernel_i,
                                             selected_masks,
                                             reduce=False)
            loss_kernels.append(loss_kernel_i)
        loss_kernels = torch.mean(torch.stack(loss_kernels, dim=1), dim=1)
        iou_kernel = iou((kernels[:, -1, :, :] > 0).long(),
                         gt_kernels[:, -1, :, :],
                         training_masks * gt_texts,
                         reduce=False)
        losses.update(dict(loss_kernels=loss_kernels, iou_kernel=iou_kernel))

        # embedding loss
        loss_emb = self.emb_loss(embs,
                                 gt_instances,
                                 gt_kernels[:, -1, :, :],
                                 training_masks,
                                 gt_bboxes,
                                 reduce=False)
        losses.update(dict(loss_emb=loss_emb))

        return losses


class student_test(nn.Module):
    def __init__(self, backbone='resnet18', neck='FPEM_v1', detection_head='PA_Head'):
        super(student, self).__init__()
        if backbone == 'resnet18':
            self.backbone = resnet18(pretrained=True)
        else:
            print('error backbone')

        if neck == 'FPEM_v1':
            neck = FPEM_v1(in_channels=(64, 128, 256, 512),
        out_channels=128)
        elif neck == 'FPEM_v2':
            neck = FPEM_v2(in_channels=(64, 128, 256, 512),
                           out_channels=128)
        else:
            print('error neck')

        if detection_head == 'PA_Head':
            self.det_head = det_Head(in_channels=512,
        hidden_dim=128,
        num_classes=6)
        else:
            print('error detection_head')

        in_channels = (64, 128, 256, 512)
        self.reduce_layer1 = Conv_BN_ReLU(in_channels[0], 128)
        self.reduce_layer2 = Conv_BN_ReLU(in_channels[1], 128)
        self.reduce_layer3 = Conv_BN_ReLU(in_channels[2], 128)
        self.reduce_layer4 = Conv_BN_ReLU(in_channels[3], 128)

        self.reduce_layer_ts = Conv_BN_ReLU(512, 512)

        self.fpem1 = neck
        self.fpem2 = neck

        self.tokenizer = FilterBasedTokenizer(128, 128, 32)

        self.transformer = Transformer(128)

        self.projector = Projector(128, 128)

    def _upsample(self, x, size, scale=1):
        _, _, H, W = size
        return F.upsample(x, size=(H // scale, W // scale), mode='bilinear')

    def forward(self,
                imgs,
                gt_texts=None,
                gt_kernels=None,
                training_masks=None,
                gt_instances=None,
                gt_bboxes=None,
                img_metas=None,
                cfg=None):
        outputs = dict()
        bs, ch, h, w = imgs.shape

        # backbone
        f = self.backbone(imgs)

        # reduce channel
        f1 = self.reduce_layer1(f[0])
        f2 = self.reduce_layer2(f[1])
        f3 = self.reduce_layer3(f[2])
        f4 = self.reduce_layer4(f[3])

        # FPEM
        f1_1, f2_1, f3_1, f4_1 = self.fpem1(f1, f2, f3, f4)
        f1_2, f2_2, f3_2, f4_2 = self.fpem2(f1_1, f2_1, f3_1, f4_1)

        # FFM
        f1 = f1_1 + f1_2
        f2 = f2_1 + f2_2
        f3 = f3_1 + f3_2
        f4 = f4_1 + f4_2

        # transformer
        t1 = torch.flatten(f1, start_dim=2)
        t2 = torch.flatten(f2, start_dim=2)
        t3 = torch.flatten(f3, start_dim=2)
        t4 = torch.flatten(f4, start_dim=2)

        token_t1 = self.tokenizer(t1)
        token_t2 = self.tokenizer(t2)
        token_t3 = self.tokenizer(t3)
        token_t4 = self.tokenizer(t4)

        all_token = torch.cat((token_t1, token_t2, token_t3, token_t4), dim=2)
        encoder1 = self.transformer(all_token)
        encoder2 = self.transformer(encoder1)

        e1, e2, e3, e4 = torch.split(encoder2, 32, dim=2)

        f1 = self.projector(t1, e1).view(-1, 128, h // 4, w // 4)
        f2 = self.projector(t2, e2).view(-1, 128, h //8, w // 8)
        f3 = self.projector(t3, e3).view(-1, 128, h // 16, w // 16)
        f4 = self.projector(t4, e4).view(-1, 128, h // 32, w //32)

        f2 = self._upsample(f2, f1.size())
        f3 = self._upsample(f3, f1.size())
        f4 = self._upsample(f4, f1.size())

        f = torch.cat((f1, f2, f3, f4), 1)
        f_s = self.reduce_layer_ts(f)

        # detection
        det_out = self.det_head(f_s)

        det_out = self._upsample(det_out, imgs.size(), 4)
        det_res = self.det_head.get_results(det_out, img_metas, cfg)
        outputs.update(det_res)

        return outputs



