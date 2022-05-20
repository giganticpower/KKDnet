import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .backbone.resnet import resnet50
from .neck.fpem_v2 import FPEM_v2
from .neck.fpem_v1 import FPEM_v1
from .utils import Conv_BN_ReLU
from models.neck.visual_transformer_noxin import FilterBasedTokenizer, Transformer, Projector
import cv2
import numpy as np

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


class teacher_train(nn.Module):
    def __init__(self, backbone='resnet50', neck='FPEM_v1', detection_head='PA_Head'):
        super(teacher_train, self).__init__()
        if backbone == 'resnet50':
            self.backbone = resnet50(pretrained=False)
        else:
            print('error backbone')

        if neck == 'FPEM_v1':
            neck = FPEM_v1(in_channels=(256, 512, 1024, 2048),
        out_channels=128)
        elif neck == 'FPEM_v2':
            neck = FPEM_v2(in_channels=(256, 512, 1024, 2048),
        out_channels=128)
        else:
            print('error neck')

        if detection_head == 'PA_Head':
            self.det_head = det_Head(in_channels=512,
        hidden_dim=128,
        num_classes=6)
        else:
            print('error detection_head')

        in_channels = (256, 512, 1024, 2048)
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

        if not self.training:
            torch.cuda.synchronize()
            start = time.time()

        # backbone
        f = self.backbone(imgs)

        if not self.training:
            torch.cuda.synchronize()
            outputs.update(dict(backbone_time=time.time() - start))
            start = time.time()

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

        # detection
        det_out = self.det_head(f)

        if self.training:
            det_out = self._upsample(det_out, imgs.size())
            det_loss = self.det_head.loss(det_out, gt_texts, gt_kernels,
                                          training_masks, gt_instances,
                                          gt_bboxes)
            outputs.update(det_loss)
        else:
            det_out = self._upsample(det_out, imgs.size(), 4)
            det_res = self.det_head.get_results(det_out, img_metas, cfg)
            outputs.update(det_res)

        return outputs
