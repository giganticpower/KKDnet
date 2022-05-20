import argparse
import json
import os
import os.path as osp
import random
import sys
import time
import numpy as np
import torch
from mmcv import Config
import cv2
from dataset import build_data_loader
from models import build_model
from utils import AverageMeter
from dataset.detect.kkd_ctw import KKD_CTW
from dataset.detect.kkd_tt import KKD_TT
from dataset.detect.kkd_msra import KKD_MSRA
from models.knowledge_distillation_t import teacher
from models.knowledge_distillation_s import student
from models.loss.functions import cal_loss
import viz_image_func
from utils.summary import LogSummary

torch.manual_seed(123456)
torch.cuda.manual_seed(123456)
np.random.seed(123456)
random.seed(123456)

def train(train_loader, model_t, model_s, optimizer, epoch, start_iter, logger, args):
    model_t.eval()
    for pram in model_t.parameters():
        pram.requires_grad = False
    model_s.train()

    # meters
    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses = AverageMeter()
    losses_ts = AverageMeter()
    losses_text = AverageMeter()
    losses_kernels = AverageMeter()
    losses_emb = AverageMeter()
    losses_rec = AverageMeter()

    ious_text = AverageMeter()
    ious_kernel = AverageMeter()
    accs_rec = AverageMeter()


    # start time
    start = time.time()
    for iter, data in enumerate(train_loader):
        # skip previous iterations
        if iter < start_iter:
            print('Skipping iter: %d' % iter)
            sys.stdout.flush()
            continue

        # time cost of data loader
        data_time.update(time.time() - start)

        # adjust learning rate
        adjust_learning_rate(optimizer, train_loader, epoch, iter, args)

        data = {key: data[key].cuda() for key in data}
        # forward
        outputs_t = model_t(**data)
        outputs_s, train_mask, outputs = model_s(**data)

        loss_ts = cal_loss(outputs_t, outputs_s, train_mask)
        losses_ts.update(loss_ts.item())
        loss_text = torch.mean(outputs['loss_text'])
        losses_text.update(loss_text.item())

        loss_kernels = torch.mean(outputs['loss_kernels'])
        losses_kernels.update(loss_kernels.item())
        if 'loss_emb' in outputs.keys():
            loss_emb = torch.mean(outputs['loss_emb'])
            losses_emb.update(loss_emb.item())
            loss = loss_text + loss_kernels + loss_emb + loss_ts
        else:
            loss = loss_text + loss_kernels + 0.5 * loss_ts

        iou_text = torch.mean(outputs['iou_text'])
        ious_text.update(iou_text.item())
        iou_kernel = torch.mean(outputs['iou_kernel'])
        ious_kernel.update(iou_kernel.item())

        losses.update(loss.item())

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - start)

        # update start time
        start = time.time()

        # print log
        if iter % 20 == 0:
            length = len(train_loader)
            log = f'({iter + 1}/{length}) ' \
                  f'LR: {optimizer.param_groups[0]["lr"]:.6f} | ' \
                  f'Batch: {batch_time.avg:.3f}s | ' \
                  f'Total: {batch_time.avg * iter / 60.0:.0f}min | ' \
                  f'ETA: {batch_time.avg * (length - iter) / 60.0:.0f}min | ' \
                  f'Loss: {losses.avg:.3f} | ' \
                  f'Loss(text/kernel/emb{""}): ' \
                  f'{losses_text.avg:.3f}/{losses_kernels.avg:.3f}/' \
                  f'{losses_emb.avg:.3f} |' \
                  f'IoU(text/kernel): {ious_text.avg:.3f}/{ious_kernel.avg:.3f} |'\
                  f'{losses_ts.avg:.3f} |'
            print(log)
            sys.stdout.flush()

        if iter % 100 == 0:
            logger.write_scalars({
                'loss': loss.item(),
                'text_loss': loss_text.item(),
                'kernels_loss': loss_kernels.item(),
                'emb_loss': loss_emb.item(),
                'ts_loss': loss_ts.item(),
            }, tag='train', n_iter=iter)


def adjust_learning_rate(optimizer, dataloader, epoch, iter, args):
    schedule = 'polylr'
    if isinstance(schedule, str):
        assert schedule == 'polylr', 'Error: schedule should be polylr!'
        cur_iter = epoch * len(dataloader) + iter
        max_iter_num = args.end_epoch * len(dataloader)
        lr = args.lr * (1 - float(cur_iter) / max_iter_num)**0.9
    elif isinstance(schedule, tuple):
        lr = args.lr
        for i in range(len(schedule)):
            if epoch < schedule[i]:
                break
            lr = lr * 0.1

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_checkpoint(state, checkpoint_path, args):
    file_path = osp.join(checkpoint_path, 'checkpoint.pth.tar')
    torch.save(state, file_path)

    if (state['iter'] == 0 and state['epoch'] > args.end_epoch - 200 and
            (state['epoch'] % 10 == 0 or state['epoch'] % 10 == 5)):
        file_name = 'checkpoint_%dep.pth.tar' % state['epoch']
        file_path = osp.join(checkpoint_path, file_name)
        torch.save(state, file_path)


def main(args):
    checkpoint_path = 'knowledge_distillation_checkpoints'
    print('Checkpoint path: %s.' % checkpoint_path)

    # data loader ctw
    data_loader = KKD_CTW(split='train',
        is_transform=True,
        img_size=640,
        short_size=640,
        kernel_scale=0.7,
        read_type='cv2')

    # data loader  tt
    # data_loader = KKD_TT(split='train',
    #                      is_transform=True,
    #                      img_size=640,
    #                      short_size=640,
    #                      kernel_scale=0.7,
    #                      read_type='cv2')
    #
    # data mrsa
    # data_loader = KKD_MSRA(split='train',
    #     is_transform=True,
    #     img_size=736,
    #     short_size=736,
    #     kernel_scale=0.7,
    #     read_type='cv2')

    train_loader = torch.utils.data.DataLoader(data_loader,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=12,
                                               drop_last=True,
                                               pin_memory=True)


    # log
    log_dir = os.path.join('./logs/', args.exp_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger = LogSummary(log_dir)

    # model
    model_t = teacher(backbone='resnet50', neck='FPEM_v2', detection_head='PA_Head')
    model_s = student(backbone='resnet18', neck='FPEM_v2', detection_head='PA_Head')

    model_t = torch.nn.DataParallel(model_t).cuda()
    model_s = torch.nn.DataParallel(model_s).cuda()
    # model_t = model_t.cuda()
    # model_s = model_s.cuda()

    from collections import OrderedDict

    # loading checkpoints
    checkpoint_t = torch.load(args.checkpoint_t)

    # new_state_dict = OrderedDict()
    # for k, v in checkpoint_t['state_dict'].items():
    #     name = k[7:]  # remove `module.`
    #     new_state_dict[name] = v

    model_t.load_state_dict(checkpoint_t['state_dict'])
    print("loading teacher pre-model successfully")
    if args.checkpoint_s:
        checkpoint_s = torch.load(args.checkpoint_s)
        # new_state_dict1 = OrderedDict()
        # for k, v in checkpoint_s['state_dict'].items():
        #     name = k[7:]  # remove `module.`
        #     new_state_dict1[name] = v
        model_s.load_state_dict(checkpoint_s['state_dict'], strict=False)
        print("loading student pre-model successfully")
    else:
        print("no use student pre-model")

    # optimizer
    optimizer = torch.optim.Adam(model_s.parameters(), lr=args.lr)
    #optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9)

    start_epoch = 0
    start_iter = 0

    for epoch in range(start_epoch, args.end_epoch):
        print('\nEpoch: [%d | %d]' % (epoch + 1, args.end_epoch))

        train(train_loader, model_t, model_s, optimizer, epoch, start_iter, logger, args)

        state = dict(epoch=epoch + 1,
                     iter=0,
                     state_dict=model_s.state_dict(),
                     optimizer=optimizer.state_dict())
        save_checkpoint(state, checkpoint_path, args)


if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--checkpoint_t', type=str, default='/checkpoints/teacher/checkpoint_566ep.pth.tar')
    parser.add_argument('--checkpoint_s', type=str, default=None)
    parser.add_argument('--resume', nargs='?', type=str, default=None)
    parser.add_argument('--end_epoch', type=int, default=600)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--exp_name', type=str, default='ctw')
    args = parser.parse_args()
    # with torch.cuda.device(2):
    # CUDA_VISIBLE_DEVICES=1
    main(args)
