import argparse
import json
import os
import os.path as osp
import sys

import cv2
import numpy as np
import torch
from mmcv import Config
from eval.tt.eval import eval_tt
from dataset import build_data_loader
from models.knowledge_distillation_s_test import student_test
from models.utils import fuse_module
from utils import AverageMeter, Corrector, ResultFormat
from eval.ctw.eval import eval_batch
from eval.msra.eval import eval_mrsa
#from eval.ic15_end2end_rec.script_batch_eval import eval_batch

def model_structure(model):
    blank = ' '
    print('-' * 90)
    print('|' + ' ' * 11 + 'weight name' + ' ' * 10 + '|' \
          + ' ' * 15 + 'weight shape' + ' ' * 15 + '|' \
          + ' ' * 3 + 'number' + ' ' * 3 + '|')
    print('-' * 90)
    num_para = 0
    type_size = 1  ##如果是浮点数就是4

    for index, (key, w_variable) in enumerate(model.named_parameters()):
        if len(key) <= 30:
            key = key + (30 - len(key)) * blank
        shape = str(w_variable.shape)
        if len(shape) <= 40:
            shape = shape + (40 - len(shape)) * blank
        each_para = 1
        for k in w_variable.shape:
            each_para *= k
        num_para += each_para
        str_num = str(each_para)
        if len(str_num) <= 10:
            str_num = str_num + (10 - len(str_num)) * blank

        print('| {} | {} | {} |'.format(key, shape, str_num))
    print('-' * 90)
    print('The total number of parameters: ' + str(num_para))
    print('The parameters of Model {}: {:4f}M'.format(model._get_name(), num_para * type_size / 1000 / 1000))
    print('-' * 90)
    

def report_speed(outputs, speed_meters):
    total_time = 0
    for key in outputs:
        if 'time' in key:
            total_time += outputs[key]
            speed_meters[key].update(outputs[key])
            print('%s: %.4f' % (key, speed_meters[key].avg))

    speed_meters['total_time'].update(total_time)
    print('FPS: %.1f' % (1.0 / speed_meters['total_time'].avg))


def test(test_loader, model, cfg):
    model.eval()

    with_rec = hasattr(cfg.model, 'recognition_head')
    if with_rec:
        pp = Corrector(cfg.data.test.type, **cfg.test_cfg.rec_post_process)
    rf = ResultFormat(cfg.data.test.type, cfg.test_cfg.result_path)

    if cfg.report_speed:
        speed_meters = dict(backbone_time=AverageMeter(500),
                            neck_time=AverageMeter(500),
                            det_head_time=AverageMeter(500),
                            det_post_time=AverageMeter(500),
                            rec_time=AverageMeter(500),
                            total_time=AverageMeter(500))

    for idx, data in enumerate(test_loader):
        # print()
        print('Testing %d/%d\r' % (idx, len(test_loader)), flush=True, end='')
    
        # prepare input
        data['imgs'] = data['imgs'].cuda()
        data.update(dict(cfg=cfg))

        # img_show = data['imgs'].squeeze(0).permute(1, 2, 0).cpu().numpy()
        # mean = [0.485, 0.456, 0.406]
        # std = [0.229, 0.224, 0.225]
        # img_show = ((img_show * std + mean) * 255).astype(np.uint8)

        # forward
        with torch.no_grad():
            # outputs = model(**data)
            outputs = model(**data)
            # outputs, det_out = model(**data)

        if cfg.report_speed:
            report_speed(outputs, speed_meters)
        # post process of recognition
        if with_rec:
            outputs = pp.process(outputs)

        # save result
        image_name, _ = osp.splitext(
            osp.basename(test_loader.dataset.img_paths[idx]))
        rf.write_result(image_name, outputs)



def main(args):
    cfg = Config.fromfile(args.config)
    for d in [cfg, cfg.data.test]:
        d.update(dict(report_speed=args.report_speed))
    print(json.dumps(cfg._cfg_dict, indent=4))
    sys.stdout.flush()

    # data loader
    data_loader = build_data_loader(cfg.data.test)
    test_loader = torch.utils.data.DataLoader(
        data_loader,
        batch_size=1,
        shuffle=False,
        num_workers=2,
    )
    # model
    model = student_test(backbone='resnet18', neck='FPEM_v2', detection_head='PA_Head')
    model = model.cuda()

    if args.checkpoint is not None:
        if os.path.isfile(args.checkpoint):
            print("Loading model and optimizer from checkpoint '{}'".format(
                args.checkpoint))
            sys.stdout.flush()

            checkpoint = torch.load(args.checkpoint)

            d = dict()
            for key, value in checkpoint['state_dict'].items():
                tmp = key[7:]
                d[tmp] = value
            model.load_state_dict(checkpoint['state_dict'])
        else:
            print("No checkpoint found at '{}'".format(args.resume))
            raise

    # fuse conv and bn
    model = fuse_module(model)
    model_structure(model)

    # test
    test(test_loader, model, cfg)
    # eval ctw-1500
    eval_batch(checkpoint['epoch'])
    # eval total-text
    #eval_tt(checkpoint['epoch'])
    # eval mrsa
    #eval_mrsa(checkpoint['epoch'])
    # eval_e2e_icdar2015
    #eval_batch(checkpoint['epoch'])



def test_checkpoint(checkpoint="./checkpoints/checkpoint.pth.tar"):
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('config', help='config file path', default='./config/r18_ctw.py')
    parser.add_argument('checkpoint', nargs='?', type=str,
                        default=checkpoint)
    parser.add_argument('--report_speed', action='store_true')
    args = parser.parse_args()

    main(args)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('config', help='config file path')
    parser.add_argument('checkpoint', nargs='?', type=str, default='./checkpoints/checkpoint_566ep.pth.tar')
    parser.add_argument('--report_speed', action='store_true')
    args = parser.parse_args()

    main(args)
