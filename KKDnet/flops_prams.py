from models.knowledge_distillation_s import student
import torch

if __name__ == '__main__':
    from thop import profile

    model = student(backbone='resnet18', neck='FPEM_v2', detection_head='PA_Head')
    input = torch.randn(1, 3, 640, 640)
    flops, pram = profile(model, inputs=(input,))
    print("flops= {}".format(flops / 1e9))
    print("prams= {}".format(pram / 1e6))