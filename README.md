# KKDnet
Kernel-mask Knowledge Distillation (KKD)


## News
- (2022/05/20) KKDnet 

## Introduction

In this paper, we proposed a novel distillation method for efficient and accurate arbitrary-shaped text detection, termed Kernel-mask Knowledge Distillation (KKD), to improve the efficiency and accuracy of text detection.

![image.png](figure3.png)The overall network architecture diagram will be uploaded after publication

## Installation

First, clone the repository locally:

```shell
git clone https://github.com/giganticpower/KKDnet.git
```

required:

PyTorch 1.1.0+

torchvision 0.3.0+

pip install -r requirement.txt

compile codes of post-processing:
```
sh ./compile.sh
```

## Dataset
Please refer to [dataset/README.md](dataset/README.md) for dataset preparation.

## Training

first stage:
we train the teacher network
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py ${CONFIG_FILE}
```
For example:
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py config/kkd/r50_ctw.py
```

second stage: knowledge distillation -> training the student network

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python knowledge_distillation_train.py
```
NOTE:
in knowledge_distillation_train.py, we should chose the dataloader in the code 162 - 184 lines

## Testing

### Evaluate

one checkpoint testing:

```shell
python test.py ${CONFIG_FILE} ${CHECKPOINT_FILE}
```
For example:
```shell
python test.py config/kkd/r18_ctw.py checkpoints/checkpoint.pth.tar
```
batch checkpoint testing:

```shell
python batch_eval.py ${CONFIG_FILE}
```
For example:
```shell
python batch_eval.py config/kkd/r18_ctw.py
```
note:

1.when batch checkpoint testing, you should change the checkpoint path in batch_eval.py

2.you should change the eval dataset in test.py 151-160 lines. (Here's a quick prediction link we've designed for convenience)


## Citation

Please cite the related works in your publications if it helps your research:

### KKDnet

```
Stay tuned
```

## License

This project is developed and maintained by [Fuzhou University].

