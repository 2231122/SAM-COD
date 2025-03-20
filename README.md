# [ECCV2024] SAM-COD: SAM-guided Unified Framework for Weakly-Supervised Camouflaged Object Detection

![Framework](figure/Framework.png)


# Prerequisites
- python==3.7.5
- torch==1.13.1
- Torchvision==0.14.1
- Scikit_image==0.19.2
- Skimage==0.0
- timm==0.3.2
- tensorboard==2.11.2
- tensorboardX==2.5.1
- tqdm==4.64.1
- einops==0.4.1
- markdown==3.4.3
- markplotlib==3.5.2
- numpy==1.12.6
- opencv-python==4.7.0.72
- openpyxl==3.1.2
- pillow==9.5.0
- pysodmetrics==1.4.0
- PyYAML==6.0
- tabulate==0.9.0

# Download P-COD and B-COD Dataset
- Point supervised dataset P-COD: [google](https://drive.google.com/file/d/17oa6-IU2Dr9Q1KKQ74UoL0hoFd5F7bOd/view?usp=sharing)
- Box supervised dataset B-COD: [google](https://drive.google.com/file/d/1Ds1kBbk1Ifq6awWcIqbQrF79PVwGZW-G/view?usp=sharing)

# Using Segment Anything Model
- Following the [SAM](https://github.com/facebookresearch/segment-anything) to create an environment.

- Box-prompt:
```shell
train.py  
```
- Point-prompt:
```shell
train.py
```
- Scribble-prompt:
```shell
train.py
``` 
# Encoder&Decoder
- Just download the dataset and pretrained model. 
- The pretrained model weight can be found here: [Pretrain_model](https://drive.google.com/file/d/1169AvHlRnyKdScEHm6yWKSyne3j0N2EZ/view?usp=sharing) . (put it in './SAM-guided-Unified-Framework-for-Weakly-Supervised-Camouflaged-Object-Detection/Pretrain_model.pth')
- The masks for distillation are in the path './CodDataset/train/masks'

- Train:
```shell
python train.py
```
- Test and Evaluate:
```shell
python test.py
```
# Using SAM 
I've been a little busy lately. The coding coming soon...

# Adapter&Filter&Matcher

# Dataset

