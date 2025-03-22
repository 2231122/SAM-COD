#!/usr/bin/python3
#coding=utf-8

from configparser import Interpolation
import cv2
from torchvision.transforms.functional import rotate, InterpolationMode
import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
class Compose(object):
    def __init__(self, *ops):
        self.ops = ops

    def __call__(self, image, mask,mask1):
        for op in self.ops:
            image, mask,mask1 = op(image, mask,mask1)
        return image, mask,mask1

class RGBDCompose(object):
    def __init__(self, *ops):
        self.ops = ops

    def __call__(self, image, depth, mask):
        for op in self.ops:
            image, depth, mask = op(image, depth, mask)
        return image, depth, mask


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std  = std

    def __call__(self, image, mask,mask1):
        image = (image - self.mean)/self.std
        # mask /= 255
        return image, mask,mask1

class RGBDNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std  = std

    def __call__(self, image, depth, mask):
        image = (image - self.mean)/self.std
        depth = (depth - self.mean)/self.std
        mask /= 255
        return image, mask

class Resize(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, image, mask,mask1):
        image = cv2.resize(image, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        mask  = cv2.resize( mask, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        mask1 = cv2.resize(mask1, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        return image, mask,mask1

class RandomCrop(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, image, mask,mask1):
        H,W,_ = image.shape
        xmin  = np.random.randint(W-self.W+1)
        ymin  = np.random.randint(H-self.H+1)
        image = image[ymin:ymin+self.H, xmin:xmin+self.W, :]
        mask  = mask[ymin:ymin+self.H, xmin:xmin+self.W, :]
        mask1 = mask1[ymin:ymin + self.H, xmin:xmin + self.W, :]
        return image, mask,mask1

class RandomHorizontalFlip(object):
    def __call__(self, image, mask,mask1):
        if np.random.randint(2)==1:
            image = image[:,::-1,:].copy()
            mask  =  mask[:,::-1,:].copy()
            mask1 = mask1[:, ::-1, :].copy()
        return image, mask,mask1

class ToTensor(object):
    def __call__(self, image, mask,mask1):
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)
        mask  = torch.from_numpy(mask)
        mask  = mask.permute(2, 0, 1)

        mask1 = torch.from_numpy(mask1)
        mask1 = mask1.permute(2, 0, 1)
        return image, mask.mean(dim=0, keepdim=True),mask1.mean(dim=0, keepdim=True)

class Flip:
    def __init__(self, flip_num):
        assert flip_num in [0,1,2]
        self.flip = flip_num
    def __call__(self, img, msk = None):
        if self.flip==1:
            img = img.flip(-2)
        elif self.flip==2:
            img = img.flip(-1)
        return img, msk

class Rotate:
    def __init__(self, rot_degree):
        self.rot = rot_degree
    def __call__(self, img, msk = None):
        if np.random.randint(2)==1:
            img = rotate(img, self.rot, interpolation=InterpolationMode.BILINEAR)
        return img, msk

class RandomNoise:
    def __init__(self, noise_level):
        self.noise_level = noise_level
    def __call__(self, img, msk = None):
        noise = torch.randn_like(img) * self.noise_level
        img = img + noise
        return img, msk
#by chf:
# class Color(object):
#     def __call__(self, image, mask):
#         image=Image.fromarray(image.astype('uint8')).convert('RGB')
#         #transforms.RandomApply([color_jitter],p=0.8),
#         image=transforms.RandomGrayscale(p=0.2)(image)
#         image=np.array(image)
#         return image,mask



