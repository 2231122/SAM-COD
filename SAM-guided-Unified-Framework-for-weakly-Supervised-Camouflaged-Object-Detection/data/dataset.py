#!/usr/bin/python3
#coding=utf-8
import json
import os
import os.path as osp
import cv2
import torch
import numpy as np
try:
    from . import transform
except:
     import transform

from torch.utils.data import Dataset, DataLoader
from lib.data_prefetcher import DataPrefetcher
import torchvision.transforms as transforms



class Config(object):
    def __init__(self, **kwargs):
        if kwargs.get('label_dir') is None:
            kwargs['label_dir'] = 'Scribble'
        self.kwargs    = kwargs
        print('\nParameters...')
        for k, v in self.kwargs.items():
            print('%-10s: %s'%(k, v))

        if 'ECSSD' in self.kwargs['datapath']:
            self.mean      = np.array([[[117.15, 112.48, 92.86]]])
            self.std       = np.array([[[ 56.36,  53.82, 54.23]]])
        elif 'DUTS' in self.kwargs['datapath']:
            self.mean      = np.array([[[124.55, 118.90, 102.94]]])
            self.std       = np.array([[[ 56.77,  55.97,  57.50]]])
        elif 'DUT-OMRON' in self.kwargs['datapath']:
            self.mean      = np.array([[[120.61, 121.86, 114.92]]])
            self.std       = np.array([[[ 58.10,  57.16,  61.09]]])
        elif 'MSRA-10K' in self.kwargs['datapath']:
            self.mean      = np.array([[[115.57, 110.48, 100.00]]])
            self.std       = np.array([[[ 57.55,  54.89,  55.30]]])
        elif 'MSRA-B' in self.kwargs['datapath']:
            self.mean      = np.array([[[114.87, 110.47,  95.76]]])
            self.std       = np.array([[[ 58.12,  55.30,  55.82]]])
        elif 'SED2' in self.kwargs['datapath']:
            self.mean      = np.array([[[126.34, 133.87, 133.72]]])
            self.std       = np.array([[[ 45.88,  45.59,  48.13]]])
        elif 'PASCAL-S' in self.kwargs['datapath']:
            self.mean      = np.array([[[117.02, 112.75, 102.48]]])
            self.std       = np.array([[[ 59.81,  58.96,  60.44]]])
        elif 'HKU-IS' in self.kwargs['datapath']:
            self.mean      = np.array([[[123.58, 121.69, 104.22]]])
            self.std       = np.array([[[ 55.40,  53.55,  55.19]]])
        elif 'SOD' in self.kwargs['datapath']:
            self.mean      = np.array([[[109.91, 112.13,  93.90]]])
            self.std       = np.array([[[ 53.29,  50.45,  48.06]]])
        elif 'THUR15K' in self.kwargs['datapath']:
            self.mean      = np.array([[[122.60, 120.28, 104.46]]])
            self.std       = np.array([[[ 55.99,  55.39,  56.97]]])
        elif 'SOC' in self.kwargs['datapath']:
            self.mean      = np.array([[[120.48, 111.78, 101.27]]])
            self.std       = np.array([[[ 58.51,  56.73,  56.38]]])
        else:
            #raise ValueError
            self.mean = np.array([[[0.485*256, 0.456*256, 0.406*256]]])
            self.std = np.array([[[0.229*256, 0.224*256, 0.225*256]]])
            # self.std, self.mean = np.array([0.1861761914527739, 0.19748777412623036, 0.2032849354904543])[None,None]*255, np.array([0.3320486163733052, 0.432231354815684, 0.449829585669272])[None,None]*255

    def __getattr__(self, name):
        if name in self.kwargs:
            return self.kwargs[name]
        else:
            return None


class Data(Dataset):
    def __init__(self, cfg):

        self.cfg = cfg

        self.data_name = cfg.datapath.split('/')[-1]
        #val_cfg = [Dataset.Config(datapath=f'{root}/test/{i}', mode='test') for i in ['CHAMELEON', 'CAMO', 'COD10K']]

        with open(cfg.datapath+'/'+cfg.mode+'.txt', 'r') as lines:
            self.samples = []
            if cfg.mode == 'train':
              for line in lines:

                  # imagepath = cfg.datapath +'/'+cfg.mode+ '/Imgs/' + line.strip() + '.jpg'
                  # maskpath  = cfg.datapath + '/'+cfg.mode+f'/{cfg.label_dir}/'  + line.strip() + '.png'
                  imagepath = cfg.datapath + '/' + cfg.mode + '/Imgs/' + line.strip() + '.jpg'

                  maskpath = cfg.datapath + '/' + cfg.mode + f'/{cfg.label_dir}/' + line.strip() + '.png'
                  maskpath1=cfg.datapath+'/'+cfg.mode+'/'+'S_GT'+'/'+line.strip()+'.png'



                  pointpath = cfg.datapath + '/' +cfg.mode+ '/Point/'+line.strip()+'.json'
                  # self.samples.append([imagepath, maskpath])
                  self.samples.append([imagepath, maskpath,pointpath,maskpath1])
            else:
              for line in lines:
                  # print(cfg.datapath)
                  # print('----')

                  if cfg.datapath == './CodDataset/test/WSA4':
                      imagepath = cfg.datapath + '/Imgs/' + line.strip() + '.png'
                  else:
                      imagepath =  cfg.datapath + '/Imgs/' + line.strip() + '.jpg'
                  #imagepath = cfg.datapath + '/Imgs/' + line.strip() + '.jpg'
                  maskpath  = cfg.datapath +'/GT/'  + line.strip() + '.png'
                  maskpath1 = cfg.datapath + '/GT/' + line.strip() + '.png'
                  pointpath ='./CodDataset' + '/' + 'train' + '/Point/' + 'camourflage_00001' + '.json'
                  # print(imagepath)
                  # print(maskpath)
                  # print('------')
                  self.samples.append([imagepath, maskpath,pointpath,maskpath1])

        if cfg.mode == 'train':
            self.transform = transform.Compose(transform.Normalize(mean=cfg.mean, std=cfg.std),
                                               transform.Resize(512, 512),
                                               # transform.Color(),
                                               transform.RandomHorizontalFlip(),
                                               transform.RandomCrop(512, 512),
                                               transform.ToTensor())
            # self.transform = transforms.Compose([transforms.Normalize(mean=cfg.mean, std=cfg.std),
            #                                         transforms.Resize([320, 320]),
            #                                         transforms.RandomGrayscale(p=0.2),
            #                                         transforms.RandomHorizontalFlip(),
            #                                         transforms.RandomCrop(320, 320),
            #                                         transforms.ToTensor()])
        elif cfg.mode == 'test':
            self.transform = transform.Compose(transform.Normalize(mean=cfg.mean, std=cfg.std),
                                                    transform.Resize(512, 512),
                                                    transform.ToTensor()
                                                )
        else:
            raise ValueError

    def __getitem__(self, idx):
        imagepath, maskpath,pointpath,maskpath1 = self.samples[idx]

        image               = cv2.imread(imagepath).astype(np.float32)[:,:,::-1]#cv2.imread->[h,w,3]
        mask                = cv2.imread(maskpath).astype(np.float32)[:,:,::-1]
        mask_gt=cv2.imread(maskpath1).astype(np.float32)[:,:,::-1]
        """Point_process"""
        H, W, C = mask.shape
        point_json = json.load(open(pointpath))
        for point in point_json['shapes']:
            if point['label']=='fg':
                x_h=point['points'][0][0]#W
                y_w=point['points'][0][1]#H
                x_h=(x_h*320)/W
                y_w=(y_w*320)/H

        """xxxxxxxxxxxxx"""

        if self.cfg.mode == 'train':
            image, mask,mask_gt         = self.transform(image, mask,mask_gt)
            mask[mask == 0.] = 255.
            mask[mask == 2.] = 0.

            mask_gt[mask_gt==0.]=255.
            mask_gt[mask_gt==2.]=0.
        else:
            image, _ ,_        = self.transform(image, mask,mask_gt)
            mask = torch.from_numpy(mask.copy()).permute(2,0,1)
            mask = mask.mean(dim=0, keepdim=True)
            mask /= 255

            mask_gt = torch.from_numpy(mask_gt.copy()).permute(2, 0, 1)
            mask_gt = mask_gt.mean(dim=0, keepdim=True)
            mask_gt /= 255
        # print(image.max(), image.min())
        return image, mask, (H, W), maskpath.split('/')[-1],x_h,y_w,mask_gt

    def __len__(self):
        return len(self.samples)


if __name__=='__main__':
    import matplotlib.pyplot as plt
    plt.ion()
    cfg  = Config(mode='test', datapath='/dataC/qhd/cod/CodDataset')
    data = Data(cfg)
    loader = DataLoader(data, batch_size=1, shuffle=True, num_workers=8)
    prefetcher = DataPrefetcher(loader)
    batch_idx = -1
    image, mask = prefetcher.next()
    image = image[0].permute(1,2,0).cpu().numpy()*cfg.std + cfg.mean
    mask  = mask[0].cpu().numpy()
    plt.subplot(121)
    plt.imshow(np.uint8(image))
    plt.subplot(122)
    plt.imshow(mask)
    input()

