import os
import random
from tracemalloc import Snapshot

import cv2
import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import affine
from torchvision.utils import draw_segmentation_masks
from PIL import Image
import tqdm

device = torch.device("cuda:0")
criterion=nn.CosineSimilarity(dim=1).to(device)
def get_featuremap(h, x):
    w = h.weight
    b = h.bias
    c = w.shape[1]
    c1 = F.conv2d(x, w.transpose(0,1), padding=(1,1), groups=c)
    return c1, b

def ToLabel(E):
    fgs = np.argmax(E, axis=1).astype(np.float32)
    return fgs.astype(np.uint8)


def SSIM(x, y):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = nn.AvgPool2d(3, 1, 1)(x)
    mu_y = nn.AvgPool2d(3, 1, 1)(y)
    mu_x_mu_y = mu_x * mu_y
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)

    sigma_x = nn.AvgPool2d(3, 1, 1)(x * x) - mu_x_sq
    sigma_y = nn.AvgPool2d(3, 1, 1)(y * y) - mu_y_sq
    sigma_xy = nn.AvgPool2d(3, 1, 1)(x * y) - mu_x_mu_y

    SSIM_n = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
    SSIM = SSIM_n / SSIM_d

    return torch.clamp((1 - SSIM) / 2, 0, 1)

"""chf"""
def generate_guass(f,sigma,w,h):
    #w,h=sigma.shape
    w=int(w.cpu())
    h=int(h.cpu())
    #C,H,W
    f[h,w]=1
    heatmap=cv2.GaussianBlur(f,sigma,0)
    am=np.amax(heatmap)
    heatmap/=am
    return heatmap

def Self_Knowledge_distilation(x, y, mask,x_h,y_w):
    l1_loss = torch.mean(torch.abs(x - y))
    return l1_loss



def SaliencyStructureConsistencynossim(x, y):
    l1_loss = torch.mean(torch.abs(x-y))
    return l1_loss


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class Flip:
    def __init__(self, flip):
        self.flip = flip
    def __call__(self, img):
        if self.flip==0:
            return img.flip(-1)
        else:
            return img.flip(-2)

class Translate:
    def __init__(self, fct):
        '''Translate offset factor'''
        drct = np.random.randint(0, 4)
        self.signed_x = drct>=2 or -1
        self.signed_y = drct%2 or -1
        self.fct = fct
    def __call__(self, img):
        angle = 0
        scale = 1
        h, w = img.shape[-2:]
        h, w = int(h*self.fct), int(w*self.fct)
        return affine(img, angle, (h*self.signed_y,w*self.signed_x), scale, shear=0, interpolation=InterpolationMode.BILINEAR)

class Crop:
    def __init__(self, H, W):
        '''keep the relative ratio for offset'''
        self.h = H
        self.w = W
        self.xm  = np.random.uniform()
        self.ym  = np.random.uniform()
        # print(self.xm, self.ym)
    def __call__(self, img):
        H,W = img.shape[-2:]
        sh = int(self.h*H)
        sw = int(self.w*W)
        ymin = int((H-sh+1)*self.ym)
        xmin = int((W-sw+1)*self.xm)
        img = img[..., ymin:ymin+ sh, xmin:xmin+ sw]
        img = F.interpolate(img, size=(H,W), mode='bilinear', align_corners=False)
        return img
#### by chf #### combin gray and glass:
from torchvision import transforms
from PIL import ImageFilter

class gray_scale:
    def __init__(self,p=0.2):
        self.p=p
        self.transf=transforms.Grayscale(3)
    def __call__(self,img):
        if random.random()<self.p:
            return self.transf(img)
        else:
            return img
class GaussianBlur(object):
    """blur a single image on CPU"""
    def __init__(self, kernel_size):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=3)
        self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=3)
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )

        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, img,msk):
        #img = self.pil_to_tensor(img).unsqueeze(0)
        #img=img.unsqueeze(0)
        img=img.cpu()
        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        #img = self.tensor_to_pil(img)

        return img
class mask1:
    def __init__(self):
        self.x=0
    def __call__(self, image,mask):
        n, c, w, h = mask.shape

        # FF = mask.squeeze(1).long()
        FF = mask.flatten(2)  # [0,1,255]
        x_mask_random = torch.zeros_like(FF)
        #x_mask_random1 = torch.zeros_like(FF)

        x_mask_random[:, :, 0:76800] = 1
        #x_mask_random1[:, :, 0:76800] = 1

        idx = torch.randperm(x_mask_random.shape[2])
        #idx1 = torch.randperm(x_mask_random1.shape[2])

        x_mask_random = x_mask_random[:, :, idx].view(x_mask_random.size())
        #x_mask_random1 = x_mask_random1[:, :, idx1].view(x_mask_random1.size())

        M_T = mask.long().flatten(2)
        # M_T[M_T != 1] = 2
        # M_T[M_T == 1] = 0
        # M_T[M_T == 2] = 1
        M_T[M_T == 255] = 2
        M_T[M_T != 2] = 0
        M_T[M_T == 2] = 1

        final_mask = x_mask_random + M_T
        final_mask = (final_mask > 0)

        # final_mask1=x_mask_random+M_T
        #final_mask1 = x_mask_random1 + M_T
        #final_mask1 = (final_mask1 > 0)
        # FF=(FF*final_mask).reshape([n,c,w,h])

        final_mask = final_mask.reshape([n, c, w, h])
        #final_mask1 = final_mask1.reshape([n, c, w, h])
        # FF1=()
        image_tr = image.clone()
        image_tr = image_tr * final_mask
        #image_tr[image_tr==0]=0
        #image = image * final_mask1
        return image_tr
class Color_jitter:
    """blur a single image on CPU"""
    def __init__(self):
        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()
        self.color_jitter = transforms.ColorJitter(0.8 * 1, 0.8 * 1, 0.8, 0.2)
    def __call__(self, img,msk):
        B,_,_,_=img.size()
        for i in range(B):
            imge=img[i,:,:,:]
            imge=self.tensor_to_pil(imge)
        #img = self.tensor_to_pil(img)
            imge = self.color_jitter(imge)

            imge =self.pil_to_tensor(imge)
            img[i,:,:,:]=imge
        return img


# import random
# import numpy as np
# from PIL import Image
# from torchvision.transforms import functional as F

class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.#固定长度的正方形边长
    """
    def __init__(self, p):
        self.p = p

        self.pil_to_tensor = transforms.ToTensor()
    def __call__(self, img,mask):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(2)
        w = img.size(3)

        grid_size = 8
        B, _, _, _ = img.size()
        for i in range(B):
            imge = img[i, :, :, :]
            #正方形区域中心点随机出现
            if grid_size > 0:
                imge=imge.cpu()
                imge = np.asarray(imge)
                for x in range(0, w, grid_size):
                    for y in range(0, h, grid_size):
                        x_end = min(w, x + grid_size)
                        y_end = min(h, y + grid_size)
                        if random.random() <= self.p:
                            imge[:,x:x_end, y:y_end] = 0
            #mask = torch.from_numpy(mask)
            #mask = mask.expand_as(imge)
            device = torch.device("cuda:1")
            imge=torch.from_numpy(imge)
            imge=imge.to(device)
            img[i, :, :, :] = imge
        return img

class HidePatch(object):
    def __init__(self, hide_prob=0.5):
        self.hide_prob = hide_prob

    def __call__(self, img):
        # get width and height of the image
        wd, ht = F.get_image_size(img)

        grid_size = 8  # For cifar, the patch size is set to be 8.

        # hide the patches
        if grid_size > 0:
            img = np.asarray(img)
            for x in range(0, wd, grid_size):
                for y in range(0, ht, grid_size):
                    x_end = min(wd, x + grid_size)
                    y_end = min(ht, y + grid_size)
                    if random.random() <= self.hide_prob:
                        img[x:x_end, y:y_end, :] = 0

        return Image.fromarray(img)