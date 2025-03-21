import cv2
from skimage import morphology, draw

import matplotlib.pyplot as plt

import os
import cv2

image_path = './Scribble'
save_path = './point-set'

file_list_mask = sorted(os.listdir(image_path))

alpha=0.075

for file in file_list_mask[0:2]:
    input_image = cv2.imread(os.path.join(image_path, file), cv2.IMREAD_GRAYSCALE)
    W,H=input_image.shape

    rate_W=int(alpha*W)
    rate_H=int(alpha*H)
    rate_x=min(rate_H,rate_W)

    time_x=W//rate_x
    time_y=H//rate_x
    grid_mask=input_image.copy()
    grid_mask[grid_mask!=0]=0
    for i in range(time_x):
        for j in range(H):
            grid_mask[i*rate_x-2,j]=1
    for i in range(W):
        for j in range(time_y):
            grid_mask[i,j*rate_x-2]=1

    fg_image = input_image.copy()
    bg_image = input_image.copy()
    fg_image[fg_image != 1] = 0
    bg_image[bg_image != 2] = 0
    bg_image[bg_image == 2] = 1
    fg_image = fg_image* 255
    bg_image = bg_image* 255

    fg_image = morphology.skeletonize(fg_image)+0
    bg_image = morphology.skeletonize(bg_image)+0
    bg_image[bg_image==1]=2

    input_image[input_image != 0] = 0

    input_image=input_image+fg_image+bg_image
    input_image = input_image * grid_mask
    for i in range(W):
        for j in range(H):
            if input_image[i,j]!=0:
                boo=input_image[i,j]
                input_image[i:i+rate_x-1,j:j+rate_x-1]=0
                input_image[i,j]=boo

    cv2.imwrite(os.path.join(save_path,file), input_image)
