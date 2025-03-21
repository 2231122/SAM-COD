#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Copyright (c) Meta Platforms, Inc. and affiliates.


# # Object masks from prompts with SAM

# The Segment Anything Model (SAM) predicts object masks given prompts that indicate the desired object. The model first converts the image into an image embedding that allows high quality masks to be efficiently produced from a prompt. 
# 
# The `SamPredictor` class provides an easy interface to the model for prompting the model. It allows the user to first set an image using the `set_image` method, which calculates the necessary image embeddings. Then, prompts can be provided via the `predict` method to efficiently predict masks from those prompts. The model can take as input both point and box prompts, as well as masks from the previous iteration of prediction.

import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import json
import sys
import os
from segment_anything import sam_model_registry, SamPredictor
# In[6]:


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    

sys.path.append("..")

sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor=SamPredictor(sam)

Image_pth='./Images'
M0_save_pth='./Masks_S_0'
M1_save_pth='./Masks_S_1'
M2_save_pth='./Masks_S_2'
Json_pth='./Box_json'
file_score_path='./score.txt'

Image_list=sorted(os.listdir(Image_pth))
count=0
for I in Image_list:
    image = cv2.imread(os.path.join(Image_pth,I))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)

    # [x,y,x,y]=[w,h,w,h]
    box_coor=[]
    data=json.load(open(os.path.join(Json_pth,I.split('.')[0]+'.json')))
    for point in data['shapes']:
        if point['label'] == 'bg':
            w_min=int(point['points'][0][0])
            h_min=int(point['points'][0][1])
            w_max=int(point['points'][1][0])
            h_max=int(point['points'][1][1])
            boox=[w_min,h_min,w_max,h_max]
            box_coor.append(boox)

    input_boxes = torch.tensor(box_coor, device=predictor.device)

    transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
    masks,scores,logits = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=True,
    )


    score_b = 0
    score_min = 50.0
    score_max = 0.0


    """ multimask_output==False  """
    # mask_f=masks[0]
    # for mask in masks[1:]:
    #     mask_f=mask_f+mask
    # mask_f[mask_f>0]=1
    # mask_f=mask_f*255
    # mask_f=torch.squeeze(mask_f,dim=0)
    # print(mask_f.shape)
    # o_m = Image.fromarray(mask_f.detach().cpu().numpy().astype(np.uint8))
    # m_name = os.path.join(Save_pth, I)
    # m_name_png = m_name.rsplit(".", 1)
    # output = m_name_png[0] + ".png"
    # o_m.save(output)

    """ multimask_output==True """
    masks=masks.transpose(0,1)
    scores=scores.transpose(0,1)

    for i, (mask, score) in enumerate(zip(masks, scores)): # number dim
          #print(mask.shape) [3,W,H] ; score.shape=[1,3]
          score_temp=0
          for j,(mask,score) in enumerate(zip(mask,score)):
              score_temp=score_temp+score
          if score_max < score_temp:
              score_max = score_temp
          if score_min > score_temp:
              score_min = score_temp
    for i, (mask, score) in enumerate(zip(masks, scores)):
          score_temp=0
          for j, (mask, score) in enumerate(zip(mask, score)):
              if j==0:
                  mask_temp=mask
              else:
                  mask_temp=mask_temp+mask
              score_temp=score_temp+score
          if score_temp!=score_max and score_temp!=score_min:
              score_mid=score_temp
          mask_temp[mask_temp>0]=1
          mask_temp=mask_temp*255
          mask_temp=torch.squeeze(mask_temp,dim=0)
          o_m = Image.fromarray(mask_temp.detach().cpu().numpy().astype(np.uint8))

          if score_temp == score_min:
              o_m.save(os.path.join(M0_save_pth,I).rsplit(".", 1)[0]+"(0)" + ".png")
          if score_temp == score_max:
              o_m.save(os.path.join(M2_save_pth,I).rsplit(".", 1)[0]+"(2)" + ".png")
          if score_temp > score_min and score_temp < score_max:
              o_m.save(os.path.join(M1_save_pth,I).rsplit(".", 1)[0]+"(1)" + ".png")
    with open(file_score_path, 'a') as sf:
        sf.write(I + ' ')
        sf.write(str(score_min.item()) + ' ')
        sf.write(str(score_mid.item()) + ' ')
        sf.write(str(score_max.item()) + ' ')
        sf.write('\n')