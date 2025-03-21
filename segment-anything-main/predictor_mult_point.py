import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


import json
import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor

sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)



predictor=SamPredictor(sam)

Image_pth='./Images'
#'./images'
Mask_pth='./Point'
M0_save_pth='./Masks_S_0'
M1_save_pth='./Masks_S_1'
M2_save_pth='./Masks_S_2'
file_score_path='./score.txt'
import os
Image_list=sorted(os.listdir(Image_pth))
count=0
for I in Image_list:
    image = cv2.imread(os.path.join(Image_pth,I))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    m_name=os.path.join(Mask_pth,I)
    m_name_png=m_name.rsplit(".",1)
    output=m_name_png[0] + ".png"
    m=cv2.imread(output,cv2.IMREAD_GRAYSCALE)
    count=count+1
    predictor.set_image(image)
    W,H=m.shape
    input_point_list=[]
    input_label_list=[]

    #W=266 H=400(kuan)
    m_bg=m.copy()
    m_fg=m.copy()
    m_bg[m_bg!=2]=0
    m_fg[m_fg!=1]=0
    m_fg[m_fg==1]=150


    while np.max(m_fg)>105:
        P=m_fg.argmax()
        fg_x=P%H
        fg_y=P//H
        m_fg[fg_y,fg_x]=0
        point = [fg_x, fg_y]
        input_point_list.append(point)
        input_label_list.append(1)

    while np.max(m_bg) > 1:# weather need to add 5 ??
        P = m_bg.argmax()
        bg_x = P % H
        bg_y = P // H

        m_bg[bg_y,bg_x]=0

        point = [bg_x, bg_y]
        input_point_list.append(point)
        input_label_list.append(0)

    print(count)
    input_label=np.array(input_label_list)
    input_point=np.array(input_point_list)

    if input_point_list!=[]:
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
        score_b=0
        score_min=1.5
        score_max=0.0

        for i, (mask, score) in enumerate(zip(masks, scores)):
            if score>score_b:
                score_b=score
                mask_f=mask
        from PIL import Image
        for i, (mask, score) in enumerate(zip(masks, scores)):
                if score_max<score:
                    score_max=score
                if score_min>score:
                    score_min=score
        for i, (mask, score) in enumerate(zip(masks, scores)):
                if score!=score_min and score!=score_max:
                    score_mid=score
                score_b=score
                mask_f=mask

                mask = mask_f * 255
                o_m = Image.fromarray(mask.astype(np.uint8))

                if score_temp == score_min:
                    o_m.save(os.path.join(M0_save_pth, I).rsplit(".", 1)[0] + "(0)" + ".png")
                if score_temp == score_max:
                    o_m.save(os.path.join(M2_save_pth, I).rsplit(".", 1)[0] + "(2)" + ".png")
                if score_temp > score_min and score_temp < score_max:
                    o_m.save(os.path.join(M1_save_pth, I).rsplit(".", 1)[0] + "(1)" + ".png")
        with open(file_score_path, 'a') as sf:
            sf.write(I + ' ')
            sf.write(str(score_min.item()) + ' ')
            sf.write(str(score_mid.item()) + ' ')
            sf.write(str(score_max.item()) + ' ')
            sf.write('\n')