import os
import cv2
import numpy as np
import shutil


mask_score0_path='./Masks_S_0'
mask_score1_path='./Masks_S_1'
mask_score2_path='./Masks_S_2'

P_Mask_Mo_path='./P_Mask_Mo'
Save_path='./Sv'
def Filter(image,a,b):
        W, H = image.shape
        count = W * H
        image1=image.copy()

        image1[image1 < 240]=0
        image1[image1 >= 240] = 1
        count1 = np.sum(image1)

        if (count1 / count) < a or (count1 / count) > b:
                return 0
        else:
                return 1
def Matcher(image1,image2,moban,score_list,i,state):
        moban1=moban.copy()
        moban1[moban1<150]=0
        moban1[moban1>=150]=1

        I1=image1.copy()
        I2=image2.copy()

        I1[I1==255]=1
        I2[I2==255]=1

        s1=moban1+I1
        s2=moban1+I2

        s1[s1==2]=0
        s2[s2==2]=0
        if state==0:
                sc_1=score_list[i][0]
                sc_2=score_list[i][1]
        else:
                sc_1=score_list[i][1]
                sc_2=score_list[i][2]

        s11=np.sum(s1)*sc_1

        s22=np.sum(s2)*sc_2



        if s11>s22:
                return 1
        else:
                return 2
def Matcher1(image0,image1,image2,P_mask,score_list,i):
        P_mask_1=P_mask.copy()
        P_mask_1[P_mask_1<150]=0
        P_mask_1[P_mask_1>=150]=1
        I0=image0.copy()
        I1=image1.copy()
        I2=image2.copy()

        I0[I0==255]=1
        I1[I1==255]=1
        I2[I2==255]=1

        s0=P_mask_1+I0
        s1=P_mask_1+I1
        s2=P_mask_1+I2

        s0[s0==2]=0
        s1[s1==2]=0
        s2[s2==2]=0

        sc_0 = score_list[i][0]
        sc_1 = score_list[i][1]
        sc_2 = score_list[i][2]

        s00=np.sum(s0)*sc_0
        s11=np.sum(s1)*sc_1
        s22=np.sum(s2)*sc_2


        if s11>s22 and s22<s00:
                return 1
        if s11<s22 and s11<s00:
                return 2
        if s00<s11 and s00<s22:
                return 0
file_list_mask = sorted(os.listdir(P_Mask_Mo_path))

score_list=[]
with open('score.txt','r') as file:
        for line in file:
                line = line.strip()
                parts=line.split()

                S_0=float(parts[1])
                S_1=float(parts[2])
                S_2=float(parts[3])
                score_list.append([S_0,S_1,S_2])
count=0
for file in file_list_mask[0:20]:
        P_mask_Mo = cv2.imread(os.path.join(P_Mask_Mo_path, file), cv2.IMREAD_GRAYSCALE)
        moban=Filter(P_mask_Mo,0.001,0.4)

        S_0_mask = cv2.imread(os.path.join(mask_score0_path, file[:-4] + '(0).png'), cv2.IMREAD_GRAYSCALE)
        S_1_mask = cv2.imread(os.path.join(mask_score1_path, file[:-4] + '(1).png'), cv2.IMREAD_GRAYSCALE)
        S_2_mask = cv2.imread(os.path.join(mask_score2_path, file[:-4] + '(2).png'), cv2.IMREAD_GRAYSCALE)
        S_0_filter=Filter(S_0_mask,0.05,0.5)
        S_1_filter=Filter(S_1_mask,0.001,0.5)
        S_2_filter=Filter(S_2_mask,0.001,0.5)
        Squ_sum=S_2_filter*100+S_1_filter*10+S_0_filter
        if Squ_sum==0:
                if moban==1:
                        final=P_mask_Mo
                else:
                        final=P_mask_Mo
        if Squ_sum==100:
                final=S_2_mask
        if Squ_sum==101:
                final=S_2_mask
        if Squ_sum==110 :
                if moban==1:
                        if Matcher(S_1_mask,S_2_mask,P_mask_Mo,score_list,count,1)==1:
                                final=S_2_mask
                        else:
                                final=S_1_mask
                else:
                        final=S_2_mask
        if Squ_sum == 111:
                if moban==1:
                        if Matcher1(S_0_mask,S_1_mask,S_2_mask,P_mask_Mo,score_list,count)==1:
                                final=S_2_mask
                        elif Matcher1(S_0_mask,S_1_mask,S_2_mask,P_mask_Mo,score_list,count)==2:
                                final=S_1_mask
                        else:
                                final=S_0_mask
                else:
                        final=S_2_mask
        if Squ_sum==11:
                if moban==1:
                        if Matcher(S_0_mask,S_1_mask,P_mask_Mo,score_list,count,0)==1:
                                final=S_1_mask
                        else:
                                final=S_0_mask
                else:
                        final=S_1_mask
        if Squ_sum==1:
                final=S_0_mask
        if Squ_sum==10:
                final=S_1_mask
        cv2.imwrite(os.path.join(Save_path, file), final)
        count=count+1
