import torch.nn.functional as F
import torch
from feature_loss import *
from tools import *
from utils import ramps
import numpy as np
device=torch.device("cuda:3")
criterion = torch.nn.CrossEntropyLoss(weight=None, ignore_index=255, reduction='mean').to(device)
loss_lsc = FeatureLoss().to(device)
loss_lsc_kernels_desc_defaults = [{"weight": 1, "xy": 6, "rgb": 0.1}]
loss_lsc_radius = 5
l = 0.3

def get_current_consistency_weight(epoch, consistency=0.1, consistency_rampup=150):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return consistency * ramps.sigmoid_rampup(epoch, consistency_rampup)
def flood(label,P_fg,x,y,h,count):
    #label:only 1 and 0
    label=1-label#翻转前景和其他
    det_p=label*P_fg[:,1:2,:,:]

    book=label*0
    book1=label*0+1
    N,C,W,H=label.shape

    #320 ?
    m_x=H
    m_y=W
    #x and y is [16,1]
    l_up_x = np.zeros(N).astype(int)
    l_up_y = np.zeros(N).astype(int)
    for i in range(N):
        l_up_x[i] = x[i] - h
        l_up_y[i] = y[i] - h
        if l_up_x[i]<0:
            l_up_x[i]=0
        if l_up_y[i]<0:
            l_up_y[i]=0
        #next ???
        if l_up_x[i]+2*h>m_x-1:
            l_up_x[i]=m_x-1-2*h
        if l_up_y[i]+2*h>m_y-1:
            l_up_y[i]=m_y-1-2*h
        #print=(h)
        #y's suoyin is at front !

        # book[i,0,l_up_x[i],l_up_y[i]:l_up_y[i]+2*h]=1
        # book[i,0,l_up_x[i]+2*h,l_up_y[i]:l_up_y[i]+2*h]=1
        # book[i,0,l_up_x[i]:l_up_x[i]+2*h,l_up_y[i]]=1
        # book[i,0,l_up_x[i]:l_up_x[i]+2*h,l_up_y[i]+2*h]=1

        book[i, 0, l_up_y[i], l_up_x[i]:l_up_x[i] + 2 * h] = 1
        book[i, 0, l_up_y[i] + 2 * h, l_up_x[i]:l_up_x[i] + 2 * h] = 1
        book[i, 0, l_up_y[i]:l_up_y[i] + 2 * h, l_up_x[i]] = 1
        book[i, 0, l_up_y[i]:l_up_y[i] + 2 * h, l_up_x[i] + 2 * h] = 1

    soft_label=book*det_p
    book1=book1-soft_label
    count=count+np.sum(soft_label)

    return soft_label,book1,count
def softlabel(pred,label,epoch):
    #thr=0.8
    #label情况：1:前景，0:背景，255不知道
    # thr:Conservative or not !!!
    P_soft_fg=(pred.cpu().detach().numpy()>0.95).astype(int)#the thr is a hy paramater !!!
    fg=(label.cpu().numpy()==1).astype(int)
    N,_,W,H=label.size()
    # P_soft_bg=pred<0.1
    center_y=np.zeros(N)
    center_x=np.zeros(N)
    for i in range(N):
        b_f=fg[i,:,:,:]
        P = b_f.argmax()
        center_y[i] = P // H
        center_x[i] = P % H



    #center_x,center_y=center_x+5,center_y+5
    #N is mast be change to correct the result !!!
    N=300 #f(epoch,size?)
    count=0
    soft_label=label
    #or cu_label is 11x11!
    for l in range(6,15):
        soft_label_l,mut_p,count=flood(fg,P_soft_fg,center_x,center_y,l,count)
        if count<N:
            #soft_label has 255!!! 0,1,255;and all most of is 255 !!!

            soft_label=soft_label.cpu()*mut_p+soft_label_l
        else:
            break
    return soft_label
def get_transform(ops=[0,1,2]):

    op = np.random.choice(ops)
    if op==0:
        flip = np.random.randint(0, 2)
        pp = Flip(flip)
    elif op==1:
        # pp = Translate(0.3)
        pp = Translate(0.15)
    elif op==2:
        pp = Crop(0.7, 0.7)
    return pp
def get_transform_sm(ops=[0,1,2,3]):
    '''One of flip, translate, crop'''
    op = np.random.choice(ops)
    if op==0:
        pp=GaussianBlur(5)
        return pp

def get_featuremap(h, x):
    w = h.weight
    b = h.bias
    c = w.shape[1]
    c1 = F.conv2d(x, w.transpose(0,1), padding=(1,1), groups=c)
    return c1, b

def unsymmetric_grad(x, y, calc, w1, w2):
    '''
    x: strong feature
    y: weak feature'''
    return calc(x, y.detach())*w1 + calc(x.detach(), y)*w2

def train_loss(image, mask, x_h,y_w,mask_gt,net, ctx, ft_dct, w_ft=.1, ft_st = 2, ft_fct=.5, ft_head=True, mtrsf_prob=1, ops=[0,1,2], w_l2g=0, l_me=0.1, me_st=50, me_all=False, multi_sc=0, l=0.3, sl=1):

    if ctx:
        epoch = ctx['epoch']
        global_step = ctx['global_step']
        sw = ctx['sw']
        t_epo = ctx['t_epo']

    do_moretrsf = np.random.uniform()<mtrsf_prob#1
    if do_moretrsf:
        pre_transform_m=get_transform_sm()
        if pre_transform_m==None:
            pre_transform = get_transform()
            image_tr=pre_transform(image)
        else:
            image_tr=pre_transform_m(image,mask)
        # for pp in pre_transform:
        #     image_tr=pp(image_tr)
        #image_tr = pre_transform(image)# make image -> image'
        large_scale = True
    else:
        large_scale = np.random.uniform() < multi_sc#0
        image_tr = image
    sc_fct = 0.5 if large_scale else 0.3
    image_scale = F.interpolate(image_tr, scale_factor=sc_fct, mode='bilinear', align_corners=True)

    #image_sm =
    #out2, loss_c , out3, out4, out5, out6, hook0 ,fg ,bg = net(image, )
    out2, loss_c, out3, out4, out5, out6, hook0 = net(image, )

    #get_featuremap()
    # out2_org = out2

    #hh.remove()
    #out2_s, _, out3_s, out4_s, out5_s, out6_s,_ ,fg_s,bg_s= net(image_scale, )
    out2_s, _, out3_s, out4_s, out5_s, out6_s, _ = net(image_scale, )
    #out2_sm,_,out3_sm,out4_sm,out5_sm,out6_sm,_=net(image_sm,)


    ### Calc intra_consisten loss (l2 norm) / entorpy
    loss_intra = []
    #me_st too large >50 epoch
    if epoch>=me_st:
        def entrp(t):
            etp = -(F.softmax (t, dim=1) * F.log_softmax (t, dim=1)).sum(dim=1)
            msk = (etp<0.5)
            return (etp*msk).sum() / (msk.sum() or 1)
        me = lambda x: entrp(torch.cat((x*0, x), 1)) # orig: 1-x, x
        if not me_all:
            e = me(out2)
            loss_intra.append(e * get_current_consistency_weight(epoch-me_st, consistency=l_me, consistency_rampup=t_epo-me_st))
            loss_intra = loss_intra + [0,0,0,0]
            sw.add_scalar('intra entropy', e.item(), global_step)
        elif me_all:
            ga = get_current_consistency_weight(epoch-me_st, consistency=l_me, consistency_rampup=t_epo-me_st)
            for i in [out2, out3, out4, out5, out6]:
                loss_intra.append(me(i)*ga)
            sw.add_scalar('intra entropy', loss_intra[0].item(), global_step)
    else:
        loss_intra.extend([0 for _ in range(5)])

    # def out_proc(out2, out3, out4, out5, out6,fg,bg):
    #     a = [out2, out3, out4, out5, out6,fg,bg]

    def out_proc(out2, out3, out4, out5, out6):
        a = [out2, out3, out4, out5, out6]
        a = [i.sigmoid() for i in a]
        a = [torch.cat((1 - i, i), 1) for i in a]
        return a
    #sigmoid
    out2, out3, out4, out5, out6 = out_proc(out2, out3, out4, out5, out6) #init

    #out2, out3, out4, out5, out6,fg,bg = out_proc(out2, out3, out4, out5, out6,fg,bg)
    # the size of out_s is be transformered
    #out2_s, out3_s, out4_s, out5_s, out6_s,fg_s,bg_s = out_proc(out2_s, out3_s, out4_s, out5_s, out6_s,fg_s,bg_s)
    out2_s, out3_s, out4_s, out5_s, out6_s = out_proc(out2_s, out3_s, out4_s, out5_s, out6_s)
    #out2_sm, out3_sm, out4_sm, out5_sm, out6_sm = out_proc(out2_sm, out3_sm, out4_sm, out5_sm, out6_sm)

    #loss_sm = (SaliencyStructureConsistency(out2_sm.detach(), out2, 0.85) * (w_l2g + 1))
    if not do_moretrsf:
        out2_scale = F.interpolate(out2[:, 1:2], scale_factor=sc_fct, mode='bilinear', align_corners=True)
        #out2_s = out2_s[:, 1:2]
        # out2_s = F.interpolate(out2_s[:, 1:2], scale_factor=0.3/sc_fct, mode='bilinear', align_corners=True)
    else:
        #which is gudingde:
        # out2_ss=pre_transform[0](out2)
        # out2_ss=pre_transform[1](out2_ss)
        # out2_ss=pre_transform[2](out2_ss)
        if pre_transform_m==None:
            out2_ss = pre_transform(out2)
        else:
            out2_ss = out2 #7.15:only mask also can transform to out2
        out2_scale = F.interpolate(out2_ss[:, 1:2], scale_factor=1.0, mode='bilinear', align_corners=True)
        out2_s = F.interpolate(out2_s[:, 1:2], scale_factor=1.0/sc_fct, mode='bilinear', align_corners=True)

    loss_ssc = Self_Knowledge_distilation(out2_s, out2_scale.detach(), mask,x_h,y_w)


    ######   label for partial cross-entropy loss  ######
    gt = mask.squeeze(1).long()

    gt_s_gt=mask_gt.squeeze(1).long()

    #mask has
    bg_label = gt.clone()
    fg_label = gt.clone()

    bg_label_s_gt = gt_s_gt.clone()
    fg_label_s_gt = gt_s_gt.clone()
    #1=foreground 2=background 0=unknown_pixels,

    # if epoch>10:
    #     gt = softlabel(out2, mask, epoch)
    #     gt = gt.squeeze(1).long()
    #0=bg ,1=fg ,255=unknown
    bg_label[gt != 0] = 255
    fg_label[gt == 0] = 255
    """"""
    Gt = gt.clone()#0,1
    Gt=Gt+1

    Gt_p=gt_s_gt.clone()
    Gt_p[Gt_p==255]=-1

    Gt_p=Gt_p*Gt

    Gt_p=Gt_p-1
    Gt_p[Gt_p<0]=255
    """"""
    bg_label_s_gt = Gt_p.clone()
    fg_label_s_gt = Gt_p.clone()

    """"""
    bg_label_s_gt[Gt_p != 0] = 255
    fg_label_s_gt[Gt_p == 0] = 255

    ######   local saliency coherence loss (scale to realize large batchsize)  ######
    image_ = F.interpolate(image, scale_factor=0.25, mode='bilinear', align_corners=True)
    sample = {'rgb': image_}

    out2_ = F.interpolate(out2[:, 1:2], scale_factor=0.25, mode='bilinear', align_corners=True)
    loss2_lsc = loss_lsc(out2_, loss_lsc_kernels_desc_defaults, loss_lsc_radius, sample, image_.shape[2], image_.shape[3])['loss']

    loss2 =loss_ssc+(criterion(out2, fg_label) + criterion(out2, bg_label))+0.5*(criterion(out2, fg_label_s_gt) + criterion(out2, bg_label_s_gt)) + l * loss2_lsc + loss_intra[0]


    return loss2,  loss2*0.0, loss2*0.0, loss2*0.0,loss2*0.0