import torch.nn.functional as F
import torch
from feature_loss import *
from tools import *
from utils import ramps
import numpy as np
device=torch.device("cuda:2")
criterion = torch.nn.CrossEntropyLoss(weight=None, ignore_index=255, reduction='mean').to(device)#.cuda()
loss_lsc = FeatureLoss().to(device)#.cuda()
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
    '''One of flip, translate, crop'''
    op = np.random.choice(ops)
    if op==0:
        flip = np.random.randint(0, 2)
        pp = Flip(flip)
    elif op==1:
        # pp = Translate(0.3)
        pp = Translate(0.15)
    elif op==2:
        #pp = Crop(0.7, 0.7)
        pp = Crop(0.7, 0.7)
    return pp
    #pp=Translate(0.15)
    #return pp
def get_color_tranform(ops=[0,1,2,3,4,5]):
    op=np.random.choice(ops)
    if op==3:
        pp=GaussianBlur(5)
        return pp
    if op==4:
        pp=mask()
        return pp
    if op==5:
        pp=Color_jitter()
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

def train_loss(image, mask, net, ctx, ft_dct, w_ft=.1, ft_st = 2, ft_fct=.5, ft_head=True, mtrsf_prob=1, ops=[0,1,2], w_l2g=0, l_me=0.1, me_st=50, me_all=False, multi_sc=0, l=0.3, sl=1):
    #print(mask.shape) #[16,1,320,320]+torch.tensor
    #image:tensor+[16,3,320,320]


    if ctx:
        epoch = ctx['epoch']
        global_step = ctx['global_step']
        sw = ctx['sw']
        t_epo = ctx['t_epo']


    ######  saliency structure consistency loss  ######

    do_moretrsf = np.random.uniform()<mtrsf_prob
    if do_moretrsf:
        pre_color_transform=get_color_tranform()
        if pre_color_transform ==None:
            pre_transform = get_transform(ops)
            image_tr=pre_transform(image)
            #image_tr=image
        else:
            #image_tr=image

            image_tr=pre_color_transform(image,mask)
        #image_tr = pre_transform(image)
        large_scale = True
    else:
        large_scale = np.random.uniform() < multi_sc
        image_tr = image
    sc_fct = 0.5 if large_scale else 0.3
    image_scale = F.interpolate(image_tr, scale_factor=sc_fct, mode='bilinear', align_corners=True)

    out2, _, out3, out4, out5, out6, hook0 = net(image, )

    out2_s, _, out3_s, out4_s, out5_s, out6_s, _ = net(image_scale, )
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
    if not do_moretrsf:
        out2_scale = F.interpolate(out2[:, 1:2], scale_factor=sc_fct, mode='bilinear', align_corners=True)
        out2_s = out2_s[:, 1:2]
        # out2_s = F.interpolate(out2_s[:, 1:2], scale_factor=0.3/sc_fct, mode='bilinear', align_corners=True)
    else:
        #out2_ss=out2
        if pre_color_transform==None:
            out2_ss = pre_transform(out2)
            #out2_ss=out2
        else:
            out2_ss=out2
        out2_scale = F.interpolate(out2_ss[:, 1:2], scale_factor=1.0, mode='bilinear', align_corners=True)
        out2_s = F.interpolate(out2_s[:, 1:2], scale_factor=1.0/sc_fct, mode='bilinear', align_corners=True)
    # wen ding xing f
    loss_ssc = Contrastive_loss(out2_s, out2_scale.detach())

    gt = mask.squeeze(1).long()
    #mask has
    bg_label = gt.clone()
    fg_label = gt.clone()

    bg_label[gt != 0] = 255
    fg_label[gt == 0] = 255

    image_ = F.interpolate(image, scale_factor=0.25, mode='bilinear', align_corners=True)
    sample = {'rgb': image_}
    # print('sample :', image_.max(), image_.min(), image_.std())
    #out2_ = F.interpolate(out2[:, 0:1], scale_factor=0.25, mode='bilinear', align_corners=True)
    out2_ = F.interpolate(out2[:, 1:2], scale_factor=0.25, mode='bilinear', align_corners=True)
    loss2_lsc = loss_lsc(out2_, loss_lsc_kernels_desc_defaults, loss_lsc_radius, sample, image_.shape[2], image_.shape[3])['loss']


    loss2 =loss_ssc+ (criterion(out2, fg_label) + criterion(out2, bg_label)) + l * loss2_lsc + loss_intra[0]

    return loss2, loss2*0.0,loss2*0.0, loss2*0.0, loss2*0.0