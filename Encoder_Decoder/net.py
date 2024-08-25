import torch
import torch.nn as nn
import torch.nn.functional as F
from pvtv2 import pvt_v2_b2,pvt_v2_b1,pvt_v2_b3,pvt_v2_b4
from timm.models.layers import DropPath, trunc_normal_
device = torch.device("cuda:2")
criterion=nn.CosineSimilarity(dim=1).to(device)
def weight_init(module):

    for n, m in module.named_children():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m,nn.Linear):#which can be replaced as kaiming_normal_()
            trunc_normal_(m.weight,std=.02)
            if isinstance(m,nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias,0)
        elif isinstance(m,nn.Dropout) or isinstance(m,DropPath):
             m.p=0.00
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm1d):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, nn.ReLU) or isinstance(m, nn.GELU) or isinstance(m, nn.LeakyReLU) or isinstance(m,
                                                                                                           nn.AdaptiveAvgPool2d) or isinstance(
                m, nn.ReLU6) or isinstance(m, nn.MaxPool2d) or isinstance(m, nn.Softmax):
            pass
        elif isinstance(m, nn.ModuleList):
            weight_init(m)
        elif isinstance(m,nn.Upsample):
            pass
        elif isinstance(m,nn.AvgPool2d):
            pass
        # elif isinstance(m,nn.ConvTranspose2d):
        #     nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        else:
            m.initialize()


def conv3x3(in_planes, out_planes, stride=1, padding=1, dilation=1, bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, dilation=dilation, bias=bias)

def conv1x1(in_planes, out_planes, stride=1, bias=False):
    "1x1 convolution"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=bias)

class BasicConv2d(nn.Module):
    def __init__(self,in_planes,out_planes,kernel_size,stride=1,padding=0,dilation=1):
        super(BasicConv2d,self).__init__()
        self.conv=nn.Conv2d(in_planes,out_planes,kernel_size=kernel_size,
                            stride=stride,padding=padding,dilation=dilation,bias=False)
        self.bn=nn.BatchNorm2d(out_planes)
    def forward(self,x):
        x=self.conv(x)
        x=self.bn(x)
        return x
    def initialize(self):
        weight_init(self)
############################################# Pooling ##############################################
class basicConv(nn.Module):
    def __init__(self, in_channel, out_channel, k=3, s=1, p=1, g=1, d=1, bias=False, bn=True, relu=True):
        super(basicConv, self).__init__()
        conv = [nn.Conv2d(in_channel, out_channel, k, s, p, dilation=d, groups=g, bias=bias)]
        if bn:
            conv.append(nn.BatchNorm2d(out_channel))
            # conv.append(nn.LayerNorm(out_channel, eps=1e-6))
        if relu:
            conv.append(nn.GELU())
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)

    def initialize(self):
        weight_init(self)

########################################### CoordAttention #########################################
# Revised from: Coordinate Attention for Efficient Mobile Network Design, CVPR21
# https://github.com/houqb/CoordAttention
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

    def initialize(self):
        weight_init(self)


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

    def initialize(self):
        weight_init(self)

BatchNorm2d=nn.BatchNorm2d
BN_MOMENTUM=0.1
class Bottleneck(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        #because next is seted to BN ,so bias=False,
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = BatchNorm2d(planes * self.expansion,
                               momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        if planes==64:
            self.downsample = nn.Sequential(nn.Conv2d(inplanes, 64, kernel_size=1, stride=stride, bias=False),
                                   nn.BatchNorm2d(64))
        elif planes==128:
            self.downsample = nn.Sequential(nn.Conv2d(inplanes, 128, kernel_size=1, stride=stride, bias=False),
                                            nn.BatchNorm2d(128))
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        #if self.downsample is not None:
        residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    def initialize(self):
        weight_init(self)
#################################################Attention##########################################
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    def initialize(self):
        weight_init(self)
class S_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., vis=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mae=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn=(q @ k.transpose(-2,-1) ) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        weight=attn

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        # if mae:
        #     return x
        # else:
        #     return x,weight
        return x
    def initialize(self):
        weight_init(self)
class special_layer(nn.Module):
    def __init__(self,dim,num_heads,mlp_ratio=4.,qkv_bias=False,qk_scale=None,drop=0.,attn_drop=0.,
                 drop_path=0.,act_layer=nn.GELU,norm_layer=nn.LayerNorm,vis=False):
        super(special_layer, self).__init__()
        self.norm1=norm_layer(dim)
        self.attn=S_Attention(dim,num_heads=num_heads,qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, vis=vis)
        self.drop_path = DropPath(drop_path) #if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
    def forward(self,x,mae=False):

        # if mae:
        #     o = self.attn(self.norm1(x),mae)
        # else:
        #     o, weights = self.attn(self.norm1(x),mae)
        o = self.attn(self.norm1(x))
        x = x + self.drop_path(o)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        # if mae:
        #     return x
        # else:
        #     return x,weights
        return x
    def initialize(self):
        weight_init(self)

################################################ Net ###############################################
class EF(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(EF,self).__init__()
        self.relu=nn.ReLU(True)
        #self.conv_res = nn.Sequential(BasicConv2d(in_channel,out_channel,1),)#BN????
        #self.conv_front = nn.Sequential(BasicConv2d(in_channel,out_channel,1),)
        #self.conv_m3x3  = conv3x3(in_channel,out_channel)
        self.conv_m3x3=nn.Sequential(BasicConv2d(in_channel,out_channel,3,padding=1),)
        # self.conv_m5x5  =  nn.Sequential(BasicConv2d(in_channel,out_channel,5,padding=2),)
        # self.conv_m7x7  = nn.Sequential(BasicConv2d(in_channel,out_channel,7,padding=3),)
        # self.conv_final =  nn.Sequential(BasicConv2d(3*out_channel,out_channel,1),)
    def forward(self,x):
        # x_r=self.conv_res(x)
        # #x1=self.conv_front(x)
        # x_3x3=self.conv_m3x3(x)
        # x_5x5=self.conv_m5x5(x)
        # x_7x7=self.conv_m7x7(x)
        # out=self.conv_final(torch.cat([x_3x3,x_5x5,x_7x7],dim=1))
        # out=self.relu(out+x_r)
        out=self.relu(self.conv_m3x3(x))
        return out
    def initialize(self):
        weight_init(self)

from functools import partial
class Net(nn.Module):
    def __init__(self, cfg):
        super(Net, self).__init__()
        self.cfg = cfg
        self.bkbone = pvt_v2_b4()
        #load_path='./pvt_v2_b2.pth'
        load_path='./pvt_v2_b4.pth'
        pretrained_dict=torch.load(load_path)
        pretrained_dict={k:v for k,v in pretrained_dict.items() if k in self.bkbone.state_dict()}
        self.bkbone.load_state_dict(pretrained_dict)
        """by_chf:512?"""
        # self.extra=nn.ModuleList([
        #     BasicConv2d(64,64,3,padding=1),
        #     BasicConv2d(128,64,3,padding=1),
        #     BasicConv2d(320,64,3,padding=1),
        #     BasicConv2d(512,64,3,padding=1),
        # ])
        self.extra = nn.ModuleList([
            # EF(64, 128),
            # EF(128, 128),
           # EF(320, 128),
            # EF(512, 128),
            # BasicConv2d(64, 64, 3,padding=1),
            # BasicConv2d(128, 64, 3,padding=1),
            # BasicConv2d(320, 64, 3,padding=1),
            #BasicConv2d(512, 64, 3,padding=1),
            conv3x3(64,64),
            conv3x3(128,64),
            conv3x3(320,64),
            conv3x3(512,64),
            # BasicConv2d(64*4,64,3,padding=1),
            # nn.ReLU(True),
        ])
        # self.upsample = nn.ModuleList([
        #     nn.ConvTranspose2d(64,64,kernel_size=2,stride=2),
        #     nn.ConvTranspose2d(64,64,kernel_size=2,stride=2),
        #     nn.ConvTranspose2d(128,64,kernel_size=2,stride=2),
        # ])
        self.head = nn.ModuleList([
            conv3x3(64*4 , 1),
            # conv3x3(64, 1),
            # conv3x3(64, 1),
            # conv3x3(64, 1),
            # conv3x3(64, 1),
        ])
        """*****************"""
        self.relu = nn.ReLU(True)
        self.norm = nn.LayerNorm(512)#norm_layer(embed_dim)  # 这个得检查一遍前面有没有norm这个东西了，这个东西可以当后面的norm用

        self.cts_bn=nn.BatchNorm2d(1)

        self.initialize()
    def upsample_add(self,x,y):
        _,_,H,W=y.size()
        return F.interpolate(x,size=(H,W),mode='bilinear',align_corners=True)+y

    def cos_sim(self,fg,bg):
        fg=F.normalize(fg,dim=1)
        bg=F.normalize(bg,dim=1)
        sim=torch.matmul(fg,bg.T)
        return torch.clamp(sim,min=0.0005,max=0.9995)

    def contrast(self,x,x0):
        N,C,W,H=x0.size()

        x_=x.reshape(N,1,H*W)
        x0=x0.reshape(N, C, H * W).permute(0, 2, 1).contiguous()
        fg_v=(torch.matmul(x_,x0)/(H*W)).reshape(N,C)
        bg_v=(torch.matmul(1-x_,x0)/(H*W)).reshape(N,C)
        #Cos=torch.nn.CosineSimilarity(dim=1,eps=1e-6)
        sim_bf=self.cos_sim(fg_v,bg_v)
        #
        # loss1=0
        #
        # for n in range(N):
        #     loss1=loss1+sim_bf[n,n]
        sim_bb=self.cos_sim(bg_v,bg_v)
        sim_ff=self.cos_sim(fg_v,fg_v)
        loss1=-torch.log(1-sim_bf)

        loss2=-torch.log(sim_bb)
        loss2[loss2<0]=0
        _,indices=sim_bb.sort(descending=True,dim=1)
        _,rank=indices.sort(dim=1)
        rank=rank-1
        rank_weights=torch.exp(-rank.float()*0.25)
        loss2=loss2*rank_weights

        #loss2[loss2<0]=0
        # _,indices=sim_bb.sort(descending=True,dim=1)
        # _,rank=indices.sort(dim=1)
        # rank=rank-1
        # rank_weights=torch.exp(-rank.float()*0.25)
        # loss2=loss2*rank_weights

        # loss3=-torch.log(sim_ff)
        # loss3[loss3 < 0] = 0
        # _, indices = sim_ff.sort(descending=True, dim=1)
        # _, rank = indices.sort(dim=1)
        # rank = rank - 1
        # rank_weights = torch.exp(-rank.float() * 0.25)
        # loss3 = loss3 * rank_weights
        loss=torch.mean(loss2)#+torch.mean(loss1) #+torch.mean(loss3)#+torch.mean(loss3)
        # embedded_fg=F.normalize(fg_v,dim=1)
        # embedded_bg=F.normalize(bg_v,dim=1)
        # sim=torch.matmul(embedded_fg,embedded_bg.T)
        # sim=torch.clamp(sim,min=0.0005,max=0.9995)
        #loss=-torch.log(1-sim)
        #loss=torch.mean(loss)
        #loss=Cos(fg_v,bg_v) # the loss is more complite
        return loss

    def forward(self, x, shape=None, epoch=None):

        shape = x.size()[2:] if shape is None else shape
        attn_map,bk_stage5,bk_stage4,bk_stage3,bk_stage2=self.bkbone(x)#bk_stage5=[8,512,6,6]
        #cls_token4,cls_token3,cls_token2,cls_token1,\
        #    attn_map,bk_stage5,bk_stage4,bk_stage3,bk_stage2=self.bkbone(x)#bk_stage5=[8,512,6,6]

        F1 = self.extra[0](bk_stage2)  # 3x3,c->64
        F2 = self.extra[1](bk_stage3)
        F3 = self.extra[2](bk_stage4)
        F4 = self.extra[3](bk_stage5)

        # F4 = self.extra[0](bk_stage2)  # 1/4
        # F3 = self.extra[1](bk_stage3)  # 1/8
        # F2 = self.extra[2](bk_stage4)  # 1/16
        # F1 = self.extra[3](bk_stage5)  # 1/32

        f_1 = F1
        f_2 = F.interpolate(F2, size=f_1.size()[2:], mode='bilinear', align_corners=True)
        f_3 = F.interpolate(F3, size=f_1.size()[2:], mode='bilinear', align_corners=True)
        f_4 = F.interpolate(F4, size=f_1.size()[2:], mode='bilinear', align_corners=True)


        """FPN:"""
        #try to predict use each feature

        """----"""
        feature_map=torch.cat([f_1,f_2,f_3,f_4],dim=1)
        #feature_map=self.extra[4](feature_map)
        # feature_map=self.extra[5](self.extra[4](feature_map))
        out0= feature_map
        cts=f_4#or=4
        """*************"""


        """instead of the hook in feature_loss"""
        hook = out0
        w = self.head[0].weight
        c = w.shape[1]
        c1 = F.conv2d(hook, w.transpose(0, 1), padding=(1, 1), groups=c)
        """*******"""

        """Contrast losses"""
        # for contrastive:test stage5 and fuse-feature
        #x_c = torch.sigmoid(self.cts_bn(self.head[0](out0)))
        #loss=self.contrast(x_c,cts)
        """"""
        out0 = F.interpolate(self.head[0](out0), size=shape, mode='bilinear', align_corners=False)
        #

        # [16,1,320,320]
        #out0=out0+(out1*out2+out3*out4)
        if self.cfg.mode == 'train':
            return out0, 0,out0,out0,out0,out0,c1#,fg,bg
        else:
            return out0,attn_map

    def initialize(self):
        print('initialize net')
        if self.cfg.snapshot:
            self.load_state_dict(torch.load(self.cfg.snapshot), strict=False)
        else:
            weight_init(self)