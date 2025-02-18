import torch.nn as nn
import torch
import os
import BpOps
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict
from torch.autograd import Function
from torch.cuda.amp import custom_fwd, custom_bwd

class BpDist(Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, xy, idx, Valid, num, H, W):
        """
        """
        assert xy.is_contiguous()
        assert Valid.is_contiguous()
        _, Cc, M = xy.shape     # (1, 2, -1)
        B = Valid.shape[0]      # (B, 1, N')
        N = H * W
        args = torch.zeros((B, num, N), dtype=torch.long, device=xy.device)
        IPCnum = torch.zeros((B, Cc, num, N), dtype=xy.dtype, device=xy.device)     # 应该是临近点的坐标偏移量
        for b in range(B):
            Pc = torch.masked_select(xy, Valid[b:b + 1].view(1, 1, N)).reshape(1, 2, -1)    # 选择valid处的坐标
            BpOps.Dist(Pc, IPCnum[b:b + 1], args[b:b + 1], H, W)
            idx_valid = torch.masked_select(idx, Valid[b:b + 1].view(1, 1, N))              # 选择valid处的坐标索引 [1, 4, 11, 18, 25]
            args[b:b + 1] = torch.index_select(idx_valid, 0, args[b:b + 1].reshape(-1)).reshape(1, num, N)      # args应该是四个临近点的排序索引（从0~N的排序号）
        return IPCnum, args

    @staticmethod
    @custom_bwd
    def backward(ctx, ga=None, gb=None):
        return None, None, None, None

bpdist = BpDist.apply
    

def _make_fusion_block(features, use_bn, size=None, use_res1=True):
    return FeatureFusionDepthBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
        use_res1=use_res1
    )


def _make_scratch(in_shape, out_shape, groups=1, expand=False):
    scratch = nn.Module()

    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    if len(in_shape) >= 4:
        out_shape4 = out_shape

    if expand:
        out_shape1 = out_shape
        out_shape2 = out_shape*2
        out_shape3 = out_shape*4
        if len(in_shape) >= 4:
            out_shape4 = out_shape*8

    scratch.layer1_rn = nn.Conv2d(
        in_shape[0], out_shape1, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer2_rn = nn.Conv2d(
        in_shape[1], out_shape2, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer3_rn = nn.Conv2d(
        in_shape[2], out_shape3, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    if len(in_shape) >= 4:
        scratch.layer4_rn = nn.Conv2d(
            in_shape[3], out_shape4, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
        )

    return scratch


class ResidualConvUnit(nn.Module):
    """Residual convolution module.
    """

    def __init__(self, features, activation, bn):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.bn = bn

        self.groups = 1

        

        if self.bn == True:
            self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=False, groups=self.groups
            )

            self.conv2 = nn.Conv2d(
                features, features, kernel_size=3, stride=1, padding=1, bias=False, groups=self.groups
            )
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)
        
        else:
            self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups
            )

            self.conv2 = nn.Conv2d(
                features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups
            )

        self.activation = activation

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """

        out = self.activation(x)
        out = self.conv1(out)
        if self.bn == True:
            out = self.bn1(out)

        out = self.activation(out)
        out = self.conv2(out)
        if self.bn == True:
            out = self.bn2(out)

        if self.groups > 1:
            out = self.conv_merge(out)

        return self.skip_add.add(out, x)


class FeatureFusionBlock(nn.Module):
    """Feature fusion block.
    """

    def __init__(self, features, activation, deconv=False, bn=False, expand=False, align_corners=True, size=None):
        """Init.

        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners

        self.groups = 1

        self.expand = expand
        out_features = features
        if self.expand == True:
            out_features = features//2

        self.out_conv = nn.Conv2d(
            features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=1)

        self.resConfUnit1 = ResidualConvUnit(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn)

        self.skip_add = nn.quantized.FloatFunctional()

        self.size = size

    def forward(self, *xs, size=None):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)

        output = self.resConfUnit2(output)

        if (size is None) and (self.size is None):
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}

        output = nn.functional.interpolate(
            output, **modifier, mode="bilinear", align_corners=self.align_corners
        )

        output = self.out_conv(output)

        return output


class FeatureFusionControlBlock(FeatureFusionBlock):
    """Feature fusion block.
    """

    def __init__(self, features, activation, deconv=False, bn=False, expand=False, align_corners=True, size=None):
        """Init.

        Args:
            features (int): number of features
        """
        super.__init__(features, activation, deconv,
                       bn, expand, align_corners, size)
        self.copy_block = FeatureFusionBlock(
            features, activation, deconv, bn, expand, align_corners, size)

    def forward(self, *xs, size=None):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)

        output = self.resConfUnit2(output)

        if (size is None) and (self.size is None):
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}

        output = nn.functional.interpolate(
            output, **modifier, mode="bilinear", align_corners=self.align_corners
        )

        output = self.out_conv(output)

        return output


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class FeatureFusionDepthBlock(nn.Module):
    """Feature fusion block.
    """

    def __init__(self, features, activation, deconv=False, bn=False, expand=False, align_corners=True, size=None, use_res1=True):
        """Init.

        Args:
            features (int): number of features
        """
        super(FeatureFusionDepthBlock, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners

        self.groups = 1

        self.expand = expand
        out_features = features
        if self.expand == True:
            out_features = features//2

        self.out_conv = nn.Conv2d(
            features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=1)
        
        if use_res1:
            self.resConfUnit1 = ResidualConvUnit(features, activation, bn)
        else:
            self.resConfUnit1 = None
        
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn)
        self.resConfUnit_depth = nn.Sequential(
            nn.Conv2d(2, features, kernel_size=3, stride=1,
                      padding=1, bias=True, groups=1),
            # nn.BatchNorm2d(features),
            activation,
            nn.Conv2d(features, features, kernel_size=3,
                      stride=1, padding=1, bias=True, groups=1),
            # nn.BatchNorm2d(features),
            activation,
            zero_module(
                nn.Conv2d(features, features, kernel_size=3,
                          stride=1, padding=1, bias=True, groups=1)
            )
        )
        self.skip_add = nn.quantized.FloatFunctional()
        self.size = size

    def forward(self, *xs, prompt_depth=None, size=None):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)

        output = self.resConfUnit2(output)

        if prompt_depth is not None:
            prompt_depth = F.interpolate(
                prompt_depth, output.shape[2:], mode='bilinear', align_corners=False)
            res = self.resConfUnit_depth(prompt_depth)
            output = self.skip_add.add(output, res)

        if (size is None) and (self.size is None):
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}

        output = nn.functional.interpolate(
            output, **modifier, mode="bilinear", align_corners=self.align_corners
        )

        output = self.out_conv(output)

        return output

def Conv1x1(in_planes, out_planes, stride=1, bias=False, groups=1, dilation=1, padding_mode='zeros'):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)


def Conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, padding_mode='zeros', bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, padding_mode=padding_mode, groups=groups, bias=bias, dilation=dilation)

class Basic2d(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=None, kernel_size=3, padding=1, padding_mode='zeros',
                 act=nn.ReLU, stride=1):
        super().__init__()
        if norm_layer:
            conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                             stride=stride, padding=padding, bias=False, padding_mode=padding_mode)
        else:
            conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                             stride=stride, padding=padding, bias=True, padding_mode=padding_mode)
        self.conv = nn.Sequential(OrderedDict([('conv', conv)]))
        if norm_layer:
            self.conv.add_module('bn', norm_layer(out_channels))
        self.conv.add_module('relu', act())

    def forward(self, x):
        out = self.conv(x)
        return out

class Basic2dTrans(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=None, act=nn.ReLU, kernel_size=4, stride=2, padding=1):
        super().__init__()
        if norm_layer is None:
            bias = True
            norm_layer = nn.Identity
        else:
            bias = False
        self.conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                       stride=stride, padding=padding, bias=bias)
        self.bn = norm_layer(out_channels)
        self.relu = act()

    def forward(self, x):
        out = self.conv(x.contiguous())
        out = self.bn(out)
        out = self.relu(out)
        return out

class Permute(nn.Module):
    def __init__(self, level, in_channels, out_channels=128, norm_layer=nn.BatchNorm2d, act=nn.ReLU):
        super().__init__()

        self.conv = nn.Sequential(
            # Basic2dTrans(in_channels=in_channels, out_channels=in_channels, norm_layer=norm_layer, act=act),    # (B, C, 74, 246)
            # Basic2dTrans(in_channels=in_channels, out_channels=in_channels, norm_layer=norm_layer, act=act),    # (B, C, 148, 492)
            # Basic2dTrans(in_channels=in_channels, out_channels=in_channels, norm_layer=norm_layer, act=act),    # (B, C, 296, 984)
            # Conv1x1(in_channels, out_channels, bias=True)      # (B, 128, 296, 984)
            Basic2d(in_channels=in_channels, out_channels=out_channels, norm_layer=norm_layer)
        )


    def forward(self, x):
        """
        """
        fout = self.conv(x)
        return fout


class Coef(nn.Module):
    """
    """
    def __init__(self, in_channels, out_channels=3, kernel_size=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=1, padding=padding, bias=True)

    def forward(self, x):
        feat = self.conv(x)
        XF, XB, XW = torch.split(feat, [1, 1, 1], dim=1)
        return XF, XB, XW
    

class Dist(nn.Module):
    """
    """

    def __init__(self, num):
        super().__init__()
        """
        """
        self.num = num

    def forward(self, S, xx, yy):
        """
        """
        num = self.num
        B, _, height, width = S.shape
        N = height * width
        S = S.reshape(B, 1, N)
        Valid = (S > 1e-3)          # (B, 1, N)
        xy = torch.stack((xx, yy), axis=0).reshape(1, 2, -1).float()    # (1, 2, n) 一行x，一行y 所有像素坐标
        idx = torch.arange(N, device=S.device).reshape(1, 1, N)         # (1, 1, n) 所有像素坐标排序
        Ofnum, args = bpdist(xy, idx, Valid, num, height, width)    # (B, 2, 4, N) (B, 4, N)
        return Ofnum, args


class Prop(nn.Module):
    """
    dout = self.prop(fout, Pxyz, Ofnum, args)
    """

    def __init__(self, Cfi, Cfp=1, Cfo=2, act=nn.GELU, norm_layer=nn.BatchNorm2d):
        super().__init__()
        """
        """
        self.dist = lambda x: (x * x).sum(1)
        Ct = Cfo + Cfi + Cfi + Cfp
        self.convXF = nn.Sequential(
            Basic2d(in_channels=Ct, out_channels=Cfi, norm_layer=norm_layer, act=act, kernel_size=1,
                    padding=0),
            Basic2d(in_channels=Cfi, out_channels=Cfi, norm_layer=norm_layer, act=act, kernel_size=1,
                    padding=0),
        )
        self.convXL = nn.Sequential(
            Basic2d(in_channels=Cfi, out_channels=Cfi, norm_layer=norm_layer, act=act, kernel_size=1,
                    padding=0),
            Basic2d(in_channels=Cfi, out_channels=Cfi, norm_layer=norm_layer, act=nn.Identity, kernel_size=1,
                    padding=0),
        )
        self.act = act()
        self.coef = Coef(Cfi, 3)

    def forward(self, If, Pf, Ofnum, args):
        """
        If: fout
        Pf: Pxyz
        Ofnum: (B, Cc, num, N) 偏移距离
        args: (B, num, N) 临近点索引(0~N)
        """
        num = args.shape[-2]        # args应该是四个临近点的排序索引(B, num, N)
        B, Cfi, H, W = If.shape     # fout
        N = H * W
        B, Cfp, Hp, Wp = Pf.shape        # Pxyz 深度
        M = Hp * Wp
        If = If.view(B, Cfi, 1, N)
        Pf = Pf.view(B, Cfp, 1, M)
        Ifnum = If.expand(B, Cfi, num, N)  ## Ifnum is (B x Cfi x num x N) fout扩展维度
        IPfnum = torch.gather(
            input=If.expand(B, Cfi, num, N),
            dim=-1,
            index=args.view(B, 1, num, N).expand(B, Cfi, num, N))  ## IPfnum is (B x Cfi x num x N) 目标点的图像编码
        Pfnum = torch.gather(
            input=Pf.expand(B, Cfp, num, M),
            dim=-1,
            index=args.view(B, 1, num, N).expand(B, Cfp, num, N))  ## Pfnum is (B x Cfp x num x N) 目标点的深度编码
        X = torch.cat([Ifnum, IPfnum, Pfnum, Ofnum], dim=1)     # Cfi + Cfi + 3 + 2  (B, C, 4, N)
        XF = self.convXF(X) 
        XF = self.act(XF + self.convXL(XF))     # (B, Cfi, 4, N)
        Alpha, Beta, Omega = self.coef(XF)      # (B, 1, 4, N)
        Omega = torch.softmax(Omega, dim=2)
        dout = torch.sum(((Alpha + 1) * Pfnum[:, -1:] + Beta) * Omega, dim=2, keepdim=True)     # (B, 1, 4, N) ---> (B, 1, 1, N)
        return dout.view(B, 1, H, W)
    
    
class Prefill(nn.Module):
    def __init__(self, in_ch, out_ch, level, drift=1e6):
        super().__init__()
        self.level = level
        self.drift = drift
        self.permute = Permute(level=level, in_channels=in_ch, out_channels=out_ch)
        self.dist = Dist(num=4)
        self.prop = Prop(out_ch)

    def forward(self, Sp, fout):
        """
        Sp:     (B, 1, 256, 1216)
        fout:   (B, Ci, 37, 123)
        """
        W = self.permute(fout)  # (B, ci, 296, 1400)  *8倍
        fi = F.interpolate(W, Sp.shape[2:], mode="bilinear", align_corners=True)     # (B, ci, H, W)

        B, _, height, width = Sp.shape
        xx, yy = torch.meshgrid(torch.arange(width, device=Sp.device), torch.arange(height, device=Sp.device), indexing='xy')   # xx是列坐标(向右为正)，yy是行坐标(向下为正)
        Ofnum, args = self.dist(Sp, xx, yy) # (B, 2, 4, N) (B, 4, N)
        dout = self.prop(fi, Sp, Ofnum, args)

        return dout

class Prefill_nearest(nn.Module):
    def __init__(self, num=4):
        super().__init__()
        self.num = num
        self.dist = Dist(num=num)

    def forward(self, Sp):
        """
        Sp:     (B, 1, 256, 1216)
        """
        B, _, height, width = Sp.shape
        xx, yy = torch.meshgrid(torch.arange(width, device=Sp.device), torch.arange(height, device=Sp.device), indexing='xy')   # xx是列坐标(向右为正)，yy是行坐标(向下为正)
        Ofnum, args = self.dist(Sp, xx, yy) # (B, 2, 4, N) (B, 4, N)

        ss = Sp.view(B, 1, 1, -1).repeat(1, 1, self.num, 1)  # (B, 1, 1, N) --> (B, 1, 4, N)
        dout = torch.gather(ss, dim=-1, index=args.view(B, 1, self.num, -1))   # (B, 1 ,4, N)
        dout = torch.mean(dout, dim=-2) # (B, 1  N)
        dout = dout.view(B, _ , height, width)

        return dout
