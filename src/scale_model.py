import torch
import torch.nn as nn
import torch.nn.functional as F
from blocks import _make_scratch, _make_fusion_block, Prefill, Prefill_nearest
from spn import BasicDepthEncoder, Post_process_deconv
from dcn_v2 import dcn_v2_conv


def Conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def Conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

class Basic2d(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=None, kernel_size=3, padding=1):
        super().__init__()
        if norm_layer:
            conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                             stride=1, padding=padding, bias=False)
        else:
            conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                             stride=1, padding=padding, bias=True)
        self.conv = nn.Sequential(conv, )
        if norm_layer:
            self.conv.add_module('bn', norm_layer(out_channels))
        self.conv.add_module('relu', nn.ReLU(inplace=True))

    def forward(self, x):
        out = self.conv(x)
        return out

class Basic2dTrans(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3,
                                       stride=2, padding=1, output_padding=1, bias=False)
        self.bn = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out

class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None, act=True):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = Conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = Conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.act = act

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        if self.act:
            out = self.relu(out)
        return out
    
class BasicDepthEncoder(nn.Module):

    def __init__(self, kernel_size, block=BasicBlock, bc=16, norm_layer=nn.BatchNorm2d):
        super(BasicDepthEncoder, self).__init__()
        self._norm_layer = norm_layer
        self.kernel_size = kernel_size
        self.num = kernel_size*kernel_size - 1
        self.idx_ref = self.num // 2

        self.convd1 = Basic2d(1, bc * 2, norm_layer=None, kernel_size=3, padding=1)
        self.convd2 = Basic2d(bc * 2, bc * 2, norm_layer=None, kernel_size=3, padding=1)

        self.convf1 = Basic2d(bc * 2, bc * 2, norm_layer=None, kernel_size=3, padding=1)
        self.convf2 = Basic2d(bc * 2, bc * 2, norm_layer=None, kernel_size=3, padding=1)

        self.conv = Basic2d(bc * 4, bc * 4, norm_layer=None, kernel_size=3, padding=1)
        self.ref = block(bc * 4, bc * 4, norm_layer=norm_layer, act=False)
        self.conv_weight = nn.Conv2d(bc * 4, self.kernel_size**2, kernel_size=1, stride=1, padding=0)
        self.conv_offset = nn.Conv2d(bc * 4, 2*(self.kernel_size**2 - 1), kernel_size=1, stride=1, padding=0)

    def forward(self, depth, context):
        B, _, H, W = depth.shape

        d1 = self.convd1(depth)
        d2 = self.convd2(d1)

        f1 = self.convf1(context)
        f2 = self.convf2(f1)

        input_feature = torch.cat((d2, f2), dim=1)
        input_feature = self.conv(input_feature)
        feature = self.ref(input_feature)
        weight = torch.sigmoid(self.conv_weight(feature))
        offset = self.conv_offset(feature)

        # Add zero reference offset
        offset = offset.view(B, self.num, 2, H, W)
        list_offset = list(torch.chunk(offset, self.num, dim=1))
        list_offset.insert(self.idx_ref,
                           torch.zeros((B, 1, 2, H, W)).type_as(offset))
        offset = torch.cat(list_offset, dim=1).view(B, -1, H, W)

        return weight, offset


class Post_process_deconv(nn.Module):

    def __init__(self, dkn_residual, kernel_size):
        super().__init__()

        self.dkn_residual = dkn_residual

        self.w = nn.Parameter(torch.ones((1, 1, kernel_size, kernel_size)))
        self.b = nn.Parameter(torch.zeros(1))
        self.stride = 1
        self.padding = int((kernel_size - 1) / 2)
        self.dilation = 1
        self.deformable_groups = 1
        self.im2col_step = 64

    def forward(self, depth, weight, offset):

        if self.dkn_residual:
            weight = weight - torch.mean(weight, 1).unsqueeze(1).expand_as(weight)
        else:
            weight = weight / torch.sum(weight, 1).unsqueeze(1).expand_as(weight)

        output = dcn_v2_conv(
            depth, offset, weight, self.w, self.b, self.stride, self.padding,
            self.dilation, self.deformable_groups)

        if self.dkn_residual:
            output = output + depth

        return output


class Guide(nn.Module):

    def __init__(self, c1, c2, norm_layer=None, use_dino = False):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if use_dino:
            self.conv = Basic2d(c1+c2+c2, c1, norm_layer)
        else:
            self.conv = Basic2d(c1+c2, c1, norm_layer)

    def forward(self, lidar, weight, img_dino=None):
        if img_dino is not None:
            img_dino = F.interpolate(
                    img_dino, lidar.shape[2:], mode='bilinear', align_corners=False)
            weight = torch.cat((lidar, weight, img_dino), dim=1)
        else:
            weight = torch.cat((lidar, weight), dim=1)
        weight = self.conv(weight)
        return weight
    

class StoDepth_BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, prob, m, multFlag, inplanes, planes, stride=1, downsample=None):
        super(StoDepth_BasicBlock, self).__init__()
        self.conv1 = Conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = Conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.prob = prob
        self.m = m
        self.multFlag = multFlag

    def forward(self, x):

        identity = x.clone()

        if self.training:
            # if torch.equal(self.m.sample(), torch.ones(1)):
            if True:
                self.conv1.weight.requires_grad = True
                self.conv2.weight.requires_grad = True

                out = self.conv1(x)
                out = self.bn1(out)
                out = self.relu(out)
                out = self.conv2(out)
                out = self.bn2(out)

                if self.downsample is not None:
                    identity = self.downsample(x)

                out += identity
            else:
                # Resnet does not use bias terms
                self.conv1.weight.requires_grad = False
                self.conv2.weight.requires_grad = False

                if self.downsample is not None:
                    identity = self.downsample(x)

                out = identity
        else:

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.conv2(out)
            out = self.bn2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            if self.multFlag:
                out = self.prob * out + identity
            else:
                out = out + identity

        out = self.relu(out)

        return out
    

class ScaleModel(nn.Module):
    def __init__(self,
                 nclass,
                 in_channels,
                 features=256,
                 out_channels=[256, 512, 1024, 1024],
                 use_bn=False,
                 use_clstoken=False,
                 use_spn=False,
                 use_prefill=False,
                 use_dino = False,
                 output_act='sigmoid',
                 block=StoDepth_BasicBlock,
                 guide=Guide
                 ):
        
        super(ScaleModel, self).__init__()
        
        self.use_dino = use_dino
        if use_dino:
            self.nclass = nclass
            self.use_clstoken = use_clstoken
            self.use_spn = use_spn
            self.use_prefill = use_prefill

            self.projects = nn.ModuleList([
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channel,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                ) for out_channel in out_channels
            ])

            self.resize_layers = nn.ModuleList([
                nn.ConvTranspose2d(
                    in_channels=out_channels[0],
                    out_channels=out_channels[0],
                    kernel_size=4,
                    stride=4,
                    padding=0),
                nn.ConvTranspose2d(
                    in_channels=out_channels[1],
                    out_channels=out_channels[1],
                    kernel_size=2,
                    stride=2,
                    padding=0),
                nn.Identity(),
                nn.Conv2d(
                    in_channels=out_channels[3],
                    out_channels=out_channels[3],
                    kernel_size=3,
                    stride=2,
                    padding=1)
            ])

            if use_clstoken:
                self.readout_projects = nn.ModuleList()
                for _ in range(len(self.projects)):
                    self.readout_projects.append(
                        nn.Sequential(
                            nn.Linear(2 * in_channels, in_channels),
                            nn.GELU()))

            self.scratch = _make_scratch(
                out_channels,
                [128, 256, 256, 256],
                groups=1,
                expand=False,
            )
        

        bc = 16
        
        layers=(2, 2, 2, 2, 2)
        self._norm_layer=nn.BatchNorm2d
        self.preserve_input = True

        prob_0_L = (1, 0.5)
        self.multFlag = True
        self.prob_now = prob_0_L[0]
        self.prob_delta = prob_0_L[0] - prob_0_L[1]
        self.prob_step = self.prob_delta / (sum(layers) - 1)

        self.conv_img = Basic2d(3, bc * 2, norm_layer=self._norm_layer, kernel_size=5, padding=2)
        self.conv_lidar = Basic2d(1, bc * 2, norm_layer=None, kernel_size=5, padding=2)
        in_channels = bc * 2

        self.inplanes = 32     # 32
        self.layer1_img, self.layer1_lidar = self._make_layer(block, 64, layers[0], stride=1)   # 32-->64
        self.guide1 = guide(64, 64, self._norm_layer, use_dino)

        self.inplanes = 64
        self.layer2_img, self.layer2_lidar = self._make_layer(block, 128, layers[1], stride=2)   # 64-->128
        self.guide2 = guide(128, 128, self._norm_layer, use_dino)

        self.inplanes = 128
        self.layer3_img, self.layer3_lidar = self._make_layer(block, 256, layers[2], stride=2)   # 128-->256
        self.guide3 = guide(256, 256, self._norm_layer, use_dino)

        self.inplanes = 256
        self.layer4_img, self.layer4_lidar = self._make_layer(block, 256, layers[3], stride=2)   # 256-->256
        self.guide4 = guide(256, 256, self._norm_layer, use_dino)

        self.inplanes = 256
        self.layer5_img, self.layer5_lidar = self._make_layer(block, 256, layers[4], stride=2)   # 256-->256
        self.guide5 = guide(256, 256, self._norm_layer, use_dino)
        
        self.layer4d = Basic2dTrans(in_channels * 8, in_channels * 8, self._norm_layer)
        self.upproj0 = nn.Sequential(
            Basic2dTrans(in_channels * 8, in_channels * 4, self._norm_layer),
            Basic2dTrans(in_channels * 4, in_channels * 2, self._norm_layer),
            Basic2dTrans(in_channels * 2, in_channels, self._norm_layer)
        )
        self.weight_offset0 = BasicDepthEncoder(kernel_size=3, block=BasicBlock, bc=bc, norm_layer=nn.BatchNorm2d)

        self.layer3d = Basic2dTrans(in_channels * 8, in_channels * 8, self._norm_layer)
        self.upproj1 = nn.Sequential(
            Basic2dTrans(in_channels * 8, in_channels * 4, self._norm_layer),
            Basic2dTrans(in_channels * 4, in_channels, self._norm_layer)
        )
        self.weight_offset1 = BasicDepthEncoder(kernel_size=3, block=BasicBlock, bc=bc, norm_layer=nn.BatchNorm2d)

        self.layer2d = Basic2dTrans(in_channels * 8, in_channels * 4, self._norm_layer)
        self.upproj2 = nn.Sequential(
            Basic2dTrans(in_channels * 4, in_channels, self._norm_layer)
        )
        self.weight_offset2 = BasicDepthEncoder(kernel_size=3, block=BasicBlock, bc=bc, norm_layer=nn.BatchNorm2d)

        self.layer1d = Basic2dTrans(in_channels * 4, in_channels * 2, self._norm_layer)
        self.conv = Basic2d(in_channels * 2, in_channels, self._norm_layer)
        self.weight_offset3 = BasicDepthEncoder(kernel_size=3, block=BasicBlock, bc=bc, norm_layer=nn.BatchNorm2d)

        self.Post_process = Post_process_deconv(dkn_residual=True, kernel_size=3)


    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        img_downsample, depth_downsample = None, None
        if stride != 1 or self.inplanes != planes :
            img_downsample = nn.Sequential(
                Conv1x1(self.inplanes, planes, stride),
                norm_layer(planes),
            )
            depth_downsample = nn.Sequential(
                Conv1x1(self.inplanes, planes, stride),
                norm_layer(planes),
            )

        m = torch.distributions.bernoulli.Bernoulli(torch.Tensor([self.prob_now]))
        img_layers = [block(self.prob_now, m, self.multFlag, self.inplanes, planes, stride, img_downsample)]
        depth_layers = [block(self.prob_now, m, self.multFlag, self.inplanes, planes, stride, depth_downsample)]
        self.prob_now = self.prob_now - self.prob_step
        self.inplanes = planes

        for _ in range(1, blocks):
            m = torch.distributions.bernoulli.Bernoulli(torch.Tensor([self.prob_now]))
            img_layers.append(block(self.prob_now, m, self.multFlag, self.inplanes, planes))
            depth_layers.append(block(self.prob_now, m, self.multFlag, self.inplanes, planes))
            self.prob_now = self.prob_now - self.prob_step

        return nn.Sequential(*img_layers), nn.Sequential(*depth_layers)



    def forward(self, out_features=None, patch_h=None, patch_w=None, image=None, sparse_depth=None, d_clear=None, prefill_depth=None, certainty=None, img_size=(256, 1216)):
        if self.use_dino:
            out = []
            for i, x in enumerate(out_features):
                x = x[0]
                x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))    # (1, ci, 37, 123)
                x = self.projects[i](x)         # 大小不变 分别输出四个channel的特征    (1, ci, 37, 123)
                y = self.resize_layers[i](x)    # (1, c1, 148, 492) (1, c2, 74, 246) (1, c3, 37, 123) (1, c4, 19, 62)
                out.append(y)

            layer_0, layer_1, layer_2, layer_3, layer_4 = out    # (1, 96, 148, 704) (b, 192, 74, 352) (b, 384, 37, 176) (b, 768, 19, 88)
        
            c1_img_dino = self.scratch.layer1_rn(layer_0)    # 都把通道数转化成指定值 feature：64
            c2_img_dino = self.scratch.layer1_rn(layer_1)    # 都把通道数转化成指定值 feature：128
            c3_img_dino = self.scratch.layer2_rn(layer_2)    # 256
            c4_img_dino = self.scratch.layer3_rn(layer_3)    # 256
            c5_img_dino = self.scratch.layer4_rn(layer_4)    # 256
        else:
            c1_img_dino, c2_img_dino, c3_img_dino, c4_img_dino, c5_img_dino = None, None, None, None, None

        c0_img = self.conv_img(image)
        c0_lidar = self.conv_lidar(sparse_depth)    # 32

        c1_img = self.layer1_img(c0_img)
        c1_lidar = self.layer1_lidar(c0_lidar)      # 64
        c1_lidar_dyn = self.guide1(c1_lidar, c1_img, c1_img_dino)

        c2_img = self.layer2_img(c1_img)
        c2_lidar = self.layer2_lidar(c1_lidar_dyn)      # 128
        c2_lidar_dyn = self.guide2(c2_lidar, c2_img, c2_img_dino)

        c3_img = self.layer3_img(c2_img)
        c3_lidar = self.layer3_lidar(c2_lidar_dyn)  # 256
        c3_lidar_dyn = self.guide3(c3_lidar, c3_img, c3_img_dino)

        c4_img = self.layer4_img(c3_img)
        c4_lidar = self.layer4_lidar(c3_lidar_dyn)  # 256
        c4_lidar_dyn = self.guide4(c4_lidar, c4_img, c4_img_dino)
        
        c5_img = self.layer5_img(c4_img)
        c5_lidar = self.layer5_lidar(c4_lidar_dyn)  # 256
        c5_lidar_dyn = self.guide5(c5_lidar, c5_img, c5_img_dino)

        depth_predictions = []
        c5 = c5_lidar_dyn
        dc4 = self.layer4d(c5)
        c4 = dc4 + c4_lidar_dyn
        c4_up = self.upproj0(c4)
        if self.preserve_input:
            mask = torch.sum(d_clear > 0.0, dim=1, keepdim=True)
            mask = (mask > 0.0).type_as(d_clear)
            prefill_depth = (1.0 - mask) * prefill_depth + mask * d_clear
        else:
            prefill_depth = prefill_depth
        prefill_depth = prefill_depth.detach()
        weight0, offset0 = self.weight_offset0(prefill_depth, c4_up)
        output = self.Post_process(prefill_depth, weight0, offset0)
        depth_predictions.append(output)

        dc3 = self.layer3d(c4)
        c3 = dc3 + c3_lidar_dyn
        c3_up = self.upproj1(c3)
        if self.preserve_input:
            mask = torch.sum(d_clear > 0.0, dim=1, keepdim=True)
            mask = (mask > 0.0).type_as(d_clear)
            output = (1.0 - mask) * output + mask * d_clear
        else:
            output = output
        output = output.detach()
        weight1, offset1 = self.weight_offset1(output, c3_up)
        output = self.Post_process(output, weight1, offset1)
        depth_predictions.append(output)

        dc2 = self.layer2d(c3)
        c2 = dc2 + c2_lidar_dyn
        c2_up = self.upproj2(c2)
        if self.preserve_input:
            mask = torch.sum(d_clear > 0.0, dim=1, keepdim=True)
            mask = (mask > 0.0).type_as(d_clear)
            output = (1.0 - mask) * output + mask * d_clear
        else:
            output = output
        output = output.detach()
        weight2, offset2 = self.weight_offset2(output, c2_up)
        output = self.Post_process(output, weight2, offset2)
        depth_predictions.append(output)

        dc1 = self.layer1d(c2)
        c1 = dc1 + c1_lidar_dyn
        c1 = self.conv(c1)
        c0 = c1 + c0_lidar
        if self.preserve_input:
            mask = torch.sum(d_clear > 0.0, dim=1, keepdim=True)
            mask = (mask > 0.0).type_as(d_clear)
            output = (1.0 - mask) * output + mask * d_clear
        else:
            output = output
        output = output.detach()
        weight3, offset3 = self.weight_offset3(output, c0)
        output = self.Post_process(output, weight3, offset3)

        depth_predictions.append(output)


        # output = {'results': depth_predictions}

        return depth_predictions

    



