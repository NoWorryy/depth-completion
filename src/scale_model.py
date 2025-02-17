import torch
import torch.nn as nn
import torch.nn.functional as F
from blocks import _make_scratch, _make_fusion_block, Prefill, Prefill_nearest
from spn import BasicDepthEncoder, Post_process_deconv


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
                 output_act='sigmoid'):
        
        super(ScaleModel, self).__init__()

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
            features,
            groups=1,
            expand=False,
        )

        self.scratch.stem_transpose = None

        self.scratch.refinenet1 = _make_fusion_block(
            features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(
            features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(
            features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(
            features, use_bn, use_res1=False)

        head_features_1 = features
        head_features_2 = 32

        act_func = nn.Sigmoid() if output_act == 'sigmoid' else nn.Identity()

        if nclass > 1:
            self.scratch.output_conv = nn.Sequential(
                nn.Conv2d(head_features_1, head_features_1,
                          kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.Conv2d(head_features_1, nclass,
                          kernel_size=1, stride=1, padding=0),
            )
        else:
            self.scratch.output_conv1 = nn.Conv2d(
                head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)

            self.scratch.output_conv2 = nn.Sequential(
                nn.Conv2d(head_features_1 // 2, head_features_2,
                          kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(head_features_2),
                nn.ReLU(True),
                nn.Conv2d(head_features_2, 1, kernel_size=1,
                          stride=1, padding=0, bias=False),
                nn.BatchNorm2d(1),
                act_func,
            )
        
        if self.use_spn:
            self.weight_offset = BasicDepthEncoder(kernel_size=3, bc=64, norm_layer=nn.BatchNorm2d)
            self.Post_process = Post_process_deconv(dkn_residual=True, kernel_size=3)
        
        if self.use_prefill:
            # self.prefill1 = Prefill(in_ch=out_channels[0], out_ch=32, level=1)
            # self.prefill2 = Prefill(in_ch=out_channels[1], out_ch=32, level=2)
            # self.prefill3 = Prefill(in_ch=out_channels[2], out_ch=64, level=3)
            # self.prefill4 = Prefill(in_ch=out_channels[3], out_ch=64, level=4)

            self.prefill = Prefill_nearest(num=4)


    def forward(self, out_features, patch_h, patch_w, sparse_scale=None, certainty=None, img_size=(256, 1216)):
        out0 = []
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]

            x = x.permute(0, 2, 1).reshape(
                (x.shape[0], x.shape[-1], patch_h, patch_w))    # (1, ci, 37, 123)

            x = self.projects[i](x)         # 大小不变 分别输出四个channel的特征    (1, ci, 37, 123)
            # out0.append(x)

            y = self.resize_layers[i](x)    # (1, c1, 148, 492) (1, c2, 74, 246) (1, c3, 37, 123) (1, c4, 19, 62)

            out.append(y)

        # layer1, layer2, layer3, layer4 = out0       # (1, ci, 37, 123)
        layer_1, layer_2, layer_3, layer_4 = out    # (1, ci, hi, wi)
        
        if self.use_prefill:
            # mid_scale1 = self.prefill1(sparse_scale, layer1)
            # mid_scale2 = self.prefill2(sparse_scale, layer2)
            # mid_scale3 = self.prefill3(sparse_scale, layer3)
            # mid_scale4 = self.prefill4(sparse_scale, layer4)

            mid_scale = self.prefill(sparse_scale)
        
        if certainty is not None:
            # mid_scale1 = torch.cat((mid_scale1, certainty), 1)  # (B, 2, H0, W0)
            # mid_scale2 = torch.cat((mid_scale2, certainty), 1)  # (B, 2, H0, W0)
            # mid_scale3 = torch.cat((mid_scale3, certainty), 1)  # (B, 2, H0, W0)
            # mid_scale4 = torch.cat((mid_scale4, certainty), 1)  # (B, 2, H0, W0)
            
            mid_scale = torch.cat((mid_scale, certainty), 1)  # (B, 2, H0, W0)

        layer_1_rn = self.scratch.layer1_rn(layer_1)    # 都把通道数转化成指定值 feature：128
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:], prompt_depth=mid_scale)          # (B, 128, H3, W3)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:], prompt_depth=mid_scale)  # (B, 128, H2, W2)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:], prompt_depth=mid_scale)  # (B, 128, H1, W1)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn, prompt_depth=mid_scale)     # (B, 128, 2H1, 2W1)
        

        out = self.scratch.output_conv1(path_1)     # (B, 64, 2H1, 2W1)
        out_feat = F.interpolate(out, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True)  # (B, 64, H0, W0)
        out = self.scratch.output_conv2(out_feat)   # (B, 1, H0, W0) H0=518

        if self.use_spn:
            weight, offset = self.weight_offset(out, out_feat)
            out = self.Post_process(out, weight, offset)

        ouput_scale = F.interpolate(out, img_size, mode="bilinear", align_corners=True)
               
        return ouput_scale