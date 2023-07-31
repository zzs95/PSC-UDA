"""UNet based on ResNet34"""
import copy
import math
import torch
import torch_scatter
import torch.nn as nn
import torch.nn.functional as F
# from torchvision.models.resnet import resnet34
from monai.networks.nets.resnet import resnet34
from collections import OrderedDict
from data.utils.edge_utils import random_sample_edge_point, sample_point_feat
import os
import sys
sys.path.append(os.getcwd())

class FCNResNet34(nn.Module):
    def __init__(self, config):
        super(FCNResNet34, self).__init__()
        self.pretrained = config.pretrained
        self.hiden_size = config.hiden_size
        net = resnet34(n_input_channels=1, spatial_dims=3, shortcut_type='A')
        if self.pretrained:
            pretrained_dict = torch.load('./models/resnet_34.pth')['state_dict']
            net_dict = net.state_dict()
            new_state_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in net_dict.keys()}
            net_dict.update(new_state_dict)
            net.load_state_dict(net_dict)

        conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=1, padding=3, bias=False)
        conv1.weight.data = net.conv1.weight.data
        self.layer_in = nn.Sequential(
            conv1, net.bn1, net.relu,
            net.maxpool,)
        self.layer_in_src = None
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4

        # Decoder
        self.deconv_layer1 = nn.Sequential(
            nn.Conv3d(64, self.hiden_size, kernel_size=7, stride=1, padding=3, bias=False),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
        )
        self.deconv_layer2 = nn.Sequential(
            nn.Conv3d(128, self.hiden_size, kernel_size=7, stride=1, padding=3, bias=False),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode='nearest'),
        )
        self.deconv_layer3 = nn.Sequential(
            nn.Conv3d(256, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(64, self.hiden_size, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode='nearest'),
        )
        self.deconv_layer4 = nn.Sequential(
            nn.Conv3d(512, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(64, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(64, self.hiden_size, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode='nearest'),
        )

    def init_src_encoder(self):
        self.layer_in_src = copy.deepcopy(self.layer_in)
        self.layer2_src = copy.deepcopy(self.layer2)
        self.layer3_src = copy.deepcopy(self.layer3)
        self.layer4_src = copy.deepcopy(self.layer4)

    def forward(self, data_dict, src=False):
        # pad input to be divisible by 16 = 2 ** 4
        x = data_dict['img']
        d, h, w = x.shape[2], x.shape[3], x.shape[4]
        if h % 16 != 0 or w % 16 != 0:
            d_pad = 16 * math.ceil(d / 16)
            h_pad = 16 * math.ceil(h / 16)
            w_pad = 16 * math.ceil(w / 16)
            x = F.pad(x, [0,d_pad-d, 0,h_pad-h, 0,w_pad-w], "constant")
            # assert False, "invalid input size: {}".format(x.shape)

        # Encoder
        if src:
            layer1_out = self.layer_in_src(x)
            layer2_out = self.layer2_src(layer1_out)
            layer3_out = self.layer3_src(layer2_out)
            layer4_out = self.layer4_src(layer3_out)
        else:
            layer1_out = self.layer_in(x)
            layer2_out = self.layer2(layer1_out)
            layer3_out = self.layer3(layer2_out)
            layer4_out = self.layer4(layer3_out)

        # Deconv
        layer1_out = self.deconv_layer1(layer1_out)
        layer2_out = self.deconv_layer2(layer2_out)
        layer3_out = self.deconv_layer3(layer3_out)
        layer4_out = self.deconv_layer4(layer4_out)

        data_dict['img_scale2'] = layer1_out
        data_dict['img_scale4'] = layer2_out
        data_dict['img_scale8'] = layer3_out
        data_dict['img_scale16'] = layer4_out

        return data_dict



def runtest():
    b, c, d, h, w = 2, 1, 32, 32, 32
    image = torch.randn(b, c, d, h, w).cuda()
    net_trg = FCNResNet34(pretrained=False)
    net_src = net_trg.init_src_encoder()
    net_trg.cuda()
    feats = net_trg({'img': image})
    print('feats', feats.shape)


if __name__ == '__main__':
    runtest()
