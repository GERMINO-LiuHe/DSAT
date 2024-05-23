import torch
import torch.nn as nn
import torch.nn.functional as F
from CTrans import ChannelTransformer
from Improved_semhash import maskSemhash


def conv3x3(in_planes, out_planes, strd=1, padding=1, bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stride=strd, padding=padding, bias=bias)


class ConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(ConvBlock, self).__init__()
        planes = int(out_planes / 2)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, bias=False)

        if in_planes != out_planes:
            self.downsample = nn.Sequential(
                nn.BatchNorm2d(in_planes),
                nn.ReLU(True),
                nn.Conv2d(in_planes, out_planes,
                          kernel_size=1, stride=1, bias=False),
            )
        else:
            self.downsample = None

    def forward(self, x):
        residual = x

        out1 = self.bn1(x)
        out1 = F.relu(out1, True)
        out1 = self.conv1(out1)

        out2 = self.bn2(out1)
        out2 = F.relu(out2, True)
        out2 = self.conv2(out2)

        out3 = self.bn3(out2)
        out3 = F.relu(out3, True)
        out3 = self.conv3(out3)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out3 += residual

        return out3


class Upsample(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(Upsample, self).__init__()
        self.upsample = nn.ConvTranspose2d(dim_in, dim_out, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.upsample(x)


class HourGlass(nn.Module):
    def __init__(self, config, depth, num_features, vis=False, img_size=128):
        super(HourGlass, self).__init__()
        self.depth = depth
        self.features = num_features
        self.Upsample = Upsample(256, 256)
        self._generate_network(self.depth)
        self.mtc = ChannelTransformer(config, vis, img_size,
                                      channel_num=[num_features, num_features, num_features, num_features],
                                      patchSize=config.patch_sizes)

    def _generate_network(self, level):
        self.add_module('b1_' + str(level), ConvBlock(256, 256))

        self.add_module('b2_' + str(level), ConvBlock(256, 256))

        if level > 1:
            self._generate_network(level - 1)
        else:
            self.add_module('b2_plus_' + str(level), ConvBlock(256, 256))

        self.add_module('b3_' + str(level), ConvBlock(256, 256))

    def _forward(self, level, inp):
        # Upper branch
        up1_1 = inp
        up1_1 = self._modules['b1_' + str(level)](up1_1)
        # Lower branch
        low1_1 = F.avg_pool2d(inp, 2, stride=2)
        low1_1 = self._modules['b2_' + str(level)](low1_1)

        # Upper branch
        up2_1 = low1_1
        up2_1 = self._modules['b1_' + str(level - 1)](up2_1)
        # Lower branch
        low2_1 = F.avg_pool2d(low1_1, 2, stride=2)
        low2_1 = self._modules['b2_' + str(level - 1)](low2_1)

        # Upper branch
        up3_1 = low2_1
        up3_1 = self._modules['b1_' + str(level - 2)](up3_1)
        # Lower branch
        low3_1 = F.avg_pool2d(low2_1, 2, stride=2)
        low3_1 = self._modules['b2_' + str(level - 2)](low3_1)

        # Upper branch
        up4_1 = low3_1
        up4_1 = self._modules['b1_' + str(level - 3)](up4_1)
        # Lower branch
        low4_1 = F.avg_pool2d(low3_1, 2, stride=2)
        low4_1 = self._modules['b2_' + str(level - 3)](low4_1)

        # 4th layer
        low4_2 = self._modules['b2_plus_' + str(level - 3)](low4_1)

        low1_1, low2_1, low3_1, low4_1, att_weights = self.mtc(low1_1, low2_1, low3_1, low4_1)

        low4_3 = low4_2 + low4_1
        low4_3 = self._modules['b3_' + str(level - 3)](low4_3)
        up4_2 = self.Upsample(low4_3)
        low3_3 = up4_2 + up4_1

        # 3th layer
        low3_3 = low3_3 + low3_1
        low3_3 = self._modules['b3_' + str(level - 2)](low3_3)
        up3_2 = self.Upsample(low3_3)
        low2_3 = up3_2 + up3_1

        # 2th layer
        low2_3 = low2_3 + low2_1
        low2_3 = self._modules['b3_' + str(level - 1)](low2_3)
        up2_2 = self.Upsample(low2_3)
        low1_3 = up2_2 + up2_1

        # 1th layer
        low1_3 = low1_3 + low1_1
        low1_3 = self._modules['b3_' + str(level - 0)](low1_3)
        up1_2 = self.Upsample(low1_3)
        low0_3 = up1_2 + up1_1
        return low0_3

    def forward(self, x):
        return self._forward(self.depth, x)


class HourGlass_2(nn.Module):
    def __init__(self, depth, num_features):
        super(HourGlass_2, self).__init__()
        self.depth = depth
        self.features = num_features
        self.Upsample = Upsample(256, 256)
        self._generate_network(self.depth)

    def _generate_network(self, level):
        self.add_module('b1_' + str(level), ConvBlock(256, 256))

        self.add_module('b2_' + str(level), ConvBlock(256, 256))

        if level > 1:
            self._generate_network(level - 1)
        else:
            self.add_module('b2_plus_' + str(level), ConvBlock(256, 256))

        self.add_module('b3_' + str(level), ConvBlock(256, 256))

    def _forward(self, level, inp):
        # Upper branch
        up1 = inp
        up1 = self._modules['b1_' + str(level)](up1)

        # Lower branch
        low1 = F.avg_pool2d(inp, 2, stride=2)
        low1 = self._modules['b2_' + str(level)](low1)

        if level > 1:
            low2 = self._forward(level - 1, low1)
        else:
            low2 = low1
            low2 = self._modules['b2_plus_' + str(level)](low2)

        low3 = low2
        low3 = self._modules['b3_' + str(level)](low3)

        up2 = self.Upsample(low3)

        return up1 + up2

    def forward(self, x):
        return self._forward(self.depth, x)


class FAN(nn.Module):

    def __init__(self, config, inplanes, outplanes, bn=False):
        super(FAN, self).__init__()
        self.bn = bn
        if bn:
            self.bn = nn.BatchNorm2d(inplanes)

        # Base part
        self.conv1 = nn.Conv2d(inplanes, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = ConvBlock(64, 128)
        self.conv3 = ConvBlock(128, 256)
        self.mask_1 = maskSemhash(10, 256, 64)
        self.conv4_1 = HourGlass(config, 4, 256)
        self.conv4_2 = HourGlass(config, 4, 256)
        self.conv4_3 = HourGlass(config, 4, 256)
        self.conv4_4 = HourGlass(config, 4, 256)
        self.conv5 = ConvBlock(256, 128)
        self.conv6 = conv3x3(128, outplanes)
        self.Upsample = Upsample(128, 128)

    def forward(self, x):

        if self.bn:
            x = self.bn(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2, stride=2)
        x = self.conv3(x)

        mask = self.mask_1(x)
        x = x * mask + x
        # mask = self.mask_2(x)
        # x = x * mask + x
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x = self.conv4_4(x)

        x = self.conv5(x)
        x = self.Upsample(x)
        out = self.conv6(x)

        return out, mask
