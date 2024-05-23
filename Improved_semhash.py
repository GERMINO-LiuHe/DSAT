import torch
from torch import nn
import torch.nn.functional as F


class Improved_Semhash(nn.Module):
    def __init__(self):
        super().__init__()
        self.threshold = 0.

    def saturating_sigmoid(self, x):
        value_1 = torch.ones_like(x)
        value_0 = torch.zeros_like(x)
        return torch.max(value_0, torch.min(value_1, 1.2*torch.sigmoid(x)-0.1))

    def forward(self, scores):
        B, clip_num, _ = scores.size()
        if self.training:
            gauss_nosie = torch.randn(B, clip_num, 1).cuda()
        else:
            gauss_nosie = torch.zeros(B, clip_num, 1).cuda()
        score_n = scores + gauss_nosie #{g_e}'
        v1 = self.saturating_sigmoid(score_n) #g_a
        v2 = (score_n > self.threshold).float() - torch.sigmoid(score_n - self.threshold).detach() + torch.sigmoid(score_n - self.threshold)
        v2 += v1 - v1.detach()
        seed = torch.rand(1)
        if self.training:
            return v1 if seed > 0.5 else v2
        else:
            return v2


def conv3x3(in_planes, out_planes, strd=1, padding=1, bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stride=strd, padding=padding, bias=bias)


class ConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(ConvBlock, self).__init__()
        planes = int(out_planes/2)
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


def mychangesize(input, imagesize):
    #  input[B,C 1]  output [B,C,IMAGESIZE,IMAGESIZE]
    input = torch.repeat_interleave(input, imagesize, dim=2)
    input = torch.unsqueeze(input, dim=-1)
    output = torch.repeat_interleave(input, imagesize, dim=3)

    return output


class maskSemhash(nn.Module):
    def __init__(self, b_size, c_size, img_size):
        super().__init__()
        self.b_size = b_size
        self.c_size = c_size
        self.img_size = img_size
        self.cursize = img_size // 4

        # self.conlayer1 = ConvBlock(c_size, c_size) # 特征提取器
        # self.conlayer2 = ConvBlock(c_size, img_size)
        #
        #
        # self.connected_layer_1 = nn.Linear(in_features=self.cursize*self.cursize*self.img_size, out_features=self.cursize*self.cursize)
        # self.connected_layer_2 = nn.Linear(in_features=self.cursize*self.cursize, out_features=self.c_size)
        # self.bn1 = nn.BatchNorm1d(self.cursize*self.cursize)
        self.semhash = Improved_Semhash()

    def forward(self, input):
        B, C, W, H = input.size()
        input = F.adaptive_avg_pool2d(input, (1, 1))
        output = torch.squeeze(input, -1)
        score = self.semhash(output)
        score = torch.unsqueeze(score, -1)
        return score


if __name__ == '__main__':
    # a = torch.randn(2, 10, 1)
    # semhash = Improved_Semhash()
    # score = semhash(a)
    # print(score)

    # connected_layer_1 = nn.Linear(in_features=64 * 64 * 64, out_features=64*64)
    # connected_layer_2 = nn.Linear(in_features=64 * 64, out_features=64)
    # bn1 = nn.BatchNorm1d(64*64)
    # conlayer = ConvBlock(256, 64)
    # input = torch.randn(2, 256, 64, 64)
    # input = conlayer(input)
    # input = input.view(2, 64*64*64)
    # # print(input.shape)
    # output = connected_layer_1(input)
    # output = bn1(output)
    # output = F.relu(output, True)
    # output = connected_layer_2(output)
    # print(output)
    # output = torch.reshape(output, (2, 64, 1))
    # output = output.view(2, 64, 1)
    # semhash = Improved_Semhash()
    # score = semhash(output)
    # print(score)

    input = torch.randn(2, 10, 5, 5).cuda()
    # output = torch.reshape(input, (2, 64, 1))
    model = maskSemhash(2, 10, 5).cuda()
    # model.eval()
    score = model(input)
    print(score*input)


    # output = torch.randn(2, 10, 1)
    # print(output)
    # output = torch.repeat_interleave(output, 5, dim=2)
    # output = torch.unsqueeze(output, dim=-1)
    # print(output.shape)
    # output = torch.repeat_interleave(output, 5, dim=3)
    # print(output)
