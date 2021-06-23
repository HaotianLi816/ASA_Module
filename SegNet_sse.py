import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2 as cv
import numpy as np
from PIL import Image
import torch.utils.data as Data
import os
import time
import argparse
import math

bn_momentum = 0.1  # BN层的momentum
torch.cuda.manual_seed(1)  # 设置随机种子

class Se_module_diff(nn.Module):
    def __init__(self, inp, oup, Avg_size = 1, se_ratio = 4):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d((Avg_size, Avg_size))
        num_squeezed_channels = max(1,int(inp / se_ratio))
        self._se_reduce = nn.Conv2d(in_channels=inp, out_channels=num_squeezed_channels, kernel_size=1)
        self._se_expand = nn.Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)
        self.Avg_size = Avg_size
        self.reset_parameters()

    #x and z are different conv layer and z pass through more convs
    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


    def forward(self, x, z):
        SIZE = z.size()
        y = self.avg(x)
        y = self._se_reduce(y)
        y = y * torch.sigmoid(y)
        y = self._se_expand(y)
        if self.Avg_size != 1:
            y = F.upsample_bilinear(y, size=[SIZE[2], SIZE[3]])
        z = torch.sigmoid(y) * z
        return z

# 编码器
class Encoder(nn.Module):
    def __init__(self, input_channels):
        super(Encoder, self).__init__()

        self.sse_lstm2 = Se_module_diff(64, 128)
        self.sse_lstm3 = Se_module_diff(128, 256)
        self.sse_lstm4 = Se_module_diff(256, 512)
        self.sse_lstm5 = Se_module_diff(512, 512)


        self.enco1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, momentum=bn_momentum),
            nn.ReLU()
        )
        self.enco2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, momentum=bn_momentum),
            nn.ReLU()
        )
        self.enco3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, momentum=bn_momentum),
            nn.ReLU()
        )
        self.enco4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=bn_momentum),
            nn.ReLU()
        )
        self.enco5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=bn_momentum),
            nn.ReLU()
        )

    def forward(self, x):
        id = []

        x1 = self.enco1(x)
        x1, id1 = F.max_pool2d(x1, kernel_size=2, stride=2, return_indices=True)  # 保留最大值的位置
        id.append(id1)

        x2 = self.enco2(x1)
        x2, id2 = F.max_pool2d(x2, kernel_size=2, stride=2, return_indices=True)
        id.append(id2)
        sse2 = self.sse_lstm2(x1, x2)

        x3 = self.enco3(sse2)
        x3, id3 = F.max_pool2d(x3, kernel_size=2, stride=2, return_indices=True)
        id.append(id3)
        sse3 = self.sse_lstm3(sse2, x3)

        x4 = self.enco4(sse3)
        x4, id4 = F.max_pool2d(x4, kernel_size=2, stride=2, return_indices=True)
        id.append(id4)
        sse4 = self.sse_lstm4(sse3, x4)

        x5 = self.enco5(sse4)
        x5, id5 = F.max_pool2d(x5, kernel_size=2, stride=2, return_indices=True)
        id.append(id5)
        sse5 = self.sse_lstm5(sse4, x5)

        return sse5, id


# 编码器+解码器
class SegNet(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(SegNet, self).__init__()

        self.weights_new = self.state_dict()
        self.encoder = Encoder(input_channels)

        self.sse_lstm6 = Se_module_diff(512, 512)
        self.sse_lstm7 = Se_module_diff(512, 256)
        self.sse_lstm8 = Se_module_diff(256, 128)
        self.sse_lstm9 = Se_module_diff(128, 64)
        self.sse_lstm10 = Se_module_diff(64, 1)

        self.deco1 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=bn_momentum),
            nn.ReLU()
        )
        self.deco2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, momentum=bn_momentum),
            nn.ReLU()
        )
        self.deco3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, momentum=bn_momentum),
            nn.ReLU()
        )
        self.deco4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, momentum=bn_momentum),
            nn.ReLU()
        )
        self.deco5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(64, output_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        x, id = self.encoder(x)

        x6 = F.max_unpool2d(x, id[4], kernel_size=2, stride=2)
        x6 = self.deco1(x6)
        sse6 = self.sse_lstm6(x, x6)

        x7 = F.max_unpool2d(sse6, id[3], kernel_size=2, stride=2)
        x7 = self.deco2(x7)
        sse7 = self.sse_lstm7(sse6, x7)

        x8 = F.max_unpool2d(sse7, id[2], kernel_size=2, stride=2)
        x8 = self.deco3(x8)
        sse8 = self.sse_lstm8(sse7, x8)

        x9 = F.max_unpool2d(sse8, id[1], kernel_size=2, stride=2)
        x9 = self.deco4(x9)
        sse9 = self.sse_lstm9(sse8, x9)

        x10 = F.max_unpool2d(sse9, id[0], kernel_size=2, stride=2)
        x10 = self.deco5(x10)
        sse10 = self.sse_lstm10(sse9, x10)

        return sse10

    # 删掉VGG-16后面三个全连接层的权重
    def load_weights(self, weights_path):
        weights = torch.load(weights_path)
        del weights["classifier.0.weight"]
        del weights["classifier.0.bias"]
        del weights["classifier.3.weight"]
        del weights["classifier.3.bias"]
        del weights["classifier.6.weight"]
        del weights["classifier.6.bias"]

        names = []
        for key, value in self.encoder.state_dict().items():
            if "num_batches_tracked" in key:
                continue
            names.append(key)

        for name, dict in zip(names, weights.items()):
            self.weights_new[name] = dict[1]

        self.encoder.load_state_dict(self.weights_new)


class MyDataset(Data.Dataset):
    def __init__(self, txt_path):
        super(MyDataset, self).__init__()

        paths = open(txt_path, "r")

        image_label = []
        for line in paths:
            line.rstrip("\n")
            line.lstrip("\n")
            path = line.split()
            image_label.append((path[0], path[1]))

        self.image_label = image_label

    def __getitem__(self, item):
        image, label = self.image_label[item]

        image = cv.imread(image)
        image = cv.resize(image, (224, 224))
        image = image/255.0  # 归一化输入
        image = torch.Tensor(image)
        image = image.permute(2, 0, 1)  # 将图片的维度转换成网络输入的维度（channel, width, height）

        label = cv.imread(label, 0)
        label = cv.resize(label, (224, 224))
        label = torch.Tensor(label)

        return image, label

    def __len__(self):
        return  len(self.image_label)