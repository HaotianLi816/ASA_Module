import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2 as cv
import numpy as np
from PIL import Image
import torch.utils.data as Data
import math
import os
import time
import argparse


bn_momentum = 0.1  # BN层的momentum
torch.cuda.manual_seed(1)  # 设置随机种子

class sse_lstm(nn.Module):
    def __init__(self, inp, oup, Avg_size = 1, se_ratio = 4):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d((Avg_size, Avg_size))
        num_squeezed_channels = max(1,int(inp / se_ratio))
        self._se_reduce = nn.Conv2d(in_channels=inp, out_channels=num_squeezed_channels, kernel_size=1)
        self._se_expand = nn.Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)
        self.Avg_size = Avg_size
        self.reset_parameters()
        self.layer_norm = nn.LayerNorm(oup, eps=1e-05, elementwise_affine=True)


        self.lstm = nn.LSTM(oup, oup, batch_first=True)   # (x1,x2,x3,x4)   input_size = x3
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


    def forward(self, FM, h, c, seq_len=1):                     # c = h = (BS,IF,1,1)
        c = self._se_reduce(c)                              # BS*IS/R*1*1
        c = c * torch.sigmoid(c)                            # BS*IS/R*1*1
        c = self._se_expand(c)                              # BS*HS*1*1
        c = self.layer_norm(c.view(-1, c.size(1)))#.size()[1:])
        c = c.view(1, -1, c.size(1))                          # BS*1*HS

        h = self._se_reduce(h)
        h = h * torch.sigmoid(h)
        h = self._se_expand(h)
        h = self.layer_norm(h.view(-1, h.size(1)))                     # layer normalization
        h = h.view(1, -1, h.size(1))

        FM1 = self.avg(FM)                                  # BS*HS*1*1
        FM1 = FM1.view(-1, seq_len, FM1.size(1))              # BS*seq_len*HS
        output, (hn, cn) = self.lstm(FM1, (h, c))             # cn = hn = (batch,num_layers,HS)

        hn = hn.view(-1, hn.size(2), 1, 1)
        cn = cn.view(-1, cn.size(2), 1, 1)

        FM2 = torch.sigmoid(hn) * FM

        return FM2, hn, cn

# 编码器
class Encoder(nn.Module):
    def __init__(self, input_channels, batch_size=1):
        super(Encoder, self).__init__()

        self.h0 = torch.Tensor(batch_size, 64, 1, 1).cuda()
        self.c0 = torch.Tensor(batch_size, 64, 1, 1).cuda()

        self.sse_lstm1 = sse_lstm(64, 64)
        self.sse_lstm2 = sse_lstm(64, 128)
        self.sse_lstm3 = sse_lstm(128, 256)
        self.sse_lstm4 = sse_lstm(256, 512)
        self.sse_lstm5 = sse_lstm(512, 512)


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
        nn.init.normal_(self.h0, 0, 0.1)
        nn.init.normal_(self.c0, 0, 0.1)

        x = self.enco1(x)
        sse1, h1, c1 = self.sse_lstm1(x, self.h0, self.c0)
        x, id1 = F.max_pool2d(sse1, kernel_size=2, stride=2, return_indices=True)  # 保留最大值的位置
        id.append(id1)
        x = self.enco2(x)
        sse2, h2, c2 = self.sse_lstm2(x, h1, c1)
        x, id2 = F.max_pool2d(sse2, kernel_size=2, stride=2, return_indices=True)
        id.append(id2)
        x = self.enco3(x)
        sse3, h3, c3 = self.sse_lstm3(x, h2, c2)
        x, id3 = F.max_pool2d(sse3, kernel_size=2, stride=2, return_indices=True)
        id.append(id3)
        x = self.enco4(x)
        sse4, h4, c4 = self.sse_lstm4(x, h3, c3)
        x, id4 = F.max_pool2d(sse4, kernel_size=2, stride=2, return_indices=True)
        id.append(id4)
        x = self.enco5(x)
        sse5, h5, c5 = self.sse_lstm5(x, h4, c4)
        x, id5 = F.max_pool2d(sse5, kernel_size=2, stride=2, return_indices=True)

        id.append(id5)

        return x, id, h5, c5


# 编码器+解码器
class SegNet(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(SegNet, self).__init__()

        self.weights_new = self.state_dict()
        self.encoder = Encoder(input_channels)

        self.sse_lstm1 = sse_lstm(512, 512)
        self.sse_lstm2 = sse_lstm(512, 256)
        self.sse_lstm3 = sse_lstm(256, 128)
        self.sse_lstm4 = sse_lstm(128, 64)
        self.sse_lstm5 = sse_lstm(64, 1)

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
        x, id, h, c = self.encoder(x)

        x = F.max_unpool2d(x, id[4], kernel_size=2, stride=2)
        x = self.deco1(x)
        sse1, h1, c1 = self.sse_lstm1(x, h, c)

        x = F.max_unpool2d(sse1, id[3], kernel_size=2, stride=2)
        x = self.deco2(x)
        sse2, h2, c2 = self.sse_lstm2(x, h1, c1)

        x = F.max_unpool2d(sse2, id[2], kernel_size=2, stride=2)
        x = self.deco3(x)
        sse3, h3, c3 = self.sse_lstm3(x, h2, c2)

        x = F.max_unpool2d(sse3, id[1], kernel_size=2, stride=2)
        x = self.deco4(x)
        sse4, h4, c4 = self.sse_lstm4(x, h3, c3)

        x = F.max_unpool2d(sse4, id[0], kernel_size=2, stride=2)
        x = self.deco5(x)
        sse5, h5, c5 = self.sse_lstm5(x, h4, c4)

        return sse5

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