from torch import nn
from torch.nn import functional as F
import torch
from torchvision import models
import torchvision
import math
input_size = (512, 512)

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

class Interpolate(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=False):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode
        self.scale_factor = scale_factor
        self.align_corners = align_corners

    def forward(self, x):
        x = self.interp(x, size=self.size, scale_factor=self.scale_factor,
                        mode=self.mode, align_corners=self.align_corners)
        return x

def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)


class ConvRelu(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class DecoderBlockV2(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=True):
        super(DecoderBlockV2, self).__init__()
        self.in_channels = in_channels

        if is_deconv:
            """
                Paramaters for Deconvolution were chosen to avoid artifacts, following
                link https://distill.pub/2016/deconv-checkerboard/
            """

            self.block = nn.Sequential(
                ConvRelu(in_channels, middle_channels),
                nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2,
                                   padding=1),
                nn.ReLU(inplace=True)
            )
        else:
            self.block = nn.Sequential(
                Interpolate(scale_factor=2, mode='bilinear'),
                ConvRelu(in_channels, middle_channels),
                ConvRelu(middle_channels, out_channels),
            )
            # self.block = nn.Sequential(
            #     # Interpolate(scale_factor=2, mode='bilinear'),
            #     ConvRelu(in_channels, middle_channels),
            #     ConvRelu(middle_channels, out_channels),
            # )

    def forward(self, x):
        return self.block(x)

class UNet16(nn.Module):
    def __init__(self, num_classes=1, num_filters=32, pretrained=False, is_deconv=False):
        """
        :param num_classes:
        :param num_filters:
        :param pretrained:
            False - no pre-trained network used
            True - encoder pre-trained with VGG16
        :is_deconv:
            False: bilinear interpolation is used in decoder
            True: deconvolution is used in decoder
        """
        super().__init__()
        self.num_classes = num_classes

        self.pool = nn.MaxPool2d(2, 2)

        #print(torchvision.models.vgg16(pretrained=pretrained))

        self.encoder = torchvision.models.vgg16(pretrained=pretrained).features

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(self.encoder[0],
                                   self.relu,
                                   self.encoder[2],
                                   self.relu)

        self.conv2 = nn.Sequential(self.encoder[5],
                                   self.relu,
                                   self.encoder[7],
                                   self.relu)

        self.conv3 = nn.Sequential(self.encoder[10],
                                   self.relu,
                                   self.encoder[12],
                                   self.relu,
                                   self.encoder[14],
                                   self.relu)

        self.conv4 = nn.Sequential(self.encoder[17],
                                   self.relu,
                                   self.encoder[19],
                                   self.relu,
                                   self.encoder[21],
                                   self.relu)

        self.conv5 = nn.Sequential(self.encoder[24],
                                   self.relu,
                                   self.encoder[26],
                                   self.relu,
                                   self.encoder[28],
                                   self.relu)

        self.center = DecoderBlockV2(512, num_filters * 8 * 2, num_filters * 8, is_deconv)

        self.dec5 = DecoderBlockV2(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec4 = DecoderBlockV2(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec3 = DecoderBlockV2(256 + num_filters * 8, num_filters * 4 * 2, num_filters * 2, is_deconv)
        self.dec2 = DecoderBlockV2(128 + num_filters * 2, num_filters * 2 * 2, num_filters, is_deconv)
        self.dec1 = ConvRelu(64 + num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.pool(conv1))
        conv3 = self.conv3(self.pool(conv2))
        conv4 = self.conv4(self.pool(conv3))
        conv5 = self.conv5(self.pool(conv4))

        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([center, conv5], 1))

        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))

        if self.num_classes > 1:
            x_out = F.log_softmax(self.final(dec1), dim=1)
        else:
            x_out = self.final(dec1)
            #x_out = F.sigmoid(x_out)

        return x_out

class UNetResNet(nn.Module):
    """PyTorch U-Net model using ResNet(34, 101 or 152) encoder.
    UNet: https://arxiv.org/abs/1505.04597
    ResNet: https://arxiv.org/abs/1512.03385
    Proposed by Alexander Buslaev: https://www.linkedin.com/in/al-buslaev/
    Args:
            encoder_depth (int): Depth of a ResNet encoder (34, 101 or 152).
            num_classes (int): Number of output classes.
            num_filters (int, optional): Number of filters in the last layer of decoder. Defaults to 32.
            dropout_2d (float, optional): Probability factor of dropout layer before output layer. Defaults to 0.2.
            pretrained (bool, optional):
                False - no pre-trained weights are being used.
                True  - ResNet encoder is pre-trained on ImageNet.
                Defaults to False.
            is_deconv (bool, optional):
                False: bilinear interpolation is used in decoder.
                True: deconvolution is used in decoder.
                Defaults to False.
    """

    def __init__(self, encoder_depth, num_classes, num_filters=32, dropout_2d=0.2, batch_size=1,
                 pretrained=False, is_deconv=False):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_2d = dropout_2d

        if encoder_depth == 34:
            self.encoder = torchvision.models.resnet34(pretrained=pretrained)
            bottom_channel_nr = 512
        elif encoder_depth == 18:
            self.encoder = torchvision.models.resnet18(pretrained=pretrained)
            bottom_channel_nr = 512
        elif encoder_depth == 101:
            self.encoder = torchvision.models.resnet101(pretrained=pretrained)
            bottom_channel_nr = 2048
        elif encoder_depth == 152:
            self.encoder = torchvision.models.resnet152(pretrained=pretrained)
            bottom_channel_nr = 2048
        else:
            raise NotImplementedError('only 34, 101, 152 version of Resnet are implemented')

        self.h0 = torch.Tensor(batch_size, 64, 1, 1).cuda()
        self.c0 = torch.Tensor(batch_size, 64, 1, 1).cuda()

        self.sse_lstm1 = sse_lstm(64, 64)
        self.sse_lstm2 = sse_lstm(64, 64)
        self.sse_lstm3 = sse_lstm(64, 128)
        self.sse_lstm4 = sse_lstm(128, 256)
        # self.sse_lstm5 = sse_lstm(256, 512)
        self.sse_lstm6 = sse_lstm(256, 256)
        # self.sse_lstm7 = sse_lstm(256, 256)
        self.sse_lstm8 = sse_lstm(256, 256)
        self.sse_lstm9 = sse_lstm(256, 128)
        self.sse_lstm10 = sse_lstm(128, 64)
        self.sse_lstm11 = sse_lstm(64, 32)

        self.pool = nn.MaxPool2d(2, 2)

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(self.encoder.conv1,
                                   self.encoder.bn1,
                                   self.encoder.relu,
                                   self.pool)

        self.conv2 = self.encoder.layer1

        self.conv3 = self.encoder.layer2

        self.conv4 = self.encoder.layer3

        # self.conv5 = self.encoder.layer4

        self.center = DecoderBlockV2(bottom_channel_nr // 2, num_filters * 8 * 2, num_filters * 8, is_deconv)

        # self.dec5 = DecoderBlockV2(bottom_channel_nr + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)

        self.dec4 = DecoderBlockV2(bottom_channel_nr // 2 + num_filters * 8, num_filters * 8 * 2, num_filters * 8,
                                   is_deconv)
        self.dec3 = DecoderBlockV2(bottom_channel_nr // 4 + num_filters * 8, num_filters * 4 * 2, num_filters * 2 * 2,
                                   is_deconv)
        self.dec2 = DecoderBlockV2(bottom_channel_nr // 8 + num_filters * 2 * 2, num_filters * 2 * 2, num_filters * 2,
                                   is_deconv)
        self.dec1 = DecoderBlockV2(num_filters * 2, num_filters * 2, num_filters, is_deconv)
        self.dec0 = ConvRelu(num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        nn.init.normal_(self.h0, 0, 0.1)
        nn.init.normal_(self.c0, 0, 0.1)
        conv1 = self.conv1(x)
        sse1, h1, c1 = self.sse_lstm1(conv1, self.h0, self.c0)
        conv2 = self.conv2(sse1)
        sse2, h2, c2 = self.sse_lstm2(conv2, h1, c1)
        conv3 = self.conv3(sse2)
        sse3, h3, c3 = self.sse_lstm3(conv3, h2, c2)
        conv4 = self.conv4(sse3)
        sse4, h4, c4 = self.sse_lstm4(conv4, h3, c3)
        # conv5 = self.conv5(sse4)
        # sse5, h5, c5 = self.sse_lstm5(conv5, h4, c4)

        pool = self.pool(sse4)
        center = self.center(pool)
        sse6, h6, c6 = self.sse_lstm6(center, h4, c4)

        # dec5 = self.dec5(torch.cat([sse6, sse5], 1))
        # sse7, h7, c7 = self.sse_lstm7(dec5, h6, c6)

        dec4 = self.dec4(torch.cat([sse6, sse4], 1))
        sse8, h8, c8 = self.sse_lstm8(dec4, h6, c6)

        dec3 = self.dec3(torch.cat([sse8, sse3], 1))
        sse9, h9, c9 = self.sse_lstm9(dec3, h6, c6)

        dec2 = self.dec2(torch.cat([sse9, sse2], 1))
        sse10, h10, c10 = self.sse_lstm10(dec2, h9, c9)

        dec1 = self.dec1(sse10)
        sse11, h11, c11 = self.sse_lstm11(dec1, h10, c10)

        dec0 = self.dec0(sse11)

        return self.final(F.dropout2d(dec0, p=self.dropout_2d))