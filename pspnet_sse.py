import torch
from torch import nn
from torch.nn import functional as F

from extractors import ResNet_sse, Se_module_diff



class PSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.interpolate(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)


class PSPUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )

    def forward(self, x):
        h, w = 2 * x.size(2), 2 * x.size(3)
        p = F.interpolate(input=x, size=(h, w), mode='bilinear')
        return self.conv(p)


class PSPNet(nn.Module):
    def __init__(self, n_classes=18, sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend=34,
                 pretrained=True):
        super().__init__()

        self.sse_lstm1 = Se_module_diff(512, 1024)
        self.sse_lstm2 = Se_module_diff(1024, 512)
        self.sse_lstm3 = Se_module_diff(512, 256)
        self.sse_lstm4 = Se_module_diff(256, 128)
        self.sse_lstm5 = Se_module_diff(128, 64)
        self.sse_lstm6 = Se_module_diff(64, 64)

        self.feats = ResNet_sse(backend, pretrained=True)
        self.psp = PSPModule(psp_size, 1024, sizes)
        self.drop_1 = nn.Dropout2d(p=0.3)

        self.up_1 = PSPUpsample(1024, 512)
        self.up_2 = PSPUpsample(512, 256)
        self.up_3 = PSPUpsample(256, 128)
        self.up_4 = PSPUpsample(128, 64)
        self.up_5 = PSPUpsample(64, 64)

        self.drop_2 = nn.Dropout2d(p=0.15)
        self.final = nn.Conv2d(64, n_classes, kernel_size=1)

        self.classifier = nn.Sequential(
            nn.Linear(deep_features_size, 256),
            nn.ReLU(),
            nn.Linear(256, n_classes)
        )

    def forward(self, x):
        f = self.feats(x)
        p = self.psp(f)
        sse1 = self.sse_lstm1(f, p)

        p = self.up_1(sse1)
        p = self.drop_1(p)
        sse2 = self.sse_lstm2(sse1, p)   # 1024/512

        p = self.up_2(sse2)
        p = self.drop_2(p)
        sse3 = self.sse_lstm3(sse2, p)    #512/256

        p = self.up_3(sse3)
        p = self.drop_2(p)
        sse4 = self.sse_lstm4(sse3, p)  # 256/128

        p = self.up_4(sse4)
        p = self.drop_2(p)
        sse5 = self.sse_lstm5(sse4, p)  # 128/64

        p = self.up_5(sse5)
        sse6 = self.sse_lstm6(sse5, p)  # 64/64
        p = self.final(sse6)

        return p
