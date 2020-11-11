import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as M
from torch.nn.utils.spectral_norm import spectral_norm
import utils

class ResNeXtBottleneck(nn.Module):
    def __init__(self, in_channels=256, out_channels=256, stride=1, cardinality=32, dilate=1):
        super(ResNeXtBottleneck, self).__init__()
        D = out_channels // 2
        self.out_channels = out_channels
        self.conv_reduce = nn.Conv2d(in_channels, D, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_conv = nn.Conv2d(D, D, kernel_size=2 + stride, stride=stride, padding=dilate, dilation=dilate,
                                   groups=cardinality,
                                   bias=False)
        self.conv_expand = nn.Conv2d(D, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.shortcut = nn.Sequential()
        if stride != 1:
            self.shortcut.add_module('shortcut',
                                     nn.AvgPool2d(2, stride=2))

    def forward(self, x):
        bottleneck = self.conv_reduce.forward(x)
        bottleneck = F.leaky_relu(bottleneck, 0.2, True)
        bottleneck = self.conv_conv.forward(bottleneck)
        bottleneck = F.leaky_relu(bottleneck, 0.2, True)
        bottleneck = self.conv_expand.forward(bottleneck)
        x = self.shortcut.forward(x)
        return x + bottleneck

class Generator(nn.Module):
    def __init__(self, nf=64):
        super(Generator, self).__init__()

        self.toH = nn.Sequential(nn.Conv2d(4, nf, kernel_size=7, stride=1, padding=3), nn.LeakyReLU(0.2, True))

        self.to0 = nn.Sequential(nn.Conv2d(1, nf // 2, kernel_size=3, stride=1, padding=1),  # 512
                                 nn.LeakyReLU(0.2, True))
        self.to1 = nn.Sequential(nn.Conv2d(nf // 2, nf, kernel_size=4, stride=2, padding=1),  # 256
                                 nn.LeakyReLU(0.2, True))
        self.to2 = nn.Sequential(nn.Conv2d(nf, nf * 2, kernel_size=4, stride=2, padding=1),  # 128
                                 nn.LeakyReLU(0.2, True))
        self.to3 = nn.Sequential(nn.Conv2d(nf * 3, nf * 4, kernel_size=4, stride=2, padding=1),  # 64
                                 nn.LeakyReLU(0.2, True))
        self.to4 = nn.Sequential(nn.Conv2d(nf * 4, nf * 8, kernel_size=4, stride=2, padding=1),  # 32
                                 nn.LeakyReLU(0.2, True))

        tunnel4 = nn.Sequential(*[ResNeXtBottleneck(nf * 8, nf * 8, cardinality=32, dilate=1) for _ in range(20)])

        self.tunnel4 = nn.Sequential(nn.Conv2d(nf * 8, nf * 8, kernel_size=3, stride=1, padding=1),
                                     nn.LeakyReLU(0.2, True),
                                     tunnel4,
                                     nn.Conv2d(nf * 8, nf * 4 * 4, kernel_size=3, stride=1, padding=1),
                                     nn.PixelShuffle(2),
                                     nn.LeakyReLU(0.2, True)
                                     )  # 64

        depth = 2
        tunnel = [ResNeXtBottleneck(nf * 4, nf * 4, cardinality=32, dilate=1) for _ in range(depth)]
        tunnel += [ResNeXtBottleneck(nf * 4, nf * 4, cardinality=32, dilate=2) for _ in range(depth)]
        tunnel += [ResNeXtBottleneck(nf * 4, nf * 4, cardinality=32, dilate=4) for _ in range(depth)]
        tunnel += [ResNeXtBottleneck(nf * 4, nf * 4, cardinality=32, dilate=2),
                   ResNeXtBottleneck(nf * 4, nf * 4, cardinality=32, dilate=1)]
        tunnel3 = nn.Sequential(*tunnel)

        self.tunnel3 = nn.Sequential(nn.Conv2d(nf * 8, nf * 4, kernel_size=3, stride=1, padding=1),
                                     nn.LeakyReLU(0.2, True),
                                     tunnel3,
                                     nn.Conv2d(nf * 4, nf * 2 * 4, kernel_size=3, stride=1, padding=1),
                                     nn.PixelShuffle(2),
                                     nn.LeakyReLU(0.2, True)
                                     )  # 128

        tunnel = [ResNeXtBottleneck(nf * 2, nf * 2, cardinality=32, dilate=1) for _ in range(depth)]
        tunnel += [ResNeXtBottleneck(nf * 2, nf * 2, cardinality=32, dilate=2) for _ in range(depth)]
        tunnel += [ResNeXtBottleneck(nf * 2, nf * 2, cardinality=32, dilate=4) for _ in range(depth)]
        tunnel += [ResNeXtBottleneck(nf * 2, nf * 2, cardinality=32, dilate=2),
                   ResNeXtBottleneck(nf * 2, nf * 2, cardinality=32, dilate=1)]
        tunnel2 = nn.Sequential(*tunnel)

        self.tunnel2 = nn.Sequential(nn.Conv2d(nf * 4, nf * 2, kernel_size=3, stride=1, padding=1),
                                     nn.LeakyReLU(0.2, True),
                                     tunnel2,
                                     nn.Conv2d(nf * 2, nf * 4, kernel_size=3, stride=1, padding=1),
                                     nn.PixelShuffle(2),
                                     nn.LeakyReLU(0.2, True)
                                     )

        tunnel = [ResNeXtBottleneck(nf, nf, cardinality=16, dilate=1)]
        tunnel += [ResNeXtBottleneck(nf, nf, cardinality=16, dilate=2)]
        tunnel += [ResNeXtBottleneck(nf, nf, cardinality=16, dilate=4)]
        tunnel += [ResNeXtBottleneck(nf, nf, cardinality=16, dilate=2),
                   ResNeXtBottleneck(nf, nf, cardinality=16, dilate=1)]
        tunnel1 = nn.Sequential(*tunnel)

        self.tunnel1 = nn.Sequential(nn.Conv2d(nf * 2, nf, kernel_size=3, stride=1, padding=1),
                                     nn.LeakyReLU(0.2, True),
                                     tunnel1,
                                     nn.Conv2d(nf, nf * 2, kernel_size=3, stride=1, padding=1),
                                     nn.PixelShuffle(2),
                                     nn.LeakyReLU(0.2, True)
                                     )

        self.exit = nn.Conv2d(nf, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, sketch, hint):
        hint = self.toH(hint)

        x0 = self.to0(sketch)
        x1 = self.to1(x0)
        x2 = self.to2(x1)
        x3 = self.to3(torch.cat([x2, hint], 1))  # !
        x4 = self.to4(x3)

        x = self.tunnel4(x4)
        x = self.tunnel3(torch.cat([x, x3], 1))
        x = self.tunnel2(torch.cat([x, x2], 1))
        x = self.tunnel1(torch.cat([x, x1], 1))
        x = torch.tanh(self.exit(torch.cat([x, x0], 1)))

        return x

class Discriminator(nn.Module):
    # initializers
    def __init__(self, in_nc, out_nc, nf=32):
        super(Discriminator, self).__init__()
        self.input_nc = in_nc
        self.output_nc = out_nc
        self.nf = nf
        self.activation = nn.LeakyReLU(0.2, True)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_nc, nf, 3, 1, 1),
            nn.LeakyReLU(0.2, True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(nf, nf * 2, 3, 2, 1),
            nn.LeakyReLU(0.2, True),
            spectral_norm(nn.Conv2d(nf * 2, nf * 4, 3, 1, 1)),
            nn.LeakyReLU(0.2, True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(nf * 4, nf * 4, 3, 2, 1),
            nn.LeakyReLU(0.2, True),
            spectral_norm(nn.Conv2d(nf * 4, nf * 8, 3, 1, 1)),
            nn.LeakyReLU(0.2, True)
        )

        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(nf * 8, nf * 8, 3, 1, 1)),
            nn.LeakyReLU(0.2, True)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(nf * 8, out_nc, 3, 1, 1)
        )

        # utils.initialize_weights(self)

    # forward method
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = torch.sigmoid(x)

        return x

class VGG19(nn.Module):
    def __init__(self, init_weights=None, feature_mode=False, batch_norm=False, num_classes=1000):
        super(VGG19, self).__init__()
        self.cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
        self.init_weights = init_weights
        self.feature_mode = feature_mode
        self.batch_norm = batch_norm
        self.num_clases = num_classes
        self.features = self.make_layers(self.cfg, batch_norm)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if not init_weights == None:
            self.load_state_dict(torch.load(init_weights))

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.feature_mode:
            module_list = list(self.features.modules())
            for l in module_list[1:27]:                 # conv4_4
                x = l(x)
        if not self.feature_mode:
            x = x.view(x.size(0), -1)
            x = self.classifier(x)

        return x