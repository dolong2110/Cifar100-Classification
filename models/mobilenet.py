"""mobilenet in pytorch
[1] Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand,
    Marco Andreetto, Hartwig Adam
    MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications
    https://arxiv.org/abs/1704.04861
"""

import torch.nn as nn
import torch.nn.functional as F

from training.classification_model import ImageClassificationBase

class DepthSeperabelConv2d(ImageClassificationBase):

    def __init__(self, input_channels, output_channels, kernel_size, **kwargs):
        super().__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(
                input_channels,
                input_channels,
                kernel_size,
                groups=input_channels,
                **kwargs),
            nn.BatchNorm2d(input_channels),
            nn.ReLU(inplace=True)
        )

        self.pointwise = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)

        return x


class BasicConv2d(ImageClassificationBase):

    def __init__(self, input_channels, output_channels, kernel_size, **kwargs):

        super().__init__()
        self.conv = nn.Conv2d(
            input_channels, output_channels, kernel_size, **kwargs)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class MobileNet(ImageClassificationBase):

    """
    Args:
        width multipler: The role of the width multiplier α is to thin
                         a network uniformly at each layer. For a given
                         layer and width multiplier α, the number of
                         input channels M becomes αM and the number of
                         output channels N becomes αN.
    """

    def __init__(self, width_multiplier=1, class_num=100):
        super().__init__()

        alpha = width_multiplier
        self.stem = nn.Sequential(
            BasicConv2d(3, int(32 * alpha), 3, padding=1, bias=False),
            DepthSeperabelConv2d(
                int(32 * alpha),
                int(64 * alpha),
                3,
                padding=1,
                bias=False
            )
        )

        # downsample
        self.conv1 = nn.Sequential(
            DepthSeperabelConv2d(
                int(64 * alpha),
                int(128 * alpha),
                3,
                stride=2,
                padding=1,
                bias=False
            ),
            DepthSeperabelConv2d(
                int(128 * alpha),
                int(128 * alpha),
                3,
                padding=1,
                bias=False
            )
        )

        # downsample
        self.conv2 = nn.Sequential(
            DepthSeperabelConv2d(
                int(128 * alpha),
                int(256 * alpha),
                3,
                stride=2,
                padding=1,
                bias=False
            ),
            DepthSeperabelConv2d(
                int(256 * alpha),
                int(256 * alpha),
                3,
                padding=1,
                bias=False
            )
        )

        # downsample
        self.conv3 = nn.Sequential(
            DepthSeperabelConv2d(
                int(256 * alpha),
                int(512 * alpha),
                3,
                stride=2,
                padding=1,
                bias=False
            ),

            DepthSeperabelConv2d(
                int(512 * alpha),
                int(512 * alpha),
                3,
                padding=1,
                bias=False
            ),
            DepthSeperabelConv2d(
                int(512 * alpha),
                int(512 * alpha),
                3,
                padding=1,
                bias=False
            ),
            DepthSeperabelConv2d(
                int(512 * alpha),
                int(512 * alpha),
                3,
                padding=1,
                bias=False
            ),
            DepthSeperabelConv2d(
                int(512 * alpha),
                int(512 * alpha),
                3,
                padding=1,
                bias=False
            ),
            DepthSeperabelConv2d(
                int(512 * alpha),
                int(512 * alpha),
                3,
                padding=1,
                bias=False
            )
        )

        # downsample
        self.conv4 = nn.Sequential(
            DepthSeperabelConv2d(
                int(512 * alpha),
                int(1024 * alpha),
                3,
                stride=2,
                padding=1,
                bias=False
            ),
            DepthSeperabelConv2d(
                int(1024 * alpha),
                int(1024 * alpha),
                3,
                padding=1,
                bias=False
            )
        )

        self.fc = nn.Linear(int(1024 * alpha), class_num)
        self.avg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.stem(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

"""mobilenetv2 in pytorch
[1] Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen
    MobileNetV2: Inverted Residuals and Linear Bottlenecks
    https://arxiv.org/abs/1801.04381
"""

class LinearBottleNeck(ImageClassificationBase):

    def __init__(self, in_channels, out_channels, stride, t=6, class_num=100):
        super().__init__()

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * t, 1),
            nn.BatchNorm2d(in_channels * t),
            nn.ReLU6(inplace=True),

            nn.Conv2d(in_channels * t, in_channels * t, 3, stride=stride, padding=1, groups=in_channels * t),
            nn.BatchNorm2d(in_channels * t),
            nn.ReLU6(inplace=True),

            nn.Conv2d(in_channels * t, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )

        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):

        residual = self.residual(x)

        if self.stride == 1 and self.in_channels == self.out_channels:
            residual += x

        return residual

class MobileNetV2(ImageClassificationBase):

    def __init__(self, class_num=100):
        super().__init__()

        self.pre = nn.Sequential(
            nn.Conv2d(3, 32, 1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )

        self.stage1 = LinearBottleNeck(32, 16, 1, 1)
        self.stage2 = self._make_stage(2, 16, 24, 2, 6)
        self.stage3 = self._make_stage(3, 24, 32, 2, 6)
        self.stage4 = self._make_stage(4, 32, 64, 2, 6)
        self.stage5 = self._make_stage(3, 64, 96, 1, 6)
        self.stage6 = self._make_stage(3, 96, 160, 1, 6)
        self.stage7 = LinearBottleNeck(160, 320, 1, 6)

        self.conv1 = nn.Sequential(
            nn.Conv2d(320, 1280, 1),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True)
        )

        self.conv2 = nn.Conv2d(1280, class_num, 1)

    def forward(self, x):
        x = self.pre(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x)
        x = self.stage7(x)
        x = self.conv1(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)

        return x

    @staticmethod
    def _make_stage(repeat, in_channels, out_channels, stride, t):

        layers = [LinearBottleNeck(in_channels, out_channels, stride, t)]

        while repeat - 1:
            layers.append(LinearBottleNeck(out_channels, out_channels, 1, t))
            repeat -= 1

        return nn.Sequential(*layers)

def mobilenetv2():
    return MobileNetV2()

def mobilenet(alpha=1, class_num=100):
    return MobileNet(alpha, class_num)