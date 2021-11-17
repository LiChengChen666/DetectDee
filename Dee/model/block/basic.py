import torch
from torch import nn
from torch.nn import functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
            super(BasicBlock, self).__init__()
            self.output_channel = planes
            self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                                        padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                                        padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)
            self.downsample = downsample
            self.stride = stride
        def get_output_channel(self):
                return self.output_channel

        def forward(self, x):
                residual = x

                out = self.conv1(x)
                out = self.bn1(out)
                out = self.relu(out)

                out = self.conv2(out)
                out = self.bn2(out)

                if self.downsample is not None:
                        residual = self.downsample(x)
                out += residual
                out = self.relu(out)

                return out