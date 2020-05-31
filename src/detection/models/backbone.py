"""Backbones for Object Detection SSD model"""

import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from torchvision.models.vgg import vgg16, vgg19


class VGG(nn.Module):
    """
    VGG backbone for SSD
    
    :param backbone : CNN feature extractor e.g. vgg16, vgg19
    :type backbone : str
    :param pretrained : use pretrained model or not
    :type pretrained : bool
    """
    def __init__(self, backbone='vgg16', pretrained=True):
        super().__init__()

        if backbone == 'vgg16':
            backbone = vgg16(pretrained=pretrained)
        elif backbone == 'vgg19':
            backbone = vgg19(pretrained=pretrained)
        else:
            raise Exception('ERROR : backbone not supported : {}'.format(backbone))

        # additional layers SSD-specific to get 8732 boxes overall
        layers = list(backbone.features[:-1])
        layers[16] = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True)
        layers.append(nn.MaxPool2d(kernel_size=3, stride=1, padding=1))
        
        self.features = nn.Sequential(*layers)
        self.out_channels = [512, 1024, 512, 256, 256, 256] # prediction layer channels

    def forward(self, x):
        return self.features(x)


class ResNet(nn.Module):
    """
    ResNet backbone for SSD
    
    :param backbone : CNN feature extractor e.g. resnet18, resnet50
    :type backbone : str
    :param pretrained : use pretrained model or not
    :type pretrained : bool
    """
    def __init__(self, backbone='resnet50', pretrained=True):
        super().__init__()
        if backbone == 'resnet18':
            backbone = resnet18(pretrained=pretrained)
            self.out_channels = [256, 512, 512, 256, 256, 128]
        elif backbone == 'resnet34':
            backbone = resnet34(pretrained=pretrained)
            self.out_channels = [256, 512, 512, 256, 256, 256]
        elif backbone == 'resnet50':
            backbone = resnet50(pretrained=pretrained)
            self.out_channels = [1024, 512, 512, 256, 256, 256]
        elif backbone == 'resnet101':
            backbone = resnet101(pretrained=pretrained)
            self.out_channels = [1024, 512, 512, 256, 256, 256]
        else:
            raise Exception('ERROR : backbone not supported : {}'.format(backbone))

        self.features = nn.Sequential(*list(backbone.children())[:7])

        conv4_block1 = self.features[-1][0]
        conv4_block1.conv1.stride = (1, 1)
        conv4_block1.conv2.stride = (1, 1)
        conv4_block1.downsample[0].stride = (1, 1)

    def forward(self, x):
        return self.features(x)