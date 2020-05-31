"""SSD Object Detection Model Class"""
import torch
import torch.nn as nn
from .backbone import VGG, ResNet


class SSDResNet(nn.Module):
    """
    SSD Object Detection Model with ResNet backbone
    
    :param backbone : CNN feature extractor e.g. resnet18, resnet50
    :type backbone : str
    :param num_classes : number of output classes
    :type num_classes : int
    """
    def __init__(self, backbone='resnet50', num_classes=2, **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.anchor_boxes = [4, 6, 6, 6, 4, 4]

        self.backbone = ResNet(backbone, **kwargs)
        self.extra_layers = self._build_extra_layers()
        self.loc, self.conf = self._build_prediction_layers()

    def forward(self, x):
        x = self.backbone(x) # extract features from base network

        detection_feed = [x] # extract output for prediction layers
        for l in self.extra_layers:
            x = l(x) # output for auxillary layers
            detection_feed.append(x) # extract output for pred layers

        # Input to pred layers : feature map 38x38, 19x19, 10x10, 5x5, 3x3, 1x1

        # Compute loc coordinates and conf from pred layers
        loc, conf = [], []
        for (x, l, c) in zip(detection_feed, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1) 
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1) 

        return (loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes)) # (n, 8732, 4), (n, 8732, num_classes)
    
    def _build_extra_layers(self):
        """
        utility function to build auxillary layers of SSD
        NOTE : configuration of layers might vary for different backbones
        """

        input_size = self.backbone.out_channels
        extra_layers = []
        for i, (input_size, output_size, channels) in enumerate(zip(input_size[:-1], input_size[1:], [256, 256, 128, 128, 128])):
            if i < 3:
                layer = nn.Sequential(
                    nn.Conv2d(input_size, channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels, output_size, kernel_size=3, padding=1, stride=2, bias=False),
                    nn.BatchNorm2d(output_size),
                    nn.ReLU(inplace=True),
                )
            else:
                layer = nn.Sequential(
                    nn.Conv2d(input_size, channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels, output_size, kernel_size=3, bias=False),
                    nn.BatchNorm2d(output_size),
                    nn.ReLU(inplace=True),
                )

            extra_layers.append(layer)

        return nn.ModuleList(extra_layers)

    def _build_prediction_layers(self):
        """Build prediction layers to output in terms of 8732 boxes - location and conf scores 
        NOTE : configuration remains same given 8732 boxes
        """
        loc, conf = [], []

        for nd, oc in zip(self.anchor_boxes, self.backbone.out_channels):
            loc.append(nn.Conv2d(oc, nd * 4, kernel_size=3, padding=1))
            conf.append(nn.Conv2d(oc, nd * self.num_classes, kernel_size=3, padding=1))

        return nn.ModuleList(loc), nn.ModuleList(conf)



class SSDVGG(nn.Module):
    """
    SSD Object Detection Model with VGG backbone
    
    :param backbone : CNN feature extractor e.g. vgg16, vgg19
    :type backbone : str
    :param num_classes : number of output classes
    :type num_classes : int
    """

    def __init__(self, backbone='vgg16', num_classes=2, **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.anchor_boxes = [4, 6, 6, 6, 4, 4]

        self.backbone = VGG(backbone, **kwargs)
        self.extra_layers = self._build_extra_layers()
        self.loc, self.conf = self._build_prediction_layers()

    def forward(self, x):

        detection_feed = []
        
        x = self.backbone.features[:23](x)
        detection_feed.append(x) # extract output for pred layers
        x = self.backbone.features[23:](x) # compute base features

        for l in self.extra_layers:
            x = l(x) # Compute auxillary features
            detection_feed.append(x) # extract output for pred layers

        # Input to pred layers : feature map 38x38, 19x19, 10x10, 5x5, 3x3, 1x1
        
        loc, conf = [], []
        for (x, l, c) in zip(detection_feed, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1) 
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        return (loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes)) # (n, 8732, 4), (n, 8732, num_classes)

    def _build_extra_layers(self):
        """
        utility function to build auxillary layers of SSD
        NOTE : configuration of layers might vary for different backbones
        """
        extras = [
            nn.Sequential(
                nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
                nn.ReLU(inplace=True),
                nn.Conv2d(1024, 1024, kernel_size=1),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3),
                nn.ReLU(inplace=True)
            ),
        ]
        return nn.ModuleList(extras)
    
    def _build_prediction_layers(self):
        """Build prediction layers to output in terms of 8732 boxes - location and conf scores 
        NOTE : configuration remains same given 8732 boxes
        """
        loc, conf = [], []

        for nd, oc in zip(self.anchor_boxes, self.backbone.out_channels):
            loc.append(nn.Conv2d(oc, nd * 4, kernel_size=3, padding=1))
            conf.append(nn.Conv2d(oc, nd * self.num_classes, kernel_size=3, padding=1))

        return nn.ModuleList(loc), nn.ModuleList(conf)



class SSD(nn.Module):
    """
    SSD Object Detection Model with custom backbone
        
        :param backbone: specify which backbone to use e.g. vgg16, resnet18
        :type backbone: string
        :param num_classes : number of output classes (includes background class)
        :type num_classes: int
        :param **kwargs: other arguments like pretrained=True
        :type **kwargs: dict
    """
    def __init__(self, backbone='vgg16', num_classes=2, **kwargs):
        super().__init__()

        if 'vgg' in backbone:
            self.model = SSDVGG(backbone, num_classes, **kwargs)
        elif 'resnet' in backbone:
            self.model = SSDResNet(backbone, num_classes, **kwargs)
        else:
            raise Exception('backbone not supported : {}!'.format(backbone))

    def forward(self, x):
        """Do forward pass of the model"""
        return self.model(x)
