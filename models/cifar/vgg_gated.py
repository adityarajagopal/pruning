'''VGG for CIFAR10. FC layers are removed.
(c) YANG, Wei 
'''
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math
import sys

from src.ar4414.pruning.pruning_layers import GatedConv2d 

__all__ = ['VGGGated', 'vgg16_gated']

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
}

class VGGGated(nn.Module):

    def __init__(self, features, num_classes=1000):
        super(VGGGated, self).__init__()
        self.features = features
        self.classifier = nn.Linear(512, num_classes)
        # self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()
                m.bias.data.fill_(1.0)
    
    def load_state_dict(self, state_dict, strict=True, initialise=True):
        if initialise == True:
            relevantLayerNames = []
            relevantLayers = []
            for n,m in self.named_modules():
                if isinstance(m, nn.Conv2d):
                    relevantLayerNames.append(n)
                    relevantLayers.append(m)
            
            baselineConvLayers = [0,3,7,10,14,17,20,24,27,30,34,37,40]
            baselineRelevantLayers = ['features.'+str(x)+'.weight' for x in baselineConvLayers]

            for k,v in state_dict.items():
                if k in baselineRelevantLayers:
                    idx = baselineRelevantLayers.index(k)
                    relevantLayers[idx].weight.data = v
            
            return (None, None)
        
        else:
            return super().load_state_dict(state_dict, strict)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = GatedConv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d]
            # if batch_norm:
            #     layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            # else:
            #     layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg11(**kwargs):
    """VGG 11-layer model (configuration "A")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['A']), **kwargs)
    return model


def vgg11_bn(**kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    model = VGG(make_layers(cfg['A'], batch_norm=True), **kwargs)
    return model


def vgg13(**kwargs):
    """VGG 13-layer model (configuration "B")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['B']), **kwargs)
    return model


def vgg13_bn(**kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization"""
    model = VGG(make_layers(cfg['B'], batch_norm=True), **kwargs)
    return model


def vgg16_gated(**kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGGGated(make_layers(cfg['D']), **kwargs)
    return model


def vgg16_bn(**kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    model = VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)
    return model


def vgg19(**kwargs):
    """VGG 19-layer model (configuration "E")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['E']), **kwargs)
    return model


def vgg19_bn(**kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    equential
    return model
