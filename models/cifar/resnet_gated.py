from __future__ import absolute_import

'''Resnet for cifar dataset.
Ported form
https://github.com/facebook/fb.resnet.torch
and
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
(c) YANG, Wei
'''
import torch
import torch.nn as nn
import math
import sys
import torch.nn.functional as F
import time

from src.ar4414.pruning.pruning_layers import GatedConv2d 

__all__ = ['resnet_gated']

class UniversalIterator(object):
    def __init__(self, layers):
        self.layersIterator = layers.items().__iter__()
        self.quantLayers = []
        self.index = 0 

    def __iter__(self):
        return self 
    
    def __next__(self):
        if self.index == len(self.quantLayers):
            k,v = next(self.layersIterator)
            while('Gated' not in str(v)):
                k,v = next(self.layersIterator)
            
            topEntity = v 
            self.quantLayers = []
            self.index = 0
            self.parse_description(topEntity)
            quantLayer = self.quantLayers[self.index]
            self.index += 1
        else:
            quantLayer = self.quantLayers[self.index]
            self.index += 1
        
        return quantLayer 
    
    def parse_description(self, v):
        name = v.__class__.__name__
        if name == 'GatedConv2d':
            self.quantLayers.append(v)
        else :
            for k,v in v._modules.items():
                if 'Gated' in str(v):
                    self.parse_description(v)

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return GatedConv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        print('Start of basic block - {}'.format(str(x.shape)))
        residual = x
        
        out = self.conv1(x)
        print('conv1 done')
        out = self.conv2(out)
        print('conv2 done')

        if self.downsample is not None:
            print('downsample')
            residual = self.downsample(x)

        out += residual
        print('added residual - {}, {}'.format(torch.max(out), torch.max(residual)))
        out = F.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = GatedConv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = GatedConv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = GatedConv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNetGated(nn.Module):

    def __init__(self, depth, num_classes=1000):
        super(ResNetGated, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth - 2) // 6

        block = Bottleneck if depth >=44 else BasicBlock

        self.inplanes = 16
        self.conv1 = GatedConv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 16, n)
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        self.layer3 = self._make_layer(block, 64, n, stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for n,m in self.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                if m.affine == True:
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        
        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
    
    def load_state_dict(self, state_dict, strict=True, initialise=True):
        if initialise == True:
            # convLayers = []
            # for v in UniversalIterator(self._modules):
            #     convLayers.append(v)
            # 
            # convLayerNames = [x for x in state_dict.keys() if 'conv' in str(x) or 'downsample.0' in str(x) and 'weight' in str(x)]
            # sameLayerNames = [x for x in self.state_dict().keys() if x in state_dict.keys()]
            # print(len(convLayerNames), len(convLayers), len(sameLayerNames), len(self.state_dict().keys()), len(state_dict.keys()))
            
            # convCount = 0 
            # for k,v in state_dict.items():
            #     if k in convLayerNames:
            #         print(k, convLayers[convCount].conv.weight.shape, v.shape)
            #         convLayers[convCount].conv.weight.data = v
            #         convCount += 1
            
            relevantLayers = []
            relevantLayerNames = []
            for n,m in self.named_modules():
                if isinstance(m, GatedConv2d):
                    relevantLayerNames.append(str(n)+'.weight')
                    relevantLayers.append(m)
            
            for k,v in state_dict.items():
                if k in relevantLayerNames:
                    idx = relevantLayerNames.index(k)
                    print(k, relevantLayers[idx].conv.weight.shape, v.shape)
                    relevantLayers[idx].conv.weight.data = v
            
            return (None, None)
                
        else:
            return super().load_state_dict(state_dict, strict)


def resnet_gated(**kwargs):
    """
    Constructs a ResNet model.
    """
    return ResNetGated(**kwargs)
