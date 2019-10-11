'''AlexNet for CIFAR10. FC layers are removed. Paddings are adjusted.
Without BN, the start learning rate should be 0.01
(c) YANG, Wei 
'''
import torch
import torch.nn as nn
import sys
import time
import math 

from src.ar4414.pruning.pruning_layers import GatedConv2d 

__all__ = ['alexnet_gated']

class AlexNetGated(nn.Module):

    def __init__(self, num_classes=10):
        super(AlexNetGated, self).__init__()
        self.conv1 = GatedConv2d(3, 64, kernel_size=11, stride=4, padding=5)
        self.conv2 = GatedConv2d(64, 192, kernel_size=5, padding=2)
        self.conv3 = GatedConv2d(192, 384, kernel_size=3, padding=1)
        self.conv4 = GatedConv2d(384, 256, kernel_size=3, padding=1)
        self.conv5 = GatedConv2d(256, 256, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool2d(x)
        x = self.conv2(x) 
        x = self.relu(x)
        x = self.maxpool2d(x)
        x = self.conv3(x) 
        x = self.relu(x)
        x = self.conv4(x) 
        x = self.relu(x)
        x = self.conv5(x) 
        x = self.relu(x)
        x = self.maxpool2d(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x

    def load_state_dict(self, state_dict, strict=True, initialise=True):
        # print(self.__dict__['_modules']['conv1'].conv.weight)
        if initialise == True:
            relevantLayers = self.__dict__['_modules']
            relevantLayerNames = relevantLayers.keys()
            for k,v in state_dict.items():
                layer = k.split('.')[0]
                tensorType = k.split('.')[1]
                if layer in relevantLayerNames and tensorType == 'weight':
                    if 'conv' in layer:
                        # print(layer, relevantLayers[layer].conv.weight.shape, k, v.shape)
                        # print(type(relevantLayers[layer].conv.weight.data), type(v))
                        relevantLayers[layer].conv.weight.data = v
                    # else:
                    #     # print(layer, relevantLayers[layer].weight.shape, k, v.shape)
                    #     # print(type(relevantLayers[layer].weight.data), type(v))
                    #     relevantLayers[layer].weight.data = v
            return (None, None)
        
        else:
            return super().load_state_dict(state_dict, strict)

def alexnet_gated(**kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    """
    model = AlexNetGated(**kwargs)
    return model
