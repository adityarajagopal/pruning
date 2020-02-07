import sys
import csv
import os
import numpy as np
import time
from tqdm import tqdm
import json
import pickle
import subprocess
import importlib
import math
import copy

from src.ar4414.pruning.pruners.base import BasicPruning

import torch
import torch.nn as nn

class SqueezeNetPruning(BasicPruning):
#{{{
    def __init__(self, params, model):
    #{{{
        self.fileName = 'squeezenet_{}.py'.format(int(params.pruningPerc))
        self.netName = 'SqueezeNet'

        # selects only convs and fc layers 
        # used in get_layer_params to get sizes of only convs and fcs 
        self.convs_and_fcs = lambda lName : True if ('conv' in lName and 'weight' in lName) else False
        
        # function that specifies conv layers to skip when pruning
        # used in structure_l1_weight
        self.layerSkip = lambda lName : True if 'conv2' in lName and 'fire' not in lName else False

        super().__init__(params, model)
    #}}}
    
    def inc_prune_rate(self, layerName):
    #{{{
        lParam = str(layerName) + '.weight'
        self.layerSizes[lParam][0] -= 1 

        nextLayerName = self.layersInOrder[lParam]
        nextLayerSize = [self.layerSizes[n] for n in nextLayerName]
        currLayerSize = self.layerSizes[lParam]
        paramsPruned = currLayerSize[1]*currLayerSize[2]*currLayerSize[3]
        # check if FC layer
        if nextLayerName == ['module.conv2.weight']: 
            paramsPruned += nextLayerSize[0][0]
        else:
            for i,nLS in enumerate(nextLayerSize):
                paramsPruned += nLS[0]*nLS[2]*nLS[3] 
                nLN = nextLayerName[i]
                self.layerSizes[nLN][1] -= 1
        
        return (100.* paramsPruned / self.totalParams)
    #}}}
    
    def get_layer_params(self):
    #{{{
        for p in self.model.named_parameters():
            paramsInLayer = 1
            for dim in p[1].size():
                paramsInLayer *= dim
            self.totalParams += paramsInLayer
            
            if self.convs_and_fcs(p[0]):
                self.layerSizes[p[0]] = list(p[1].size())
        
        # construct layers in order to have the fire.conv1 as next layer after both fire.conv2 and fire.conv3
        # since concatenation occurs 
        self.layersInOrder = list(self.layerSizes.keys())
        newOrder = {}
        for i,l in enumerate(self.layersInOrder):
            if 'fire' not in l:
                if 'conv1' in l:
                    newOrder[l] = [self.layersInOrder[i+1]]
            elif 'fire' in l:
                if 'conv1' in l:
                    newOrder[l] = [self.layersInOrder[i+1], self.layersInOrder[i+2]]
                elif 'conv2' in l:
                    newOrder[l] = [self.layersInOrder[i+2]]
                elif 'conv3' in l:
                    newOrder[l] = [self.layersInOrder[i+1]]
        
        self.layersInOrder = newOrder
    #}}}

    def is_conv_or_fc(self, lParam):
    #{{{
        if 'conv' in lParam and 'weight' in lParam: 
            return True
        else:
            return False
    #}}}

    def prune_layer(self, lParam):
    #{{{
        if 'conv' in lParam and 'weight' in lParam:
            return True
        else:
            return False
    #}}}

    def skip_layer(self, lName):
    #{{{
        if 'conv2' in lName and 'fire' not in lName:
            return True
        else:
            return False
    #}}} 
        
    def write_net(self):
    #{{{
        def fprint(text):
            print(text, file=self.modelDesc)
        
        self.modelDesc = open(self.filePath, 'w+')

        fprint('import torch')
        fprint('import torch.nn as nn')
        fprint('import torch.nn.functional as F')
    
        fprint('')
        fprint('class {}(nn.Module):'.format(self.netName))
        fprint('\tdef __init__(self, num_classes=10):')
        fprint('\t\tsuper().__init__()')
        fprint('')

        channelsPruned = {l:len(v) for l,v in self.channelsToPrune.items()}
        start = True
        currentIpChannels = 3

        linesToWrite = {}
        prunedModel = copy.deepcopy(self.model)
        for n,m in prunedModel.named_modules():
        #{{{
            if not m._modules:
                if n in channelsPruned.keys():
                    m.out_channels -= channelsPruned[n] 
                    m.in_channels = currentIpChannels if not start else m.in_channels

                    if 'fire' in n:
                        if 'conv2' in n:
                            expandOpChannels = m.out_channels
                        elif 'conv3' in n:
                            expandOpChannels += m.out_channels
                            currentIpChannels = expandOpChannels
                        else:
                            currentIpChannels = m.out_channels
                    else:
                        currentIpChannels = m.out_channels
                    
                    bnIpChannels = m.out_channels

                    if start:
                        start = False
                
                elif isinstance(m, nn.BatchNorm2d):
                    m.num_features = bnIpChannels

                elif isinstance(m, nn.Linear):
                    m.in_features = currentIpChannels
                
                elif isinstance(m, nn.LogSoftmax):
                    linesToWrite[n] = '\t\tself.{} = nn.LogSoftmax(dim=1)'.format('_'.join(n.split('.')[1:]))
                    continue

                elif isinstance(m, nn.ReLU):
                    continue

                linesToWrite[n] = '\t\tself.{} = nn.{}'.format('_'.join(n.split('.')[1:]), str(m))
        #}}}

        self.orderedKeys = list(linesToWrite.keys())
        [fprint(linesToWrite[k]) for k in self.orderedKeys]

        fprint('')
        fprint('\tdef forward(self, x):')

        i = 0
        while i < len(self.orderedKeys): 
            layer = self.orderedKeys[i]
            
            if 'fire' in layer:
                if 'conv1' in layer:
                    fprint('\t\tx = F.relu(self.{}(self.{}(x)))'.format('_'.join(self.orderedKeys[i+1].split('.')[1:]), '_'.join(layer.split('.')[1:])))
                    i = i+2
                
                elif 'conv2' in layer:
                    fprint('\t\tout1x1 = self.{}(self.{}(x))'.format('_'.join(self.orderedKeys[i+1].split('.')[1:]), '_'.join(layer.split('.')[1:])))
                    i = i+2
                
                elif 'conv3' in layer:
                    fprint('\t\tout3x3 = self.{}(self.{}(x))'.format('_'.join(self.orderedKeys[i+1].split('.')[1:]), '_'.join(layer.split('.')[1:])))
                    fprint('\t\tx = F.relu(torch.cat([out1x1, out3x3], 1))')
                    i = i+2
            
            elif 'conv' in layer:
                if 'bn' in self.orderedKeys[i+1]:
                    fprint('\t\tx = F.relu(self.{}(self.{}(x)))'.format('_'.join(self.orderedKeys[i+1].split('.')[1:]), '_'.join(layer.split('.')[1:])))
                    i = i+2
                else:
                    fprint('\t\tx = self.{}(x)'.format('_'.join(layer.split('.')[1:])))
                    i = i+1
            
            elif ('maxpool' in layer) or ('avg_pool' in layer) or ('softmax' in layer):
                fprint('\t\tx = self.{}(x)'.format('_'.join(layer.split('.')[1:])))
                i = i+1
            
        fprint('\t\treturn x.squeeze()')
        fprint('')
        fprint('def squeezenet(**kwargs):')
        fprint('\treturn SqueezeNet(**kwargs)')

        self.modelDesc.close()
        #}}}                  

    def transfer_weights(self, oModel, pModel):
    #{{{
        parentModel = oModel.state_dict() 
        prunedModel = pModel.state_dict() 

        ipChannelsToPrune = []
        ipChannelsKept = []
        opChannelsKept = []
        fireIpChannelsPruned = []
        for k in self.orderedKeys:
            if 'conv' in k:
            #{{{
                layer = k
                param = k + '.weight'
                paramB = k + '.bias'
                pParam = k.split('.')[0] + '.' + '_'.join(k.split('.')[1:]) + '.weight'
                pParamB = k.split('.')[0] + '.' + '_'.join(k.split('.')[1:]) + '.bias'

                opChannelsToPrune = self.channelsToPrune[layer].copy()

                allIpChannels = list(range(parentModel[param].shape[1]))
                allOpChannels = list(range(parentModel[param].shape[0]))
                ipChannelsKept = list(set(allIpChannels) - set(ipChannelsToPrune))
                opChannelsKept = list(set(allOpChannels) - set(opChannelsToPrune))
                
                tmpW = parentModel[param][opChannelsKept,:]
                prunedModel[pParam] = tmpW[:,ipChannelsKept] 
                prunedModel[pParamB] = parentModel[paramB][opChannelsKept]
                
                if 'fire' in k:
                    if 'conv2' in k:
                        fireIpChannelsPruned = opChannelsToPrune.copy()
                        offset = len(allOpChannels)
                    elif 'conv3' in k:
                        fireIpChannelsPruned += [x + offset for x in opChannelsToPrune]
                        ipChannelsToPrune = fireIpChannelsPruned
                    else:
                        ipChannelsToPrune = opChannelsToPrune
                else:
                    ipChannelsToPrune = opChannelsToPrune
            #}}}
            
            elif 'bn' in k:
            #{{{
                layer = k
                
                paramW = k + '.weight'
                paramB = k + '.bias'
                paramM = k + '.running_mean'
                paramV = k + '.running_var'
                paramNB = k + '.num_batches_tracked'
                
                pParamW = k.split('.')[0] + '.' + '_'.join(k.split('.')[1:]) + '.weight'
                pParamB = k.split('.')[0] + '.' + '_'.join(k.split('.')[1:]) + '.bias'
                pParamM = k.split('.')[0] + '.' + '_'.join(k.split('.')[1:]) + '.running_mean'
                pParamV = k.split('.')[0] + '.' + '_'.join(k.split('.')[1:]) + '.running_var'
                pParamNB = k.split('.')[0] + '.' + '_'.join(k.split('.')[1:]) + '.num_batches_tracked'

                prunedModel[pParamW] = parentModel[paramW][opChannelsKept]
                prunedModel[pParamB] = parentModel[paramB][opChannelsKept]
                
                prunedModel[pParamM] = parentModel[paramM][opChannelsKept]
                prunedModel[pParamV] = parentModel[paramV][opChannelsKept]
                prunedModel[pParamNB] = parentModel[paramNB]
            #}}}
            
        pModel.load_state_dict(prunedModel)

        return pModel
    #}}}
#}}}
