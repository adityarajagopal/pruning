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

from src.ar4414.pruning.pruners.base import BasicPruning

import torch
import torch.nn as nn

class AlexNetPruning(BasicPruning):
#{{{
    def __init__(self, params, model):
        self.fileName = 'alexnet_{}.py'.format(int(params.pruningPerc))
        self.netName = 'AlexNet'
        super().__init__(params, model)

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
        for n,m in self.model.named_modules():
        #{{{
            if not m._modules:
                if n in channelsPruned.keys():
                    m.out_channels -= channelsPruned[n] 
                    m.in_channels = currentIpChannels if not start else m.in_channels
                    currentIpChannels = m.out_channels
                    if start:
                        start = False
                
                elif isinstance(m, nn.BatchNorm2d):
                    m.num_features = currentIpChannels

                elif isinstance(m, nn.Linear):
                    m.in_features = currentIpChannels

                elif isinstance(m, nn.ReLU):
                    continue

                linesToWrite[n] = '\t\tself.{} = nn.{}'.format('_'.join(n.split('.')[1:]), str(m))
        #}}}
        
        self.orderedKeys = list(linesToWrite.keys())
        [fprint(linesToWrite[k]) for k in self.orderedKeys]
                    
        fprint('')
        fprint('\tdef forward(self, x):')

        i = 0
        maxpoolConvs = ['module.conv1', 'module.conv2', 'module.conv5']
        maxPoolLayerName = [x for x in self.orderedKeys if 'maxpool' in x][0]
        while i < len(self.orderedKeys): 
            if 'conv' in self.orderedKeys[i]:
                fprint('\t\tx = F.relu(self.{}(x))'.format('_'.join(self.orderedKeys[i].split('.')[1:])))
                if self.orderedKeys[i] in maxpoolConvs:
                    fprint('\t\tx = self.{}(x)'.format('_'.join(maxPoolLayerName.split('.')[1:])))
                i = i+1
            
            elif 'linear' in self.orderedKeys[i] or 'classifier' in self.orderedKeys[i]:
                fprint('\t\tx = x.view(x.size(0), -1)')
                fprint('\t\tx = self.{}(x)'.format('_'.join(self.orderedKeys[i].split('.')[1:])))
                i += 1

            elif 'maxpool' in self.orderedKeys[i]:
                i += 1

        fprint('\t\treturn x')
        fprint('')
        fprint('def alexnet(**kwargs):')
        fprint('\treturn AlexNet(**kwargs)')
        
        self.modelDesc.close()
        #}}}                  

    def transfer_weights(self, oModel, pModel):
    #{{{
        parentModel = oModel.state_dict() 
        prunedModel = pModel.state_dict() 

        ipChannelsToPrune = []
        ipChannelsKept = []
        opChannelsKept = []
        for k in self.orderedKeys:
            if 'conv' in k:
                layer = k
                param = k + '.weight'
                paramB = k + '.bias'
                pParam = k.split('.')[0] + '.' + '_'.join(k.split('.')[1:]) + '.weight'
                pParamB = k.split('.')[0] + '.' + '_'.join(k.split('.')[1:]) + '.bias'

                opChannelsToPrune = self.channelsToPrune[layer]

                allIpChannels = list(range(parentModel[param].shape[1]))
                allOpChannels = list(range(parentModel[param].shape[0]))
                ipChannelsKept = list(set(allIpChannels) - set(ipChannelsToPrune))
                opChannelsKept = list(set(allOpChannels) - set(opChannelsToPrune))
                
                tmpW = parentModel[param][opChannelsKept,:]
                prunedModel[pParam] = tmpW[:,ipChannelsKept] 
                prunedModel[pParamB] = parentModel[paramB][opChannelsKept]
                
                ipChannelsToPrune = opChannelsToPrune
            
            elif 'linear' in k or 'classifier' in k:
                layer = k
                paramW = k + '.weight'
                paramB = k + '.bias'
                pParamW = k.split('.')[0] + '.' + '_'.join(k.split('.')[1:]) + '.weight'
                pParamB = k.split('.')[0] + '.' + '_'.join(k.split('.')[1:]) + '.bias'

                prunedModel[pParamB] = parentModel[paramB]
                prunedModel[pParamW] = parentModel[paramW][:,opChannelsKept]
            
        pModel.load_state_dict(prunedModel)

        return pModel
    #}}}
#}}}














