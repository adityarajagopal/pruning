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

class AlexNetPruning(BasicPruning):
#{{{
    def __init__(self, params, model):
    #{{{
        self.fileName = 'alexnet_{}.py'.format(int(params.pruningPerc))
        self.netName = 'AlexNet'
        
        super().__init__(params, model)
    #}}} 

    def is_conv_or_fc(self, lParam):
    #{{{
        if ('conv' in lParam or 'classifier' in lParam) and ('weight' in lParam):
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
        return False
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
