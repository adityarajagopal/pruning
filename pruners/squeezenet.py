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
    
    # def inc_prune_rate(self, layerName):
    # #{{{
    #     lParam = str(layerName) + '.weight'
    #     self.layerSizes[lParam][0] -= 1 

    #     nextLayerName = self.layersInOrder[lParam]
    #     nextLayerSize = [self.layerSizes[n] for n in nextLayerName]
    #     currLayerSize = self.layerSizes[lParam]
    #     paramsPruned = currLayerSize[1]*currLayerSize[2]*currLayerSize[3]
    #     # check if FC layer
    #     if nextLayerName == ['module.conv2.weight']: 
    #         paramsPruned += nextLayerSize[0][0]
    #     else:
    #         for i,nLS in enumerate(nextLayerSize):
    #             paramsPruned += nLS[0]*nLS[2]*nLS[3] 
    #             nLN = nextLayerName[i]
    #             self.layerSizes[nLN][1] -= 1
    #     
    #     return (100.* paramsPruned / self.totalParams)
    # #}}}
     
    # def get_layer_params(self):
    # #{{{
    #     for p in self.model.named_parameters():
    #         paramsInLayer = 1
    #         for dim in p[1].size():
    #             paramsInLayer *= dim
    #         self.totalParams += paramsInLayer
    #         
    #         if self.convs_and_fcs(p[0]):
    #             self.layerSizes[p[0]] = list(p[1].size())
    #     
    #     # construct layers in order to have the fire.conv1 as next layer after both fire.conv2 and fire.conv3
    #     # since concatenation occurs 
    #     self.layersInOrder = list(self.layerSizes.keys())
    #     newOrder = {}
    #     for i,l in enumerate(self.layersInOrder):
    #         if 'fire' not in l:
    #             if 'conv1' in l:
    #                 newOrder[l] = [self.layersInOrder[i+1]]
    #         elif 'fire' in l:
    #             if 'conv1' in l:
    #                 newOrder[l] = [self.layersInOrder[i+1], self.layersInOrder[i+2]]
    #             elif 'conv2' in l:
    #                 newOrder[l] = [self.layersInOrder[i+2]]
    #             elif 'conv3' in l:
    #                 newOrder[l] = [self.layersInOrder[i+1]]
    #     
    #     self.layersInOrder = newOrder
    # #}}}

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
#}}}

