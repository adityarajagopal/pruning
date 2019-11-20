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

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

class FBSPruning(object):
    #{{{
    def __init__(self, params, model):
        #{{{
        self.params = params
        self.layers = model._modules['module']._modules
        self.channelProbs = {}
        
        numConvLayers = 0
        for k,v in self.layers.items():
            if 'conv' in k:
                self.channelProbs['conv' + str(numConvLayers)] = {} 
            numConvLayers += 1
        
        numConvLayers = 0
        for x in model.named_parameters():
            if 'conv.weight' in x[0]:
                name = 'conv' + str(numConvLayers)
                outChannels = x[1].shape[0]
                self.channelProbs[name] = [0.0 for x in range(outChannels)] 
                numConvLayers += 1
        #}}} 
    
    def prune_model(self, model):
        #{{{
        for k,v in self.layers.items():
            if 'conv' in k:
                v.unprunedRatio = self.params.unprunedRatio
        return model
        #}}} 
    
    def prune_rate(self, model, verbose=False):
        return self.params.unprunedRatio
    
    def log_prune_rate(self, rootFolder, params): 
        #{{{
        if params.printOnly == True:
            return 

        prunePerc = '{:2.1f}'.format(1.0 - params.unprunedRatio).replace('.','_')
        fileName = 'prunePerc_' + prunePerc + '_channels_by_layer.json'
        jsonName = os.path.join(rootFolder, fileName)
        with open(jsonName, 'w') as jsonFile:
            json.dump(self.channelProbs, jsonFile)
        #}}}
    #}}}

class BasicPruning(ABC):
#{{{
    def __init__(self, params, model, layerSkip=(lambda lName : False)):
        #{{{
        self.params = params
        self.model = model
        
        self.metricValues = []
        self.channelsToPrune = {}
        self.gpu_list = [int(x) for x in self.params.gpu_id.split(',')]
        
        self.layerSkip = layerSkip

        self.totalParams = 0
        self.layerSizes = {}
        self.get_layer_params()

        # create model directory and file
        dirName = 'models/pruned/{}/{}'.format(params.dataset, params.subsetName)
        self.filePath = os.path.join(dirName, self.fileName)
        
        ## create dir if it doesn't exist
        cmd = 'mkdir -p {}'.format(dirName)
        subprocess.check_call(cmd, shell=True)
        
        self.importPath = 'src.ar4414.pruning.{}.{}'.format('.'.join(dirName.split('/')), self.fileName.split('.')[0])
        #}}} 

    def log_pruned_channels(self, rootFolder, params, totalPrunedPerc, channelsPruned): 
        #{{{
        if params.printOnly == True:
            return 
        
        jsonName = os.path.join(rootFolder, 'pruned_channels.json')
        channelsPruned['prunePerc'] = totalPrunedPerc
        summary = {}
        summary[str(params.curr_epoch)] = channelsPruned

        with open(jsonName, 'w') as sumFile:
            json.dump(summary, sumFile)
        
        return summary
        #}}} 

    def import_pruned_model(self):
    #{{{
        pModel = importlib.import_module(self.importPath).__dict__[self.netName]
        prunedModel = pModel(num_classes=100)
        prunedModel = torch.nn.DataParallel(prunedModel, self.gpu_list).cuda()

        return prunedModel
    #}}}
    
    def prune_model(self, model):
        #{{{
        if self.params.pruneFilters == True: 
            # pruning based on l1 norm of weights
            if self.params.pruningMetric == 'weights':
                tqdm.write("Pruning filters - Weights")
                channelsPruned = self.structured_l1_weight(model)

                # perform pruning by writing out pruned network
                self.gpu = 'cuda:{}'.format(self.gpu_list[0])
                self.write_net()
                
                prunedModel = self.import_pruned_model()
                
                prunedModel = self.transfer_weights(model, prunedModel)
                optimiser = torch.optim.SGD(prunedModel.parameters(), lr=self.params.lr, momentum=self.params.momentum, weight_decay=self.params.weight_decay)

                return channelsPruned, prunedModel, optimiser
            
            # pruning based on activations 
            if self.params.pruningMetric == 'activations':
                tqdm.write("Pruning filters - Mean Activation")
                return self.structured_activations(model)
        #}}}
        
    def non_zero_argmin(self, array): 
        minIdx = np.argmin(array[np.nonzero(array)]) 
        return (minIdx, array[minIdx])     
    
    def inc_prune_rate(self, layerName):
    #{{{
        lParam = str(layerName) + '.weight'
        self.layerSizes[lParam][0] -= 1 

        nextLayerName = self.layersInOrder[self.layersInOrder.index(lParam) + 1]
        nextLayerSize = self.layerSizes[nextLayerName]
        currLayerSize = self.layerSizes[lParam]
        paramsPruned = currLayerSize[1]*currLayerSize[2]*currLayerSize[3]
        # check if FC layer
        if len(nextLayerSize) == 2: 
            paramsPruned += nextLayerSize[0]
        else:
            paramsPruned += nextLayerSize[0]*nextLayerSize[2]*nextLayerSize[3]
            self.layerSizes[nextLayerName][1] -= 1
        
        return (100.* paramsPruned / self.totalParams)
    #}}}
    
    def prune_rate(self, pModel):
    #{{{
        prunedParams = 0
        for p in pModel.named_parameters():
            params = 1
            for dim in p[1].size():
                params *= dim 
            prunedParams += params
        
        return 100.*((self.totalParams - prunedParams) / self.totalParams), (prunedParams * 4) / 1e6, (self.totalParams * 4)/1e6
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
        
        self.layersInOrder = list(self.layerSizes.keys())
    #}}}
    
    @abstractmethod
    def structured_l1_weight(self, model):
        pass

    @abstractmethod
    def write_net(self, subsetName=None):
        pass
    
    @abstractmethod
    def transfer_weights(self, oModel, pModel): 
        pass
#}}}














