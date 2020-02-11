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
import functools
import copy

from abc import ABC, abstractmethod

import src.ar4414.pruning.pruners.dependencies as dependSrc

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
    def __init__(self, params, model):
    #{{{
        self.params = params
        self.model = model

        self.minFiltersInLayer = 2

        self.metricValues = []
        self.channelsToPrune = {}
        self.gpu_list = [int(x) for x in self.params.gpu_id.split(',')]
        
        self.totalParams = 0
        self.layerSizes = {}
        
        self.get_layer_params()

        # create model directory and file
        dirName = 'models/pruned/{}/{}'.format(params.dataset, params.subsetName)
        self.filePath = os.path.join(dirName, self.fileName)
        
        ## create dir if it doesn't exist
        cmd = 'mkdir -p {}'.format(dirName)
        subprocess.check_call(cmd, shell=True)

        self.depBlock = dependSrc.DependencyBlock(model)

        self.importPath = 'src.ar4414.pruning.{}.{}'.format('.'.join(dirName.split('/')), self.fileName.split('.')[0])
    #}}} 
    
    def get_layer_params(self):
    #{{{
        for p in self.model.named_parameters():
            paramsInLayer = 1
            for dim in p[1].size():
                paramsInLayer *= dim
            self.totalParams += paramsInLayer
            
            if self.is_conv_or_fc(p[0]):
                self.layerSizes[p[0]] = list(p[1].size())
        
        self.layersInOrder = list(self.layerSizes.keys())
    #}}}

    def log_pruned_channels(self, rootFolder, params, totalPrunedPerc, channelsPruned): 
    #{{{
        if params.printOnly == True:
            return 
        
        # write pruned channels as json to log folder
        jsonName = os.path.join(rootFolder, 'pruned_channels.json')
        channelsPruned['prunePerc'] = totalPrunedPerc
        summary = {}
        summary[str(params.curr_epoch)] = channelsPruned
        with open(jsonName, 'w') as sumFile:
            json.dump(summary, sumFile)
        
        # copy pruned network description to log folder
        cmd = "cp {} {}/pruned_model.py".format(self.filePath, rootFolder)
        subprocess.check_call(cmd, shell=True)

        return summary
    #}}} 

    def get_random_init_model(self, finetunePath):    
    #{{{
        importPath = finetunePath.split('/')
        baseIdx = importPath.index('src')
        importPath = importPath[baseIdx:]
        importPath.append('pruned_model')
        self.importPath = '.'.join(importPath)
        prunedModel = self.import_pruned_model()
        optimiser = torch.optim.SGD(prunedModel.parameters(), lr=self.params.lr, momentum=self.params.momentum, weight_decay=self.params.weight_decay)
        return prunedModel, optimiser
    #}}}

    def import_pruned_model(self):
    #{{{
        module = importlib.import_module(self.importPath)
        pModel = module.__dict__[self.netName]
        prunedModel = pModel(num_classes=100)
        prunedModel = torch.nn.DataParallel(prunedModel, self.gpu_list).cuda()
        return prunedModel
    #}}}
    
    def prune_model(self, model, transferWeights=True):
    #{{{
        if self.params.pruneFilters == True: 
            # pruning based on l1 norm of weights
            if self.params.pruningMetric == 'weights':
                tqdm.write("Pruning filters - Weights")
                channelsPruned = self.structured_l1_weight(model)
                self.write_net()
                prunedModel = self.import_pruned_model()
                # prunedModel = self.transfer_weights(model, prunedModel)
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
        # remove 1 output filter from current layer
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
            # remove 1 input activation from next layer
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
    
    def calculate_metric(self, param):
    #{{{
        #l1-norm
        metric = np.absolute(param).reshape(param.shape[0], -1).sum(axis=1)
        metric /= (param.shape[1]*param.shape[2]*param.shape[3])
        return metric
    #}}}

    def rank_filters(self, model):
    #{{{
        localRanking = {} 
        globalRanking = []
        
        # create global ranking
        for p in model.named_parameters():
        #{{{
            if self.prune_layer(p[0]):
                layerName = '.'.join(p[0].split('.')[:-1])
                if self.skip_layer(layerName):
                    continue
            
                pNp = p[1].data.cpu().numpy()
            
                # calculate metric
                metric = self.calculate_metric(pNp)

                globalRanking += [(layerName, i, x) for i,x in enumerate(metric)]
                localRanking[layerName] = sorted([(i, x) for i,x in enumerate(metric)], key=lambda tup:tup[1])
        #}}}
        
        globalRanking = sorted(globalRanking, key=lambda i: i[2]) 
        self.channelsToPrune = {l:[] for l,m in model.named_modules() if isinstance(m, nn.Conv2d)}

        return localRanking, globalRanking
    #}}}

    def remove_filters(self, localRanking, globalRanking, dependencies, groupPruningLimits):
    #{{{
        currentPruneRate = 0
        listIdx = 0
        while (currentPruneRate < self.params.pruningPerc) and (listIdx < len(globalRanking)):
            layerName, filterNum, _ = globalRanking[listIdx]

            depLayers = []
            pruningLimit = self.minFiltersInLayer
            for i, group in enumerate(dependencies):
                if layerName in group:            
                    depLayers = group
                    pruningLimit = groupPruningLimits[i]
                    break
            
            # if layer not in group, just remove filter from layer 
            # if layer is in a dependent group remove corresponding filter from each layer
            depLayers = [layerName] if depLayers == [] else depLayers
            for layerName in depLayers:
                if len(localRanking[layerName]) <= pruningLimit:
                    listIdx += 1
                    continue
            
                # if filter has already been pruned, continue
                # could happen to due to dependencies
                if filterNum in self.channelsToPrune[layerName]:
                    listIdx += 1
                    continue 
                
                localRanking[layerName].pop(0)
                self.channelsToPrune[layerName].append(filterNum)
                
                currentPruneRate += self.inc_prune_rate(layerName) 
            
            listIdx += 1

        return self.channelsToPrune
    #}}}
    
    def structured_l1_weight(self, model):
    #{{{
        localRanking, globalRanking = self.rank_filters(model)
        
        internalDeps, externalDeps = self.depBlock.get_dependencies()
        
        numChannelsInDependencyGroup = [len(localRanking[k[0]]) for k in externalDeps]
        if self.params.pruningPerc >= 50.0:
            groupPruningLimits = [int(math.ceil(gs * (1.0 - self.params.pruningPerc/100.0))) for gs in numChannelsInDependencyGroup]
        else:
            groupPruningLimits = [int(math.ceil(gs * self.params.pruningPerc/100.0)) for gs in numChannelsInDependencyGroup]

        dependencies = internalDeps + externalDeps
        groupPruningLimits = [2]*len(internalDeps) + groupPruningLimits
        
        self.remove_filters(localRanking, globalRanking, dependencies, groupPruningLimits)

        return self.channelsToPrune
    #}}}

    # selects only convs and fc layers 
    # used in get_layer_params to get sizes of only convs and fcs 
    # lParam is from loop over named_parameters()
    @abstractmethod
    def is_conv_or_fc(self, lParam): 
        pass

    # function that selects all the layers to be pruned (currently only convs)
    # used in structure_l1_weight to go through layers that are to be pruned
    # lParam is from loop over named_parameters()
    @abstractmethod
    def prune_layer(self, lParam):
        pass
        
    # function that specifies conv layers to skip when pruning
    # used in structure_l1_weight
    # lName here is module name, so doesn't have 'weight'/'bias' keyword
    @abstractmethod
    def skip_layer(self, lName):
        pass
    
    @abstractmethod
    def write_net(self, subsetName=None):
        pass
    
    @abstractmethod
    def transfer_weights(self, oModel, pModel): 
        pass
#}}}
