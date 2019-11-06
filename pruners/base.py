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
        self.totalParams = 0
        self.paramsPerLayer = []
        self.channelsToPrune = {}
                
        self.gpu_list = [int(x) for x in self.params.gpu_id.split(',')]

        self.masks = {}
        for p in model.named_parameters():
            paramsInLayer = 1
            for dim in p[1].size():
                paramsInLayer *= dim
            self.paramsPerLayer.append(paramsInLayer)
            self.totalParams += paramsInLayer

            device = 'cuda:' + str(self.params.gpuList[0])
            if 'conv' in p[0]:
                layerName = '.'.join(p[0].split('.')[:-1])
                
                if layerSkip(layerName):
                    continue

                if layerName not in self.masks.keys():
                    self.masks[layerName] = [torch.tensor((), dtype=torch.float32).new_ones(p[1].size()).cuda(device)]
                else:
                    self.masks[layerName].append(torch.tensor((), dtype=torch.float32).new_ones(p[1].size()).cuda(device))
        
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
            # pruning based on l2 norm of weights
            if self.params.pruningMetric == 'weights':
                tqdm.write("Pruning filters - Weights")
                channelsPruned = self.structured_l2_weight(model)

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
    
    def prune_rate(self, model, verbose=False):
    #{{{
        totalPrunedParams = 0
        totalPrunedParams1 = 0
        prunedParamsPerLayer = {}

        if self.masks == {}:
            return 0.
        for layer, mask in self.masks.items():
            for x in mask:
                if layer not in prunedParamsPerLayer.keys():
                    prunedParamsPerLayer[layer] = np.count_nonzero((x == 0).data.cpu().numpy())
                else:
                    prunedParamsPerLayer[layer] += np.count_nonzero((x == 0).data.cpu().numpy())
            totalPrunedParams += prunedParamsPerLayer[layer]
        
        return 100.*(totalPrunedParams/self.totalParams) 
    #}}}        

    def structured_l2_weight(self, model):
    #{{{
        self.metricValues = []
        
        namedParams = dict(model.named_parameters())
        for layerName, mask in self.masks.items():
            pNp = namedParams[layerName + '.weight'].data.cpu().numpy()
            
            # calculate metric
            
            # l2-norm
            # metric = np.square(pNp).reshape(pNp.shape[0], -1).sum(axis=1)
            # metric /= (pNp.shape[1]*pNp.shape[2]*pNp.shape[3])
            # metric /= np.sqrt(np.square(metric).sum())

            #l1-norm
            metric = np.absolute(pNp).reshape(pNp.shape[0], -1).sum(axis=1)
            metric /= (pNp.shape[1]*pNp.shape[2]*pNp.shape[3])

            # calculte incremental prune percentage of pruning filter
            incPrunePerc = 100.*(pNp.shape[1]*pNp.shape[2]*pNp.shape[3]) / self.totalParams
            
            # store calculated values to be sorted by l2 norm mag and used later
            values = [(layerName, i, x, incPrunePerc) for i,x in enumerate(metric) if not np.all((mask[0][i] == 0.).data.cpu().numpy())]
            self.metricValues += values
        
        self.metricValues = sorted(self.metricValues, key=lambda tup: tup[2])

        self.channelsToPrune = {l:[] for l,m in model.named_modules() if isinstance(m, nn.Conv2d)}

        currentPruneRate = 0
        listIdx = 0
        while (currentPruneRate < self.params.pruningPerc) and (listIdx < len(self.metricValues)):
            filterToPrune = self.metricValues[listIdx]
            layerName = filterToPrune[0]
            filterNum = filterToPrune[1]
            
            for x in self.masks[layerName]:
                x[filterNum] = 0.
            self.channelsToPrune[layerName].append(filterNum)

            currentPruneRate += filterToPrune[3]
            listIdx += 1
        
        return self.channelsToPrune
    #}}}

    @abstractmethod
    def write_net(self, subsetName=None):
        pass
    
    @abstractmethod
    def transfer_weights(self, oModel, pModel): 
        pass
#}}}














