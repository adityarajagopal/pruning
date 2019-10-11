import sys
import csv
import os
import numpy as np
import time
import torch
from tqdm import tqdm
import json
import pickle

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

class PruningHook(object):
    #{{{
    def __init__(self, module):
        self.channels = []
        module.register_forward_hook(self.prune)
    
    def update_channels(self, channel):
        self.channels.append(channel)

    def prune(self, module, input, output):
        if self.channels != []:
            output[:,self.channels] = 0
    #}}}

class BasicPruning(object):
    #{{{
    def __init__(self, params, model, inferer, valLoader):
        #{{{
        self.params = params
        self.metricValues = []
        self.totalParams = 0
        self.paramsPerLayer = []

        self.layers = model._modules['module']._modules
        self.inferer = inferer
        self.valLoader = valLoader
        self.numLayers = 0

        self.masks = {}
        self.hooks = {}
        for n,m in model.named_modules():
            if isinstance(m, nn.Conv2d):
                # self.masks[n] = []
                # self.hooks.append(PruningHook(m)) 
                self.hooks[n] = PruningHook(m)
        
        for p in model.named_parameters():
            paramsInLayer = 1
            for dim in p[1].size():
                paramsInLayer *= dim
            self.paramsPerLayer.append(paramsInLayer)
            self.totalParams += paramsInLayer
            
            device = 'cuda:' + str(self.params.gpuList[0])
            if 'conv' in p[0] and 'weight' in p[0]:
                layerName = '.'.join(p[0].split('.')[:-1])
                # self.masks.append(torch.tensor((), dtype=torch.float32).new_ones(p[1].size()).cuda(device))
                self.masks[layerName] = torch.tensor((), dtype=torch.float32).new_ones(p[1].size()).cuda(device)
                self.numLayers += 1
        #}}} 
    
    def log_prune_rate(self, rootFolder, params, totalPrunedPerc): 
        #{{{
        if params.printOnly == True:
            return 
        csvName = os.path.join(rootFolder, 'layer_prune_rate.csv')
        with open(csvName, 'a') as csvFile:
            writer = csv.writer(csvFile, delimiter=',')
            writer.writerow([params.curr_epoch] + params.prunePercPerLayer + [totalPrunedPerc])
        #}}} 
    
    def prune_model(self, model):
        #{{{
        if self.params.pruneFilters == True: 
            # pruning based on l2 norm of weights
            if self.params.pruningMetric == 'weights':
                tqdm.write("Pruning filters - Weights")
                return self.structured_l2_weight(model)
            
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
        # prunedParamsPerLayer = []
        prunedParamsPerLayer = {}

        # if self.masks == []:
        #     return 0.
        # for mask in self.masks:
        #     prunedParamsPerLayer.append(np.count_nonzero((mask == 0).data.cpu().numpy()))
        if self.masks == {}:
            return 0.
        for layer, mask in self.masks.items():
            prunedParamsPerLayer[layer] = np.count_nonzero((mask == 0).data.cpu().numpy())
        
        layerNum = 0
        convNum = 0
        for p in model.named_parameters():
            if 'conv' in p[0] and 'weight' in p[0]:
                layerName = '.'.join(p[0].split('.')[:-1])
                # prunedParams = prunedParamsPerLayer[convNum]  
                prunedParams = prunedParamsPerLayer[layerName]  
                totalPrunedParams += prunedParams
                if verbose:
                    self.params.prunePercPerLayer.append((prunedParams / self.paramsPerLayer[layerNum]))
                convNum += 1
            layerNum += 1
        
        return 100.*(totalPrunedParams/self.totalParams) 
        #}}}        
    
    def structured_l2_weight(self, model):
        #{{{
        layerNum = 0
        self.metricValues = []
        for p in model.named_parameters():
            if 'conv' in p[0] and 'weight' in p[0]:
                if layerNum < self.params.thisLayerUp:
                    layerNum += 1
                    continue
                
                layerName = '.'.join(p[0].split('.')[:-1])
                pNp = p[1].data.cpu().numpy()
                
                # calculate metric
                metric = np.square(pNp).reshape(pNp.shape[0], -1).sum(axis=1)
                metric /= (pNp.shape[1]*pNp.shape[2]*pNp.shape[3])
                metric /= np.sqrt(np.square(metric).sum())

                # calculte incremental prune percentage of pruning filter
                incPrunePerc = 100.*(pNp.shape[1]*pNp.shape[2]*pNp.shape[3]) / self.totalParams
                
                # store calculated values to be sorted by l2 norm mag and used later
                # values = [(layerNum, i, x, incPrunePerc) for i,x in enumerate(metric) if not np.all((self.masks[layerNum][i] == 0.).data.cpu().numpy())]
                values = [(layerName, i, x, incPrunePerc) for i,x in enumerate(metric) if not np.all((self.masks[layerName][i] == 0.).data.cpu().numpy())]
                self.metricValues += values
                layerNum += 1

        self.metricValues = sorted(self.metricValues, key=lambda tup: tup[2])

        channelsToPrune = {l:[] for l,m in model.named_modules() if isinstance(m, nn.Conv2d)}

        # currentPruneRate = self.prune_rate(model)
        currentPruneRate = 0
        listIdx = 0
        while (currentPruneRate < self.params.pruningPerc) and (listIdx < len(self.metricValues)):
            filterToPrune = self.metricValues[listIdx]
            # layerNum = filterToPrune[0]
            layerName = filterToPrune[0]
            filterNum = filterToPrune[1]
            
            # self.masks[layerNum][filterNum] = 0.
            # self.hooks[layerNum].update_channels(filterNum)
            # channelsToPrune['module.conv'+str(layerNum+1)].append(filterNum)
            self.masks[layerName][filterNum] = 0.
            self.hooks[layerName].update_channels(filterNum)
            channelsToPrune[layerName].append(filterNum)

            currentPruneRate += filterToPrune[3]
            listIdx += 1
      
        return model, channelsToPrune
        #}}}

    def structured_activations(self, model):
        #{{{
        # potentially have to change finetuning structure to prune single feature map each iteration
        # and alternate between finetuning and pruning till desired sparsity is reached
        # will take a lot more iterations

        handles = []
        for k,v in self.layers.items():
            if 'Conv' in str(v):
                handles.append(v.register_forward_hook(self.mean_activations))
        
        incPrunePerc = []
        for p in model.named_parameters(): 
            if 'conv' in p[0] and 'weight' in p[0]:
                pNp = p[1].data.cpu().numpy()
                tmp = 100.*(pNp.shape[1]*pNp.shape[2]*pNp.shape[3]) / self.totalParams
                incPrunePerc.append(tmp)

        numConvLayers = len(incPrunePerc)
        numBatches = 0
        self.meanActivations = {i:[] for i in range(numConvLayers)}
                
        for batchIdx, (inputs, targets) in enumerate(self.valLoader):
            self.layerNum = 0
            numBatches = len(targets)
            self.inferer.run_single_forward(self.params, inputs, targets, model)
        
        layerNum = 0
        for p in model.named_parameters(): 
            if 'conv' in p[0] and 'weight' in p[0]:
                act = self.meanActivations[layerNum]
                act /= numBatches
                magAct = act.shape[1] * act.shape[2] 

                metric = act.reshape(act.shape[0], -1).sum(axis=1)
                metric /= magAct
                metric /= np.sqrt(np.square(metric).sum())

                self.meanActivations[layerNum] = metric
                
                layerNum += 1
        
        meanActivationsByFilter = []
        for layerNum, x in self.meanActivations.items():
            # x /= numBatches
            meanActivationsByFilter += [(int(layerNum), int(filterNum), float(np.abs(act))) for filterNum, act in enumerate(x) if not np.all((self.masks[layerNum][filterNum] == 0.).data.cpu().numpy())]
                        
        meanActivationsByFilter = sorted(meanActivationsByFilter, key=lambda tup:tup[2])
        
        currentPruneRate = self.prune_rate(model)

        listIdx = 0
        while currentPruneRate < self.params.pruningPerc:
            filterToPrune = meanActivationsByFilter[listIdx]
            layerNum = filterToPrune[0]
            filterNum = filterToPrune[1]
            self.masks[layerNum][filterNum] = 0.
            model.module.set_masks(self.masks)
            currentPruneRate += incPrunePerc[layerNum] 
            listIdx += 1
        
        [handle.remove() for handle in handles]
        
        return model
        #}}}

    def mean_activations(self, module, input, output):
        #{{{
        # try averaging activations and then performing metric calculation at the end instead 

        outNp = output[0].data.cpu().numpy()
        # magAct = outNp.shape[1] * outNp.shape[2]

        # metric = outNp.reshape(outNp.shape[0], -1).sum(axis=1)
        # metric /= magAct
        # metric /= np.sqrt(np.square(metric).sum())

        if self.meanActivations[self.layerNum] == []:
            # self.meanActivations[self.layerNum] = metric
            self.meanActivations[self.layerNum] = outNp
        else:
            # self.meanActivations[self.layerNum] += metric 
            self.meanActivations[self.layerNum] += outNp

        self.layerNum += 1
        #}}}



            
        
                


