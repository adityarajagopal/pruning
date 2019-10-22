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

class BasicPruning(object):
#{{{
    def __init__(self, params, model, inferer, valLoader):
        #{{{
        self.params = params
        self.metricValues = []
        self.totalParams = 0
        self.paramsPerLayer = []
        self.channelsToPrune = {}

        self.layers = model._modules['module']._modules
        self.inferer = inferer
        self.valLoader = valLoader

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
                if layerName not in self.masks.keys():
                    self.masks[layerName] = [torch.tensor((), dtype=torch.float32).new_ones(p[1].size()).cuda(device)]
                else:
                    self.masks[layerName].append(torch.tensor((), dtype=torch.float32).new_ones(p[1].size()).cuda(device))
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
            metric = np.square(pNp).reshape(pNp.shape[0], -1).sum(axis=1)
            metric /= (pNp.shape[1]*pNp.shape[2]*pNp.shape[3])
            metric /= np.sqrt(np.square(metric).sum())

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
        
        # perform pruning 
        layersToPrune = self.masks.keys()
        for p in model.named_parameters():
            if 'conv' in p[0]:
                layerName = '.'.join(p[0].split('.')[:-1])
                if layerName in layersToPrune:
                    # print(p[0], layerName, np.count_nonzero((self.masks[layerName][0 if 'weight' in p[0] else 1] == 0).data.cpu().numpy()))
                    p[1].data = p[1].mul(self.masks[layerName][0 if 'weight' in p[0] else 1])
                    p[1].requires_grad = True
      
        return self.channelsToPrune
    #}}}

#{{{
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
#}}}
#}}}

class MobileNetV2Pruning(BasicPruning):
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

        self.channelsToPrune = {}

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
                
                # with dw convs, the initial 1x1 conv and the 3x3 dw need to have the same number of channels
                # so instead of pruning both together, only the final 1x1 which completes the dw conv is pruned
                if 'layers' in layerName and 'conv3' not in layerName:
                        continue
                
                if layerName not in self.masks.keys():
                    self.masks[layerName] = [torch.tensor((), dtype=torch.float32).new_ones(p[1].size()).cuda(device)]
                else:
                    self.masks[layerName].append(torch.tensor((), dtype=torch.float32).new_ones(p[1].size()).cuda(device))
        
        # identify blocks with residuals
        self.resNames = []
        self.layerCount = 0
        for n,m in model.named_modules():
            if 'layers' in n and len(n.split('.')) == 3:
                if m._modules['conv2'].stride[0] == 1:
                    m.register_forward_hook(self.residual_mod)
                    self.resNames.append(n)
    #}}} 
    
    def residual_mod(self, module, input, output):
    #{{{
        if self.channelsToPrune == {}:
            return 

        layerName = self.resNames[self.layerCount] + '.conv3'
        channelsToPrune = self.channelsToPrune[layerName]

        device = 'cuda:' + str(self.params.gpuList[0])
        mask = torch.tensor((), dtype=torch.float32, requires_grad=False).new_ones(output.shape).cuda(device)  
        mask[:,channelsToPrune] = 0.
        output.data = output.mul(mask)

        self.layerCount += 1
        if self.layerCount == len(self.resNames):
            self.layerCount = 0
    #}}}
#}}}

class ResNet20Pruning(BasicPruning):
#{{{
    def __init__(self, params, model, inferer, valLoader):  
    #{{{
        super().__init__(params, model, inferer, valLoader)

        self.layerCount = 0
        self.resNames = []
        for n,m in model.named_modules():
            if 'layer' in n and len(n.split('.')) == 3:
                m.register_forward_hook(self.residual_mod)
                self.resNames.append(n)
    #}}} 

    def residual_mod(self, module, input, output):
    #{{{
        if self.channelsToPrune == {}:
            return 

        layerName = self.resNames[self.layerCount] + '.conv2'
        channelsToPrune = self.channelsToPrune[layerName]

        device = 'cuda:' + str(self.params.gpuList[0])
        mask = torch.tensor((), dtype=torch.float32, requires_grad=False).new_ones(output.shape).cuda(device)  
        mask[:,channelsToPrune] = 0.
        output.data = output.mul(mask)

        self.layerCount += 1
        if self.layerCount == len(self.resNames):
            self.layerCount = 0
    #}}}
#}}}
