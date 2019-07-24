import sys
import csv
import numpy as np
import time
import torch
from tqdm import tqdm

class Pruning(object):
    def __init__(self, params, model):
        self.params = params
        self.metricValues = []
        self.masks = []
        self.totalParams = 0
        self.paramsPerLayer = []

        for p in model.named_parameters():
            paramsInLayer = 1
            for dim in p[1].size():
                paramsInLayer *= dim
            self.paramsPerLayer.append(paramsInLayer)
            self.totalParams += paramsInLayer
            
            device = 'cuda:' + str(self.params.gpuList[0])
            if 'conv' in p[0] and 'weight' in p[0]:
                self.masks.append(torch.tensor((), dtype=torch.float32).new_ones(p[1].size()).cuda(device))
        
    def prune_model(self, model):
        if self.params.pruneFilters == True: 
            tqdm.write("Pruning filters")
            return self.structured_l2_weight(model)

    def non_zero_argmin(self, array): 
        minIdx = np.argmin(array[np.nonzero(array)]) 
        return (minIdx, array[minIdx])     
    
    def prune_rate(self, model, verbose=False):
        totalPrunedParams = 0
        prunedParamsPerLayer = []

        if self.masks == []:
            return 0.
        
        for mask in self.masks:
            prunedParamsPerLayer.append(np.count_nonzero((mask == 0).data.cpu().numpy()))
        
        # modules = model.module.__dict__['_modules']
        # for k,v in modules.items():
        #     if 'conv' in k:
        #         if v.mask_flag == False:
        #             return 0
        #         mask = v.get_mask()
        #         prunedParamsPerLayer.append(np.count_nonzero((mask == 0).data.cpu().numpy()))
        
        layerNum = 0
        convNum = 0
        for p in model.named_parameters():
            if 'conv' in p[0] and 'weight' in p[0]:
                prunedParams = prunedParamsPerLayer[convNum]  
                totalPrunedParams += prunedParams
                if verbose:
                    self.params.prunePercPerLayer.append((prunedParams / self.paramsPerLayer[layerNum]))
                convNum += 1
            layerNum += 1
        
        return 100.*(totalPrunedParams/self.totalParams) 
                
    def structured_l2_weight(self, model):
        layerNum = 0
        self.metricValues = []
        for p in model.named_parameters():
            if 'conv' in p[0] and 'weight' in p[0]:
                if layerNum < self.params.thisLayerUp:
                    layerNum += 1
                    continue
                pNp = p[1].data.cpu().numpy()
                
                # calculate metric
                metric = np.square(pNp).reshape(pNp.shape[0], -1).sum(axis=1)
                metric /= (pNp.shape[1]*pNp.shape[2]*pNp.shape[3])
                metric /= np.sqrt(np.square(metric).sum())

                # calculte incremental prune percentage of pruning filter
                incPrunePerc = 100.*(pNp.shape[1]*pNp.shape[2]*pNp.shape[3]) / self.totalParams
                
                # store calculated values to be sorted by l2 norm mag and used later
                values = [(layerNum, i, x, incPrunePerc) for i,x in enumerate(metric) if not np.all((self.masks[layerNum][i] == 0.).data.cpu().numpy())]
                self.metricValues += values
                layerNum += 1

        self.metricValues = sorted(self.metricValues, key=lambda tup: tup[2])

        currentPruneRate = self.prune_rate(model)
        listIdx = 0
        while currentPruneRate < self.params.pruningPerc:
            filterToPrune = self.metricValues[listIdx]
            layerNum = filterToPrune[0]
            filterNum = filterToPrune[1]
            self.masks[layerNum][filterNum] = 0.
            model.module.set_masks(self.masks)
            currentPruneRate += filterToPrune[3]
            listIdx += 1
        
        return model

            
        
                


