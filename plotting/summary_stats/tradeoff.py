import os
import sys
import math
import json
import argparse
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

import src.ar4414.pruning.plotting.summary_stats.collector as collector

def get_tradeoff_points(binSearchResults, inferenceTimes, unprunedData, pruneAfter): 
#{{{
    nets = list(inferenceTimes.keys())
    subsets = list(binSearchResults[nets[0]].keys())
    
    data = {net:{subset:None for subset in subsets} for net in nets}

    for net in nets:
        for subset in subsets:
            dataPoints = pd.DataFrame({'Type':[], 'TestAcc':[], 'Time':[], 'Metric':[]})
            
            searchCost = binSearchResults[net][subset]['cost'].drop(columns=['Gops'])
            searchEpochData = binSearchResults[net][subset]['epoch_data']
            
            # get values for unpruned network
            unprunedStats = unprunedData.loc[(unprunedData['Network'] == net) & (unprunedData['Subset'] == subset)] 
            targetAcc = int(unprunedStats['TestAcc'])
            baseMetric = float(inferenceTimes[net]['0'])

            # get points which match or exceed original accuracy
            validPoints = searchCost.loc[searchCost.index > pruneAfter-1].loc[searchCost['TestAcc'] >= targetAcc]
            validEpochs = validPoints.index.values.tolist()
            prunePercs = [list(filter(lambda x : x[1] >= epoch, searchEpochData))[0][0] for epoch in validEpochs]  
            uniquePp = []
            uniqueValidEpochs = []
            for idx,pp in enumerate(prunePercs): 
                if pp not in uniquePp:
                    uniquePp.append(pp)
                    uniqueValidEpochs.append(validEpochs[idx])
            
            dp = validPoints.loc[validPoints.index.isin(uniqueValidEpochs)]
            inference = [inferenceTimes[net][str(pp)] for pp in uniquePp]
            metric = []
            types = []
            for idx,row in dp.reset_index().iterrows(): 
                metric.append(float(inference[idx]))
                types.append('valid')
            dp['Metric'] = metric
            dp['Type'] = types
            
            basePoint = {'Type':'base', 'TestAcc':targetAcc, 'Time':0, 'Metric':baseMetric}
            dp = dp.append(basePoint, ignore_index=True)

            #{{{
            # get best accuracy per precision searched
            # bestAccs = []
            # searchTime = []
            # metric = []
            # types = []
            # excludeValidPoints = searchCost.loc[~(searchCost.index.isin(validEpochs)) & (searchCost.index > pruneAfter-1)]
            # invalidEpochs = excludeValidPoints.index.values.tolist()
            # initEpoch = invalidEpochs[0]
            # for pp,epoch in searchEpochData: 
            #     if initEpoch >= epoch:
            #         continue
            #     
            #     indices = list(filter(lambda x : x >= initEpoch and x < epoch, invalidEpochs))   
            #     infTime = inferenceTimes[net][str(pp)]
            #     subPoints = excludeValidPoints.loc[excludeValidPoints.index.isin(indices)]
            #     
            #     if len(subPoints['TestAcc']) == 0:
            #         initEpoch == epoch
            #         continue
            #     
            #     bestAcc = subPoints['TestAcc'].max()
            #     bestAccIdx = subPoints['TestAcc'].idxmax()
            #     
            #     types.append('invalid')
            #     bestAccs.append(bestAcc)
            #     searchTime.append(subPoints['Time'][bestAccIdx])
            #     metric.append(float(infTime))

            #     initEpoch = epoch
            # 
            # invalidPoints = pd.DataFrame({'Type':types, 'TestAcc':bestAccs, 'Time':searchTime, 'Metric':metric})
            # dp = dp.append(invalidPoints)
            #}}}

            dataPoints = dataPoints.append(dp)
            data[net][subset] = dataPoints
    return data
#}}}

def plot_tradeoff(data, saveLoc=None): 
#{{{
    nets = list(data.keys()) 
    subsets = list(data[nets[0]].keys())
    for net in nets: 
        for subset in subsets:
            ax = plt.subplots(1,1)
            ax[1].set_title("Search Time to Inference Time tradeoff for {}-{}".format(net.capitalize(),subset.capitalize()))
            ax[1].set_xlabel("Search Time (s)")
            ax[1].set_ylabel("Inference Time for searched model (s)")
            for ptType, pts in data[net][subset].groupby(['Type']): 
                colour = 'blue' if ptType == 'base' else 'red' if ptType == 'invalid' else 'green'
                label = 'No Pruning' if ptType == 'base' else 'Pruned Model'
                ax[1].scatter(pts['Time'], pts['Metric'], color=colour, label=label)
                ax[1].legend()
            
            if saveLoc is not None: 
                plt.tight_layout()
                figFile = os.path.join(saveLoc, '{}_{}.png'.format(net, subset))
                plt.savefig(figFile)
#}}}









