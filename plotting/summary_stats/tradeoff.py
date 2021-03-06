import os
import sys
import math
import json
import argparse
import itertools
from operator import itemgetter

import numpy as np
import pandas as pd
from matplotlib import rc 
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from src.ar4414.pruning.plotting.config import datasetTranslate
import src.ar4414.pruning.plotting.summary_stats.collector as collector

def get_data_valid_points(net, validPoints, searchEpochData, inferenceTimes, inferenceGops, ptType): 
#{{{
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
    gops = [float(inferenceGops.loc[inferenceGops['PrunePerc'] == pp]['InferenceGops']) for pp in uniquePp]
    metric = []
    types = []
    infGops = []
    for idx,row in dp.reset_index().iterrows(): 
        metric.append(float(inference[idx]))
        infGops.append(gops[idx])
        types.append(ptType)
    dp['Metric'] = metric
    dp['InferenceGops'] = infGops
    dp['Type'] = types
    dp['PruningLevel'] = uniquePp
    
    return dp
#}}}

def get_tradeoff_points(subsetAwareData, binSearchResults, inferenceTimes, unprunedData, pruneAfter): 
#{{{
    nets = list(inferenceTimes.keys())
    subsets = list(binSearchResults[nets[0]].keys())
    
    data = {net:{subset:None for subset in subsets} for net in nets}

    for net in nets:
        for subset in subsets:
            dataPoints = pd.DataFrame({'Type':[], 'PruningLevel':[], 'TestAcc':[], 'Time':[], 'Metric':[], 'InferenceGops':[]})
            
            searchCost = binSearchResults[net][subset]['cost'].drop(columns=['Gops'])
            searchEpochData = binSearchResults[net][subset]['epoch_data']
            
            # get values for unpruned network
            unprunedStats = unprunedData.loc[(unprunedData['Network'] == net) & (unprunedData['Subset'] == datasetTranslate[subset])] 
            targetAcc = float(unprunedStats['TestAcc'])
            baseMetric = float(inferenceTimes[net]['0'])
            infGops = subsetAwareData.loc[(subsetAwareData['Network'] == net) & (subsetAwareData['Dataset'] == datasetTranslate[subset])]            

            # get points which match or exceed original accuracy
            validPoints = searchCost.loc[searchCost.index > pruneAfter-1].loc[searchCost['TestAcc'] >= targetAcc]
            dpValid = get_data_valid_points(net, validPoints, searchEpochData, inferenceTimes, infGops, '>= Unpruned Network')
            otherDatapoints = []
            # get points which drop by <1%
            # condition1 = (searchCost['TestAcc'] < targetAcc) & (searchCost['TestAcc'] >= (targetAcc-1.0))
            # validPoints = searchCost.loc[searchCost.index > pruneAfter-1].loc[condition1]
            # otherDatapoints.append(get_data_valid_points(net, validPoints, searchEpochData, inferenceTimes, 'Error < 1%'))
            
            # get points which drop by <2%
            # condition2 = (searchCost['TestAcc'] < (targetAcc-1.0)) & (searchCost['TestAcc'] >= (targetAcc-2.0))
            # validPoints = searchCost.loc[searchCost.index > pruneAfter-1].loc[condition2]
            # otherDatapoints.append(get_data_valid_points(net, validPoints, searchEpochData, inferenceTimes, 'Error < 2%'))
            
            # get points which drop by <4%
            # condition3 = (searchCost['TestAcc'] < (targetAcc-2.0)) & (searchCost['TestAcc'] >= (targetAcc-4.0)) 
            # validPoints = searchCost.loc[searchCost.index > pruneAfter-1].loc[condition3]
            # otherDatapoints.append(get_data_valid_points(net, validPoints, searchEpochData, inferenceTimes, 'Error < 4%'))
            
            dp = dpValid
            for x in otherDatapoints:
                dp = dp.append(x)
            
            basePoint = {'Type':'Unpruned', 'TestAcc':targetAcc, 'Time':0, 'Metric':baseMetric, 'InferenceGops':float(unprunedStats['InferenceGops'])}
            dp = dp.append(basePoint, ignore_index=True)

            dataPoints = dataPoints.append(dp)
            data[net][subset] = dataPoints
    return data
#}}}

def plot_tradeoff(data, saveLoc=None): 
#{{{
    colours = {
                'Unpruned'            : 'red',
                '>= Unpruned Network' : 'green',
                'Error < 1%'          : 'yellow',
                'Error < 2%'          : 'orange',
                'Error < 4%'          : 'dark red'
              }
    
    labels = {'Unpruned' : "$\mathcal{M}_\mathcal{D}$ on $\mathcal{D}'$", '>= Unpruned Network' : "$\mathcal{M}_\mathcal{D}'$ on $\mathcal{D}'$"}

    nets = list(data.keys()) 
    subsets = list(data[nets[0]].keys())
    globalData = []
    for net in nets: 
        for subset in subsets:
            ax = plt.subplots(1,1)
            ax[1].set_title("Search Time to Inference Time tradeoff for {}-{}".format(net.capitalize(),datasetTranslate[subset].capitalize()))
            ax[1].set_xlabel("Search Time (s)")
            ax[1].set_ylabel("Inference Time for searched model (s)")
            
            # get unpruned gops
            df = data[net][subset]
            unprunedGops = float(df.loc[df['Type'] == 'Unpruned']['InferenceGops'])
            unprunedTime = float(df.loc[df['Type'] == 'Unpruned']['Metric'])

            for ptType, pts in df.groupby(['Type']): 
                label = ptType
                ax[1].scatter(pts['Time'], pts['Metric'], color=colours[ptType], label=labels[ptType])
                ax[1].legend()
                if ptType is not 'Unpruned':
                    for idx, time in enumerate(list(pts['Time'])):
                        prunedGops = float(pts['InferenceGops'][idx])
                        gopsGain = 100. * (unprunedGops - prunedGops)/unprunedGops
                        gopsX = unprunedGops / prunedGops
                        pruningLevel = str(pts['PruningLevel'][idx])
                        text = "{:.2f}x,{}%".format(gopsX, pruningLevel)
                        loc = (time, pts['Metric'][idx])
                        textOffsetPixels = (-65,-17)
                        ax[1].annotate(text, loc, xytext=textOffsetPixels, textcoords='offset pixels')

                        globalData.append((net, subset, unprunedTime/float(pts['Metric'][idx]), gopsX, pruningLevel)) 
            
            if saveLoc is not None: 
                plt.tight_layout()
                figFile = os.path.join(saveLoc, '{}_{}.png'.format(net, subset))
                plt.savefig(figFile)
    
    maxLatencyGain = max(globalData, key=itemgetter(2))
    maxGopsGain = max(globalData, key=itemgetter(3))
    maxPruningLvl = max(globalData, key=itemgetter(4))
    print("Max latency gain = {} \n Max gops gain = {} \n Max Pruning level = {}".format(maxLatencyGain, maxGopsGain, maxPruningLvl))
#}}}

