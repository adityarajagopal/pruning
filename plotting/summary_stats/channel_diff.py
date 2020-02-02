import sys
import os
import json
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_pruned_channels(basePath, log):
#{{{
    logFile = os.path.join(basePath, log, 'pruned_channels.json')
    with open(logFile, 'r') as jFile:
        channelsPruned = json.load(jFile)    
    channelsPruned = list(channelsPruned.values())[0]
    channelsPruned.pop('prunePerc')

    return channelsPruned
#}}}

def calc_perc_diff(preFtChannelsPruned, postFtChannelsPruned):
#{{{
    totChannelsPruned = 0
    totOverlap = 0
    for k,currChannelsPruned in postFtChannelsPruned.items(): 
        origChannelsPruned = preFtChannelsPruned[k]
    
        numChanPruned = len(list(set(currChannelsPruned) | set(origChannelsPruned)))
        overlap = len(list(set(currChannelsPruned) & set(origChannelsPruned)))
        totChannelsPruned += numChanPruned
        totOverlap += overlap  
    
        if numChanPruned != 0:
            pDiff = 1.0 - (overlap / numChanPruned)
        else:
            pDiff = 0
        
    pDiffGlobal = 1. - (totOverlap / totChannelsPruned)
    return pDiffGlobal
#}}}

def compute_differences(preFtChannelsPruned, prunedChannels):
#{{{
    pDiffPerRun = [calc_perc_diff(preFtChannelsPruned, postFtChannelsPruned) for postFtChannelsPruned in prunedChannels]
    pDiffBetweenRuns = [calc_perc_diff(x[0],x[1]) for x in list(itertools.combinations(prunedChannels,2))]
    return np.mean(pDiffPerRun), np.mean(pDiffBetweenRuns) 
#}}}

def plot_channel_diff_by_pp(summaryData, saveLoc=None):
#{{{
    for (dataset, net), data in summaryData.groupby(['Dataset', 'Network']):
        ax = plt.subplots(1,1)[1]
        data.plot.bar(x='PrunePerc', y=['PreFtDiff', 'PostFtDiff'], title='Difference in Channels Pruned for {} on {}'.format(net, dataset), ax=ax)
        ax.set_xlabel('Pruning Percentage (%)')
        ax.set_ylabel('Percentage Difference in Channels Pruned (%)')

        if saveLoc is not None:
            plt.tight_layout
            figFile = os.path.join(saveLoc, '{}_{}.png'.format(net, dataset))
            plt.savefig(figFile)
#}}}

def plot_channel_diff_by_subset(summaryData, saveLoc=None):
#{{{
    transformedData = {}
    for subset, data in summaryData.groupby(['Dataset']):
        ax = plt.subplots(1,1)[1]
        for net, changes in data.groupby(['Network']):
            transformedData['pp'] = list(changes['PrunePerc'])
            transformedData[net] = list(changes['PreFtDiff'])
        newDf = pd.DataFrame(transformedData)
        newDf.plot.bar(x='pp', title='Difference in Channels Pruned with finetuning on the {} subset'.format(subset.capitalize()), ax=ax)
        ax.set_xlabel('Pruning Percentage (%)')
        ax.set_ylabel('Percentage Difference in Channels Pruned before and after finetuning (%)')

        if saveLoc is not None:
            plt.tight_layout
            figFile = os.path.join(saveLoc, '{}.png'.format(subset))
            plt.savefig(figFile)
#}}}




