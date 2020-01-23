import sys
import os
import itertools
import math
import json
import argparse

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import src.ar4414.pruning.plotting.summary_stats.acc as mod_acc
import src.ar4414.pruning.plotting.summary_stats.gops as mod_gops
import src.ar4414.pruning.plotting.summary_stats.channel_diff as mod_channel_diff
import src.ar4414.pruning.plotting.summary_stats.l1_norms as mod_l1_norms

def summary_statistics(logs, networks, datasets, prunePercs):
#{{{    
    data = {'Network':[], 'Dataset':[], 'PrunePerc':[], 'AvgTestAcc':[], 'StdTestAcc':[], 'PreFtDiff':[], 'PostFtDiff':[], 'InferenceGops':[]}
    
    for network in networks:
        for dataset in datasets:
            for pp in prunePercs:
                preFtChannelsPruned = '/home/ar4414/pytorch_training/src/ar4414/pruning/prunedChannels/{}/pre_ft_pp_{}.pth.tar'.format(network,pp)
                preFtChannelsPruned = torch.load(preFtChannelsPruned)
                basePath = logs[network][dataset]['base_path'] 

                runs = logs[network][dataset][pp] 
                tmpPruned = []
                tmpAcc = []
                tmpInfGops = []
                for i,run in enumerate(runs): 
                    log = 'pp_{}/{}/orig'.format(pp, run)
    
                   # extract pruned channels 
                    channelsPruned = mod_channel_diff.get_pruned_channels(basePath, log)
                    tmpPruned.append(channelsPruned)

                    # extract accuracy 
                    accFile = os.path.join(basePath, log, 'log.csv')
                    logReader = mod_acc.LogReader(accFile)
                    bestTest = logReader.get_best_test_acc()
                    tmpAcc.append(bestTest)
    
                    #extract inferenceGops
                    gops = mod_gops.get_gops(basePath, log)
                    tmpInfGops.append(gops['inf'])
                
                pDiffPerRun, pDiffBetweenRuns = mod_channel_diff.compute_differences(preFtChannelsPruned, tmpPruned)
                avgTestAcc = np.mean(tmpAcc) 
                stdTestAcc = np.std(tmpAcc)
                
                data['Network'].append(network)
                data['Dataset'].append(dataset)
                data['PrunePerc'].append(int(pp))
                data['AvgTestAcc'].append(avgTestAcc)
                data['StdTestAcc'].append(stdTestAcc)
                data['PreFtDiff'].append(pDiffPerRun)
                data['PostFtDiff'].append(pDiffBetweenRuns)
                data['InferenceGops'].append(np.mean(tmpInfGops))
    
    df = pd.DataFrame(data)

    return df
#}}}

def l1_norm_statistics(logs, networks, datasets, prunePercs): 
#{{{
    
    data = {net:{dataset:{pp:None for pp in prunePercs+['stats']} for dataset in datasets} for net in networks}

    for network in networks:
        for dataset in datasets:
            tmpPre = np.array([])
            tmpPost = np.array([])
            tmpDiff = np.array([])
            
            for i,pp in enumerate(prunePercs):
                runs = logs[network][dataset][pp] 
                
                tmpCutoff = []

                for j,run in enumerate(runs): 
                    basePath = logs[network][dataset]['base_path']
                    log = 'pp_{}/{}/orig'.format(pp, run)
                    
                    channelsPruned = mod_channel_diff.get_pruned_channels(basePath, log)
                    preL1Norms, postL1Norms, diffL1Norms, layerIndexOffset = mod_l1_norms.get_l1_norms(logs, network, dataset, pp, run)
                    tmpCutoff.append(mod_l1_norms.get_cutoff_l1_norms(np.array(postL1Norms), channelsPruned, layerIndexOffset))

                    if len(tmpPre) == 0:
                        tmpPre = preL1Norms
                        tmpPost = postL1Norms
                        tmpDiff = diffL1Norms
                    else:
                        tmpPre += preL1Norms
                        tmpPost += postL1Norms
                        tmpDiff += diffL1Norms
                
                avgCutoff = np.mean(tmpCutoff)
                data[network][dataset][pp] = avgCutoff

            numRuns = (i+1)*(j+1)
            meanPreFtL1 = tmpPre / numRuns 
            meanPostFtL1 = tmpPost / numRuns
            meanDiffL1 = tmpDiff / numRuns
            
            l1NormStats = pd.DataFrame({'Pre_Ft_L1_Norms':meanPreFtL1, 'Post_Ft_L1_Norms':meanPostFtL1, 'Diff_L1_Norms':meanDiffL1})

            data[network][dataset]['stats'] = l1NormStats
            
    return data
#}}}
