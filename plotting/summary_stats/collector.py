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

import src.ar4414.pruning.plotting.summary_stats.acc as accSrc
import src.ar4414.pruning.plotting.summary_stats.gops as gopSrc
import src.ar4414.pruning.plotting.summary_stats.channel_diff as channeDiffSrc
import src.ar4414.pruning.plotting.summary_stats.l1_norms as l1NormsSrc

def summary_statistics(logs, networks, datasets, prunePercs):
#{{{    
    # data refers to runs that have been pruned based on finetuning on the dataset
    data = {'Network':[], 'Dataset':[], 'PrunePerc':[], 'AvgTestAcc':[], 'StdTestAcc':[], 'PreFtDiff':[], 'PostFtDiff':[], 'InferenceGops':[], 'FinetuneGops':[], 'Memory(MB)':[]}
    
    for network in networks:
        for dataset in datasets:
            for pp in prunePercs:
                preFtChannelsPruned = '/home/ar4414/pytorch_training/src/ar4414/pruning/logs/{}/{}/baseline/pre_ft_pp_{}.pth.tar'.format(network,'cifar100',pp)
                preFtChannelsPruned = torch.load(preFtChannelsPruned)
                basePath = logs[network][dataset]['base_path'] 

                runs = logs[network][dataset][pp] 
                tmpPruned = []
                tmpAcc = []
                tmpInfGops = []
                tmpFtGops = []
                tmpMem = []
                for i,run in enumerate(runs): 
                    log = 'pp_{}/{}/orig'.format(pp, run)
    
                   # extract pruned channels 
                    channelsPruned = channeDiffSrc.get_pruned_channels(basePath, log)
                    tmpPruned.append(channelsPruned)

                    # extract accuracy 
                    accFile = os.path.join(basePath, log, 'log.csv')
                    logReader = accSrc.LogReader(accFile)
                    bestTest = logReader.get_best_test_acc()
                    tmpAcc.append(bestTest)
    
                    #extract gops
                    infGops, totalFtGops, modelSize = gopSrc.get_gops(basePath, log)
                    tmpInfGops.append(infGops)
                    tmpFtGops.append(totalFtGops)
                    tmpMem.append(modelSize)

                pDiffPerRun, pDiffBetweenRuns = channeDiffSrc.compute_differences(preFtChannelsPruned, tmpPruned)
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
                data['FinetuneGops'].append(np.mean(tmpFtGops))
                data['Memory(MB)'].append(np.mean(tmpMem))
    
    df = pd.DataFrame(data)

    return df
#}}}

def subset_agnostic_summary_statistics(logs, networks, datasets, prunePercs):
#{{{
    # data refers to runs that have not been pruned or pruned without finetuning on the subset 
    data = {'Network':[], 'Subset':[], 'PrunePerc':[], 'TestAcc':[], 'InferenceGops':[]}
    
    for network in networks:
        for subset in datasets:
            pp = 0
            data['Network'].append(network)
            data['Subset'].append(subset)
            data['PrunePerc'].append(pp)
            data['TestAcc'].append(logs[network][subset]['unpruned_inference']['test_top1'])
            data['InferenceGops'].append(logs[network][subset]['unpruned_inference']['gops'])
    
    df = pd.DataFrame(data)
    return df
#}}}

def per_epoch_statistics(logs, networks, datasets, prunePercs):
#{{{
    data = {net:{subset:{pp:None for pp in prunePercs} for subset in datasets} for net in networks}
    for network in networks:
        for dataset in datasets:
            for pp in prunePercs:
                basePath = logs[network][dataset]['base_path'] 
                runs = logs[network][dataset][pp] 

                for i,run in enumerate(runs):
                    log = 'pp_{}/{}/orig'.format(pp, run)
                    
                    #get epochs and acc per epoch
                    accFile = os.path.join(basePath, log, 'log.csv')
                    logReader = accSrc.LogReader(accFile)
                    tmpPerEpochStats = logReader.get_acc_per_epoch()

                    #get gops by epoch
                    gops, modelSize = gopSrc.get_gops(basePath, log, perEpoch=True)
                    
                    tmpPerEpochStats['Ft_Gops'] = gops                                        

                    if i == 0:
                        perEpochStats = tmpPerEpochStats
                    else:
                        perEpochStats = pd.concat((perEpochStats, tmpPerEpochStats))
                
                perEpochStats = perEpochStats.groupby(perEpochStats.index).mean()
                perEpochStats['Epoch'] += 1
                
                data[network][dataset][pp] = perEpochStats
    
    return data
#}}}

def single_search(perEpochData, currCost, targetAcc, pruneAfter):
#{{{
    testAccs = list(perEpochData['Test_Top1'])
    bestTestAcc = max(testAccs[pruneAfter:])
    bestIdx = testAccs.index(bestTestAcc)

    # remove cost of finetuning as this happens only once
    gops = np.array(list(perEpochData['Ft_Gops'])) - perEpochData['Ft_Gops'][pruneAfter-1]
    
    currTotalGops = currCost['Gops'][-1]
    cost = {'Gops': currCost['Gops'] + list(currTotalGops + gops[pruneAfter:]), 'TestAcc': currCost['TestAcc'] + testAccs[pruneAfter:]}
    
    if int(bestTestAcc) < int(targetAcc):
        return -1, cost
    else:
        return 1, cost
    # elif int(bestTestAcc) > int(targetAcc): 
    #     return 1, cost
    # else:
    #     return 0, cost
#}}}

def bin_search_cost(logs, networks, datasets, prunePercs):
#{{{
    # perform binary search to find pruning percentage that give no accuracy loss
    data = {net:{subset:None for subset in datasets} for net in networks}
    initPp = 50                     
    pruneAfter = 5
    ppToSearch = [int(x) for x in prunePercs]

    for net in networks:
        for subset in datasets:
            prevPp = 0
            currPp = initPp
            uB = ppToSearch[-1]
            lB = ppToSearch[0]
            bestPp = 0
            
            # find best test accuracy and gops obtained after initial 5 epochs of finetuning
            perEpochData = per_epoch_statistics(logs, networks, datasets, [str(initPp)])[net][subset][str(initPp)]
            targetAcc = max(list(perEpochData['Test_Top1'])[:pruneAfter])
            cost = {'Gops':list(perEpochData['Ft_Gops'])[:pruneAfter], 'TestAcc':list(perEpochData['Test_Top1'])[:pruneAfter]}
            
            # perform binary search
            while prevPp != currPp:
                print("Pruning level to search = {} %".format(currPp)) 
                
                perEpochData = per_epoch_statistics(logs, networks, datasets, [str(currPp)])[net][subset][str(currPp)]
                state, cost = single_search(perEpochData, cost, targetAcc, pruneAfter)
                
                # prune less
                if state == -1: 
                    tmp = (lB + currPp) / 2.
                    uB = currPp 

                # try to prune more, but return previous model if state goes to -1
                elif state == 1:
                    tmp = (uB + currPp) / 2.
                    lB = currPp 
                    bestPp = currPp 

                # elif state == 0:
                #     break

                prevPp = currPp
                currPp = 5 * math.ceil(tmp/5)  
            
            print('Best Pruning Perc = {}'.format(bestPp))
            pd.DataFrame(cost).plot(x='Gops', y='TestAcc')
            plt.show()
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
                    
                    channelsPruned = channeDiffSrc.get_pruned_channels(basePath, log)
                    preL1Norms, postL1Norms, diffL1Norms, layerIndexOffset = l1NormsSrc.get_l1_norms(logs, network, dataset, pp, run)
                    tmpCutoff.append(l1NormsSrc.get_cutoff_l1_norms(np.array(postL1Norms), channelsPruned, layerIndexOffset))

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


