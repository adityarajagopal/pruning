import os
import sys 
import math
import json
import glob
import argparse
import itertools

import configparser as cp

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt

from src.ar4414.pruning.plotting.config import datasetTranslate
import src.ar4414.pruning.plotting.summary_stats.acc as accSrc
import src.ar4414.pruning.plotting.summary_stats.gops as gopSrc
import src.ar4414.pruning.plotting.summary_stats.timing as timingSrc
import src.ar4414.pruning.plotting.summary_stats.l1_norms as l1NormsSrc
import src.ar4414.pruning.plotting.summary_stats.channel_diff as channeDiffSrc

def summary_statistics(logs, networks, datasets, prunePercs):
#{{{    
    global datasetTranslate
    
    # data refers to runs that have been pruned based on finetuning on the dataset
    data = {'Network':[], 'Dataset':[], 'PrunePerc':[], 'AvgTestAcc':[], 'StdTestAcc':[], 'PreFtDiff':[], 'PostFtDiff':[], 'InferenceGops':[], 'FinetuneGops':[], 'Memory(MB)':[]}
    
    pruneAfter = -1
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
                    
                    # get epochs after which pruning was performed
                    if pruneAfter == -1:
                        config = cp.ConfigParser() 
                        configFile = glob.glob(os.path.join(basePath, log, '*.ini'))[0]
                        config.read(configFile)
                        pruneAfter = config.getint('pruning_hyperparameters', 'prune_after')

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
                data['Dataset'].append(datasetTranslate[dataset])
                # data['Dataset'].append(dataset)
                data['PrunePerc'].append(int(pp))
                data['AvgTestAcc'].append(avgTestAcc)
                data['StdTestAcc'].append(stdTestAcc)
                data['PreFtDiff'].append(pDiffPerRun)
                data['PostFtDiff'].append(pDiffBetweenRuns)
                data['InferenceGops'].append(np.mean(tmpInfGops))
                data['FinetuneGops'].append(np.mean(tmpFtGops))
                data['Memory(MB)'].append(np.mean(tmpMem))
    
    df = pd.DataFrame(data)

    return df, pruneAfter
#}}}

def subset_agnostic_summary_statistics(logs, networks, datasets, prunePercs, subsetAgnosticLogs):
#{{{
    global datasetTranslate

    # data refers to runs that have not been pruned or pruned without finetuning on the subset 
    data = {'Network':[], 'Subset':[], 'PrunePerc':[], 'TestAcc':[], 'InferenceGops':[]}

    prunePercs = ['0'] + prunePercs
                
    for network in networks:
        for subset in datasets:
            for pp in prunePercs:
                data['Network'].append(network)
                # data['Subset'].append(subset)
                data['Subset'].append(datasetTranslate[subset])
                data['PrunePerc'].append(pp)
                if pp == '0':
                    data['TestAcc'].append(logs[network][subset]['unpruned_inference']['test_top1'])
                    data['InferenceGops'].append(logs[network][subset]['unpruned_inference']['gops'])
                else:
                    sALog = subsetAgnosticLogs[network]['entire_dataset']
                    data['TestAcc'].append(sALog['{}_inference'.format(subset)][pp])
                    gops,_,_ = gopSrc.get_gops(sALog['base_path'], 'pp_{}/{}/orig/'.format(pp, sALog[pp][0]))
                    data['InferenceGops'].append(gops)
    
    df = pd.DataFrame(data)
    return df
#}}}

def per_epoch_statistics(logs, networks, datasets, prunePercs):
#{{{
    pruneAfter = -1
    data = {net:{subset:{pp:None for pp in prunePercs} for subset in datasets} for net in networks}
    for network in networks:
        for dataset in datasets:
            for pp in prunePercs:
                basePath = logs[network][dataset]['base_path'] 
                runs = logs[network][dataset][pp] 

                for i,run in enumerate(runs):
                    log = 'pp_{}/{}/orig'.format(pp, run)
    
                    # get epochs after which pruning was performed
                    if pruneAfter == -1:
                        config = cp.ConfigParser() 
                        configFile = glob.glob(os.path.join(basePath, log, '*.ini'))[0]
                        config.read(configFile)
                        pruneAfter = config.getint('pruning_hyperparameters', 'prune_after')
                    
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
    
    return data, pruneAfter
#}}}

def l1_norm_statistics(logs, networks, datasets, prunePercs): 
#{{{
    
    data = {net:{dataset:{pp:None for pp in prunePercs+['stats']} for dataset in datasets} for net in networks}

    for network in networks:
        for dataset in datasets:
            print("Collecting for {} on {}".format(network, dataset))
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

def timing_statistics(pruneAfter, logPath, networks, datasets, prunePercs, totalEpochs=30): 
#{{{
    inferenceData = {net:{pp:{} for pp in prunePercs} for net in networks}
    trainingData = {net:{pp:{} for pp in prunePercs} for net in networks}

    for net in networks: 
        unprunedTraining = []
        prunedTraining = []

        for dataset in datasets: 
            for count,pp in enumerate(prunePercs): 
                logFile = os.path.join(logPath, net, dataset, pp, 'timing_data.pth.tar')
                log = torch.load(logFile) 
                
                # inference times
                unprunedTime = 1000.*(log['inference']['minibatch'][0][0] / 128)
                prunedTime = 1000.*(log['inference']['minibatch'][0][1] / 128)

                if count == 0:
                    inferenceData[net]['0'] = unprunedTime
                inferenceData[net][pp] = prunedTime

                # finetune times
                trainMbTimes = {epoch: np.mean(times) for epoch, times in log['training']['minibatch'].items()}                
                [unprunedTraining.append(time) if epoch < pruneAfter else prunedTraining.append(time) for epoch,time in trainMbTimes.items()]
                unprunedFt = np.mean(unprunedTraining)
                prunedFt = np.mean(prunedTraining)
                #TODO : make this 30 also be read from file 
                ftTimes = np.cumsum([unprunedFt if epoch < pruneAfter else prunedFt for epoch in range(totalEpochs)])
                ftTime = {'Epoch':list(range(1,totalEpochs+1)), 'FtTime':ftTimes}
                trainingData[net][pp] = pd.DataFrame(ftTime)
    
    return inferenceData, trainingData
#}}}


