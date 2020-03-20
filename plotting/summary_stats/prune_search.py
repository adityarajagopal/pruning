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

datasetTranslate = {'aquatic':'aquatic', 'subset1':'outdoor', 'indoors':'indoor', 'natural':'natural', 'random1':'random'}

def check_stopping(mode, state, prevPp, currPp):
#{{{
    if mode == 'memory_opt':
        if prevPp == currPp:
            return True
    elif mode == 'cost_opt':
        if state == 1:
            return True

    return False
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
    
    if 'Ft_Time' in list(perEpochData.columns):
        time = np.array(list(perEpochData['Ft_Time'])) - perEpochData['Ft_Time'][pruneAfter-1]
        currTotalTime = currCost['Time'][-1]
        cost['Time'] = currCost['Time'] + list(currTotalTime + time[pruneAfter:])
    
    if int(bestTestAcc) < int(targetAcc):
        retVal = -1
    else:
        retVal = 1

    return retVal, cost, bestTestAcc
#}}}

def bin_search_cost(logs, networks, datasets, prunePercs, mode='memory_opt', profLogPath=None, targetData=None):
#{{{
    global datasetTranslate
    
    # perform binary search to find pruning percentage that give no accuracy loss
    data = {net:{subset:None for subset in datasets} for net in networks}
    initPp = 50                     
    ppToSearch = [int(x) for x in prunePercs]

    for net in networks:
        for subset in datasets:
            print("Searching for {} on {}".format(net, subset))
            prevPp = 0
            currPp = initPp
            uB = ppToSearch[-1]
            lB = ppToSearch[0]
            bestPp = 0

            # find best test accuracy and gops obtained after initial 5 epochs of finetuning
            # perEpochData, pruneAfter = collector.per_epoch_statistics(logs, networks, datasets, [str(initPp)], profLogs)[net][subset][str(initPp)]
            aggPerEpochData, pruneAfter = collector.per_epoch_statistics(logs, networks, datasets, [str(initPp)])
            perEpochData = aggPerEpochData[net][subset][str(initPp)]
            
            if profLogPath is not None:
                _, ftTime =collector.timing_statistics(pruneAfter, profLogPath, [net], ['cifar100'], [str(initPp)])  
                perEpochData['Ft_Time'] = ftTime[net][str(initPp)]['FtTime']
                cost = {'Time':list(perEpochData['Ft_Time'])[:pruneAfter], 'Gops':list(perEpochData['Ft_Gops'])[:pruneAfter], 'TestAcc':list(perEpochData['Test_Top1'])[:pruneAfter]}
            else:
                cost = {'Gops':list(perEpochData['Ft_Gops'])[:pruneAfter], 'TestAcc':list(perEpochData['Test_Top1'])[:pruneAfter]}

            if targetData is None:
                targetAcc = max(list(perEpochData['Test_Top1'])[:pruneAfter])
            else:
                # targetAcc = float(targetData.loc[(targetData['Network'] == net) & (targetData['Subset'] == subset)]['TestAcc'])
                targetAcc = float(targetData.loc[(targetData['Network'] == net) & (targetData['Subset'] == datasetTranslate[subset])]['TestAcc'])
            state = 0
            bestTestAcc = targetAcc
            epochCount = pruneAfter
            epochData = []
            
            # perform binary search
            while not check_stopping(mode, state, prevPp, currPp):
                # perEpochData, pruneAfter = collector.per_epoch_statistics(logs, networks, datasets, [str(currPp)], profLogs)[net][subset][str(currPp)]
                epochCount += int(perEpochData['Epoch'].tail(1)) - pruneAfter
                epochData.append((currPp, epochCount))
                aggPerEpochData, pruneAfter = collector.per_epoch_statistics(logs, networks, datasets, [str(currPp)])
                perEpochData = aggPerEpochData[net][subset][str(currPp)]
                
                if profLogPath is not None:
                    _, ftTime =collector.timing_statistics(pruneAfter, profLogPath, [net], ['cifar100'], [str(currPp)])  
                    perEpochData['Ft_Time'] = ftTime[net][str(currPp)]['FtTime']
                
                state, cost, highestTestAcc = single_search(perEpochData, cost, targetAcc, pruneAfter)
                
                # prune less
                if state == -1: 
                    tmp = (lB + currPp) / 2.
                    uB = currPp 

                # try to prune more, but return previous model if state goes to -1
                elif state == 1:
                    tmp = (uB + currPp) / 2.
                    lB = currPp 
                    bestPp = currPp 
                    bestTestAcc = highestTestAcc

                prevPp = currPp
                currPp = 5 * math.ceil(tmp/5)  
            
            data[net][subset] = {'cost':pd.DataFrame(cost), 'best_pp':bestPp, 'best_acc':bestTestAcc, 'target_acc':targetAcc, 'epoch_data':epochData}
    
    return data 
#}}}

def plot_bin_search_cost(data, saveLoc=None, time=False):
#{{{
    nets = list(data.keys())
    subsets = list(data[nets[0]].keys())

    for net in nets:
        for subset in subsets: 
            cost = data[net][subset]['cost']
            bestPp = data[net][subset]['best_pp']
            bestTestAcc = data[net][subset]['best_acc']
            
            if not time:
                title = "Cost of finding best pruning percentage for {} on {} \n Best Pruning level = {}%".format(net.capitalize(), subset.capitalize(), bestPp)
                ax = cost.plot(x='Gops', y='TestAcc', title=title)
                ax.axhline(bestTestAcc, label='Best Test Accuracy', color='red')
                ax.set_xlabel('Cost (GOps) of performing binary search to find best pruning level')
                ax.set_ylabel('Test Top1 (%)')
                ax.legend()
            else:
                title = "Binary Search cost of finding best pruning percentage for \n {} on {} \n Best Pruning level = {}%".format(net.capitalize(), subset.capitalize(), bestPp)
                ax2 = cost.plot(x='Time', y='TestAcc', title=title)
                ax2.axhline(bestTestAcc, label='Best Test Accuracy', color='red')
                ax2.set_xlabel('Cost per minibatch (seconds) of on Nvidia Jetson TX2')
                ax2.set_ylabel('Test Top1 (%)')
                ax2.legend()

            if saveLoc is not None: 
                plt.tight_layout()
                figFile = os.path.join(saveLoc, '{}_{}.png'.format(net, subset))
                plt.savefig(figFile)
#}}}




