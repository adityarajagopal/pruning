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

import src.ar4414.pruning.plotting.summary_stats.collector as collector

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
    
    if int(bestTestAcc) < int(targetAcc):
        retVal = -1
    else:
        retVal = 1
    
    return retVal, cost, bestTestAcc
#}}}

def bin_search_cost(logs, networks, datasets, prunePercs, mode='memory_opt'):
#{{{
    # perform binary search to find pruning percentage that give no accuracy loss
    data = {net:{subset:None for subset in datasets} for net in networks}
    initPp = 50                     
    pruneAfter = 5
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
            perEpochData = collector.per_epoch_statistics(logs, networks, datasets, [str(initPp)])[net][subset][str(initPp)]
            targetAcc = max(list(perEpochData['Test_Top1'])[:pruneAfter])
            cost = {'Gops':list(perEpochData['Ft_Gops'])[:pruneAfter], 'TestAcc':list(perEpochData['Test_Top1'])[:pruneAfter]}
            state = 0
            bestTestAcc = targetAcc
            
            # perform binary search
            while not check_stopping(mode, state, prevPp, currPp):
                perEpochData = collector.per_epoch_statistics(logs, networks, datasets, [str(currPp)])[net][subset][str(currPp)]
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

            data[net][subset] = {'cost':pd.DataFrame(cost), 'best_pp':bestPp, 'best_acc':bestTestAcc}
    
    return data 
#}}}

def plot_bin_search_cost(data):
#{{{
    nets = list(data.keys())
    subsets = list(data[nets[0]].keys())

    for net in nets:
        for subset in subsets: 
            cost = data[net][subset]['cost']
            bestPp = data[net][subset]['best_pp']
            bestTestAcc = data[net][subset]['best_acc']
            title = "Cost of finding best pruning percentage for {} on {} \n Best Pruning level = {}%".format(net.capitalize(), subset.capitalize(), bestPp)
            ax = cost.plot(x='Gops', y='TestAcc', title=title)
            ax.axhline(bestTestAcc, label='Best Test Accuracy', color='red')
            ax.set_xlabel('Cost (GOps) of performing binary search to find best pruning level')
            ax.set_ylabel('Test Top1 (%)')
            ax.legend()
#}}}




