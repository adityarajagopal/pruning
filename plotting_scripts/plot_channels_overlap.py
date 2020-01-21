import sys
import itertools
import torch
import matplotlib.pyplot as plt
import os
import json
import torch.nn as nn
import numpy as np
import subprocess
import itertools
from scipy.spatial import distance
import pandas as pd

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

with open('/home/ar4414/pytorch_training/src/ar4414/pruning/plotting_scripts/logs.json', 'r') as jFile:
    logs = json.load(jFile)

networks = ['mobilenetv2', 'resnet']
datasets = ['entire_dataset', 'subset1', 'aquatic']
prunePercs = ['5', '10', '25', '50', '60', '75', '85', '95']
prettyPrint = False

data = {'Network':[], 'Dataset':[], 'PrunePerc':[], 'AvgTestAcc':[], 'StdTestAcc':[], 'PreFtDiff':[], 'PostFtDiff':[]}
for network in networks:
#{{{
    for dataset in datasets:
        if prettyPrint:
            print("============ For {} on {} ============ ".format(network, dataset))
            print("\tPruning Perc (%)\t|\tPerc Diff to PreFt (%)\t|\tPerc Diff between PostFt (%)\t|\tAverage Test Acc (%)\t|")
            print("============================================================================================================================================")
        for pp in prunePercs:
            preFt = '/home/ar4414/pytorch_training/src/ar4414/pruning/prunedChannels/{}/pre_ft_pp_{}.pth.tar'.format(network,pp)
            preFtChannelsPruned = torch.load(preFt)
            basePath = "/home/ar4414/pytorch_training/src/ar4414/pruning/logs/{}/cifar100/{}/l1_prune".format(network, dataset)

            # logs[network][dataset]['base_path'] = basePath
            # with open('/home/ar4414/pytorch_training/src/ar4414/pruning/plotting_scripts/logs_new.json', 'w') as jFile:
            #     json.dump(logs, jFile, indent=2)

            runs = logs[network][dataset][pp] 
            tmpPruned = []
            tmpAcc = []
            for run in runs: 
                log = 'pp_{}/{}/orig'.format(pp, run)
                
                # extract pruned channels 
                logFile = os.path.join(basePath, log, 'pruned_channels.json')
                with open(logFile, 'r') as jFile:
                    channelsPruned = json.load(jFile)    
                channelsPruned = list(channelsPruned.values())[0]
                channelsPruned.pop('prunePerc')
                tmpPruned.append(channelsPruned)

                # extract accuracy 
                accFile = os.path.join(basePath, log, 'log.csv')
                accFile = pd.read_csv(accFile, delimiter=',\t', engine='python')
                
                trainTop1 = accFile['Train_Top1'].dropna()
                testTop1 = accFile['Test_Top1'].dropna()
                valTop1 = accFile['Val_Top1'].dropna()

                if len(valTop1) <= 5: 
                    continue

                bestValIdx = valTop1[5:].idxmax()
                bestTest = testTop1[bestValIdx]

                tmpAcc.append(bestTest)

            pDiffPerRun = [calc_perc_diff(preFtChannelsPruned, postFtChannelsPruned) for postFtChannelsPruned in tmpPruned]
            pDiffBetweenRuns = [calc_perc_diff(x[0],x[1]) for x in list(itertools.combinations(tmpPruned,2))]
            avgTestAcc = np.mean(tmpAcc) 
            stdTestAcc = np.std(tmpAcc)
            
            data['Network'].append(network)
            data['Dataset'].append(dataset)
            data['PrunePerc'].append(int(pp))
            data['AvgTestAcc'].append(avgTestAcc)
            data['StdTestAcc'].append(stdTestAcc)
            data['PreFtDiff'].append(np.mean(pDiffPerRun))
            data['PostFtDiff'].append(np.mean(pDiffBetweenRuns))

            if prettyPrint:
                print("\t\t{}\t\t|\t\t{:3f}\t|\t\t{:3f}\t\t|\t{:3f} pm {:3f}\t|".format(pp, np.mean(pDiffPerRun), np.mean(pDiffBetweenRuns), avgTestAcc, stdTestAcc))
#}}}

df = pd.DataFrame(data)

for (dataset, net), data in df.groupby(['Dataset', 'Network']):
    ax = data.plot.bar(x='PrunePerc', y=['PreFtDiff', 'PostFtDiff'], title='Difference in Channels Pruned for {} on {}'.format(net, dataset))
    ax.set_xlabel('Pruning Percentage (%)')
    ax.set_ylabel('Percentage Difference in Channels Pruned (%)')

#subplots returns fig,ax tuple
axAccs = [plt.subplots(1,1)[1] for i in range(3)]
for (dataset, net), data in df.groupby(['Dataset', 'Network']):
    colour = 'red' if 'mobilenetv2' in net else 'blue'
    if 'entire_dataset' in dataset:
        ax = data.plot.scatter(x='PrunePerc', y='AvgTestAcc', ax=axAccs[0], c=colour, label=net, title='Test Accuracy (%) on subset {}'.format(dataset))
    elif 'subset1' in dataset:
        ax = data.plot.scatter(x='PrunePerc', y='AvgTestAcc', ax=axAccs[1], c=colour, label=net, title='Test Accuracy (%) on subset {}'.format(dataset))
    elif 'aquatic' in dataset:
        ax = data.plot.scatter(x='PrunePerc', y='AvgTestAcc', ax=axAccs[2], c=colour, label=net, title='Test Accuracy (%) on subset {}'.format(dataset))
    
    ax.set_xlabel('Pruning Percentage (%)')
    ax.set_ylabel('Test Accuracy (%)')

plt.show()
