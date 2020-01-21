import sys
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

# basePath = "/home/ar4414/pytorch_training/src/ar4414/pruning/logs/mobilenetv2/cifar100/entire_dataset/l1_prune"
# logA = "pp_60/2020-01-20-17-58-34/orig"
# logB = "pp_60/2019-11-22-23-45-13/orig" 
# basePath = "/home/ar4414/pytorch_training/src/ar4414/pruning/logs/resnet/cifar100/entire_dataset/l1_prune"
# logA = "pp_60/2020-01-20-23-04-18/orig"
# logB = "pp_60/2019-11-22-21-00-12/orig" 

# basePath = "/home/ar4414/pytorch_training/src/ar4414/pruning/logs/mobilenetv2/cifar100/subset1/l1_prune"
# logA = "pp_60/2020-01-20-18-07-15/orig"
# logB = "pp_60/2019-11-22-21-57-37/orig" 
# basePath = "/home/ar4414/pytorch_training/src/ar4414/pruning/logs/resnet/cifar100/subset1/l1_prune"
# logA = "pp_60/2020-01-20-19-52-50/orig"
# logB = "pp_60/2019-11-22-19-50-10/orig" 

# basePath = "/home/ar4414/pytorch_training/src/ar4414/pruning/logs/mobilenetv2/cifar100/aquatic/l1_prune"
# logA = "pp_60/2020-01-20-15-48-28/orig"
# logB = "pp_60/2019-11-22-22-30-19/orig" 
# basePath = "/home/ar4414/pytorch_training/src/ar4414/pruning/logs/resnet/cifar100/aquatic/l1_prune"
# logA = "pp_60/2020-01-20-21-00-22/orig"
# logB = "pp_60/2019-11-22-20-14-06/orig" 

# logFile = '/home/ar4414/pytorch_training/src/ar4414/pruning/prunedChannels/mobilenetv2/pre_ft_pp_60.pth.tar'
# postFtChannelsPruned = torch.load(logFile)
                
logFile = os.path.join(basePath, logA, 'pruned_channels.json')
with open(logFile, 'r') as jFile:
    preFtChannelsPruned = json.load(jFile)    
preFtChannelsPruned = list(preFtChannelsPruned.values())[0]
preFtChannelsPruned.pop('prunePerc')

logFile = os.path.join(basePath, logB, 'pruned_channels.json')
with open(logFile, 'r') as jFile:
    postFtChannelsPruned = json.load(jFile)    
postFtChannelsPruned = list(postFtChannelsPruned.values())[0]
postFtChannelsPruned.pop('prunePerc')

layerNames = list(postFtChannelsPruned.keys())

numChanChanged = []
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
        # print("For layer {}, percentage of channels pruned that were different = {}".format(k,pDiff))
    else:
        pDiff = 0
    
    numChanChanged.append(pDiff)      

pDiffGlobal = 1. - (totOverlap / totChannelsPruned)
print("Across network, percentage of channels that were different = {}".format(pDiffGlobal))

