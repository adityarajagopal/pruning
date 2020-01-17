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
import math

def calculate_metric(param):
#{{{
    #l1-norm
    metric = np.absolute(param).reshape(param.shape[0], -1).sum(axis=1)
    metric /= (param.shape[1]*param.shape[2]*param.shape[3])
    return metric
#}}}

def calc_l1(model):
#{{{
    l1Norms = []
    
    # create global ranking
    for k,p in model.items():
        if 'conv' in k and 'weight' in k:
            pNp = p.data.cpu().numpy()
            # calculate metric
            metric = calculate_metric(pNp)
            l1Norms += list(metric)

    return l1Norms
#}}}

subsets = ['entire_dataset', 'subset1', 'aquatic']

nets = ['alexnet', 'resnet', 'mobilenetv2', 'squeezenet']
net = nets[2]
subset = subsets[0]
log = 'pp_5/2019-11-22-22-53-17/orig' 
basePath = '/home/ar4414/pytorch_training/src/ar4414/pruning/logs/{}/cifar100/{}/l1_prune'.format(net, subset)

baseModelFile = '/mnt/users/ar4414/pruning_logs/mobilenetv2/cifar100/baseline/2019-10-07-15-17-32/orig/111-model.pth.tar'
modelFile = os.path.join(basePath, log, 'best-model.pth.tar') 

baseModel = torch.load(baseModelFile)
newModel = torch.load(modelFile)
basel1Norms = calc_l1(baseModel) 
newl1Norms = calc_l1(newModel)

data = {'base':basel1Norms[:len(newl1Norms)], 'new':newl1Norms}
df = pd.DataFrame(data)
df.plot.hist(bins=int(math.sqrt(len(newl1Norms))), alpha=0.5)
plt.show()
