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

nets = ['alexnet', 'resnet', 'mobilenetv2', 'squeezenet']
subsets = ['entire_dataset', 'subset1', 'aquatic']
logs = {'mobilenetv2':['pp_0/2020-01-20-10-10-47/orig', 'pp_0/2020-01-20-11-21-06/orig', 'pp_0/2020-01-20-13-49-12/orig']}

baseModelFile = {'mobilenetv2':'/mnt/users/ar4414/pruning_logs/mobilenetv2/cifar100/baseline/2019-10-07-15-17-32/orig/111-model.pth.tar'}

net = nets[2]

data = {}

baseModel = torch.load(baseModelFile[net])
basel1Norms = calc_l1(baseModel) 
# data['base'] = basel1Norms

for i in [0,1,2]:
    subset = subsets[i]
    log = logs[net][i]
    basePath = '/home/ar4414/pytorch_training/src/ar4414/pruning/logs/{}/cifar100/{}/l1_prune'.format(net, subset)
    
    modelFile = os.path.join(basePath, log, 'pre_pruning.pth.tar') 
    newModel = torch.load(modelFile)
    newl1Norms = calc_l1(newModel)

    # data[subset] = newl1Norms

    diff = np.array(basel1Norms) - np.array(newl1Norms)
    diff = [x/basel1Norms[i] for i,x in enumerate(diff)]

    data[subset] = diff

df = pd.DataFrame(data)

for plot in subsets:
    ax = df.plot(title='Difference in l1-norm before and after finetuning by filter', y=plot)
    ax.set_xlabel('Filter Number')
    ax.set_ylabel('Difference in l1-norm')
    plt.tight_layout()
    plt.show()
