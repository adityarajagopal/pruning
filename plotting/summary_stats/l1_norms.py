import os
import sys
import math

import torch
import numpy as np
import matplotlib.pyplot as plt

def calculate_metric(param):
#{{{
    #l1-norm
    metric = np.absolute(param).reshape(param.shape[0], -1).sum(axis=1)
    metric /= (param.shape[1]*param.shape[2]*param.shape[3])
    return metric
#}}}

def calculate_l1_norms(model):
#{{{
    l1Norms = []
    for p in model.items():
        if 'conv' in p[0] and 'weight' in p[0]:
            pNp = p[1].data.cpu().numpy()
            metric = calculate_metric(pNp)
            l1Norms += list(metric)
    return l1Norms
#}}}

def get_l1_norms(logs, network, dataset, pp, run):
#{{{
    preFtModel = torch.load(logs[network]['pre_ft_model'])
    postFtModel = torch.load(os.path.join(logs[network][dataset]['base_path'], "pp_{}/{}/orig/pre_pruning.pth.tar".format(pp,run)))

    preL1Norms = np.array(calculate_l1_norms(preFtModel))
    postL1Norms = np.array(calculate_l1_norms(postFtModel))
    diffL1Norms = postL1Norms - preL1Norms

    return(preL1Norms, postL1Norms, diffL1Norms)    
#}}}

def plot_histograms(normsDict):
#{{{
    for net,v in normsDict.items():
        for dataset,l1NormStats in v.items():
            ax = plt.subplots(1,1)[1]
            title = "Histogram of difference in l1-norms before and after finetuning \n Net:{}, Subset:{}".format(net, dataset)
            l1NormStats['Diff_L1_Norms'].plot.hist(bins=int(math.sqrt(len(l1NormStats.index))), ax=ax, title=title)
            ax.set_xlabel('Difference in l1-norm before and after finetuning')
            ax.set_ylabel('Frequency occured')
            
            ax = plt.subplots(1,1)[1]
            title = "Histogram of l1-norms before and after finetuning \n Net:{}, Subset:{}".format(net, dataset)
            l1NormStats[['Pre_Ft_L1_Norms', 'Post_Ft_L1_Norms']].plot.hist(bins=int(math.sqrt(len(l1NormStats.index))), ax=ax, title=title)
            ax.set_xlabel('L1-norm before and after finetuning')
            ax.set_ylabel('Frequency occured')
#}}}

