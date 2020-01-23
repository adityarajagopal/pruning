import os
import sys
import math

import torch
import matplotlib
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
    layerNameIndexOffset = {}
    for p in model.items():
        if 'conv' in p[0] and 'weight' in p[0]:
            layerName = '.'.join(p[0].split('.')[:-1])
            layerNameIndexOffset[layerName] = len(l1Norms)
            
            pNp = p[1].data.cpu().numpy()
            metric = calculate_metric(pNp)
            l1Norms += list(metric)

    return l1Norms, layerNameIndexOffset
#}}}

def get_l1_norms(logs, network, dataset, pp, run):
#{{{
    preFtModel = torch.load(logs[network]['pre_ft_model'])
    postFtModel = torch.load(os.path.join(logs[network][dataset]['base_path'], "pp_{}/{}/orig/pre_pruning.pth.tar".format(pp,run)))

    preL1Norms, layerIndexOffset = calculate_l1_norms(preFtModel)
    postL1Norms, _ = calculate_l1_norms(postFtModel)
    preL1Norms = np.array(preL1Norms)
    postL1Norms = np.array(postL1Norms)
    diffL1Norms = postL1Norms - preL1Norms

    return(preL1Norms, postL1Norms, diffL1Norms, layerIndexOffset)    
#}}}

def get_cutoff_l1_norms(l1Norms, channelsPruned, layerIndexOffset):
#{{{
    l1NormsPrunedFilters = []
    
    for k,v in channelsPruned.items(): 
        if k not in layerIndexOffset.keys():
            continue
        offset = int(layerIndexOffset[k])
        filters = np.array(v)
        if len(filters) == 0:
            continue
        l1NormsPrunedFilters += list(l1Norms[offset + filters])
    
    return max(l1NormsPrunedFilters)
#}}}

def plot_histograms(normsDict):
#{{{
    for net,v in normsDict.items():
        for dataset,l1NormStats in v.items():
            ax = plt.subplots(1,1)[1]
            title = "Histogram of difference in l1-norms before and after finetuning \n Net:{}, Subset:{}".format(net, dataset)
            diffDf = l1NormStats['stats']['Diff_L1_Norms']
            diffDf.plot.hist(bins=int(math.sqrt(len(diffDf.index))), ax=ax, title=title)
            ax.set_xlabel('Difference in l1-norm before and after finetuning')
            ax.set_ylabel('Frequency occured')

            ax = plt.subplots(1,1)[1]
            title = "Histogram of l1-norms before and after finetuning \n Net:{}, Subset:{}".format(net, dataset)
            l1NormDf = l1NormStats['stats'][['Pre_Ft_L1_Norms', 'Post_Ft_L1_Norms']]
            l1NormDf.plot.hist(bins=int(math.sqrt(len(l1NormDf.index))), ax=ax, title=title)
            ax.set_xlabel('L1-norm before and after finetuning')
            ax.set_ylabel('Frequency occured')
            
            pp = list(l1NormStats.keys())
            pp.remove('stats')
            colourMap = matplotlib.cm.get_cmap('Dark2')
            colours = colourMap(np.linspace(0,1,num=len(pp)))
            for i,p in enumerate(pp):
                ax.axvline(l1NormStats[p], color=colours[i], label='{}-%'.format(p))
            plt.legend()
#}}}

