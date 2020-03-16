import os
import sys
import glob
import json
import math

import configparser as cp

import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt

def get_gops(basePath, log, perEpoch=False):
#{{{
    gopsFile = os.path.join(basePath, log, 'gops.json')
    with open(gopsFile, 'r') as jFile:
        gops = json.load(jFile)    
    
    config = cp.ConfigParser() 
    configFile = glob.glob(os.path.join(basePath, log, '*.ini'))[0]
    config.read(configFile)
    
    dataset = config.get('dataset', 'dataset')
    subset = config.get('pruning_hyperparameters', 'sub_name')
    batchSize = config.getint('training_hyperparameters', 'train_batch')    
    pruneAfter = config.getint('pruning_hyperparameters', 'prune_after')
    epochs = config.getint('pruning_hyperparameters', 'finetune_budget')
    
    # not required anymore as ft gops is number of batches independent
    # trainValSplit = config.getfloat('training_hyperparameters', 'train_val_split')
    # numClasses = len(config.get('pruning_hyperparameters', 'sub_classes').split()) if subset != 'entire_dataset' else 100
    # datasetSizes = {'cifar100':{'entire_dataset':50000, 'subset1':17500, 'aquatic':5000, 'indoors':6000, 'natural':16000, 'random1':26000}} 
    # trainingImgsPerClass = 500
    # numBatches = trainValSplit * (5*numClasses) * trainingImgsPerClass
    # ftGops = [(numBatches * gops['ft']['unpruned']) if epoch < pruneAfter else (numBatches * gops['ft']['pruned']) for epoch in range(epochs)]

    infGops = gops['inf'] / batchSize
    ftGops = [gops['ft']['unpruned'] if epoch < pruneAfter else gops['ft']['pruned'] for epoch in range(epochs)]
    epochFtGops = list(np.cumsum(np.array(ftGops)))
    totalFtGops = epochFtGops[-1]

    if perEpoch:
        return (epochFtGops, gops['mem']['pruned'])
    else:
        return (infGops, totalFtGops, gops['mem']['pruned'])
#}}}

def plot_inf_gops_vs_acc(summaryData, subsetAgnosticSummaryData, saveLoc=None, time=False):
#{{{
    xAxis = 'InferenceGops' if not time else 'InferenceTime'
    yAxis = 'AvgTestAcc'

    subsets = set(list(summaryData['Dataset']))
    networks = set(list(summaryData['Network']))
    #plt.subplots returns fig,ax
    axes = {net:{subset:plt.subplots(1,1) for subset in subsets} for net in networks} 

    for (dataset, net), data in summaryData.groupby(['Dataset', 'Network']):
        title = 'Top1 Test Accuracy (%) for {} on {}'.format(net.capitalize(), dataset.capitalize()) 
        label = 'Subset Aware Pruning'
        
        ax = data.plot.scatter(x=xAxis, y=yAxis, ax=axes[net][dataset][1], label=label, title=title)
        
        xlabel = 'Inference GOps' if not time else 'Inference Time per image (ms)'
        # ax.set_xlabel('Inference GOps')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Test Accuracy (%)')
    
    for (subset, net), data in subsetAgnosticSummaryData.groupby(['Subset', 'Network']):
        for count, (pp, points) in enumerate(data.groupby(['PrunePerc'])):
            labels = ['Subset Agnostic Unpruned', 'Subset Agnostic Pruning', '']
            labelIdx = 0 if pp == "0" else 1 if count == 1 else 2
            marker = 'x' if pp == "0" else '+'
            
            # axes[net][subset][1].plot([float(points['InferenceGops'])], [float(points['TestAcc'])], label=labels[labelIdx], marker=marker, markersize=4, color='red', linestyle="None")
            axes[net][subset][1].plot([float(points[xAxis])], [float(points['TestAcc'])], label=labels[labelIdx], marker=marker, markersize=4, color='red', linestyle="None")
            axes[net][subset][1].legend()

    if saveLoc is not None:
        for net,plots in axes.items(): 
            for subset,plot in plots.items(): 
                plt.tight_layout()
                figFile = os.path.join(saveLoc, '{}_{}.png'.format(net, subset))
                plot[0].savefig(figFile)
#}}}

def plot_inf_gops_vs_acc_errorbar(summaryData, subsetAgnosticSummaryData, saveLoc=None, time=False):
#{{{
    xAxis = 'InferenceGops' if not time else 'InferenceTime'
    yAxis = 'AvgTestAcc'
    yErr = 'StdTestAcc'

    subsets = set(list(summaryData['Dataset']))
    networks = set(list(summaryData['Network']))
    #plt.subplots returns fig,ax
    axes = {net:plt.subplots(1,1) for net in networks} 
    errorBars = {net:{'Type':[], 'PrunePerc':[], yAxis:[], yErr:[], xAxis:[]} for net in networks}

    for (network, pp), data in summaryData.groupby(['Network', 'PrunePerc']):
        errorBars[network]['Type'].append('Subset Aware Pruning')
        errorBars[network]['PrunePerc'].append(pp)
        errorBars[network][yAxis].append(data[yAxis].mean())
        errorBars[network][yErr].append(data[yAxis].std())
        errorBars[network][xAxis].append(data[xAxis].mean())

    for (network, pp), data in subsetAgnosticSummaryData.groupby(['Network', 'PrunePerc']):
        ptType = 'Subset Agnostic Pruning' if pp != '0' else 'Unpruned'
        errorBars[network]['Type'].append(ptType)
        errorBars[network]['PrunePerc'].append(int(pp))
        errorBars[network][yAxis].append(data['TestAcc'].mean())
        errorBars[network][yErr].append(data['TestAcc'].std())
        errorBars[network][xAxis].append(data[xAxis].mean())

    errorBarsDf = {net : pd.DataFrame(val) for net,val in errorBars.items()}
    plotData = {
                'Unpruned' : {'colour': 'red', 'marker': 'x'},
                'Subset Agnostic Pruning' : {'colour':'blue', 'marker':'*'},
                'Subset Aware Pruning' : {'colour':'green', 'marker':'o'}
              }
    
    plt.rcParams['errorbar.capsize'] = 4
    for net,df in errorBarsDf.items():
        for ptType, data in df.groupby(['Type']):
            title = 'Top1 Test Accuracy (Mean +/- Std %) \n for {} across subsets'.format(net.capitalize()) 
            xlabel = 'Inference Ops (GOps)' if not time else 'Inference Time (ms)'
            ax = data.plot.scatter(x=xAxis, y=yAxis, ax=axes[net][1], label=ptType, yerr=yErr, color=plotData[ptType]['colour'], marker=plotData[ptType]['marker'], title=title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel('Test Accuracy (%)')
    
    if saveLoc is not None:
        for net,plot in axes.items(): 
           plt.tight_layout()
           figFile = os.path.join(saveLoc, '{}.png'.format(net))
           plot[0].savefig(figFile)
#}}}

def plot_ft_gops_vs_acc(summaryData):
#{{{
    axAccs = [plt.subplots(1,1)[1] for i in range(3)] #subplots returns fig,ax tuple
    xAxis = 'FinetuneGops'
    yAxis = 'AvgTestAcc'

    for (dataset, net), data in summaryData.groupby(['Dataset', 'Network']):
        colour = 'red' if 'mobilenetv2' in net else 'blue'
        title = 'Top1 Test Accuracy (%) for {} on Subset-{}'.format(net, dataset.capitalize()) 
        
        if 'entire_dataset' in dataset:
            ax = data.plot.scatter(x=xAxis, y=yAxis, ax=axAccs[0], c=colour, label=net, title=title)
        elif 'subset1' in dataset:
            ax = data.plot.scatter(x=xAxis, y=yAxis, ax=axAccs[1], c=colour, label=net, title=title)
        elif 'aquatic' in dataset:
            ax = data.plot.scatter(x=xAxis, y=yAxis, ax=axAccs[2], c=colour, label=net, title=title)
        
        ax.set_xlabel('Finetune GOps')
        ax.set_ylabel('Test Accuracy (%)')
#}}}

def plot_ft_gops_by_epoch(data, plotAsLine=None, _accMetric='Test_Top1'):
#{{{
    nets = list(data.keys())
    subsets = list(data[nets[0]].keys()) 
    prunePercs = list(data[nets[0]][subsets[0]].keys())

    colours = cm.rainbow(np.linspace(0,1,len(prunePercs))) 
    accMetric = _accMetric
    xAxis = 'Ft_Gops'
    lab = '{}-%'

    linePlots = ['25', '50', '75', '95'] if plotAsLine is None else plotAsLine

    for net in nets: 
        for subset in subsets:
            for i,pp in enumerate(prunePercs): 
                df = data[net][subset][pp]
                colour = [colours[i]]*len(df)
                
                if i == 0:
                    ax = df.plot(x=xAxis, y=accMetric, color=colour, label=lab.format(pp), title='{} vs Finetune Gops \n Network-{} on Subset-{}'.format(accMetric, net.capitalize(), subset.capitalize()))
                    ax.set_xlabel('Finetune Gops')
                    ax.set_ylabel(accMetric)
                else:
                    if pp in linePlots:
                        df.plot(x=xAxis, y=accMetric, ax=ax, color=colour, label=lab.format(pp))
                    
                    ax.plot([df[xAxis][len(df)-1]], [df[accMetric][len(df)-1]], marker="o", markersize=4, color='red')
                    ax.set_xlabel('Finetune Gops')
                    ax.set_ylabel(accMetric)
#}}}

