import os
import glob
import json
import math

import configparser as cp

import matplotlib.pyplot as plt

def get_gops(basePath, log):
#{{{
    datasetSizes = {'cifar100':{'entire_dataset':50000, 'subset1':20000, 'aquatic':5000}} 

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

    numBatches = math.ceil(datasetSizes[dataset][subset]/batchSize)

    infGops = gops['inf'] / batchSize
    ftGops = sum([(numBatches * gops['ft']['unpruned']) if epoch < pruneAfter else (numBatches * gops['ft']['pruned']) for epoch in range(epochs)])

    return (infGops, ftGops, gops['mem']['pruned'])
#}}}

def plot_inf_gops_vs_acc(summaryData):
#{{{
    axAccs = [plt.subplots(1,1)[1] for i in range(3)] #subplots returns fig,ax tuple
    xAxis = 'InferenceGops'
    yAxis = 'AvgTestAcc'

    colours = {'mobilenetv2':'red', 'resnet':'blue', 'alexnet':'green', 'squeezenet':'orange'}

    for (dataset, net), data in summaryData.groupby(['Dataset', 'Network']):
        colour = colours[net]
        title = 'Top1 Test Accuracy (%) for {} on Subset-{}'.format(net, dataset.capitalize()) 
        
        if 'entire_dataset' in dataset:
            ax = data.plot.scatter(x=xAxis, y=yAxis, ax=axAccs[0], c=colour, label=net, title=title)
        elif 'subset1' in dataset:
            ax = data.plot.scatter(x=xAxis, y=yAxis, ax=axAccs[1], c=colour, label=net, title=title)
        elif 'aquatic' in dataset:
            ax = data.plot.scatter(x=xAxis, y=yAxis, ax=axAccs[2], c=colour, label=net, title=title)
        
        ax.set_xlabel('Inference GOps')
        ax.set_ylabel('Test Accuracy (%)')
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



