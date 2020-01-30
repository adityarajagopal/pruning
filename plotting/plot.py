import os
import sys
import math
import json
import argparse
import itertools

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from tabulate import tabulate
import matplotlib.pyplot as plt

# set python path so it points to src
parentDir = os.path.split(os.getcwd())
childDir = parentDir[1]
while childDir != 'src':
    parentDir = os.path.split(parentDir[0])
    childDir = parentDir[1]
sys.path.append(parentDir[0])

from src.ar4414.pruning.plotting import log_updater 
from src.ar4414.pruning.plotting.summary_stats import collector 
from src.ar4414.pruning.plotting.summary_stats import prune_search as searchSrc
from src.ar4414.pruning.plotting.summary_stats import gops as gopSrc 
from src.ar4414.pruning.plotting.summary_stats import l1_norms as l1NormsSrc
from src.ar4414.pruning.plotting.summary_stats import channel_diff as channeDiffSrc

def parse_arguments():
#{{{
    print('Parsing Arguments')
    parser = argparse.ArgumentParser()
    
    # plot data for only a subset of networks / datasets
    parser.add_argument('--networks', type=str, nargs='+', default=None, help='name of networks to display')
    parser.add_argument('--subsets', type=str, nargs='+', default=None, help='name of subsets to display')
    
    # types of plots
    parser.add_argument('--channel_diff', action='store_true', help='plot difference in channels before and after finetuning')
    parser.add_argument('--inf_gops', action='store_true', help='plot inference gops vs test accuracy')
    parser.add_argument('--ft_gops', action='store_true', help='plot finetune gops vs test accuracy')
    parser.add_argument('--l1_norm', action='store_true', help='plot histograms of l1-norms and change in l1-norms before and after finetuning')
    parser.add_argument('--pretty_print', action='store_true', help='pretty print summary data table')
    
    parser.add_argument('--ft_epoch_gops', action='store_true', help='plot finetune gops vs test accuracy')
    parser.add_argument('--plot_as_line', type=str, nargs='+', default=None, help='pruning percentages to plot as a line')
    parser.add_argument('--acc_metric', type=str, default='Test_Top1', help='y-axis metric : one of (Train_Top1, Test_Top1 and Val_Top1')
    
    parser.add_argument('--bin_search_cost', action='store_true', help='plot cost of binary search')
    parser.add_argument('--mode', type=str, default='memory_opt', help='how to prioritse binary search : one of memory_opt or cost_opt')
    
    # update logs.json with timestamps
    parser.add_argument('--update_logs', action='store_true', help='update logs.json file with relevant timestamps')
    parser.add_argument('--as_of', type=str, help='year-month-day including and after which to store the logs')
   
    # add network to logs.json
    parser.add_argument('--add_network', action='store_true', help='add a network to logs')
    parser.add_argument('--name', type=str, help='name of network to add')
    parser.add_argument('--pre_ft_path', type=str, help='path to model before finetuning')
    parser.add_argument('--base_folder', type=str, help='folder name where timestamped logs are to be placed')


    args = parser.parse_args()
    
    return args
#}}}

if __name__ == '__main__':
    args = parse_arguments()

    if len(sys.argv) == 1:
        print('No arguments passed, hence nothing will run')
        sys.exit()
    
    networks = ['alexnet', 'mobilenetv2', 'resnet', 'squeezenet'] if args.networks is None else args.networks
    # datasets = ['entire_dataset', 'subset1', 'aquatic'] if args.subsets is None else args.subsets
    datasets = ['subset1', 'aquatic'] if args.subsets is None else args.subsets
    prunePercs = [str(i) for i in range(5,100,5)]
    logsJson = '/home/ar4414/pytorch_training/src/ar4414/pruning/logs/logs.json'
    
    # load json with log file locations
    with open(logsJson, 'r') as jFile:
        logs = json.load(jFile)

    if args.add_network:
        print("==> Updating json with new network")
        logs = log_updater.add_network(logs, args.name, datasets, args.base_folder, args.pre_ft_path)
        with open(logsJson, 'w') as jFile:
            logs = json.dump(logs, jFile, indent=2)

    if args.update_logs: 
        print("==> Updating logs.json with new timestamps")
        logs = log_updater.update_timestamps(logs, networks, datasets, prunePercs, asOf=args.as_of)
        with open(logsJson, 'w') as jFile:
            logs = json.dump(logs, jFile, indent=2)

    if args.channel_diff or args.inf_gops or args.ft_gops:
        print("==> Collecting Accuracy and Gops statistics")
        summaryData = collector.summary_statistics(logs, networks, datasets, prunePercs)
        subsetAgnosticSummaryData = collector.subset_agnostic_summary_statistics(logs, networks, datasets, prunePercs)

        if args.pretty_print:
            print(tabulate(summaryData, headers='keys', tablefmt='psql'))
            print(tabulate(subsetAgnosticSummaryData, headers='keys', tablefmt='psql'))
    
    if args.ft_epoch_gops:
        print("==> Collecting cumulative gops by epoch statistics")
        gopsByEpochData = collector.per_epoch_statistics(logs, networks, datasets, prunePercs)
        print("==> Plotting cumulative gops by epoch statistics")
        gopSrc.plot_ft_gops_by_epoch(gopsByEpochData, args.plot_as_line, args.acc_metric)

    if args.bin_search_cost:
        print("==> Performing binary search to get pruning percentage that gives no accuracy loss")
        binSearchCost = searchSrc.bin_search_cost(logs, networks, datasets, prunePercs, args.mode)
        searchSrc.plot_bin_search_cost(binSearchCost)

    if args.l1_norm:
        print("==> L1-Norm Statistics")
        normsDict = collector.l1_norm_statistics(logs, networks, datasets, prunePercs)
        
        if args.pretty_print:
            for net,v in normsDict.items():
                for dataset,df in v.items():
                    print("=============== L1-Norm per filter or Network {} and Subset {} ==================".format(net, dataset))
                    print(tabulate(df, headers='keys', tablefmt='psql'))
                    print()
    
    # plot difference in channels pruned by percentage pruned
    if args.channel_diff:
        print("==> Plotting Difference in Channels Pruned before and after finetuning")
        channeDiffSrc.plot_channel_diff_by_pp(summaryData)

    # plot inference gops vs accuracy tradeoff
    if args.inf_gops:
        print("==> Plotting GOps for inference vs best test top1 accuracy obtained")
        gopSrc.plot_inf_gops_vs_acc(summaryData, subsetAgnosticSummaryData)
    
    # plot finetune gops vs accuracy tradeoff 
    if args.ft_gops:
        print("==> Plotting GOps for finetuning vs best test top1 accuracy obtained")
        gopSrc.plot_ft_gops_vs_acc(summaryData)

    # plot difference in l1-norms and l1-norms 
    if args.l1_norm: 
        print("==> Plotting l1-norm histograms and histograms in difference in l1-norm before and after finetuning")
        l1NormsSrc.plot_histograms(normsDict)        
     
    plt.show()

