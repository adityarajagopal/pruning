import os
import sys
import math
import json
import argparse
import itertools
import subprocess

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
from src.ar4414.pruning.plotting.summary_stats import gops as gopSrc 
from src.ar4414.pruning.plotting.summary_stats import l1_norms as l1NormsSrc
from src.ar4414.pruning.plotting.summary_stats import tradeoff as tradeoffSrc
from src.ar4414.pruning.plotting.summary_stats import prune_search as searchSrc
from src.ar4414.pruning.plotting.summary_stats import channel_diff as channeDiffSrc

def parse_arguments():
#{{{
    print('Parsing Arguments')
    parser = argparse.ArgumentParser()
    
    # plot data for only a subset of networks / datasets
    parser.add_argument('--networks', type=str, nargs='+', default=None, help='name of networks to display')
    parser.add_argument('--subsets', type=str, nargs='+', default=None, help='name of subsets to display')
    parser.add_argument('--logs_json', type=str, default='/home/ar4414/pytorch_training/src/ar4414/pruning/logs/logs_v1.json', help='full file path of json file where logs summary to be placed')
    parser.add_argument('--prof_logs', type=str, default=None, help='path to profiling logs - filepath after this should be <net>/<dataset>/<pruning_perc>/timing_data.pth.tar')
    
    parser.add_argument('--silent', action='store_true', help="don't show figures")
    parser.add_argument('--save', action='store_true', help='save figures')
    parser.add_argument('--loc', type=str, default='recent', help='folder under graphs/ where images should be saved')
    
    # add network to logs.json
    parser.add_argument('--add_network', action='store_true', help='add a network to logs')
    parser.add_argument('--pre_ft_path', type=str, default=None, help='path to model before finetuning')
    
    # add dataset to all networks in logs.json
    parser.add_argument('--add_dataset', action='store_true', help='add a dataset to all networks in logs')
    
    # common arguments between above 2 
    parser.add_argument('--name', type=str, help='name of network/dataset to add')
    parser.add_argument('--base_folder', type=str, help='folder name where timestamped logs are to be placed')
    
    # update logs.json with timestamps
    parser.add_argument('--update_logs', action='store_true', help='update logs.json file with relevant timestamps')
    parser.add_argument('--as_of', type=str, help='year-month-day including and after which to store the logs')

    # types of plots
    parser.add_argument('--ft_gops', action='store_true', help='plot finetune gops vs test accuracy')
    parser.add_argument('--l1_norm', action='store_true', help='plot histograms of l1-norms and change in l1-norms before and after finetuning')
    parser.add_argument('--pretty_print', action='store_true', help='pretty print summary data table')
    
    parser.add_argument('--channel_diff', action='store_true', help='plot difference in channels before and after finetuning')
    parser.add_argument('--pre_post_ft', action='store_true', help='plot difference in channels per network and subset between pre-post ft and models reached post ft')
    parser.add_argument('--across_networks', action='store_true', help='plot difference in channels before and after finetuning compared across networks')
    
    parser.add_argument('--inf_gops', action='store_true', help='plot inference gops vs test accuracy')
    parser.add_argument('--subset_agnostic_logs', type=str, default='/home/ar4414/pytorch_training/src/ar4414/pruning/logs/subset_agnostic_logs.json', help='full file path of json file where logs for subset agnostic pruning was performed')
    
    parser.add_argument('--ft_epoch_gops', action='store_true', help='plot finetune gops vs test accuracy')
    parser.add_argument('--plot_as_line', type=str, nargs='+', default=None, help='pruning percentages to plot as a line')
    parser.add_argument('--acc_metric', type=str, default='Test_Top1', help='y-axis metric : one of (Train_Top1, Test_Top1 and Val_Top1')
    
    parser.add_argument('--bin_search_cost', action='store_true', help='plot cost of binary search')
    parser.add_argument('--mode', type=str, default='memory_opt', help='how to prioritse binary search : one of memory_opt or cost_opt')

    parser.add_argument('--time_tradeoff', action='store_true', help='plot inf_time vs training_time for various points that match accuracy')
    
    args = parser.parse_args()
    
    return args
#}}}

def get_save_location(args):
#{{{
    saveLoc = None
    if args.save:
        if args.bin_search_cost:
            if args.prof_logs is None:
                saveLoc = '/home/ar4414/pytorch_training/src/ar4414/pruning/graphs/{}/bin_search_cost/{}/'.format(args.loc, args.mode)
            else:
                saveLoc = '/home/ar4414/pytorch_training/src/ar4414/pruning/graphs/{}/bin_search_cost_time/{}/'.format(args.loc, args.mode)
        elif args.inf_gops:
            if args.prof_logs is None:
                saveLoc = '/home/ar4414/pytorch_training/src/ar4414/pruning/graphs/{}/inference_gops/'.format(args.loc)
            else:
                saveLoc = '/home/ar4414/pytorch_training/src/ar4414/pruning/graphs/{}/inference_time/'.format(args.loc)
        elif args.time_tradeoff:
            saveLoc = '/home/ar4414/pytorch_training/src/ar4414/pruning/graphs/{}/search_inf_time_tradeoff/'.format(args.loc)
            # saveLoc = '/home/ar4414/pytorch_training/src/ar4414/pruning/graphs/{}/search_inf_time_tradeoff_v1/'.format(args.loc)
        elif args.pre_post_ft:
            saveLoc = '/home/ar4414/pytorch_training/src/ar4414/pruning/graphs/{}/difference_in_channels_pruned_per_network_subset/'.format(args.loc)
        elif args.across_networks:
            saveLoc = '/home/ar4414/pytorch_training/src/ar4414/pruning/graphs/{}/difference_in_channels_pruned_per_subset/'.format(args.loc)
        elif args.l1_norm:
            saveLoc = '/home/ar4414/pytorch_training/src/ar4414/pruning/graphs/{}/difference_in_l1_norm_of_weights/'.format(args.loc)

        print("Saving graphs to {}".format(saveLoc))
        if not os.path.isdir(saveLoc):
            cmd = 'mkdir -p ' + saveLoc 
            subprocess.check_call(cmd, shell=True)
    
    return saveLoc
#}}}

if __name__ == '__main__':
#{{{
    args = parse_arguments()

    if len(sys.argv) == 1:
        print('No arguments passed, hence nothing will run')
        sys.exit()
    
    networks = ['alexnet', 'mobilenetv2', 'resnet', 'squeezenet'] if args.networks is None else args.networks
    datasets = ['subset1', 'aquatic', 'indoors', 'natural', 'random1'] if args.subsets is None else args.subsets
    prunePercs = [str(i) for i in range(5,100,5)]
    logsJson = args.logs_json 
    
    # load json with log file locations
    try:
        with open(logsJson, 'r') as jFile:
            logs = json.load(jFile)
    except FileNotFoundError:
        with open(logsJson, 'w+') as jFile:
            emptyJson = {}
            json.dump(emptyJson, jFile)
        with open(logsJson, 'r') as jFile:
            logs = json.load(jFile)

    if args.add_network:
    #{{{
        print("==> Updating json with new network")
        logs = log_updater.add_network(logs, args.name, datasets, args.base_folder, args.pre_ft_path)
        with open(logsJson, 'w') as jFile:
            logs = json.dump(logs, jFile, indent=2)
    #}}}

    if args.add_dataset: 
    #{{{
        print("==> Updating json with new dataset")
        logs = log_updater.add_dataset(logs, args.name, args.base_folder)
        with open(logsJson, 'w') as jFile:
            logs = json.dump(logs, jFile, indent=2)
    #}}}

    if args.update_logs: 
    #{{{
        print("==> Updating logs.json with new timestamps")
        logs = log_updater.update_timestamps(logs, networks, datasets, prunePercs, asOf=args.as_of)
        with open(logsJson, 'w') as jFile:
            logs = json.dump(logs, jFile, indent=2)
    #}}}

    if args.channel_diff or args.inf_gops or args.ft_gops:
    #{{{
        print("==> Collecting Accuracy and Gops statistics")
        summaryData, pruneAfter = collector.summary_statistics(logs, networks, datasets, prunePercs)
        
        with open(args.subset_agnostic_logs, 'r') as jFile:
            subsetAgnosticLogs = json.load(jFile)
        subsetAgnosticSummaryData = collector.subset_agnostic_summary_statistics(logs, networks, datasets, prunePercs, subsetAgnosticLogs)

        if args.prof_logs is not None:
            infTime, _ = collector.timing_statistics(pruneAfter, args.prof_logs, networks, ['cifar100'], prunePercs)  
        
            sdInf = []
            for idx, row in summaryData.iterrows(): 
                sdInf.append(infTime[row['Network']][str(row['PrunePerc'])])
            summaryData['InferenceTime'] = sdInf
            
            sasdInf = []
            for idx, row in subsetAgnosticSummaryData.iterrows(): 
                sasdInf.append(infTime[row['Network']][str(row['PrunePerc'])])
            subsetAgnosticSummaryData['InferenceTime'] = sasdInf

        if args.pretty_print:
            print(tabulate(summaryData, headers='keys', tablefmt='psql'))
            print(tabulate(subsetAgnosticSummaryData, headers='keys', tablefmt='psql'))
            sys.exit()
    
        saveLoc = get_save_location(args) 
        # plot inference gops vs accuracy tradeoff
        if args.inf_gops:
            print("==> Plotting GOps for inference vs best test top1 accuracy obtained")
            gopSrc.plot_inf_gops_vs_acc(summaryData, subsetAgnosticSummaryData, saveLoc, (args.prof_logs is not None))
    
        # plot difference in channels pruned by percentage pruned
        if args.pre_post_ft:
            print("==> Plotting change in channels pruned between pre-post ft and post ft models")
            channeDiffSrc.plot_channel_diff_by_pp(summaryData, saveLoc)
        
        # plot difference in channels pruned by percentage pruned
        if args.across_networks:
            print("==> Plotting Difference in Channels Pruned before and after finetuning")
            channeDiffSrc.plot_channel_diff_by_subset(summaryData, saveLoc)
    
        # plot finetune gops vs accuracy tradeoff 
        if args.ft_gops:
            print("==> Plotting GOps for finetuning vs best test top1 accuracy obtained")
            gopSrc.plot_ft_gops_vs_acc(summaryData)
    #}}}

    if args.bin_search_cost:
    #{{{
        print("==> Performing binary search to get pruning percentage that gives no accuracy loss")
        binSearchCost = searchSrc.bin_search_cost(logs, networks, datasets, prunePercs, args.mode, args.prof_logs)
        saveLoc = get_save_location(args)
        searchSrc.plot_bin_search_cost(binSearchCost, saveLoc, (args.prof_logs is not None))
    #}}}

    if args.time_tradeoff:
    #{{{
        print('==> Plotting inference time vs training time tradeoff across points that produce no accuracy loss')
        
        with open(args.subset_agnostic_logs, 'r') as jFile:
            subsetAgnosticLogs = json.load(jFile)
        subsetAgnosticSummaryData = collector.subset_agnostic_summary_statistics(logs, networks, datasets, prunePercs, subsetAgnosticLogs)
        targetData = subsetAgnosticSummaryData[subsetAgnosticSummaryData['PrunePerc'] == '0']
        
        binSearchMemoryOptimised = searchSrc.bin_search_cost(logs, networks, datasets, prunePercs, 'memory_opt', args.prof_logs, targetData)
        infTime, _ = collector.timing_statistics(5, args.prof_logs, networks, ['cifar100'], prunePercs)  
        
        tradeoffPoints = tradeoffSrc.get_tradeoff_points(binSearchMemoryOptimised, infTime, targetData, 5)
        
        saveLoc = get_save_location(args)
        tradeoffSrc.plot_tradeoff(tradeoffPoints, saveLoc)
    #}}}
    
    if args.l1_norm:
    #{{{
        print("==> L1-Norm Statistics")
        normsDict = collector.l1_norm_statistics(logs, networks, datasets, prunePercs)
        
        print("==> Plotting l1-norm histograms and histograms in difference in l1-norm before and after finetuning")
        saveLoc = get_save_location(args) 
        l1NormsSrc.plot_histograms(normsDict, saveLoc)        
        
        if args.pretty_print:
            for net,v in normsDict.items():
                for dataset,df in v.items():
                    print("=============== L1-Norm per filter or Network {} and Subset {} ==================".format(net, dataset))
                    print(tabulate(df, headers='keys', tablefmt='psql'))
                    print()
    #}}}
    
    if args.ft_epoch_gops:
    #{{{
        print("==> Collecting cumulative gops by epoch statistics")
        gopsByEpochData = collector.per_epoch_statistics(logs, networks, datasets, prunePercs)
        print("==> Plotting cumulative gops by epoch statistics")
        gopSrc.plot_ft_gops_by_epoch(gopsByEpochData, args.plot_as_line, args.acc_metric)
    #}}}
    
    if not args.silent:
        plt.show()
#}}}


