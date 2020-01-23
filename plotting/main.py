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

from src.ar4414.pruning.plotting.summary_stats import collector 
from src.ar4414.pruning.plotting.summary_stats import gops as mod_gops 
from src.ar4414.pruning.plotting.summary_stats import l1_norms as mod_l1_norms
from src.ar4414.pruning.plotting.summary_stats import channel_diff as mod_channel_diff

def parse_arguments():
#{{{
    print('Parsing Arguments')
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--channel_diff', action='store_true', help='plot difference in channels before and after finetuning')
    parser.add_argument('--inf_gops', action='store_true', help='plot inference gops vs test accuracy')
    parser.add_argument('--l1_norm', action='store_true', help='plot histograms of l1-norms and change in l1-norms before and after finetuning')
    parser.add_argument('--pretty_print', action='store_true', help='pretty print summary data table')

    args = parser.parse_args()
    
    return args
#}}}

if __name__ == '__main__':
    args = parse_arguments()

    if len(sys.argv) == 1:
        print('No arguments passed, hence nothing will run')
        sys.exit()

    # load json with log file locations
    with open('/home/ar4414/pytorch_training/src/ar4414/pruning/plotting/logs.json', 'r') as jFile:
        logs = json.load(jFile)
    
    networks = ['mobilenetv2', 'resnet']
    datasets = ['entire_dataset', 'subset1', 'aquatic']
    prunePercs = ['5', '10', '25', '50', '60', '75', '85', '95']
    
    if args.channel_diff or args.inf_gops:
        summaryData = collector.summary_statistics(logs, networks, datasets, prunePercs)
        
        if args.pretty_print:
            print(tabulate(summaryData, headers='keys', tablefmt='psql'))
    
    if args.l1_norm:
        normsDict = collector.l1_norm_statistics(logs, networks, datasets, prunePercs)
        
        if args.pretty_print:
            for net,v in normsDict.items():
                for dataset,df in v.items():
                    print("=============== L1-Norm per filter or Network {} and Subset {} ==================".format(net, dataset))
                    print(tabulate(df, headers='keys', tablefmt='psql'))
                    print()
    
    # plot difference in channels pruned by percentage pruned
    if args.channel_diff:
        mod_channel_diff.plot_channel_diff_by_pp(summaryData)

    # plot inference time vs accuracy tradeoff
    if args.inf_gops:
        mod_gops.plot_inf_gops_vs_acc(summaryData)

    # plot difference in l1-norms and l1-norms 
    if args.l1_norm: 
        mod_l1_norms.plot_histograms(normsDict)        
    
    plt.show()

