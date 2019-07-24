from __future__ import print_function

import argparse
import configparser as cp
import sys

import src.param_parser as ppSrc

class Params(ppSrc.Params) : 
    def __init__(self, config_file) : 
        super().__init__(config_file)
        
        self.getGops = config_file.getboolean('pruning_hyperparameters', 'get_gops')
        self.sub_classes = config_file.get('pruning_hyperparameters', 'sub_classes').split() 
        self.thisLayerUp = config_file.getint('pruning_hyperparameters', 'this_layer_up') 
        self.pruningPerc = config_file.getfloat('pruning_hyperparameters', 'pruning_perc')
        self.prunePercIncrement = config_file.getint('pruning_hyperparameters', 'iterative_pruning_increment') 
        self.pruneAfter = config_file.getint('pruning_hyperparameters', 'iterative_pruning_epochs') 
        self.pruneWeights = config_file.getboolean('pruning_hyperparameters', 'prune_weights')
        self.pruneFilters = config_file.getboolean('pruning_hyperparameters', 'prune_filters')
        self.finetune = config_file.getboolean('pruning_hyperparameters', 'finetune')
        assert not (self.pruneWeights == True and self.pruneFilters == True), 'Cannot prune both weights and filters'

        # internal pruning attributes 
        self.prunePercPerLayer = []

def parse_command_line_args() : 
    parser = argparse.ArgumentParser(description='PyTorch Pruning')

    # Command line vs Config File
    parser.add_argument('--config-file', default='None', type=str, help='config file with training parameters')
    
    args = parser.parse_args()

    return args
