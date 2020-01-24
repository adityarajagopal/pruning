from __future__ import print_function

import argparse
import configparser as cp
import sys

import src.param_parser as ppSrc

class Params(ppSrc.Params) : 
    def __init__(self, config_file) : 
        super().__init__(config_file)
        
        self.getGops = config_file.getboolean('pruning_hyperparameters', 'get_gops')
        # self.plotInferenceGops = config_file.getboolean('pruning_hyperparameters', 'plot_inference_gops', fallback=None)
        self.inferenceGops = config_file.getboolean('pruning_hyperparameters', 'inference_gops', fallback=None)
        self.logs = config_file.get('pruning_hyperparameters', 'logs', fallback=None)
        
        self.plotType = config_file.get('pruning_hyperparameters', 'plot_type', fallback='joint')
        assert (self.plotType == 'number' or self.plotType == 'hamming' or self.plotType == 'joint'), 'Plot Type must be number, hamming or joint - provided {}'.format(self.plotType)
        self.plotChannels = config_file.get('pruning_hyperparameters', 'plot_channels', fallback='').split()

        self.changeInRanking = config_file.getboolean('pruning_hyperparameters', 'change_in_rank', fallback=False)
        
        self.subsetName = config_file.get('pruning_hyperparameters', 'sub_name', fallback='subset1')
        self.sub_classes = config_file.get('pruning_hyperparameters', 'sub_classes').split() 
        
        self.retrain = config_file.getboolean('pruning_hyperparameters', 'retrain', fallback=False)
        self.channelsPruned = config_file.get('pruning_hyperparameters', 'channels_pruned', fallback='')
        
        # --------------------------------  
        self.finetune = config_file.getboolean('pruning_hyperparameters', 'finetune')
        self.static = config_file.getboolean('pruning_hyperparameters', 'static', fallback=True)
        self.thisLayerUp = config_file.getint('pruning_hyperparameters', 'this_layer_up') 
        self.pruningPerc = config_file.getfloat('pruning_hyperparameters', 'pruning_perc')
        self.prunePercIncrement = config_file.getint('pruning_hyperparameters', 'iterative_pruning_increment') 
        self.iterPruneInc = config_file.getint('pruning_hyperparameters', 'iterative_pruning_increment') 
        if self.static:
            self.pruneAfter = config_file.getint('pruning_hyperparameters', 'prune_after', fallback=-2) 
        self.finetuneBudget = config_file.getint('pruning_hyperparameters', 'finetune_budget', fallback=0) 
       
        self.pruneWeights = config_file.getboolean('pruning_hyperparameters', 'prune_weights')
        self.pruneFilters = config_file.getboolean('pruning_hyperparameters', 'prune_filters')
        self.pruningMetric = config_file.get('pruning_hyperparameters', 'metric', fallback='filters')
        assert not (self.pruneWeights == True and self.pruneFilters == True), 'Cannot prune both weights and filters'
       
        # --------------------------------  
        self.fbsPruning = config_file.getboolean('pruning_hyperparameters', 'fbs_pruning', fallback=False)
        self.fbsFinetune = config_file.getboolean('pruning_hyperparameters', 'fbs_finetune', fallback=False)
        self.unprunedRatio = config_file.getfloat('pruning_hyperparameters', 'unpruned_ratio', fallback=1.0)
        self.unprunedLB = config_file.getfloat('pruning_hyperparameters', 'unpruned_lb', fallback=0.1)
        self.batchLim = config_file.getint('pruning_hyperparameters', 'batch_lim', fallback=-1)
        self.logDir = config_file.get('pruning_hyperparameters', 'logdir', fallback='/home/ar4414/pytorch_training/src/ar4414/pruning/logs')
        self.logFiles = config_file.get('pruning_hyperparameters', 'logfiles', fallback='').split()
        
        # --------------------------------  
        self.entropy = config_file.getboolean('entropy_hyperparameters', 'entropy', fallback=False)
        self.eChannels = config_file.getint('entropy_hyperparameters', 'channels', fallback=-1)
        self.numBatches = config_file.getint('entropy_hyperparameters', 'num_batches', fallback=1)
        self.eLayers = config_file.get('entropy_hyperparameters', 'layers', fallback=[])
        if self.eLayers != []:
            self.eLayers = self.eLayers.split()
        # self.entropyLocalPruning = config_file.getboolean('entropy_hyperparameters', 'entropy_local_pruning', fallback=False)
        self.entropyGlobalPruning = config_file.getboolean('entropy_hyperparameters', 'entropy_global_pruning', fallback=False)
        # assert not (self.entropyLocalPruning == True and self.entropyGlobalPruning == True), 'Cannot prune by ranking both locally and globally'

        # internal pruning attributes 
        self.prunePercPerLayer = []

def parse_command_line_args() : 
    parser = argparse.ArgumentParser(description='PyTorch Pruning')

    # Command line vs Config File
    parser.add_argument('--config-file', default='None', type=str, help='config file with training parameters')
    
    args = parser.parse_args()

    return args
