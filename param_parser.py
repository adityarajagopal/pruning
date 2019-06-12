from __future__ import print_function

import argparse
import configparser as cp

class Params() : 
    def __init__(self, config_file) : 
        # attributes read in from config file 
        self.dataset = config_file.get('dataset', 'dataset')
        self.data_location = config_file.get('dataset', 'dataset_location')

        self.arch = config_file.get('cnn', 'architecture')        
        self.depth = config_file.get('cnn', 'depth')       
        self.cardinality = config_file.get('cnn', 'cardinality')
        self.widen_factor = config_file.get('cnn', 'widen_factor')
        self.growth_rate = config_file.get('cnn', 'growth_rate')
        self.compression_rate = config_file.get('cnn', 'compression_rate')

        # self.start_epoch = config_file.getint('training_hyperparameters', 'start_epoch')
        self.printOnly = config_file.getboolean('training_hyperparameters', 'print_only')
        self.epochs = config_file.getint('training_hyperparameters', 'total_epochs')
        self.train_batch = config_file.getint('training_hyperparameters', 'train_batch')
        self.test_batch = config_file.getint('training_hyperparameters', 'test_batch') 
        self.lr = config_file.getfloat('training_hyperparameters', 'learning_rate')
        self.dropout = config_file.getfloat('training_hyperparameters', 'dropout_ratio')
        self.gamma = config_file.getfloat('training_hyperparameters', 'gamma')
        self.momentum = config_file.getfloat('training_hyperparameters', 'momentum') 
        self.weight_decay = config_file.getfloat('training_hyperparameters', 'weight_decay') 
        self.mo_schedule = [self.__to_num(i) for i in config_file.get('training_hyperparameters', 'momentum_schedule').split()]
        self.lr_schedule = [self.__to_num(i) for i in config_file.get('training_hyperparameters', 'lr_schedule').split()]
        self.trainValSplit = config_file.getfloat('training_hyperparameters', 'train_val_split')
        
        self.sub_classes = config_file.get('pruning_hyperparameters', 'sub_classes').split() 
        self.getGops = config_file.getboolean('pruning_hyperparameters', 'get_gops')
        self.pruneType = config_file.get('pruning_hyperparameters', 'prune_type')
        
        # self.this_layer_up = config_file.getint('pruning_hyperparameters', 'this_layer_up') 
        # self.pruneAfter = config_file.getint('pruning_hyperparameters', 'iterative_pruning_epochs') 
        # self.pruneIncrement = config_file.getint('pruning_hyperparameters', 'iterative_pruning_increment') 
        # self.finetune = config_file.getboolean('pruning_hyperparameters', 'finetune')
        # self.prune_weights = config_file.getboolean('pruning_hyperparameters', 'prune_weights')
        # self.prune_filters = config_file.getboolean('pruning_hyperparameters', 'prune_filters')
        # self.pruning_perc = config_file.getfloat('pruning_hyperparameters', 'pruning_perc')
        # self.pollution = config_file.getfloat('pruning_hyperparameters', 'test_set_pollution')
        # assert not (self.prune_weights == True and self.prune_filters == True), 'Cannot prune both weights and filters'

        self.manual_seed = config_file.getint('pytorch_parameters', 'manual_seed')
        self.workers = config_file.getint('pytorch_parameters', 'data_loading_workers')
        self.gpu_id = config_file.get('pytorch_parameters', 'gpu_id')
        self.pretrained = config_file.get('pytorch_parameters', 'pretrained')        
        self.checkpoint = config_file.get('pytorch_parameters', 'checkpoint_path')
        self.test_name = config_file.get('pytorch_parameters', 'test_name')
        self.resume = config_file.getboolean('pytorch_parameters', 'resume')
        self.branch = config_file.getboolean('pytorch_parameters', 'branch')
        self.evaluate = config_file.getboolean('pytorch_parameters', 'evaluate')
        self.tee_printing = config_file.get('pytorch_parameters', 'tee_printing')

        # attributes used internally
        self.use_cuda = True
        self.start_epoch = 0 
        self.curr_epoch = 0 
        self.train_loss = 0 
        self.train_top1 = 1
        self.train_top5 = 1
        self.test_loss = 0 
        self.test_top1 = 1
        self.test_top5 = 1
        self.bestValidLoss = 0.0

    def get_state(self) : 
        return self.__dict__

    def __to_num(self, x) : 
        try : 
            return int(x) 
        except ValueError: 
            return float(x) 

def parse_command_line_args() : 
    parser = argparse.ArgumentParser(description='PyTorch Pruning')

    # Command line vs Config File
    parser.add_argument('--config-file', default='None', type=str, help='config file with training parameters')
    
    args = parser.parse_args()

    return args
