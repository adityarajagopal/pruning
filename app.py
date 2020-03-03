from src.ar4414.pruning.timers import Timer
import src.ar4414.pruning.param_parser as ppSrc 
import src.ar4414.pruning.model_creator as mcSrc
import src.ar4414.pruning.training as trainingSrc
import src.ar4414.pruning.inference as inferenceSrc
import src.ar4414.pruning.checkpointing as checkpointingSrc

import src.ar4414.pruning.pruners.base as pruningSrc
from src.ar4414.pruning.pruners.alexnet import AlexNetPruning
from src.ar4414.pruning.pruners.squeezenet import SqueezeNetPruning 
from src.ar4414.pruning.pruners.resnet import ResNet20PruningDependency as ResNetPruning
from src.ar4414.pruning.pruners.mobilenetv2 import MobileNetV2PruningDependency as MobileNetV2Pruning 

import src.app as appSrc
import src.input_preprocessor as preprocSrc

import os
import sys
import json
import time
import random
import subprocess

import configparser as cp

import torch
import torch.cuda
import torch.nn as nn

import math
import numpy as np

class Application(appSrc.Application):
    def main(self):
    #{{{
        self.setup_dataset()
        self.setup_model()
        self.setup_tee_printing()
        self.setup_pruners()

        if self.params.profilePruning: 
            # setting class attribute enabled will set enabled 
            # true for all objects of this class
            Timer.enabled = True

        if self.params.pruneFilters == True:
        #{{{
            print('=========Baseline Accuracy==========')
            testStats = self.run_inference()
            print('==========================')

            if self.params.finetune:
            #{{{
                # run finetuning
                self.run_finetune()

                # if timers have been set, log timers 
                if self.params.profilePruning: 
                #{{{
                    data = {'training':None, 'inference':None}
                    data['training'] = {epoch:self.trainer.dataTimer.stats[epoch] + time + self.trainer.pruneTimer.stats[epoch] for epoch,time in self.trainer.mbTimer.stats.items()} 
                    data['inference'] = self.inferer.infTimer.stats['minibatch_time']
                    logDir = 'profiling_logs/tx2/{}/{}/{}'.format(self.params.arch, self.params.dataset, int(self.params.pruningPerc))
                    Timer.log_dict(logDir, data)
                #}}}
            #}}}

            elif self.params.retrain:
            #{{{
                self.model, self.optimiser = self.pruner.get_random_init_model()
                self.run_training()
            #}}}

            else:
            #{{{
                channelsPruned, prunedModel, optimiser = self.pruner.prune_model(self.model)
                pruneRate, prunedSize, origSize = self.pruner.prune_rate(prunedModel)
                print('Pruned Percentage = {:.2f}%, NewModelSize = {:.2f}MB, OrigModelSize = {:.2f}MB'.format(pruneRate, prunedSize, origSize))
                self.inferer.test_network(self.params, self.test_loader, prunedModel, self.criterion, optimiser)
                print('==========================')
            #}}}
        #}}}

        elif self.params.evaluate == True:
            self.run_inference()

        else : 
            self.run_training()
    #}}} 
    
    def setup_pruners(self):
    #{{{
        if 'alexnet' in self.params.arch:
            self.pruner = AlexNetPruning(self.params, self.model)
            self.netName = 'AlexNet'
            self.trainableLayers = ['classifier']
        elif 'resnet' in self.params.arch:
            self.pruner = ResNetPruning(self.params, self.model)
            self.netName = 'ResNet{}'.format(self.params.depth)
            self.trainableLayers = ['fc']
        elif 'mobilenet' in self.params.arch:
            self.pruner = MobileNetV2Pruning(self.params, self.model)
            self.netName = 'MobileNetv2'
            self.trainableLayers = ['linear']
        elif 'squeezenet' in self.params.arch:
            self.pruner = SqueezeNetPruning(self.params, self.model)
            self.netName = 'SqueezeNet'
            self.trainableLayers = ['module.conv2']
        else:
            raise ValueError("Pruning not implemented for architecture ({})".format(self.params.arch))
    #}}}

    def setup_lr_schedule(self):
    #{{{
        # adjust lr based on pruning percentage
        if self.params.static:
        #{{{
            initLrIdx = self.params.lr_schedule.index(self.params.pruneAfter)
            initLrIdx = initLrIdx-1 if initLrIdx != 0 else initLrIdx+1
            initLr = self.params.lr_schedule[initLrIdx]
            initPrunedLrIdx = self.params.lr_schedule.index(self.params.pruneAfter) + 1

            initPrunedLr = initLr / (self.params.gamma)
            listEnd = initPrunedLrIdx + 5
            
            self.params.lr_schedule[initPrunedLrIdx] = initPrunedLr
            self.params.lr_schedule = self.params.lr_schedule[:listEnd]
        #}}}
        else:
        #{{{
            initLr = self.params.lr_schedule[1]
            initPrunedLrIdx = 3
            
            if self.params.pruningPerc <= 25.0:
                initPrunedLr = initLr
                listEnd = initPrunedLrIdx + 1
            else:
                initPrunedLr = initLr / (self.params.gamma * self.params.gamma)
                listEnd = initPrunedLrIdx + 5
            
            self.params.lr_schedule[initPrunedLrIdx] = initPrunedLr
            self.params.lr_schedule = self.params.lr_schedule[:listEnd]
        #}}}
    #}}}
    
    def run_finetune(self):
    #{{{
        if self.params.entropy and self.params.entropyGlobalPruning:
            print('==> Performing Activation Entropy Pruning Finetune')
            self.trainer.finetune_entropy(self.params, self.pruner, self.checkpointer, self.train_loader, self.test_loader, self.valLoader, self.model, self.criterion, self.optimiser, self.inferer) 
        elif self.params.pruneFilters:
            print('==> Performing l1-weight Pruning Finetune')
            if self.params.static:
                self.trainer.static_finetune_l1_weights(self.params, self.pruner, self.checkpointer, self.train_loader, self.test_loader, self.valLoader, self.model, self.criterion, self.optimiser, self.inferer) 
            else:
                self.trainer.validation_finetune_l1_weights(self.params, self.pruner, self.checkpointer, self.train_loader, self.test_loader, self.valLoader, self.model, self.criterion, self.optimiser, self.inferer) 
    #}}}
    
    def setup_param_checkpoint(self, configFile):
    #{{{
        config = cp.ConfigParser() 
        config.read(configFile)
        self.params = ppSrc.Params(config)
        self.checkpointer = checkpointingSrc.Checkpointer(self.params, configFile)
        self.setup_params()
    #}}}
    
    def setup_others(self):
    #{{{
        self.preproc = preprocSrc.Preproc()
        self.mc = mcSrc.ModelCreator()
        self.trainer = trainingSrc.Trainer(self.params)
        self.inferer = inferenceSrc.Inferer()
    #}}}

    def run_inference(self):
    #{{{
        # perform inference only
        print('==> Performing Inference')
        return self.inferer.test_network(self.params, self.test_loader, self.model, self.criterion, self.optimiser)
    #}}}
       
