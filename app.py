import src.ar4414.pruning.gop_calculator as gopSrc
import src.ar4414.pruning.param_parser as ppSrc 
import src.ar4414.pruning.model_creator as mcSrc
import src.ar4414.pruning.inference as inferenceSrc
import src.ar4414.pruning.checkpointing as checkpointingSrc
import src.ar4414.pruning.training as trainingSrc

import src.ar4414.pruning.pruners.base as pruningSrc
from src.ar4414.pruning.pruners.alexnet import AlexNetPruning
from src.ar4414.pruning.pruners.resnet import ResNet20PruningDependency as ResNetPruning
from src.ar4414.pruning.pruners.mobilenetv2 import MobileNetV2PruningDependency as MobileNetV2Pruning 
from src.ar4414.pruning.pruners.squeezenet import SqueezeNetPruning 

import src.app as appSrc
import src.input_preprocessor as preprocSrc

import os
import random
import sys
import json
import subprocess
import time

import configparser as cp

import torch
import torch.cuda
import torch.nn as nn

# import matplotlib.pyplot as plt
# import matplotlib
import math
import numpy as np

class Application(appSrc.Application):
    def main(self):
    #{{{
        self.setup_dataset()
        self.setup_model()
        self.setup_tee_printing()
        self.setup_pruners()

        if self.params.getGops: 
        #{{{
            logs = self.params.logs
            with open(logs, 'r') as jFile:
                logs = json.load(jFile)    
            log = logs[self.params.arch][self.params.subsetName]
            prunedPercs = list(filter(lambda x: (x != 'base_path') and ('_inference' not in x), list(log.keys())))
            
            if self.params.inferenceGops:
            #{{{
                for pp in prunedPercs:
                    for run in log[pp]:
                        finetunePath = os.path.join(log['base_path'], "pp_{}/{}/orig".format(pp, run))
                        
                        # get inference gops for pruned model
                        self.model, self.optimiser = self.pruner.get_random_init_model(finetunePath)
                        _,tfg,_,_ = gopSrc.calc_inference_gops(self)
                        gopSrc.store_gops_json(finetunePath, inf=tfg)                        

                        print("{},{:.3f}".format(finetunePath, tfg))
            #}}}

            else:
            #{{{
                for pp in prunedPercs:
                    for run in log[pp]:
                        finetunePath = os.path.join(log['base_path'], "pp_{}/{}/orig".format(pp, run))
                        
                        #get unpruned training gops
                        _,tfg,_,tbg = gopSrc.calc_training_gops(self)
                        unprunedTrainGops = tfg + tbg
                        unprunedModelMB = mcSrc.get_model_size(self.model)                   
                        
                        #get pruned gops
                        try:
                            with open(os.path.join(finetunePath, 'pruned_channels.json'), 'r') as jFile:
                                channelsPruned = json.load(jFile)
                        except FileNotFoundError:
                            print("'pruned_channesl.json' does'n exist in log dir {}".format(finetunePath))
                            print("Check path or run finetuning to generate file")
                
                        prunedModel, optimiser = self.pruner.get_random_init_model(finetunePath)
                        _,tfg,_,tbg = gopSrc.calc_training_gops(self, optimiser, prunedModel)
                        prunedTrainGops = tfg + tbg
                        prunedModelMB = mcSrc.get_model_size(prunedModel)        

                        gopSrc.store_gops_json(finetunePath, unpruned=unprunedTrainGops, pruned=prunedTrainGops, memUP=unprunedModelMB, memP=prunedModelMB)
                        
                        print("{},{:.3f},{:.3f},{:.3f},{:.3f}".format(finetunePath, unprunedTrainGops, unprunedModelMB, prunedTrainGops, prunedModelMB))
            #}}}
        #}}}

        elif self.params.unprunedTestAcc: 
        #{{{
            logs = self.params.logs
            with open(logs, 'r') as jFile:
                logJson = json.load(jFile)    
            
            _,tfg,_,_ = gopSrc.calc_inference_gops(self)
            infGopsPerImage = tfg / self.params.test_batch

            logNetSub = logJson[self.params.arch][self.params.subsetName]
            loss, top1, top5 = self.run_inference()
            
            logNetSub['unpruned_inference'] = {'test_top1':top1, 'gops':infGopsPerImage}
            
            with open(logs, 'w') as jFile: 
                json.dump(logJson, jFile, indent=2)
        #}}}

        elif self.params.prunedTestAcc:
        #{{{
            logs = self.params.logs
            with open(logs, 'r') as jFile:
                logs = json.load(jFile)    
            log = logs[self.params.arch][self.params.trainedOn]
            prunedPercs = list(filter(lambda x: (x != 'base_path') and ('_inference' not in x), list(log.keys())))
                    
            log['{}_inference'.format(self.params.subsetName)] = {pp:0 for pp in prunedPercs}
            
            for pp in prunedPercs:
                print("==========> Pruning Perc = {}".format(pp))
                for run in log[pp]:
                    finetunePath = os.path.join(log['base_path'], "pp_{}/{}/orig".format(pp, run))

                    # get inference acc for pruned model
                    self.model, self.optimiser = self.pruner.get_random_init_model(finetunePath)
                    self.params.pretrained = os.path.join(finetunePath, "best-model.pth.tar")
                    self.params.evaluate = True
                    self.model = self.mc.load_pretrained(self.params, self.model)
                    loss, top1, top5 = self.run_inference()
                    
                    #update log file
                    log['{}_inference'.format(self.params.subsetName)][pp] = top1 
            
            with open(self.params.logs, 'w') as jFile:
                json.dump(logs, jFile, indent=2)
        #}}}

        elif self.params.noFtChannelsPruned:
        #{{{
            logs = self.params.logs
            with open(logs, 'r') as jFile:
                logs = json.load(jFile)    
            logs = logs[self.params.arch][self.params.subsetName]
            prunePercs = list(logs.keys())
            prunePercs.remove('base_path')

            for pp in prunePercs:
                if pp == 0.:
                    continue
                
                print("=========== Prune Perc = {}% ===========".format(pp))
                self.params.pruningPerc = int(pp)
                self.setup_pruners()
                preFtChannelsPruned = self.pruner.structured_l1_weight(self.model)
    
                logLocation = '/home/ar4414/pytorch_training/src/ar4414/pruning/logs/{}/{}/baseline/pre_ft_pp_{}.pth.tar'.format(self.params.arch, self.params.dataset, int(pp))
                torch.save(preFtChannelsPruned, logLocation)
        #}}}
        
        elif self.params.pruneFilters == True:
        #{{{
            print('=========Baseline Accuracy==========')
            testStats = self.run_inference()
            print('==========================')

            if self.params.finetune:
            #{{{
                # run finetuning
                self.run_finetune()
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

        # elif self.params.entropy == True:
        # #{{{
        #     print('=========Baseline Accuracy==========')
        #     testStats = self.run_inference()
        #     print('==========================')
        #     
        #     #{{{
        #     # if self.params.entropyLocalPruning == True:            
        #     #     prunePercentages = [x/100.0 for x in range(10,100,10)]
        #     #     # for pp in prunePercentages:
        #     #     for pp in [0.7, 0.8, 0.9]:
        #     #         for n,m in self.model.named_modules():
        #     #             if isinstance(m, nn.Conv2d):
        #     #                 if n in ['module.conv3', 'module.conv4', 'module.conv5']:
        #     #                     entropySrc.EntropyLocalPruner(n, m, self.params, pp)
        #     #         
        #     #         print('==========No finetune accuracy after pruning {}%==========='.format(pp*100.0))
        #     #         self.run_inference()
        #     #}}}
        #     
        #     if self.params.entropyGlobalPruning == True:
        #     #{{{
        #         if self.params.finetune == True:
        #             self.pruner = entropySrc.EntropyGlobalPruner(self.model, self.params, self.params.pruningPerc, [])
        #             self.run_finetune()

        #         else:
        #         #{{{
        #             prunePercs = [10, 40, 80]
        #             
        #             if self.params.plotChannels:
        #                 # channels = {l:list(range(m.out_channels)) for l,m in self.model.named_modules() if isinstance(m, nn.Conv2d)}
        #                 channels = {l:list(range(m.out_channels)) for l,m in self.model.named_modules() if isinstance(m, nn.Conv2d) if l in ['module.conv3', 'module.conv4', 'module.conv5']}
        #                 fig,ax = plt.subplots(len(prunePercs), len(channels.keys()), sharex=True, sharey=True)
        #                 fig.add_subplot(111, frameon=False)
        #             
        #             for i, pp in enumerate(prunePercs):
        #                 egp = entropySrc.EntropyGlobalPruner(self.model, self.params, pp, ['module.conv3', 'module.conv4', 'module.conv5'])
        #                 channelsPruned = egp.channelsToPrune
        #                 loss, top1, top5 = self.run_inference()
        #                 print('==========================')
        #                 tmp = [len(x) for l,x in channelsPruned.items()]
        #                 print(sum(tmp))
        #                 
        #                 if self.params.plotChannels:
        #                     for j,(l,x) in enumerate(channels.items()):
        #                         # if l not in channelsPruned.keys():
        #                         #     continue
        #                         
        #                         y = [0 for t in x]                             
        #                         for t in channelsPruned[l]:
        #                             y[t] = 1
        #                         ax[i][j].bar(x,y)
        #                         ax[i][j].get_yaxis().set_ticks([])
        #                         
        #                         if i == len(channels.keys()) - 1:
        #                             ax[i][j].set_xlabel('Layer-{}'.format(l.split('.')[1]))
        #                     
        #                     ax[i][0].set_ylabel('Pruned  = {}% \n Top1 = {:.2f}%'.format(pp*100, top1))
        #                 
        #             if self.params.plotChannels:
        #                 plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        #                 plt.xlabel('\nChannel Number')
        #                 plt.title('Channels taken when pruning based on Entropy')
        #                 plt.show()
        #         #}}}
        #     #}}}
        #     
        #     else:
        #     #{{{
        #         calculators = []
        #         layerNames = []
        #         for n,m in self.model.named_modules():
        #             if isinstance(m, nn.Conv2d):                                  
        #                 calculators.append(entropySrc.Entropy(n, m, self.params, min(self.params.numBatches, len(self.test_loader))))
        #                 calculators[-1].register_hooks()
        #                 layerNames.append(n)
        #                 
        #         self.inferer.run_n_minibatches(self.params, self.test_loader, self.model, self.params.numBatches)

        #         logger = entropySrc.EntropyLogger(self.params, calculators, layerNames) 
        #         logger.log_entropies(testStats)
        #     #}}}
        # #}}}
        
        elif self.params.fbsPruning == True:
        #{{{
            self.pruner = pruningSrc.FBSPruning(self.params, self.model)

            if self.params.fbsFinetune == True:
                self.run_finetune()
            
            elif self.params.evaluate == True:
                if self.params.unprunedRatio < 1.0:
                    self.chanProbCalculator = fbsChanSrc.ChannelProbability(self.model, self.params, self.pruner)
                    self.chanProbCalculator.register_hooks()
                    self.model = self.pruner.prune_model(self.model)
                
                self.chanProbCalculator = fbsChanSrc.ChannelProbability(self.model, self.params, self.pruner)
                self.chanProbCalculator.register_hooks()
                self.run_inference()
                
                if self.params.unprunedRatio < 1.0:
                    rootFolder = self.params.pretrained.split('/')[:-1]
                    rootFolder = '/'.join(rootFolder)
                    
                    self.pruner.log_prune_rate(rootFolder, self.params)
            
            else:
                # self.chanProbCalculator = fbsChanSrc.ChannelProbability(self.model, self.params, self.pruner)
                # self.chanProbCalculator.register_hooks()
                self.run_training()
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
       
