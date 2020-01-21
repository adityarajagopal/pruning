import src.ar4414.pruning.gop_calculator as gopSrc
import src.ar4414.pruning.fbs_channel_probability as fbsChanSrc
import src.ar4414.pruning.entropy as entropySrc
import src.ar4414.pruning.param_parser as ppSrc 
import src.ar4414.pruning.model_creator as mcSrc
import src.ar4414.pruning.inference as inferenceSrc
import src.ar4414.pruning.checkpointing as checkpointingSrc
import src.ar4414.pruning.training as trainingSrc
from src.ar4414.pruning.pruners.alexnet import AlexNetPruning
from src.ar4414.pruning.pruners.resnet import ResNet20PruningDependency as ResNetPruning
from src.ar4414.pruning.pruners.mobilenetv2 import MobileNetV2PruningDependency as MobileNetV2Pruning 
from src.ar4414.pruning.pruners.squeezenet import SqueezeNetPruning 
from src.ar4414.pruning.plotter import *

import src.app as appSrc
import src.input_preprocessor as preprocSrc
from src.ar4414.pruning.rbo import *

import os
import random
import sys
import json
import subprocess
import time

import configparser as cp

import tensorboardX as tbx

import torch
import torch.cuda
import torch.nn as nn

import matplotlib.pyplot as plt
import matplotlib
import math
import numpy as np
import pandas as pd

class Application(appSrc.Application):
    def main(self):
    #{{{
        self.setup_dataset()
        self.setup_model()
        self.setup_tee_printing()
        
        if self.params.pruneFilters:
            self.setup_pruners()

        if self.params.getGops:
        #{{{
            if self.params.pruneFilters:
            #{{{
                if self.params.finetune:
                #{{{
                    fig, axes = plt.subplots(1,1,figsize=(10,5))
                    listPrunePercs = [0,5,10,25,50,60,75,85,95]
                    
                    for i, logFile in enumerate(self.params.logFiles):
                    #{{{
                        logDir = os.path.join(self.params.logDir, logFile)
                        
                        try:
                            with open(os.path.join(logDir, 'pruned_channels.json'), 'r') as cpFile:
                                channelsPruned = json.load(cpFile)
                        except FileNotFoundError:
                            print("File : {} does not exist.".format(os.path.join(logDir, 'pruned_channels.json')))
                            print("Either the log directory is wrong or run finetuning without GetGops to generate file before running this command.")
                            sys.exit()
                        
                        pruneEpoch = int(list(channelsPruned.keys())[0])
                        numBatches = len(self.train_loader)
                        channelsPruned = list(channelsPruned.values())[0]
                        prunePerc = channelsPruned.pop('prunePerc')

                        self.params.pruningPerc = listPrunePercs[i]
                        newImportPath = self.pruner.importPath.split('.')
                        newFileName = newImportPath[-1].split('_')
                        newFileName[-1] = str(listPrunePercs[i])
                        newFileName = '_'.join(newFileName)
                        newImportPath[-1] = newFileName
                        newImportPath = '.'.join(newImportPath)
                        self.pruner.importPath = newImportPath

                        # get unpruned gops
                        self.trainGopCalc = gopSrc.GopCalculator(self.model, self.params.arch) 
                        self.trainGopCalc.register_hooks()
                        self.trainer.single_forward_backward(self.params, self.model, self.criterion, self.optimiser, self.train_loader)      
                        self.trainGopCalc.remove_hooks()
                        _, tfg, _, tbg = self.trainGopCalc.get_gops()
                        unprunedGops = tfg + tbg

                        totalUnprunedParams = 0
                        for p in self.model.named_parameters():
                            paramsInLayer = 1
                            for dim in p[1].size():
                                paramsInLayer *= dim
                            totalUnprunedParams += (paramsInLayer * 4) / 1e6

                        # get pruned gops
                        prunedModel = self.pruner.import_pruned_model()
                        optimiser = torch.optim.SGD(prunedModel.parameters(), lr=self.params.lr, momentum=self.params.momentum, weight_decay=self.params.weight_decay)
                        self.trainGopCalc = gopSrc.GopCalculator(prunedModel, self.params.arch) 
                        self.trainGopCalc.register_hooks()
                        self.trainer.single_forward_backward(self.params, prunedModel, self.criterion, optimiser, self.train_loader)      
                        self.trainGopCalc.remove_hooks()
                        _, tfg, _, tbg = self.trainGopCalc.get_gops()
                        prunedGops = tfg + tbg
                        
                        totalPrunedParams = 0
                        for p in prunedModel.named_parameters():
                            paramsInLayer = 1
                            for dim in p[1].size():
                                paramsInLayer *= dim
                            totalPrunedParams += (paramsInLayer * 4) / 1e6

                        print('Pruned Percentage = {:.2f}'.format(prunePerc))
                        print('Total Unpruned GOps = {:.2f}'.format(unprunedGops))
                        print('Total Unpruned Params = {:.2f}MB'.format(totalUnprunedParams))
                        print('Total Pruned GOps = {:.2f}'.format(prunedGops))
                        print('Total Pruned Params = {:.2f}MB'.format(totalPrunedParams))

                        # log = os.path.join(self.params.logDir, 'log.csv')
                        log = os.path.join(logDir, 'log.csv')
                        log = pd.read_csv(log, delimiter = ',\t', engine='python')

                        gops = [(numBatches * unprunedGops) if epoch < pruneEpoch else (numBatches * prunedGops) for epoch in log['Epoch']]
                        log['Gops'] = np.cumsum(gops)

                        print(log)

                        log.plot(x='Gops', y='Test_Top1', ax=axes, label="{:.2f}%,{:.2f}MB".format(prunePerc, totalPrunedParams))
                    #}}}
                    
                    axes.set_ylabel('Top1 Test Accuracy')
                    axes.set_xlabel('GOps')
                    axes.set_title('Cost of finetuning ({}) in GOps [{}]'.format(self.netName, self.params.subsetName))
                    axes.legend()
                    
                    plt.show()
                    
                    folder = os.path.join('/home/ar4414/remote_copy/gop_graphs/')
                    figName = os.path.join(folder, '{}_{}.png'.format(self.params.arch, self.params.subsetName))
                    cmd = 'mkdir -p {}'.format(folder)
                    subprocess.check_call(cmd, shell=True)
                    print('Saving - {}'.format(figName))
                    fig.savefig(figName, format='png') 
                #}}}
                
                else:
                #{{{
                    self.trainGopCalc = gopSrc.GopCalculator(self.model, self.params.arch) 
                    self.trainGopCalc.register_hooks()
                    self.trainer.single_forward_backward(self.params, self.model, self.criterion, self.optimiser, self.train_loader)      
                    self.trainGopCalc.remove_hooks()
                    _, utfg, _, utbg = self.trainGopCalc.get_gops()
                    
                    print('Unpruned Performance ==============')
                    loss, top1, top5 = self.run_inference()
                    print('Total Unpruned Forward GOps = {}'.format(utfg))
                    print('Total Unpruned Backward GOps = {}'.format(utbg))
                    print('Total Unpruned GOps = {}'.format(utfg + utbg))
                    
                    channelsPruned, prunedModel, optimiser = self.pruner.prune_model(self.model)
                    self.trainGopCalc = gopSrc.GopCalculator(prunedModel, self.params.arch) 
                    self.trainGopCalc.register_hooks()
                    self.trainer.single_forward_backward(self.params, prunedModel, self.criterion, optimiser, self.train_loader)      
                    self.trainGopCalc.remove_hooks()
                    _, tfg, _, tbg = self.trainGopCalc.get_gops()

                    print('Prune Performance (without finetuning) ============')
                    self.inferer.test_network(self.params, self.test_loader, prunedModel, self.criterion, optimiser)
                    pruneRate, _, _ = self.pruner.prune_rate(prunedModel)
                    print('Pruned Percentage = {:.2f}%'.format(pruneRate))
                    print('Total Pruned Forward GOps = {}'.format(tfg))
                    print('Total Pruned Backward GOps = {}'.format(tbg))
                    print('Total Pruned GOps = {}'.format(tfg + tbg))
                #}}} 
            #}}}
            
            else:
                raise ValueError('Gop calculation not implemented for specified architecture')
        #}}}

        elif self.params.plotChannels != []:
        #{{{
            if self.params.pruneFilters:
                plotter = ChannelPlotter(self.params, self.model)
                plotter.plot_channels()
        #}}}

        elif self.params.plotInferenceGops:
        #{{{
            inferenceLogs = self.params.inferenceLogs
            logCsv = pd.read_csv(inferenceLogs, header=None)            
            path = '/'.join(inferenceLogs.split('/')[:-1])
            
            plotter = RetrainPlotter()
            
            for idx, log in logCsv.iterrows():
                randInitPath = os.path.join(path, log[0])
                finetunePath = os.path.join(path, log[1])

                # get inference gops for pruned model
                self.model, self.optimiser = self.pruner.get_random_init_model(finetunePath)
                infGopCalc = gopSrc.GopCalculator(self.model, self.params.arch) 
                infGopCalc.register_hooks()
                self.run_gop_calc()
                infGopCalc.remove_hooks()
                _, tfg, _, _ = infGopCalc.get_gops()
            
                # get best test accuracy for both pruned and unpruned models
                randLog = os.path.join(randInitPath, 'log.csv')
                ftLog = os.path.join(finetunePath, 'log.csv')
                rBest = plotter.get_best_acc(randLog)
                fBest = plotter.get_best_acc(ftLog, fromEpoch=self.params.pruneAfter) 

                plotter.update_stats(tfg, rBest, fBest)
            
            subsetName = path.split('/')[-1]
            title = 'Best Acheived Accuracy vs. Inference GOps for {} [{}]'.format(self.netName, subsetName)
            logFile = '/home/ar4414/remote_copy/retrain/{}/{}.png'.format(self.netName, subsetName)
            plotter.plot(title=title, logFile=logFile) 
        #}}}

        elif self.params.changeInRanking:
        #{{{
            log = [x for x in self.params.logFiles if 'pp_{}'.format(str(int(self.params.pruningPerc))) in x][0]
            changeByLayers = {}
            prunePercs = []
            changeByPrunePerc = [] 

            basePP = {}
            
            for log in self.params.logFiles:
                ppStr = log.split('/')[0]
                pp = float(ppStr.split('_')[1])
                self.params.pruningPerc = pp 
                
                if pp == 0.:
                    continue
                
                print("=========== Prune Perc = {}% ===========".format(pp))
                self.setup_pruners()
                preFtChannelsPruned = self.pruner.structured_l1_weight(self.model)
    
                torch.save(preFtChannelsPruned, 'prunedChannels/{}/pre_ft_pp_{}.pth.tar'.format(self.params.arch, int(pp)))
                continue
                
                logFile = os.path.join(self.params.logDir, log, 'pruned_channels.json')
                with open(logFile, 'r') as jFile:
                    postFtChannelsPruned = json.load(jFile)    
                postFtChannelsPruned = list(postFtChannelsPruned.values())[0]
                postFtChannelsPruned.pop('prunePerc')
            
                layerNames = list(postFtChannelsPruned.keys())

                numChanChanged = []
                totChannelsPruned = 0
                totOverlap = 0
                for k,currChannelsPruned in postFtChannelsPruned.items(): 
                    origChannelsPruned = preFtChannelsPruned[k]

                    numChanPruned = len(list(set(currChannelsPruned) | set(origChannelsPruned)))
                    overlap = len(list(set(currChannelsPruned) & set(origChannelsPruned)))
                    totChannelsPruned += numChanPruned
                    totOverlap += overlap  

                    if numChanPruned != 0:
                        pDiff = 1.0 - (overlap / numChanPruned)
                        # print("For layer {}, percentage of channels pruned that were different = {}".format(k,pDiff))
                    else:
                        pDiff = 0
                    
                    numChanChanged.append(pDiff)      
                
                changeByLayers[ppStr] = numChanChanged
            
                pDiffGlobal = 1. - (totOverlap / totChannelsPruned)
                print("Across network, percentage of channels that were different = {}".format(pDiffGlobal))
                prunePercs.append("{}-%".format(pp))
                changeByPrunePerc.append(pDiffGlobal)  
            
            ############ comment in if need to store in order to use scripts/plot_channels.py
            ############ to plot a cross network comparison graph of global pruning difference
            toStore = {'pp':prunePercs, '%-diff':changeByPrunePerc}
            torch.save(toStore, 'prunedChannels/{}.pth.tar'.format(self.params.arch))
            
            ############ plots for each level of pruning globally what is the difference in the 
            ############ channels selected
            # title = 'Percent difference in channels pruned globally by percentage of network pruned \n ({})'.format(self.params.subsetName)
            # xlabel = "Percentage of network pruned"
            # ylabel = 'Percentage of channels pruned differently'
            # plot_channel_difference({'%-diff':changeByPrunePerc}, prunePercs, title, xlabel, ylabel) 

            ########### plots by layer what is the difference for the different levels of purning
            # title='Percentage difference in channels pruned after finetuning \n ({})'.format(self.params.subsetName)
            # xlabel = '{} - Layer Name'.format(self.params.arch.capitalize())
            # ylabel = 'Percentage of channels pruned differently'
            # plot_channel_difference(changeByLayers, layerNames, title, xlabel, ylabel) 
        #}}}

        elif self.params.entropy == True:
        #{{{
            print('=========Baseline Accuracy==========')
            testStats = self.run_inference()
            print('==========================')
            
            #{{{
            # if self.params.entropyLocalPruning == True:            
            #     prunePercentages = [x/100.0 for x in range(10,100,10)]
            #     # for pp in prunePercentages:
            #     for pp in [0.7, 0.8, 0.9]:
            #         for n,m in self.model.named_modules():
            #             if isinstance(m, nn.Conv2d):
            #                 if n in ['module.conv3', 'module.conv4', 'module.conv5']:
            #                     entropySrc.EntropyLocalPruner(n, m, self.params, pp)
            #         
            #         print('==========No finetune accuracy after pruning {}%==========='.format(pp*100.0))
            #         self.run_inference()
            #}}}
            
            if self.params.entropyGlobalPruning == True:
            #{{{
                if self.params.finetune == True:
                    self.pruner = entropySrc.EntropyGlobalPruner(self.model, self.params, self.params.pruningPerc, [])
                    self.run_finetune()

                else:
                #{{{
                    prunePercs = [10, 40, 80]
                    
                    if self.params.plotChannels:
                        # channels = {l:list(range(m.out_channels)) for l,m in self.model.named_modules() if isinstance(m, nn.Conv2d)}
                        channels = {l:list(range(m.out_channels)) for l,m in self.model.named_modules() if isinstance(m, nn.Conv2d) if l in ['module.conv3', 'module.conv4', 'module.conv5']}
                        fig,ax = plt.subplots(len(prunePercs), len(channels.keys()), sharex=True, sharey=True)
                        fig.add_subplot(111, frameon=False)
                    
                    for i, pp in enumerate(prunePercs):
                        egp = entropySrc.EntropyGlobalPruner(self.model, self.params, pp, ['module.conv3', 'module.conv4', 'module.conv5'])
                        channelsPruned = egp.channelsToPrune
                        loss, top1, top5 = self.run_inference()
                        print('==========================')
                        tmp = [len(x) for l,x in channelsPruned.items()]
                        print(sum(tmp))
                        
                        if self.params.plotChannels:
                            for j,(l,x) in enumerate(channels.items()):
                                # if l not in channelsPruned.keys():
                                #     continue
                                
                                y = [0 for t in x]                             
                                for t in channelsPruned[l]:
                                    y[t] = 1
                                ax[i][j].bar(x,y)
                                ax[i][j].get_yaxis().set_ticks([])
                                
                                if i == len(channels.keys()) - 1:
                                    ax[i][j].set_xlabel('Layer-{}'.format(l.split('.')[1]))
                            
                            ax[i][0].set_ylabel('Pruned  = {}% \n Top1 = {:.2f}%'.format(pp*100, top1))
                        
                    if self.params.plotChannels:
                        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
                        plt.xlabel('\nChannel Number')
                        plt.title('Channels taken when pruning based on Entropy')
                        plt.show()
                #}}}
            #}}}
            
            else:
            #{{{
                calculators = []
                layerNames = []
                for n,m in self.model.named_modules():
                    if isinstance(m, nn.Conv2d):                                  
                        calculators.append(entropySrc.Entropy(n, m, self.params, min(self.params.numBatches, len(self.test_loader))))
                        calculators[-1].register_hooks()
                        layerNames.append(n)
                        
                self.inferer.run_n_minibatches(self.params, self.test_loader, self.model, self.params.numBatches)

                logger = entropySrc.EntropyLogger(self.params, calculators, layerNames) 
                logger.log_entropies(testStats)
            #}}}
        #}}}
        
        elif self.params.pruneFilters == True:
        #{{{
            print('=========Baseline Accuracy==========')
            testStats = self.run_inference()
            print('==========================')

            if self.params.finetune:
            #{{{
                self.setup_lr_schedule()
                
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
            initLr = self.params.lr_schedule[self.params.lr_schedule.index(self.params.pruneAfter) - 1]
            initPrunedLrIdx = self.params.lr_schedule.index(self.params.pruneAfter) + 1
            
            if self.params.pruningPerc <= 25.0:
                initPrunedLr = initLr
                listEnd = initPrunedLrIdx + 1
            else:
                initPrunedLr = initLr / (self.params.gamma * self.params.gamma)
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

    def run_gop_calc(self):
        self.inferer.run_single_minibatch(self.params, self.test_loader, self.model)
    
    def run_inference(self):
        # perform inference only
        print('==> Performing Inference')
        return self.inferer.test_network(self.params, self.test_loader, self.model, self.criterion, self.optimiser)
       
