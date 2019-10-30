import src.ar4414.pruning.gop_calculator as gopSrc
import src.ar4414.pruning.fbs_channel_probability as fbsChanSrc
import src.ar4414.pruning.entropy as entropySrc
import src.ar4414.pruning.param_parser as ppSrc 
import src.ar4414.pruning.model_creator as mcSrc
import src.ar4414.pruning.inference as inferenceSrc
import src.ar4414.pruning.checkpointing as checkpointingSrc
import src.ar4414.pruning.training as trainingSrc
import src.ar4414.pruning.prune as pruningSrc

import src.app as appSrc
import src.input_preprocessor as preprocSrc

import os
import random
import sys
import json

import configparser as cp

import tensorboardX as tbx

import torch
import torch.cuda
import torch.nn as nn

import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd

class Application(appSrc.Application):
    def main(self):
        self.setup_dataset()
        self.setup_model()
        self.setup_tee_printing()
        
        # done here as within both get gops and 
        if self.params.pruneFilters:
            if 'mobilenet' in self.params.arch:
                self.pruner = pruningSrc.MobileNetV2Pruning(self.params, self.model)
            elif 'resnet' in self.params.arch:
                self.pruner = pruningSrc.ResNet20PruningConcat(self.params, self.model)
            elif 'alexnet' in self.params.arch:
                self.pruner = pruningSrc.AlexNetPruning(self.params, self.model)
            else:
                raise ValueError("Pruning not implemented for architecture ({})".format(self.params.arch))

        if self.params.getGops:
        #{{{
            if self.params.pruneFilters:
            #{{{
                if self.params.finetune:
                #{{{
                    try:
                        with open(os.path.join(self.params.logDir, 'pruned_channels.json'), 'r') as cpFile:
                            channelsPruned = json.load(cpFile)
                    except FileNotFoundError:
                        print("File : {} does not exist.".format(os.path.join(self.params.logDir, 'pruned_channels.json')))
                        print("Either the log directory is wrong or run finetuning without GetGops to generate file before running this command.")
                        sys.exit()
                    
                    assert float(self.params.logDir.split('/')[-3].split('_')[1]) == float(self.params.pruningPerc), 'Pruning percentage specified in config file does not correspond to log file\'s pruning percentage'
                    
                    pruneEpoch = int(list(channelsPruned.keys())[0])
                    channelsPruned = list(channelsPruned.values())[0]
                    prunePerc = channelsPruned.pop('prunePerc')
                    numBatches = len(self.train_loader)

                    # get unpruned gops
                    self.trainGopCalc = gopSrc.GopCalculator(self.model, self.params.arch) 
                    self.trainGopCalc.register_hooks()
                    self.trainer.single_forward_backward(self.params, self.model, self.criterion, self.optimiser, self.train_loader)      
                    self.trainGopCalc.remove_hooks()
                    _, tfg, _, tbg = self.trainGopCalc.get_gops()
                    unprunedGops = tfg + tbg

                    # get pruned gops
                    prunedModel = self.pruner.import_pruned_model()
                    optimiser = torch.optim.SGD(prunedModel.parameters(), lr=self.params.lr, momentum=self.params.momentum, weight_decay=self.params.weight_decay)
                    self.trainGopCalc = gopSrc.GopCalculator(prunedModel, self.params.arch) 
                    self.trainGopCalc.register_hooks()
                    self.trainer.single_forward_backward(self.params, prunedModel, self.criterion, optimiser, self.train_loader)      
                    self.trainGopCalc.remove_hooks()
                    _, tfg, _, tbg = self.trainGopCalc.get_gops()
                    prunedGops = tfg + tbg

                    print('Pruned Percentage = {}'.format(prunePerc))
                    print('Total Unpruned GOps = {}'.format(unprunedGops))
                    print('Total Pruned GOps = {}'.format(prunedGops))

                    log = os.path.join(self.params.logDir, 'log.csv')
                    log = pd.read_csv(log, delimiter = ',\t', engine='python')
                    fig, axes = plt.subplots(1,1)

                    gops = [(numBatches * unprunedGops) if epoch < pruneEpoch else (numBatches * prunedGops) for epoch in log['Epoch']]
                    log['Gops'] = np.cumsum(gops)

                    print(log)

                    log.plot(x='Gops', y='Test_Top1', ax=axes)
                    axes.set_ylabel('Top1 Test Accuracy')
                    axes.set_xlabel('GOps')
                    axes.set_title('Cost of performing finetuning in GOps \n ({:.2f}% pruning)'.format(prunePerc))
                    
                    plt.show()
                #}}}
                
                else:
                #{{{
                    channelsPruned, _, _ = self.pruner.prune_model(self.model)

                    self.trainGopCalc = gopSrc.GopCalculator(self.model, self.params.arch, channelsPruned) 
                    self.trainGopCalc.register_hooks()
                    
                    self.trainer.single_forward_backward(self.params, self.model, self.criterion, self.optimiser, self.train_loader)      
                    self.trainGopCalc.remove_hooks()

                    _, tfg, _, tbg = self.trainGopCalc.get_gops()
                    
                    loss, top1, top5 = self.run_inference()
                    print('Pruned Percentage = {}'.format(self.pruner.prune_rate(self.model, True)))
                    print('Total Forward GOps = {}'.format(tfg))
                    print('Total Backward GOps = {}'.format(tbg))
                    print('Total GOps = {}'.format(tfg + tbg))
                #}}} 
            #}}}
            
            elif 'googlenet' in self.params.arch:
            #{{{
                # register hooks
                self.gopCalculator = gopSrc.GoogleNetGopCalculator(self.model, self.params)
                self.gopCalculator.register_hooks()
                self.run_gop_calc()
                print('Unpruned Gops = ', self.gopCalculator.baseTotalGops)
                print('Pruned Gops = ', self.gopCalculator.prunedTotalGops)
            #}}} 

            else:
                raise ValueError('Gop calculation not implemented for specified architecture')
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
            
            if self.params.finetune == True:
                self.run_finetune()
            
            else:
            #{{{
                #{{{
                # if self.params.plotChannels:
                #     # channels = {l:list(range(m.out_channels)) for l,m in self.model.named_modules() if isinstance(m, nn.Conv2d)}
                #     channels = {l:list(range(m.out_channels)) for l,m in self.model.named_modules() if isinstance(m, nn.Conv2d) if l in ['module.conv3', 'module.conv4', 'module.conv5']}
                #     fig,ax = plt.subplots(len(prunePercs), len(channels.keys()), sharex=True, sharey=True)
                #     fig.add_subplot(111, frameon=False)
                #}}}
                
                channelsPruned, prunedModel, optimiser = self.pruner.prune_model(self.model)
                print('Pruned Percentage = {}'.format(self.pruner.prune_rate(self.model, True)))
                self.inferer.test_network(self.params, self.test_loader, prunedModel, self.criterion, optimiser)
                print('==========================')
                    
                #{{{
                #     if self.params.plotChannels:
                #         for j,(l,x) in enumerate(channels.items()):
                #             y = [0 for t in x]                             
                #             for t in channelsPruned[l]:
                #                 y[t] = 1
                #             ax[i][j].bar(x,y)
                #             ax[i][j].get_yaxis().set_ticks([])
                #             
                #             if i == len(channels.keys()) - 1:
                #                 ax[i][j].set_xlabel('Layer-{}'.format(l.split('.')[1]))
                #         
                #         ax[i][0].set_ylabel('Pruned  = {}% \n Top1 = {:.2f}%'.format(pp, top1))
                #     
                # if self.params.plotChannels:
                #     plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
                #     plt.xlabel('\nChannel Number')
                #     plt.title('Channels taken when pruning based on weight l2-norm')
                #     plt.show()
                #}}}
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

    def run_finetune(self):
    #{{{
        if self.params.entropy and self.params.entropyGlobalPruning:
            print('==> Performing Activation Entropy Pruning Finetune')
            self.trainer.finetune_entropy(self.params, self.pruner, self.checkpointer, self.train_loader, self.test_loader, self.valLoader, self.model, self.criterion, self.optimiser, self.inferer) 
        elif self.params.pruneFilters:
            print('==> Performing l2-weight Pruning Finetune')
            self.trainer.finetune_l2_weights(self.params, self.pruner, self.checkpointer, self.train_loader, self.test_loader, self.valLoader, self.model, self.criterion, self.optimiser, self.inferer) 
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
       
