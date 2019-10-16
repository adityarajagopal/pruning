import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

import sys 
import subprocess
import random
import math
import os
import time

class Entropy(object):
#{{{
    def __init__(self, moduleName, module, params, numBatches):
        #{{{
        self.module = module
        self.params = params
        self.gpu = 'cuda:' + self.params.gpu_id[0]
        self.temporalActivations = torch.FloatTensor([]).cuda(self.gpu)
        
        if 'mobilenet' in params.arch:
            logFolder = os.path.join(params.logDir, 'activations')
            if not os.path.isdir(logFolder):
                cmd = 'mkdir -p ' + logFolder
                subprocess.check_call(cmd, shell=True)
            
            self.logFile = os.path.join(params.logDir, 'activations', str(moduleName).replace('.','-') + '.pth.tar')
            if not os.path.isfile(self.logFile):
                torch.save(self.temporalActivations, self.logFile)
            
            self.count = 0
            self.numBatches = numBatches
        #}}}
    
    def register_hooks(self): 
        #{{{
        allChannels = range(self.module.out_channels)
        if self.params.eChannels == -1:
            self.channelsToMonitor = list(allChannels)
        else:
            self.channelsToMonitor = np.random.choice(allChannels, self.params.eChannels, replace=False) 
        
        handle = self.module.register_forward_hook(self.collect_act)
        
        return handle
        #}}}

    def collect_act(self, module, input, output):
        #{{{
        if 'mobilenet' in self.params.arch: 
            self.count += 1
        
        self.temporalActivations = torch.cat((self.temporalActivations, output[:, self.channelsToMonitor]), dim=0)

        if 'mobilenet' in self.params.arch and ((self.count % 5 == 0) or (self.count == self.numBatches)):
            prevAct = torch.load(self.logFile)
            toSave = torch.cat((prevAct, self.temporalActivations), dim=0)
            torch.save(toSave, self.logFile)
            self.temporalActivations.resize_(0)
        #}}}
    
    def calc_volumetric_entropy(self):
        #{{{
        if len(self.temporalActivations.shape) == 3:
            self.temporalActivations = torch.unsqueeze(self.temporalActivations, 1)
        
        temporalAct = self.temporalActivations.permute(1,0,2,3)
        entropies = []
        for channel in temporalAct:
            channel = channel.contiguous().view(-1)
            count = torch.histc(channel, bins=100)
            entropies.append(self.entropy(count))
            
        return entropies
        #}}}
    
    def calc_2d_entropy(self, tempAct):
        #{{{
        if 'mobilenet' in self.params.arch:
            self.temporalActivations = tempAct
        
        if len(self.temporalActivations.shape) == 3:
            self.temporalActivations = torch.unsqueeze(self.temporalActivations, 1)
        
        print(self.temporalActivations.shape)
        temporalAct = self.temporalActivations.permute(1,0,2,3)
        entropies = []
        for channel in temporalAct:
            activation = channel.view(channel.shape[0], -1).permute(1,0)
            temporalPixel = torch.unbind(activation, 0)
            counts = [torch.histc(pixel, bins=int(math.sqrt(pixel.shape[0]))) for pixel in temporalPixel]
            entropy2d = [self.entropy(count) for count in counts] 
            entropies.append(entropy2d)
        
        meanAct = torch.mean(temporalAct, dim=1)
        
        return entropies, meanAct
        #}}}

    @staticmethod
    def entropy(count):
        #{{{
        count = count.cpu().numpy()
        pA = [x / count.sum() for x in count if x != 0]
        entropy = -np.sum(pA * np.log(pA))
        return entropy
        #}}}

    @staticmethod
    def get_stat(func, values):
        #{{{
        value = func(values)
        try:
            idx = values.index(value)
        except:
            idx = -1     
        
        return(value, idx)
        #}}}
#}}}

class EntropyLogger(object):
#{{{
    def __init__(self, params, entropyCalcs, layerNames):
        #{{{
        self.calculators = entropyCalcs
        self.layerNames = layerNames
        self.params = params
        #}}}

    def log_entropies(self, testStats):
        #{{{
        logDir = self.params.logDir
            
        if self.params.sub_classes == []:
            subClasses = 'all'
        else:
            subClasses = '+'.join(self.params.sub_classes)
        
        if not self.params.printOnly:
            cmd = 'mkdir -p ' + logDir
            subprocess.check_call(cmd, shell=True)
            rawLogFile = os.path.join(logDir, 'raw_log.csv')
            summaryLogFile = os.path.join(logDir, 'summary_log.csv')
        
            rawHeaders = ['SubClass','Layer','Channel Number','Mean Entropy', 'Test Loss', 'Test Top1', 'Test Top5']
            summaryHeaders = ['SubClass','Layer','Min Channel Number','Min Mean Entropy', 'Max Channel Number', 'Max Mean Entropy', 'Test Loss', 'Test Top1', 'Test Top5'] 
            
            if os.path.exists(rawLogFile):
                rLF = open(rawLogFile, 'a')
                sLF = open(summaryLogFile, 'a')
            else:
                rLF = open(rawLogFile, 'w+')
                sLF = open(summaryLogFile, 'w+')

                rLine = ','.join(rawHeaders) + '\n'
                rLF.write(rLine)
                
                sLine = ','.join(summaryHeaders) + '\n'
                sLF.write(sLine)

        print("Super Class examined: {}".format(subClasses))
        for i, calc in enumerate(self.calculators):
            print('=============================================')
            
            if 'mobilenet' in self.params.arch:
                actLog = os.path.join(self.params.logDir, 'activations', self.layerNames[i].replace('.','-') + '.pth.tar')
                tempAct = torch.load(actLog).cpu()
                print(tempAct.shape)
                entropies, meanAct = calc.calc_2d_entropy(tempAct)
            else: 
                entropies, meanAct = calc.calc_2d_entropy([])
            
            meanEntropies = [np.mean(entropy) for entropy in entropies] 
            
            (minEntropy, minEntropyIdx) = calc.get_stat(min, meanEntropies)
            (maxEntropy, maxEntropyIdx) = calc.get_stat(max, meanEntropies)

            print('Channel: LayerName = {}, MinEntropy = {}, ChannelNum = {}'.format(self.layerNames[i], minEntropy, minEntropyIdx)) 

            print('Channel: LayerName = {}, MaxEntropy = {}, ChannelNum = {}'.format(self.layerNames[i], maxEntropy, maxEntropyIdx)) 

            if not self.params.printOnly:            
                sLine = [subClasses, self.layerNames[i], str(minEntropyIdx), str(minEntropy), str(maxEntropyIdx), str(maxEntropy),str(testStats[0]), str(testStats[1]), str(testStats[2])]
                sLine = ','.join(sLine) + '\n'
                sLF.write(sLine)
                
                for k, entropy in enumerate(meanEntropies):
                    rLine = [subClasses, self.layerNames[i], str(k), str(entropy), str(testStats[0]), str(testStats[1]), str(testStats[2])]
                    rLine = ','.join(rLine) + '\n'
                    rLF.write(rLine)

        if not self.params.printOnly:            
            rLF.close()            
            sLF.close()
        #}}}
#}}}

class GlobalHook(object):
#{{{
    def __init__(self, module, channels):
        self.channels = channels
        module.register_forward_hook(self.prune)
    
    def prune(self, module, input, output):
        output[:,self.channels] = 0
#}}}

class EntropyGlobalPruner(object):
#{{{
    def __init__(self, model, params, prunePerc, layers):
        self.gpu = 'cuda:' + params.gpu_id[0]
        
        rawLogFile = os.path.join(params.logDir, 'raw_log.csv')
        log = pd.read_csv(rawLogFile)
        if params.sub_classes == []:
            subClasses = 'all'
        else:
            subClasses = '+'.join(params.sub_classes)
        log = log[log.SubClass == subClasses]
        
        # if layers != []:
        #     layersToFilter = ''
        #     for l in layers:
        #         layersToFilter += '('+l+')|'
        #     layersToFilter = layersToFilter[:-1]
        #     data = log[log.Layer.str.match(layersToFilter)]
        #     sortedData = data.sort_values('Mean Entropy')
        # else:
        sortedData = log.sort_values('Mean Entropy')

        totalParameters = 0
        for p in model.named_parameters():
           # print(p[0], p[1].shape, torch.prod(torch.tensor(p[1].shape)))
           totalParameters += torch.prod(torch.tensor(p[1].shape)) 
        
        # marginal percentage of parameters per output kernel by layer
        # has some innacuracy as number of weights for next layer also reduces by pruning a channel in current layer
        mppByLayer = {}
        for n,m in model.named_modules():
            if isinstance(m, nn.Conv2d):
                # if layers != []:
                #     if n in layers:
                #         mppByLayer[n] = 100 * (m.in_channels * m.kernel_size[0] * m.kernel_size[1]) / float(totalParameters)
                # else: 
                mppByLayer[n] = 100 * ((m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]) / float(totalParameters)
        
        row = 0
        percPruned = 0
        self.channelsToPrune = {l:[] for l,m in model.named_modules() if isinstance(m, nn.Conv2d)}
        while(percPruned < prunePerc and row < sortedData.shape[0]):
            series = sortedData.iloc[row]
            percPruned += mppByLayer[series['Layer']] 
            self.channelsToPrune[series['Layer']].append(series['Channel Number']) 
            row += 1
        print(row, sortedData.shape[0])  
        print("Pruning by activation entropy ranking")
        print("Pruned Percentage = {}".format(percPruned))

        for n,m in model.named_modules():
            # if isinstance(m, nn.Conv2d) and n in layers:
            if isinstance(m, nn.Conv2d):
                GlobalHook(m, self.channelsToPrune[n])
#}}}
    
## Local Pruning doesn't make sense in this case, as it might be less efficient
## to prune uniformly across all 3 layers 
#{{{
# class EntropyLocalPruner(object):
#     def __init__(self, moduleName, module, params, prunePerc):
#         self.gpu = 'cuda:' + params.gpu_id[0]
# 
#         mppByLayer = {'module.conv3': 0.0006924752624433156, 'module.conv4': 0.0013849505248866311, 'module.conv5': 0.0009233003499244208}
#         
#         rawLogFile = os.path.join(params.logDir, 'raw_log.csv')
#         log = pd.read_csv(rawLogFile)
#         if params.sub_classes == []:
#             subClasses = 'all'
#         else:
#             subClasses = '+'.join(params.sub_classes)
#         log = log[log.SubClass == subClasses]
#         
#         channelData = log[log.Layer == moduleName]
#         sortedChannelData = channelData.sort_values('Mean Entropy') 
# 
#         pruned = 0
#         row = 0
#         filtersToPrune = []
#         while(pruned < prunePerc and row < sortedChannelData.shape[0]):
#            print(moduleName, row, pruned, prunePerc, sortedChannelData.shape[0])
#            filtersToPrune.append(sortedChannelData['Channel Number'].iloc[row])
#            row += 1
#            pruned += mppByLayer[moduleName]
#         
#         numChannelsPruned = int(module.out_channels * prunePerc)
#         self.filtersToPrune = sortedChannelData['Channel Number'].head(n=numChannelsPruned).tolist()
# 
#         module.register_forward_hook(self.entropy_pruning)
# 
#     def entropy_pruning(self, module, input, output): 
#         output[:,self.filtersToPrune] = 0
#}}}


