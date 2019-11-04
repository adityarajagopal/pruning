import sys
import csv
import os
import numpy as np
import time
from tqdm import tqdm
import json
import pickle
import subprocess
import importlib
import math

from src.ar4414.pruning.prune import BasicPruning

import torch
import torch.nn as nn

class MobileNetV2Pruning(BasicPruning):
#{{{
    def __init__(self, params, model):
        self.fileName = 'mobilenetv2_{}.py'.format(int(params.pruningPerc))
        self.netName = 'MobileNetV2'
        skipFn = lambda lName : True if 'layers' in lName and 'conv3' not in lName else False
        super().__init__(params, model, layerSkip = skipFn)
    
    def write_net(self):
    #{{{
        def fprint(text):
            print(text, file=self.modelDesc)
        
        self.modelDesc = open(self.filePath, 'w+')

        fprint('import torch')
        fprint('import torch.nn as nn')
        fprint('import torch.nn.functional as F')
    
        fprint('')
        fprint('class {}(nn.Module):'.format(self.netName))
        fprint('\tdef __init__(self, num_classes=10):')
        fprint('\t\tsuper().__init__()')
        fprint('')

        channelsPruned = {l:len(v) for l,v in self.channelsToPrune.items()}
        start = True
        currentIpChannels = 3

        linesToWrite = {}
        for n,m in self.model.named_modules():
        #{{{
            if not m._modules:
                if 'shortcut' not in n:
                    if n in channelsPruned.keys():
                        m.out_channels -= channelsPruned[n] 
                        m.in_channels = currentIpChannels if not start else m.in_channels
                        currentIpChannels = m.out_channels
                        if start:
                            start = False
                    
                    elif isinstance(m, nn.BatchNorm2d):
                        m.num_features = currentIpChannels

                    elif isinstance(m, nn.Linear):
                        m.in_features = currentIpChannels
                    
                    linesToWrite[n] = '\t\tself.{} = nn.{}'.format('_'.join(n.split('.')[1:]), str(m))
        #}}}

        #{{{
        blockInChannels = {}
        for n,m in self.model.named_modules():
            if 'layers' in n and len(n.split('.')) == 3:
                if m._modules['conv2'].stride[0] == 1:
                    blockInChannels[n] = (m._modules['conv1'].in_channels, m._modules['conv3'].out_channels)
        
        self.orderedKeys = list(linesToWrite.keys())
        for k,v in blockInChannels.items():
            if v[0] == v[1]:
                newKey = k + '.shortcut'
                self.orderedKeys.insert(self.orderedKeys.index(k + '.bn3')+1, newKey)
                m = nn.Sequential()
                linesToWrite[newKey] = '\t\tself.{} = nn.{}'.format('_'.join(newKey.split('.')[1:]), str(m))
            
            else:
                newKey = k + '.shortcut.0'
                self.orderedKeys.insert(self.orderedKeys.index(k + '.bn3')+1, newKey)
                m = nn.Conv2d(v[0], v[1], kernel_size=1, stride=1, padding=0, bias=False)
                linesToWrite[newKey] = '\t\tself.{} = nn.{}'.format('_'.join(newKey.split('.')[1:]), str(m))

                newKey = k + '.shortcut.1'
                self.orderedKeys.insert(self.orderedKeys.index(k + '.shortcut.0')+1, newKey)
                m = nn.BatchNorm2d(v[1])
                linesToWrite[newKey] = '\t\tself.{} = nn.{}'.format('_'.join(newKey.split('.')[1:]), str(m))

        [fprint(linesToWrite[k]) for k in self.orderedKeys]
                    
        fprint('')
        fprint('\tdef forward(self, x):')

        i = 0
        while i < len(self.orderedKeys): 
            if 'layers' in self.orderedKeys[i]:
                fprint('\t\tout = F.relu(self.{}(self.{}(x)))'.format('_'.join(self.orderedKeys[i+1].split('.')[1:]), '_'.join(self.orderedKeys[i].split('.')[1:])))
                i = i+2
                fprint('\t\tout = F.relu(self.{}(self.{}(out)))'.format('_'.join(self.orderedKeys[i+1].split('.')[1:]), '_'.join(self.orderedKeys[i].split('.')[1:])))
                i = i+2
                fprint('\t\tout = self.{}(self.{}(out))'.format('_'.join(self.orderedKeys[i+1].split('.')[1:]), '_'.join(self.orderedKeys[i].split('.')[1:])))
                i = i+2
                if 'shortcut.0' in self.orderedKeys[i]:
                    fprint('\t\tx = out + self.{}(self.{}(x))'.format('_'.join(self.orderedKeys[i+1].split('.')[1:]), '_'.join(self.orderedKeys[i].split('.')[1:])))
                    i = i+2
                elif 'shortcut' in self.orderedKeys[i]:
                    fprint('\t\tx = out + self.{}(x)'.format('_'.join(self.orderedKeys[i].split('.')[1:])))
                    i = i+1
                else:
                    fprint('\t\tx = out')
            elif 'linear' in self.orderedKeys[i]:
                fprint('\t\tx = F.avg_pool2d(x,4)')
                fprint('\t\tx = x.view(x.size(0), -1)')
                fprint('\t\tx = self.{}(x)'.format('_'.join(self.orderedKeys[i].split('.')[1:])))
                i += 1
            elif 'conv' in self.orderedKeys[i]:
                fprint('\t\tx = F.relu(self.{}(self.{}(x)))'.format('_'.join(self.orderedKeys[i+1].split('.')[1:]), '_'.join(self.orderedKeys[i].split('.')[1:])))
                i = i+2

        fprint('\t\treturn x')
        fprint('')
        fprint('def mobilenetv2(**kwargs):')
        fprint('\treturn MobileNetV2(**kwargs)')
        #}}}                  

        self.modelDesc.close()
    #}}}

    def transfer_weights(self, oModel, pModel):
    #{{{
        parentModel = oModel.state_dict() 
        prunedModel = pModel.state_dict() 
        
        ipChannelsToPrune = []
        ipChannelsKept = []
        opChannelsKept = []
        for k in self.orderedKeys:
            if 'conv' in k:
                layer = k
                param = k + '.weight'
                pParam = k.split('.')[0] + '.' + '_'.join(k.split('.')[1:]) + '.weight'

                opChannelsToPrune = self.channelsToPrune[layer]

                allIpChannels = list(range(parentModel[param].shape[1]))
                allOpChannels = list(range(parentModel[param].shape[0]))
                ipChannelsKept = list(set(allIpChannels) - set(ipChannelsToPrune))
                opChannelsKept = list(set(allOpChannels) - set(opChannelsToPrune))
                tmp = parentModel[param][opChannelsKept,:]
                prunedModel[pParam] = tmp[:,ipChannelsKept] 
                
                ipChannelsToPrune = opChannelsToPrune
            
            elif 'bn' in k:
                layer = k
                
                paramW = k + '.weight'
                paramB = k + '.bias'
                paramM = k + '.running_mean'
                paramV = k + '.running_var'
                paramNB = k + '.num_batches_tracked'
                
                pParamW = k.split('.')[0] + '.' + '_'.join(k.split('.')[1:]) + '.weight'
                pParamB = k.split('.')[0] + '.' + '_'.join(k.split('.')[1:]) + '.bias'
                pParamM = k.split('.')[0] + '.' + '_'.join(k.split('.')[1:]) + '.running_mean'
                pParamV = k.split('.')[0] + '.' + '_'.join(k.split('.')[1:]) + '.running_var'
                pParamNB = k.split('.')[0] + '.' + '_'.join(k.split('.')[1:]) + '.num_batches_tracked'

                prunedModel[pParamW] = parentModel[paramW][opChannelsKept]
                prunedModel[pParamB] = parentModel[paramB][opChannelsKept]
                
                prunedModel[pParamM] = parentModel[paramM][opChannelsKept]
                prunedModel[pParamV] = parentModel[paramV][opChannelsKept]
                prunedModel[pParamNB] = parentModel[paramNB]
                # prunedModel[pParamM] = torch.zeros(len(opChannelsKept))
                # prunedModel[pParamV] = torch.ones(len(opChannelsKept))
                # prunedModel[pParamNB] = torch.tensor(0, dtype=torch.long)
            
            elif 'linear' in k:
                layer = k
                paramW = k + '.weight'
                paramB = k + '.bias'
                pParamW = k.split('.')[0] + '.' + '_'.join(k.split('.')[1:]) + '.weight'
                pParamB = k.split('.')[0] + '.' + '_'.join(k.split('.')[1:]) + '.bias'
                
                prunedModel[pParamB] = parentModel[paramB]
                prunedModel[pParamW] = parentModel[paramW][:,opChannelsKept]
                
        pModel.load_state_dict(prunedModel)

        return pModel
    #}}}
#}}}

class MobileNetV2PruningDependency(BasicPruning):
#{{{
    def __init__(self, params, model):
        self.fileName = 'mobilenetv2_{}.py'.format(int(params.pruningPerc))
        self.netName = 'MobileNetV2'
        super().__init__(params, model)
        # skipFn = lambda lName : True if 'layers' in lName and 'conv3' not in lName else False
        # super().__init__(params, model, layerSkip = skipFn)
    
    def structured_l2_weight(self, model):
    #{{{
        localRanking = {} 
        globalRanking = []
        namedParams = dict(model.named_parameters())
        
        for layerName, mask in self.masks.items():
            pNp = namedParams[layerName + '.weight'].data.cpu().numpy()
            
            # calculate metric
            
            # l2-norm
            # metric = np.square(pNp).reshape(pNp.shape[0], -1).sum(axis=1)
            # metric /= (pNp.shape[1]*pNp.shape[2]*pNp.shape[3])
            # metric /= np.sqrt(np.square(metric).sum())

            #l1-norm
            metric = np.absolute(pNp).reshape(pNp.shape[0], -1).sum(axis=1)
            metric /= (pNp.shape[1]*pNp.shape[2]*pNp.shape[3])

            # calculte incremental prune percentage of pruning filter
            incPrunePerc = 100.*(pNp.shape[1]*pNp.shape[2]*pNp.shape[3]) / self.totalParams
            
            # store calculated values to be sorted by l1 norm mag and used later
            globalRanking += [(layerName, i, x, incPrunePerc) for i,x in enumerate(metric) if not np.all((mask[0][i] == 0.).data.cpu().numpy())]
            localRanking[layerName] = sorted([(i, x, incPrunePerc) for i,x in enumerate(metric) if not np.all((mask[0][i] == 0.).data.cpu().numpy())], key=lambda tup:tup[1])

        globalRanking = sorted(globalRanking, key=lambda i: i[2]) 

        self.channelsToPrune = {l:[] for l,m in model.named_modules() if isinstance(m, nn.Conv2d)}

        # build dependency lists
        dwDepLayers = [] 
        groupIdx = 0
        for n,m in model.named_modules():
            if 'layers.' in n and len(m._modules) != 0 and 'shortcut' not in n:
                dwDepLayers.append([])
                dwDepLayers[groupIdx] += [n+'.conv1', n+'.conv2']
                groupIdx += 1
        
        resDepLayers = [[list(localRanking.keys())[0]]]
        groupIdx = 0
        pruneLimit = []
        for n,m in model.named_modules():
            if 'layers.' in n and len(m._modules) != 0 and 'shortcut' not in n:
                if len(m.shortcut._modules) > 0 or m.conv2.stride[0] > 1:
                    if len(resDepLayers[groupIdx]) == 1:
                        resDepLayers[groupIdx][0] = n+'.conv3'
                    else:
                        resDepLayers.append([n+'.conv3'])
                        pruneLimit.append(len(localRanking[resDepLayers[groupIdx][0]]))
                        groupIdx += 1
                else:
                    resDepLayers[groupIdx] += [n+'.conv3']
        resDepLayers.pop(-1)        

        if self.params.pruningPerc >= 50.0:
            groupLimits = [int(math.ceil(gs * (1.0 - self.params.pruningPerc/100.0))) for gs in pruneLimit]
        else:
            groupLimits = [int(math.ceil(gs * self.params.pruningPerc/100.0)) for gs in pruneLimit]

        currentPruneRate = 0
        listIdx = 0

        def remove_filter(layerName, filterNum):
            if filterNum in self.channelsToPrune[layerName]:
                return

            for x in self.masks[layerName]:
                x[filterNum] = 0.
            self.channelsToPrune[layerName].append(filterNum)
        
        while (currentPruneRate < self.params.pruningPerc) and (listIdx < len(globalRanking)):
            layerName, filterNum, _, incPrunePerc = globalRanking[listIdx]

            depLayers = []
            limit = 2
            dependencies = resDepLayers if ('layers' in layerName and 'conv3' in layerName) else dwDepLayers
            for i, group in enumerate(dependencies):
                if layerName in group:            
                    depLayers = group
                    if 'layers' in layerName and 'conv3' in layerName:
                        groupIdx = i
                        limit = groupLimits[groupIdx]
                    break
            
            if depLayers == []: 
                if len(localRanking[layerName]) <= 2:
                    listIdx += 1
                    continue
                localRanking[layerName].pop(0)
                remove_filter(layerName, filterNum)
                currentPruneRate += incPrunePerc
            else: 
                for layerName in depLayers:
                    # if len(localRanking[layerName]) <= groupLimits[groupIdx]:
                    if len(localRanking[layerName]) <= limit:
                        listIdx += 1
                        continue
                    filt = localRanking[layerName].pop(0)
                    remove_filter(layerName, filt[0])
                    currentPruneRate += filt[2]
                
            listIdx += 1
        
        return self.channelsToPrune
    #}}}
    
    def write_net(self):
    #{{{
        def fprint(text):
            print(text, file=self.modelDesc)
        
        self.modelDesc = open(self.filePath, 'w+')

        fprint('import torch')
        fprint('import torch.nn as nn')
        fprint('import torch.nn.functional as F')
    
        fprint('')
        fprint('class {}(nn.Module):'.format(self.netName))
        fprint('\tdef __init__(self, num_classes=10):')
        fprint('\t\tsuper().__init__()')
        fprint('')

        channelsPruned = {l:len(v) for l,v in self.channelsToPrune.items()}
        start = True
        currentIpChannels = 3

        linesToWrite = {}
        for n,m in self.model.named_modules():
        #{{{
            if not m._modules:
                if 'shortcut' not in n:
                    if n in channelsPruned.keys():
                        m.out_channels -= channelsPruned[n] 
                        m.in_channels = currentIpChannels if not start else m.in_channels
                        if 'layers' in n and 'conv2' in n:
                            m.groups = m.in_channels
                        currentIpChannels = m.out_channels
                        if start:
                            start = False
                    
                    elif isinstance(m, nn.BatchNorm2d):
                        m.num_features = currentIpChannels

                    elif isinstance(m, nn.Linear):
                        m.in_features = currentIpChannels
                    
                    linesToWrite[n] = '\t\tself.{} = nn.{}'.format('_'.join(n.split('.')[1:]), str(m))
        #}}}

        #{{{
        blockInChannels = {}
        for n,m in self.model.named_modules():
            if 'layers' in n and len(n.split('.')) == 3:
                if m._modules['conv2'].stride[0] == 1:
                    blockInChannels[n] = (m._modules['conv1'].in_channels, m._modules['conv3'].out_channels)
        
        self.orderedKeys = list(linesToWrite.keys())
        for k,v in blockInChannels.items():
            if v[0] == v[1]:
                newKey = k + '.shortcut'
                self.orderedKeys.insert(self.orderedKeys.index(k + '.bn3')+1, newKey)
                m = nn.Sequential()
                linesToWrite[newKey] = '\t\tself.{} = nn.{}'.format('_'.join(newKey.split('.')[1:]), str(m))
            
            else:
                newKey = k + '.shortcut.0'
                self.orderedKeys.insert(self.orderedKeys.index(k + '.bn3')+1, newKey)
                m = nn.Conv2d(v[0], v[1], kernel_size=1, stride=1, padding=0, bias=False)
                linesToWrite[newKey] = '\t\tself.{} = nn.{}'.format('_'.join(newKey.split('.')[1:]), str(m))

                newKey = k + '.shortcut.1'
                self.orderedKeys.insert(self.orderedKeys.index(k + '.shortcut.0')+1, newKey)
                m = nn.BatchNorm2d(v[1])
                linesToWrite[newKey] = '\t\tself.{} = nn.{}'.format('_'.join(newKey.split('.')[1:]), str(m))

        [fprint(linesToWrite[k]) for k in self.orderedKeys]
                    
        fprint('')
        fprint('\tdef forward(self, x):')

        i = 0
        while i < len(self.orderedKeys): 
            if 'layers' in self.orderedKeys[i]:
                fprint('\t\tout = F.relu(self.{}(self.{}(x)))'.format('_'.join(self.orderedKeys[i+1].split('.')[1:]), '_'.join(self.orderedKeys[i].split('.')[1:])))
                i = i+2
                fprint('\t\tout = F.relu(self.{}(self.{}(out)))'.format('_'.join(self.orderedKeys[i+1].split('.')[1:]), '_'.join(self.orderedKeys[i].split('.')[1:])))
                i = i+2
                fprint('\t\tout = self.{}(self.{}(out))'.format('_'.join(self.orderedKeys[i+1].split('.')[1:]), '_'.join(self.orderedKeys[i].split('.')[1:])))
                i = i+2
                if 'shortcut.0' in self.orderedKeys[i]:
                    fprint('\t\tx = out + self.{}(self.{}(x))'.format('_'.join(self.orderedKeys[i+1].split('.')[1:]), '_'.join(self.orderedKeys[i].split('.')[1:])))
                    i = i+2
                elif 'shortcut' in self.orderedKeys[i]:
                    fprint('\t\tx = out + self.{}(x)'.format('_'.join(self.orderedKeys[i].split('.')[1:])))
                    i = i+1
                else:
                    fprint('\t\tx = out')
            elif 'linear' in self.orderedKeys[i]:
                fprint('\t\tx = F.avg_pool2d(x,4)')
                fprint('\t\tx = x.view(x.size(0), -1)')
                fprint('\t\tx = self.{}(x)'.format('_'.join(self.orderedKeys[i].split('.')[1:])))
                i += 1
            elif 'conv' in self.orderedKeys[i]:
                fprint('\t\tx = F.relu(self.{}(self.{}(x)))'.format('_'.join(self.orderedKeys[i+1].split('.')[1:]), '_'.join(self.orderedKeys[i].split('.')[1:])))
                i = i+2

        fprint('\t\treturn x')
        fprint('')
        fprint('def mobilenetv2(**kwargs):')
        fprint('\treturn MobileNetV2(**kwargs)')
        #}}}                  

        self.modelDesc.close()
    #}}}

    def transfer_weights(self, oModel, pModel):
    #{{{
        parentModel = oModel.state_dict() 
        prunedModel = pModel.state_dict() 
        
        ipChannelsToPrune = []
        ipChannelsKept = []
        opChannelsKept = []
        for k in self.orderedKeys:
            if 'conv' in k:
            #{{{
                layer = k
                param = k + '.weight'
                pParam = k.split('.')[0] + '.' + '_'.join(k.split('.')[1:]) + '.weight'

                opChannelsToPrune = self.channelsToPrune[layer]

                allIpChannels = list(range(parentModel[param].shape[1]))
                allOpChannels = list(range(parentModel[param].shape[0]))
                if 'layers' in k and 'conv2' in k:
                    ipChannelsKept = allIpChannels
                else:
                    ipChannelsKept = list(set(allIpChannels) - set(ipChannelsToPrune))
                opChannelsKept = list(set(allOpChannels) - set(opChannelsToPrune))
                tmp = parentModel[param][opChannelsKept,:]
                prunedModel[pParam] = tmp[:,ipChannelsKept] 
                
                if 'layers' in k and 'conv1' in k:
                    groupIpPruned = ipChannelsToPrune
                
                ipChannelsToPrune = opChannelsToPrune
            #}}}
            
            elif 'bn' in k:
            #{{{
                layer = k
                
                paramW = k + '.weight'
                paramB = k + '.bias'
                paramM = k + '.running_mean'
                paramV = k + '.running_var'
                paramNB = k + '.num_batches_tracked'
                
                pParamW = k.split('.')[0] + '.' + '_'.join(k.split('.')[1:]) + '.weight'
                pParamB = k.split('.')[0] + '.' + '_'.join(k.split('.')[1:]) + '.bias'
                pParamM = k.split('.')[0] + '.' + '_'.join(k.split('.')[1:]) + '.running_mean'
                pParamV = k.split('.')[0] + '.' + '_'.join(k.split('.')[1:]) + '.running_var'
                pParamNB = k.split('.')[0] + '.' + '_'.join(k.split('.')[1:]) + '.num_batches_tracked'

                prunedModel[pParamW] = parentModel[paramW][opChannelsKept]
                prunedModel[pParamB] = parentModel[paramB][opChannelsKept]
                
                prunedModel[pParamM] = parentModel[paramM][opChannelsKept]
                prunedModel[pParamV] = parentModel[paramV][opChannelsKept]
                prunedModel[pParamNB] = parentModel[paramNB]
            #}}}
            
            elif 'linear' in k:
            #{{{
                layer = k
                paramW = k + '.weight'
                paramB = k + '.bias'
                pParamW = k.split('.')[0] + '.' + '_'.join(k.split('.')[1:]) + '.weight'
                pParamB = k.split('.')[0] + '.' + '_'.join(k.split('.')[1:]) + '.bias'
                
                prunedModel[pParamB] = parentModel[paramB]
                prunedModel[pParamW] = parentModel[paramW][:,opChannelsKept]
            #}}}

            elif 'shortcut.0' in k:
            #{{{
                layer = k
                param = k + '.weight'
                pParam = k.split('.')[0] + '.' + '_'.join(k.split('.')[1:]) + '.weight'
                
                allIpChannels = list(range(parentModel[param].shape[1]))
                allOpChannels = list(range(parentModel[param].shape[0]))
                ipChannelsKept = list(set(allIpChannels) - set(groupIpPruned))
                opChannelsKept = list(set(allOpChannels) - set(opChannelsToPrune))
                tmp = parentModel[param][opChannelsKept,:]
                prunedModel[pParam] = tmp[:,ipChannelsKept] 
                
            #}}}

            elif 'shortcut.1' in k:
            #{{{
                layer = k
                
                paramW = k + '.weight'
                paramB = k + '.bias'
                paramM = k + '.running_mean'
                paramV = k + '.running_var'
                paramNB = k + '.num_batches_tracked'
                
                pParamW = k.split('.')[0] + '.' + '_'.join(k.split('.')[1:]) + '.weight'
                pParamB = k.split('.')[0] + '.' + '_'.join(k.split('.')[1:]) + '.bias'
                pParamM = k.split('.')[0] + '.' + '_'.join(k.split('.')[1:]) + '.running_mean'
                pParamV = k.split('.')[0] + '.' + '_'.join(k.split('.')[1:]) + '.running_var'
                pParamNB = k.split('.')[0] + '.' + '_'.join(k.split('.')[1:]) + '.num_batches_tracked'

                prunedModel[pParamW] = parentModel[paramW][opChannelsKept]
                prunedModel[pParamB] = parentModel[paramB][opChannelsKept]
                
                prunedModel[pParamM] = parentModel[paramM][opChannelsKept]
                prunedModel[pParamV] = parentModel[paramV][opChannelsKept]
                prunedModel[pParamNB] = parentModel[paramNB]
            #}}}
        
        pModel.load_state_dict(prunedModel)

        return pModel
    #}}}
#}}}














