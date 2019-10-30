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

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

class FBSPruning(object):
    #{{{
    def __init__(self, params, model):
        #{{{
        self.params = params
        self.layers = model._modules['module']._modules
        self.channelProbs = {}
        
        numConvLayers = 0
        for k,v in self.layers.items():
            if 'conv' in k:
                self.channelProbs['conv' + str(numConvLayers)] = {} 
            numConvLayers += 1
        
        numConvLayers = 0
        for x in model.named_parameters():
            if 'conv.weight' in x[0]:
                name = 'conv' + str(numConvLayers)
                outChannels = x[1].shape[0]
                self.channelProbs[name] = [0.0 for x in range(outChannels)] 
                numConvLayers += 1
        #}}} 
    
    def prune_model(self, model):
        #{{{
        for k,v in self.layers.items():
            if 'conv' in k:
                v.unprunedRatio = self.params.unprunedRatio
        return model
        #}}} 
    
    def prune_rate(self, model, verbose=False):
        return self.params.unprunedRatio
    
    def log_prune_rate(self, rootFolder, params): 
        #{{{
        if params.printOnly == True:
            return 

        prunePerc = '{:2.1f}'.format(1.0 - params.unprunedRatio).replace('.','_')
        fileName = 'prunePerc_' + prunePerc + '_channels_by_layer.json'
        jsonName = os.path.join(rootFolder, fileName)
        with open(jsonName, 'w') as jsonFile:
            json.dump(self.channelProbs, jsonFile)
        #}}}
    #}}}

class BasicPruning(ABC):
#{{{
    def __init__(self, params, model, layerSkip=(lambda lName : False)):
        #{{{
        self.params = params
        self.model = model
        
        self.metricValues = []
        self.totalParams = 0
        self.paramsPerLayer = []
        self.channelsToPrune = {}

        self.masks = {}
        for p in model.named_parameters():
            paramsInLayer = 1
            for dim in p[1].size():
                paramsInLayer *= dim
            self.paramsPerLayer.append(paramsInLayer)
            self.totalParams += paramsInLayer

            device = 'cuda:' + str(self.params.gpuList[0])
            if 'conv' in p[0]:
                layerName = '.'.join(p[0].split('.')[:-1])
                
                if layerSkip(layerName):
                    continue

                if layerName not in self.masks.keys():
                    self.masks[layerName] = [torch.tensor((), dtype=torch.float32).new_ones(p[1].size()).cuda(device)]
                else:
                    self.masks[layerName].append(torch.tensor((), dtype=torch.float32).new_ones(p[1].size()).cuda(device))
        
        # create model directory and file
        dirName = 'models/pruned/{}/{}'.format(params.dataset, params.subsetName)
        self.filePath = os.path.join(dirName, self.fileName)
        
        ## create dir if it doesn't exist
        cmd = 'mkdir -p {}'.format(dirName)
        subprocess.check_call(cmd, shell=True)
        
        self.importPath = 'src.ar4414.pruning.{}.{}'.format('.'.join(dirName.split('/')), self.fileName.split('.')[0])
        #}}} 
    
    def log_pruned_channels(self, rootFolder, params, totalPrunedPerc, channelsPruned): 
        #{{{
        if params.printOnly == True:
            return 

        jsonName = os.path.join(rootFolder, 'pruned_channels.json')
        channelsPruned['prunePerc'] = totalPrunedPerc
        summary = {}
        summary[str(params.curr_epoch)] = channelsPruned
        
        with open(jsonName, 'w') as sumFile:
            json.dump(summary, sumFile)
        
        return summary
        #}}} 

    def import_pruned_model(self):
    #{{{
        pModel = importlib.import_module(self.importPath).__dict__[self.netName]
        prunedModel = pModel(num_classes=100)
        prunedModel = torch.nn.DataParallel(prunedModel, self.gpu_list).cuda()

        return prunedModel
    #}}}
    
    def prune_model(self, model):
        #{{{
        if self.params.pruneFilters == True: 
            # pruning based on l2 norm of weights
            if self.params.pruningMetric == 'weights':
                tqdm.write("Pruning filters - Weights")
                channelsPruned = self.structured_l2_weight(model)

                # perform pruning by writing out pruned network
                self.gpu_list = [int(x) for x in self.params.gpu_id.split(',')]
                self.gpu = 'cuda:{}'.format(self.gpu_list[0])
                self.write_net()
                prunedModel = self.import_pruned_model()
                prunedModel = self.transfer_weights(model, prunedModel)
                optimiser = torch.optim.SGD(prunedModel.parameters(), lr=self.params.lr, momentum=self.params.momentum, weight_decay=self.params.weight_decay)

                return channelsPruned, prunedModel, optimiser
            
            # pruning based on activations 
            if self.params.pruningMetric == 'activations':
                tqdm.write("Pruning filters - Mean Activation")
                return self.structured_activations(model)
        #}}}
        
    def non_zero_argmin(self, array): 
        minIdx = np.argmin(array[np.nonzero(array)]) 
        return (minIdx, array[minIdx])     
    
    def prune_rate(self, model, verbose=False):
    #{{{
        totalPrunedParams = 0
        totalPrunedParams1 = 0
        prunedParamsPerLayer = {}

        if self.masks == {}:
            return 0.
        for layer, mask in self.masks.items():
            for x in mask:
                if layer not in prunedParamsPerLayer.keys():
                    prunedParamsPerLayer[layer] = np.count_nonzero((x == 0).data.cpu().numpy())
                else:
                    prunedParamsPerLayer[layer] += np.count_nonzero((x == 0).data.cpu().numpy())
            totalPrunedParams += prunedParamsPerLayer[layer]
        
        return 100.*(totalPrunedParams/self.totalParams) 
    #}}}        

    def structured_l2_weight(self, model):
    #{{{
        self.metricValues = []
        
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
            
            # store calculated values to be sorted by l2 norm mag and used later
            values = [(layerName, i, x, incPrunePerc) for i,x in enumerate(metric) if not np.all((mask[0][i] == 0.).data.cpu().numpy())]
            self.metricValues += values
        
        self.metricValues = sorted(self.metricValues, key=lambda tup: tup[2])

        self.channelsToPrune = {l:[] for l,m in model.named_modules() if isinstance(m, nn.Conv2d)}

        currentPruneRate = 0
        listIdx = 0
        while (currentPruneRate < self.params.pruningPerc) and (listIdx < len(self.metricValues)):
            filterToPrune = self.metricValues[listIdx]
            layerName = filterToPrune[0]
            filterNum = filterToPrune[1]
            
            for x in self.masks[layerName]:
                x[filterNum] = 0.
            self.channelsToPrune[layerName].append(filterNum)

            currentPruneRate += filterToPrune[3]
            listIdx += 1
        
        return self.channelsToPrune
    #}}}

    @abstractmethod
    def write_net(self, subsetName=None):
        pass
    
    @abstractmethod
    def transfer_weights(self, oModel, pModel): 
        pass
#}}}

class AlexNetPruning(BasicPruning):
#{{{
    def __init__(self, params, model):
        self.fileName = 'alexnet_{}.py'.format(int(params.pruningPerc))
        self.netName = 'AlexNet'
        super().__init__(params, model)

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

                elif isinstance(m, nn.ReLU):
                    continue

                linesToWrite[n] = '\t\tself.{} = nn.{}'.format('_'.join(n.split('.')[1:]), str(m))
        #}}}
        
        self.orderedKeys = list(linesToWrite.keys())
        [fprint(linesToWrite[k]) for k in self.orderedKeys]
                    
        fprint('')
        fprint('\tdef forward(self, x):')

        i = 0
        maxpoolConvs = ['module.conv1', 'module.conv2', 'module.conv5']
        maxPoolLayerName = [x for x in self.orderedKeys if 'maxpool' in x][0]
        while i < len(self.orderedKeys): 
            if 'conv' in self.orderedKeys[i]:
                fprint('\t\tx = F.relu(self.{}(x))'.format('_'.join(self.orderedKeys[i].split('.')[1:])))
                if self.orderedKeys[i] in maxpoolConvs:
                    fprint('\t\tx = self.{}(x)'.format('_'.join(maxPoolLayerName.split('.')[1:])))
                i = i+1
            
            elif 'linear' in self.orderedKeys[i] or 'classifier' in self.orderedKeys[i]:
                fprint('\t\tx = x.view(x.size(0), -1)')
                fprint('\t\tx = self.{}(x)'.format('_'.join(self.orderedKeys[i].split('.')[1:])))
                i += 1

            elif 'maxpool' in self.orderedKeys[i]:
                i += 1

        fprint('\t\treturn x')
        fprint('')
        fprint('def alexnet(**kwargs):')
        fprint('\treturn AlexNet(**kwargs)')
        
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
                paramB = k + '.bias'
                pParam = k.split('.')[0] + '.' + '_'.join(k.split('.')[1:]) + '.weight'
                pParamB = k.split('.')[0] + '.' + '_'.join(k.split('.')[1:]) + '.bias'

                opChannelsToPrune = self.channelsToPrune[layer]

                allIpChannels = list(range(parentModel[param].shape[1]))
                allOpChannels = list(range(parentModel[param].shape[0]))
                ipChannelsKept = list(set(allIpChannels) - set(ipChannelsToPrune))
                opChannelsKept = list(set(allOpChannels) - set(opChannelsToPrune))
                
                tmpW = parentModel[param][opChannelsKept,:]
                prunedModel[pParam] = tmpW[:,ipChannelsKept] 
                prunedModel[pParamB] = parentModel[paramB][opChannelsKept]
                
                ipChannelsToPrune = opChannelsToPrune
            
            elif 'linear' in k or 'classifier' in k:
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

class ResNet20Pruning(BasicPruning):
#{{{
    def __init__(self, params, model):  
        self.fileName = 'resnet{}_{}.py'.format(int(params.depth), int(params.pruningPerc))
        self.netName = 'ResNet{}'.format(int(params.depth))
        super().__init__(params, model)

    def write_net(self):
    #{{{
        def fprint(text):
            print(text, file=self.modelDesc)
        
        self.modelDesc = open(self.filePath, 'w+')

        fprint('import torch')
        fprint('import torch.nn as nn')
        fprint('import torch.nn.functional as F')
    
        fprint('')
        # fprint('class ResNet_{}(nn.Module):'.format(self.params.pruningPerc))
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
                if 'downsample' not in n:
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

                    elif isinstance(m, nn.ReLU):
                        continue
                    
                    linesToWrite[n] = '\t\tself.{} = nn.{}'.format('_'.join(n.split('.')[1:]), str(m))
        #}}}

        #{{{
        blockInChannels = {}
        for n,m in self.model.named_modules():
            if 'layer' in n and len(n.split('.')) == 3:
                blockInChannels[n] = (m._modules['conv1'].in_channels, m._modules['conv2'].out_channels, m._modules['conv1'].stride)
        
        self.orderedKeys = list(linesToWrite.keys())
        for k,v in blockInChannels.items():
            if v[0] == v[1]:
                newKey = k + '.downsample'
                self.orderedKeys.insert(self.orderedKeys.index(k + '.bn2')+1, newKey)
                m = nn.Sequential()
                linesToWrite[newKey] = '\t\tself.{} = nn.{}'.format('_'.join(newKey.split('.')[1:]), str(m))
            
            else:
                newKey = k + '.downsample.0'
                self.orderedKeys.insert(self.orderedKeys.index(k + '.bn2')+1, newKey)
                m = nn.Conv2d(v[0], v[1], kernel_size=1, stride=v[2], padding=0, bias=False)
                linesToWrite[newKey] = '\t\tself.{} = nn.{}'.format('_'.join(newKey.split('.')[1:]), str(m))

                newKey = k + '.downsample.1'
                self.orderedKeys.insert(self.orderedKeys.index(k + '.downsample.0')+1, newKey)
                m = nn.BatchNorm2d(v[1])
                linesToWrite[newKey] = '\t\tself.{} = nn.{}'.format('_'.join(newKey.split('.')[1:]), str(m))

        [fprint(linesToWrite[k]) for k in self.orderedKeys]
        
        fprint('')
        fprint('\tdef forward(self, x):')

        i = 0
        while i < len(self.orderedKeys): 
            if 'layer' in self.orderedKeys[i]:
                fprint('\t\tout = F.relu(self.{}(self.{}(x)))'.format('_'.join(self.orderedKeys[i+1].split('.')[1:]), '_'.join(self.orderedKeys[i].split('.')[1:])))
                i = i+2
                fprint('\t\tout = self.{}(self.{}(out))'.format('_'.join(self.orderedKeys[i+1].split('.')[1:]), '_'.join(self.orderedKeys[i].split('.')[1:])))
                i = i+2
                if 'downsample.0' in self.orderedKeys[i]:
                    fprint('\t\tx = F.relu(out + self.{}(self.{}(x)))'.format('_'.join(self.orderedKeys[i+1].split('.')[1:]), '_'.join(self.orderedKeys[i].split('.')[1:])))
                    i = i+2
                elif 'downsample' in self.orderedKeys[i]:
                    fprint('\t\tx = F.relu(out + self.{}(x))'.format('_'.join(self.orderedKeys[i].split('.')[1:])))
                    i = i+1
                else:
                    fprint('\t\tx = F.relu(out)')
            
            elif 'fc' in self.orderedKeys[i]:
                fprint('\t\tx = x.view(x.size(0), -1)')
                fprint('\t\tx = self.{}(x)'.format('_'.join(self.orderedKeys[i].split('.')[1:])))
                i += 1
            
            elif 'conv' in self.orderedKeys[i]:
                fprint('\t\tx = F.relu(self.{}(self.{}(x)))'.format('_'.join(self.orderedKeys[i+1].split('.')[1:]), '_'.join(self.orderedKeys[i].split('.')[1:])))
                i = i+2
            
            elif 'avgpool' in self.orderedKeys[i]:
                fprint('\t\tx = self.{}(x)'.format('_'.join(self.orderedKeys[i].split('.')[1:])))
                i += 1

        fprint('\t\treturn x')
        fprint('')
        # fprint('def resnet_{}(**kwargs):'.format(self.params.pruningPerc))
        # fprint('\treturn ResNet_{}(**kwargs)'.format(self.params.pruningPerc))
        fprint('def resnet(**kwargs):')
        fprint('\treturn ResNet(**kwargs)')
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
                ipChannelsKept = list(set(allIpChannels) - set(ipChannelsToPrune))
                opChannelsKept = list(set(allOpChannels) - set(opChannelsToPrune))
                tmp = parentModel[param][opChannelsKept,:]
                prunedModel[pParam] = tmp[:,ipChannelsKept] 
                
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
            
            elif 'fc' in k:
            #{{{
                layer = k
                paramW = k + '.weight'
                paramB = k + '.bias'
                pParamW = k.split('.')[0] + '.' + '_'.join(k.split('.')[1:]) + '.weight'
                pParamB = k.split('.')[0] + '.' + '_'.join(k.split('.')[1:]) + '.bias'
                
                prunedModel[pParamB] = parentModel[paramB]
                prunedModel[pParamW] = parentModel[paramW][:,opChannelsKept]
            #}}}

        pModel.load_state_dict(prunedModel)

        return pModel
    #}}}
#}}}

class ResNet20PruningConcat(BasicPruning):
#{{{
    def __init__(self, params, model):  
        self.fileName = 'resnet{}_{}.py'.format(int(params.depth), int(params.pruningPerc))
        self.netName = 'ResNet{}'.format(int(params.depth))
        super().__init__(params, model)

    def write_net(self):
    #{{{
        def fprint(text):
            print(text, file=self.modelDesc)
        
        self.modelDesc = open(self.filePath, 'w+')
        # self.modelDesc = sys.stdout 

        fprint('import torch')
        fprint('import torch.nn as nn')
        fprint('import torch.nn.functional as F')
    
        fprint('')
        # fprint('class ResNet_{}(nn.Module):'.format(self.params.pruningPerc))
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
                if 'downsample' not in n:
                    if n in channelsPruned.keys():
                        m.out_channels -= channelsPruned[n] 
                        m.in_channels = currentIpChannels if not start else m.in_channels
                        currentIpChannels = m.out_channels
                        if start:
                            start = False
                    
                    if isinstance(m, nn.BatchNorm2d):
                        m.num_features = currentIpChannels

                    elif isinstance(m, nn.Linear):
                        m.in_features = currentIpChannels

                    elif isinstance(m, nn.ReLU):
                        continue
                    
                    linesToWrite[n] = '\t\tself.{} = nn.{}'.format('_'.join(n.split('.')[1:]), str(m))
        #}}}

        #{{{
        blockInChannels = {}
        for n,m in self.model.named_modules():
            if 'layer' in n and len(n.split('.')) == 3:
                blockInChannels[n] = [m._modules['conv1'].in_channels, m._modules['conv2'].out_channels, m._modules['conv1'].stride]
        
        self.orderedKeys = list(linesToWrite.keys())
        updatedIpChannels = {k:v[0] for k,v in blockInChannels.items()}
        for i,(k,v) in enumerate(blockInChannels.items()):

            ic = updatedIpChannels[k]
            
            if v[2][0] == 2:
                newKey = k + '.downsample.0'
                self.orderedKeys.insert(self.orderedKeys.index(k + '.bn2')+1, newKey)
                m = nn.Conv2d(ic, v[1], kernel_size=1, stride=v[2], padding=0, bias=False)
                linesToWrite[newKey] = '\t\tself.{} = nn.{}'.format('_'.join(newKey.split('.')[1:]), str(m))

                newKey = k + '.downsample.1'
                self.orderedKeys.insert(self.orderedKeys.index(k + '.downsample.0')+1, newKey)
                m = nn.BatchNorm2d(v[1])
                linesToWrite[newKey] = '\t\tself.{} = nn.{}'.format('_'.join(newKey.split('.')[1:]), str(m))
            
            else:
                newKey = 'ignore:res' if ic == v[1] else 'ignore:pad_out' if ic > v[1] else 'ignore:pad_x'
                self.orderedKeys.insert(self.orderedKeys.index(k + '.bn2') + 1, newKey)
                
                if newKey == 'ignore:pad_out':
                    toEdit = self.orderedKeys[self.orderedKeys.index(k + '.bn2') + 2]
                    lineToEdit = linesToWrite[toEdit]
                    modName, module = lineToEdit.split('=',1)
                    print(modName)
                    module = eval(module)
                    module.in_channels = v[0]
                    linesToWrite[toEdit] = '\t\t{} = nn.{}'.format(modName.strip(), str(module))

                    layerName = '.'.join(toEdit.split('.')[:-1])
                    updatedIpChannels[layerName] = v[0]
        
        # if last layer has a concatenated output, then fc layer that follows this needs to also have a different input size
        if newKey == 'ignore:pad_out':
            i = self.orderedKeys.index(k + '.bn2')
            while('fc' not in self.orderedKeys[i]):
                i += 1
            toEdit = self.orderedKeys[i]
            lineToEdit = linesToWrite[toEdit]
            modName, module = lineToEdit.split('=',1)
            module = eval(module)
            module.in_features = ic
            linesToWrite[toEdit] = '\t\t{} = nn.{}'.format(modName.strip(), str(module))

        [fprint(linesToWrite[k]) for k in self.orderedKeys if 'ignore' not in k]

        fprint('')
        fprint('\tdef forward(self, x):')

        i = 0
        while i < len(self.orderedKeys): 
            if 'layer' in self.orderedKeys[i]:
                fprint('\t\tout = F.relu(self.{}(self.{}(x)))'.format('_'.join(self.orderedKeys[i+1].split('.')[1:]), '_'.join(self.orderedKeys[i].split('.')[1:])))
                i = i+2
                fprint('\t\tout = self.{}(self.{}(out))'.format('_'.join(self.orderedKeys[i+1].split('.')[1:]), '_'.join(self.orderedKeys[i].split('.')[1:])))
                i = i+2
                
                if 'downsample.0' in self.orderedKeys[i]:
                    fprint('\t\tx = F.relu(out + self.{}(self.{}(x)))'.format('_'.join(self.orderedKeys[i+1].split('.')[1:]), '_'.join(self.orderedKeys[i].split('.')[1:])))
                    i = i+2
                elif self.orderedKeys[i] == 'ignore:res':
                    i += 1
                elif self.orderedKeys[i] == 'ignore:pad_x':
                    fprint("\t\ttmp = torch.zeros(out.shape[0], out.shape[1] - x.shape[1], out.shape[2], out.shape[3], requires_grad=False).cuda('{}')".format(self.gpu))
                    fprint("\t\tx = torch.cat([x, tmp], dim=1)")
                    i += 1 
                elif self.orderedKeys[i] == 'ignore:pad_out':
                    fprint("\t\ttmp = torch.zeros(x.shape[0], x.shape[1] - out.shape[1], x.shape[2], x.shape[3], requires_grad=False).cuda('{}')".format(self.gpu))
                    fprint("\t\tout = torch.cat([out, tmp], dim=1)")
                    i += 1 
                
                fprint('\t\tx = F.relu(out + x)')
            
            elif 'fc' in self.orderedKeys[i]:
                fprint('\t\tx = x.view(x.size(0), -1)')
                fprint('\t\tx = self.{}(x)'.format('_'.join(self.orderedKeys[i].split('.')[1:])))
                i += 1
            
            elif 'conv' in self.orderedKeys[i]:
                fprint('\t\tx = F.relu(self.{}(self.{}(x)))'.format('_'.join(self.orderedKeys[i+1].split('.')[1:]), '_'.join(self.orderedKeys[i].split('.')[1:])))
                i = i+2
            
            elif 'avgpool' in self.orderedKeys[i]:
                fprint('\t\tx = self.{}(x)'.format('_'.join(self.orderedKeys[i].split('.')[1:])))
                i += 1

        fprint('\t\treturn x')
        fprint('')
        # fprint('def resnet_{}(**kwargs):'.format(self.params.pruningPerc))
        # fprint('\treturn ResNet_{}(**kwargs)'.format(self.params.pruningPerc))
        fprint('def resnet(**kwargs):')
        fprint('\treturn ResNet(**kwargs)')
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
                ipChannelsKept = list(set(allIpChannels) - set(ipChannelsToPrune))
                opChannelsKept = list(set(allOpChannels) - set(opChannelsToPrune))
                tmp = parentModel[param][opChannelsKept,:]
                prunedModel[pParam][:,:len(ipChannelsKept)] = tmp[:,ipChannelsKept] 

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
            
            elif 'fc' in k:
            #{{{
                layer = k
                
                paramW = k + '.weight'
                paramB = k + '.bias'
                pParamW = k.split('.')[0] + '.' + '_'.join(k.split('.')[1:]) + '.weight'
                pParamB = k.split('.')[0] + '.' + '_'.join(k.split('.')[1:]) + '.bias'

                prunedModel[pParamB] = parentModel[paramB]
                prunedModel[pParamW][:,:len(opChannelsKept)] = parentModel[paramW][:,opChannelsKept]
            #}}}

        pModel.load_state_dict(prunedModel)

        return pModel
    #}}}
#}}}

class ResNet20PruningDependency(BasicPruning):
#{{{
    def __init__(self, params, model):  
        self.fileName = 'resnet{}_{}.py'.format(int(params.depth), int(params.pruningPerc))
        self.netName = 'ResNet{}'.format(int(params.depth))
        super().__init__(params, model)

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

        # build dependency list 
        depLayerNames = [list(localRanking.keys())[0]]
        depLayerNames += [x for x in list(localRanking.keys()) if 'layer' in x and 'conv2' in x]
        groupSizes = list(len(localRanking[k]) for k in depLayerNames)
        if self.params.pruningPerc >= 50.0:
            minFilters = [int(math.ceil(gs * (1.0 - self.params.pruningPerc/100.0))) for gs in groupSizes]
        else:
            minFilters = [int(math.ceil(gs * self.params.pruningPerc/100.0)) for gs in groupSizes]
        
        dependencies = [[]]
        groupLimits = [minFilters[0]]
        prevGs = groupSizes[0]
        groupIdx = 0
        for i,currGs in enumerate(groupSizes):
            if currGs == prevGs:
                dependencies[groupIdx] += [depLayerNames[i]]
            else:
                dependencies.append([depLayerNames[i]])        
                groupLimits.append(minFilters[i])
                groupIdx += 1
            prevGs = currGs
        
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
            for i, group in enumerate(dependencies):
                if layerName in group:            
                    depLayers = group
                    groupIdx = i
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
                    if len(localRanking[layerName]) <= groupLimits[groupIdx]:
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
        # self.modelDesc = sys.stdout 

        fprint('import torch')
        fprint('import torch.nn as nn')
        fprint('import torch.nn.functional as F')
    
        fprint('')
        # fprint('class ResNet_{}(nn.Module):'.format(self.params.pruningPerc))
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
                if 'downsample' not in n:
                    if n in channelsPruned.keys():
                        m.out_channels -= channelsPruned[n] 
                        m.in_channels = currentIpChannels if not start else m.in_channels
                        currentIpChannels = m.out_channels
                        if start:
                            start = False
                    
                    if isinstance(m, nn.BatchNorm2d):
                        m.num_features = currentIpChannels

                    elif isinstance(m, nn.Linear):
                        m.in_features = currentIpChannels

                    elif isinstance(m, nn.ReLU):
                        continue
                    
                    linesToWrite[n] = '\t\tself.{} = nn.{}'.format('_'.join(n.split('.')[1:]), str(m))
        #}}}

        #{{{
        blockInChannels = {}
        for n,m in self.model.named_modules():
            if 'layer' in n and len(n.split('.')) == 3:
                blockInChannels[n] = [m._modules['conv1'].in_channels, m._modules['conv2'].out_channels, m._modules['conv1'].stride]
        
        self.orderedKeys = list(linesToWrite.keys())
        updatedIpChannels = {k:v[0] for k,v in blockInChannels.items()}
        for i,(k,v) in enumerate(blockInChannels.items()):

            ic = updatedIpChannels[k]
            
            if v[2][0] == 2:
                newKey = k + '.downsample.0'
                self.orderedKeys.insert(self.orderedKeys.index(k + '.bn2')+1, newKey)
                m = nn.Conv2d(ic, v[1], kernel_size=1, stride=v[2], padding=0, bias=False)
                linesToWrite[newKey] = '\t\tself.{} = nn.{}'.format('_'.join(newKey.split('.')[1:]), str(m))

                newKey = k + '.downsample.1'
                self.orderedKeys.insert(self.orderedKeys.index(k + '.downsample.0')+1, newKey)
                m = nn.BatchNorm2d(v[1])
                linesToWrite[newKey] = '\t\tself.{} = nn.{}'.format('_'.join(newKey.split('.')[1:]), str(m))
            
            else:
                newKey = 'ignore:res' if ic == v[1] else 'ignore:pad_out' if ic > v[1] else 'ignore:pad_x'
                self.orderedKeys.insert(self.orderedKeys.index(k + '.bn2') + 1, newKey)
                
                if newKey == 'ignore:pad_out':
                    toEdit = self.orderedKeys[self.orderedKeys.index(k + '.bn2') + 2]
                    lineToEdit = linesToWrite[toEdit]
                    modName, module = lineToEdit.split('=',1)
                    print(modName)
                    module = eval(module)
                    module.in_channels = v[0]
                    linesToWrite[toEdit] = '\t\t{} = nn.{}'.format(modName.strip(), str(module))

                    layerName = '.'.join(toEdit.split('.')[:-1])
                    updatedIpChannels[layerName] = v[0]
        
        # if last layer has a concatenated output, then fc layer that follows this needs to also have a different input size
        if newKey == 'ignore:pad_out':
            i = self.orderedKeys.index(k + '.bn2')
            while('fc' not in self.orderedKeys[i]):
                i += 1
            toEdit = self.orderedKeys[i]
            lineToEdit = linesToWrite[toEdit]
            modName, module = lineToEdit.split('=',1)
            module = eval(module)
            module.in_features = ic
            linesToWrite[toEdit] = '\t\t{} = nn.{}'.format(modName.strip(), str(module))

        [fprint(linesToWrite[k]) for k in self.orderedKeys if 'ignore' not in k]

        fprint('')
        fprint('\tdef forward(self, x):')

        i = 0
        while i < len(self.orderedKeys): 
            if 'layer' in self.orderedKeys[i]:
                fprint('\t\tout = F.relu(self.{}(self.{}(x)))'.format('_'.join(self.orderedKeys[i+1].split('.')[1:]), '_'.join(self.orderedKeys[i].split('.')[1:])))
                i = i+2
                fprint('\t\tout = self.{}(self.{}(out))'.format('_'.join(self.orderedKeys[i+1].split('.')[1:]), '_'.join(self.orderedKeys[i].split('.')[1:])))
                i = i+2
                
                if 'downsample.0' in self.orderedKeys[i]:
                    fprint('\t\tx = F.relu(out + self.{}(self.{}(x)))'.format('_'.join(self.orderedKeys[i+1].split('.')[1:]), '_'.join(self.orderedKeys[i].split('.')[1:])))
                    i = i+2
                elif self.orderedKeys[i] == 'ignore:res':
                    i += 1
                elif self.orderedKeys[i] == 'ignore:pad_x':
                    fprint("\t\ttmp = torch.zeros(out.shape[0], out.shape[1] - x.shape[1], out.shape[2], out.shape[3], requires_grad=False).cuda('{}')".format(self.gpu))
                    fprint("\t\tx = torch.cat([x, tmp], dim=1)")
                    i += 1 
                elif self.orderedKeys[i] == 'ignore:pad_out':
                    fprint("\t\ttmp = torch.zeros(x.shape[0], x.shape[1] - out.shape[1], x.shape[2], x.shape[3], requires_grad=False).cuda('{}')".format(self.gpu))
                    fprint("\t\tout = torch.cat([out, tmp], dim=1)")
                    i += 1 
                
                fprint('\t\tx = F.relu(out + x)')
            
            elif 'fc' in self.orderedKeys[i]:
                fprint('\t\tx = x.view(x.size(0), -1)')
                fprint('\t\tx = self.{}(x)'.format('_'.join(self.orderedKeys[i].split('.')[1:])))
                i += 1
            
            elif 'conv' in self.orderedKeys[i]:
                fprint('\t\tx = F.relu(self.{}(self.{}(x)))'.format('_'.join(self.orderedKeys[i+1].split('.')[1:]), '_'.join(self.orderedKeys[i].split('.')[1:])))
                i = i+2
            
            elif 'avgpool' in self.orderedKeys[i]:
                fprint('\t\tx = self.{}(x)'.format('_'.join(self.orderedKeys[i].split('.')[1:])))
                i += 1

        fprint('\t\treturn x')
        fprint('')
        # fprint('def resnet_{}(**kwargs):'.format(self.params.pruningPerc))
        # fprint('\treturn ResNet_{}(**kwargs)'.format(self.params.pruningPerc))
        fprint('def resnet(**kwargs):')
        fprint('\treturn ResNet(**kwargs)')
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

        groupIpPruned = []

        for k in self.orderedKeys:
            if 'conv' in k:
            #{{{
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

                if 'layer' in k and 'conv1' in k:
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
            
            elif 'fc' in k:
            #{{{
                layer = k
                
                paramW = k + '.weight'
                paramB = k + '.bias'
                pParamW = k.split('.')[0] + '.' + '_'.join(k.split('.')[1:]) + '.weight'
                pParamB = k.split('.')[0] + '.' + '_'.join(k.split('.')[1:]) + '.bias'

                prunedModel[pParamB] = parentModel[paramB]
                prunedModel[pParamW] = parentModel[paramW][:,opChannelsKept]
            #}}}

            elif 'downsample.0' in k:
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
            
            elif 'downsample.1' in k:
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















