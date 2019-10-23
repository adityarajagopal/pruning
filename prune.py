import sys
import csv
import os
import numpy as np
import time
import torch
from tqdm import tqdm
import json
import pickle

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

class BasicPruning(object):
#{{{
    def __init__(self, params, model, inferer, valLoader):
        #{{{
        self.params = params
        self.metricValues = []
        self.totalParams = 0
        self.paramsPerLayer = []
        self.channelsToPrune = {}
        self.model = model

        self.layers = model._modules['module']._modules
        self.inferer = inferer
        self.valLoader = valLoader

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
                if layerName not in self.masks.keys():
                    self.masks[layerName] = [torch.tensor((), dtype=torch.float32).new_ones(p[1].size()).cuda(device)]
                else:
                    self.masks[layerName].append(torch.tensor((), dtype=torch.float32).new_ones(p[1].size()).cuda(device))
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
    
    def prune_model(self, model):
        #{{{
        if self.params.pruneFilters == True: 
            # pruning based on l2 norm of weights
            if self.params.pruningMetric == 'weights':
                tqdm.write("Pruning filters - Weights")
                return self.structured_l2_weight(model)
            
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

    def prune(self, module, input, output):
    #{{{
        layerName = self.pruneLayerNames[self.pruneCount]
        channelsToPrune = self.channelsToPrune[layerName]

        device = 'cuda:' + str(self.params.gpuList[0])
        mask = torch.tensor((), dtype=torch.float32).new_ones(output.shape).cuda(device)  
        mask[:,channelsToPrune] = 0.
        output.data = output.mul(mask)

        self.pruneCount += 1
        if self.pruneCount == len(self.pruneLayerNames):
            self.pruneCount = 0
    #}}} 
    
    def structured_l2_weight(self, model):
    #{{{
        self.metricValues = []
        
        namedParams = dict(model.named_parameters())
        for layerName, mask in self.masks.items():
            pNp = namedParams[layerName + '.weight'].data.cpu().numpy()
            
            # calculate metric
            metric = np.square(pNp).reshape(pNp.shape[0], -1).sum(axis=1)
            metric /= (pNp.shape[1]*pNp.shape[2]*pNp.shape[3])
            metric /= np.sqrt(np.square(metric).sum())

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
        
        # perform pruning 
        # layersToPrune = self.masks.keys()
        # for p in model.named_parameters():
        #     if 'conv' in p[0]:
        #         layerName = '.'.join(p[0].split('.')[:-1])
        #         if layerName in layersToPrune:
        #             # print(p[0], layerName, np.count_nonzero((self.masks[layerName][0 if 'weight' in p[0] else 1] == 0).data.cpu().numpy()))
        #             p[1].data = p[1].mul(self.masks[layerName][0 if 'weight' in p[0] else 1])
        #             p[1].requires_grad = True
        
        # perform pruning (with hooks)
        # self.pruneCount = 0
        # self.pruneLayerNames = []
        # layersToPrune = self.masks.keys()
        # for n,m in model.named_modules():
        #     if n in layersToPrune:
        #         m.register_forward_hook(self.prune)
        #         self.pruneLayerNames.append(n)
      
        return self.channelsToPrune
    #}}}

#{{{
    def structured_activations(self, model):
        #{{{
        # potentially have to change finetuning structure to prune single feature map each iteration
        # and alternate between finetuning and pruning till desired sparsity is reached
        # will take a lot more iterations

        handles = []
        for k,v in self.layers.items():
            if 'Conv' in str(v):
                handles.append(v.register_forward_hook(self.mean_activations))
        
        incPrunePerc = []
        for p in model.named_parameters(): 
            if 'conv' in p[0] and 'weight' in p[0]:
                pNp = p[1].data.cpu().numpy()
                tmp = 100.*(pNp.shape[1]*pNp.shape[2]*pNp.shape[3]) / self.totalParams
                incPrunePerc.append(tmp)

        numConvLayers = len(incPrunePerc)
        numBatches = 0
        self.meanActivations = {i:[] for i in range(numConvLayers)}
                
        for batchIdx, (inputs, targets) in enumerate(self.valLoader):
            self.layerNum = 0
            numBatches = len(targets)
            self.inferer.run_single_forward(self.params, inputs, targets, model)
        
        layerNum = 0
        for p in model.named_parameters(): 
            if 'conv' in p[0] and 'weight' in p[0]:
                act = self.meanActivations[layerNum]
                act /= numBatches
                magAct = act.shape[1] * act.shape[2] 

                metric = act.reshape(act.shape[0], -1).sum(axis=1)
                metric /= magAct
                metric /= np.sqrt(np.square(metric).sum())

                self.meanActivations[layerNum] = metric
                
                layerNum += 1
        
        meanActivationsByFilter = []
        for layerNum, x in self.meanActivations.items():
            # x /= numBatches
            meanActivationsByFilter += [(int(layerNum), int(filterNum), float(np.abs(act))) for filterNum, act in enumerate(x) if not np.all((self.masks[layerNum][filterNum] == 0.).data.cpu().numpy())]
                        
        meanActivationsByFilter = sorted(meanActivationsByFilter, key=lambda tup:tup[2])
        
        currentPruneRate = self.prune_rate(model)

        listIdx = 0
        while currentPruneRate < self.params.pruningPerc:
            filterToPrune = meanActivationsByFilter[listIdx]
            layerNum = filterToPrune[0]
            filterNum = filterToPrune[1]
            self.masks[layerNum][filterNum] = 0.
            model.module.set_masks(self.masks)
            currentPruneRate += incPrunePerc[layerNum] 
            listIdx += 1
        
        [handle.remove() for handle in handles]
        
        return model
        #}}}

    def mean_activations(self, module, input, output):
        #{{{
        # try averaging activations and then performing metric calculation at the end instead 

        outNp = output[0].data.cpu().numpy()
        # magAct = outNp.shape[1] * outNp.shape[2]

        # metric = outNp.reshape(outNp.shape[0], -1).sum(axis=1)
        # metric /= magAct
        # metric /= np.sqrt(np.square(metric).sum())

        if self.meanActivations[self.layerNum] == []:
            # self.meanActivations[self.layerNum] = metric
            self.meanActivations[self.layerNum] = outNp
        else:
            # self.meanActivations[self.layerNum] += metric 
            self.meanActivations[self.layerNum] += outNp

        self.layerNum += 1
        #}}}
#}}}
#}}}

class AlexNetPruning(BasicPruning):
#{{{
    def __init__(self, params, model, inferer, valLoader):
        super().__init__(params, model, inferer, valLoader)
        
    def write_net(self):
    #{{{
        def fprint(text):
            print(text, file=self.modelDesc)
        
        self.modelDesc = open('models/cifar/pruned/alexnet.py', 'w+')

        fprint('import torch')
        fprint('import torch.nn as nn')
        fprint('import torch.nn.functional as F')
    
        fprint('')
        fprint('class AlexNet(nn.Module):')
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
    def __init__(self, params, model, inferer, valLoader):
    #{{{
        self.params = params
        self.model = model
        
        self.metricValues = []
        self.totalParams = 0
        self.paramsPerLayer = []

        self.layers = model._modules['module']._modules
        self.inferer = inferer
        self.valLoader = valLoader

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
                
                # with dw convs, the initial 1x1 conv and the 3x3 dw need to have the same number of channels
                # so instead of pruning both together, only the final 1x1 which completes the dw conv is pruned
                if 'layers' in layerName and 'conv3' not in layerName:
                        continue
                
                if layerName not in self.masks.keys():
                    self.masks[layerName] = [torch.tensor((), dtype=torch.float32).new_ones(p[1].size()).cuda(device)]
                else:
                    self.masks[layerName].append(torch.tensor((), dtype=torch.float32).new_ones(p[1].size()).cuda(device))
        
        # identify blocks with residuals
        # self.resNames = []
        # self.layerCount = 0
        # for n,m in model.named_modules():
        #     if 'layers' in n and len(n.split('.')) == 3:
        #         if m._modules['conv2'].stride[0] == 1:
        #             # m.register_forward_hook(self.residual_mod)
        #             self.resNames.append(n)
    #}}} 
    
    # def residual_mod(self, module, input, output):
    # #{{{
    #     if self.channelsToPrune == {}:
    #         return 

    #     layerName = self.resNames[self.layerCount] + '.conv3'
    #     channelsToPrune = self.channelsToPrune[layerName]

    #     device = 'cuda:' + str(self.params.gpuList[0])
    #     mask = torch.tensor((), dtype=torch.float32).new_ones(output.shape).cuda(device)  
    #     mask[:,channelsToPrune] = 0.
    #     output.data = output.mul(mask)

    #     self.layerCount += 1
    #     if self.layerCount == len(self.resNames):
    #         self.layerCount = 0
    # #}}}
    
    def write_net(self):
    #{{{
        def fprint(text):
            print(text, file=self.modelDesc)
        
        self.modelDesc = open('models/cifar/pruned/mobilenetv2.py', 'w+')

        fprint('import torch')
        fprint('import torch.nn as nn')
        fprint('import torch.nn.functional as F')
    
        fprint('')
        fprint('class MobileNetV2(nn.Module):')
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
    def __init__(self, params, model, inferer, valLoader):  
        super().__init__(params, model, inferer, valLoader)

    def write_net(self):
    #{{{
        def fprint(text):
            print(text, file=self.modelDesc)
        
        # self.modelDesc = open('models/cifar/pruned/resnet_{}.py'.format(self.params.pruningPerc), 'w+')
        self.modelDesc = open('models/cifar/pruned/resnet.py', 'w+')

        fprint('import torch')
        fprint('import torch.nn as nn')
        fprint('import torch.nn.functional as F')
    
        fprint('')
        # fprint('class ResNet_{}(nn.Module):'.format(self.params.pruningPerc))
        fprint('class ResNet(nn.Module):')
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
            
            elif 'fc' in k:
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
    
    # def residual_mod(self, module, input, output):
    # #{{{
    #     if self.channelsToPrune == {}:
    #         return 

    #     layerName = self.resNames[self.layerCount] + '.conv2'
    #     channelsToPrune = self.channelsToPrune[layerName]

    #     device = 'cuda:' + str(self.params.gpuList[0])
    #     mask = torch.tensor((), dtype=torch.float32, requires_grad=False).new_ones(output.shape).cuda(device)  
    #     mask[:,channelsToPrune] = 0.
    #     output.data = output.mul(mask)

    #     self.layerCount += 1
    #     if self.layerCount == len(self.resNames):
    #         self.layerCount = 0
    # #}}}
#}}}















