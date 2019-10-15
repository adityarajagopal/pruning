import sys
import torch
import math
import torch.nn as nn

class GoogleNetGopCalculator(object):
#{{{
    def __init__(self, model, params):
        #{{{
        self.modules = model._modules
        self.layers = self.modules['module']._modules
        self.params = params

        self.outChannelLog = []
        self.baseTotalGops = 0
        self.prunedTotalGops = 0
        #}}}

    def register_hooks(self):
        #{{{
        for k,v in self.layers.items():
            if 'Conv' in str(v):
                v.register_forward_hook(self.conv_gops)
            if 'Linear' in str(v):
                v.register_forward_hook(self.linear_gops)
            if 'ReLU' in str(v):
                v.register_forward_hook(self.relu_gops)
            if 'BatchNorm' in str(v): 
                v.register_forward_hook(self.bn_gops)
            if 'AvgPool' in str(v):
                v.register_forward_hook(self.avgpool_gops)
        #}}}
    
    def relu_gops(self, module, input, output):
        #{{{
        gops = input[0].numel() / 1e9
        self.baseTotalGops += gops  

        # if weight pruning
        if self.params.pruneWeights == True:
            self.prunedTotalGops += gops 

        # if filter pruning
        if self.params.pruneFilters == True:
            inputSize = input[0].size()
            inChannels = inputSize[1]
            tmp = [x[1] for x in reversed(self.outChannelLog) if x[0] == inChannels]
            inChannels = tmp[0]

            self.prunedTotalGops += (inputSize[0] * inChannels * inputSize[2] * inputSize[3]) / 1e9
        #}}}

    def avgpool_gops(self, module, input, output): 
        #{{{
        kernelSize = module.kernel_size
        batchSize = input[0].size()[0]
        inChannels = input[0].size()[1]
        
        numKernelOps = output.size()[1] * output.size()[2] * batchSize * inChannels
        opsPerKernel = kernelSize * kernelSize + 1

        gops = opsPerKernel * numKernelOps / 1e9
        self.baseTotalGops += gops

        # if weight prune
        if self.params.pruneWeights == True:
            self.prunedTotalGops += gops

        # if filter prune
        if self.params.pruneFilters == True:
            idx = [1,3,5,7]
            inChannels = sum([self.outChannelLog[i][1] for i in idx])
            numKernelOps = output.size()[1] * output.size()[2] * batchSize * inChannels
            gops = opsPerKernel * numKernelOps / 1e9
            self.prunedTotalGops += gops
        #}}}

    def bn_gops(self, module, input, output):
        #{{{
        inputSize = input[0].size()
        batchSize = inputSize[0]
        elementsPerImage = inputSize[1] * inputSize[2] * inputSize[3]
                
        # mean calculation, variance calculation, normalisation, weight and bias mac
        gops = elementsPerImage * ((batchSize + 1) + (2*batchSize + 1) + (2*batchSize + 2) + 2*batchSize) / 1e9
        self.baseTotalGops += gops

        # if weight pruning
        if self.params.pruneWeights == True:
            self.prunedTotalGops += gops 

        # if filter pruning
        if self.params.pruneFilters == True:
            inChannels = inputSize[1]                         
            tmp = [x[1] for x in reversed(self.outChannelLog) if x[0] == inChannels]
            inChannels = tmp[0]
            elementsPerImage = inChannels * inputSize[2] * inputSize[3]
            gops = elementsPerImage * ((batchSize + 1) + (2*batchSize + 1) + (2*batchSize + 2) + 2*batchSize) / 1e9
            self.prunedTotalGops += gops
        #}}}

    def linear_gops(self, module, input, output):
        #{{{
        batchSize = input[0].size()[0]
        outputFeatures = module._parameters['weight'].size()[0]
        commonDim = input[0].size()[1]
 
        biasOps = module._parameters['bias'].size()[0]

        gops = (batchSize * outputFeatures * commonDim + biasOps) / 1e9
        self.baseTotalGops += gops

        # if weight pruning 
        if self.params.pruneWeights == True:
            self.prunedTotalGops += gops
        
        # if filter pruning
        if self.params.pruneFilters == True:
            idx = [1,3,5,7]
            commonDim = sum([self.outChannelLog[i][1] for i in idx])
            gops = (batchSize * commonDim * outputFeatures + biasOps) / 1e9
            self.prunedTotalGops += gops 
        #}}}

    def get_conv_gops_for_layer(self, inChannels, kernelSize, outChannels, batchSize, outputSize):
        #{{{
        # weight operation 
        opsPerKernel = 2 * kernelSize * kernelSize * inChannels 
        numKernels = outChannels 
        totalKernelOps = batchSize * outputSize * outputSize

        biasOps = outChannels
        
        gops = opsPerKernel * numKernels * totalKernelOps + biasOps

        return (gops / 1e9)
        #}}}

    def conv_gops(self, module, input, output):
        #{{{
        mask = module._buffers['mask']

        weightMatSize = module._parameters['weight'].size()
        inputMatSize = input[0].size()
        outputMatSize = output[0].size()

        outChannels = weightMatSize[0]
        inChannels = weightMatSize[1]
        kernelSize = weightMatSize[2]
        batchSize = inputMatSize[0]
        outputSize = outputMatSize[1]

        gops = self.get_conv_gops_for_layer(inChannels, kernelSize, outChannels, batchSize, outputSize)
        self.baseTotalGops += gops
        # print(str(module), 'Base GOps = ', gops)
        
        if self.params.pruneFilters == True:
            # get number of pruned filters which is equal to new outChannels
            newOutChannels = sum([1 for filt in mask if torch.nonzero(filt).nelement() != 0])
            
            if self.outChannelLog != []:
                tmp = [x[1] for x in reversed(self.outChannelLog) if x[0] == inChannels]
                
                if len(self.outChannelLog) == 8:
                    idx = [1,3,5,7]
                    newInChannels = sum([self.outChannelLog[i][1] for i in idx])
                    self.outChannelLog = [(inChannels, newInChannels)]
                    inChannels = newInChannels
                else:
                    if tmp == []:
                        inChannels = self.outChannelLog[-1]
                    else:
                        inChannels = tmp[0]
            
            self.outChannelLog.append((outChannels,newOutChannels))
            outChannels = newOutChannels

            gops = self.get_conv_gops_for_layer(inChannels, kernelSize, outChannels, batchSize, outputSize)
            self.prunedTotalGops += gops
            # print(str(module), 'Filter Pruned GOps = ', gops)
        
        elif self.params.pruneWeights == True:
            totalNodes = mask.numel()
            prunedNodes = torch.sum((mask == 1))
            prunePerc = float(prunedNodes) / totalNodes
            kernelSize = math.sqrt(prunePerc) * kernelSize
            
            gops = self.get_conv_gops_for_layer(inChannels, kernelSize, outChannels, batchSize, outputSize)
            self.prunedTotalGops += gops
            # print(str(module), 'Weight Pruned GOps = ', gops)
        #}}}
#}}}

class GopCalculator(object):
#{{{
    def __init__(self, model, network, channelsPruned=None):
    #{{{
        self.model = model
        self.network = network

        self.backwardGops = []
        self.forwardGops = []
        self.layerNames = []

        self.hooks = []

        self.channelsPruned = [0] # no input channels pruned for 1st layer
        self.inPtr = 0
        self.outPtr = 1
        if channelsPruned is not None:
            self.prune = True
            self.channelsPruned += [len(v) for l,v in channelsPruned.items()]
        else:
            self.prune = False
            self.channelsPruned = None
    #}}}

    def register_hooks(self):
    #{{{
        for n,m in self.model.named_modules():
            if isinstance(m, nn.Conv2d):
                self.layerNames.append(n)
                self.hooks.append(m.register_forward_hook(self.forward_conv_hook))
                self.hooks.append(m.register_backward_hook(self.backward_conv_hook))
            if isinstance(m, nn.Linear):
                self.layerNames.append(n)
                self.hooks.append(m.register_forward_hook(self.forward_fc_hook))
                self.hooks.append(m.register_backward_hook(self.backward_fc_hook))
    #}}}

    def remove_hooks(self):
        [hook.remove() for hook in self.hooks]

    def get_gops(self):
    #{{{
        self.backwardGops.reverse()
        forwardGopsPerLayer = dict(zip(self.layerNames, self.forwardGops))
        backwardGopsPerLayer = dict(zip(self.layerNames, self.backwardGops))

        totalForwardGops = sum(forwardGopsPerLayer.values())
        totalBackwardGops = sum(backwardGopsPerLayer.values())
        
        return (forwardGopsPerLayer, totalForwardGops, backwardGopsPerLayer, totalBackwardGops)
    #}}}
    
    def forward_conv_hook(self, module, input, output):
    #{{{
        gops = 0.
        
        batchSize = input[0].shape[0]
        
        inChannelsPruned = self.channelsPruned[self.inPtr] if self.prune else 0
        self.inPtr += 1
        inChannels = input[0].shape[1] - inChannelsPruned
        
        outChannelsPruned = self.channelsPruned[self.outPtr] if self.prune else 0
        self.outPtr += 1
        outChannels = output.shape[1] - outChannelsPruned 

        # print(input[0].shape[1], inChannelsPruned)
        # print(output.shape[1], outChannelsPruned)

        outputSize = output.shape[2]
        kernelSize = module.kernel_size[0]

        numCalcs = batchSize * outChannels * outputSize * outputSize
        opsPerCalc = 2 * kernelSize * kernelSize * (inChannels // module.groups)
        
        gops += (opsPerCalc * numCalcs)

        self.forwardGops.append((gops / 1e9))
     #}}}

    def forward_fc_hook(self, module, input, output):
    #{{{
        batchSize = input[0].shape[0]
        outputFeatures = output.shape[1]
        commonDim = input[0].shape[1]
 
        biasOps = outputFeatures 

        gops = (batchSize * outputFeatures * commonDim + biasOps) / 1e9

        self.forwardGops.append(gops)
    #}}}

    def backward_conv_hook(self, module, grad_input, grad_output):
    #{{{
        gops = 0.

        gradWrtOp = grad_output[0].shape

        if grad_input[0] is not None:
            gradWrtIp = grad_input[0].shape
        else:
            gradWrtIp = []
        gradWrtW = grad_input[1].shape
        
        if gradWrtIp != []:
            ix = gradWrtIp[2]
        
        batchSize = gradWrtOp[0]
        ox = gradWrtOp[2]
         
        self.outPtr -= 1
        outChannelsPruned = self.channelsPruned[self.outPtr] if self.prune else 0
        outputChannels = gradWrtW[0] - outChannelsPruned
        
        self.inPtr -= 1
        if module.groups == 1:
            inChannelsPruned = self.channelsPruned[self.inPtr] if self.prune else 0
        else:
            inChannelsPruned = 0
        inputChannels = gradWrtW[1] - inChannelsPruned
        
        kernelSize = gradWrtW[2]

        # gradient wrt inputs
        if gradWrtIp != []:
            numCalcs = batchSize * outputChannels * ix * ix * inputChannels
            opsPerCalc = 2 * kernelSize * kernelSize 
            gops += (opsPerCalc * numCalcs) 

        # gradient wrt weights 
        numCalcs = batchSize * outputChannels * kernelSize * kernelSize * inputChannels
        opsPerCalc = 2 * ox * ox
        gops += (opsPerCalc * numCalcs) 

        # gradient wrt bias (sum over dE/do per opChannel per image)
        gops += batchSize * outputChannels * ox * ox 

        self.backwardGops.append((gops / 1e9))
    #}}}

    def backward_fc_hook(self, module, grad_input, grad_output):
    #{{{
        # matrix vector multiplies
        gops = 0.
        
        gradWrtOp = grad_output[0].shape
        gradWrtW = grad_input[2].shape 

        batchSize = gradWrtW[0]
        
        # grad wrt weights
        gops += 2 * gradWrtOp[0] * gradWrtOp[1] * batchSize

        # grad wrt input 
        gops += 2 * gradWrtW[0] * gradWrtW[1] * batchSize

        self.backwardGops.append(gops / 1e9)
    #}}}
#}}}

