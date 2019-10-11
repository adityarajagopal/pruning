import sys
import torch
import math

class GoogleNetGopCalculator(object):
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
