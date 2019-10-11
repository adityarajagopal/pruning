import torch 
import sys

from src.ar4414.pruning.pruning_layers import GatedConv2d

class ChannelProbability(object):
    def __init__(self, model, params, pruner):
        self.model = model
        self.modules = model._modules
        self.layers = self.modules['module']._modules
        
        self.params = params
        self.pruner = pruner

        self.convNum = 0
        self.numConvLayers = 0

    def register_hooks(self):
        for n,m in self.model.named_modules():
            if isinstance(m, GatedConv2d):
                m.register_forward_hook(self.record_probabilities)
                self.numConvLayers += 1
        
    def record_probabilities(self, module, input, output):
        if self.params.printOnly == True:
            return
        convLayer = 'conv'+str(self.convNum)
        for idx in module.prunedChannelIdx:
            for i in idx:
                self.pruner.channelProbs[convLayer][i] += 1
        self.convNum = (self.convNum + 1) % self.numConvLayers
    
    def debugger(self, module, input, output):
        if torch.isnan(output).any(): 
            print(str(module))
            sys.exit()

        
