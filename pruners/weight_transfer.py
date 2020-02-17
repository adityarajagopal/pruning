import sys
import copy

import torch.nn as nn
import torch 

master = torch.load('/home/ar4414/pytorch_training/src/ar4414/pruning/master.pth.tar')
    
class WeightTransferUnit(object): 
#{{{
    def __init__(self, prunedModel, channelsPruned, depBlk): 
    #{{{
        self.channelsPruned = channelsPruned
        self.depBlk = depBlk
        self.pModel = prunedModel
        
        self.ipChannelsPruned = []

        baseMods = {'basic': nn_conv2d, 'relu': nn_relu, 'maxpool2d':nn_maxpool2d, 'avgpool2d':nn_avgpool2d, 'adaptiveavgpool2d':nn_adaptiveavgpool2d, 'batchnorm2d': nn_batchnorm2d, 'linear': nn_linear, 'logsoftmax': nn_logsoftmax}
        if hasattr(self, 'wtFuncs'):
            self.wtFuncs.update(baseMods)
        else:
            sef.wtFuncs = baseMods 
    #}}}

    @classmethod 
    def register_transfer_func(cls, blockName, func):
    #{{{
        if hasattr(cls, 'wtFuncs'):
            cls.wtFuncs[blockName] = func
        else: 
            setattr(cls, 'wtFuncs', {blockName:func})
    #}}}
        
    def transfer_weights(self, lType, modName, module): 
    #{{{
        self.wtFuncs[lType](self, modName, module)
        pass
    #}}}
#}}}

# torch.nn modules
def nn_conv2d(wtu, modName, module, ipChannelsPruned=None, opChannelsPruned=None, dw=False): 
#{{{
    allIpChannels = list(range(module.in_channels))
    allOpChannels = list(range(module.out_channels))
    ipPruned = wtu.ipChannelsPruned if ipChannelsPruned is None else ipChannelsPruned
    opPruned = wtu.channelsPruned[modName] if opChannelsPruned is None else opChannelsPruned
    ipChannels = list(set(allIpChannels) - set(ipPruned)) if not dw else [0]
    opChannels = list(set(allOpChannels) - set(opPruned))
    wtu.ipChannelsPruned = opPruned 
    
    pWeight = 'module.{}.weight'.format('_'.join(modName.split('.')[1:]))
    pBias = 'module.{}.bias'.format('_'.join(modName.split('.')[1:]))
    wtu.pModel[pWeight] = module._parameters['weight'][opChannels,:][:,ipChannels]
    if module._parameters['bias'] is not None:
        wtu.pModel[pBias] = module._parameters['bias'][opChannels]

    # eq = all([(wtu.pModel[pWeight] == master[pWeight]).all()])
    # print(modName, eq)
    # if not eq:
    #     breakpoint()
#}}}

def nn_batchnorm2d(wtu, modName, module): 
#{{{
    allFeatures = list(range(module.num_features))
    numFeaturesKept = list(set(allFeatures) - set(wtu.ipChannelsPruned))
    key = 'module.{}'.format('_'.join(modName.split('.')[1:]))
    wtu.pModel['{}.weight'.format(key)] = module._parameters['weight'][numFeaturesKept]
    wtu.pModel['{}.bias'.format(key)] = module._parameters['bias'][numFeaturesKept]
    wtu.pModel['{}.running_mean'.format(key)] = module._buffers['running_mean'][numFeaturesKept]
    wtu.pModel['{}.running_var'.format(key)] = module._buffers['running_var'][numFeaturesKept]
    wtu.pModel['{}.num_batches_tracked'.format(key)] = module._buffers['num_batches_tracked']

    # params = ['weight', 'bias', 'running_mean', 'running_var', 'num_batches_tracked']
    # eq = all([(wtu.pModel['{}.{}'.format(key,x)] == master['{}.{}'.format(key,x)]).all() for x in params])
    # print(modName, eq)
    # if not eq:
    #     breakpoint()
#}}}

def nn_linear(wtu, modName, module): 
#{{{
    allIpChannels = list(range(module.in_features))
    ipChannelsKept = list(set(allIpChannels) - set(wtu.ipChannelsPruned))
    
    key = 'module.{}'.format('_'.join(modName.split('.')[1:]))
    wtu.pModel['{}.weight'.format(key)] = module._parameters['weight'][:,ipChannelsKept]
    wtu.pModel['{}.bias'.format(key)] = module._parameters['bias']

    # params = ['weight', 'bias']
    # eq = all([(wtu.pModel['{}.{}'.format(key,x)] == master['{}.{}'.format(key,x)]).all() for x in params])
    # print(modName, eq)
    # if not eq:
    #     breakpoint()
#}}}

def nn_relu(wtu, modName, module): 
#{{{
    pass
#}}}

def nn_maxpool2d(wtu, modName, module): 
#{{{
    pass
#}}}

def nn_avgpool2d(wtu, modName, module): 
#{{{
    pass
#}}}

def nn_adaptiveavgpool2d(wtu, modName, module): 
#{{{
    pass
#}}}

def nn_logsoftmax(wtu, modName, module): 
#{{{
    pass
#}}}

# custom modules
def residual_backbone(wtu, modName, module, main_branch, residual_branch, aggregation_op):
#{{{
    inputToBlock = wtu.ipChannelsPruned
    idx = wtu.depBlk.instances.index(type(module))

    # main path through residual
    for n,m in module.named_modules(): 
        fullName = "{}.{}".format(modName, n)
        if not any(x in n for x in wtu.depBlk.dsLayers[idx]):
            main_branch(n, m, fullName, wtu)
    
    outputOfBlock = wtu.ipChannelsPruned

    if residual_branch is not None:
        # downsampling path if exists
        for n,m in module.named_modules(): 
            fullName = "{}.{}".format(modName, n)
            if any(x in n for x in wtu.depBlk.dsLayers[idx]):
                residual_branch(n, m, fullName, wtu, inputToBlock, outputOfBlock)
        
        if aggregation_op is not None:
            aggregation_op(wtu, opNode, opNode1)
#}}}

def split_and_aggregate_backbone(wtu, parentModName, parentModule, branchStarts, branchProcs, aggregation_op): 
#{{{
    assert len(branchStarts) == len(branchProcs), 'For each branch a processing function must be provided - branches = {}, procFuns = {}'.format(len(branchConvs), len(branchProcs))

    inputToBlock = wtu.currIpChannels
    forVarBkp = wtu.forVar

    branchOpChannels = []
    opNodes = []
            
    for idx in range(len(branchStarts)):
        branchVar = "{}_{}".format(wtu.forVar, idx)
        opNodes.append(branchVar)
        wtu.toWrite['forward'].append("\t\t{} = {}".format(branchVar, wtu.forVar))
        wtu.forVar = branchVar
        wtu.currIpChannels = inputToBlock

        inBranch = False
        for n,m in parentModule._modules.items(): 
            if n in branchStarts and not inBranch:
                inBranch = True
                branchStarts.pop(0)
            elif n in branchStarts and inBranch: 
                break
            
            if inBranch:
                fullName = "{}.{}".format(parentModName, n)
                branchProcs[idx](wtu, fullName, m)
        
        branchOpChannels.append(wtu.currIpChannels)
        wtu.forVar = forVarBkp

    if aggregation_op is not None:
        aggregation_op(wtu, opNodes, branchOpChannels)  
#}}}

def residual(wtu, modName, module):
#{{{
    def main_branch(n, m, fullName, wtu): 
    #{{{
        if isinstance(m, nn.Conv2d): 
            nn_conv2d(wtu, fullName, m)

        elif isinstance(m, nn.BatchNorm2d): 
            nn_batchnorm2d(wtu, fullName, m)
    #}}}
    
    def residual_branch(n, m, fullName, wtu, ipToBlock, opOfBlock): 
    #{{{
        if isinstance(m, nn.Conv2d): 
            nn_conv2d(wtu, fullName, m, ipToBlock, opOfBlock)
            
        elif isinstance(m, nn.BatchNorm2d):
            nn_batchnorm2d(wtu, fullName, m)
    #}}}

    residual_backbone(wtu, modName, module, main_branch, residual_branch, None)
#}}}

def mb_conv(wtu, modName, module):
#{{{
    def main_branch(n, m, fullName, wtu): 
    #{{{
        if isinstance(m, nn.Conv2d): 
            idx = wtu.depBlk.instances.index(type(module))
            wtu.convIdx = wtu.depBlk.convs[idx].index(n)
            nn_conv2d(wtu, fullName, m, None, None, dw=(wtu.convIdx==1))

        elif isinstance(m, nn.BatchNorm2d): 
            nn_batchnorm2d(wtu, fullName, m)
    #}}}
    
    def residual_branch(n, m, fullName, wtu, ipToBlock, opOfBlock): 
    #{{{
        if isinstance(m, nn.Conv2d): 
            nn_conv2d(wtu, fullName, m, ipToBlock, opOfBlock)
        
        elif isinstance(m, nn.BatchNorm2d):
            nn_batchnorm2d(wtu, fullName, m)
    #}}}
    
    idx = wtu.depBlk.instances.index(type(module))
    midConv = wtu.depBlk.convs[idx][1] 
    stride = module._modules[midConv].stride[0]
    if stride == 2: 
        residual_backbone(wtu, modName, module, main_branch, None, None)
    else:
        residual_backbone(wtu, modName, module, main_branch, residual_branch, None)
#}}}

def fire(wtu, modName, module): 
#{{{
    def basic(wtu, fullName, module): 
    #{{{
        if isinstance(module, nn.Conv2d): 
            nn_conv2d(wtu, fullName, module)

        if isinstance(module, nn.BatchNorm2d): 
            nn_batchnorm2d(wtu, fullName, module)
    #}}}

    def aggregation_op(wtu, opNodes, branchOpChannels): 
    #{{{
        wtu.currIpChannels = sum(branchOpChannels)
        nodes = "[{}]".format(','.join(opNodes))
        wtu.toWrite['forward'].append("\t\t{} = torch.cat({}, 1)".format(wtu.forVar, nodes))
        nn_relu(wtu, None, None)
    #}}}
    
    inputToBlock = wtu.currIpChannels
    idx = wtu.depBlk.instances.index(type(module))
    convs = wtu.depBlk.convs[idx]

    for n,m in module.named_modules(): 
        fullName = "{}.{}".format(modName, n)
        
        if isinstance(m, nn.Conv2d): 
            if n == convs[0]:
                nn_conv2d(wtu, fullName, m)
        
        if isinstance(m, nn.BatchNorm2d):
            nn_batchnorm2d(wtu, fullName, m)
        
        if isinstance(m, nn.ReLU): 
            nn_relu(wtu, fullName, m)     
            split_and_aggregate_backbone(wtu, modName, module, convs[1:], [basic,basic], aggregation_op)
            break
#}}}
