import sys
import copy

import torch.nn as nn
    
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
def nn_conv2d(wtu, modName, module, dw=False): 
#{{{
    allIpChannels = list(range(module.in_channels))
    allOpChannels = list(range(module.out_channels))
    ipChannels = list(set(allIpChannels) - set(wtu.ipChannelsPruned))
    opChannels = list(set(allOpChannels) - set(wtu.channelsPruned[modName]))
    pMod = eval('wtu.pModel.module.{}'.format('_'.join(modName.split('.')[1:])))
    pMod._parameters['weight'] = module._parameters['weight'][opChannels,:][:,ipChannels,:,:]
    if pMod._parameters['bias'] is not None: 
        pMod._parameters['bias'] = module._parameters['bias'][opChannels]
#}}}

def nn_relu(writer, modName, module): 
#{{{
    pass
#}}}

def nn_maxpool2d(writer, modName, module): 
#{{{
    pass
#}}}

def nn_avgpool2d(writer, modName, module): 
#{{{
    pass
#}}}

def nn_adaptiveavgpool2d(writer, modName, module): 
#{{{
    pass
#}}}

def nn_batchnorm2d(writer, modName, module): 
#{{{
    module.num_features = writer.currIpChannels
    
    writer.write_module_desc(modName, module)
    writer.write_module_forward(modName)
#}}}

def nn_linear(writer, modName, module): 
#{{{
    module.in_features = writer.currIpChannels
    
    writer.write_module_desc(modName, module)
    
    writer.toWrite['forward'].append('\t\t{} = {}.view({}.size(0), -1)'.format(writer.forVar, writer.forVar, writer.forVar))
    writer.write_module_forward(modName)
#}}}

def nn_logsoftmax(writer, modName, module): 
#{{{
    layerName = '_'.join(modName.split('.')[1:])
    moduleStr = str(module).split('(')[0]
    mod = '\t\tself.{} = nn.{}(dim={})'.format(layerName, moduleStr, module.dim)
    writer.toWrite['modules'].append(mod)
    
    writer.write_module_forward(modName)
#}}}

# custom modules
def residual_backbone(writer, modName, module, main_branch, residual_branch, aggregation_op):
#{{{
    inputToBlock = writer.currIpChannels
    idx = writer.depBlk.instances.index(type(module))
    forVarBkp = writer.forVar

    # main path through residual
    if residual_branch is not None:
        opNode = '{}_main'.format(writer.forVar)
        forward = '\t\t{} = {}'.format(opNode, writer.forVar)
        writer.toWrite['forward'].append(forward)
        writer.forVar = opNode 
    
    for n,m in module.named_modules(): 
        fullName = "{}.{}".format(modName, n)
        
        if not any(x in n for x in writer.depBlk.dsLayers[idx]):
            main_branch(n, m, fullName, writer)
    
    writer.forVar = forVarBkp
    outputOfBlock = writer.currIpChannels

    if residual_branch is not None:
        # downsampling path if exists
        opNode1 = '{}_residual'.format(writer.forVar)
        forward = '\t\t{} = {}'.format(opNode1, writer.forVar)
        writer.toWrite['forward'].append(forward)
        forVarBkp = writer.forVar
        writer.forVar = opNode1
        
        for n,m in module.named_modules(): 
            fullName = "{}.{}".format(modName, n)
            if any(x in n for x in writer.depBlk.dsLayers[idx]):
                residual_branch(n, m, fullName, writer, inputToBlock, outputOfBlock)
        
        writer.forVar = forVarBkp
        
        if aggregation_op is not None:
            aggregation_op(writer, opNode, opNode1)
#}}}

def split_and_aggregate_backbone(writer, parentModName, parentModule, branchStarts, branchProcs, aggregation_op): 
#{{{
    assert len(branchStarts) == len(branchProcs), 'For each branch a processing function must be provided - branches = {}, procFuns = {}'.format(len(branchConvs), len(branchProcs))

    inputToBlock = writer.currIpChannels
    forVarBkp = writer.forVar

    branchOpChannels = []
    opNodes = []
            
    for idx in range(len(branchStarts)):
        branchVar = "{}_{}".format(writer.forVar, idx)
        opNodes.append(branchVar)
        writer.toWrite['forward'].append("\t\t{} = {}".format(branchVar, writer.forVar))
        writer.forVar = branchVar
        writer.currIpChannels = inputToBlock

        inBranch = False
        for n,m in parentModule._modules.items(): 
            if n in branchStarts and not inBranch:
                inBranch = True
                branchStarts.pop(0)
            elif n in branchStarts and inBranch: 
                break
            
            if inBranch:
                fullName = "{}.{}".format(parentModName, n)
                branchProcs[idx](writer, fullName, m)
        
        branchOpChannels.append(writer.currIpChannels)
        writer.forVar = forVarBkp

    if aggregation_op is not None:
        aggregation_op(writer, opNodes, branchOpChannels)  
#}}}

def residual(writer, modName, module):
#{{{
    def main_branch(n, m, fullName, writer): 
    #{{{
        if isinstance(m, nn.Conv2d): 
            idx = writer.depBlk.instances.index(type(module))
            writer.addRelu = (n != writer.depBlk.convs[idx][0])
            nn_conv2d(writer, fullName, m)

        elif isinstance(m, nn.BatchNorm2d): 
            nn_batchnorm2d(writer, fullName, m)
            if writer.addRelu: 
                nn_relu(writer, fullName, m)
    #}}}
    
    def residual_branch(n, m, fullName, writer, ipToBlock, opOfBlock): 
    #{{{
        if isinstance(m, nn.Conv2d): 
            m.in_channels = ipToBlock
            m.out_channels = opOfBlock
            writer.currIpChannels = opOfBlock
            
            writer.write_module_desc(fullName, m)
            writer.write_module_forward(fullName)
        
        elif isinstance(m, nn.BatchNorm2d):
            nn_batchnorm2d(writer, fullName, m)
    #}}}

    def aggregation_op(writer, node1, node2):
    #{{{
        forward = '\t\t{} = F.relu({} + {})'.format(writer.forVar, node1, node2)
        writer.toWrite['forward'].append(forward)
    #}}}
    
    residual_backbone(writer, modName, module, main_branch, residual_branch, aggregation_op)
#}}}

def mb_conv(writer, modName, module):
#{{{
    def main_branch(n, m, fullName, writer): 
    #{{{
        if isinstance(m, nn.Conv2d): 
            idx = writer.depBlk.instances.index(type(module))
            writer.convIdx = writer.depBlk.convs[idx].index(n)
            nn_conv2d(writer, fullName, m, dw=(writer.convIdx==1))

        elif isinstance(m, nn.BatchNorm2d): 
            nn_batchnorm2d(writer, fullName, m)
            if writer.convIdx == 0 or writer.convIdx == 1: 
                nn_relu(writer, fullName, m)
    #}}}
    
    def residual_branch(n, m, fullName, writer, ipToBlock, opOfBlock): 
    #{{{
        if isinstance(m, nn.Conv2d): 
            m.in_channels = ipToBlock
            m.out_channels = opOfBlock
            writer.currIpChannels = opOfBlock
            
            writer.write_module_desc(fullName, m)
            writer.write_module_forward(fullName)
        
        elif isinstance(m, nn.BatchNorm2d):
            nn_batchnorm2d(writer, fullName, m)
    #}}}
    
    def aggregation_op(writer, node1, node2):
    #{{{
        forward = '\t\t{} = {} + {}'.format(writer.forVar, node1, node2)
        writer.toWrite['forward'].append(forward)
    #}}}
    
    idx = writer.depBlk.instances.index(type(module))
    midConv = writer.depBlk.convs[idx][1] 
    for n,m in module.named_modules(): 
        if n == midConv: 
            stride = m.stride[0]
    if stride == 2: 
        residual_backbone(writer, modName, module, main_branch, None, None)
    else:
        residual_backbone(writer, modName, module, main_branch, residual_branch, aggregation_op)
#}}}

def fire(writer, modName, module): 
#{{{
    def basic(writer, fullName, module): 
    #{{{
        if isinstance(module, nn.Conv2d): 
            nn_conv2d(writer, fullName, module)

        if isinstance(module, nn.BatchNorm2d): 
            nn_batchnorm2d(writer, fullName, module)
    #}}}

    def aggregation_op(writer, opNodes, branchOpChannels): 
    #{{{
        writer.currIpChannels = sum(branchOpChannels)
        nodes = "[{}]".format(','.join(opNodes))
        writer.toWrite['forward'].append("\t\t{} = torch.cat({}, 1)".format(writer.forVar, nodes))
        nn_relu(writer, None, None)
    #}}}
    
    inputToBlock = writer.currIpChannels
    idx = writer.depBlk.instances.index(type(module))
    convs = writer.depBlk.convs[idx]

    for n,m in module.named_modules(): 
        fullName = "{}.{}".format(modName, n)
        
        if isinstance(m, nn.Conv2d): 
            if n == convs[0]:
                nn_conv2d(writer, fullName, m)
        
        if isinstance(m, nn.BatchNorm2d):
            nn_batchnorm2d(writer, fullName, m)
        
        if isinstance(m, nn.ReLU): 
            nn_relu(writer, fullName, m)     
            split_and_aggregate_backbone(writer, modName, module, convs[1:], [basic,basic], aggregation_op)
            break
#}}}
