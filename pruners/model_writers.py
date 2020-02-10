import sys

import torch.nn as nn
    
def register_writer(blockName, func): 
#{{{
    writerFunctions[blockName] = func 
#}}}

def write_module(toWrite, modName, module): 
#{{{
    layerName = '_'.join(modName.split('.')[1:])
    mod = '\t\tself.{} = nn.{}'.format(layerName, str(module))
    toWrite['modules'].append(mod)
#}}}

def write_forward(toWrite, modName, ipNode, opNode, silent=False):
#{{{
    if not silent:
        layerName = '_'.join(modName.split('.')[1:])
        forward = '\t\t{} = self.{}({})'.format(ipNode, layerName, opNode) 
        toWrite['forward'].append(forward)
#}}}

def nn_conv2d(toWrite, modName, module, currIpChannels, channelsPruned, depBlk=None, silent=False): 
#{{{
    module.in_channels = currIpChannels 
    module.out_channels -= channelsPruned[modName]
    currIpChannels = module.out_channels
    
    write_module(toWrite, modName, module)
    write_forward(toWrite, modName, 'x', 'x', silent)

    return currIpChannels
#}}}

def nn_relu(toWrite, modName, module, currIpChannels=None, channelsPruned=None, depBlk=None, silent=False): 
#{{{
    if not silent:
        toWrite['forward'].append('\t\tx = F.relu(x)') 
#}}}

def nn_avgpool2d(toWrite, modName, module, currIpChannels=None, channelsPruned=None, depBlk=None, silent=False): 
#{{{
    write_module(toWrite, modName, module)
    write_forward(toWrite, modName, 'x', 'x', silent)
#}}}

def nn_batchnorm2d(toWrite, modName, module, currIpChannels, channelsPruned=None, depBlk=None, silent=False): 
#{{{
    module.num_features = currIpChannels
    
    write_module(toWrite, modName, module)
    write_forward(toWrite, modName, 'x', 'x', silent)
    
    return currIpChannels
#}}}

def nn_linear(toWrite, modName, module, currIpChannels, channelsPruned=None, depBlk=None, silent=False): 
#{{{
    module.in_features = currIpChannels
    
    write_module(toWrite, modName, module)
    write_forward(toWrite, modName, 'x', 'x', silent)
    
    return currIpChannels
#}}}

def residual_basic(toWrite, modName, module, currIpChannels, channelsPruned, depBlk):
#{{{
    inputToBlock = currIpChannels
    idx = depBlk.instances.index(type(module))

    # main path through residual
    opNode = 'out'
    forward = '\t\t{} = x'.format(opNode)
    toWrite['forward'].append(forward)
    
    for n,m in module.named_modules(): 
        fullName = "{}.{}".format(modName, n)
        
        if not any(x in n for x in depBlk.dsLayers[idx]):
            if isinstance(m, nn.Conv2d): 
                currIpChannels = nn_conv2d(toWrite, fullName, m, currIpChannels, channelsPruned, silent=True)
                write_forward(toWrite, fullName, opNode, opNode)

            elif isinstance(m, nn.BatchNorm2d): 
                currIpChannels = nn_batchnorm2d(toWrite, n, m, currIpChannels, silent=True)
                write_forward(toWrite, fullName, opNode, opNode)
            
            elif isinstance(m, nn.ReLU): 
                nn_relu(toWrite, n, m, silent=True)
                forward = '\t\t{} = F.relu({})'.format(opNode, opNode) 
                toWrite['forward'].append(forward)
    
    outputOfBlock = currIpChannels

    # downsampling path if exists
    opNode1 = 'out_res'
    forward = '\t\t{} = x'.format(opNode1)
    toWrite['forward'].append(forward)
    
    for n,m in module.named_modules(): 
        fullName = "{}.{}".format(modName, n)
        if any(x in n for x in depBlk.dsLayers[idx]):
            if isinstance(m, nn.Conv2d): 
                m.in_channels = inputToBlock
                m.out_channels = outputOfBlock
                currIpChannels = outputOfBlock
    
                write_module(toWrite, fullName, m)
                write_forward(toWrite, fullName, opNode1, opNode1)
            
            elif isinstance(m, nn.BatchNorm2d):
                m.num_features = currIpChannels 
                
                write_module(toWrite, fullName, m)
                write_forward(toWrite, fullName, opNode1, opNode1)
    
    forward = '\t\tx = F.relu({} + {})'.format(opNode, opNode1)
    toWrite['forward'].append(forward)

    return outputOfBlock
#}}}

def residual_bottleneck(toWrite, modName, module, currIpChannels, channelsPruned, depBlk):
    pass

def mb_conv(toWrite, modName, module, currIpChannels, channelsPruned, depBlk): 
    pass

writerFunctions = {'basic': nn_conv2d, 'relu': nn_relu, 'avgpool2d':nn_avgpool2d, 'batchnorm2d': nn_batchnorm2d, 'linear': nn_linear}
