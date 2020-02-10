import sys

import torch.nn as nn
from abc import ABC,abstractmethod 

class DependencyCalculator(ABC):
#{{{
    def __init__(self):
        pass

    @abstractmethod
    def dependent_conv(self, layerName, convs): 
        pass

    @abstractmethod
    def internal_dependency(self, module, mType, convs, ds):
        pass
    
    @abstractmethod
    def external_dependency(self, module, mType, convs, ds):
        pass
#}}}

class Basic(DependencyCalculator):
#{{{
    def __init__(self):
        pass
    
    def dependent_conv(self, layerName, convs): 
        """Basic conv itself is the dependent conv"""
        return layerName
    
    def internal_dependency(self, module, mType, convs, ds):
        """Basic conv blocks don't have internal dependencies"""
        return False,None
    
    def external_dependency(self, module, mType, convs, ds):
        """Basic conv blocks don't have external dependencies"""
        return False,None
#}}}

class Residual(DependencyCalculator):
#{{{
    def __init__(self):
        pass
    
    def dependent_conv(self, layerName, convs): 
        """convs variable only has 1 conv with residuals"""
        return "{}.{}".format(layerName, convs[0])

    def internal_dependency(self, module, mType, convs, ds):
        """Residual blocks don't have internal dependencies"""
        return False,None

    def external_dependency(self, module, mType, convs, ds): 
    #{{{
        """
        Checks if module has a downsampling layer or not
        Some modules implement downsampling as just a nn.Sequential that's empty, so checks deeper to see if there is actually a conv inside
        Returns whether dependency exists and list of dependent layers
        """
        
        childrenAreDsLayers = [(c in ds) for c in list(module._modules.keys())]
        if any(childrenAreDsLayers):
            #check if empty sequential
            idx = childrenAreDsLayers.index(True)
            layerName = list(module._modules.keys())[idx]
            return not DependencyBlock.check_children(module._modules[layerName], [nn.Conv2d]),[convs[0]]
        else:
            return True, [convs[0]]
    #}}}
#}}}

class MBConv(DependencyCalculator):
#{{{
    def __init__(self):
        pass
    
    def dependent_conv(self, layerName, convs): 
        """conv3 is the externally dependent conv with mb_conv blocks"""
        return "{}.{}".format(layerName, convs[2])

    def internal_dependency(self, module, mType, convs, ds):
        """convs 1 and 2 (dw convs) are internally dependent in mb_convs blocks"""
        return True, [convs[0], convs[1]]

    def external_dependency(self, module, mType, convs, ds): 
    #{{{
        """
        If no downsampling layer exists, then dependency exists
        If downsampling layer exists and it is of instance nn.conv2d, then no dependency exists
        If downsampling layer exists and does not have an nn.conv2d, if stide of conv2 is 1 dependency exists otherwise no
        Returns whether dependency exists and list of dependent layers
        """
        depLayers = [convs[2]]
        
        childrenAreDsLayers = [(c in ds) for c in list(module._modules.keys())]
        if any(childrenAreDsLayers):
            #check if empty sequential
            idx = childrenAreDsLayers.index(True)
            layerName = list(module._modules.keys())[idx]

            if DependencyBlock.check_children(module._modules[layerName], [nn.Conv2d]):
                return False, depLayers 
            else:
                if module._modules[convs[1]].stride[0] == 1:
                    return True, depLayers 
                else:
                    return False, depLayers
        else:
            return True, depLayers 
    #}}}
#}}}

class DependencyBlock(object):
#{{{
    def __init__(self, model):
    #{{{
        self.model = model
        
        try:
            self.types = self.dependentLayers['type']
            self.instances = self.dependentLayers['instance']
            self.convs = self.dependentLayers['conv']
            self.dsLayers = self.dependentLayers['downsample']
        except AttributeError: 
            print("Instantiating dependency block without decorators on model or for class without dependencies")

        self.linkedConvs = self.create_conv_graph()

        if hasattr(self, 'depCalcs'):
            self.depCalcs['basic'] = Basic()
        else: 
            self.depCalcs = {'basic': Basic()}
    #}}}
    
    @classmethod
    def update_block_names(cls, blockInst, *args):
    #{{{
        if hasattr(cls, 'dependentLayers'):
            cls.dependentLayers['type'].append(args[0])
            cls.dependentLayers['instance'].append(blockInst)
            cls.dependentLayers['conv'].append(args[1])
            cls.dependentLayers['downsample'].append(args[2])
        else:
            setattr(cls, 'dependentLayers', {'type':[args[0]], 'instance':[blockInst], 'conv':[args[1]], 'downsample':[args[2]]})
    #}}}

    @classmethod 
    def register_dependency_calculator(cls, blockName, calcFunc):
    #{{{
        if hasattr(cls, 'depCalcs'): 
            cls.depCalcs[blockName] = calcFunc
        else: 
            setattr(cls, 'depCalcs', {blockName: calcFunc})
    #}}}
    
    @classmethod
    def check_children(cls, module, instances): 
    #{{{
        """Checks if module has any children that are of type in list instances""" 
        check = []
        for m in module.modules():
            check += [any(isinstance(m,inst) for inst in instances)]
        return any(check)
    #}}}

    @classmethod
    def check_inst(cls, module, instances): 
        """Checks if module is of one of the types in list instances"""
        return any(isinstance(module, inst) for inst in instances)

    def create_conv_graph(self): 
    #{{{
        """
        Returns a list which has order of convs and modules which have an instance of module in instances in the entire network
        eg. conv1 -> module1(which has as an mb_conv) -> conv2 -> module2 ...    
        """
        linkedConvs = []
        for n,m in self.model.module.named_children(): 
            if not DependencyBlock.check_children(m, self.instances):
                if isinstance(m,nn.Conv2d):
                    linkedConvs.append(('basic',"module.{}".format(n)))
            else:
                for _n,_m in m.named_modules():
                    if DependencyBlock.check_inst(_m, self.instances):
                        idx = self.instances.index(type(_m))
                        mType = self.types[idx]
                        linkedConvs.append((mType, "module.{}.{}".format(n,_n)))
        
        return linkedConvs
    #}}}

    def get_dependencies(self):
    #{{{
        intDeps = []
        extDeps = []
        tmpDeps = []
        for n,m in self.model.named_modules(): 
            if DependencyBlock.check_inst(m, self.instances):
                idx = self.instances.index(type(m))
                mType = self.types[idx]
                convs = self.convs[idx]
                ds = self.dsLayers[idx]
                
                internallyDep, depLayers = self.depCalcs[mType].internal_dependency(m, mType, convs, ds)
                if internallyDep: 
                    intDeps.append(["{}.{}".format(n,x) for x in depLayers])
                
                externallyDep, depLayers = self.depCalcs[mType].external_dependency(m, mType, convs, ds)
                if externallyDep: 
                    depLayers = ["{}.{}".format(n,x) for x in depLayers]

                    if len(tmpDeps) != 0: 
                        tmpDeps += depLayers
                    else: 
                        bType,name = zip(*self.linkedConvs)
                        idx = name.index(n)
                        prevType = bType[idx-1]
                        prev = self.depCalcs[prevType].dependent_conv(name[idx-1], convs)
                        tmpDeps = [prev,*depLayers]
                else: 
                    if len(tmpDeps) != 0:
                        extDeps.append(tmpDeps)
                    tmpDeps = []
        
        if len(tmpDeps) != 0: 
            extDeps.append(tmpDeps)

        return intDeps,extDeps
    #}}}
#}}}

