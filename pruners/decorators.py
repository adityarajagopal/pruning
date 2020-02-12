import sys

import src.ar4414.pruning.pruners.dependencies as dependencies
import src.ar4414.pruning.pruners.model_writers as writers

def check_kwargs(**kwargs): 
#{{{
    """Check that kwargs has atleast the lType argument, if not it is an invalid decorator"""
    if 'lType' in kwargs.keys(): 
        return 
    else: 
        print("ERROR : Decorator without lType argument declared. This is invalid")
        sys.exit()
#}}}

def basic_block(**kwargs):
#{{{
    def decorator(block): 
        check_kwargs(**kwargs)
        dependencies.DependencyBlock.update_block_names(block, **kwargs)
        dependencies.DependencyBlock.register_dependency_calculator(kwargs['lType'], dependencies.Residual())
        writers.Writer.register_writer(kwargs['lType'], writers.residual)
        return block
    return decorator
#}}}

def bottleneck(**kwargs):
#{{{
    def decorator(block): 
        check_kwargs(**kwargs)
        dependencies.DependencyBlock.update_block_names(block, **kwargs)
        dependencies.DependencyBlock.register_dependency_calculator(kwargs['lType'], dependencies.Residual())
        writers.Writer.register_writer(kwargs['lType'], writers.residual)
        return block
    return decorator
#}}}

def mb_conv(**kwargs):
#{{{
    def decorator(block): 
        check_kwargs(**kwargs)
        dependencies.DependencyBlock.update_block_names(block, **kwargs)
        dependencies.DependencyBlock.register_dependency_calculator(kwargs['lType'], dependencies.MBConv())
        writers.Writer.register_writer(kwargs['lType'], writers.mb_conv)
        return block
    return decorator
#}}}

def fire(**kwargs):
#{{{
    def decorator(block): 
        check_kwargs(**kwargs)
        dependencies.DependencyBlock.update_block_names(block, **kwargs)
        writers.Writer.register_writer(kwargs['lType'], writers.fire)
        return block
    return decorator
#}}}
