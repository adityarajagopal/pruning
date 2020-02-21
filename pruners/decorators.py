import sys

import src.ar4414.pruning.pruners.dependencies as dependencies
import src.ar4414.pruning.pruners.model_writers as writers
import src.ar4414.pruning.pruners.weight_transfer as weight_transfer

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
        weight_transfer.WeightTransferUnit.register_transfer_func(kwargs['lType'], weight_transfer.residual)
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
        weight_transfer.WeightTransferUnit.register_transfer_func(kwargs['lType'], weight_transfer.residual)
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
        weight_transfer.WeightTransferUnit.register_transfer_func(kwargs['lType'], weight_transfer.mb_conv)
        return block
    return decorator
#}}}

def fire(**kwargs):
#{{{
    def decorator(block): 
        check_kwargs(**kwargs)
        dependencies.DependencyBlock.update_block_names(block, **kwargs)
        dependencies.DependencyBlock.register_dependency_calculator(kwargs['lType'], dependencies.Fire())
        writers.Writer.register_writer(kwargs['lType'], writers.fire)
        weight_transfer.WeightTransferUnit.register_transfer_func(kwargs['lType'], weight_transfer.fire)
        return block
    return decorator
#}}}

def skip(**kwargs): 
#{{{
    def decorator(block): 
        dependencies.DependencyBlock.skip_layers(block, **kwargs)
        return block 
    return decorator
#}}}
