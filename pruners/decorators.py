import src.ar4414.pruning.pruners.dependencies as dependencies
import src.ar4414.pruning.pruners.model_writers as writers

def basic_block(*args, **kwargs):
#{{{
    def decorator(block): 
        dependencies.DependencyBlock.update_block_names(block, *args)
        dependencies.DependencyBlock.register_dependency_calculator(args[0], dependencies.Residual())
        writers.Writer.register_writer(args[0], writers.residual)
        return block
    return decorator
#}}}

def bottleneck(*args, **kwargs):
#{{{
    def decorator(block): 
        dependencies.DependencyBlock.update_block_names(block, *args)
        dependencies.DependencyBlock.register_dependency_calculator(args[0], dependencies.Residual())
        writers.Writer.register_writer(args[0], writers.residual)
        return block
    return decorator
#}}}

def mb_conv(*args, **kwargs):
#{{{
    def decorator(block): 
        dependencies.DependencyBlock.update_block_names(block, *args)
        dependencies.DependencyBlock.register_dependency_calculator(args[0], dependencies.MBConv())
        writers.Writer.register_writer(args[0], writers.mb_conv)
        return block
    return decorator
#}}}
