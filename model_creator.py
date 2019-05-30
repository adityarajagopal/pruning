import torch.nn
import torch.backends
import torchvision 

import src.model_creator as mcSrc

class ModelCreator(mcSrc.ModelCreator):
    
    def read_model(self, params):
        if params.dataset == 'cifar10' : 
            import src.ar4414.pruning.models.cifar as models 
            num_classes = 10
    
        elif params.dataset == 'cifar100' : 
            import src.ar4414.pruning.models.cifar as models 
            num_classes = 100
    
        else : 
            import src.ar4414.pruning.models.imagenet as models 
            num_classes = 1000
    
        print("Creating Model %s" % params.arch)
        
        if params.arch.endswith('resnet'):
            model = models.__dict__[params.arch](
                        num_classes=num_classes,
                        depth=params.depth
                    )
        else:
            model = models.__dict__[params.arch](num_classes=num_classes)

        return model
