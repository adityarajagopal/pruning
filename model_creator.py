import sys
import os

import torch.nn
import torch.backends
import torchvision 

import src.model_creator as mcSrc

class ModelCreator(mcSrc.ModelCreator):
#{{{
    def setup_optimiser(self, params, model):
        if params.optimiser == 'sgd':
            return torch.optim.SGD(model.parameters(), lr=params.lr, momentum=params.momentum, weight_decay=params.weight_decay)
        elif params.optimiser == 'rmsprop':
            return torch.optim.RMSprop(model.parameters(), lr=params.lr, momentum=params.momentum, weight_decay=params.weight_decay)
    
    def read_model(self, params):
    #{{{
        if params.dataset == 'cifar10' : 
            import src.ar4414.pruning.models.cifar as models 
            num_classes = 10
            scale = 224/32
    
        elif params.dataset == 'cifar100' : 
            import src.ar4414.pruning.models.cifar as models 
            num_classes = 100
            scale = 224/32
    
        else : 
            import src.ar4414.pruning.models.imagenet as models 
            num_classes = 1000
            scale = 1 
    
        print("Creating Model %s" % params.arch)
        
        if 'resnet' in params.arch:
            if 'cifar' in params.dataset:
                model = models.__dict__[params.arch](num_classes=num_classes, depth=params.depth)
            else:
                model = models.__dict__['resnet{}'.format(params.depth)](pretrained=False, progress=False)
        elif 'efficientnet' in params.arch: 
            #TODO: make all the parameters below configurable
            model = models.__dict__[params.arch](num_classes=num_classes, scale=scale)
        else:
            model = models.__dict__[params.arch](num_classes=num_classes)

        return model
    #}}}
    
    def load_pretrained(self, params, model):
    #{{{
        if params.resume or params.branch or params.entropy or params.pruneFilters: 
            checkpoint = torch.load(params.pretrained)
            model.load_state_dict(checkpoint)

        elif params.fbsPruning:
            device_id = params.gpu_list[0]
            location = 'cuda:'+str(device_id)
            checkpoint = torch.load(params.pretrained, map_location=location)

            if 'E_g_x' in str(checkpoint.keys()):
                checkpoint = {k.replace('module.','') : v for k,v in checkpoint.items()}
                model.module.load_state_dict(checkpoint, initialise=False)
            else:
                checkpoint = {k.replace('module.','') : v for k,v in checkpoint.items()}
                model.module.load_state_dict(checkpoint, initialise=True)
            
        elif params.finetune or params.getGops:
            device_id = params.gpu_list[0]
            location = 'cuda:'+str(device_id)
            checkpoint = torch.load(params.pretrained, map_location=location)

            if params.getGops == True:
                masks = [v for k,v in checkpoint.items() if 'mask' in k]
                if masks != []:
                    print('Setting pruning masks')
                    model.module.set_masks(masks)
            
            # model.module.load_state_dict(checkpoint)
            model.load_state_dict(checkpoint)
    
        elif params.evaluate or params.unprunedTestAcc: 
            device_id = params.gpu_list[0]
            location = 'cuda:'+str(device_id)
            checkpoint = torch.load(params.pretrained, map_location=location)
            model.load_state_dict(checkpoint)
            
        torch.backends.cudnn.benchmark = True
        print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

        return model
    #}}}
#}}}

def get_model_size(model):
#{{{
    params = 0
    for p in model.named_parameters():
        paramsInLayer = 1
        for dim in p[1].size():
            paramsInLayer *= dim
        params += (paramsInLayer * 4) / 1e6

    return params
#}}}
