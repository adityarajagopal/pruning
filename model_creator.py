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
    
    def load_pretrained(self, params, model):
        if params.resume == True or params.branch == True: 
            checkpoint = torch.load(params.pretrained)
            model.load_state_dict(checkpoint)

        elif params.getGops == True:
            device_id = params.gpu_list[0]
            location = 'cuda:'+str(device_id)
            checkpoint = torch.load(params.pretrained, map_location=location)
            model.load_state_dict(checkpoint, strict=False)
            
            masks = [v.cpu() for k,v in checkpoint.items() if 'mask' in k]
            if masks != []:
                print('Setting pruning masks')
                model.module.set_masks(masks)
    
        if params.evaluate == True : 
            checkpoint = torch.load(params.pretrained)
            model.load_state_dict(checkpoint['state_dict'])
            
        torch.backends.cudnn.benchmark = True
        print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

        return model
