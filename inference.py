import torch.autograd
from tqdm import tqdm

import src.utils as utils

class Inferer(object):
    def run_single_forward(self, params, test_loader, model) :  
        model.eval()

        inputs, targets = next(iter(test_loader))
        with torch.no_grad():
            if params.use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
        
        outputs = model(inputs) 
        
        return
    
    def test_network(self, params, test_loader, model, criterion, optimiser) :  
        model.eval()
            
        losses = utils.AverageMeter()
        top1 = utils.AverageMeter()
        top5 = utils.AverageMeter()

        for batch_idx, (inputs, targets) in tqdm(enumerate(test_loader), total=len(test_loader)-1, desc='inference', leave=False) : 
            # move inputs and targets to GPU
            with torch.no_grad():
                if params.use_cuda : 
                    inputs, targets = inputs.cuda(), targets.cuda()
                inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
                
                # perform inference 
                outputs = model(inputs) 
                loss = criterion(outputs, targets)
            
            prec1, prec5 = utils.accuracy(outputs.data, targets.data)
    
            losses.update(loss) 
            top1.update(prec1) 
            top5.update(prec5)
    
        if params.evaluate == True : 
            tqdm.write('Loss: {}, Top1: {}, Top5: {}'.format(losses.avg, top1.avg, top5.avg))
        return (losses.avg, top1.avg, top5.avg)
