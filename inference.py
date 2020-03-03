import sys
import numpy as np
from tqdm import tqdm

import torch

import src.utils as utils
import src.inference as infSrc
from src.ar4414.pruning.timers import Timer

class Inferer(infSrc.Inferer):
    def run_n_minibatches(self, params, test_loader, model, numMB) :  
        #{{{
        model.eval()
            
        for batch_idx, (inputs, targets) in tqdm(enumerate(test_loader), total=len(test_loader)-1, desc='inference', leave=False) : 
            # move inputs and targets to GPU
            with torch.no_grad():
                device = 'cuda:' + str(params.gpuList[0])
                if params.use_cuda : 
                    inputs, targets = inputs.cuda(device, non_blocking=True), targets.cuda(device, non_blocking=True)
                
                # perform inference 
                outputs = model(inputs) 
            
            if (batch_idx + 1) == numMB:
                return
        #}}}
        
    def run_single_minibatch(self, params, test_loader, model) :  
        #{{{
        model.eval()

        inputs, targets = next(iter(test_loader))
        with torch.no_grad():
            if params.use_cuda:
                device = 'cuda:' + str(params.gpu_id)
                inputs, targets = inputs.cuda(device, non_blocking=True), targets.cuda( device, non_blocking=True)
        
        outputs = model(inputs) 
        
        return
        #}}}
    
    def test_network(self, params, test_loader, model, criterion, optimiser, verbose=True) :  
    #{{{
        model.eval()
            
        losses = utils.AverageMeter()
        top1 = utils.AverageMeter()
        top5 = utils.AverageMeter()

        self.infTimer = Timer('Inference')
    
        for batch_idx, (inputs, targets) in tqdm(enumerate(test_loader), total=len(test_loader)-1, desc='inference', leave=False) : 
            # move inputs and targets to GPU
            with torch.no_grad():
                device = 'cuda:' + str(params.gpuList[0])
                if params.use_cuda : 
                    inputs, targets = inputs.cuda(device, non_blocking=True), targets.cuda(device, non_blocking=True)
                
                # perform inference 
                with self.infTimer:
                    outputs = model(inputs) 

                loss = criterion(outputs, targets)
            
            prec1, prec5 = utils.accuracy(outputs.data, targets.data)
    
            losses.update(loss.item()) 
            top1.update(prec1.item()) 
            top5.update(prec5.item())

        self.infTimer.update_stats('minibatch_time', np.mean(self.infTimer.timestep))
        self.infTimer.reset()

        # if params.evaluate == True or (params.finetune == False and (params.entropy or params.pruneFilters)): 
        # if params.evaluate or params.entropy or (params.pruneFilters and not params.finetune): 
        if verbose: 
            tqdm.write('Loss: {}, Top1: {}, Top5: {}'.format(losses.avg, top1.avg, top5.avg))
        
        return (losses.avg, top1.avg, top5.avg)
    #}}}
    
