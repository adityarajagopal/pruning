import torch.autograd
from tqdm import tqdm
import sys

import src.utils as utils
import src.inference as infSrc

class Inferer(infSrc.Inferer):
    def run_single_forward(self, params, test_loader, model) :  
        model.eval()

        inputs, targets = next(iter(test_loader))
        with torch.no_grad():
            if params.use_cuda:
                device = 'cuda:' + str(params.gpu_id)
                inputs, targets = inputs.cuda(device, non_blocking=True), targets.cuda( device, non_blocking=True)
        
        outputs = model(inputs) 
        
        return
    
    # def test_network(self, params, test_loader, model, criterion, optimiser) :  
    #     model.eval()
    #         
    #     losses = utils.AverageMeter()
    #     top1 = utils.AverageMeter()
    #     top5 = utils.AverageMeter()

    #     for batch_idx, (inputs, targets) in tqdm(enumerate(test_loader), total=len(test_loader)-1, desc='inference', leave=False) : 
    #         # move inputs and targets to GPU
    #         with torch.no_grad():
    #             device = 'cuda:' + str(params.gpu_id)
    #             if params.use_cuda : 
    #                 inputs, targets = inputs.cuda(device, non_blocking=True), targets.cuda(device, non_blocking=True)
    #             
    #             # perform inference 
    #             outputs = model(inputs) 
    #             loss = criterion(outputs, targets)
    #         
    #         prec1, prec5 = utils.accuracy(outputs.data, targets.data)
    # 
    #         losses.update(loss) 
    #         top1.update(prec1) 
    #         top5.update(prec5)
    # 
    #     if params.evaluate == True : 
    #         tqdm.write('Loss: {}, Top1: {}, Top5: {}'.format(losses.avg, top1.avg, top5.avg))
    #     return (losses.avg, top1.avg, top5.avg)
