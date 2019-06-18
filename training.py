import sys
import time
from tqdm import tqdm

import torch.autograd

import src.utils as utils
import src.training as trainingSrc

import pruning.methods as pruningMethods
import pruning.utils as pruningUtils

class Trainer(trainingSrc.Trainer):
    def finetune_network(self, params, tbx_writer, checkpointer, train_loader, test_loader, valLoader, model, criterion, optimiser, inferer):  
        print('Epoch,\tLR,\tTrain_Loss,\tTrain_Top1,\tTrain_Top5,\tTest_Loss,\tTest_Top1,\tTest_Top5,\tVal_Loss,\tVal_Top1,\tVal_Top5')
        
        for epoch in tqdm(range(params.start_epoch, params.epochs), desc='training', leave=False) : 
            params.curr_epoch = epoch
            state = self.update_lr(params, optimiser)
    
            losses = utils.AverageMeter()
            top1 = utils.AverageMeter()
            top5 = utils.AverageMeter()
            
            self.batch_iter(model, criterion, optimiser, train_loader, params, losses, top1, top5)
            
            params.train_loss = losses.avg        
            params.train_top1 = top1.avg        
            params.train_top5 = top5.avg        
            
            print('pruneafter: ', params.pruneAfter)
            print('prune weights: ', params.pruneWeights)
            print('epoch', epoch)

            # perform pruning 
            if (params.pruneWeights == True or params.pruneFilters == True) and ((epoch+1) % params.pruneAfter == 0): 
                tqdm.write('Pruning Network')
                model = pruningMethods.prune_model(params, model)
                params.pruningPerc += params.prunePercIncrement
                totalPrunedPerc = pruningUtils.prune_rate(params, model)
                tqdm.write('Pruned Percentage = {}'.format(totalPrunedPerc))
                checkpointer.log_prune_rate(params, totalPrunedPerc)
                params.prunePercPerLayer = []
    
            # get test loss
            params.test_loss, params.test_top1, params.test_top5 = inferer.test_network(params, test_loader, model, criterion, optimiser)
            params.val_loss, params.val_top1, params.val_top5 = inferer.test_network(params, valLoader, model, criterion, optimiser)
            
            checkpointer.save_checkpoint(model.state_dict(), optimiser.state_dict(), params)
            
            tqdm.write("{},\t{:10.5f},\t{:10.5f},\t{:10.5f},\t{:10.5f},\t{:10.5f},\t{:10.5f},\t{:10.5f},\t{:10.5f},\t{:10.5f},\t{:10.5f}".format(epoch, params.lr, params.train_loss, params.train_top1, params.train_top5, params.test_loss, params.test_top1, params.test_top5, params.val_loss, params.val_top1, params.val_top5))
