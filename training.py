import gc
import sys
import time

import src.utils as utils
import src.training as trainingSrc
from src.ar4414.pruning.timers import Timer

import numpy as np
from tqdm import tqdm

import torch

class Trainer(trainingSrc.Trainer):
    def __init__(self, params):
        super().__init__()
        self.fbsPruning = params.fbsPruning
        self.mbTimer = Timer("Minibatch")
        self.pruneTimer = Timer("Pruning")
        self.dataTimer = Timer("Data-loading")

    def train(self, model, criterion, optimiser, inputs, targets) : 
    #{{{
        model.train()
        
        with self.mbTimer:
            outputs = model(inputs) 
            loss = criterion(outputs, targets)
            model.zero_grad() 
            loss.backward() 
            optimiser.step()
        
        prec1, prec5 = utils.accuracy(outputs.data, targets.data) 
        
        return (loss, prec1, prec5)
    #}}} 
    
    def batch_iter(self, model, criterion, optimiser, train_loader, params, losses, top1, top5):
    #{{{
        for batch_idx, (inputs, targets) in tqdm(enumerate(train_loader), total=len(train_loader)-1, desc='epoch', leave=False): 
            # move inputs and targets to GPU
            device = 'cuda:'+str(params.gpuList[0])
            with self.dataTimer:
                if params.use_cuda : 
                    inputs, targets = inputs.cuda(device, non_blocking=True), targets.cuda(device, non_blocking=True)
            
            # train model
            loss, prec1, prec5 = self.train(model, criterion, optimiser, inputs, targets)
            
            losses.update(loss) 
            top1.update(prec1) 
            top5.update(prec5)

            if params.batchLim != -1 and batch_idx == params.batchLim:
                return
    #}}}
    
    def static_finetune_l1_weights(self, params, pruner, checkpointer, train_loader, test_loader, valLoader, model, criterion, optimiser, inferer):  
    #{{{
        print('Epoch,\tLR,\tTrain_Loss,\tTrain_Top1,\tTrain_Top5,\tTest_Loss,\tTest_Top1,\tTest_Top5,\tVal_Loss,\tVal_Top1,\tVal_Top5')
            
        def update_timers():
        #{{{
            self.mbTimer.update_stats(epoch, sum(self.mbTimer.timestep))
            self.pruneTimer.update_stats(epoch, sum(self.pruneTimer.timestep))
            self.dataTimer.update_stats(epoch, sum(self.dataTimer.timestep))
            self.mbTimer.reset()
            self.pruneTimer.reset()
            self.dataTimer.reset()
        #}}}

        for epoch in tqdm(range(params.start_epoch, params.finetuneBudget), desc='training', leave=False) : 
            params.curr_epoch = epoch
            state = self.update_lr(params, optimiser)

            # perform pruning 
            if params.pruneFilters == True and epoch == params.pruneAfter: 
                checkpointer.save_model_only(model.state_dict(), params.printOnly, 'pre_pruning')
                tqdm.write('Pruning Network')
                with self.pruneTimer:
                    channelsPruned, model, optimiser = pruner.prune_model(model)
                totalPrunedPerc, _, _ = pruner.prune_rate(model)
                tqdm.write('Pruned Percentage = {:.2f}%'.format(totalPrunedPerc))
                summary = pruner.log_pruned_channels(checkpointer.root, params, totalPrunedPerc, channelsPruned)

            losses = utils.AverageMeter()
            top1 = utils.AverageMeter()
            top5 = utils.AverageMeter()

            self.batch_iter(model, criterion, optimiser, train_loader, params, losses, top1, top5)

            update_timers()

            params.train_loss = losses.avg        
            params.train_top1 = top1.avg        
            params.train_top5 = top5.avg        
            
            # get test loss
            params.test_loss, params.test_top1, params.test_top5 = inferer.test_network(params, test_loader, model, criterion, optimiser, verbose=False)
            params.val_loss, params.val_top1, params.val_top5 = inferer.test_network(params, valLoader, model, criterion, optimiser, verbose=False)

            checkpointer.save_checkpoint(model.state_dict(), optimiser.state_dict(), params)
            
            tqdm.write("{},\t{:10.5f},\t{:10.5f},\t{:10.5f},\t{:10.5f},\t{:10.5f},\t{:10.5f},\t{:10.5f},\t{:10.5f},\t{:10.5f},\t{:10.5f}".format(epoch, params.lr, params.train_loss, params.train_top1, params.train_top5, params.test_loss, params.test_top1, params.test_top5, params.val_loss, params.val_top1, params.val_top5))
    #}}} 

    def perform_pruning(self, epoch, params):
    #{{{
        if self.networkPruned:
            return False

        if epoch == 15: 
            return True
        
        if epoch < 2:
            self.overfitCount = 0
        elif epoch == 2: 
            self.overfitCount = 0
            self.diff = params.val_loss - params.train_loss.data 
        else:
            currDiff = params.val_loss - params.train_loss.data
            threshold = self.diff
            if currDiff > threshold:
                self.overfitCount += 1
            else:
                self.overfitCount = 0
            tqdm.write("diff_thresh {:10.5f}, curr_diff {:10.5f}, oc {}".format(threshold, currDiff, self.overfitCount))
        
        if self.overfitCount == 3:
            return True
        else:
            return False
    #}}}
    
    def validation_finetune_l1_weights(self, params, pruner, checkpointer, train_loader, test_loader, valLoader, model, criterion, optimiser, inferer):  
    #{{{
        print('Epoch,\tLR,\tTrain_Loss,\tTrain_Top1,\tTrain_Top5,\tTest_Loss,\tTest_Top1,\tTest_Top5,\tVal_Loss,\tVal_Top1,\tVal_Top5')

        self.networkPruned = False        
        totalFinetuneEpochs = params.epochs
        for epoch in tqdm(range(params.start_epoch, params.epochs), desc='training', leave=False) : 
            params.curr_epoch = epoch

            if epoch == totalFinetuneEpochs:
                break
            
            # perform pruning 
            if self.perform_pruning(epoch, params): 
                tqdm.write('Pruning Network')
                channelsPruned, model, optimiser = pruner.prune_model(model)
                totalPrunedPerc, _, _ = pruner.prune_rate(model)
                tqdm.write('Pruned Percentage = {:.2f}%'.format(totalPrunedPerc))
                summary = pruner.log_pruned_channels(checkpointer.root, params, totalPrunedPerc, channelsPruned)
                self.networkPruned = True

                # update lr-schedule with epoch at which pruning occured before calling update_lr
                lrChangeInterval = params.finetuneBudget / 3
                params.lr_schedule[2] = epoch
                if len(params.lr_schedule) > 4:
                    params.lr_schedule[4] = int(epoch + lrChangeInterval)
                    params.lr_schedule[6] = int(epoch + 2 * lrChangeInterval)
                totalFinetuneEpochs = int(epoch + params.finetuneBudget)

                tqdm.write(" ".join([str(x) for x in params.lr_schedule]))

            state = self.update_lr(params, optimiser)
           
            losses = utils.AverageMeter()
            top1 = utils.AverageMeter()
            top5 = utils.AverageMeter()

            self.batch_iter(model, criterion, optimiser, train_loader, params, losses, top1, top5)

            params.train_loss = losses.avg        
            params.train_top1 = top1.avg        
            params.train_top5 = top5.avg        
            
            # get test loss
            params.test_loss, params.test_top1, params.test_top5 = inferer.test_network(params, test_loader, model, criterion, optimiser, verbose=False)
            params.val_loss, params.val_top1, params.val_top5 = inferer.test_network(params, valLoader, model, criterion, optimiser, verbose=False)

            checkpointer.save_checkpoint(model.state_dict(), optimiser.state_dict(), params)
            
            tqdm.write("{},\t{:10.5f},\t{:10.5f},\t{:10.5f},\t{:10.5f},\t{:10.5f},\t{:10.5f},\t{:10.5f},\t{:10.5f},\t{:10.5f},\t{:10.5f}".format(epoch, params.lr, params.train_loss, params.train_top1, params.train_top5, params.test_loss, params.test_top1, params.test_top5, params.val_loss, params.val_top1, params.val_top5))
    #}}} 
    
    def finetune_entropy(self, params, pruner, checkpointer, train_loader, test_loader, valLoader, model, criterion, optimiser, inferer):  
    #{{{
        print('Epoch,\tLR,\tTrain_Loss,\tTrain_Top1,\tTrain_Top5,\tTest_Loss,\tTest_Top1,\tTest_Top5,\tVal_Loss,\tVal_Top1,\tVal_Top5')
        
        for epoch in tqdm(range(params.start_epoch, params.finetuneBudget), desc='training', leave=False) : 
            params.curr_epoch = epoch
            state = self.update_lr(params, optimiser)
            
            losses = utils.AverageMeter()
            top1 = utils.AverageMeter()
            top5 = utils.AverageMeter()

            self.batch_iter(model, criterion, optimiser, train_loader, params, losses, top1, top5)

            params.train_loss = losses.avg        
            params.train_top1 = top1.avg        
            params.train_top5 = top5.avg        
            
            # get test loss
            params.test_loss, params.test_top1, params.test_top5 = inferer.test_network(params, test_loader, model, criterion, optimiser, verbose=False)
            params.val_loss, params.val_top1, params.val_top5 = inferer.test_network(params, valLoader, model, criterion, optimiser, verbose=False)

            checkpointer.save_checkpoint(model.state_dict(), optimiser.state_dict(), params)
            
            tqdm.write("{},\t{:10.5f},\t{:10.5f},\t{:10.5f},\t{:10.5f},\t{:10.5f},\t{:10.5f},\t{:10.5f},\t{:10.5f},\t{:10.5f},\t{:10.5f}".format(epoch, params.lr, params.train_loss, params.train_top1, params.train_top5, params.test_loss, params.test_top1, params.test_top5, params.val_loss, params.val_top1, params.val_top5))
    #}}} 
    
    def single_forward_backward(self, params, model, criterion, optimiser, train_loader): 
    #{{{
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            if params.use_cuda : 
                inputs, targets = inputs.cuda(), targets.cuda()
                
            loss, prec1, prec5 = self.train(model, criterion, optimiser, inputs, targets)
            
            return
    #}}}
    
