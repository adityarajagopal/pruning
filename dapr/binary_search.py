import gc
import sys
import time
import math

import torch
import torch.autograd

from tqdm import tqdm
from pytorch_memlab import profile 

import src.utils as utils
import src.training as trainingSrc

class BinarySearch(trainingSrc.Trainer):
    def __init__(self, params):
        super().__init__()
    
    def update_lr(self, params, optimiser) : 
    #{{{
        assert params.lr_schedule != [], "lr_schedule in config file cannot be blank for binary search"
        
        # get epochs to change at and lr at each of those changes
        # ::2 gets every other element starting at 0 
        changeEpochs = params.lr_schedule[::2]
        newLrs = params.lr_schedule[1::2]
        epoch = params.curr_epoch

        # effectiveEpoch = epoch if (epoch < params.pruneAfter) else (((epoch - params.pruneAfter) % params.finetuneBudget) + params.pruneAfter)
        effectiveEpoch = epoch if (epoch < params.pruneAfter) else (((epoch - params.pruneAfter) % params.finetuneBudget) + params.pruneAfter)

        if effectiveEpoch in changeEpochs:
            new_lr = newLrs[changeEpochs.index(effectiveEpoch)]
            if new_lr == -1 :
                params.lr *= params.gamma
            else : 
                params.lr = new_lr
         
        for param_group in optimiser.param_groups : 
            param_group['lr'] = params.lr
    
        return params
    #}}}

    def train(self, model, criterion, optimiser, inputs, targets) : 
    #{{{
        outputs = model(inputs) 
        loss = criterion(outputs, targets)

        prec1, prec5 = utils.accuracy(outputs.data, targets.data) 
    
        optimiser.zero_grad() 
        loss.backward() 
        optimiser.step()

        return (loss.item(), prec1.item(), prec5.item())
    #}}} 
    
    def batch_iter(self, model, criterion, optimiser, train_loader, params, losses, top1, top5):
    #{{{
        model.train()
        for batch_idx, (inputs, targets) in tqdm(enumerate(train_loader), total=len(train_loader)-1, desc='epoch', leave=False): 
            # move inputs and targets to GPU
            device = 'cuda:'+str(params.gpuList[0])
            if params.use_cuda: 
                inputs, targets = inputs.cuda(device, non_blocking=True), targets.cuda(device, non_blocking=True)
            
            # train model
            loss, prec1, prec5 = self.train(model, criterion, optimiser, inputs, targets)
            
            losses.update(loss) 
            top1.update(prec1) 
            top5.update(prec5)

            if params.batchLim != -1 and batch_idx == params.batchLim:
                return
    #}}}
    
    def perform_search(self, app):  
    #{{{
        # def check_stopping(mode, state, prevPp, currPp):
        def check_stopping():
        #{{{
            if prevPp == currPp:
                return True
            # if mode == 'memory_opt':
            #     if prevPp == currPp:
            #         return True
            # elif mode == 'cost_opt':
            #     if state == 1:
            #         return True
        
            return False
        #}}}
        
        # get unpruned test accuracy 
        loss, top1, top5 = app.run_inference()
        targetAcc = top1
        print('Epoch,\tLR,\tTrain_Loss,\tTrain_Top1,\tTrain_Top5,\tTest_Loss,\tTest_Top1,\tTest_Top5,\tVal_Loss,\tVal_Top1,\tVal_Top5')
        
        # perform finetuning once
        for epoch in tqdm(range(app.params.start_epoch, app.params.pruneAfter), desc='finetuning', leave=False) : 
        #{{{
            app.params.curr_epoch = epoch
            state = self.update_lr(app.params, app.optimiser)
            
            losses = utils.AverageMeter()
            top1 = utils.AverageMeter()
            top5 = utils.AverageMeter()

            self.batch_iter(app.model, app.criterion, app.optimiser, app.train_loader, app.params, losses, top1, top5)

            app.params.train_loss = losses.avg        
            app.params.train_top1 = top1.avg        
            app.params.train_top5 = top5.avg        
            
            # get test loss
            app.params.test_loss, app.params.test_top1, app.params.test_top5 = app.inferer.test_network(app.params, app.test_loader, app.model, app.criterion, app.optimiser, verbose=False)
            app.params.val_loss, app.params.val_top1, app.params.val_top5 = app.inferer.test_network(app.params, app.valLoader, app.model, app.criterion, app.optimiser, verbose=False)

            app.checkpointer.save_checkpoint(app.model.state_dict(), app.optimiser.state_dict(), app.params)
            
            tqdm.write("{},\t{:10.5f},\t{:10.5f},\t{:10.5f},\t{:10.5f},\t{:10.5f},\t{:10.5f},\t{:10.5f},\t{:10.5f},\t{:10.5f},\t{:10.5f}".format(epoch, app.params.lr, app.params.train_loss, app.params.train_top1, app.params.train_top5, app.params.test_loss, app.params.test_top1, app.params.test_top5, app.params.val_loss, app.params.val_top1, app.params.val_top5))
        #}}}

        # store model to revert to 
        finetunedModel = app.model
        app.checkpointer.save_model_only(app.model.state_dict(), app.params.printOnly, 'pre_pruning')
        app.params.curr_epoch += 1
            
        # initialise search
        app.params.start_epoch = app.params.pruneAfter
        initPp = 50
        prevPp = 0
        currPp = initPp
        uB = 95
        lB = 5 
        bestPp = 0
        state = 0
        bestTestAcc = targetAcc

        # while not check_stopping(mode, state, prevPp, currPp):  
        while not check_stopping(): 
            # prune model 
            app.params.pruningPerc = currPp
            app.model = finetunedModel 
            app.setup_pruners()
            tqdm.write('Pruning Network')
            channelsPruned, prunedModel, app.optimiser = app.pruner.prune_model(finetunedModel)
            totalPrunedPerc, _, _ = app.pruner.prune_rate(prunedModel)
            tqdm.write('Pruned Percentage = {:.2f}%'.format(totalPrunedPerc))
            summary = app.pruner.log_pruned_channels(app.checkpointer.root, app.params, totalPrunedPerc, channelsPruned)
            
            # perform retraining
            testAccs = [1]
            validAccs = [1]
            for epoch in tqdm(range(app.params.curr_epoch, app.params.curr_epoch + app.params.finetuneBudget), desc='training', leave=False) : 
            #{{{
                app.params.curr_epoch = epoch
                state = self.update_lr(app.params, app.optimiser)

                losses = utils.AverageMeter()
                top1 = utils.AverageMeter()
                top5 = utils.AverageMeter()

                self.batch_iter(prunedModel, app.criterion, app.optimiser, app.train_loader, app.params, losses, top1, top5)

                app.params.train_loss = losses.avg        
                app.params.train_top1 = top1.avg        
                app.params.train_top5 = top5.avg        

                # get test loss
                app.params.test_loss, app.params.test_top1, app.params.test_top5 = app.inferer.test_network(app.params, app.test_loader, prunedModel, app.criterion, app.optimiser, verbose=False)
                app.params.val_loss, app.params.val_top1, app.params.val_top5 = app.inferer.test_network(app.params, app.valLoader, prunedModel, app.criterion, app.optimiser, verbose=False)

                testAccs.append(app.params.test_top1)
                validAccs.append(app.params.val_top1)

                app.checkpointer.save_checkpoint(prunedModel.state_dict(), app.optimiser.state_dict(), app.params)
                
                tqdm.write("{},\t{:10.5f},\t{:10.5f},\t{:10.5f},\t{:10.5f},\t{:10.5f},\t{:10.5f},\t{:10.5f},\t{:10.5f},\t{:10.5f},\t{:10.5f}".format(epoch, app.params.lr, app.params.train_loss, app.params.train_top1, app.params.train_top5, app.params.test_loss, app.params.test_top1, app.params.test_top5, app.params.val_loss, app.params.val_top1, app.params.val_top5))
            #}}}

            highestTestAcc = testAccs[validAccs.index(max(validAccs))]
            if int(highestTestAcc) < int(targetAcc): 
                state = -1
            else:
                state = 1
            
            # prune less
            if state == -1: 
                tmp = (lB + currPp) / 2.
                uB = currPp 

            # try to prune more, but return previous model if state goes to -1
            elif state == 1:
                tmp = (uB + currPp) / 2.
                lB = currPp 
                bestPp = currPp 
                bestTestAcc = highestTestAcc

            prevPp = currPp
            currPp = 5 * math.ceil(tmp/5)  
        
        return bestPp, bestTestAcc
    #}}} 
