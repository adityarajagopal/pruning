import sys
import time
from tqdm import tqdm
import numpy as np

import torch.autograd
import torch

import src.utils as utils
import src.training as trainingSrc

class Trainer(trainingSrc.Trainer):
    def __init__(self, params):
        super().__init__()
        self.fbsPruning = params.fbsPruning

    def train(self, model, criterion, optimiser, inputs, targets) : 
    #{{{
        model.train()

        outputs = model(inputs) 
        loss = criterion(outputs, targets)

        if self.fbsPruning == True:
            for x in model.named_buffers():
                if 'g_x' in x[0]:
                    loss += 1e-8 * x[1]
        
        prec1, prec5 = utils.accuracy(outputs.data, targets.data) 
    
        model.zero_grad() 
        loss.backward() 

        optimiser.step()

        return (loss, prec1, prec5)
    #}}} 
    
    def batch_iter(self, model, criterion, optimiser, train_loader, params, losses, top1, top5):
    #{{{
        for batch_idx, (inputs, targets) in tqdm(enumerate(train_loader), total=len(train_loader)-1, desc='epoch', leave=False): 
            # move inputs and targets to GPU
            if params.use_cuda : 
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
            
            # train model
            loss, prec1, prec5 = self.train(model, criterion, optimiser, inputs, targets)
            
            losses.update(loss) 
            top1.update(prec1) 
            top5.update(prec5)

            if params.batchLim != -1 and batch_idx == params.batchLim:
                return
    #}}}
    
    def finetune_l2_weights(self, params, pruner, checkpointer, train_loader, test_loader, valLoader, model, criterion, optimiser, inferer):  
    #{{{
        print('Epoch,\tLR,\tTrain_Loss,\tTrain_Top1,\tTrain_Top5,\tTest_Loss,\tTest_Top1,\tTest_Top5,\tVal_Loss,\tVal_Top1,\tVal_Top5')
        
        for epoch in tqdm(range(params.start_epoch, params.finetuneBudget), desc='training', leave=False) : 
            params.curr_epoch = epoch
            state = self.update_lr(params, optimiser)
            
            # perform pruning 
            if params.pruneFilters == True and epoch == params.pruneAfter: 
                tqdm.write('Pruning Network')
                channelsPruned = pruner.prune_model(model)
                totalPrunedPerc = pruner.prune_rate(model, True)
                tqdm.write('Pruned Percentage = {}'.format(totalPrunedPerc))
                summary = pruner.log_pruned_channels(checkpointer.root, params, totalPrunedPerc, channelsPruned)
                
                pruner.write_net()
                import src.ar4414.pruning.models.cifar.pruned as pModel
                # prunedModel = pModel.__dict__[str(params.arch) + '_' + str(int(params.pruningPerc))](num_classes = 100)
                prunedModel = pModel.__dict__[str(params.arch)](num_classes = 100)
                gpu_list = [int(x) for x in params.gpu_id.split(',')]
                prunedModel = torch.nn.DataParallel(prunedModel, gpu_list).cuda()
                model = pruner.transfer_weights(model, prunedModel)
                optimiser = torch.optim.SGD(model.parameters(), lr=params.lr, momentum=params.momentum, weight_decay=params.weight_decay)
    
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
    
    #{{{
    # def finetune_newtork(self, params, pruner, checkpointer, train_loader, test_loader, valLoader, model, criterion, optimiser, inferer):  
    #     print('Epoch,\tLR,\tTrain_Loss,\tTrain_Top1,\tTrain_Top5,\tTest_Loss,\tTest_Top1,\tTest_Top5,\tVal_Loss,\tVal_Top1,\tVal_Top5')
    #     
    #     for epoch in tqdm(range(params.start_epoch, params.epochs), desc='training', leave=False) : 
    #         params.curr_epoch = epoch
    #         state = self.update_lr(params, optimiser)
    # 
    #         losses = utils.AverageMeter()
    #         top1 = utils.AverageMeter()
    #         top5 = utils.AverageMeter()

    #         self.batch_iter(model, criterion, optimiser, train_loader, params, losses, top1, top5)

    #         params.train_loss = losses.avg        
    #         params.train_top1 = top1.avg        
    #         params.train_top5 = top5.avg        
    #         
    #         # perform pruning 
    #         # if (params.pruneWeights == True or params.pruneFilters == True) and ((epoch+1) % params.pruneAfter == 0): 
    #         if (params.pruneWeights == True or params.pruneFilters == True) and (epoch = params.finetuneBudget): 
    #             tqdm.write('Pruning Network')
    #             model = pruner.prune_model(model)
    #             params.pruningPerc += params.prunePercIncrement
    #             totalPrunedPerc = pruner.prune_rate(model, True)
    #             tqdm.write('Pruned Percentage = {}'.format(totalPrunedPerc))
    #             # checkpointer.log_prune_rate(params, totalPrunedPerc)
    #             pruner.log_prune_rate(checkpointer.root, params, totalPrunedPerc)
    #             params.prunePercPerLayer = []

    #         if params.fbsPruning == True and ((epoch + 1) % params.pruneAfter == 0):
    #             params.unprunedRatio -= (params.prunePercIncrement / 100.0)
    #             # return if pruning become lower than lower bound
    #             if params.unprunedRatio <= params.unprunedLB: 
    #                 return
    #             
    #             tqdm.write('Pruning Network with FBS')
    #             model = pruner.prune_model(model)
    #             tqdm.write('Pruned Percentage = {}'.format(1.0 - params.unprunedRatio))
    # 
    #         # get test loss
    #         params.test_loss, params.test_top1, params.test_top5 = inferer.test_network(params, test_loader, model, criterion, optimiser)
    #         params.val_loss, params.val_top1, params.val_top5 = inferer.test_network(params, valLoader, model, criterion, optimiser)

    #         checkpointer.save_checkpoint(model.state_dict(), optimiser.state_dict(), params)
    #         
    #         tqdm.write("{},\t{:10.5f},\t{:10.5f},\t{:10.5f},\t{:10.5f},\t{:10.5f},\t{:10.5f},\t{:10.5f},\t{:10.5f},\t{:10.5f},\t{:10.5f}".format(epoch, params.lr, params.train_loss, params.train_top1, params.train_top5, params.test_loss, params.test_top1, params.test_top5, params.val_loss, params.val_top1, params.val_top5))
    #}}}
