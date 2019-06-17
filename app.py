import src.ar4414.pruning.gop_calculator as gopSrc
import src.ar4414.pruning.param_parser as ppSrc 
import src.ar4414.pruning.model_creator as mcSrc
import src.ar4414.pruning.inference as inferenceSrc
import src.ar4414.pruning.checkpointing as checkpointingSrc
import src.ar4414.pruning.training as trainingSrc

import src.app as appSrc
# import src.checkpointing as checkpointingSrc
import src.input_preprocessor as preprocSrc
# import src.training as trainingSrc

import os
import random
import sys

import configparser as cp

import tensorboardX as tbx

import torch
import torch.cuda

class Application(appSrc.Application):
    def main(self):
        self.setup_dataset()
        self.setup_model()
        self.setup_tee_printing()
        
        # setup tensorboardX and checkpointer  
        self.tbx_writer = tbx.SummaryWriter(comment='-test-1')

        if self.params.getGops:
            if 'googlenet' in self.params.arch:
                # register hooks
                self.gopCalculator = gopSrc.GoogleNetGopCalculator(self.model, self.params)
                self.gopCalculator.register_hooks()
            else:
                raise ValueError('Gop calculation not implemented for specified architecture')
            
            self.run_gop_calc()
            print('Unpruned Gops = ', self.gopCalculator.baseTotalGops)
            print('Pruned Gops = ', self.gopCalculator.prunedTotalGops)
        
        elif self.params.finetune == True:
            self.run_finetune()
        
        elif self.params.evaluate == False : 
            setIndicesf.run_training()
        
        else : 
            self.run_inference()

        self.tbx_writer.close()

    def run_finetune(self):
        print('==> Performing Pruning Finetune')
        self.trainer.finetune_network(self.params, self.tbx_writer, self.checkpointer, self.train_loader, self.test_loader, self.valLoader, self.model, self.criterion, self.optimiser, self.inferer) 

    def setup_param_checkpoint(self, configFile):
        config = cp.ConfigParser() 
        config.read(configFile)
        self.params = ppSrc.Params(config)
        self.checkpointer = checkpointingSrc.Checkpointer(self.params, configFile)
        self.setup_params()
    
    def setup_others(self):
        self.preproc = preprocSrc.Preproc()
        self.mc = mcSrc.ModelCreator()
        self.trainer = trainingSrc.Trainer()
        self.inferer = inferenceSrc.Inferer()

    def run_gop_calc(self):
        self.inferer.run_single_forward(self.params, self.test_loader, self.model)
    
