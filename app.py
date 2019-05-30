import src.ar4414.pruning.hooks as hookSrc
import src.ar4414.pruning.param_parser as ppSrc 
import src.ar4414.pruning.model_creator as mcSrc

import src.app as appSrc
import src.checkpointing as checkpointingSrc
import src.input_preprocessor as preprocSrc
import src.training as trainingSrc
import src.inference as inferenceSrc

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

        if self.params.getGops:
            # register hooks
            self.hooks = hookSrc.Hook(self.model)
            self.hooks.register_hooks()
        
        if self.params.getGops:
            self.run_inference()
        elif self.params.evaluate == False : 
            self.run_training()
        else : 
            self.run_inference()

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
    
