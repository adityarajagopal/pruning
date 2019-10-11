import os 
import subprocess
import sys
import torch
import csv

import time
import datetime

import src.checkpointing as cpSrc

class Checkpointer(cpSrc.Checkpointer) : 
    def get_root(self):
        return self.root

    def setup_values(self, params): 
        self.values = [params.curr_epoch, params.lr, params.train_loss.item(), \
                       params.train_top1.item(), params.train_top5.item(), params.test_loss, \
                       params.test_top1, params.test_top5, params.val_loss, \
                       params.val_top1, params.val_top5]

    def restore_state(self, params): 
        # get state to load from
        if params.resume == True or params.branch == True or params.getGops == True or params.entropy == True : 
            file_to_load = params.pretrained.replace('model', 'state')        
            device = 'cuda:' + str(params.gpu_id)
            prev_state_dict = torch.load(file_to_load, map_location=device)
        
        # if resume, load from old state completely, ignore parameters in config file
        if params.resume == True : 
            # ensure path to pretrained has new path and new state know it is in resume 
            prev_state_dict['pretrained'] = params.pretrained
            prev_state_dict['resume'] = True
            prev_state_dict['gpu_id'] = params.gpu_id
            prev_state_dict['workers'] = params.workers
            
            params.__dict__.update(**prev_state_dict)

            # update new start epoch as epoch after the epoch that was resumed from
            params.start_epoch = prev_state_dict['curr_epoch'] + 1

        # if there's a branch copy the save state files to new branch folder
        # ignore previous state and use parameters in config file directly
        elif params.branch == True:
            # copy epoch checkpoint from root of branch
            prev_epoch = str(prev_state_dict['curr_epoch'])
            old_root_list = params.pretrained.split('/')
            old_root = os.path.join('/', *old_root_list[:-1])
            
            self.__create_dir(self.root)
            self.__create_copy_log(self.root, old_root, prev_epoch)
            
            cmd = 'cp ' + os.path.join(old_root, prev_epoch + '-*') + ' ' + self.root           
            subprocess.check_call(cmd, shell=True)

            params.start_epoch = prev_state_dict['curr_epoch'] + 1

        # if evaluate, use state as specified in config file
        elif params.evaluate == True : 
            pass 

        elif params.getGops == True:
            params.arch = prev_state_dict['arch']
            params.pruneWeights = prev_state_dict['prune_weights']
            params.pruneFilters = prev_state_dict['prune_filters']
            # params.__dict__.update(**prev_state_dict)
        
        # if all false, start from epoch 0 and use config file 
        else : 
            params.start_epoch = 0                            

        return params

