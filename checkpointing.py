import os 
import subprocess
import sys
import torch
import csv

import time
import datetime

import src.checkpointing as cpSrc

class Checkpointer(cpSrc.Checkpointer) : 

    def restore_state(self, params): 
        # get state to load from
        if params.resume == True or params.branch == True or params.getGops == True : 
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

        if params.getGops == True:
            params.__dict__.update(**prev_state_dict)

        # if all false, start from epoch 0 and use config file 
        else : 
            params.start_epoch = 0                            

        return params

    def log_prune_rate(self, params, totalPrunedPerc): 
        if params.printOnly == True:
            return 
        csvName = os.path.join(self.root, 'layer_prune_rate.csv')
        with open(csvName, 'a') as csvFile:
            writer = csv.writer(csvFile, delimiter=',')
            writer.writerow([params.curr_epoch] + params.prunePercPerLayer + [totalPrunedPerc])
        
