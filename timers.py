import os
import sys
import pdb
import time
import subprocess
from tqdm import tqdm
from contextlib import contextmanager

import torch

@contextmanager
def timer(description): 
    start = time.perf_counter()
    yield
    end = time.perf_counter() 
    tqdm.write("[{}] : {}s".format(description, end-start))

class Timer(object): 
    enabled = False
    
    def __init__(self, description):
        self.desc = description 
        self.timestep = []
        self.stats = {}

    def __enter__(self): 
        if self.enabled:
            self.start = time.perf_counter() 

    def __exit__(self, type, value, traceback): 
        if self.enabled:
            self.end = time.perf_counter()
            self.timestep.append(self.end - self.start)
    
    @staticmethod
    def log_dict(logDir, dataDict): 
    #{{{
        completePath = "{}/{}".format(os.getcwd(), logDir)
        cmd = "mkdir -p {}".format(completePath)         
        subprocess.check_call(cmd, shell=True)
        print("Saving timing logs to : {}".format(completePath))
        torch.save(dataDict, "{}/timing_data.pth.tar".format(completePath))
    #}}}
    
    def reset(self): 
        if self.enabled: 
            self.timestep = []

    def update_stats(self, key, value): 
        if self.enabled:
            self.stats[key] = value
    
    def acc_stats(self, key, value): 
        if self.enabled:
            if key in self.stats.keys(): 
                self.stats[key] += value 
            else:
                self.stats[key] = value
