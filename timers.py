import sys
import time
from tqdm import tqdm
from contextlib import contextmanager

@contextmanager
def timer(description): 
    start = time.perf_counter()
    yield
    end = time.perf_counter() 
    tqdm.write("[{}] : {}s".format(description, end-start))

class Timer(object): 
    enabled = True
    
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
    
    def reset(self): 
        if self.enabled: 
            self.timestep = []

    def update_stats(self, key, value): 
        if self.enabled:
            self.stats[key] = value
