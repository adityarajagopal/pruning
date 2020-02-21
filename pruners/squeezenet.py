import sys
import csv
import os
import numpy as np
import time
from tqdm import tqdm
import json
import pickle
import subprocess
import importlib
import math
import copy

from src.ar4414.pruning.pruners.base import BasicPruning

import torch
import torch.nn as nn

class SqueezeNetPruning(BasicPruning):
#{{{
    def __init__(self, params, model):
    #{{{
        self.fileName = 'squeezenet_{}.py'.format(int(params.pruningPerc))
        self.netName = 'SqueezeNet'

        # selects only convs and fc layers 
        # used in get_layer_params to get sizes of only convs and fcs 
        self.convs_and_fcs = lambda lName : True if ('conv' in lName and 'weight' in lName) else False
        
        # function that specifies conv layers to skip when pruning
        # used in structure_l1_weight
        self.layerSkip = lambda lName : True if 'conv2' in lName and 'fire' not in lName else False

        super().__init__(params, model)
    #}}}
#}}}

