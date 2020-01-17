import sys
import torch
import matplotlib.pyplot as plt
import os
import json
import torch.nn as nn
import numpy as np
import subprocess
import itertools
from scipy.spatial import distance
import pandas as pd

nets = ['alexnet', 'resnet', 'mobilenetv2', 'squeezenet']
subset = 'entire_dataset'

data = {}
for net in nets:
    channelsPruned = torch.load('prunedChannels/{}.pth.tar'.format(net))
    indices = channelsPruned['pp']
    data[net] = channelsPruned['%-diff'] 

title = 'Percent difference in channels pruned globally by percentage of network pruned \n ({})'.format(subset)
xlab = "Percentage of network pruned"
ylab = 'Percentage of channels pruned differently'

df = pd.DataFrame(data, index=indices)
ax = df.plot.bar(title = title)
ax.set_xlabel(xlab)
ax.set_ylabel(ylab)
ax.set_xticklabels(indices, rotation=45, ha='right')
plt.tight_layout()
plt.show() 

