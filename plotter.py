import sys
import matplotlib.pyplot as plt
import os
import json
import torch.nn as nn
import numpy as np
import subprocess

class ChannelPlotter(object):
    def __init__(self, params, model):
    #{{{
        self.params = params
        self.model = model
        if 'resnet' in self.params.arch:
            net = str(self.params.arch) + str(self.params.depth)
        else:
            net = str(self.params.arch)
        self.subsets = ['entire_dataset', 'subset1', 'aquatic']
        self.logDir = '/home/ar4414/pytorch_training/src/ar4414/pruning/logs/' + net +'/cifar100/{}/l1_prune'
    #}}}
        
    def plot_channels(self):
    #{{{
        self.pivotDict = {}
        for i, logFile in enumerate(self.params.plotChannels):
            logDir = os.path.join(self.logDir.format(self.subsets[i%3]), logFile)
            
            try:
                with open(os.path.join(logDir, 'pruned_channels.json'), 'r') as cpFile:
                    channelsPruned = json.load(cpFile)
            except FileNotFoundError:
                print("File : {} does not exist.".format(os.path.join(logDir, 'pruned_channels.json')))
                print("Either the log directory is wrong or run finetuning without GetGops to generate file before running this command.")
                sys.exit()
            
            pruneEpoch = int(list(channelsPruned.keys())[0])
            channelsPruned = list(channelsPruned.values())[0]
            prunePerc = channelsPruned.pop('prunePerc')
            allChannelsByLayer = {l:np.zeros(m.out_channels,dtype=int) for l,m in self.model.named_modules() if isinstance(m, nn.Conv2d)}

            if prunePerc == 0.:
                continue
            
            for j,(l,c) in enumerate(allChannelsByLayer.items()):
                c[channelsPruned[l]] = 1                              

            self.layerNames = ['.'.join(x.split('.')[1:]) for x in list(allChannelsByLayer.keys())]
            self.numChannelsPerLayer = [len(v) for v in allChannelsByLayer.values()]

            self.pivotDict[self.subsets[i%3]] = {'grad':allChannelsByLayer, 'pp':prunePerc}
            
            if (i+1)%3 == 0:
                self.plot_channel_comparisons()
                self.pivotDict = {}
    #}}}        

    def plot_channel_comparisons(self):
    #{{{
        fig, ax = plt.subplots()
        bar = ax.barh(self.layerNames, self.numChannelsPerLayer)
        
        ax = bar[0].axes
        lim = ax.get_xlim() + ax.get_ylim()
        
        for i, bar in enumerate(bar):
            layer = 'module.' + self.layerNames[i]
            
            grads = []
            for k,v in self.pivotDict.items():
                grads.append(v['grad'][layer])
                prunePerc = v['pp']
            
            if (np.array_equal(grads[0], grads[1])) and (np.array_equal(grads[1], grads[2])): 
                grad = np.array(grads[0])
                grad = np.expand_dims(grad, 1).T
                bar.set_zorder(1)
                bar.set_facecolor("none")
                x,y = bar.get_xy()
                w,h = bar.get_width(), bar.get_height()
                ax.imshow(grad, extent=[x,x+w,y,y+h], aspect="auto", zorder=0)
            else:
                inc = bar.get_height() / 3.
                cmaps = ['viridis', 'winter', 'rainbow']
                for i in range(3):
                    grad = np.array(grads[i])
                    grad = np.expand_dims(grad, 1).T
                    bar.set_zorder(1)
                    bar.set_facecolor("none")
                    x,y = bar.get_xy()
                    w,h = bar.get_width(), bar.get_height()
                    ax.imshow(grad, extent=[x,x+w,y+(i*inc),y+((i+1)*inc)], aspect="auto", cmap=cmaps[i], zorder=0)
        
        ax.axis(lim)
        ax.set_title('Channels Pruned By Layer for {:.2f}% pruning'.format(prunePerc))
        plt.tight_layout()
        
        folder = '/home/ar4414/remote_copy/channels/'
        fig = os.path.join(folder, 'p_{}.png'.format(int(prunePerc)))
        cmd = 'mkdir -p {}'.format(folder)
        subprocess.check_call(cmd, shell=True)
        print('Saving - {}'.format(fig))
        plt.savefig(fig, format='png') 
    #}}} 
