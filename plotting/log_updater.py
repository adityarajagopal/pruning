import os
import sys
import glob
import time
import datetime

def add_network(logs, networkName, datasets, baseFolder, preFtModel):
#{{{
    logs[networkName] = {dataset:{} for dataset in datasets}
    if preFtModel is not None:
        logs[networkName]['pre_ft_model'] = preFtModel
    for dataset in datasets:
        logs[networkName][dataset]['base_path'] = '/home/ar4414/pytorch_training/src/ar4414/pruning/logs/{}/cifar100/{}/{}'.format(networkName, dataset, baseFolder)

    return logs
#}}}

def add_dataset(logs, datasetName, baseFolder):  
#{{{
    for net, value in logs.items(): 
        value.update({datasetName:{'base_path': '/home/ar4414/pytorch_training/src/ar4414/pruning/logs/{}/cifar100/{}/{}'.format(net, datasetName, baseFolder)}})

    return logs
#}}}

def update_timestamps(logs, networks, datasets, prunePercs, asOf=None):
#{{{
    if asOf is None:
        print("Getting logs generated as of today onwards")
        ts = time.time()
        timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    else: 
        print("Getting logs generated as of {}".format(asOf))
        timeStamp = asOf

    date = "{}-00-00-00".format(timeStamp)
    for net in networks: 
        for dataset in datasets: 
            basePath = logs[net][dataset]['base_path']
            logFiles = logs[net][dataset]
            for pp in prunePercs:
                try:
                    timeStamps = os.listdir(os.path.join(basePath,"pp_{}".format(pp)))  
                except FileNotFoundError:
                    continue
                timeStamps = list(filter(lambda x: x >= date, timeStamps))
                if len(timeStamps) > 0:
                    if pp in logFiles.keys():
                        [logFiles[pp].append(ts) for ts in timeStamps if ts not in logFiles[pp]]
                    else:
                        logFiles[pp] = timeStamps
            
    return logs
#}}}

