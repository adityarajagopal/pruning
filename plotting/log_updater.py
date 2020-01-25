import os
import sys
import glob
import time
import datetime

def add_network(logs, networkName, datasets, baseFolder, preFtModel):
#{{{
    logs[networkName] = {dataset:{} for dataset in datasets}
    logs[networkName]['pre_ft_model'] = preFtModel
    for dataset in datasets:
        logs[networkName][dataset]['base_path'] = '/home/ar4414/pytorch_training/src/ar4414/pruning/logs/alexnet/cifar100/{}/{}'.format(dataset, baseFolder)

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
    
    day = timeStamp.split('-')[-1]
    day = "[{}-3][{}-9]".format(day[0], day[1])
    timeStamp = timeStamp.split('-')
    timeStamp[-1] = day
    timeStamp = '-'.join(timeStamp)

    for net in networks: 
        for dataset in datasets: 
            basePath = logs[net][dataset]['base_path']
            logFiles = logs[net][dataset]
            for pp in prunePercs:
                wildCard = "pp_{}/{}-*".format(pp, timeStamp)
                logList = glob.glob(os.path.join(basePath, wildCard))
                timeStamps = [x.split('/')[-1] for x in logList]
                if len(timeStamps) > 0:
                    if pp in logFiles.keys():
                        [logFiles[pp].append(ts) for ts in timeStamps if ts not in logFiles[pp]]
                    else:
                        logFiles[pp] = timeStamps
            
    return logs
#}}}

