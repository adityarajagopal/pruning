import os
import configparser as cp
import subprocess
import sys

nets = ['resnet', 'mobilenetv2', 'alexnet', 'squeezenet']
pruningPercs = [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95]
subset = ['subset1', 'aquatic']
sub_classes = ["large_man-made_outdoor_things large_natural_outdoor_scenes vehicles_1 vehicles_2 trees small_mammals people", "aquatic_mammals fish", ""]
batchSize = []
ftBudget = []
lrSchedule = []
configPath = '/home/ar4414/pytorch_training/src/ar4414/pruning/configs/pruned_acc'
runFileBase = '/home/ar4414/pytorch_training/src/ar4414/pruning/scripts_conf_logs/'
config = cp.ConfigParser()

cmd = 'mkdir -p ' + configPath
subprocess.check_call(cmd, shell=True)

for netCount, net in enumerate(nets):
    testCount = 0
    configFile = '/home/ar4414/pytorch_training/src/ar4414/pruning/configs/' + str(net) + '.ini'
    config.read(configFile)
            
    config['training_hyperparameters']['print_only'] = "True"
    
    config['pytorch_parameters']['gpu_id'] = "0"
    config['pytorch_parameters']['resume'] = "False"
    config['pytorch_parameters']['branch'] = "False"
    config['pytorch_parameters']['evaluate'] = "False"
    
    config['pruning_hyperparameters']['logs'] = "/home/ar4414/pytorch_training/src/ar4414/pruning/logs/subset_agnostic_logs.json"
    
    config['pruning_hyperparameters']['get_gops'] = "False"
    config['pruning_hyperparameters']['inference_gops'] = "False"
    
    config['pruning_hyperparameters']['unpruned_test_acc'] = "False"
    
    config['pruning_hyperparameters']['pruned_test_acc'] = "True"
    config['pruning_hyperparameters']['trained_on'] = "entire_dataset"
    
    config['pruning_hyperparameters']['no_finetune_channels_pruned'] = "False"
    
    config['pruning_hyperparameters']['prune_filters'] = "False"
    config['pruning_hyperparameters']['finetune'] = "False"
    config['pruning_hyperparameters']['static'] = "False"
    config['pruning_hyperparameters']['retrain'] = "False"
    config['pruning_hyperparameters']['finetune_budget'] = "30"
    config['pruning_hyperparameters']['prune_after'] = "5"

    for ssCount, ss in enumerate(subset):
        config['pruning_hyperparameters']['sub_name'] = ss
        config['pruning_hyperparameters']['sub_classes'] = sub_classes[ssCount]
        
        runFileName = 'run_pruned_acc.sh'
        runFile = os.path.join(runFileBase, runFileName)

        testConfig = os.path.join(configPath, str(net) + '_' + str(testCount) + '.ini')
        with open(testConfig, 'w+') as tcFile:
            config.write(tcFile)
        
        with open(runFile, 'a+') as rFile:
            rFile.write('python main.py --config-file ' + testConfig + '\n')
        
        testCount += 1





