import os
import configparser as cp
import subprocess
import sys

nets = ['resnet', 'mobilenetv2', 'alexnet', 'squeezenet']
pruningPercs = [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95]
subset = ['subset1', 'aquatic']
sub_classes = ["large_man-made_outdoor_things large_natural_outdoor_scenes vehicles_1 vehicles_2 trees small_mammals people", "aquatic_mammals fish"]
config = cp.ConfigParser()

base = '/home/ar4414/pytorch_training/src/ar4414/pruning/'

configPath = os.path.join(base, 'configs', 'no_ft_channels_pruned')
runFileBase = os.path.join(base, 'scripts', 'no_ft_channels_pruned')

cmd = 'mkdir -p ' + configPath
subprocess.check_call(cmd, shell=True)
cmd = 'mkdir -p ' + runFileBase
subprocess.check_call(cmd, shell=True)

for netCount, net in enumerate(nets):
    testCount = 0
    configFile = os.path.join(base, 'configs', str(net) + '.ini')
    config.read(configFile)
            
    repeats = 1
    gpu = "0"
    runFile = os.path.join(runFileBase, 'run_{}.sh'.format(gpu))
    
    config['training_hyperparameters']['print_only'] = "True"
    
    config['pytorch_parameters']['gpu_id'] = gpu
    config['pytorch_parameters']['resume'] = "False"
    config['pytorch_parameters']['branch'] = "False"
    config['pytorch_parameters']['evaluate'] = "False"
    
    config['pruning_hyperparameters']['get_gops'] = "False"
    config['pruning_hyperparameters']['inference_gops'] = "False"
    
    config['pruning_hyperparameters']['logs'] = "/home/ar4414/pytorch_training/src/ar4414/pruning/logs/logs.json"
    config['pruning_hyperparameters']['no_finetune_channels_pruned'] = "True"
    
    config['pruning_hyperparameters']['prune_filters'] = "False"
    config['pruning_hyperparameters']['finetune'] = "False"
    config['pruning_hyperparameters']['static'] = "False"
    config['pruning_hyperparameters']['retrain'] = "False"
    config['pruning_hyperparameters']['finetune_budget'] = "30"
    config['pruning_hyperparameters']['prune_after'] = "5"

    config['pruning_hyperparameters']['sub_name'] = 'aquatic'
    config['pruning_hyperparameters']['sub_classes'] = sub_classes[1]

    testConfig = os.path.join(configPath, str(net) + '_' + str(testCount) + '.ini')
    with open(testConfig, 'w+') as tcFile:
        config.write(tcFile)
    
    with open(runFile, 'a+') as rFile:
        for i in range(repeats):
            rFile.write('python main.py --config-file ' + testConfig + '\n')
    
    testCount += 1

os.chmod(runFile, 0o755)




