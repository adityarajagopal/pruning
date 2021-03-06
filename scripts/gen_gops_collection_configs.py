import os
import sys
import stat
import subprocess
import configparser as cp

nets = ['resnet', 'mobilenetv2', 'alexnet', 'squeezenet']
pruningPercs = [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95]
# subset = ['subset1', 'aquatic']
# sub_classes = ["large_man-made_outdoor_things large_natural_outdoor_scenes vehicles_1 vehicles_2 trees small_mammals people", "aquatic_mammals fish", ""]
subset = ['indoors', 'natural', 'random1']
sub_classes = ['food_containers household_electrical_devices household_furniture', 'flowers fruit_and_vegetables insects large_omnivores_and_herbivores medium_mammals non-insect_invertebrates small_mammals reptiles', 'aquatic_mammals fish flowers fruit_and_vegetables household_furniture large_man-made_outdoor_things large_omnivores_and_herbivores medium_mammals non-insect_invertebrates people reptiles trees vehicles_2']
# subset = ['entire_dataset']
# sub_classes = ['']
batchSize = []
ftBudget = []
lrSchedule = []

base = '/home/ar4414/pytorch_training/src/ar4414/pruning'
testName = 'gops_calculation'
configPath = os.path.join(base, 'configs', testName)
runFileBase = os.path.join(base, 'scripts', testName)
config = cp.ConfigParser()

cmd = 'mkdir -p ' + configPath
subprocess.check_call(cmd, shell=True)
cmd = 'mkdir -p ' + runFileBase 
subprocess.check_call(cmd, shell=True)

for netCount, net in enumerate(nets):
    testCount = 0
    configFile = os.path.join(base, 'configs', str(net) + '.ini')
    config.read(configFile)
            
    config['training_hyperparameters']['print_only'] = "True"
    
    config['pytorch_parameters']['gpu_id'] = "0"
    config['pytorch_parameters']['resume'] = "False"
    config['pytorch_parameters']['branch'] = "False"
    config['pytorch_parameters']['evaluate'] = "False"
    
    config['pruning_hyperparameters']['logs'] = "/home/ar4414/pytorch_training/src/ar4414/pruning/logs/logs.json"
    config['pruning_hyperparameters']['get_gops'] = "True"
    
    config['pruning_hyperparameters']['prune_filters'] = "False"
    config['pruning_hyperparameters']['finetune'] = "False"
    config['pruning_hyperparameters']['static'] = "False"
    config['pruning_hyperparameters']['retrain'] = "False"
    config['pruning_hyperparameters']['finetune_budget'] = "30"
    config['pruning_hyperparameters']['prune_after'] = "5"

    for ssCount, ss in enumerate(subset):
        config['pruning_hyperparameters']['sub_name'] = ss
        config['pruning_hyperparameters']['sub_classes'] = sub_classes[ssCount]
        
        for t in range(2): 
            config['pruning_hyperparameters']['inference_gops'] = "False" if t == 1 else "True"
            runFileName = 'run.sh'
            runFile = os.path.join(runFileBase, runFileName)

            testConfig = os.path.join(configPath, str(net) + '_' + str(testCount) + '.ini')
            with open(testConfig, 'w+') as tcFile:
                config.write(tcFile)
            
            with open(runFile, 'a+') as rFile:
                rFile.write('python main.py --config-file ' + testConfig + '\n')
            
            testCount += 1

os.chmod(runFile, 0o755)


