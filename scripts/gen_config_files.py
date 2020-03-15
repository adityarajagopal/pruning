import os
import configparser as cp
import subprocess
import sys

# nets = ['alexnet', 'mobilenetv2', 'squeezenet', 'resnet']
# lrSchedules = [
#                 '0 0.0001 5 0.001 15 -1 25 -1',
#                 '0 0.001 5 0.01 15 -1 25 -1', 
#                 '0 0.0008 5 0.02 15 -1 25 -1',
#                 '0 0.001 5 0.01 15 -1 25 -1'
#               ]
nets = ['mobilenetv2']
lrSchedules = [
                '0 0.001 5 0.01 15 -1 25 -1'
              ]
# lrSchedules = [
#                 '0 0.001 10 -1 20 -1', 
#                 '0 0.01 10 -1 20 -1', 
#                 '0 0.02 10 -1 20 -1',
#                 '0 0.01 10 -1 20 -1'
#               ]
# pruningPercs = [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95]
pruningPercs = [60,65,70,75,80,85,90,95]
# subset = ['aquatic', 'subset1', 'indoors', 'natural', 'random1']
# sub_classes = [
#                 'aquatic_mammals fish',
#                 'large_man-made_outdoor_things large_natural_outdoor_scenes vehicles_1 vehicles_2 trees small_mammals people',
#                 'food_containers household_electrical_devices household_furniture',
#                 'flowers fruit_and_vegetables insects large_omnivores_and_herbivores medium_mammals non-insect_invertebrates small_mammals reptiles',
#                 'aquatic_mammals fish flowers fruit_and_vegetables household_furniture large_man-made_outdoor_things large_omnivores_and_herbivores medium_mammals non-insect_invertebrates people reptiles trees vehicles_2'
#               ]
subset = ['indoors', 'natural', 'random1']
sub_classes = [
                'food_containers household_electrical_devices household_furniture',
                'flowers fruit_and_vegetables insects large_omnivores_and_herbivores medium_mammals non-insect_invertebrates small_mammals reptiles',
                'aquatic_mammals fish flowers fruit_and_vegetables household_furniture large_man-made_outdoor_things large_omnivores_and_herbivores medium_mammals non-insect_invertebrates people reptiles trees vehicles_2'
              ]
config = cp.ConfigParser()

base = '/home/ar4414/pytorch_training/src/ar4414/pruning/'

testName = 'l1_prune'
configPath = os.path.join(base, 'configs', testName)
runFileBase = os.path.join(base, 'scripts', testName)
cpRoot = os.path.join(base, 'logs/{}/cifar100/{}/{}')

cmd = 'mkdir -p ' + configPath
subprocess.check_call(cmd, shell=True)
cmd = 'mkdir -p ' + runFileBase
subprocess.check_call(cmd, shell=True)

runFiles = []
for netCount, net in enumerate(nets):
    testCount = 0
    configFile = os.path.join(base, 'configs', str(net) + '.ini')
    config.read(configFile)
            
    repeats = 5
    # gpu = netCount % 3 
    gpu = 1
    runFile = os.path.join(runFileBase, 'run_{}.sh'.format(gpu))
    runFiles.append(runFile)
    
    config['training_hyperparameters']['print_only'] = "False"
    config['training_hyperparameters']['lr_schedule'] = lrSchedules[netCount] 
    
    config['pytorch_parameters']['gpu_id'] = str(gpu)
    config['pytorch_parameters']['resume'] = "False"
    config['pytorch_parameters']['branch'] = "False"
    config['pytorch_parameters']['evaluate'] = "False"
    
    config['pruning_hyperparameters']['get_gops'] = "False"
    config['pruning_hyperparameters']['inference_gops'] = "False"
    
    config['pruning_hyperparameters']['prune_filters'] = "True"
    config['pruning_hyperparameters']['finetune'] = "True"
    config['pruning_hyperparameters']['static'] = "True"
    config['pruning_hyperparameters']['retrain'] = "False"
    config['pruning_hyperparameters']['finetune_budget'] = "30"
    config['pruning_hyperparameters']['prune_after'] = "5"

    for ssCount, ss in enumerate(subset):
        config['pruning_hyperparameters']['sub_name'] = ss
        config['pruning_hyperparameters']['sub_classes'] = sub_classes[ssCount]
        config['pytorch_parameters']['checkpoint_path'] = cpRoot.format(net, ss, testName) 

        for ppCount, pp in enumerate(pruningPercs):
            config['pruning_hyperparameters']['pruning_perc'] = str(pp)
            config['pytorch_parameters']['test_name'] = 'pp_{}'.format(pp)

            testConfig = os.path.join(configPath, str(net) + '_' + str(testCount) + '.ini')
            with open(testConfig, 'w+') as tcFile:
                config.write(tcFile)
            
            with open(runFile, 'a+') as rFile:
                for i in range(repeats):
                    rFile.write('python main.py --config-file ' + testConfig + '\n')
            
            testCount += 1

[os.chmod(rf,0o755) for rf in runFiles]

