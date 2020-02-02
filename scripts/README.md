# Scripts 
The various scripts here can be used to quickly generate config files and bash scripts to run multiple different tests on various networks and subsets
The parameters common to most of the script generation files are as follows: 
- nets: networks to generate configs for
- pruningPercs: pruning percentages to cycle through
- subset: name of subsets 
- sub_classes: classes within the corresponding subsets index in subset wise
- configPath: folder inside configs where the new config files should be placed 
- runFileBase: location where bash scripts with call to each test should be placed  
- cpRoot: if logs are going to be created by the runs then this is the path to the folder where the checkpoint needs to be placed

## gen_config_files.py
* generates basic finetune + pruning + retrain files or any of the other  
* generates a test per network, subset, and pruning percentage 
* can change the gpus to run on (generates different bash script per gpu) 
* can change the number of repeats of a test

## gen_gops_collection_configs.py
* generates a test per network and subset combination
* for each combination it gerneates one config to collect inference gops and another to collect training gops

## gen_pruned_acc_configs.py + gen_unpruned_acc_configs.py 
* generates a test per network and subset combination 
* need to set logs parameter to correct log file
* tests store inference on subset of a pruned and unpruned model that was retrained on the entire_dataset after pruning
