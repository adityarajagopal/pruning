# Pruning Configurations 
As you go down the config file, within the pruning_hyperpamaters section, each true before will cancel out the following true or falses, i.e. priority is top down.
This README has details of the main pruning framework. Details on scripts available are in a README in the scripts folder and plotting options are in a README in the plotting folder. 

## Get GOps for a given design
- **Get\_Gops** : True
- **Logs** : path to json file with the time stamps of all the logs  
- **Inference\_Gops** : 
    * if True, inference gops stored in gops.json file inside folder with logs 
    * if False, training gops stored in gops.json file inside folder with logs
    * inference gops is gops per minibatch, training gops is minibatch independent as it stores gops per epoch

## Performance of subset on network trained on entire dataset
This performs inference on the network trained on the entire dataset, but with images in the subset only.
There is no retraining performed, and provides data on performance if original network is directly deployed.
- **Unpruned\_Test\_Acc**: True
    * stores into logs json the dict {'test_top1':#, 'gops':infernece\_gops} per network and subset

## Performance of subset on network pruned and then trained on entire dataset 
This looks at performance if pruning was performed in a data agnostic manner, i.e. the network is pruned at epoch 0, 
and retrained for the same epoch budget as the data aware case. 
For each pruning percentage, inference is performed on the various subsets and recorded in the logs.json file. 
The prequisite for this is that the above training must be performed and the json file updated with logs of training 
on the entire dataset. (gen_data_agnostic_l1_prune.py in the scripts folder generates scripts for this training) 
- **Pruned\_Test\_Acc** : True
- **Trained\_on** : 
    * default 'entire\_dataset' works at the moment 
    * this parameter specifies the pruned models have been retrained after pruning on what subset/dataset
- **Logs** : path to json file which holds the timestamps for logs from runs where pruning was data agnostic 

## Channels that would've been pruned before performing finetuning
- **No_Finetune_Channels_Pruned** : True
- **Pretrained** : path to model that was trained originally before any finetuning

* location is currently set by default to '{path to pruning dir}/logs/{arch}/{dataset}/baseline/pre\_ft\_pp\_{pruning percentage}.pth.tar'

## Prune filters 
Multiple configurations possible. Base parameters are as follows: 
- **Prune_Filters** : True
- **Pruning_Perc** : percentage of network (memory wise) to be pruned
- **Finetune_Budget** : Number of epochs after pruning that retraining should be performed 
- **Prune_After** : Epoch at which pruning should be performed 
### Finetune on subset before pruning
- **Finetune** : True 
- **Static** : True 
    * currently no other mode exists for finetuning
    * static refers to setting a pre-defined and fixed number of epochs for finetuning and retraining
### Prune and retrain from scratch
- **Finetune** : False
- **Retrain** : True
* Prune model without any finetuning and retrain from scratch on dataset specified by Sub_Name and Sub_Classes parameters
### Perform inference on pruned model without any retraining
- **Finetune** : False
- **Retrain** : False
* will prune model specified in Pretrained to pruning percentage sepcified by Pruning_Perc and report inference statistics 

## Performing binary search DaPR methodology 
- **Binary_Search** : True
    * setting this to true will perform the binary search algorithm described in the paper "Now that I can see, I can improve : Enabling data-driven finetuning of CNNs on the edge"
    * the other settings that need to be set are 'Pruning_After'(*n<sub>f</sub>* in paper), 'Finetune_Budget(*n<sub>r</sub>* in paper)'
- Currently the values of *p<sub>l</sub>* is fixed to 5%, *p<sub>u</sub>* is fixed to 95% and *p<sub>i</sub>* is fixed to 5% corresponding to the paper.

**Pruning a Custom Network** 
==============================
- If network has a special module, and this module has a conv, a decorator needs to be created (decorators.py)

sub-heading
-----------------------
text

