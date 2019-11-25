# Pruning Configurations 
- Prune without retraining
    - Prune_Filters : True
    - Finetune : False
    - Get_Gops : False
    - Plot_Channels : Commented out 

- Prune with retraining 
    - Prune_Filters : True
    - Finetune : True
    - Get_Gops : False 
    - Plot_Channels : Commented out

- Get Acc vs Pruning GOps graph (based on log-file)
    - Prune_Filters : True
    - Finetune : True
    - Get_Gops : True
    - Plot_Channels : Commented out
    - Sub_Name, Sub_Classes, LogDir, LogFiles : Need to be filled correctly 

- Get Pruned and Unpruned GOps for a certain level of pruning  
    - Prune_Filters : True
    - Finetune : False
    - Get_Gops : True
    - Plot_Channels : Commented out
    - PrunePerc : Set to desired value

- Get plot of stats about channels pruned   
    - Prune_Filters : True
    - Finetune : N/A
    - Get_Gops : False
    - Plot_Type : joint / number / hamming
    - Plot_Channels : Filled correctly 

 

