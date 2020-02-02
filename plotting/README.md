# Updating the logs JSON

## Creating new json file for storing time stamped directories 
### Use the --add_network command in plot.py 
- **--name** : name of the new network that you want to add 
- **--pre_ft_path** : path to the model that was used before finetuning
    * if finetuning was not performed then this can be left empty 
- **--base_folder** : name of the folder that holds the timestamped pp\_{} directories
- **--logs_json** : full path to the json file you want to create

* Optional : set the **--subsets** if you want only certain subsets to be add per network

## Updating the json with logs
### Use the --update_logs command 
- **--as_of** : date including and after which logs will be added in the format y-m-d
- **--logs_json** passes the full path to the json file you want to update

* Optional : set the **--subsets** if you want only certain subsets to be add per network
* The command uses the base path present in the json file to automatically find the directories 
* The command also ensure no duplicate logs are added

# Plotting

## Plotting inference gops vs accuracy tradeoff
### Use the --inf_gops command
- **--subset_agnostic_logs** : path to the json file containing the timestamps for data agnostic pruning, 
  i.e. pruning and retraining on entire dataset instead of specific to a subset

* Set **--networks** and **--subsets** to specific ones if only certain graphs required

## Plotting cost of performing binary search to find best pruning percentage 
### Use the --bin_search_cost command 
- **--mode** : 
    * memory_opt - performs binary search to find smallest model size that results in test accuracy >= target 
    * cost_opt - performs binary search to find first model that results in test accuracy >= target, optimises for search cost

* Set **--networks** and **--subsets** to specific ones if only certain graphs required

## Plotting the difference in channels pruned 
### Use the --channel_diff command 
- **--pre_post_ft** : 
    * plots difference in channels pruned between models before and after finetuning as well as difference in channels pruned between post finetune models
    * produces a different plot per network and subset

- **--across_networks**: 
    * plots difference in channels pruned between models before and after finetuning on a subset
    * produces a differen plot per subset only 
    * each x-axis point has a bar per network 

## Plotting l1_norm histograms
### Use the --l1_norm command
- Plots per network subset, a histogram of l1-norms of the model before and after finetuning as well as a plot of the histogram of the difference between the l1-norms before and after fineutning  

## Plotting finetuning gops vs test accuracy 
### Use the --ft_gops command
- Plots per network and subset a graph of total finetuning gops on the x-axis vs test accuracy on the y-axis

## Plotting per epoch cost of finetuning 
### Use the --ft_epoch_gops command 
- Plots per network and subset a graph of cumulative cost in GOps in the x-axis and test accuracy on the y-axis
- **--plot_as_line** : selects pruning percentages to plot as a line, while percentages no in this are simply plotted as a point at the end (similar to **--ft_gops**
- **--acc_metric** : change the metric on the y-axis (default - Test_Top1)
