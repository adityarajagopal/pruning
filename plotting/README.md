## Creating new json file for storing time stamped directories 
- Use the --add_network command in plot.py 
- --name takes the name of the new network that you want to add 
- --pre_ft_path is the path to the model that was used before finetuning
    - if finetuning was not performed then this can be left empty 
- --base_folder is the name of the folder that holds the timestamped pp\_{} directories
- --logs_json passes the full path to the json file you want to create

-- Optional : set the --subsets if you want only certain subsets to be add per network

## Updating the json with logs
- Use the --update_logs command 
- --as_of takes the date including and after which logs will be added in the format y-m-d
- --logs_json passes the full path to the json file you want to update

-- Optional : set the --subsets if you want only certain subsets to be add per network
-- The command uses the base path present in the json file to automatically find the directories 
-- The command also ensure no duplicate logs are added
