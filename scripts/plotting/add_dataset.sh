python plotting/plot.py --add_dataset --name indoors --base_folder l1_prune --logs_json /home/ar4414/pytorch_training/src/ar4414/pruning/logs/logs.json 

# need to add the name of the subset added to plot.py before calling this
python plotting/plot.py --update_logs --as_of 2020-03-03 --logs_json /home/ar4414/pytorch_training/src/ar4414/pruning/logs/logs.json
