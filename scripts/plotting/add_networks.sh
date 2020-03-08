python plotting/plot.py --add_network --name alexnet --pre_ft_path /home/ar4414/pytorch_training/src/ar4414/pruning/logs/alexnet/cifar100/baseline/2019-08-28-10-40-35/orig/61-model.pth.tar --base_folder l1_prune --logs_json /home/ar4414/pytorch_training/src/ar4414/pruning/logs/logs.json 

python plotting/plot.py --add_network --name resnet --pre_ft_path /home/ar4414/pytorch_training/src/ar4414/pruning/logs/resnet/cifar100/baseline/2019-08-27-11-58-14/orig/118-model.pth.tar --base_folder l1_prune --logs_json /home/ar4414/pytorch_training/src/ar4414/pruning/logs/logs.json 

python plotting/plot.py --add_network --name mobilenetv2 --pre_ft_path /home/ar4414/pytorch_training/src/ar4414/pruning/logs/mobilenetv2/cifar100/baseline/2019-10-07-15-17-32/orig/111-model.pth.tar --base_folder l1_prune --logs_json /home/ar4414/pytorch_training/src/ar4414/pruning/logs/logs.json 

python plotting/plot.py --add_network --name squeezenet --pre_ft_path /home/ar4414/pytorch_training/src/ar4414/pruning/logs/squeezenet/cifar100/baseline/2019-11-05-12-10-50/orig/167-model.pth.tar --base_folder l1_prune --logs_json /home/ar4414/pytorch_training/src/ar4414/pruning/logs/logs.json 

python plotting/plot.py --update_logs --as_of 2020-03-03 --logs_json /home/ar4414/pytorch_training/src/ar4414/pruning/logs/logs.json
