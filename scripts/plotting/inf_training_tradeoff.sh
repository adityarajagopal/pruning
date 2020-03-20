# python plotting/plot.py --logs_json logs/logs.json --subset_agnostic_logs logs/logs.json --prof_logs profiling_logs/tx2 --time_tradeoff --subsets indoors natural random1
# python plotting/plot.py --logs_json logs/logs.json --subset_agnostic_logs logs/logs.json --prof_logs profiling_logs/tx2 --time_tradeoff  --networks resnet
# python plotting/plot.py --logs_json logs/logs.json --subset_agnostic_logs logs/logs.json --prof_logs profiling_logs/tx2 --time_tradeoff  --networks alexnet
# python plotting/plot.py --logs_json logs/logs.json --subset_agnostic_logs logs/logs.json --prof_logs profiling_logs/tx2 --time_tradeoff  --networks mobilenetv2
python plotting/plot.py --logs_json logs/logs.json --subset_agnostic_logs logs/logs.json --prof_logs profiling_logs/tx2 --time_tradeoff  --networks squeezenet
# python plotting/plot.py --logs_json logs/logs.json --subset_agnostic_logs logs/logs.json --prof_logs profiling_logs/tx2 --time_tradeoff --save --loc l1_norm --silent --subsets indoors natural random1
