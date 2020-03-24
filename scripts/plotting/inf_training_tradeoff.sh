#===================== testing runs =====================
# python plotting/plot.py --time_tradeoff --logs_json logs/logs.json --subset_agnostic_logs logs/logs.json --prof_logs profiling_logs/tx2 --networks alexnet
# python plotting/plot.py --logs_json logs/logs.json --subset_agnostic_logs logs/logs.json --prof_logs profiling_logs/tx2 --time_tradeoff  --networks resnet
# python plotting/plot.py --logs_json logs/logs.json --subset_agnostic_logs logs/logs.json --prof_logs profiling_logs/tx2 --time_tradeoff  --networks alexnet
# python plotting/plot.py --logs_json logs/logs.json --subset_agnostic_logs logs/logs.json --prof_logs profiling_logs/tx2 --time_tradeoff  --networks mobilenetv2
# python plotting/plot.py --logs_json logs/logs.json --subset_agnostic_logs logs/logs.json --prof_logs profiling_logs/tx2 --time_tradeoff  --networks squeezenet 

#===================== collection runs =====================
python plotting/plot.py --time_tradeoff --logs_json logs/logs.json --subset_agnostic_logs logs/logs.json --prof_logs profiling_logs/tx2 --save --loc l1_norm --silent
