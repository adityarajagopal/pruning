# python plotting/plot.py --logs_json logs/logs.json --inf_gops --subset_agnostic_logs logs/logs.json --subsets indoors natural random1
# python plotting/plot.py --logs_json logs/logs.json --prof_logs profiling_logs/tx2 --inf_gops --subset_agnostic_logs logs/logs.json --subsets indoors natural random1
python plotting/plot.py --logs_json logs/logs.json --inf_gops --subset_agnostic_logs logs/logs.json --subsets indoors natural random1 --save --loc l1_norm --silent
python plotting/plot.py --logs_json logs/logs.json --prof_logs profiling_logs/tx2 --inf_gops --subset_agnostic_logs logs/logs.json --save --loc l1_norm --subsets indoors natural random1 --silent
