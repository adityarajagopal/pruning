# python plotting/plot.py --logs_json logs/logs.json --bin_search_cost --mode memory_opt
# python plotting/plot.py --logs_json logs/logs.json --bin_search_cost --mode cost_opt --save --loc l1_norm --silent
# python plotting/plot.py --prof_logs profiling_logs/tx2 --logs_json logs/logs.json --bin_search_cost --mode cost_opt --save --loc l1_norm --silent
# python plotting/plot.py --prof_logs profiling_logs/tx2 --logs_json logs/logs.json --bin_search_cost --mode memory_opt --save --loc l1_norm --silent

python plotting/plot.py --prof_logs profiling_logs/tx2 --logs_json logs/logs.json --bin_search_cost --mode memory_opt 
