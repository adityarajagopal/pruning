# -------- testing runs ----------------------
# python plotting/plot.py --logs_json logs/logs.json --channel_diff --pre_post_ft --subset_agnostic_logs logs/logs.json --subsets indoors natural random1 
# python plotting/plot.py --logs_json logs/logs.json --channel_diff --across_networks --subset_agnostic_logs logs/logs.json --subsets indoors natural random1
# python plotting/plot.py --logs_json logs/logs.json --subset_agnostic_logs logs/logs.json --l1_norm

# -------- collection runs --------------------
python plotting/plot.py --channel_diff --pre_post_ft --logs_json logs/logs.json --subset_agnostic_logs logs/logs.json --save --loc l1_norm --silent 
python plotting/plot.py --channel_diff --across_networks --logs_json logs/logs.json --subset_agnostic_logs logs/logs.json --save --loc l1_norm --silent
# python plotting/plot.py --logs_json logs/logs.json --subset_agnostic_logs logs/logs.json --l1_norm --save --loc l1_norm --silent
