#!/bin/bash

# Configurable via env vars; defaults assume layout under repo root.
BIRD_DEV_ROOT="${BIRD_DEV_ROOT:-data/bird/dev}"
db_root_path="${BIRD_DB_ROOT:-${BIRD_DEV_ROOT}/dev_databases/}"
data_mode="dev"
diff_json_path="${BIRD_DIFF:-${BIRD_DEV_ROOT}/dev.json}"
predicted_sql_json_path="${PREDICTED_SQL_JSON:?set PREDICTED_SQL_JSON to your predict_dev.json}"
ground_truth_sql_path="${BIRD_GOLD:-${BIRD_DEV_ROOT}/dev_gold.sql}"
num_cpus=12
meta_time_out=30.0
time_out=60
mode_gt="gt"
mode_predict="gpt"

# evaluate EX
echo "Evaluate BIRD EX begin!"
python ./official_bird_evaluation/evaluation_bird_ex.py --db_root_path $db_root_path \
    --predicted_sql_json_path $predicted_sql_json_path \
    --data_mode $data_mode \
    --ground_truth_sql_path $ground_truth_sql_path \
    --num_cpus $num_cpus \
    --mode_predict $mode_predict \
    --diff_json_path $diff_json_path \
    --meta_time_out $meta_time_out
echo "Evaluate EX done!"

# # evaluate VES
# echo "Evaluate BIRD VES begin!"
# python ./evaluation/evaluation_bird_ves.py \
#     --db_root_path $db_root_path \
#     --predicted_sql_json_path $predicted_sql_json_path \
#     --data_mode $data_mode \
#     --ground_truth_sql_path $ground_truth_sql_path \
#     --num_cpus $num_cpus --meta_time_out $time_out \
#     --mode_gt $mode_gt --mode_predict $mode_predict \
#     --diff_json_path $diff_json_path
# echo "Evaluate VES done!"
