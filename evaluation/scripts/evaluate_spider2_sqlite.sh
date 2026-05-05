python -u evaluate_spider2_sqlite.py \
    --model-path /home/datht/huggingface/griffith-bigdata/FINER-SQL-3B-BIRD \
    --data-path /home/datht/grast-sql/end2end/data/spider2_sqlite_top40 \
    -n 20 \
    --temperature 1.0 \
    --batch-size 16 \
    --max-samples -1 \
    --output-dir output/spider2_sqlite/FINER-SQL-3B-BIRD-top40-n20-t1.0 > FINER-SQL-3B-BIRD-top40-n20-t1.0.log
