python -u evaluate_spider2_sqlite.py \
    --model-path /home/datht/huggingface/griffith-bigdata/FINER-SQL-3B-BIRD \
    --data-path /home/datht/grast-sql/end2end/data/spider2_sqlite_top30 \
    -n 50 \
    --temperature 1.0 \
    --batch-size 32 \
    --max-samples -1 \
    --output-dir output/spider2_sqlite/FINER-SQL-3B-BIRD-top30 > FINER-SQL-3B-BIRD-top30.log
