mkdir -p logs

for n in 1 5 10 20 30 40 50; 
do
echo $n
python -u evaluate_bird_dev.py \
    --model-path /home/datht/huggingface/griffith-bigdata/FINER-SQL-3B-BIRD  \
    --data-path /home/datht/grast-sql/end2end/data-dev/grpo_sql_writer_bird_dev/ \
    -n $n \
    --temperature 1.0 \
    --batch-size 32 \
    --max-samples -1 \
    --output-dir output/bird/FINER-SQL-3B-BIRD-dev-n$n > logs/FINER-SQL-3B-BIRD-dev-n$n.log
done