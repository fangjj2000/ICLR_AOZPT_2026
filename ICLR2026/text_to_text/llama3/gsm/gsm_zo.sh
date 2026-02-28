
cd /hy-tmp/ICLR2026/text_to_text
export CUDA_VISIBLE_DEVICES=1
seeds=${2:-"14 42 81"}
learning_rate=0.01
mu=0.01
alpha=0
use_confident=0
beta=0.99
window_size=50
for seed in $seeds;do
    python llama3/gsm/zo/gsm_zo.py \
        --input_file /hy-tmp/ICLR2026/text_to_text/datasets/gsm.csv \
        --output_csv_file output/gsm_llama3/gsm_zo_${learning_rate}_${mu}_${window_size}_${alpha}_${beta}/${seed}.csv \
        --seed $seed \
        --learning_rate $learning_rate\
        --mu $mu\
        --alpha $alpha\
        --use_confident $use_confident\
        --beta $beta \
        --window_size $window_size
done