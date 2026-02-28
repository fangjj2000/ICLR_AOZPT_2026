export CUDA_VISIBLE_DEVICES=0
seeds=${2:-"14 42 81"}
learning_rate=0.01
mu=0.01
alpha=0
use_confident=0.95
beta=0.99
window_size=50
cd /hy-tmp/ICLR2026/text_to_text
for seed in $seeds;do
    python llama3/cnn/zo/cnn_zo.py \
        --input_file /hy-tmp/ICLR2026/text_to_text/datasets/cnn.csv \
        --output_csv_file output/cnn_llama3/cnn_${learning_rate}_${mu}_${window_size}_${alpha}_${beta}/${seed}.csv \
        --seed $seed \
        --learning_rate $learning_rate\
        --mu $mu\
        --alpha $alpha\
        --beta $beta \
        --window_size $window_size
done