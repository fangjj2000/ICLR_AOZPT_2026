export CUDA_VISIBLE_DEVICES=2
task=${1:-"cnn"}
seeds=${2:-"14 42 81"}
gpt_model=${3:-"gpt-4o-mini"}
learning_rate=0.01
mu=0.01
alpha=0.95
use_confident=0
beta=0.99
window_size=50
cd /hy-tmp/ICLR2026/text_to_text
for seed in $seeds;do
    python gpt4/zo/main_gpt4.py \
        --input_file /hy-tmp/ICLR2026/text_to_text/datasets/${task}.csv \
        --output_csv_file output/${task}_${gpt_model}/${learning_rate}_${mu}_${window_size}_${alpha}_${beta}/${seed}.csv \
        --seed $seed \
        --task $task \
        --model_name_or_path $gpt_model \
        --learning_rate $learning_rate\
        --mu $mu\
        --alpha $alpha\
        --beta $beta \
        --window_size $window_size
done
