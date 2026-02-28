#export https_proxy="127.0.0.1:1081"
#export http_proxy="127.0.0.1:1081"
export CUDA_VISIBLE_DEVICES=3
tasks=${1:-"gsm"} #cnn gsm wmt14
seeds=${2:-"14 42 81"}
gpt_model=${3:-"gpt-4o-mini"}
learning_rate=0.01
mu=0.01
alpha=0
use_confident=0
beta=0.99
window_size=50
# -m debugpy --listen 5678 --wait-for-client
# python -m debugpy --listen 1234 --wait-for-client gsm_ZO-OGDWAC.py \S
cd /hy-tmp/ICLR2026/text_to_text
for task in $tasks;do
    for seed in $seeds;do
        python gpt4/mp/main_gpt4.py \
            --input_file /hy-tmp/ICLR2026/text_to_text/datasets/${task}.csv \
            --output_csv_file output/mp/${task}_${gpt_model}/${seed}.csv \
            --seed $seed \
            --task $task \
            --model_name_or_path $gpt_model \
            --zero_shot \
            --learning_rate $learning_rate \
            --mu $mu\
            --alpha $alpha \
            --beta $beta \
            --window_size $window_size
    done
done