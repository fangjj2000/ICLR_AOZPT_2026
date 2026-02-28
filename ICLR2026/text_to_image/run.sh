export CUDA_VISIBLE_DEVICES=3
cd /hy-tmp/ICLR2026/text_to_image
seeds=${2:-"14 42 81"}
window_sizes=50
alpha=0.9
beta=0.95
learning_rate=0.01
mu=0.01  
for seed in $seeds;do
    python main.py --random_proj uniform --seed $seed --mu $mu --learning_rate $learning_rate --window_size $window_size --alpha_w $alpha --beta_w $beta \
    --sdmodel_name stable-diffusion-v1-5/stable-diffusion-v1-5 \
    --model_name vicuna \
    --input_file ./datasets/anime.csv \
    --HF_cache_dir lmsys/vicuna-13b-v1.3
done


# --model_name vicuna wizardlm openchat
# --HF_cache_dir lmsys/vicuna-13b-v1.3 QuixiAI/WizardLM-13B-Uncensored openchat/openchat-3.5-0106/WeiboAI/VibeThinker-1.5B