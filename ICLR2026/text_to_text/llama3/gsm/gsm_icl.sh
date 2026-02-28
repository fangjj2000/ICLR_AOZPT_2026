seeds=${2:-"14 42 81"}
cd /hy-tmp/ICLR2026/text_to_text
export CUDA_VISIBLE_DEVICES=1
for seed in $seeds;do
    python llama3/gsm/zo/gsm_icl.py \
        --input_file /hy-tmp/ICLR2026/text_to_text/datasets/gsm.csv \
        --output_csv_file output/gsm_llama3/gsm_icl/${seed}.csv \
        --seed $seed 
done