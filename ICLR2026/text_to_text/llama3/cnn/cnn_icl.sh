seeds=${2:-"14 42 81"}
cd /hy-tmp/ICLR2026/text_to_text
export CUDA_VISIBLE_DEVICES=0
for seed in $seeds;do
    python llama3/cnn/zo/cnn_icl.py \
        --input_file /hy-tmp/ICLR2026/text_to_text/datasets/cnn.csv \
        --output_csv_file output/cnn_llama3/cnn_icl/${seed}.csv \
        --seed $seed 
done