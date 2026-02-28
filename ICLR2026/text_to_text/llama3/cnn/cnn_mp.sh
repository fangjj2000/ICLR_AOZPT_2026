seeds=${2:-"14 42 81"}
cd /hy-tmp/ICLR2026/text_to_text
export CUDA_VISIBLE_DEVICES=0
for seed in $seeds;do
    python /hy-tmp/ICLR2026/text_to_text/llama3/cnn/zo/cnn_mp.py \
        --input_file /hy-tmp/ICLR2026/text_to_text/datasets/cnn.csv \
        --output_csv_file /hy-tmp/ICLR2026/text_to_text/output_new/cnn_mp_${seed}.csv \
        --seed $seed 
done

