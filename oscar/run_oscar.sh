#!/bin/bash -u
stage=-1			# start from -1 if you need to start from data download
stop_stage=0			# stage at which to stop

datasource=./Sourcedata
dsetdir=./dataset
outdir=./output
cashedir=./CASHE

#ln -s /home/wentao/Desktop/Memotion3/dataset /home/wentao/Desktop/Oscar/Oscar
#ln -s /home/wentao/Desktop/Memotion3/Sourcedata /home/wentao/Desktop/Oscar/Oscar
<<'COMMEN'
python oscar/run_hate_pretrain.py \
    --data_dir dataset \
    --output_dir output/pretrain_taskA \
    --num_labels 2 \
    --model_name_or_path subtask/imgtextmatching/output/checkpoint-7-11136 \
    --do_lower_case \
    --do_train \
    --evaluate_during_training \
    --num_captions_per_img_val 1 \
    --per_gpu_train_batch_size 8 \
    --learning_rate 2e-5 \
    --num_train_epochs 38 \
    --weight_decay 0.05 \
    --warmup_steps 10600 \
    --save_steps 2120 \
    --add_od_labels \
    --od_label_type vg \
    --max_seq_length 70 
COMMEN

python oscar/run_hate_train_taskA.py \
    --data_dir dataset \
    --output_dir output/train_taskA \
    --num_labels 3 \
    --model_name_or_path output_wrong_f1/pretrain_taskA/checkpoint-7-16960 \
    --do_lower_case \
    --do_train \
    --evaluate_during_training \
    --num_captions_per_img_val 1 \
    --per_gpu_train_batch_size 8 \
    --learning_rate 2e-5 \
    --num_train_epochs 38 \
    --weight_decay 0.05 \
    --warmup_steps 4300 \
    --save_steps 860 \
    --add_od_labels \
    --od_label_type vg \
    --max_seq_length 70 || exit 1


for type in 'humorous' 'sarcastic' 'offensive' 'motivation'; do #'humorous'

python oscar/run_hate_train_taskC.py \
    --data_dir dataset \
    --type $type \
    --output_dir output/train_taskC/${type} \
    --model_name_or_path output/train_taskA \
    --do_lower_case \
    --do_train \
    --evaluate_during_training \
    --num_captions_per_img_val 1 \
    --per_gpu_train_batch_size 8 \
    --learning_rate 2e-5 \
    --num_train_epochs 38 \
    --weight_decay 0.05 \
    --warmup_steps 4300 \
    --save_steps  860 \
    --add_od_labels \
    --od_label_type vg \
    --max_seq_length 70 || exit 1;

done
#860
