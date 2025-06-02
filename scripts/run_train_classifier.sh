#!/bin/bash

task=${task:-wikitext}
cache_dir=${cache_dir:-/scratch/gpfs/PLI/ydong/hf_cache}
max_seq_len=${max_seq_len:-512}
n_embd=${n_embd:-512}
n_head=${n_head:-16}
max_steps=${max_steps:-6000}
shaped_attention=${shaped_attention:-mixing}
n_layer=${n_layer:-8}
learning_rate=${learning_rate:-5e-3}
per_device_train_batch_size=${per_device_train_batch_size:-64}
per_device_eval_batch_size=${per_device_eval_batch_size:-64}
job_hours=${job_hours:-1}
n_gpu=${n_gpu:-4}
postfix=${postfix:-""}
# llama3=True
model_name_or_path=${model_name_or_path:-llama}
activation_cminus=${activation_cminus:-1}
freeze_attention=${freeze_attention:-0}
depth_alpha=${depth_alpha:-0}
freeze_mlp=${freeze_mlp:-False}

job_name=${shaped_attention}_${task}_seq${max_seq_len}_layer${n_layer}_emb${n_embd}_bs${per_device_train_batch_size}_steps${max_steps}_lr${learning_rate}_gpu${n_gpu}_nhead${n_head}_${model_name_or_path}_cminus${activation_cminus}freeze_att${freeze_attention}freeze_mlp${freeze_mlp}_depth_alpha${depth_alpha}${postfix}


task=${task} freeze_attention=${freeze_attention} freeze_mlp=${freeze_mlp} depth_alpha=${depth_alpha} model_name_or_path=${model_name_or_path} cache_dir=${cache_dir} max_seq_len=${max_seq_len} n_embd=${n_embd} max_steps=${max_steps} shaped_attention=${shaped_attention} n_layer=${n_layer} n_head=${n_head} per_device_eval_batch_size=${per_device_eval_batch_size} per_device_train_batch_size=${per_device_train_batch_size} learning_rate=${learning_rate} sbatch -J ${job_name} -N 1 --ntasks-per-node ${n_gpu} --mem=20G --cpus-per-task 10  --gres=gpu:${n_gpu} -p pli-c --mail-user=ydong@princeton.edu --mail-type=end --mail-type=begin --output=slurm/${job_name} --time=${job_hours}:00:00 --constraint="rh9" scripts/train_classifier.sh
