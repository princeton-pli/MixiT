#!/bin/bash

task=${task:-wikitext}
cache_dir=${cache_dir:-/scratch/gpfs/PLI/ydong/hf_cache}
max_seq_len=${max_seq_len:-512}
n_embd=${n_embd:-512}
n_head=${n_head:-16}
max_steps=${max_steps:-3000}
shaped_attention=${shaped_attention:-mixing}
n_layer=${n_layer:-8}
learning_rate=${learning_rate:-5e-3}
per_device_train_batch_size=${per_device_train_batch_size:-64}
per_device_eval_batch_size=${per_device_eval_batch_size:-64}
model_name_or_path=${model_name_or_path:-llama}
# Specifying this to avoid address already in use errors.
master_port=${master_port:-29500}
activation_cminus=${activation_cminus:-1}
depth_alpha=${depth_alpha:-1.0}
freeze_attention=${freeze_attention:-False}
freeze_mlp=${freeze_mlp:-False}

streaming_train_root=/scratch/gpfs/PLI/tianyug/conditional_pretraining/packed
streaming_val_root=/scratch/gpfs/PLI/tianyug/conditional_pretraining/packed
domains_and_proportions_train="{'dclm-0-99-complete'=1.0}"
domains_and_proportions_val="{'refinedweb-172b-len8k'=1.0}"

echo n_layer ${n_layer}
echo ${domains_and_proportions_train}

source /home/ydong/miniforge3/etc/profile.d/conda.sh
conda init
conda activate /home/ydong/miniforge3

# Multi-GPU
if [ -z "$SLURM_NTASKS_PER_NODE" ]
then
    SLURM_NTASKS_PER_NODE=$(expr $SLURM_NTASKS / $SLURM_NNODES)
fi

# FIRSTNODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# export MASTER_ADDR=$FIRSTNODE

# export WORLD_SIZE=$(expr $SLURM_NTASKS_PER_NODE \* $SLURM_NNODES)
export LOCAL_WORLD_SIZE=$SLURM_NTASKS_PER_NODE
# export LOCAL_RANK=$SLURM_LOCALID
# export RANK=$(expr $SLURM_NODEID \* $SLURM_NTASKS_PER_NODE + $SLURM_LOCALID)
export OMP_NUM_THREADS=8

# echo world size $WORLD_SIZE, local world size $LOCAL_WORLD_SIZE, local rank $LOCAL_RANK, rank $RANK

export WANDB_MODE=offline


output_dir="results/${shaped_attention}_${task}_seq${max_seq_len}_layer${n_layer}_emb${n_embd}_bs${per_device_train_batch_size}_steps${max_steps}_lr${learning_rate}_gpu${WORLD_SIZE}_nhead${n_head}_${model_name_or_path}_freeze_attention${freeze_attention}"

            
torchrun --nnodes=1 --master_port=${master_port} --nproc_per_node=${LOCAL_WORLD_SIZE} trainer.py --task ${task} --output_dir ${output_dir} --cache_dir ${cache_dir} --model_name_or_path ${model_name_or_path} --max_seq_len ${max_seq_len} --n_embd ${n_embd} --n_head ${n_head} --max_steps=${max_steps} --shaped_attention ${shaped_attention} --eval_steps 1500 --logging_steps=1500 --n_layer ${n_layer} --per_device_train_batch_size ${per_device_train_batch_size} --per_device_eval_batch_size ${per_device_eval_batch_size} --streaming_train_root=${streaming_train_root} --streaming_val_root=${streaming_val_root} --domains_and_proportions_train=${domains_and_proportions_train} --domains_and_proportions_val=${domains_and_proportions_val} --learning_rate=${learning_rate} --activation_cminus=${activation_cminus} --depth_alpha=${depth_alpha} --freeze_attention=${freeze_attention} --freeze_mlp=${freeze_mlp}
