#!/bin/bash
##SBATCH --job-name=${job_name}
#SBATCH --nodes=1                # node count
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --ntasks-per-node=1      # total number of gpus per node
##SBATCH --gpu-bind=single:1
#SBATCH --cpus-per-task=10
#SBATCH --time=12:00:00          # total run time limit (HH:MM:SS)
#SBATCH --partition=pli-c
##SBATCH --qos=pli-cp
#SBATCH --constraint=rh9
##SBATCH --array=1-2 # 1-4
#SBATCH --mem=20G 
#SBATCH --mail-user=ydong@princeton.edu
#SBATCH --mail-type=end
#SBATCH --mail-type=begin
#SBATCH --output=slurm/${job_name}

task=${task:-"additionstream_10"}
cache_dir=${cache_dir:-/scratch/gpfs/PLI/ydong/hf_cache}
max_seq_len=${max_seq_len:-512}
n_embd=${n_embd:-768}
# n_head=${n_head:-16}
max_steps=${max_steps:-500}
shaped_attention=${shaped_attention:-mixing}
n_layer=${n_layer:-8}
learning_rate=${learning_rate:-5e-3}
per_device_train_batch_size=${per_device_train_batch_size:-128}
per_device_eval_batch_size=${per_device_eval_batch_size:-128}
job_hours=${job_hours:-1}
n_gpu=${n_gpu:-4}
# llama3=True                                                                                            
model_name_or_path=${model_name_or_path:-llama}
freeze_attention=${freeze_attention:-0}
freeze_mlp=${freeze_mlp:-0}
depth_alpha=${depth_alpha:-0}

# conda init bash
# source ~/.bashrc
source /home/ydong/miniforge3/etc/profile.d/conda.sh
conda init
conda activate /scratch/gpfs/ydong/miniforge3
export WANDB_MODE=offline
# export WANDB_CACHE_DIR=/scratch/gpfs/mkhodak/smt/wandb/.cache

# task=${task:-"additionstream_10"}
shaped_attention=${shaped_attention:-"mixing"}
# n_head=$(( 4 * SLURM_ARRAY_TASK_ID ))
weight_frozen=${weight_frozen:-1}

if [[ ${shaped_attention} == "mixing" ]]; then
    # n_head=$(( 8 * SLURM_ARRAY_TASK_ID ))
    # n_head=$(( 4 * SLURM_ARRAY_TASK_ID ))
    all_n_head=(4 32 64)
    # all_layers=(4 8 12)
    all_layers=(2)
    all_n_embds=(512 1024)
elif [[ ${shaped_attention} == "shaped" ]]; then
    n_head=$(( 8 * SLURM_ARRAY_TASK_ID ))
    all_layers=(2 4 8 12)
    all_n_embds=(512 1024 2048)
else
    # n_head=$(( 8 * SLURM_ARRAY_TASK_ID ))
    # all_layers=(2 4 8 12)
    # all_n_embds=(512 1024)
    all_n_head=(4)
    # all_layers=(4 8 12)
    all_layers=(2)
    all_n_embds=(512)
fi

if [[ ${task} == "memorize" ]]; then
    script="hdmemorize_exp.py"
else
    script="depth.py"
fi

# for task in additionstream_10; do
# for layers in 2 4 6 8 10 ; do
for n_embd in "${all_n_embds[@]}"; do
for n_layer in "${all_layers[@]}"; do
for n_head in "${all_n_head[@]}"; do

    python ${script} --task=$task --max_steps=${max_steps} --eval_steps=1000 --logging_steps=1000 --weight_frozen=${weight_frozen} --n_layer=$n_layer --shaped_attention=${shaped_attention} --n_embd=${n_embd} --n_head=${n_head} --model_name_or_path=${model_name_or_path} --per_device_train_batch_size=${per_device_train_batch_size} --per_device_eval_batch_size=${per_device_eval_batch_size} --llama3=True --freeze_attention=${freeze_attention} --freeze_mlp=${freeze_mlp} --depth_alpha=${depth_alpha}
    
    # --n_embd ${n_embd} --n_head ${n_head} --max_steps=${max_steps} --shaped_attention ${shaped_attention} --eval_steps 1500 --logging_steps=1500 --n_layer ${n_layer} --per_device_train_batch_size ${per_device_train_batch_size} --per_device_eval_batch_size ${per_device_eval_batch_size} --streaming_train_root=${streaming_train_root} --streaming_val_root=${streaming_val_root} --domains_and_proportions_train=${domains_and_proportions_train} --domains_and_proportions_val=${domains_and_proportions_val} --learning_rate ${learning_rate} --activation_cminus=${activation_cminus}

    # python depth.py --task=$task --train_steps=500 --eval_steps=100 --weight_frozen=0 --n_layer=${n_layer} --shaping=$attn --n_embd=512 --n_head=${n_head}
done
done
done
# done
