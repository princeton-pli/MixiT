#!/bin/bash
##SBATCH --job-name=add-shaped
#SBATCH --nodes=1                # node count
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --ntasks-per-node=1      # total number of gpus per node
##SBATCH --gpu-bind=single:1
#SBATCH --cpus-per-task=10
#SBATCH --time=6:00:00          # total run time limit (HH:MM:SS)
#SBATCH --partition=pli-c
##SBATCH --qos=pli-cp
#SBATCH --constraint=rh8|rh9
##SBATCH --array=1-1 # 1-4


# conda init bash
# source ~/.bashrc
source /home/ydong/miniforge3/etc/profile.d/conda.sh
conda init
conda activate /scratch/gpfs/ydong/miniforge3
export WANDB_MODE=offline
export WANDB_CACHE_DIR=/scratch/gpfs/mkhodak/smt/wandb/.cache

task=${task:-"additionstream_10"}
attn=${shaped_attention:-"mixing"}
# heads=$(( 4 ** SLURM_ARRAY_TASK_ID ))
n_head=${n_head:-128}


# for layers in 2 4 6 8 10 ; do
for layers in 8; do

  python depth.py --task=$task --train_steps=500 --eval_steps=100 --weight_frozen=0 --n_layer=$layers --shaping=$attn --n_embd=512 --n_head=${n_head}

done
