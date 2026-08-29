#!/bin/bash
#SBATCH -J duo-lm1b-wrap-medium              # Job name
#SBATCH -o watch_folder/dcs/duo_lm1b_wrap_medium/%x_%j.out     # output file (%j expands to jobID)
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH -t 06:00:00                   # Time limit (hh:mm:ss)
#SBATCH --cpu-freq=Performance        # Cpu frequency
#SBATCH --partition=dcs-2024          # Request partition
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --ntasks-per-node=4           # Processes per compute node
#SBATCH --gres=nvme,gpu:32g:4         # Type/number of GPUs needed
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon pre-emption

# To enable preemption re-loading, set `hydra.run.dir` or 
# `checkpointing.save_dir` explicitly.

# This task requires a global batch_size of 512
# Since we have 4 * 32 < 512, gradient accummulation (of 4 steps) will happen.  
# Run up to 1M steps

#[0.01, 0.04, 0.07, 0.1, 0.13, 0.16, 0.19, 0.22, 0.25, 0.28, 0.31] 
#[0.34, 0.37, 0.4, 0.43, 0.46, 0.49, 0.52, 0.55, 0.58, 0.61, 0.64]
#[0.67, 0.7, 0.73, 0.76, 0.79, 0.82, 0.85, 0.88, 0.91, 0.94, 0.97]
#[1.00]

size=medium
subset=0.0001
cache_dir=/gpfs/u/home/HPDM/HPDMphmb/scratch/diffusion-duality/text_data/

export WANDB_MODE=offline
export WANDB_DISABLE_UPDATE_CHECK=true
#export WANDB_SILENT=true
export NO_PROXY="localhost,127.0.0.1"

srun --export=ALL python -u -m main \
  loader.batch_size=32 \
  loader.eval_batch_size=32 \
  data=lm1b-wrap \
  data.cache_dir=${cache_dir} \
  wandb.name=duo-lm1b-wrap-${subset}-${size} \
  wandb.project=lm1b-unpack-${size} \
  model=${size} \
  algo=duo \
  model.length=128 \
  algo.gumbel_tau_log10_start=-3.0 \
  algo.gumbel_tau_log10_end=-3.0 \
  algo.gamma_min=-3.5 \
  algo.gamma_max=-1.75 \
  algo.curriculum_start=0 \
  algo.curriculum_end=500000 \
  training.loss_precision=16-mixed \
  trainer.precision=16-mixed \
  trainer.num_nodes=1 \
  data.subset=${subset} 