#!/bin/bash

#SUBSET_SIZES=(0.000719 0.001338 0.001956 0.002575 0.003194 0.003813 0.004431 0.005669 0.006288 0.006906 0.007525 0.008144 0.008762 0.009381)

SUBSET_SIZES=(0.001338 0.001956 0.002575 0.003194 0.003813 0.004431 0.005669 0.006288 0.006906 0.007525 0.008144 0.008762 0.009381)

num_jobs=100
size=tiny

num_gpus=4
num_nodes=1

cache_dir=/gpfs/u/home/HPDM/HPDMphmb/scratch/diffusion-duality/text_data/

for subset in "${SUBSET_SIZES[@]}"; do
    # Define the checkpoint path variable
    ckpt_path="${ckpt_dir}/lm1b-${size}/lm1b-${size}-${subset}.ckpt"

    sbatch <<EOT
#!/bin/bash
#SBATCH -J lm1b-${size}-${subset}              # Job name
#SBATCH -o watch_folder/npl-extras/lm1b_wrap_${size}/%x_%j.out     # output file (%j expands to jobID)
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH -t 06:00:00                   # Time limit (hh:mm:ss)
#SBATCH --cpu-freq=Performance        # Cpu frequency
#SBATCH --partition=npl-2024          # Request partition
#SBATCH -N ${num_nodes}                          # Total number of nodes requested
#SBATCH --ntasks-per-node=${num_gpus}           # Processes per compute node
#SBATCH --gres=nvme,gpu:32g:${num_gpus}         # Type/number of GPUs needed
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon pre-emption
#SBATCH --array=1-${num_jobs}%1

export WANDB_MODE=offline
export WANDB_DISABLE_UPDATE_CHECK=true
export NO_PROXY="localhost,127.0.0.1"

srun --export=ALL python -u -m main \
  loader.batch_size=64 \
  loader.eval_batch_size=64 \
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
  trainer.num_nodes=${num_nodes} \
  data.subset=${subset} 
EOT
done 