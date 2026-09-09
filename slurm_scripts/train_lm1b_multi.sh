#!/bin/bash

#SUBSET_SIZES=(0.000719 0.001338 0.001956 0.003194 0.003813 0.004431 0.005669 0.006288 0.006906 0.008144 0.008762 0.009381)

SUBSET_SIZES=(0.000719 0.001338 0.001956)

#SUBSET_SIZES=(0.02 0.03 0.05 0.06)

#SUBSET_SIZES=(0.02 0.06)

SUBSET_SIZES=(0.06)

global_batch_size=512
batch_size=128
num_nodes=1
num_gpus=4

num_jobs=1

effective_batch_size=$((batch_size * num_gpus * num_nodes))
accumulation_steps=$((global_batch_size / effective_batch_size))

echo "Accumulation Steps: $accumulation_steps"

size=small

gpu_type="nvidia_h100_80gb_hbm3" # or nvidia_h200

time_limit="36:00:00"

work_dir="${WORK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cache_dir="${work_dir}/text_data/"

for subset in "${SUBSET_SIZES[@]}"; do
    sbatch <<EOT
#!/bin/bash
#SBATCH -J lm1b-${size}-${subset}              # Job name
#SBATCH -o watch_folder/lm1b-wrap-${size}/%x_%j.out     # output file (%j expands to jobID)
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH -t ${time_limit}              # Time limit (hh:mm:ss)
#SBATCH --cpu-freq=Performance        # Cpu frequency
#SBATCH --partition=rpi               # Request partition
#SBATCH -N ${num_nodes}                          # Total number of nodes requested
#SBATCH --ntasks-per-node=${num_gpus}            # Processes per compute node
#SBATCH --gres=gpu:${gpu_type}:${num_gpus}
#SBATCH --open-mode=append                       # Do not overwrite logs
#SBATCH --requeue                                # Requeue upon pre-emption
#SBATCH --array=1-${num_jobs}%1

export WANDB_MODE=offline
export WANDB_DISABLE_UPDATE_CHECK=true
export NO_PROXY="localhost,127.0.0.1"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((num_gpus - 1)))

conda activate duo

echo "------------------------------------------------"
echo "CONDA ENVIRONMENT CHECK"
echo "Active Env: $CONDA_DEFAULT_ENV"
echo "Python Path: $(which python)"
echo "------------------------------------------------"

srun --export=ALL python -u -m main \
  trainer.devices=${num_gpus} \
  trainer.num_nodes=${num_nodes} \
  trainer.accumulate_grad_batches=${accumulation_steps} \
  loader.global_batch_size=${global_batch_size} \
  loader.batch_size=${batch_size} \
  loader.eval_batch_size=${batch_size} \
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