#!/bin/bash

# ================= CONFIGURATION =================
#SUBSET_SIZES=(0.0001 0.002575 0.00505 0.007525 0.01 0.04 0.07 0.1 0.13 0.16 0.19 0.22 0.25 0.28 0.31 0.34 0.37 0.4 0.43 0.46 0.49 0.52 0.55 0.58 0.61 0.64 0.67 0.7 0.73 0.76 0.79 0.82 0.85 0.88 0.91 0.94 0.97 1.0)

SUBSET_SIZES=(0.000719 0.001338 0.001956 0.003194 0.003813 0.004431 0.005669 0.006288 0.006906 0.008144 0.008762 0.009381)

SUBSET_SIZES=(0.02 0.03 0.05 0.06)

SUBSET_SIZES=(0.02)


num_nodes=1
num_gpus=1
size=small

num_samples=100000
batch_size=384

data=lm1b-wrap
size_per_file=2500
work_dir=/mnt/home/phamd/diffusion-duality/
ckpt_dir="${work_dir}/model-ckpts" 
out_dir=synthetic_data_large

num_jobs=1
time_limit="36:00:00"

mkdir -p ./slurm_logs
# =================================================

for subset in "${SUBSET_SIZES[@]}"; do
    echo "Submitting job for subset: ${subset}"

    # Define the checkpoint path variable
    ckpt_path="${ckpt_dir}/lm1b-${size}/lm1b-${size}-${subset}.ckpt"

    # Check if the file exists
    if [[ -f "$ckpt_path" ]]; then
        echo "Running evaluation for subset: ${subset}"

        sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=multi-gen-${data}-${size}-${subset}
#SBATCH -o watch_folder/gens/lm1b-wrap-${size}/%x_%j.out     # output file (%j expands to jobID)
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH -t ${time_limit}              # Time limit (hh:mm:ss)
#SBATCH --cpu-freq=Performance        # Cpu frequency
#SBATCH --partition=rpi               # Request partition
#SBATCH -N ${num_nodes}                          # Total number of nodes requested
#SBATCH --ntasks-per-node=${num_gpus}           # Processes per compute node
#SBATCH --gpus-per-node=${num_gpus}
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon pre-emption
#SBATCH --array=1-${num_jobs}%1

# Load environment
# source ~/.bashrc
# conda activate my_env

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

echo "------------------------------------------------"
echo "CONDA ENVIRONMENT CHECK"
echo "Active Env: $CONDA_DEFAULT_ENV"
echo "Python Path: $(which python)"
echo "------------------------------------------------"

# --- RUN COMMAND ---
srun --export=ALL python -u -m generate_multi \\
      hydra.run.dir=${work_dir}/${out_dir}/"${data}-${size}"/${subset} \\
      mode=sample_eval \\
      data=${data} \\
      model=${size} \\
      algo=duo \\
      data.cache_dir=${work_dir}/text_data \\
      eval.checkpoint_path=${work_dir}/model-ckpts/lm1b-${size}/lm1b-${size}-${subset}.ckpt \\
      model.length=128 \\
      wandb.name=duo-lm1b-wrap-${subset}-${size} \\
      wandb.project=eval-lm1b-${size} \\
      algo.gumbel_tau_log10_start=-3.0 \\
      algo.gumbel_tau_log10_end=-3.0 \\
      algo.gamma_min=-3.5 \\
      algo.gamma_max=-1.75 \\
      algo.curriculum_start=0 \\
      algo.curriculum_end=500000 \\
      data.subset=${subset} \\
      sampling.noise_removal=greedy \\
      +eval.num_samples=${num_samples} \\
      +eval.batch_size=${batch_size} \\
      +eval.size_per_file=${size_per_file}
EOT
    else
        echo "Skipping subset ${subset}: Checkpoint not found at ${ckpt_path}"
    fi
done