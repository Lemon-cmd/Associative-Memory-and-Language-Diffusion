#!/bin/bash

# ================= CONFIGURATION =================
#SUBSET_SIZES=(0.0001 0.002575 0.00505 0.007525 0.01 0.04 0.07 0.1 0.13 0.16 0.19 0.22 0.25 0.28 0.31 0.34 0.37 0.4 0.43 0.46 0.49 0.52 0.55 0.58 0.61 0.64 0.67 0.7 0.73 0.76 0.79 0.82 0.85 0.88 0.91 0.94 0.97 1.0)

SUBSET_SIZES=(0.25 0.4 0.73 0.79 0.94)


num_nodes=1
num_gpus=6
size=small
num_samples=20000
data=lm1b-wrap
work_dir=/gpfs/u/home/HPDM/HPDMphmb/scratch/diffusion-duality

# --- Dynamic Batch Size Logic ---
if [[ "$size" == "small" || "$size" == "tiny" ]]; then
    batch_size=128
else
    batch_size=128
fi

mkdir -p ./slurm_logs
# =================================================

echo "Using batch size: ${batch_size}"
for subset in "${SUBSET_SIZES[@]}"; do
    # Define the checkpoint path variable
    ckpt_path="${work_dir}/model-ckpts/lm1b-${size}/lm1b-${size}-${subset}.ckpt"

    # Check if the file exists
    if [[ -f "$ckpt_path" ]]; then
        echo "Submitting job for subset: ${subset}"

        sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=gen-${data}-${size}-${subset}
#SBATCH --error=watch_folder/errs/${data}-${size}/%x_%j.err
#SBATCH --output=watch_folder/gens/${data}-${size}/%x_%j.out
#SBATCH --nodes=${num_nodes}
#SBATCH --gres=nvme,gpu:32g:${num_gpus}
#SBATCH --ntasks-per-node=${num_gpus}
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --partition=dcs-2024

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

srun --export=ALL python -u -m generate \\
      hydra.run.dir=${work_dir}/synthetic_data/"${data}-${size}"/${subset} \\
      mode=sample_eval \\
      data=${data} \\
      model=${size} \\
      algo=duo \\
      data.cache_dir=${work_dir}/text_data \\
      eval.checkpoint_path=${ckpt_path} \\
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
      +eval.batch_size=${batch_size}
EOT
    else
        echo "Skipping subset ${subset}: Checkpoint not found at ${ckpt_path}"
    fi
done