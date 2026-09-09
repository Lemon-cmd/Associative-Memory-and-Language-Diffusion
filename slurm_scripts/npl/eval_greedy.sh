#!/bin/bash

# Define your list of subset sizes here
SUBSET_SIZES=(0.0001 0.002575 0.00505 0.007525 0.01 0.04 0.07 0.1 0.13 0.16 0.19 0.22 0.25 0.28 0.31 0.34 0.37 0.4 0.43 0.46 0.49 0.52 0.55 0.58 0.61 0.64 0.67 0.7 0.73 0.76 0.79 0.82 0.85 0.88 0.91 0.94 0.97 1.0)

# Define your list of eval times here
TIMES=(1.0)

# Convert Bash Array to Hydra-formatted String "[t1,t2,t3]"
# and join the array with commas
printf -v TIMES_STR "%s," "${TIMES[@]}"
TIMES_STR="[${TIMES_STR%,}]" # Remove trailing comma and add brackets

# Other constants
size=small
batch_size=128

num_samples=5000
data=lm1b-wrap

work_dir="${WORK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
ckpt_dir="${work_dir}/model-ckpts" 
result_name="stability_greedy"

eval_mode=random
override=false

# CHANGE THESE SLURM VARIABLES APPROPRIATELY
num_nodes=1
num_gpus=6
num_jobs=1

eval_ppl=false
perform_perturb=false

skip_special_tokens=false
deterministic_posterior=true

for subset in "${SUBSET_SIZES[@]}"; do
    # Define the checkpoint path variable
    ckpt_path="${ckpt_dir}/lm1b-${size}/lm1b-${size}-${subset}.ckpt"

    # Check if the file exists
    if [[ -f "$ckpt_path" ]]; then
        echo "Running evaluation for subset: ${subset}"

        sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=stability-${data}-${size}-${subset}
#SBATCH --error=watch_folder/dcs-errs/${result_name}/${data}-${size}/%x_%j.err
#SBATCH --output=watch_folder/dcs-outs/${result_name}/${data}-${size}/%x_%j.out
#SBATCH --nodes=${num_nodes}
#SBATCH --gres=nvme,gpu:32g:${num_gpus}
#SBATCH --ntasks-per-node=${num_gpus}
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=1-${num_jobs}%1
#SBATCH --time=06:00:00  
#SBATCH --partition=npl-2024

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

srun --export=ALL python -u -m eval_fixed_point_multi \
      hydra.run.dir=${work_dir}/experiments/${result_name}/"${data}-${size}"/${subset} \
      mode=sample_eval \
      data=${data} \
      model=${size} \
      algo=duo \
      data.cache_dir=${work_dir}/text_data \
      eval.checkpoint_path=${ckpt_path} \
      model.length=128 \
      wandb.name=duo-lm1b-wrap-${subset}-${size} \
      wandb.project=eval-lm1b-${size} \
      algo.gumbel_tau_log10_start=-3.0 \
      algo.gumbel_tau_log10_end=-3.0 \
      algo.gamma_min=-3.5 \
      algo.gamma_max=-1.75 \
      algo.curriculum_start=0 \
      algo.curriculum_end=500000 \
      data.subset=${subset} \
      +eval.eval_mode=${eval_mode} \
      +eval.num_samples=${num_samples} \
      +eval.batch_size=${batch_size} \
      +eval.eval_times=${TIMES_STR} \
      +eval.override=${override} \
      +eval.eval_ppl=${eval_ppl} \
      +eval.skip_special_tokens=${skip_special_tokens} \
      +eval.perform_perturb=${perform_perturb} \
      +sampling.deterministic_posterior=${deterministic_posterior}
EOT
    else
        echo "Skipping subset ${subset}: Checkpoint not found at ${ckpt_path}"
    fi
done