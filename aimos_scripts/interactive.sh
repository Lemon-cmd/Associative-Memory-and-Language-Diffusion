#!/bin/bash

# --- 1. Test Configuration ---
subset=0.06
batch_size=128
size=small
num_gpus=1              # Start with 1 GPU for testing
gpu_type="nvidia_h100"  # Updated for your H200 nodes
num_nodes=1
time_limit="00:30:00"   # Short time for quick testing
cache_dir=/mnt/home/phamd/diffusion-duality/text_data/

# --- 2. Request the Interactive Node ---
# This command will stop here and wait for the allocation. 
# Once granted, it executes the commands below ON the compute node.
srun -p rpi \
     -N ${num_nodes} \
     --gres=gpu:${gpu_type}:${num_gpus} \
     -t ${time_limit} \
     --pty bash -c "
        # --- 3. Environment Setup (Inside the node) ---
        export WANDB_MODE=offline
        export WANDB_DISABLE_UPDATE_CHECK=true
        export NO_PROXY='localhost,127.0.0.1'
        export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

        # Initialize Conda (Use the full path to conda if 'conda activate' fails)
        source \$(conda info --base)/etc/profile.d/conda.sh
        conda activate duo

        # --- 4. Execute Main (with debug flags) ---
        python -u -m main \
          loader.batch_size=${batch_size} \
          loader.eval_batch_size=64 \
          data=lm1b-wrap \
          data.cache_dir=${cache_dir} \
          wandb.name=test-interactive-${subset} \
          wandb.project=test-debug \
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
          trainer.max_steps=5 \
          data.subset=${subset}
"