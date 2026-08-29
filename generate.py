import os
import torch
import torch.distributed as dist
import hydra
import lightning as L
import pandas as pd
from datasets import Dataset
from omegaconf import OmegaConf

import algo
import dataloader
import utils

from time import time
from eval_fixed_point import BasinEvaluator


class SyntheticGenerator(BasinEvaluator):
    def __init__(self, config, checkpoint_path, device, eps: float = 1e-5):
        self.config = config
        self.device = device
        self.eps = eps
        self.logger = utils.get_logger(__name__)

        # Load Model (Reusing logic from your existing setup)
        # Assuming algo.load_model or similar logic exists as per your eval_spectral.py imports
        self.model = self._load_model(
            checkpoint_path
        )  # already loaded to device and in eval()

    @torch.no_grad()
    def generate_batch(self, batch_size, seq_len):
        """
        Runs the reverse diffusion process from t=1 to t=0 to generate samples.
        """
        x = torch.randint(
            0, self.model.vocab_size, (batch_size, seq_len), device=self.device
        )
        return self._run_reverse_process(x, 1.0, self.eps)[0]


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config):
    # Distributed Setup
    # 1. Try Standard Torchrun/Environment Variables
    if "LOCAL_RANK" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

        if local_rank == 0:
            print("Initializing DDP via torchrun environment variables.")

    # 2. Fallback to Slurm Variables (if running via 'srun python ...')
    elif "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        local_rank = int(os.environ["SLURM_LOCALID"])
        world_size = int(os.environ["SLURM_NTASKS"])

        if local_rank == 0:
            print("Initializing DDP via Slurm environment variables.")

        # We also need to explicitly set these for dist.init_process_group
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(local_rank)

        # If Master Addr is not set, assume the first node in the hostlist
        if "MASTER_ADDR" not in os.environ:
            import subprocess

            try:
                # Get the first hostname from scontrol
                cmd = "scontrol show hostnames $SLURM_JOB_NODELIST"
                stdout = subprocess.check_output(cmd, shell=True)
                master_node = stdout.decode().splitlines()[0]
                os.environ["MASTER_ADDR"] = master_node
                os.environ["MASTER_PORT"] = "29500"  # Default port
            except:
                print("Warning: Could not auto-detect MASTER_ADDR from Slurm.")
    else:
        # Single GPU / Debug Mode
        print("Single GPU / Debug Mode. DDP not initialized.")
        local_rank = rank = 0
        world_size = 1

    if torch.cuda.is_available():
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
        torch.cuda.empty_cache()
    else:
        # Fallback for debugging on CPU
        device = "cpu"

    L.seed_everything(config.seed + local_rank)  # Seed offset by rank for diversity

    # --- 2. Configuration & Initialization ---
    # Total samples desired
    total_samples = config.eval.get("num_samples", 1000)
    batch_size = config.eval.get("batch_size", 128)
    batch_size = min(batch_size, total_samples // world_size)  # Avoid oversized batches
    seq_len = config.model.length  # Assuming this exists in your data config
    
    # Calculate samples per GPU
    samples_per_gpu = total_samples // world_size
    remainder = total_samples % world_size
    if rank == world_size - 1 and remainder > 0:
        samples_per_gpu += batch_size

    eps = config.sampling.get("eps", 1e-5)
    generator = SyntheticGenerator(
        config, config.eval.checkpoint_path, device=device, eps=eps
    )
    output_dir = config.eval.get("output_dir", "synthetic_data")

    # check if directory is empty
    if os.path.exists(output_dir) and os.listdir(output_dir):
        raise ValueError(
            f"Output directory {output_dir} already exists and is not empty. Please specify an empty directory to save generated data."
        )
        dist.destroy_process_group()

    # --- 3. Generation Loop ---
    local_samples = []
    generated_count = 0

    if rank == 0:
        generator.logger.info(
            f"Starting generation of {total_samples} samples across {world_size} GPUs using batch size of {batch_size}."
        )

    iterations = (samples_per_gpu + batch_size) // batch_size

    one_time_log = True
    st = time()

    for i in range(iterations):
        if generated_count >= samples_per_gpu:
            break

        # Generate indices [Batch, Seq]
        batch_ids = generator.generate_batch(batch_size, seq_len)

        if rank == 0 and one_time_log:
            elapsed_time = time() - st  # seconds
            one_time_log = False
            generator.logger.info(
                f"Generated {batch_size} samples took {elapsed_time:.2f}s. Expected duration: {elapsed_time * (samples_per_gpu // batch_size) / 60:.2f}mins."
            )

        # Move to CPU to save memory during collection
        local_samples.append(batch_ids.cpu())
        generated_count += batch_size

    # Concatenate all local batches
    if len(local_samples) > 0:
        local_tensor = torch.cat(local_samples, dim=0)  # [N_local, Seq]
    else:
        local_tensor = torch.empty((0, seq_len), dtype=torch.long)

    # --- 4. Gather Results ---
    gathered_data = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_data, local_tensor)

    # --- 5. Save Dataset (Rank 0 only) ---
    if rank == 0:
        # 1. Flatten list of tensors
        all_tensors = [t for t in gathered_data if t is not None and t.numel() > 0]
        full_tensor = torch.cat(all_tensors, dim=0)

        # Trim to exact requested number (in case of rounding up)
        full_tensor = full_tensor[:total_samples]
        assert (
            full_tensor.shape[0] == total_samples
        ), "Mismatch in total samples after gathering."

        generator.logger.info(
            f"Generated {len(full_tensor)} samples. Saving to disk..."
        )

        # 2. Convert to list of lists (or numpy) for Dataset creation
        # We interpret these as 'input_ids'
        data_dict = {"input_ids": full_tensor.numpy()}

        # 3. Create HF Dataset
        hf_dataset = Dataset.from_dict(data_dict)

        # 4. Save to Disk
        # Define output path
        os.makedirs(output_dir, exist_ok=True)

        # You can load this later using: from datasets import load_from_disk; ds = load_from_disk('path')
        hf_dataset.save_to_disk(output_dir)
        generator.logger.info(f"Synthetic set saved to: {output_dir}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
