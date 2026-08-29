import os
import torch
import torch.distributed as dist
import hydra
import lightning as L
import pandas as pd
import glob
import uuid
import time
from datasets import Dataset
import pyarrow.parquet as pq

import algo
import utils
from eval_fixed_point import BasinEvaluator


class SyntheticGenerator(BasinEvaluator):
    def __init__(self, config, checkpoint_path, device, eps: float = 1e-5):
        self.config = config
        self.device = device
        self.eps = eps
        self.logger = utils.get_logger(__name__)
        self.model = self._load_model(checkpoint_path)

    @torch.no_grad()
    def generate_batch(self, batch_size, seq_len):
        x = torch.randint(
            0, self.model.vocab_size, (batch_size, seq_len), device=self.device
        )
        return self._run_reverse_process(x, 1.0, self.eps)[0]


def get_existing_count(output_dir):
    """
    Efficiently counts total rows across all parquet files in the directory.
    Uses metadata reading to avoid loading actual data.
    """
    if not os.path.exists(output_dir):
        return 0

    parquet_files = glob.glob(os.path.join(output_dir, "*.parquet"))
    if not parquet_files:
        return 0

    total_rows = 0
    for f in parquet_files:
        try:
            # Only read metadata (footer), extremely fast
            meta = pq.read_metadata(f)
            total_rows += meta.num_rows
        except Exception:
            # If a file is corrupt (job crashed mid-write), ignore or warn
            print(f"Warning: Could not read metadata for {f}. Removing it...")
            # remove the file
            os.remove(f)
            pass

    return total_rows


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
    else:
        # Fallback for debugging on CPU
        device = "cpu"

    # --- 2. Configuration ---
    target_total_samples = config.eval.get("num_samples", 200_000)
    batch_size = config.eval.get("batch_size", 64)
    output_dir = config.eval.get("output_dir", "synthetic_data")

    # NEW: Safety constraints
    # How often to save to disk (prevent OOM)
    size_per_file = config.eval.get("size_per_file", 5000)
    # Job time limit in minutes (e.g., 6 hours = 360 mins). Set buffer (e.g. 350).
    time_limit_mins = config.eval.get("time_limit_mins", 350)
    start_time = time.time()

    # --- 3. Resume Logic (Prevent Over-Generation) ---
    # Only Rank 0 checks the file system to avoid hammering IO
    existing_samples = torch.tensor(0, device=device)

    if rank == 0:
        count = get_existing_count(output_dir)  # Now safe to delete corrupt files
        print(f"Found {count} existing samples in {output_dir}.")
        existing_samples += count

    # Broadcast the official count to all ranks
    if world_size > 1:
        dist.broadcast(existing_samples, src=0)

    L.seed_everything(config.seed + rank + existing_samples.item())
    samples_needed = target_total_samples - existing_samples.item()
    if samples_needed <= 0:
        if rank == 0:
            print("Target sample count reached. Exiting.")
        dist.destroy_process_group()
        return

    # Distribute remaining work
    my_quota = samples_needed // world_size
    remainder = samples_needed % world_size

    if rank < remainder:
        # distribute work evenly among first 'remainder' ranks
        my_quota += 1

    if rank == 0:
        print(f"Generating {samples_needed} more samples. Quota per GPU: ~{my_quota}")
        os.makedirs(output_dir, exist_ok=True)

    # --- 4. Generation Loop ---
    generator = SyntheticGenerator(
        config,
        config.eval.checkpoint_path,
        device=device,
        eps=config.sampling.get("eps", 1e-5),
    )

    seq_len = config.model.length
    generated_count = 0
    buffer = []

    # We use a unique ID for this run to avoid filename collisions if jobs overlap
    run_id = str(uuid.uuid4())[:8]

    while generated_count < my_quota:
        # A. Check Time Limit
        elapsed_mins = (time.time() - start_time) / 60
        if elapsed_mins >= time_limit_mins:
            generator.logger.info(
                f"[Rank {rank}] Time limit reached ({elapsed_mins:.1f}m). Saving buffer and stopping."
            )
            break

        # B. Generate
        # Clip batch size to not exceed quota
        current_bs = min(batch_size, my_quota - generated_count)
        batch_ids = generator.generate_batch(current_bs, seq_len)

        # C. Buffer
        buffer.append(batch_ids.cpu())
        generated_count += current_bs

        # D. Incremental Save (Flush Buffer)
        # Check if buffer has enough samples to write a file
        current_buffer_size = sum([t.shape[0] for t in buffer])

        if current_buffer_size >= size_per_file or generated_count >= my_quota:
            # Consolidate buffer
            full_tensor = torch.cat(buffer, dim=0)

            # Create a filename: rank_timestamp_chunk.parquet
            # Using timestamp ensures uniqueness across resume runs
            fname = f"rank{rank}_{run_id}_{generated_count}.parquet"
            fpath = os.path.join(output_dir, fname)

            # Save using Pandas/PyArrow (Efficient)
            df_list = pd.DataFrame({"input_ids": list(full_tensor.numpy())})
            df_list.to_parquet(fpath, index=False)

            if rank == 0:
                generator.logger.info(f"Saved {fpath} ({current_buffer_size} samples)")

            # Clear buffer
            buffer = []

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
