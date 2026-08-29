import os
import torch
import torch.distributed as dist
import hydra
import lightning as L
import pandas as pd
import numpy as np
from datasets import load_from_disk
from omegaconf import OmegaConf

import algo
import dataloader
import utils

from time import time

# Import the parent class
from eval_fixed_point import BasinEvaluator


class StabilityEvaluator(BasinEvaluator):
    def __init__(self, config, checkpoint_path, device):
        self.config = config
        self.device = device
        self.logger = utils.get_logger(__name__)

        # Load the model using the helper from BasinEvaluator
        self.model = self._load_model(checkpoint_path)

        # Set tokenizer as model.tokenizer
        self.tokenizer = self.model.tokenizer

        # Ensure model is in eval mode
        self.model.eval()

    @torch.no_grad()
    def get_cond_entropy(self, x0, t_val=1e-5):
        """
        Computes H(x0 | xt) where xt ~ q(xt | x0, t=t_val).
        """
        batch_size = x0.shape[0]

        # 1. Prepare Time and Noise Schedule
        t = torch.ones(batch_size, 1, device=self.device) * t_val
        _, alpha_t = self.model.noise(t)

        # 2. Forward Process: Perturb x0 -> xt
        xt = self.model.q_xt(x0, alpha_t)

        # 3. Reverse Process: Predict p(x0 | xt)
        sigma = self.model._sigma_from_alphat(alpha_t)
        log_p_x0 = self.model.forward(xt, sigma)

        # 4. Compute Entropy: H = - sum(p * log p) using float32 for stability
        probs = log_p_x0.exp().float()
        token_entropy = -torch.sum(probs.float() * log_p_x0, dim=-1)  # [Batch, Seq]
        return token_entropy

    def run_eval_loop(self, evalset, samples_to_process, batch_size, eval_time):
        """
        Runs the entropy evaluation loop for a given dataloader.
        Handles iterator refreshing if the dataset is smaller than samples_to_process.
        """

        # Indices for tracking processed samples
        indices = range(len(evalset))
        entropies = []
        processed_count = 0

        one_time_print = True
        batch_size = min(len(evalset), batch_size)
        for i in range(0, len(indices), batch_size):
            # slice the batch indices
            batch_indices = indices[i : i + batch_size]

            # grab the batch
            batch = evalset.select(batch_indices)
            x = batch["input_ids"].to(self.device)

            current_bs = x.shape[0]

            st = time()
            h = self.get_cond_entropy(x, eval_time)

            if one_time_print:
                self.logger.info(
                    f"[Eval] Sample Entropy Computation Time per Batch Size: {current_bs}: {time()-st:.4f} secs."
                )
                one_time_print = False

            entropies.append(h.cpu())
            processed_count += current_bs

            # Break if we've processed enough samples
            if processed_count >= samples_to_process:
                break

        return entropies


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config):
    # --- 1. Distributed Setup ---
    if "LOCAL_RANK" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    elif "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        local_rank = int(os.environ["SLURM_LOCALID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(local_rank)
        if "MASTER_ADDR" not in os.environ:
            import subprocess

            try:
                cmd = "scontrol show hostnames $SLURM_JOB_NODELIST"
                stdout = subprocess.check_output(cmd, shell=True)
                os.environ["MASTER_ADDR"] = stdout.decode().splitlines()[0]
                os.environ["MASTER_PORT"] = "29500"
            except:
                pass
    else:
        local_rank = rank = 0
        world_size = 1

    if torch.cuda.is_available():
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        device = "cpu"

    L.seed_everything(config.seed + rank)

    # --- 2. Initialize Evaluator ---
    evaluator = StabilityEvaluator(config, config.eval.checkpoint_path, device)

    # --- 3. Configuration ---
    num_samples = config.eval.get("num_samples", 2_000)
    batch_size = config.eval.get("batch_size", 128)
    eval_time = config.eval.get("eval_time", 1e-5)
    data_subset = config.data.get("subset", 1.0)
    eval_mode = config.eval.get("eval_mode", "train")
    override = config.eval.get("override", False)

    synthetic_path = config.eval.get("synthetic_data_path")
    output_csv_path = f"entropy_gap_t={eval_time}.csv"
    output_npz_path = f"entropy_gap_t={eval_time}.npz"

    if os.path.exists(output_csv_path) and not override:
        if rank == 0:
            evaluator.logger.info(
                f"Output file {output_csv_path} already exists. Skipping."
            )
        dist.destroy_process_group()
        return

    if not synthetic_path:
        if rank == 0:
            evaluator.logger.info("Error: Must specify 'eval.synthetic_data_path'")
        dist.destroy_process_group()
        return

    # Calculate samples per GPU
    samples_per_gpu = num_samples // world_size
    remainder = num_samples % world_size
    if rank < remainder:
        samples_per_gpu += 1

    # --- 4. Data Loading ---
    # A. Real Data Dataloader
    _, train_dl = evaluator.prepare_data(
        data_subset, local_rank, world_size, num_samples, eval_mode
    )

    # B. Synthetic Data Dataloader
    try:
        synth_ds = load_from_disk(synthetic_path)
        synth_ds = synth_ds.with_format("torch")
    except Exception as e:
        if rank == 0:
            evaluator.logger.info(f"Error loading synthetic data: {e}")
        dist.destroy_process_group()
        return

    synth_dl = synth_ds.shard(num_shards=world_size, index=rank)
    # synth_dl = torch.utils.data.Dataset(synth_ds, batch_size=batch_size)

    # --- 5. Evaluation Execution ---
    if rank == 0:
        evaluator.logger.info(f"Evaluating Stability (Entropy Gap) at t={eval_time}...")
        evaluator.logger.info(f"Target Samples per GPU: {samples_per_gpu}")

    # Run Loop A: Real Data
    real_entropies = evaluator.run_eval_loop(
        evaluator.eval_subset, samples_per_gpu, batch_size, eval_time
    )  # Num_samples x Seq_len

    # Run Loop B: Synthetic Data
    synth_entropies = evaluator.run_eval_loop(
        synth_dl, samples_per_gpu, batch_size, eval_time
    )  # Num_samples x Seq_len

    # --- 6. Gather & Save Results ---
    local_real = torch.cat(real_entropies)
    local_synth = torch.cat(synth_entropies)

    gathered_real = [None for _ in range(world_size)]
    gathered_synth = [None for _ in range(world_size)]

    dist.all_gather_object(gathered_real, local_real)
    dist.all_gather_object(gathered_synth, local_synth)

    if rank == 0:
        # Filter None and Concatenate
        all_real = torch.cat([t for t in gathered_real if t is not None])
        all_synth = torch.cat([t for t in gathered_synth if t is not None])

        # Trim to exact num_samples requested
        all_real = all_real[:num_samples]
        all_synth = all_synth[:num_samples]

        # Log the sizes of these two tensors
        evaluator.logger.info(f"Collected Real Entropies: {all_real.shape}")
        evaluator.logger.info(f"Collected Synthetic Entropies: {all_synth.shape}")

        real_np = all_real.numpy()
        synth_np = all_synth.numpy()

        avg_real, avg_synth = map(np.mean, (real_np, synth_np))
        avg_entropy_gap = avg_synth - avg_real

        evaluator.logger.info("\n" + "=" * 40)
        evaluator.logger.info(f"RESULTS at t={eval_time} with subset={data_subset}")
        evaluator.logger.info("=" * 40)
        evaluator.logger.info(
            f"Real Data Entropy (Train):      {avg_real:.4f} +/- {np.std(real_np):.4f}"
        )
        evaluator.logger.info(
            f"Synthetic Data Entropy (Gen):   {avg_synth:.4f} +/- {np.std(synth_np):.4f}"
        )
        evaluator.logger.info("-" * 40)
        evaluator.logger.info(f"CONDITIONAL ENTROPY GAP:        {avg_entropy_gap:.4f}")
        evaluator.logger.info("=" * 40)

        my_dict = dict(
            {
                "entropy": np.concatenate(
                    [real_np.mean(1), synth_np.mean(1)]
                ),  # Average over seq. length
                "source": ["real"] * len(real_np) + ["synthetic"] * len(synth_np),
                "t": [eval_time] * (len(real_np) + len(synth_np)),
            }
        )
        df = pd.DataFrame(my_dict)
        df.to_csv(output_csv_path, index=False)
        np.savez_compressed(
            output_npz_path, real_entropy=real_np, synthetic_entropy=synth_np
        )
        evaluator.logger.info(
            f"Saved results to {output_csv_path} and {output_npz_path}."
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
