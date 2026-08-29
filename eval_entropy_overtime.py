import os
import torch
import torch.nn.functional as F
import torch.distributed as dist
import pandas as pd
import numpy as np
import hydra
import lightning as L
from pathlib import Path
from omegaconf import OmegaConf  # Added to save config

# Repo imports based on your scripts
import algo
import dataloader
import utils
from time import time

from eval_fixed_point import BasinEvaluator


class StabilityOverTime(BasinEvaluator):
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

    def _compute_curvature_metrics(self, log_x_theta):
        """
        Computes 2nd order statistics of the Free Energy S using O(V) memory.
        Diagonal: Local Stability (Variance)
        Off-Diagonal: Interaction/Competition (Covariance Norm)
        """
        probs = log_x_theta.exp()  # [Batch, Seq, Vocab]
        V = probs.shape[-1]

        # Entropy: H(p) = -sum(p * log(p))
        # Measures the total uncertainty in the token distribution.
        entropy = -torch.sum(probs * log_x_theta, dim=-1)  # [Batch, Seq]

        # Diagonal Stability: sum p_i * (1 - p_i)
        # Represents the local curvature of the energy basin for each token.
        diag_grad = probs * (1 - probs)
        diag_stability = torch.sum(diag_grad, dim=-1)
        diag_stability_norm = (diag_grad**2).sum(dim=-1) ** 0.5

        # Off-Diagonal Interaction (Frobenius Norm)
        # Identity: sum_{i != j} (pi * pj)^2 = (sum pi^2)^2 - sum pi^4
        # We multiply by 0.5 to get the norm of just the upper triangular part.
        p_sq_sum = torch.sum(probs**2, dim=-1) ** 2
        p_quad_sum = torch.sum(probs**4, dim=-1)

        off_diag_sq_norm = 0.5 * torch.clamp(p_sq_sum - p_quad_sum, min=1e-12)
        interaction_norm = torch.sqrt(off_diag_sq_norm)

        # Interaction Sum: sum_{i != j} -pi * pj = -((sum pi)^2 - sum pi^2)
        interaction = -(torch.sum(probs, dim=-1) ** 2 - torch.sum(probs**2, dim=-1))

        # interaction mean: interaction / (V * (V - 1)) / 2 for upper triangular only due to symmetry
        interaction_mean = interaction / (V * (V - 1)) / 2

        # 4. Spectral Gap: Top-1 vs Top-2 probability
        top_probs, _ = torch.topk(probs, k=2, dim=-1)
        spectral_gap = top_probs[..., 0] - top_probs[..., 1]

        return {
            "entropy": entropy.mean().item(),
            "diag": diag_stability.mean().item(),
            "diag_norm": diag_stability_norm.mean().item(),
            "interaction": interaction_mean.mean().item(),
            "interaction_norm": interaction_norm.mean().item(),
            "spectral_gap": spectral_gap.mean().item(),
            "max_conf": top_probs[..., 0].mean().item(),
        }

    def track_trajectory(self, xt, start_t, eps=1e-5):
        num_steps = self.config.sampling.steps
        full_timesteps = torch.linspace(1.0, eps, num_steps + 1, device=self.device)
        dt = (1.0 - eps) / num_steps

        # Find starting step index
        start_step_idx = torch.searchsorted(full_timesteps.flip(0), start_t).item()
        start_step_idx = num_steps - start_step_idx

        x = xt.clone()
        local_stats = []

        for i in range(start_step_idx, num_steps):
            t_val = full_timesteps[i]
            t = t_val * torch.ones(x.shape[0], 1, device=self.device)

            with torch.no_grad():
                _, alpha_t = self.model.noise(t)
                sigma = self.model._sigma_from_alphat(alpha_t)

                # Get predictive distribution p(x0 | xt)
                log_x_theta = self.model.forward(x, sigma)

                # Compute curvature stats
                metrics = self._compute_curvature_metrics(log_x_theta)
                metrics["t"] = t_val.item()
                local_stats.append(metrics)

                # Sample x0 from p(x0 | xt)
                if self.model.sampler == "ancestral":
                    _, x = self.model._ancestral_update(x=x, t=t, dt=dt, p_x0=None)
                elif self.model.sampler == "ancestral_cache":
                    p_x0_cache, x_next = self.model._ancestral_update(
                        x=x, t=t, dt=dt, p_x0=p_x0_cache
                    )
                    if not torch.allclose(x_next, x) or self.model.time_conditioning:
                        p_x0_cache = None
                    x = x_next
                else:  # analytic
                    x = self.model._analytic_update(x=x, t=t, dt=dt)

        with torch.no_grad():
            # Final denoiser step (t -> 0)
            t0 = full_timesteps[-1] * torch.ones(x.shape[0], 1, device=self.device)
            _, alpha_t = self.model.noise(t0)
            sigma = self.model._sigma_from_alphat(alpha_t)

            # Get predictive distribution p(x0 | xt)
            log_x_theta = self.model.forward(x, sigma)

            # Compute curvature stats
            metrics = self._compute_curvature_metrics(log_x_theta)
            metrics["t"] = full_timesteps[-1].item()
            local_stats.append(metrics)

        return pd.DataFrame(local_stats)


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
    evaluator = StabilityOverTime(config, config.eval.checkpoint_path, device)

    # --- 3. Configuration ---
    num_samples = config.eval.get("num_samples", 2_000)
    batch_size = config.eval.get("batch_size", 128)
    eval_time = config.eval.get("eval_time", 1.0)
    data_subset = config.data.get("subset", 1.0)
    eval_mode = config.eval.get("eval_mode", "train")
    override = config.eval.get("override", False)

    _, eval_dl = evaluator.prepare_data(
        data_subset, local_rank, world_size, num_samples, eval_mode=eval_mode
    )
    output_csv_path = f"entropy_from_t={eval_time}_{eval_mode}.csv"

    if os.path.exists(output_csv_path) and not override:
        if rank == 0:
            evaluator.logger.info(
                f"Output file {output_csv_path} already exists. Skipping."
            )
        dist.destroy_process_group()
        return

    # Calculate samples per GPU
    samples_per_gpu = num_samples // world_size
    remainder = num_samples % world_size
    if rank < remainder:
        samples_per_gpu += 1

    local_results = []
    indices = range(len(eval_dl))
    batch_size = min(len(indices), batch_size)

    eval_dl = eval_dl.iter(batch_size=batch_size)

    start_time = eval_time
    processed_samples = 0
    display_once = True
    st = time()
    while processed_samples < samples_per_gpu:
        for batch in eval_dl:
            x0 = batch["input_ids"].to(device)

            # Handle last batch size
            if processed_samples > samples_per_gpu:
                remainder = samples_per_gpu - (processed_samples - len(x0))
                x0 = x0[:remainder]

            processed_samples += len(x0)

            # Apply perturbation
            _, alpha_t = evaluator.model.noise(
                torch.ones(x0.shape[0], 1, device=device) * start_time
            )
            xt = evaluator.model.q_xt(x0, alpha_t)

            # Track trajectory for this batch
            batch_df = evaluator.track_trajectory(xt, start_time, 1e-5)

            # Display time per batch once
            if display_once and rank == 0:
                et = (time() - st) / len(x0)
                evaluator.logger.info(
                    f"Time per batch: {et * len(x0):.4f} seconds. Expected duration: {et * num_samples / 60:.4f} mins."
                )
                display_once = False

            local_results.append(batch_df)

            if processed_samples >= samples_per_gpu:
                break

    # Combine local results
    if local_results:
        local_df = pd.concat(local_results, ignore_index=True)
    else:
        local_df = pd.DataFrame()  # empty

    # Gather results from all processes to rank 0
    gathered_dfs = [None] * world_size
    dist.all_gather_object(gathered_dfs, local_df)
    dist.barrier()

    if rank == 0:
        valid_results = [df for df in gathered_dfs if not df.empty]

        result_df = (
            pd.concat(valid_results)
            .groupby("t")
            .mean()
            .reset_index()
            .sort_values("t", ascending=False)
        )

        # --- SAVING CONFIG ---
        config_save_path = "config.yaml"
        OmegaConf.save(config, config_save_path)
        print(f"Saved config to {config_save_path}")

        # --- SAVING RESULTS & PLOTTING ---
        result_df.to_csv(output_csv_path, index=False)
        print(f"Saved results to {output_csv_path}")

        """
        import matplotlib.pyplot as plt
        # Plot Entropy Dynamics
        plt.figure(figsize=(10, 6))
        plt.plot(
            result_df["t"],
            result_df["entropy"],
            label="Entropy",
            color="green",
            lw=2,
        )
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.gca().invert_xaxis()
        plt.title(f"Subset: {data_subset}")
        plt.xlabel("Time (t)", fontsize=14)
        plt.ylabel("Entropy", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"entropy_t={eval_time}_{eval_mode}.png")
        plt.close()
        """

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
