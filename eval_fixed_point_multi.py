import os
import torch
import hydra
import logging
import numpy as np
import pandas as pd
import lightning as L

# Repo imports
import algo
import dataloader
import utils
import math
import datasets

from time import time
import torch.distributed as dist
from eval_fixed_point import BasinEvaluator


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

    L.seed_everything(config.seed)

    ckpt = config.eval.checkpoint_path
    subset = config.data.get("subset", 1.0)
    override = config.eval.get("override", False)
    eval_mode = config.eval.get("eval_mode", "train")
    num_samples = config.eval.get("num_samples", 2000)

    eval_times = config.eval.get("eval_times", [1e-5])

    if isinstance(eval_times, (float, int)):
        eval_times = [float(eval_times)]
    elif hasattr(eval_times, "__iter__"):
        eval_times = [float(t) for t in eval_times]

    evaluator = BasinEvaluator(config, ckpt, device=device)
    num_samples, _ = evaluator.prepare_data(
        subset, local_rank, world_size, num_samples, eval_mode
    )

    for t_val in eval_times:
        csv_path = f"stability_t={t_val}_subset={subset}_{eval_mode}.csv"
        npz_path = f"stability_t={t_val}_subset={subset}_{eval_mode}.npz"

        if os.path.exists(csv_path) and os.path.exists(npz_path) and not override:
            evaluator.logger.info(
                f"Skipping evaluation: results already exist at {csv_path}"
            )
            continue

        local_df, local_tensors = evaluator.perturb_and_restore(t_val)

        # Gather DataFrames
        all_dfs = [None for _ in range(world_size)]
        if dist.is_initialized():
            dist.all_gather_object(all_dfs, local_df)
        else:
            all_dfs = [local_df]

        # Gather Tensors
        all_tensors_list = [None for _ in range(world_size)]
        if dist.is_initialized():
            dist.all_gather_object(all_tensors_list, local_tensors)
        else:
            all_tensors_list = [local_tensors]

        if rank == 0:
            # 1. CSV
            final_df = pd.concat(all_dfs, ignore_index=True)
            final_df.to_csv(csv_path, index=False)

            # 2. NPZ
            # Concatenate keys safely
            keys = ["entropy", "mask_perturbed", "mask_correct", "mask_unstable"]
            gathered_dict = {}
            for k in keys:
                t_list = [d[k] for d in all_tensors_list if k in d]
                if t_list:
                    gathered_dict[k] = torch.cat(t_list, dim=0).float().numpy()

            np.savez_compressed(npz_path, **gathered_dict)

            # ENTROPY
            avg_ent_stable = final_df["entropy_stable"].mean()
            avg_ent_unstable = final_df["entropy_unstable"].mean()
            avg_ent_recovered = final_df["entropy_recovered"].mean()
            avg_eng_failed_rec = final_df["entropy_failed_rec"].mean()

            # RATE
            avg_unstable = final_df["unstable_rate"].mean() * 100
            avg_recovery = final_df["recovery_rate"].mean() * 100
            avg_all_recovery = final_df["is_exact_match"].mean() * 100

            evaluator.logger.info(
                f"SAVED RESULTS for t={t_val} with subset={subset} and eval_mode={eval_mode}"
            )
            evaluator.logger.info("=" * 40)
            evaluator.logger.info(
                f"Entropy of Stable Tokens     : {avg_ent_stable:.6f}"
            )
            evaluator.logger.info(
                f"Entropy of Unstable Tokens   : {avg_ent_unstable:.6f}"
            )

            evaluator.logger.info(
                f"Entropy of Recovered Tokens  : {avg_ent_recovered:.6f}"
            )
            evaluator.logger.info(
                f"Entropy of Failed Rec. Tokens: {avg_eng_failed_rec:.6f}"
            )
            evaluator.logger.info(
                f"Unstable Rate (Broken Original Tokens): {avg_unstable:.3f}%"
            )
            evaluator.logger.info(
                f"Recovery Rate (for Perturbed Tokens)   : {avg_recovery:.3f}%"
            )
            evaluator.logger.info(
                f"Overall Exact Match Rate    : {avg_all_recovery:.3f}%"
            )
            evaluator.logger.info(f"Summary CSV: {csv_path}")
            evaluator.logger.info(f"Matrix NPZ : {npz_path}")
            evaluator.logger.info("=" * 40)

        # ensure all processes sync here
        dist.barrier()

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
