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


class BasinEvaluator:
    def __init__(self, config, checkpoint_path, device="cuda"):
        self.config = config
        self.device = device
        self.logger = utils.get_logger(__name__)
        self.model = self._load_model(checkpoint_path)
        self.tokenizer = self.model.tokenizer

    def _load_model(self, checkpoint_path):
        self.logger.info(f"Loading model from {checkpoint_path}")
        algo_name = self.config.algo.name

        if algo_name == "mdlm":
            model_cls = algo.MDLM
        elif algo_name == "duo":
            model_cls = algo.DUO
        elif algo_name == "sedd":
            model_cls = algo.SEDDAbsorb
        else:
            model_cls = algo.DUO_BASE

        tokenizer = dataloader.get_tokenizer(self.config)

        if "hf" in self.config.algo.backbone:
            model = model_cls(self.config, tokenizer=tokenizer)
        else:
            model = model_cls.load_from_checkpoint(
                checkpoint_path,
                tokenizer=tokenizer,
                config=self.config,
                map_location=self.device,
            )
        return model.eval()

    def calculate_ppl(self, texts):
        if not texts:
            return 0.0
        self.model.metrics.gen_ppl.reset()
        self.model.metrics.record_generative_perplexity(
            texts, max_length=self.config.model.length, device=self.device
        )
        return self.model.metrics.gen_ppl.compute().item()

    def prepare_data(
        self, subset_val, local_rank, world_size, num_samples=100, eval_mode="train"
    ):
        """
        Loads training data deterministically based on subset size/fraction.
        """
        # Enforce non-streaming to get tensors
        if eval_mode == "test":
            ds = dataloader.get_dataset(
                self.config.data.valid,
                self.tokenizer,
                mode="test",
                wrap=self.config.data.wrap,
                insert_eos=self.config.data.insert_train_eos,
                cache_dir=self.config.data.cache_dir,
                block_size=self.config.model.length,
                streaming=False,
                revision=self.config.data.get("train_revision", None),
            )
            self.logger.info("Evaluating on test dataset...")

        elif eval_mode == "train":
            ds = dataloader.get_dataset(
                self.config.data.train,
                self.tokenizer,
                mode="train",
                wrap=self.config.data.wrap,
                insert_eos=self.config.data.insert_train_eos,
                cache_dir=self.config.data.cache_dir,
                block_size=self.config.model.length,
                streaming=False,
                revision=self.config.data.get("train_revision", None),
            )
            self.logger.info("Evaluating on training dataset...")

        elif eval_mode == "random":
            vocab_size = self.model.vocab_size
            random_ids = torch.randint(
                0, vocab_size, (num_samples, self.config.model.length)
            )
            ds = datasets.Dataset.from_dict({"input_ids": random_ids}).with_format(
                "torch"
            )
            self.logger.info("Evaluating on random data...")
        else:
            raise ValueError("Unknown mode")

        # Determine subset size
        total = len(ds)
        ds = ds.shuffle(seed=self.config.seed)
        if eval_mode == "train":
            # Handle Fraction vs Absolute
            if subset_val <= 1.0:
                subset_count = int(total * subset_val)
            else:
                subset_count = int(subset_val)
            subset_count = min(subset_count, total)
            ds = ds.select(range(min(subset_count, len(ds))))
        else:
            subset_count = total

        # Deterministic Subset Selection
        ds = ds.select(range(min(num_samples, len(ds))))
        self.num_samples = len(ds)

        # Shard
        my_indices = np.array_split(range(self.num_samples), world_size)[local_rank]
        self.eval_subset = ds.select(my_indices)

        return self.num_samples, self.eval_subset

    def perturb_and_restore(self, start_t):
        self.logger.info(
            f"Running Basin Test | t={start_t} | Samples={self.num_samples}"
        )

        indices = range(len(self.eval_subset))
        batch_size = self.config.eval.get("batch_size", 32)
        perform_perturb = self.config.eval.get("perform_perturb", True)

        all_df = []

        # Tensors to store BxL matrices
        storage = {
            "entropy": [],  # Float [B, L]
            "mask_perturbed": [],  # Bool [B, L] (Input Noise)
            "mask_correct": [],  # Bool [B, L] (Final Correctness)
            "mask_unstable": [],  # Bool [B, L] (Unperturbed -> Incorrect)
        }

        for i in range(0, len(indices), batch_size):
            batch = self.eval_subset.select(indices[i : i + batch_size])
            x0 = batch["input_ids"].to(self.device)

            # Noise
            t_tensor = torch.ones(len(x0), 1, device=self.device) * start_t
            _, alpha_t = self.model.noise(t_tensor)

            with torch.no_grad():
                if perform_perturb:
                    xt = self.model.q_xt(x0, alpha_t)
                else:
                    xt = x0

                # Restore + Entropy
                x_recon, entropy = self._run_reverse_process(xt, start_t)

            # --- CALCULATE MASKS ---
            # 1. Where was noise applied?
            is_perturbed = x0 != xt

            # 2. Where is the output correct?
            is_correct = x0 == x_recon

            # 3. Where did we break a valid token? (Unperturbed AND Incorrect)
            is_unstable = (~is_perturbed) & (~is_correct)

            # Store Raw Matrices (Move to CPU)
            storage["entropy"].append(entropy.cpu())
            storage["mask_perturbed"].append(is_perturbed.cpu())
            storage["mask_correct"].append(is_correct.cpu())
            storage["mask_unstable"].append(is_unstable.cpu())

            # Compute Summary Stats
            df_batch = self._compare_sequences(x0, xt, x_recon, entropy)
            all_df.append(df_batch)

        # Concatenate everything
        final_df = pd.concat(all_df, ignore_index=True) if all_df else pd.DataFrame()

        final_tensors = {}
        if storage["entropy"]:
            final_tensors["entropy"] = torch.cat(storage["entropy"], dim=0)
            final_tensors["mask_perturbed"] = torch.cat(
                storage["mask_perturbed"], dim=0
            )
            final_tensors["mask_correct"] = torch.cat(storage["mask_correct"], dim=0)
            final_tensors["mask_unstable"] = torch.cat(storage["mask_unstable"], dim=0)

        return final_df, final_tensors

    def _run_reverse_process(self, xt, start_t, eps: float = 1e-5):
        """
        Custom generation loop starting from intermediate time t.
        Adapts logic from trainer_base.py generate_samples.
        """
        num_steps = self.config.sampling.steps

        # Create full schedule from 1.0 down to eps
        full_timesteps = torch.linspace(1, eps, num_steps + 1, device=self.device)
        dt = (1 - eps) / num_steps

        # Find the step index corresponding to start_t
        start_step_idx = torch.searchsorted(full_timesteps.flip(0), start_t).item()
        start_step_idx = num_steps - start_step_idx  # Adjust for flip

        # Clamp index just in case
        start_step_idx = max(0, min(start_step_idx, num_steps - 1))

        x = xt.clone()
        p_x0_cache = None

        # Run loop from start_step to end
        for i in range(start_step_idx, num_steps):
            t = full_timesteps[i] * torch.ones(x.shape[0], 1, device=self.device)

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

        # Final denoiser step (t -> 0)
        t0 = full_timesteps[-1] * torch.ones(x.shape[0], 1, device=self.device)
        if self.config.sampling.noise_removal == "ancestral":
            if self.model.sampler == "analytic":
                x = self.model._denoiser_update(x=x, t=t0)
            else:
                _, x = self.model._ancestral_update(
                    x=x, t=t0, dt=None, p_x0=p_x0_cache, noise_removal_step=True
                )
        elif self.config.sampling.noise_removal == "greedy":
            sigma = self.model._sigma_from_alphat(self.model.noise(t0)[1])
            x = self.model.forward(xt=x, sigma=sigma).argmax(dim=-1)

        sigma_final = self.model._sigma_from_alphat(self.model.noise(t0)[1])
        with torch.no_grad():
            logits = self.model.forward(xt=x, sigma=sigma_final)
            probs = logits.softmax(dim=-1)
            log_probs = logits.log_softmax(dim=-1)
            token_entropy = -torch.sum(probs * log_probs, dim=-1)  # [B, L]

        return x, token_entropy

    def _compare_sequences(self, x0, xt, x_recon, entropy):
        x0_np, xt_np, xr_np = map(lambda z: z.cpu().numpy(), (x0, xt, x_recon))
        ent_np = entropy.cpu().float().numpy()
        res = []

        skip_special_tokens = self.config.eval.get("skip_special_tokens", True)
        for i in range(len(x0_np)):
            # Decode for visual inspection
            orig_text, pert_text, recon_text = map(
                lambda z: self.tokenizer.decode(
                    z, skip_special_tokens=skip_special_tokens
                ),
                (x0_np[i], xt_np[i], xr_np[i]),
            )

            # Masks
            is_perturbed = x0_np[i] != xt_np[i]
            is_correct = x0_np[i] == xr_np[i]

            # --- Categories for Stats ---
            # 1. Recovered: Perturbed -> Fixed
            mask_recovered = is_perturbed & is_correct

            # 2. Stable: Unperturbed -> Stayed Correct
            mask_stable = (~is_perturbed) & is_correct

            # 3. Unstable: Unperturbed -> Broken (Important!)
            mask_unstable = (~is_perturbed) & (~is_correct)

            # 4. Failed Recovery: Perturbed -> Still Wrong
            mask_failed_rec = is_perturbed & (~is_correct)

            def safe_mean(mask):
                return ent_np[i][mask].mean() if mask.sum() > 0 else float("nan")

            num_unchanged = (~is_perturbed).sum().item()
            res.append(
                {
                    "original": orig_text,
                    "perturbed": pert_text,
                    "restored": recon_text,
                    "is_exact_match": np.array_equal(x0_np[i], xr_np[i]),
                    "perturbation_rate": is_perturbed.mean(),
                    "recovery_rate": (
                        (x0_np[i][is_perturbed] == xr_np[i][is_perturbed]).sum()
                        / is_perturbed.sum()
                        if is_perturbed.sum() > 0
                        else np.nan
                    ),
                    "unstable_rate": (
                        mask_unstable.sum() / num_unchanged
                        if num_unchanged > 0
                        else np.nan
                    ),
                    "entropy_recovered": safe_mean(mask_recovered),
                    "entropy_stable": safe_mean(mask_stable),
                    "entropy_unstable": safe_mean(mask_unstable),
                    "entropy_failed_rec": safe_mean(mask_failed_rec),
                }
            )
        return pd.DataFrame(res)


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

    L.seed_everything(config.seed)

    ckpt = config.eval.checkpoint_path
    subset = config.data.get("subset", 1.0)
    t_val = config.eval.get("eval_time", 0.1)
    override = config.eval.get("override", False)
    eval_mode = config.eval.get("eval_mode", "train")
    num_samples = config.eval.get("num_samples", 2000)

    csv_path = f"stability_t={t_val}_subset={subset}_{eval_mode}.csv"
    npz_path = f"stability_t={t_val}_subset={subset}_{eval_mode}.npz"

    if os.path.exists(csv_path) and os.path.exists(npz_path) and not override:
        evaluator.logger.info(
            f"Skipping evaluation: results already exist at {csv_path}"
        )
        return

    evaluator = BasinEvaluator(config, ckpt, device=device)
    num_samples, _ = evaluator.prepare_data(
        subset, local_rank, world_size, num_samples, eval_mode
    )

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
        evaluator.logger.info(f"Entropy of Stable Tokens     : {avg_ent_stable:.6f}")
        evaluator.logger.info(f"Entropy of Unstable Tokens   : {avg_ent_unstable:.6f}")

        evaluator.logger.info(f"Entropy of Recovered Tokens  : {avg_ent_recovered:.6f}")
        evaluator.logger.info(
            f"Entropy of Failed Rec. Tokens: {avg_eng_failed_rec:.6f}"
        )
        evaluator.logger.info(
            f"Unstable Rate (Broken Original Tokens): {avg_unstable:.3f}%"
        )
        evaluator.logger.info(
            f"Recovery Rate (for Perturbed Tokens)   : {avg_recovery:.3f}%"
        )
        evaluator.logger.info(f"Overall Exact Match Rate    : {avg_all_recovery:.3f}%")
        evaluator.logger.info(f"Summary CSV: {csv_path}")
        evaluator.logger.info(f"Matrix NPZ : {npz_path}")
        evaluator.logger.info("=" * 40)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
