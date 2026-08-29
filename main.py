import json
import os

import fsspec
import hydra
import lightning as L
import omegaconf
import rich.syntax
import rich.tree
import torch

import algo
import dataloader
import utils

omegaconf.OmegaConf.register_new_resolver("cwd", os.getcwd)
omegaconf.OmegaConf.register_new_resolver("device_count", torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver("eval", eval)
omegaconf.OmegaConf.register_new_resolver("div_up", lambda x, y: (x + y - 1) // y)


def _load_from_checkpoint(diffusion_model, config, tokenizer):
    if "hf" in config.algo.backbone:
        return diffusion_model(config, tokenizer=tokenizer).to("cuda")

    return diffusion_model.load_from_checkpoint(
        config.eval.checkpoint_path, tokenizer=tokenizer, config=config
    )


@L.pytorch.utilities.rank_zero_only
def _print_config(
    config: omegaconf.DictConfig, resolve: bool = True, save_cfg: bool = True
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
      config (DictConfig): Configuration composed by Hydra.
      resolve (bool): Whether to resolve reference fields of DictConfig.
      save_cfg (bool): Whether to save the configuration tree to a file.
    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    fields = config.keys()
    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, omegaconf.DictConfig):
            branch_content = omegaconf.OmegaConf.to_yaml(
                config_section, resolve=resolve
            )

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))
    rich.print(tree)
    if save_cfg:
        with fsspec.open(
            "{}/config_tree.txt".format(config.checkpointing.save_dir), "w"
        ) as fp:
            rich.print(tree, file=fp)


@L.pytorch.utilities.rank_zero_only
def _print_batch(train_ds, valid_ds, tokenizer, k=64):
    for dl_type, dl in [("train", train_ds), ("valid", valid_ds)]:
        print(f"Printing {dl_type} dataloader batch.")
        batch = next(iter(dl))
        print("Batch input_ids.shape", batch["input_ids"].shape)
        first = batch["input_ids"][0, :k]
        last = batch["input_ids"][0, -k:]
        print(f"First {k} tokens:", tokenizer.decode(first))
        print("ids:", first)
        print(f"Last {k} tokens:", tokenizer.decode(last))
        print("ids:", last)


def _generate_samples(diffusion_model, config, logger, tokenizer):
    logger.info("Starting Sample Eval.")
    model = _load_from_checkpoint(
        diffusion_model=diffusion_model, config=config, tokenizer=tokenizer
    )
    model.metrics.gen_ppl.reset()
    model.metrics.sample_entropy.reset()
    if config.eval.disable_ema:
        logger.info("Disabling EMA.")
        model.ema = None
    stride_length = config.sampling.stride_length
    num_strides = config.sampling.num_strides
    all_samples = []
    for _ in range(config.sampling.num_sample_batches):
        if config.sampling.semi_ar:
            _, intermediate_samples, _ = model.restore_model_and_semi_ar_sample(
                stride_length=stride_length,
                num_strides=num_strides,
                dt=1 / config.sampling.steps,
            )
            text_samples = intermediate_samples[-1]
            # Note: Samples generated using semi-ar method
            # need to to be processed before computing generative perplexity
            # since these samples contain numerous <|endoftext|> tokens
            # and diffusion.compute_generative_perplexity() discards
            # any text after the first EOS token.
        else:
            samples = model.restore_model_and_sample(num_steps=config.sampling.steps)
            model.metrics.record_entropy(samples)
            text_samples = model.tokenizer.batch_decode(samples)
            model.metrics.record_generative_perplexity(
                text_samples, config.model.length, model.device
            )
            all_samples.extend(list(text_samples))
    generative_ppl = 0.0
    entropy = 0.0
    if not config.sampling.semi_ar:
        generative_ppl = model.metrics.gen_ppl.compute().item()
        entropy = model.metrics.sample_entropy.compute().item()
        print("Generative perplexity:", generative_ppl)
        print("Sample entropy:", entropy)
    samples_path = config.eval.generated_samples_path
    with fsspec.open(samples_path, "w") as f:
        json.dump(
            {
                "generative_ppl": generative_ppl,
                "entropy": entropy,
                "generated_seqs": all_samples,
            },
            f,
            indent=4,
        )
    print("Samples saved at:", samples_path)


def _eval_ppl(diffusion_model, config, logger, tokenizer):
    logger.info("Starting Perplexity Eval.")

    model = _load_from_checkpoint(
        diffusion_model=diffusion_model, config=config, tokenizer=tokenizer
    )
    if config.eval.disable_ema:
        logger.info("Disabling EMA.")
        model.ema = None

    wandb_logger = None
    if config.get("wandb", None) is not None:
        wandb_logger = L.pytorch.loggers.WandbLogger(
            config=omegaconf.OmegaConf.to_object(config), **config.wandb
        )
    callbacks = []
    if "callbacks" in config:
        for _, callback in config.callbacks.items():
            callbacks.append(hydra.utils.instantiate(callback))
    trainer = hydra.utils.instantiate(
        config.trainer,
        default_root_dir=os.getcwd(),
        callbacks=callbacks,
        strategy=hydra.utils.instantiate(config.strategy),
        logger=wandb_logger,
    )
    _, valid_ds = dataloader.get_dataloaders(
        config, tokenizer, skip_train=True, valid_seed=config.seed
    )
    trainer.validate(model, valid_ds)


def _train(diffusion_model, config, logger, tokenizer):
    logger.info("Starting Training.")
    wandb_logger = None

    if config.get("wandb", None) is not None:
        # 1. Convert wandb config to a mutable dict to avoid OmegaConf/argument collisions
        wandb_args = omegaconf.OmegaConf.to_container(config.wandb, resolve=True)

        # 2. Define path to store the WandB Run ID (using fsspec for S3/remote compatibility)
        wandb_id_path = "{}/wandb_run_id.txt".format(config.checkpointing.save_dir)

        # 3. If restarting, check if an ID exists
        if utils.fsspec_exists(wandb_id_path):
            with fsspec.open(wandb_id_path, "r") as f:
                existing_id = f.read().strip()
            logger.info(f"Resuming WandB run with ID: {existing_id}")

            # Override the ID and set resume to allow
            wandb_args["id"] = existing_id
            wandb_args["resume"] = "allow"

        # 4. Initialize Logger using the modified args
        wandb_logger = L.pytorch.loggers.WandbLogger(
            config=omegaconf.OmegaConf.to_object(config), **wandb_args
        )

        # 5. Save the ID immediately if it's a new run
        if not utils.fsspec_exists(wandb_id_path):
            # Accessing .experiment.id creates the run if it hasn't started
            generated_id = wandb_logger.experiment.id

            if type(generated_id) != str:
                generated_id = wandb_logger.version

            with fsspec.open(wandb_id_path, "w") as f:
                f.write(generated_id)
            logger.info(f"Created new WandB run with ID: {generated_id}")

    if (
        config.checkpointing.resume_from_ckpt
        and config.checkpointing.resume_ckpt_path is not None
        and utils.fsspec_exists(config.checkpointing.resume_ckpt_path)
    ):
        ckpt_path = config.checkpointing.resume_ckpt_path
    else:
        ckpt_path = None

    # Lightning callbacks
    callbacks = []
    if "callbacks" in config:
        for _, callback in config.callbacks.items():
            callbacks.append(hydra.utils.instantiate(callback))

    train_ds, valid_ds = dataloader.get_dataloaders(config, tokenizer)
    _print_batch(train_ds, valid_ds, tokenizer)

    # --- AUTO-DISABLE VALIDATION LOGIC ---
    # If the dataset is smaller than the requested validation interval,
    # disable validation to prevent PyTorch Lightning crash.
    val_interval = config.trainer.get("val_check_interval", 1)

    # Only check if interval is an integer (step-based)
    if isinstance(val_interval, int) and hasattr(train_ds, "__len__"):
        # Get the number of nodes requested in the config
        num_nodes = config.trainer.get("num_nodes", 1)

        # Get the number of GPUs available on the current node
        devices_per_node = torch.cuda.device_count()

        # Calculate total ranks (World Size)
        total_ranks = num_nodes * devices_per_node
        num_batches_per_rank = len(train_ds) // total_ranks

        if val_interval >= num_batches_per_rank:
            logger.warning(
                f"val_check_interval ({val_interval}) is larger than total training batches ({num_batches_per_rank}). "
                "Disabling validation (limit_val_batches=0.0) to avoid crash."
            )
            # Disable validation
            omegaconf.OmegaConf.update(config, "trainer.limit_val_batches", 0)
            # Set interval to 1.0 (end of epoch) to satisfy Lightning's strict config check
            # Even though validation is off, Lightning validates this arg before checking limit_val_batches
            omegaconf.OmegaConf.update(config, "trainer.val_check_interval", 1)
    # -------------------------------------

    if config.training.finetune_path != "":
        assert utils.fsspec_exists(config.training.finetune_path)
        model = diffusion_model.load_from_checkpoint(
            config.training.finetune_path, tokenizer=tokenizer, config=config
        )
    else:
        model = diffusion_model(config, tokenizer=valid_ds.tokenizer)

    # Ensure log_every_n_steps is set for logging frequency
    # config.yaml has this set to 100, which is good.
    # This safety check ensures it defaults to 50 if removed from yaml.
    if "log_every_n_steps" not in config.trainer:
        logger.info("Setting log_every_n_steps to 50.")
        omegaconf.OmegaConf.update(config, "trainer.log_every_n_steps", 50)

    trainer = hydra.utils.instantiate(
        config.trainer,
        default_root_dir=os.getcwd(),
        callbacks=callbacks,
        strategy=hydra.utils.instantiate(config.strategy),
        logger=wandb_logger,
    )
    trainer.fit(model, train_ds, valid_ds, ckpt_path=ckpt_path)


'''
def _train(diffusion_model, config, logger, tokenizer):
    logger.info("Starting Training.")
    wandb_logger = None
    if config.get("wandb", None) is not None:
        wandb_logger = L.pytorch.loggers.WandbLogger(
            config=omegaconf.OmegaConf.to_object(config), **config.wandb
        )

    if (
        config.checkpointing.resume_from_ckpt
        and config.checkpointing.resume_ckpt_path is not None
        and utils.fsspec_exists(config.checkpointing.resume_ckpt_path)
    ):
        ckpt_path = config.checkpointing.resume_ckpt_path
    else:
        ckpt_path = None

    # Lightning callbacks
    callbacks = []
    if "callbacks" in config:
        for _, callback in config.callbacks.items():
            callbacks.append(hydra.utils.instantiate(callback))

    train_ds, valid_ds = dataloader.get_dataloaders(config, tokenizer)
    _print_batch(train_ds, valid_ds, tokenizer)

    # --- AUTO-DISABLE VALIDATION LOGIC ---
    # If the dataset is smaller than the requested validation interval,
    # disable validation to prevent PyTorch Lightning crash.
    val_interval = config.trainer.get("val_check_interval", 1)

    # Only check if interval is an integer (step-based)
    if isinstance(val_interval, int) and hasattr(train_ds, "__len__"):
        # Get the number of nodes requested in the config
        num_nodes = config.trainer.get("num_nodes", 1)

        # Get the number of GPUs available on the current node
        devices_per_node = torch.cuda.device_count()

        # Calculate total ranks (World Size)
        total_ranks = num_nodes * devices_per_node
        num_batches_per_rank = len(train_ds) // total_ranks

        if val_interval >= num_batches_per_rank:
            logger.warning(
                f"val_check_interval ({val_interval}) is larger than total training batches ({num_batches_per_rank}). "
                "Disabling validation (limit_val_batches=0.0) to avoid crash."
            )
            # Disable validation
            omegaconf.OmegaConf.update(config, "trainer.limit_val_batches", 0)
            # Set interval to 1.0 (end of epoch) to satisfy Lightning's strict config check
            # Even though validation is off, Lightning validates this arg before checking limit_val_batches
            omegaconf.OmegaConf.update(config, "trainer.val_check_interval", 1)
    # -------------------------------------

    if config.training.finetune_path != "":
        assert utils.fsspec_exists(config.training.finetune_path)
        model = diffusion_model.load_from_checkpoint(
            config.training.finetune_path, tokenizer=tokenizer, config=config
        )
    else:
        model = diffusion_model(config, tokenizer=valid_ds.tokenizer)

    trainer = hydra.utils.instantiate(
        config.trainer,
        default_root_dir=os.getcwd(),
        callbacks=callbacks,
        strategy=hydra.utils.instantiate(config.strategy),
        logger=wandb_logger,
    )
    trainer.fit(model, train_ds, valid_ds, ckpt_path=ckpt_path)
'''

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config):
    """Main entry point for training."""
    L.seed_everything(config.seed)
    _print_config(config, resolve=True, save_cfg=True)

    logger = utils.get_logger(__name__)
    tokenizer = dataloader.get_tokenizer(config)
    if config.algo.name == "ar":
        diffusion_model = algo.AR
    elif config.algo.name == "mdlm":
        diffusion_model = algo.MDLM
    elif config.algo.name == "duo_base":
        diffusion_model = algo.DUO_BASE
    elif config.algo.name == "d3pm":
        diffusion_model = algo.D3PMAbsorb
    elif config.algo.name == "sedd":
        diffusion_model = algo.SEDDAbsorb
    elif config.algo.name == "duo":
        diffusion_model = algo.DUO
    elif config.algo.name == "distillation":
        diffusion_model = algo.Distillation
    elif config.algo.name == "ot-finetune":
        diffusion_model = algo.OptimalTransportFinetune
    else:
        raise ValueError(f"Invalid algorithm name: {config.algo.name}")
    kwargs = {
        "diffusion_model": diffusion_model,
        "config": config,
        "tokenizer": tokenizer,
        "logger": logger,
    }
    if config.mode == "sample_eval":
        _generate_samples(**kwargs)
    elif config.mode == "ppl_eval":
        _eval_ppl(**kwargs)
    else:
        _train(**kwargs)


if __name__ == "__main__":
    main()
