# Associative Memory and Language Diffusion

Code for studying **associative memory and memorization in masked / discrete diffusion language
models**. The central experiment is a two-axis sweep — **model size × training-set fraction** —
that isolates when a diffusion LM stops generalizing and starts recalling its training data.

## The sweep

Three architectures are each trained on 54 nested subsets of [LM1B](https://huggingface.co/datasets/lm1b),
from 0.01% of the corpus up to 100%, all for exactly 1,000,000 steps:

| Size | Backbone | `hidden_size` | `n_blocks` | `n_heads` | Params |
|---|---|---|---|---|---|
| `tiny`   | `ddit` | 256  | 8  | 8  | 23.7 M |
| `small`  | `ddit` | 768  | 12 | 12 | 139.3 M |
| `medium` | `ddit` | 1024 | 24 | 16 | 384.0 M |

Holding architecture and step count fixed while sweeping `data.subset` is what makes the
memorization transition measurable.

## Checkpoints

All 162 trained checkpoints are published separately (they total ~473 GB and are not in git):

**https://huggingface.co/lemoncmd/lldms-associative-memory**

```python
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="lemoncmd/lldms-associative-memory",
    filename="tiny/lm1b-tiny-0.01.ckpt",
)
```

## Setup

Requires Python 3.12 and CUDA 12.6.

```bash
conda create -n duo python=3.12
conda activate duo
pip install -r requirements.txt
# flash-attn last, from the prebuilt wheel (building from source takes ~1h)
pip install flash_attn-2.7.3+cu12torch2.6cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
```

`requirements-lock.txt` pins every transitive dependency for exact reproduction.

## Usage

Configuration is [Hydra](https://hydra.cc)-based; everything in `configs/` is overridable
from the command line.

```bash
# train the medium model on 25% of LM1B
python main.py model=medium data.subset=0.25

# evaluate conditional entropy
python eval_entropy.py model=tiny data.subset=0.01 \
    eval.checkpoint_path=/path/to/lm1b-tiny-0.01.ckpt

# generate samples
python generate.py model=small data.subset=1.0 \
    eval.checkpoint_path=/path/to/lm1b-small-1.0.ckpt
```

Multi-GPU / SLURM launchers for the full sweep live in `aimos_scripts/` and `scripts/`.

## Layout

```
main.py                 training entry point (Hydra)
algo.py                 diffusion algorithms (duo, mdlm, sedd, d3pm, ar, distillation)
trainer_base.py         Lightning module: training / eval loops
dataloader.py           LM1B + OpenWebText loading, subset sampling
models/                 ddit / dimamba backbones
metrics.py              perplexity and related metrics
eval_entropy*.py        conditional-entropy evaluation
eval_fixed_point*.py    fixed-point / stability analysis
generate*.py            sampling
configs/                Hydra configs (model, data, algo, noise, lr_scheduler, ...)
aimos_scripts/          SLURM sweep launchers
nbs/                    analysis notebooks and figures
```

## Citation

_(To be added.)_
