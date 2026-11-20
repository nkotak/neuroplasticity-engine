# Adaptive Model 1A: the original prototype

**Status: superseded and kept for the record.** This was the first stage of the project. It
did not test the project's hypothesis, and the model it trained never produced usable text.
The experiment that did test the hypothesis is in [`../treering/`](../treering/); the overall
story is in the [top-level README](../README.md).

## What it is

A language model built on MLX (Apple Silicon) whose feed-forward layers change shape while it
trains:

- connections are pruned and regrown towards a fixed 40% sparsity;
- neurons are split, merged, and added on a timer;
- a controller pauses these changes when validation loss gets worse.

The model has 8 layers, 256 dimensions, a 32,000-token vocabulary and about 23 million
parameters. Two outer loops were built around it and are described in `program.md`: one in
which an agent edits the training code and keeps changes that pass a benchmark, and one in
which a larger teacher model writes practice tasks that are verified before being used for
fine-tuning.

## What happened

- The last training run was stopped at step 33,482 of 60,000. Its best validation loss was
  5.81 (perplexity 333), and training loss had been flat since about step 12,000.
- The model scored 0 of 12 on its own benchmark. Its output was repeated fragments.
- In that run the structure barely changed: 2 splits, 1 merge and 16 added neurons, taking
  the widest layer from 1,024 to 1,031 neurons.
- The only logged self-learning cycle was rejected by its own promotion gate.

## Why it did not test the hypothesis

The hypothesis is about learning new things over time without forgetting old ones. This
prototype trained once on a fixed blend of data, so nothing new ever arrived. Growth ran on a
timer and sparsity was pinned, so size could not follow what was learned. Nothing measured
whether earlier knowledge survived.

## Known problems

- The embedding and output matrices are meant to be tied (`mlx_model.py`, `_tie_weights`) but
  drift apart in training; saved checkpoints show them as different.
- The trainer has no option to switch the structural changes off, so there is no baseline.
- "Head pruning" masks slices of the combined attention output, not individual heads.
- The PyTorch half cannot run: `main.py`, `adaptive_network.py`, `inference.py`,
  `transformer_model/` and six of the nine test files import `torch`, which is not installed
  in either environment. `pytest` is not installed either.
- The data settings in `config.py` point at a folder on the original machine.

## Layout

| Path | Contents |
|---|---|
| `mlx_model.py`, `mlx_adaptive_feedforward.py`, `mlx_plasticity.py` | The model and the controller that prunes, regrows, splits and merges |
| `mlx_train.py`, `mlx_data.py`, `mlx_inference.py`, `config.py` | Training, data, generation and settings |
| `autoresearch/`, `selflearn/`, `programs/`, `program.md` | The two outer loops and their rules |
| `spikes/exact_plasticity/` | Stage 2 of the project: exact bookkeeping with silent neurons |
| `reports/`, `research_notes/` | Literature review on continual learning and growing networks |
| `docs/superpowers/specs/` | An experiment design written before the final approach replaced it |
| `main.py`, `adaptive_network.py`, `inference.py`, `transformer_model/` | Earlier PyTorch version; does not run |

Not in the repository: trained weights, run logs, the tokenizer files, and the 29 MB training
file `data/reasoning_50k/reasoning_data.jsonl` (regenerate it with
`generate_reasoning_data.py`).

## Running it

The trainer has a 50-step smoke-test mode:

```bash
./venv/bin/python mlx_train.py --quick
```

It was not re-run for this write-up, and it needs the data sources named in `config.py`,
which are not all in this repository.

This code predates the memory caps in `../treering/safety.py`. Run one job at a time and
watch memory if you start a full training run.
