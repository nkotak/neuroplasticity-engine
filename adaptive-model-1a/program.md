# Mission

Improve Adaptive Model 1A through bounded code research and verified self-learning.

Primary objective:
- improve held-out benchmark composite and validation loss
- do not destabilize training
- do not bypass trainer, plasticity, or monitoring guardrails

Supporting docs:
- `programs/code_research.md`
- `programs/selflearn.md`
- `programs/benchmarks.md`
- `programs/backends.md`

# In-Scope Files

Only the following files and directories may be edited by autonomous code-research runs:

- `mlx_train.py`
- `mlx_model.py`
- `mlx_plasticity.py`
- `mlx_adaptive_feedforward.py`
- `mlx_data.py`
- `mlx_inference.py`
- `monitor_training.py`
- `config.py`
- `autoresearch/**`
- `selflearn/**`
- `data/benchmarks/**`
- `run_autoresearch.py`
- `run_selflearn.py`

No file outside this allowlist may be edited unless a human expands scope first.

# Protected Files

The following must never be edited by autonomous code-research runs:

- held-out benchmark gold sets
- accepted selflearn corpora
- tokenizer artifacts
- historical runs
- `program.md`

# Backends

- `mlx`: primary training backend and the only backend allowed to produce promoted checkpoints in v1
- `mps`: secondary experiment and inference backend; only use training if an explicit mirrored trainer exists
- `ane`: inference and judge backend only; never use for training

If a backend lacks a required capability, do not improvise around it.

# Experiment Profiles

- `research`: short bounded experiment
- `promotion`: full training run
- `selflearn_cycle`: generate, verify, fine-tune, benchmark, and promote or reject

# Code Research Loop

1. Run a baseline first if none exists for the current sandbox.
2. Edit only allowlisted files.
3. Commit each experiment in the sandbox.
4. Run the bounded experiment runner.
5. Benchmark the result.
6. Keep or discard the change using the promotion gate.
7. Continue autonomously until stopped or a guardrail trips.

# Self-Learning Loop

1. Benchmark the current best checkpoint.
2. Identify failure buckets.
3. Generate candidate tasks and examples.
4. Verify every candidate.
5. Write accepted JSONL only.
6. Fine-tune on accepted data.
7. Benchmark again.
8. Promote or reject.

# Promotion Rules

- No code change is promoted unless it passes the benchmark gate.
- No checkpoint is promoted unless it passes the benchmark gate.
- No self-generated sample enters training unless it is verified.
- Use `best`, not `final`, as the default promoted artifact.

# Logging Rules

- Write code-research experiment results to `results.tsv`.
- Write self-learning cycle results to `cycles.tsv`.
- Do not flood stdout; redirect long-running jobs to logs.
- Always record backend, profile, metric, status, and description.

# Never Do

- Never modify protected files.
- Never train on held-out benchmark items.
- Never use unverified self-generated data for training.
- Never disable monitoring or validation to make a run look better.
- Never stop to ask permission once an autonomous loop begins, unless a guardrail trips.
