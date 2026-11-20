# Tree-ring learning

A transformer that grows in immutable rings and therefore cannot forget.

Each new body of experience gets a new ring of residual dimensions, attention heads and
feed-forward neurons. A ring reads everything older and writes only into itself, so every
earlier version of the model stays physically inside the current one, unchanged. A Bayesian
mixture over those past selves decides, token by token, which stage of the model's life
explains the current input best.

Full results and the argument for why this works are in [REPORT.md](REPORT.md).

## Files

| File | What it holds |
|---|---|
| `model.py` | `Ring` and the forward pass that returns every past self's prediction in one pass |
| `mixture.py` | The switching Bayesian mixture over selves, and dreaming from it |
| `learn.py` | Training one ring, scoring, pruning, and the try-a-ring-keep-it-if-it-helps loop |
| `baselines.py` | Plain fine-tuning, EWC, self-dream replay, joint training, separate experts, compaction |
| `data.py` | The skills stream and the local-text stream |
| `run.py` | Runs one stream for one seed, saving after every method, and resumes if interrupted |
| `report.py` | Turns the JSON files into tables |
| `safety.py` | Hard memory caps for MLX; imported by `model.py` so every run has them |
| `guard.py` | External kill switch that stops runs using too much memory |
| `launch.py` | Starts a command detached, so a long run survives the session that started it |
| `run_all.sh`, `run_queue.sh` | Run the seeds one at a time (never in parallel) |
| `results/` | Raw results per stream and seed, and summary tables |

## Running

Uses the existing environment in `../adaptive-model-1a/.venv` (MLX, NumPy, loguru, typer).
No data is downloaded: the text stream is built from files already on this Mac.

Run one job at a time. Uncapped parallel runs crashed the development machine twice; see
the incident section of the report.

```bash
../adaptive-model-1a/.venv/bin/python run.py --benchmark skills --seed 0
```

```bash
../adaptive-model-1a/.venv/bin/python run.py --benchmark text --seed 0
```

```bash
../adaptive-model-1a/.venv/bin/python report.py --benchmark skills
```

Every module also runs its own checks when executed directly, for example:

```bash
../adaptive-model-1a/.venv/bin/python model.py
```
