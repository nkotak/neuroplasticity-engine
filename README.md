# Neuroplasticity Engine

An experiment in whether a small neural network can keep learning the way a brain does: by
adding and pruning neurons, growing as it learns, and not forgetting what it knew.

**Short answer:** the hypothesis holds at small scale, but the method that made it hold is
not new, and the harder goal behind it (one self-improving model) was not reached.

## Hypothesis

Today's language models stop learning once training ends. They do not improve themselves the
way a brain does, through structural change. The hypothesis was that this can be shown at
small scale:

> A small model can learn continuously by adding and pruning neurons, and its size will
> increase as it learns more.

## What was tried

| Stage | Where | Outcome |
|---|---|---|
| 1. A language model that prunes and regrows connections while training | `adaptive-model-1a/` | Did not test the hypothesis: it trained once on fixed data, so nothing new ever arrived. The model also never learned to produce usable text. |
| 2. Exact bookkeeping: add silent neurons, solve the output weights exactly | `adaptive-model-1a/spikes/exact_plasticity/` | Forgetting eliminated exactly, but only for looking up facts on top of frozen features. Brittle to noisy inputs. |
| 3. Tree-ring learning: grow a frozen "ring" per body of material, choose between past selves by likelihood | `treering/` | The final experiment. Results below. |

A literature review written along the way is in
`adaptive-model-1a/reports/Continual learning and growing networks.md`.

## Results

A small transformer learned seven things in sequence. Final accuracy across all seven, and
how many points of earlier accuracy were lost along the way:

| Method | Final accuracy | Forgotten | Seeds |
|---|---|---|---|
| Joint training on everything at once (upper bound) | 100% | n/a | 1 |
| Separate frozen experts, chosen by likelihood | 99.1% | 0.0 | 1 |
| **Tree-ring (this project)** | **97.6%** | **0.0** | 3 |
| Fixed-size model rehearsing its own samples | 92.9% | 8.3 | 2 |
| EWC | 42.3% | 67.2 | 2 |
| Plain fine-tuning | 37.4% | 73.0 | 2 |

The same ordering held on real text (Python source, manual pages, a word list, C headers,
JSON). The run was stopped early, so some rows rest on one seed. Full tables, method and
checks are in [`treering/REPORT.md`](treering/REPORT.md).

## Final reasoning

**What the evidence supports**

- **A growing model did not forget.** Zero points were forgotten across seven stages, on
  synthetic skills and on real text, with no stored data and no task labels at test time.
  Fixed-size models forgot between 8 and 73 points.
- **Size followed novelty.** The model added capacity for each new kind of material and added
  none when shown material it already knew, in all five runs.
- **Pruning worked where there was slack.** On the skills stream 25% to 75% of each new
  block's feed-forward neurons were removed without loss. On real text none could be.
- **The protection is provable.** Every earlier version of the model stays bit-for-bit
  identical inside the grown one, and a mathematical bound on forgetting held on every
  sequence tested.

**What it does not support**

- **This is not a new solution.** Freezing old parts and adding new ones dates to 2016.
  Choosing between frozen experts by how well each explains the input has been published
  several times, including with the same kind of guarantee. This project's own addition,
  letting new parts read old ones, made early learning faster but did not improve final
  accuracy: unconnected experts did as well or slightly better.
- **Growth was not what prevented forgetting.** Freezing plus a rule for choosing between
  stages did. With that rule switched off, the same grown model forgot 50 points.
- **It is not one self-improving model.** It is a set of frozen stages with a chooser. Three
  attempts to merge them into a single model without old data fell short (at best 92% on
  skills, and failure on real text). On sequences mixing material from different stages it
  scored 82.5%, against 99.0% for joint training.

**Why the harder goal failed**

Protecting old knowledge turned out to be easy: do not overwrite it. Integrating new knowledge
with old, without seeing the old data again, is the real problem. Every attempt at it relied
on the model regenerating its own past, and the result was only as good as that regeneration.
A model this small regenerates simple synthetic tasks well and real text badly.

**Conclusion**

At small scale a model can grow as it learns and lose nothing. That part of the hypothesis is
confirmed. But the mechanism is established work, the growth itself is not what does the
protecting, and the result is an expanding collection of frozen stages, not a brain-like
system that reorganises what it knows. The open problem is integration, and nothing here
solves it.

## Running it

Everything uses the Python environment in `adaptive-model-1a/` (MLX on Apple Silicon).

```bash
cd treering
../adaptive-model-1a/.venv/bin/python run.py --benchmark skills --seed 0
```

```bash
../adaptive-model-1a/.venv/bin/python report.py --benchmark skills
```

**Memory safety.** Uncapped MLX runs crashed the development machine twice (details in the
report). `treering/safety.py` now caps memory for every run. Do not run several at once.

## Layout

| Path | Contents |
|---|---|
| `treering/` | Final experiment: code, raw results, full report |
| `adaptive-model-1a/` | Original prototype (stage 1), the stage 2 spike, the literature review, and an earlier experiment design in `docs/` |
