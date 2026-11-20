# Tree-ring learning: full write-up

Date: 2026-11-20. Hardware: one MacBook Pro (M2 Max, 96 GB). Framework: MLX.

## Summary

A small transformer was taught seven things one after another. With the method described
here it forgot nothing: final accuracy on the skills stream was 97.6% with 0.0 points
forgotten, against 37% for plain fine-tuning (73 points forgotten), 42% for EWC and 93% for
a fixed-size model that replays its own generated samples. The same pattern held on real
text. The method needs no stored data and no task labels at test time, and it comes with a
mathematical bound on forgetting that held on every sequence tested.

Three things did not work out, and they matter more than the headline:

1. **The part of the design that is original adds no accuracy.** New parts of the network
   were allowed to read older parts ("nesting"). Unconnected experts combined by the same
   rule scored 99.1%, slightly higher. Nesting only made early learning faster.
2. **No single integrated model was produced.** The method works as a collection of frozen
   stages plus a rule for choosing between them. Three attempts to merge the stages into one
   model without old data reached at best 92% on skills and failed on real text.
3. **It is not a new solution.** Freezing old parts, adding new ones, and choosing between
   experts by likelihood are all published ideas. The exact combination used here did not
   turn up in a literature search, but result 1 says the new part is not the part that works.

Joint training on all data at once, which is not allowed in continual learning, scored 100%.

## The idea

- **Rings.** Each new body of material gets a new block of the network, called a ring: new
  residual dimensions, attention heads and feed-forward neurons. A ring reads everything
  older (through a small linear adapter) and writes only into its own dimensions.
- **Selves.** Because nothing ever writes into an older ring, the model made of rings 1 to k
  computes exactly the same function forever. Each such prefix is called a self. One forward
  pass of the full model yields the predictions of every past self.
- **Mixture.** At each token the system predicts with a weighted average of all selves. A
  self's weight is the probability that it is the right one, given how well it has predicted
  the sequence so far. A "switching" variant lets the weights move mid-sequence.
- **Growth by need.** A new ring is trained on new material and kept only if it lowers the
  loss on that material by at least 0.02 nats per token. Its feed-forward neurons are pruned
  before it is frozen.

## Why it cannot forget

For every sequence `x`, every self `k`, with `K` selves and `A` switching rates:

```
-log p_mixture(x)  <=  -log p_k(x) + log K + log A
```

The mixture is a weighted sum of the selves' probabilities, so it is at least `1/(K A)` times
any one of them. Self `k` never changes, so the loss on old material can never rise by more
than `log(K A)` nats per sequence, whatever is learned later. With 6 selves, 3 rates and
95 predicted tokens that is 0.030 nats per token.

## Setup

**Model.** Three layers. First ring: 96 dimensions, 4 heads, 384 feed-forward neurons. Later
rings: 48 dimensions, 3 heads, 192 feed-forward neurons, adapter width 48. Final size was
about 785,000 parameters on skills and 1.03 million on text.

**Skills stream.** Character-level problems such as `rev:abc=cba;`. Seven eras: copy and
reverse; sort; shift and double; two-number addition; compositions (sort descending, sort
then shift); a revisit that mixes reverse and sort; an exact repeat of the first era.
Accuracy is the share of problems answered entirely correctly.

**Text stream.** Real text already on the machine, no downloads: Python standard-library
source, Unix manual pages, a dictionary word list, C headers, question-answer JSON from this
project, then Python again. Scored in nats per character (lower is better).

**Budget.** 3,000 steps for the first ring, 2,000 plus 150 pruning steps for each later one,
batch 64. Every baseline gets the same steps per era and a matched parameter count.

**Methods compared.**

| Method | What it is |
|---|---|
| Tree-ring (mixture) | The method above |
| Separate experts + mixture | One unconnected expert per era, same sizes, same mixture |
| Fixed size + own dreams | One fixed-size model that rehearses samples it generates itself |
| EWC | One fixed-size model with a penalty for changing important weights |
| Plain fine-tuning | One fixed-size model, trained on each era in turn |
| Joint training | One fixed-size model trained on all eras at once (upper bound) |

**Seeds.** The run was stopped early, so seed counts differ by row and are shown in every
table. Rows with one seed are indicative, not established.

## Results

### Skills stream

| Method | Seeds | Final accuracy | Accuracy forgotten |
|---|---|---|---|
| Joint training (upper bound) | 1 | 100% | n/a |
| Separate experts + mixture | 1 | 99.1% | -0.1 |
| **Tree-ring (mixture)** | 3 | 97.6% ± 0.4 | 0.0 ± 0.0 |
| Fixed size + own dreams | 2 | 92.9% ± 0.3 | 8.3 ± 0.3 |
| EWC (best of two strengths) | 2 | 42.3% ± 2.2 | 67.2 ± 2.5 |
| Plain fine-tuning | 2 | 37.4% ± 0.0 | 73.0 ± 0.0 |
| Tree-ring, newest self only | 3 | 52.3% ± 2.4 | 50.4 ± 1.1 |

The last row is the tree-ring model with the mixture switched off. It forgets badly, which
shows that the mixture, not the growing, is what protects old skills.

Per-era accuracy for the tree-ring mixture: 99.6, 98.0, 99.0, 94.7, 94.8, 97.6, 99.6.
The two weakest eras (addition and compositions) are learning shortfalls, not forgetting.

### Real-text stream

| Method | Seeds | Final loss (nats/char) | Loss forgotten |
|---|---|---|---|
| Joint training (upper bound) | 1 | 0.935 | n/a |
| Separate experts + mixture | 1 | 1.010 | +0.008 |
| **Tree-ring (mixture)** | 2 | 1.015 ± 0.000 | +0.004 ± 0.000 |
| Fixed size + own dreams | 1 | 1.488 | +0.672 |
| EWC | 1 | 1.591 | +0.250 |
| Plain fine-tuning | 1 | 2.185 | +1.508 |
| Tree-ring, newest self only | 2 | 5.595 ± 0.149 | +4.704 ± 0.168 |

### Growth tracked novelty

| Era (skills) | Loss before | Loss after | Rings kept | Parameters added |
|---|---|---|---|---|
| copy+reverse | n/a | 1.068 | 1 | 202,016 |
| sort | 5.759 | 1.041 | 1 | 98,000 |
| shift+double | 6.177 | 0.911 | 1 | 107,264 |
| addition | 10.923 | 0.722 | 1 | 144,368 |
| compositions | 4.251 | 1.104 | 1 | 125,792 |
| mixed revisit | 1.216 | 1.100 | 1 | 107,216 |
| exact repeat | 1.096 | 1.096 | 0 | 0 |

- **Repeats add nothing.** The exact-repeat era added no ring in any of the five runs (three
  skills seeds, two text seeds).
- **Pruning.** On skills, pruning removed 25% to 75% of each ring's feed-forward neurons. On
  text it removed none, because any removal raised the loss past the tolerance.
- **A flaw in the keep rule.** The mixed-revisit ring was kept because it lowered the loss,
  yet accuracy on that era was 97.6% before it and 97.6% after. It bought likelihood, not
  correctness, for 107,000 parameters.

### The guarantee, checked

- **Past selves are bit-identical.** The largest change in any past self's output after later
  growth was exactly 0.0, in every run and every variant.
- **Rollback is exact.** Removing the newest rings reproduced the earlier results with a
  difference of exactly 0.0.
- **The bound held.** The worst gap between the mixture and the best single self on any test
  sequence was 2.703 nats on skills (bound 2.890) and 2.585 on text (bound 2.708).
- **Measured cost of the bound.** Loss on the first era rose from 1.067 to 1.096 by the end
  of the skills stream, an increase of 0.028 against a bound of 0.030.
- **The mixture finds the right self.** On each era the ring that learned it held 94% to 100%
  of the weight on skills, and 77% to 99% on text.

### Does nesting help?

One new ring on the frozen model against a stand-alone model with the same number of new
parameters, both learning the compositions era (one seed):

| Training steps | Ring accuracy | Stand-alone accuracy |
|---|---|---|
| 200 | 28.9% | 0.0% |
| 500 | 71.6% | 39.9% |
| 1,200 | 93.9% | 99.4% |

Reading older rings makes a new ring learn faster at first and finish lower. Combined with
the separate-experts row above, nesting is a speed-up, not an accuracy gain.

### One integrated model: three attempts

The aim was a single self that holds everything, trained partly on "dreams": sequences
sampled from the model's own past selves, with no stored data.

| Attempt | Seeds | Newest self alone, skills | Newest self alone, text |
|---|---|---|---|
| New ring learns old skills from dreams | 1 | 89.2% | not run |
| New ring starts as a copy of the previous self and learns a correction | 2 | 91.7% ± 1.7 | 2.808 (failed) |
| Same, plus rows spliced from a dream and new data | 1 | 74.0% | not run |
| Fresh half-size model trained only on dreams | 1 | 64.0% | not run |

None matched the mixture (97.6%), and none beat the fixed-size dream baseline (92.9%). On
real text the single self failed outright. Dreams on skills were 93% correct; dreams of real
text from a model this small are poor, and learning from them does not work. Splicing made
both this method and the fixed-size baseline worse (the baseline fell to 80.0% from 92.6% on
the same seed).

### Sequences that mix eras

Rows mixing all eight skills, which no method was trained on:

| Method | Seeds | Accuracy |
|---|---|---|
| Joint training (upper bound) | 1 | 99.0% |
| Tree-ring (switching mixture) | 3 | 82.5% ± 0.3 |
| Separate experts + mixture | 1 | 79.7% |
| Fixed size + own dreams | 2 | 75.5% ± 3.6 |
| Tree-ring (plain Bayes mixture) | 3 | 44.3% ± 1.2 |
| Tree-ring, newest self only | 3 | 27.1% ± 3.2 |

Switching is essential here (82.5% against 44.3% without it), but a 16-point gap to joint
training remains. Each frozen self was trained only on its own era and is unreliable when
unfamiliar material sits earlier in the same sequence.

## Is it new?

No. The working parts are published:

- Freezing old columns and adding new ones that read them: Progressive Neural Networks
  (Rusu et al., 2016, [arXiv:1606.04671](https://arxiv.org/abs/1606.04671)).
- Mixing frozen model snapshots with Bayesian weights and a regret bound: the Forget-Me-Not
  Process (Milan et al., 2016,
  [NeurIPS](https://proceedings.neurips.cc/paper/6055-the-forget-me-not-process.pdf)).
- Per-domain experts in a language model, mixed by a posterior over domains, with experts
  added without forgetting: DEMix (Gururangan et al., 2021,
  [arXiv:2108.05036](https://arxiv.org/abs/2108.05036)).
- Experts each paired with a density model for task-free selection: TAME
  ([arXiv:2210.03869](https://arxiv.org/pdf/2210.03869)); also Expert Gate (2017) and
  CN-DPM (2020), cited from memory.
- Switching between experts mid-sequence: fixed share (Herbster and Warmuth, 1998, cited
  from memory; see also [arXiv:1008.4532](https://arxiv.org/abs/1008.4532)).
- A language model rehearsing its own generated samples: LAMOL (2020), cited from memory.

Seven targeted searches did not find this exact combination: nested selves all read out in
one forward pass, a switching mixture with a stated bound, and exact rollback. That is weak
evidence of novelty, and the experiments above show the nested part contributes speed only.

## Limits

- Tiny models (under 1.1 million parameters) and short streams (six or seven eras).
- Era boundaries were given during training. Only inference was free of task labels.
- The model grows with every new era and frozen rings can never be pruned or merged.
- Several rows rest on one seed.
- The guarantee is on the likelihood of whole sequences, not on answer accuracy directly.

## Incident: two machine crashes

On 2026-11-19 and 2026-11-20 these experiments crashed the machine twice. MLX keeps freed GPU
memory in a cache whose default limit on this machine is 91 GB per process. The routine that
generates dreams changed its array sizes at every step and filled that cache, and several
runs were going at once. The kernel's panic reports show the Python processes at 68 to 79 GB
each on a 96 GB machine.

Fixes, all in this directory:

- `safety.py` caps MLX at 12 GB of working memory and 1 GB of cache, and stops the process
  above 16 GB. It is imported by `model.py`, so no run can start without it.
- The dream routine now keeps every array the same size. The workload that reached 79 GB
  peaks at 1.4 GB.
- `guard.py` is an external kill switch (20 GB per run, or system free memory under 25%).
- Runs go one at a time and save after every method.

After the fix the longest run held between 0.5 and 1.6 GB and the guard never fired.

## Reproducing

```bash
cd treering
../adaptive-model-1a/.venv/bin/python run.py --benchmark skills --seed 0
```

```bash
../adaptive-model-1a/.venv/bin/python report.py --benchmark skills
```

Each module runs its own checks when executed directly, for example `python model.py`.
Raw results are in `results/`. The text stream reads
`../adaptive-model-1a/data/reasoning_50k/reasoning_data.jsonl`, which is not in the
repository; regenerate it with `../adaptive-model-1a/generate_reasoning_data.py`, or the
stream runs without that domain.

## What remains

- Run the missing seeds so every row has at least three.
- The open problem is integration: one model that holds every stage's knowledge without old
  data. In these experiments its quality tracked how faithfully the model could regenerate
  its own past, which a model this small does poorly on real text.
