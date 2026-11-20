# Continual growth experiment: design

Date: 2026-11-19
Status: awaiting approval. Nothing in this document has been built.

## Intent

The project's thesis: today's models do not keep improving after training the way a brain does. The aim is to show, at small scale, that a model can keep learning new things by adding and pruning neurons, and that its size increases as it learns more.

The existing code never tested this. It trains once on a fixed blend of text, adds two neurons every 3,000 steps on a timer, and holds sparsity at 40%. No new material ever arrives, so nothing measures whether old knowledge survives new learning.

This experiment tests the thesis directly: a tiny transformer learns a sequence of tasks one after another, grows when a task needs capacity, and is scored on every task it has seen so far.

## Goals

1. Measure whether a model that grows on demand keeps old tasks while learning new ones.
2. Measure whether its size tracks how much it has learned.
3. Compare against fixed-size models given the same task sequence, including ones with the same final size.
4. Produce a result that can be published as-is, whether it supports the thesis or not.

## Non-goals

- Natural language, the 23M-parameter model, and `mlx_train.py`.
- The fixed-sparsity controller in `mlx_plasticity.py`, and neuron split and merge.
- The self-learning and autoresearch loops.
- Growing attention heads or layers.
- Weight-regularisation methods such as EWC, and remedies for loss of plasticity in fixed-size networks.
- Any claim of novelty for the method. Freezing old weights and training new feed-forward capacity is published work (see "What the research says").

## Anti-goals

These would make the result worthless even if the numbers looked good:

- Showing the growing model any data from an earlier task during a later task.
- Telling the model which task it is doing by any route other than the prompt itself.
- Choosing any setting by looking at the scored runs.
- Reporting only the seeds that pass.

## What the existing code allows

Verified by reading the code on 2026-11-19, and confirmed by an independent reviewer:

- `MLXAdaptiveTransformer` (`mlx_model.py`) accepts a different feed-forward width per layer.
- `MLXAdaptiveFeedForward.add_neurons` (`mlx_adaptive_feedforward.py:259`) appends neurons with random input weights and output weights near zero, so adding them leaves the model's output almost unchanged. `remove_neuron` (line 350) deletes one.
- Each feed-forward block keeps a running average of every neuron's activation (`_persistent_mean_activation`), usable for pruning decisions.
- `_convergence_stats` (`mlx_plasticity.py:454`) has a slope-based stall test that can be copied.

Constraints this places on the design:

- **Only feed-forward neurons can grow.** Attention cannot. New tasks must therefore need new stored facts, not new ways of moving information between positions.
- **MLX cannot freeze individual rows of a weight matrix,** and AdamW's weight decay touches every weight. Freezing has to be enforced explicitly.
- **The embedding and output matrices are not tied,** despite the comment at `mlx_model.py:176`. The experiment treats them as separate.
- **Neither virtual environment has pytest.** It must be installed before tests can run.

## Approaches considered

1. **Freeze and grow, one model, no task switch (chosen).** After each task, every parameter that task used is frozen. New tasks get new neurons, which are the only things that train. All neurons stay active for every input. Forgetting is not guaranteed to be zero, because new neurons can fire on old-task inputs, so it is a real measurement.
2. **Per-task neuron masks selected by the task tag.** At inference, only the neurons belonging to the prompt's task are switched on. Forgetting is zero by construction, which makes the main result a foregone conclusion and hard-codes the routing instead of learning it. Kept as the follow-up if approach 1 fails its forgetting test.
3. **Everything stays trainable, with a penalty for changing important weights, plus growth.** Closest to how a brain is thought to work, but it has more settings to tune and a failure would be hard to attribute.

Approach 1 was chosen because it is the simplest design whose outcome is not known in advance.

## What the research says

From the literature review in `reports/Continual learning and growing networks.md`. Paper details below are as reported by that review and have not been re-checked against the PDFs.

- **Closest precedent.** "Grow, Don't Overwrite" (arXiv:2603.08647) widens the feed-forward layers of a 1B-parameter model and trains only the new weights, reporting old-task retention within 2%. Approach 1 is the same idea at tiny scale, with on-demand sizing and pruning added.
- **Growing versus starting large.** The evidence is mixed and leans against growth: two 2026 studies (arXiv:2609.27291, arXiv:2605.15435) found grown networks no better than same-size networks trained from scratch. The review found no published test in a transformer of growth against a same-size model given the same protection. Arm E is that test, and is the part of this experiment most likely to add something new.
- **Duplicated neurons do not diverge.** Joudaki et al. (arXiv:2510.00304) show copies of a neuron stay locked together under training. New neurons here are freshly initialised, not split from old ones.
- **Growth triggers.** NORACL (arXiv:2604.27031) found triggers based on loss or on a timer at least 1.9% worse than triggers based on saturation measures. The stall trigger here is the simple option and is known to be the weaker one.
- **Tuning leaks.** A large replication study (arXiv:2403.09066) found most continual-learning methods failed to reproduce once settings were tuned on different data from the evaluation. Hence the pilot-then-lock protocol below.
- **Copying within the prompt can pass for memory.** Hence inputs within an example never repeat.

## Design

### Scenario

Each prompt names its task with a tag. This is the easiest continual-learning setting, and results here say nothing about settings where the model must work out the task for itself.

### Task stream

- **Vocabulary:** 32 data symbols, one tag per task (8 in the main experiment, 52 in the long-stream run), and one separator. No tokenizer is needed; symbols are integer ids.
- **A task** is a tag plus a fixed random lookup table from some of the symbols to symbols. The number of symbols it covers is the number of facts in the task.
- **An example** is `TAG x1 … x8 = y1 … y8`, where each `y` is the table's entry for the matching `x`. The eight inputs are distinct, so no answer can be copied from earlier in the same example.
- **Base phase:** tasks 0 and 1, each a full reordering of all 32 symbols. Every symbol is therefore seen as both input and output before anything is frozen, and the two tables disagree on almost every input, which forces the model to use the tag.
- **Continual phase:** tasks 2 to 7, one at a time, covering random subsets of 8, 32, 16, 32, 8 and 24 symbols. The order of these six sizes is shuffled per seed, because task order is known to change forgetting.

Lookup tables are a deliberate choice: the amount learned is countable in facts, which is what "size increases as it learns more" needs. The limit is that this tests storing knowledge, not acquiring new skills.

### Model

`MLXAdaptiveTransformer` with `d_model` 64, 2 layers, 4 heads, no dropout. The starting feed-forward width is set in the pilot to the smallest value at which the base phase reaches 99% (default 64 per layer), and is the same for every arm.

### Arms

| Arm | Size | Training | Purpose |
|---|---|---|---|
| A. Grow | Starting width, grows on demand, pruned after each task | Sequential, old parameters frozen | The method under test |
| B. Fixed small | Starting width | Sequential, all trainable | Shows ordinary forgetting |
| C. Fixed large | Arm A's final width | Sequential, all trainable | Rules out "it just had more parameters" |
| D. Joint | Arm A's final width | All eight tasks mixed | Upper bound |
| E. Pre-allocated | Reaches arm A's final width in six equal blocks | Identical to A, except each new task gets one fixed block, with no on-demand growth and no pruning | Separates sizing on demand from freezing |
| F. Replay | Arm A's final width | Sequential, all trainable, 5% of each batch drawn from earlier tasks | Reference: the standard cheap fix |

Rules common to all arms:

- A task ends when its accuracy has been at least 99% for three evaluations in a row, or at the step cap. Arm D ends when every task meets that bar, or at seven times the cap.
- "Arm A's final width" is taken per seed from that seed's arm A run.
- Ten scored seeds. A seed fixes the lookup tables, the task order and the model's starting weights. Three seeds would settle A against B and C, where the gap is tens of points, but not A against E, where published gaps are about one point.

### Arm A procedure

1. **Base phase.** Train all parameters on tasks 0 and 1 mixed.
2. **Freeze everything.**
3. **For each new task:**
   - Make the new tag's embedding row trainable.
   - Add 4 neurons per layer. These and the tag row are the only trainable values.
   - Every 100 steps, evaluate on every task seen so far. If the current task is under 99% and its loss has stalled over the last five evaluations, add 4 more neurons per layer, up to 128 new neurons per layer per task. The stall window restarts after each growth event.
   - **Prune** once the task ends. Rank this task's new neurons by average activation times output-weight size. Remove the weakest while the task's accuracy stays within half a point of its level before pruning.
   - **Freeze** the survivors and record the model's size.

Freezing must be exact: a frozen value is bit-identical before and after any later task. How that is enforced (restoring frozen values after each update, or masking gradients with weight decay off) is left to the builder. A test asserts it.

The optimizer is recreated after each growth event. `_restore_optimizer_state` (`mlx_train.py:304`) is the upgrade if lost momentum turns out to matter.

### Protocol

1. **Pilot.** All settings are chosen on a development stream whose seed differs from the ten scored seeds. Each arm picks its learning rate from the same small grid by its own final average accuracy. Arm A's own settings (stall threshold, growth increment, prune tolerance, starting output weights of new neurons) are chosen here too.
2. **Lock.** Settings are written to a file and not changed afterwards.
3. **Score.** Each arm runs once per scored seed. A run is repeated only for a crash or a code defect, and the repeat is reported.

### Metrics

Accuracy means fact accuracy: the share of output positions answered correctly over 512 fresh examples. Because inputs are distinct and drawn evenly, this estimates the share of a task's facts the model holds. Whole-example accuracy is reported alongside but is not used in any criterion.

With `a[t][j]` as accuracy on task `j` after finishing task `t`, and `T` as the last task:

- **Final average accuracy:** the mean of `a[T][j]` over all tasks.
- **Learning accuracy:** `a[t][t]` for each new task.
- **Forgetting:** for each earlier task, its best accuracy at any earlier task boundary minus `a[T][j]`, averaged.
- **Worst dip:** the lowest accuracy any earlier task reaches at any evaluation during later training.
- **Size:** parameter count and neurons per layer after each task, neurons added and kept per task, and steps used per task.
- **Marginal cost:** neurons kept per fact and steps per fact for each task, in stream order. In a model that only ever adds fresh neurons, a loss of the ability to learn cannot show up as dead neurons; it shows up as this cost rising along the stream.

### Pass criteria

Fixed before any run. Each must hold for the mean over the ten scored seeds, and the number of individual seeds that meet it is reported.

1. Arm A learning accuracy is at least 95% on every new task.
2. Arm A forgetting is at most 2 points and final average accuracy is at least 95%.
3. Arms B and C each forget at least 30 points. If they do not, the stream is too easy and the experiment is invalid, not passed.
4. Arm A's parameter count rises after every task, and 32-fact tasks keep more neurons than 8-fact tasks.

Arms D, E and F are reported but are not pass criteria. A against E is reported as three separate numbers with confidence intervals: accuracy, steps used, and neurons allocated but unused. The stated expectation for E is that equal blocks under-serve the 32-fact tasks and over-serve the 8-fact ones; given the research above, E matching A is a plausible and publishable outcome.

The thesis is not supported at this scale if criterion 1 or 2 fails after the one permitted fix below.

Passing supports one claim only: structure alone, with no old data, is enough to keep old tasks in this setting. It does not show that growth beats replay (arm F is a reference line), and low forgetting is evidence for freezing, not for growth. What growth itself contributes is read from A against E and from criterion 4.

### Long-stream run

Reported, not a pass criterion. After the main experiment, arms A and E run once per seed on a stream of 50 continual tasks of 8 facts each, with the settings already locked. The output is the marginal-cost curve across the stream. A flat curve is evidence the model keeps its ability to learn; a rising one shows the frozen base becoming a poorer foundation. This is the only part of the experiment that addresses "keeps learning" as opposed to "does not forget".

### Known risks and the one permitted fix each

- **New neurons disturb old tasks.** Nothing during a new task penalises a new neuron for firing on old-task inputs. If criterion 2 fails, add a penalty on the new neurons' activation size, which needs no old data, tune it on the development stream, and report results with and without it. If it still fails, report the negative result.
- **Frozen attention cannot read a new tag.** If criterion 1 fails, start each new tag's embedding at the average of the existing tags and widen the base phase to three tasks. Report that this was done.

## Not measured

- Loss of the ability to learn in the fixed-size arms. The long-stream run covers arms A and E only.
- Whether the baselines' forgetting is deep or shallow, for example how fast they relearn an old task.
- Learning new skills, as opposed to new facts.
- Better growth triggers than a loss stall, other pruning scores, and other ways of starting new neurons.
- Distillation and EWC as further reference lines.

## Deliverables

- `continual/tasks.py`, `continual/growth.py`, `continual/experiment.py`, and `run_continual.py`, following the existing `selflearn/` and `run_selflearn.py` layout.
- `tests/test_continual.py` covering: adding neurons leaves outputs almost unchanged; frozen values are bit-identical across a task; the task generator is deterministic for a seed and never repeats an input within an example; the metrics are correct on a hand-made accuracy table.
- `runs/continual/<timestamp>/results.json` with every arm's accuracy table, sizes, steps, locked settings and seeds.
- Three plots: accuracy per task across the stream for each arm; parameter count across the stream; neurons kept against facts per task.
- A README section with the results table and a plain conclusion, whichever way it goes.

## Left to the builder

Batch size, step cap, the learning-rate grid, whether neurons are added to every layer or one, the exact split of code across files, and plot styling. Any change to the pass criteria, the protocol or the anti-goals needs the project owner's approval.

## Calls made by the author

The project owner set the thesis and asked for this spec. Everything else is the author's proposal, in particular:

- Approach 1 over approaches 2 and 3.
- Lookup tables as the task family, and the task sizes.
- Fact accuracy as the scored measure, and the thresholds 95%, 2 points and 30 points.
- Arms E and F, ten seeds, shuffled task order and the long-stream run, all added because of the literature review.
- One shared stopping rule for all arms.
- The pilot, lock and single scored run protocol.
- Leaving out distillation and EWC reference arms, which the review recommends, to keep the first version small.
