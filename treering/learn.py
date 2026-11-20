"""Training, evaluation, pruning and the grow-a-ring-per-era procedure.

Purpose
    train_ring   trains one ring (or one plain model) with every older ring frozen.
    evaluate     scores the Bayesian mixture and every self on an era.
    prune_ring   removes feed-forward neurons from the newest ring before it is frozen.
    learn_era    the tree-ring procedure: try a new ring, keep it only if it helps.

Docs
    MLX optimizers: https://ml-explore.github.io/mlx/build/html/python/optimizers.html

Sample input
    info = learn_era(rings, era, Config(), np.random.default_rng(0))

Expected output
    rings gains zero or more frozen Ring objects; info reports steps used, rings kept,
    parameters added and the era loss before and after.
"""
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from loguru import logger

import safety

from data import BOS, Batch, Era
from mixture import ALPHAS, dream, log_softmax, switching_logprobs
from model import Ring, clone_ring, count_params, lineage_forward, ring_forward

Forward = Callable[[mx.array], List[mx.array]]


@dataclass
class Config:
    vocab: int = 128
    length: int = 96
    n_layers: int = 3
    base_delta: int = 96
    base_heads: int = 4
    base_ff: int = 384
    ring_delta: int = 48
    ring_heads: int = 3
    ring_ff: int = 192
    steps_base: int = 3000
    steps_ring: int = 2000
    lr: float = 2e-3
    batch: int = 64
    weight_decay: float = 0.01
    max_rings_per_era: int = 1
    min_gain: float = 0.02          # nats per token a new ring must save to be kept
    eval_batches: int = 3
    eval_batch: int = 128
    consolidate: bool = False       # also learn from dreams of all past selves
    residual: bool = False          # a new ring outputs a correction added to the previous self
    splice: bool = False            # also train on rows spliced from a dream and new data
    n_dreams: int = 512
    dream_weight: float = 1.0
    prune: bool = True
    prune_tol: float = 0.01
    prune_steps: int = 150


def teacher_targets(teacher: Sequence[Ring], dreams: mx.array, chunk: int = 256) -> mx.array:
    """Mixture log-probabilities of a frozen teacher lineage at every dream position."""
    out = []
    for i in range(0, dreams.shape[0], chunk):
        d = dreams[i:i + chunk]
        lp = switching_logprobs(lineage_forward(teacher, d[:, :-1]), d[:, 1:])[0]
        mx.eval(lp)
        out.append(lp)
        safety.check("while scoring dreams")
    return mx.concatenate(out)


def make_splice(teacher: Sequence[Ring], dreams: mx.array, era: Era, cfg: Config, rng: np.random.Generator):
    """Rows that are part dream (old material) and part new-era data, joined at a random point.

    Every other training row comes from a single era, which lets a model decide "old or new"
    once per sequence. Spliced rows force that decision to be made locally.
    Returns (tokens, teacher log-probs, mask): mask is 1 where the next token is new-era data
    (learn it from the data) and 0 where it is dream (match the frozen teacher).
    """
    n, length = dreams.shape
    new = era.sample(n, rng).tokens
    cut = rng.integers(length // 4, 3 * length // 4, size=(n, 1))
    dream_first = rng.integers(0, 2, size=(n, 1)).astype(bool)
    is_new = (np.arange(length)[None, :] >= cut) == dream_first        # True where the token is new-era data
    tokens = mx.array(np.where(is_new, new, np.array(dreams)).astype(np.int32))
    return tokens, teacher_targets(teacher, tokens), mx.array(is_new[:, 1:].astype(np.float32))


def train_ring(frozen: Sequence[Ring], ring: Ring, sample: Callable[[int, np.random.Generator], Batch],
               steps: int, cfg: Config, rng: np.random.Generator, dreams: Optional[mx.array] = None,
               dream_lp: Optional[mx.array] = None, penalty: Optional[Callable[[Ring], mx.array]] = None,
               lr: Optional[float] = None, penalty_state: Optional[list] = None,
               splice: Optional[tuple] = None) -> float:
    """Train `ring` only; older rings are frozen and their activations are computed once per batch.

    If dreams are given, the ring is also trained to match `dream_lp`, the teacher's
    log-probabilities on those dreams.
    """
    lr = lr or cfg.lr
    warm = max(1, steps // 20)
    sched = optim.join_schedules([optim.linear_schedule(0.0, lr, warm), optim.cosine_decay(lr, max(1, steps - warm))], [warm])
    opt = optim.AdamW(learning_rate=sched, weight_decay=cfg.weight_decay)
    frozen = list(frozen)
    has_trace, has_dream, has_splice = bool(frozen), dreams is not None, splice is not None
    trace_fn = mx.compile(lambda t: lineage_forward(frozen, t, which=[], return_trace=True)[1]) if has_trace else None

    def loss_fn(*args: mx.array) -> mx.array:
        it = iter(args)
        tokens = next(it)
        loss = nn.losses.cross_entropy(ring_forward(ring, tokens[:, :-1], next(it) if has_trace else None),
                                       tokens[:, 1:], reduction="mean")
        if has_dream:
            d_tokens = next(it)
            s_lp = log_softmax(ring_forward(ring, d_tokens[:, :-1], next(it) if has_trace else None))
            t_lp = next(it)
            loss = loss + cfg.dream_weight * (mx.exp(t_lp) * (t_lp - s_lp)).sum(-1).mean()
        if has_splice:
            x_tokens = next(it)
            s_lp = log_softmax(ring_forward(ring, x_tokens[:, :-1], next(it) if has_trace else None))
            t_lp, is_new = next(it), next(it)
            learn = -mx.take_along_axis(s_lp, x_tokens[:, 1:, None], axis=-1)[..., 0]
            keep = (mx.exp(t_lp) * (t_lp - s_lp)).sum(-1)
            loss = loss + (is_new * learn + (1 - is_new) * cfg.dream_weight * keep).mean()
        if penalty is not None:
            loss = loss + penalty(ring)
        return loss

    loss_and_grad = nn.value_and_grad(ring, loss_fn)
    state = [ring.state, opt.state]

    def raw_step(*args: mx.array) -> mx.array:
        loss, grads = loss_and_grad(*args)
        grads, _ = optim.clip_grad_norm(grads, 1.0)
        opt.update(ring, grads)
        return loss

    step = mx.compile(raw_step, inputs=state + (penalty_state or []), outputs=state)
    loss = mx.array(0.0)
    for i in range(steps):
        tokens = mx.array(sample(cfg.batch, rng).tokens)
        args = [tokens] + ([trace_fn(tokens[:, :-1])] if has_trace else [])
        if has_dream:
            idx = mx.array(rng.integers(0, dreams.shape[0], cfg.batch))
            d = dreams[idx]
            args += [d] + ([trace_fn(d[:, :-1])] if has_trace else []) + [dream_lp[idx]]
        if has_splice:
            idx = mx.array(rng.integers(0, splice[0].shape[0], cfg.batch))
            x = splice[0][idx]
            args += [x] + ([trace_fn(x[:, :-1])] if has_trace else []) + [splice[1][idx], splice[2][idx]]
        loss = step(*args)
        mx.eval(state, loss)
        if i % 100 == 0:
            safety.check("while training")
    return float(loss)


def evaluate(forward: Forward, era: Era, cfg: Config, seed: int = 1234, alphas: Sequence[float] = ALPHAS) -> Dict:
    """Score the mixture and each self on fixed validation batches of one era."""
    rng = np.random.default_rng(seed)
    tokens_total, mix_nll, self_nll, max_regret, weights = 0, 0.0, None, -1e9, None
    probs_total, mix_right, self_right = 0, 0, None
    for _ in range(cfg.eval_batches):
        b = era.sample_val(cfg.eval_batch, rng)
        tokens = mx.array(b.tokens)
        logits = forward(tokens[:, :-1])
        mix_lp, self_lp, w_final = switching_logprobs(logits, tokens[:, 1:], alphas)
        tgt = mx.take_along_axis(mix_lp, tokens[:, 1:, None], axis=-1)[..., 0]
        m_seq, s_seq = np.array(-tgt.sum(-1)), np.array(-self_lp.sum(-1))
        mix_nll += m_seq.sum()
        self_nll = s_seq.sum(1) if self_nll is None else self_nll + s_seq.sum(1)
        max_regret = max(max_regret, float((m_seq - s_seq.min(0)).max()))
        w = np.array(w_final).mean(1)
        weights = w if weights is None else weights + w
        tokens_total += tgt.size
        safety.check("while evaluating")
        if b.out_mask is not None:
            m, pid = b.out_mask[:, 1:], b.problem[:, 1:]
            n_prob = int(pid.max()) + 1
            need = np.bincount(pid[m], minlength=n_prob)
            preds = [np.array(mx.argmax(mix_lp, axis=-1))] + [np.array(mx.argmax(l, axis=-1)) for l in logits]
            rights = []
            for p in preds:
                got = np.bincount(pid[m], weights=(p == b.tokens[:, 1:])[m], minlength=n_prob)
                rights.append(int(((got == need) & (need > 0)).sum()))
            probs_total += int((need > 0).sum())
            mix_right += rights[0]
            self_right = np.array(rights[1:]) if self_right is None else self_right + np.array(rights[1:])
    out = {"mix_nll": float(mix_nll / tokens_total), "self_nll": [float(v) for v in self_nll / tokens_total],
           "max_regret": float(max_regret), "weights": [float(v) for v in weights / cfg.eval_batches]}
    if probs_total:
        out["mix_acc"] = float(mix_right / probs_total)
        out["self_acc"] = [float(v) for v in self_right / probs_total]
    return out


def lineage(rings: Sequence[Ring]) -> Forward:
    return lambda tokens: lineage_forward(rings, tokens)


def prune_ring(frozen: Sequence[Ring], ring: Ring, era: Era, cfg: Config, rng: np.random.Generator,
               dreams: Optional[mx.array], dream_lp: Optional[mx.array], splice: Optional[tuple] = None) -> Ring:
    """Remove the weakest feed-forward neurons of the newest ring while its loss holds."""
    k = len(frozen)
    self_loss = lambda r: evaluate(lineage(list(frozen) + [r]), era, cfg)["self_nll"][-1]
    base = self_loss(ring)
    stats: Dict[int, List[mx.array]] = {k: []}
    lineage_forward(list(frozen) + [ring], mx.array(era.sample(cfg.eval_batch, rng).tokens)[:, :-1], which=[-1], ffn_stats=stats)
    for frac in (0.75, 0.5, 0.25):
        trial = clone_ring(ring)
        for layer in range(ring.n_layers):
            score = np.array(stats[k][layer] * mx.sqrt((ring.f2[layer].weight ** 2).sum(axis=0)))
            keep = mx.array(np.sort(np.argsort(-score)[: max(4, int(round(len(score) * (1 - frac))))]))
            trial.f1[layer].weight, trial.f1[layer].bias = ring.f1[layer].weight[keep], ring.f1[layer].bias[keep]
            trial.f2[layer].weight = ring.f2[layer].weight[:, keep]
        train_ring(frozen, trial, era.sample, cfg.prune_steps, cfg, rng, dreams, dream_lp, lr=cfg.lr * 0.25, splice=splice)
        if self_loss(trial) <= base * (1 + cfg.prune_tol):
            return trial
    return ring


def learn_era(rings: List[Ring], era: Era, cfg: Config, rng: np.random.Generator) -> Dict:
    """Grow rings for one era. A ring is kept only if it lowers the era's mixture loss."""
    info = {"era": era.name, "steps": 0, "rings_kept": 0, "rings_tried": 0, "params_added": 0, "ff_kept": []}
    prev = evaluate(lineage(rings), era, cfg)["mix_nll"] if rings else float("inf")
    info["loss_before"] = prev
    dreams = dream(rings, cfg.n_dreams, cfg.length, BOS) if (cfg.consolidate and rings) else None
    dream_lp = teacher_targets(rings, dreams) if dreams is not None else None
    splice = make_splice(rings, dreams, era, cfg, rng) if (cfg.splice and dreams is not None) else None
    for _ in range(cfg.max_rings_per_era if rings else 1):
        first = not rings
        ring = Ring(cfg.vocab, sum(r.delta for r in rings), cfg.base_delta if first else cfg.ring_delta, cfg.n_layers,
                    cfg.base_heads if first else cfg.ring_heads, cfg.base_ff if first else cfg.ring_ff, cfg.length,
                    residual=cfg.residual)
        steps = cfg.steps_base if first else cfg.steps_ring
        train_ring(rings, ring, era.sample, steps, cfg, rng, dreams, dream_lp, splice=splice)
        info["steps"] += steps
        info["rings_tried"] += 1
        after = evaluate(lineage(rings + [ring]), era, cfg)["mix_nll"]
        if not first and prev - after < cfg.min_gain:
            logger.info(f"  {era.name}: ring rejected (gain {prev - after:.4f} nats/token)")
            break
        if cfg.prune:
            ring = prune_ring(rings, ring, era, cfg, rng, dreams, dream_lp, splice)
            info["steps"] += cfg.prune_steps
        ring.freeze()
        rings.append(ring)
        prev = evaluate(lineage(rings), era, cfg)["mix_nll"]
        info["rings_kept"] += 1
        info["params_added"] += count_params(ring)
        info["ff_kept"].append(ring.d_ff_per_layer())
        logger.info(f"  {era.name}: ring kept, mixture loss {prev:.4f}, params +{count_params(ring)}, ff {ring.d_ff_per_layer()}")
    info["loss_after"] = prev
    return info


if __name__ == "__main__":
    import sys
    import time

    from data import skills_stream

    failures, total_tests = [], 0
    mx.random.seed(0)
    rng = np.random.default_rng(0)
    cfg = Config(steps_base=300, steps_ring=150, eval_batches=2, prune_steps=40, n_layers=2, base_delta=48, base_ff=96)
    eras = skills_stream(cfg.length)
    rings: List[Ring] = []

    total_tests += 1  # a short run must actually learn something on the first era
    t0 = time.time()
    info = learn_era(rings, eras[0], cfg, rng)
    rate = info["steps"] / (time.time() - t0)
    res = evaluate(lineage(rings), eras[0], cfg)
    print(f"era 0: loss {res['mix_nll']:.3f} acc {res['mix_acc']:.3f}  ({rate:.1f} steps/s incl. eval)")
    if not (len(rings) == 1 and res["mix_nll"] < 2.6):
        failures.append(f"first era did not learn: loss {res['mix_nll']:.3f}")

    total_tests += 1  # growing for a second era must leave era-0 self bit-identical
    probe = mx.array(eras[0].sample_val(8, np.random.default_rng(5)).tokens)[:, :-1]
    before = np.array(lineage_forward(rings, probe, which=[0])[0])
    learn_era(rings, eras[1], cfg, rng)
    after = np.array(lineage_forward(rings, probe, which=[0])[0])
    if np.abs(before - after).max() != 0.0:
        failures.append(f"era-0 self changed: {np.abs(before - after).max()}")

    total_tests += 1  # the mixture on era 0 stays within log K nats per sequence of the era-0 self
    res2 = evaluate(lineage(rings), eras[0], cfg)
    if res2["max_regret"] > np.log(len(rings) * len(ALPHAS)) + 1e-3:
        failures.append(f"regret {res2['max_regret']:.4f} exceeds log(K A) {np.log(len(rings) * len(ALPHAS)):.4f}")
    print(f"after era 1: era-0 mixture loss {res2['mix_nll']:.3f} (was {res['mix_nll']:.3f}), rings {len(rings)}, max regret {res2['max_regret']:.4f}")

    if failures:
        print(f"VALIDATION FAILED - {len(failures)} of {total_tests} tests failed:")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print(f"VALIDATION PASSED - all {total_tests} tests produced expected results")
