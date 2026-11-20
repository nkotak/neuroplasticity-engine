"""Fixed-size comparison methods, all built on the same transformer code as the tree-ring model.

Purpose
    plain_model    a standard transformer (a single ring) sized to a parameter budget.
    run_sequential naive fine-tuning, EWC, or self-dream replay on a fixed-size model.
    run_joint      one model trained on all eras at once (the usual upper bound).
    run_isolated   one independent expert per era, combined by the same Bayesian mixture.

Docs
    EWC: Kirkpatrick et al. 2017, https://arxiv.org/abs/1612.00796
    Self-generated rehearsal for language models (LAMOL): https://arxiv.org/abs/1909.03329

Sample input
    out = run_sequential(eras, steps_per_era, total_params, cfg, seed=0, method="naive")

Expected output
    {"matrix": [[{"mix_nll": ..., "mix_acc": ...} per era] per training stage], "params": int}
"""
from typing import Callable, Dict, List, Optional

import mlx.core as mx
import numpy as np
from loguru import logger
from mlx.utils import tree_flatten

from data import BOS, Batch, Era
from learn import Config, evaluate, make_splice, teacher_targets, train_ring
from mixture import dream
from model import Ring, clone_ring, count_params, lineage_forward


def plain_model(cfg: Config, target_params: int) -> Ring:
    """Standard transformer whose parameter count is closest to target_params."""
    best = None
    for delta in range(16, 641, 8):
        ring = Ring(cfg.vocab, 0, delta, cfg.n_layers, 4, 4 * delta, cfg.length)
        gap = abs(count_params(ring) - target_params)
        if best is None or gap < best[0]:
            best = (gap, ring)
    return best[1]


def _grid(models_fn: Callable[[], List[Ring]], eras: List[Era], cfg: Config, separate: bool = False) -> List[Dict]:
    models = models_fn()
    fwd = (lambda t: [lineage_forward([m], t)[0] for m in models]) if separate else (lambda t: lineage_forward(models, t))
    keep = ("mix_nll", "mix_acc", "max_regret", "weights")
    return [{k: v for k, v in evaluate(fwd, era, cfg).items() if k in keep} for era in eras]


def fisher_diagonal(model: Ring, era: Era, cfg: Config, rng: np.random.Generator, batches: int = 24) -> Dict[str, mx.array]:
    import mlx.nn as nn

    def loss_fn(tokens: mx.array) -> mx.array:
        logits = lineage_forward([model], tokens[:, :-1])[0]
        return nn.losses.cross_entropy(logits, tokens[:, 1:], reduction="mean")

    grad_fn = nn.value_and_grad(model, loss_fn)
    fisher: Dict[str, mx.array] = {}
    for _ in range(batches):
        _, grads = grad_fn(mx.array(era.sample(16, rng).tokens))
        for name, g in tree_flatten(grads):
            fisher[name] = fisher.get(name, 0.0) + g * g / batches
        mx.eval(list(fisher.values()))
    return fisher


def run_sequential(eras: List[Era], steps: List[int], total_params: int, cfg: Config, seed: int,
                   method: str = "naive", ewc_lambda: float = 0.0) -> Dict:
    """Train one fixed-size model on the eras in order. method: naive | ewc | selfdream | selfdream_splice."""
    mx.random.seed(seed + 100)
    rng = np.random.default_rng(seed + 100)
    model = plain_model(cfg, total_params)
    fisher: Dict[str, mx.array] = {}
    star: Dict[str, mx.array] = {}
    matrix = []
    for i, era in enumerate(eras):
        dreams, dream_lp, penalty, pstate, splice = None, None, None, None, None
        if i > 0 and method in ("selfdream", "selfdream_splice"):
            teacher = [clone_ring(model)]
            dreams = dream(teacher, cfg.n_dreams, cfg.length, BOS)
            dream_lp = teacher_targets(teacher, dreams)
            if method == "selfdream_splice":
                splice = make_splice(teacher, dreams, era, cfg, rng)
        if i > 0 and method == "ewc":
            f, s = dict(fisher), dict(star)
            penalty = lambda m: 0.5 * ewc_lambda * sum(((p - s[n]) ** 2 * f[n]).sum() for n, p in tree_flatten(m.trainable_parameters()))
            pstate = [f, s]
        if steps[i] > 0:
            train_ring([], model, era.sample, steps[i], cfg, rng, dreams=dreams, dream_lp=dream_lp, penalty=penalty, penalty_state=pstate, splice=splice)
        if method == "ewc":
            new = fisher_diagonal(model, era, cfg, rng)
            fisher = {n: fisher.get(n, 0.0) + v for n, v in new.items()}
            star = {n: mx.array(p) for n, p in tree_flatten(model.trainable_parameters())}
            mx.eval(list(fisher.values()), list(star.values()))
        matrix.append(_grid(lambda: [model], eras, cfg))
        logger.info(f"  [{method}] after {era.name}: " + " ".join(f"{c['mix_nll']:.3f}" for c in matrix[-1]))
    return {"matrix": matrix, "params": count_params(model), "_models": [model]}


def run_joint(eras: List[Era], steps: List[int], total_params: int, cfg: Config, seed: int) -> Dict:
    mx.random.seed(seed + 200)
    rng = np.random.default_rng(seed + 200)
    model = plain_model(cfg, total_params)

    def sample(n: int, r: np.random.Generator) -> Batch:
        per = -(-n // len(eras))
        toks = np.concatenate([e.sample(per, r).tokens for e in eras])
        return Batch(toks[r.permutation(len(toks))[:n]])

    train_ring([], model, sample, sum(steps), cfg, rng)
    final = _grid(lambda: [model], eras, cfg)
    logger.info("  [joint] " + " ".join(f"{c['mix_nll']:.3f}" for c in final))
    return {"final": final, "params": count_params(model), "_models": [model]}


def run_isolated(eras: List[Era], steps: List[int], params_per_era: List[int], cfg: Config, seed: int) -> Dict:
    """Independent expert per era with the same new-parameter budget, mixed by likelihood."""
    mx.random.seed(seed + 300)
    rng = np.random.default_rng(seed + 300)
    experts: List[Ring] = []
    matrix = []
    for i, era in enumerate(eras):
        if params_per_era[i] > 0:
            expert = plain_model(cfg, params_per_era[i])
            train_ring([], expert, era.sample, steps[i], cfg, rng)
            experts.append(expert)
        matrix.append(_grid(lambda: experts, eras, cfg, separate=True))
        logger.info(f"  [isolated] after {era.name}: " + " ".join(f"{c['mix_nll']:.3f}" for c in matrix[-1]))
    return {"matrix": matrix, "params": int(sum(count_params(e) for e in experts)), "_models": experts}


def train_on_dreams(teacher: List[Ring], target_params: int, steps: int, cfg: Config, seed: int,
                    refresh: int = 500) -> Ring:
    """Compaction: a fresh model that only ever sees dreams of the teacher lineage."""
    mx.random.seed(seed + 400)
    rng = np.random.default_rng(seed + 400)
    student = plain_model(cfg, target_params)
    done = 0
    while done < steps:
        dreams = dream(teacher, cfg.n_dreams, cfg.length, BOS)
        chunk = min(refresh, steps - done)
        idle = lambda n, r: Batch(np.array(dreams[mx.array(r.integers(0, dreams.shape[0], n))]))
        train_ring([], student, idle, chunk, cfg, rng, dreams=dreams, dream_lp=teacher_targets(teacher, dreams),
                   lr=cfg.lr * (1 - done / steps) + 1e-4)
        done += chunk
    return student


if __name__ == "__main__":
    import sys

    from data import skills_stream

    failures, total_tests = [], 0
    cfg = Config(n_layers=2, eval_batches=1, n_dreams=64)
    eras = skills_stream(cfg.length)[:2]

    total_tests += 1
    m = plain_model(cfg, 200_000)
    if abs(count_params(m) - 200_000) > 30_000:
        failures.append(f"plain_model missed the budget: {count_params(m)}")

    for method, lam in (("naive", 0.0), ("ewc", 1e4), ("selfdream", 0.0)):
        total_tests += 1
        out = run_sequential(eras, [60, 60], 120_000, cfg, seed=0, method=method, ewc_lambda=lam)
        vals = [c["mix_nll"] for row in out["matrix"] for c in row]
        if len(out["matrix"]) != 2 or not all(np.isfinite(vals)):
            failures.append(f"{method}: bad output {vals}")

    total_tests += 1
    iso = run_isolated(eras, [60, 60], [60_000, 30_000], cfg, seed=0)
    if len(iso["matrix"][-1][0]["weights"]) != 2:
        failures.append("isolated experts: expected two mixture weights")

    total_tests += 1
    j = run_joint(eras, [60, 60], 120_000, cfg, seed=0)
    if len(j["final"]) != 2:
        failures.append("joint: wrong result size")

    if failures:
        print(f"VALIDATION FAILED - {len(failures)} of {total_tests} tests failed:")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print(f"VALIDATION PASSED - all {total_tests} tests produced expected results")
