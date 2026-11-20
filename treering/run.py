"""Experiment runner for tree-ring learning and its baselines.

Purpose
    Runs one stream (skills or text) for one seed: the tree-ring model with and without
    dream consolidation, then every baseline with the same step and parameter budgets,
    plus the checks that back the claims (bit-exact past selves, the regret bound,
    exact rollback, forward transfer, compaction by dreaming). Writes one JSON file.

Docs
    Typer: https://typer.tiangolo.com/

Sample input
    python run.py --benchmark skills --seed 0

Expected output
    results/skills_seed0.json and a one-line summary per method on stdout.
"""
import json
import os
import time
from typing import Dict, List, Optional

import mlx.core as mx
import numpy as np
import typer
from loguru import logger

from baselines import plain_model, run_isolated, run_joint, run_sequential, train_on_dreams
from data import BOS, Era, skills_batch, skills_stream, text_stream
from learn import Config, evaluate, learn_era, lineage, train_ring
from mixture import ALPHAS, dream
from model import Ring, count_params, lineage_forward, load_rings, save_rings

app = typer.Typer(add_completion=False)
KEEP = ("mix_nll", "mix_acc", "max_regret", "weights", "self_nll", "self_acc")


def grid(rings: List[Ring], eras: List[Era], cfg: Config) -> List[Dict]:
    return [{k: v for k, v in evaluate(lineage(rings), e, cfg).items() if k in KEEP} for e in eras]


def run_treering(eras: List[Era], cfg: Config, seed: int) -> Dict:
    mx.random.seed(seed)
    rng = np.random.default_rng(seed)
    probe = mx.array(np.concatenate([e.sample_val(4, np.random.default_rng(99)).tokens for e in eras]))[:, :-1]
    rings: List[Ring] = []
    infos, matrix, snapshots, n_rings = [], [], {}, []
    for era in eras:
        info = learn_era(rings, era, cfg, rng)
        for k in range(len(snapshots), len(rings)):
            snapshots[k] = np.array(lineage_forward(rings, probe, which=[k])[0])
        infos.append(info)
        n_rings.append(len(rings))
        matrix.append(grid(rings, eras, cfg))
        logger.info(f"[tree-ring{'+dreams' if cfg.consolidate else ''}{'+corrections' if cfg.residual else ''}{'+splice' if cfg.splice else ''}] after {era.name}: rings {len(rings)} | "
                    + " ".join(f"{c['mix_nll']:.3f}" for c in matrix[-1]))
    now = lineage_forward(rings, probe)
    exact = max(float(np.abs(np.array(now[k]) - snapshots[k]).max()) for k in snapshots)
    rollback = 0.0
    for e, n in enumerate(n_rings[:-1]):
        again = grid(rings[:n], eras, cfg)
        rollback = max(rollback, max(abs(a["mix_nll"] - b["mix_nll"]) for a, b in zip(again, matrix[e])))
    samples = [bytes(np.array(row).astype(np.uint8)).decode("ascii", "replace") for row in dream(rings, 8, cfg.length, BOS)]
    return {"infos": infos, "matrix": matrix, "n_rings": n_rings, "params": int(sum(count_params(r) for r in rings)),
            "ring_params": [count_params(r) for r in rings], "exact_max_diff": exact, "rollback_max_diff": rollback,
            "dream_samples": samples, "_rings": rings}


def transfer_probe(rings: List[Ring], n_before: int, era: Era, cfg: Config, seed: int) -> List[Dict]:
    """Same new-parameter budget, same steps: a ring that can read older rings vs a model that cannot."""
    out = []
    for steps in (200, 500, 1200):
        mx.random.seed(seed + 7)
        rng = np.random.default_rng(seed + 7)
        base = list(rings[:n_before])
        ring = Ring(cfg.vocab, sum(r.delta for r in base), cfg.ring_delta, cfg.n_layers, cfg.ring_heads, cfg.ring_ff, cfg.length)
        train_ring(base, ring, era.sample, steps, cfg, rng)
        a = evaluate(lineage(base + [ring]), era, cfg)
        alone = plain_model(cfg, count_params(ring))
        train_ring([], alone, era.sample, steps, cfg, rng)
        b = evaluate(lineage([alone]), era, cfg)
        out.append({"steps": steps, "ring_params": count_params(ring), "alone_params": count_params(alone),
                    "ring_nll": a["self_nll"][-1], "alone_nll": b["self_nll"][-1],
                    "ring_acc": a.get("self_acc", [None])[-1], "alone_acc": b.get("self_acc", [None])[-1]})
        logger.info(f"[transfer] {steps} steps: ring {out[-1]['ring_nll']:.3f}/{out[-1]['ring_acc']} vs alone {out[-1]['alone_nll']:.3f}/{out[-1]['alone_acc']}")
    return out


def write(res: Dict, path: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({k: v for k, v in res.items() if v is not None}, f, indent=1, default=float)
    os.replace(tmp, path)


@app.command()
def main(benchmark: str = typer.Option("skills", help="skills or text"), seed: int = typer.Option(0),
         smoke: bool = typer.Option(False, help="tiny budgets, to test the pipeline end to end"),
         skip: str = typer.Option("", help="comma-separated stage names to leave out of this run"),
         out_dir: str = typer.Option("results")) -> None:
    """Run one stream for one seed. Saves after every method and resumes if interrupted."""
    t0 = time.time()
    cfg = Config(length=96 if benchmark == "skills" else 128)
    if smoke:
        cfg = Config(**{**cfg.__dict__, "steps_base": 150, "steps_ring": 80, "eval_batches": 1, "n_dreams": 64,
                        "prune_steps": 20, "n_layers": 2})
    eras = skills_stream(cfg.length) if benchmark == "skills" else text_stream(cfg.length)
    os.makedirs(os.path.join(out_dir, "ckpt"), exist_ok=True)
    path = os.path.join(out_dir, f"{benchmark}_seed{seed}.json")
    ckpt = os.path.join(out_dir, "ckpt", f"{benchmark}_seed{seed}_rings.npz")
    res: Dict = json.load(open(path)) if os.path.exists(path) else {
        "benchmark": benchmark, "seed": seed, "eras": [e.name for e in eras], "config": dict(cfg.__dict__), "seconds": 0.0}

    skipped = {s for s in skip.split(",") if s}
    for s in skipped:  # a skipped stage is treated as done for this invocation only
        res.setdefault(s, None)

    mixed = None
    if benchmark == "skills":  # sequences mixing skills from every era: nothing is trained on these
        tasks = ["cpy", "rev", "srt", "inc", "dup", "add", "dsc", "sri"]
        fn = lambda n, rng: skills_batch(tasks, n, cfg.length, rng)
        mixed = Era("mixed", fn, fn, True)

    def score(name: str, fwd, alphas=ALPHAS, last: bool = False) -> None:
        if mixed is None:
            return
        e = evaluate(fwd, mixed, cfg, alphas=alphas)
        res.setdefault("mixed", {})[name] = {"acc": e["self_acc"][-1] if last else e["mix_acc"],
                                              "nll": e["self_nll"][-1] if last else e["mix_nll"]}

    def done(key: str) -> None:
        res["seconds"] += time.time() - done.t
        done.t = time.time()
        write(res, path)
        logger.info(f"[saved] {key} -> {path}")

    done.t = t0

    if "treering" not in res:
        ours = run_treering(eras, cfg, seed)
        rings = ours.pop("_rings")
        save_rings(ckpt, rings)
        res["treering"] = ours
        score("Tree-ring (switching mixture)", lineage(rings))
        score("Tree-ring (plain Bayes mixture)", lineage(rings), alphas=(0.0,))
        score("Tree-ring, newest self only", lineage(rings), last=True)
        done("treering")
    else:
        rings = load_rings(ckpt)
    ours = res["treering"]
    steps = [i["steps"] for i in ours["infos"]]
    per_era = [i["params_added"] for i in ours["infos"]]

    if "treering_dreams" not in res:
        sleepy = run_treering(eras, Config(**{**cfg.__dict__, "consolidate": True}), seed)
        sleepy_rings = sleepy.pop("_rings")
        res["treering_dreams"] = sleepy
        score("Tree-ring + dreams (switching mixture)", lineage(sleepy_rings))
        score("Tree-ring + dreams, newest self only", lineage(sleepy_rings), last=True)
        done("treering_dreams")

    if "treering_residual" not in res:
        chain = run_treering(eras, Config(**{**cfg.__dict__, "consolidate": True, "residual": True}), seed)
        chain_rings = chain.pop("_rings")
        res["treering_residual"] = chain
        score("Tree-ring corrections + dreams (switching mixture)", lineage(chain_rings))
        score("Tree-ring corrections + dreams, newest self only", lineage(chain_rings), last=True)
        done("treering_residual")

    if "treering_splice" not in res:
        sp = run_treering(eras, Config(**{**cfg.__dict__, "consolidate": True, "residual": True, "splice": True}), seed)
        sp_rings = sp.pop("_rings")
        res["treering_splice"] = sp
        score("Tree-ring corrections + spliced dreams (switching mixture)", lineage(sp_rings))
        score("Tree-ring corrections + spliced dreams, newest self only", lineage(sp_rings), last=True)
        done("treering_splice")

    if "selfdream_splice" not in res:
        out = run_sequential(eras, steps, ours["params"], cfg, seed, "selfdream_splice")
        score("Fixed size + own spliced dreams", lineage(out.pop("_models")))
        res["selfdream_splice"] = out
        done("selfdream_splice")

    stages = [("naive", "Plain fine-tuning", lambda: run_sequential(eras, steps, ours["params"], cfg, seed, "naive")),
              ("ewc_1000", "EWC (strength 1000)", lambda: run_sequential(eras, steps, ours["params"], cfg, seed, "ewc", 1e3)),
              ("ewc_100000", "EWC (strength 100000)", lambda: run_sequential(eras, steps, ours["params"], cfg, seed, "ewc", 1e5)),
              ("selfdream", "Fixed size + own dreams", lambda: run_sequential(eras, steps, ours["params"], cfg, seed, "selfdream")),
              ("joint", "Joint training (upper bound)", lambda: run_joint(eras, steps, ours["params"], cfg, seed))]
    for key, label, fn_stage in stages:
        if key not in res:
            out = fn_stage()
            score(label, lineage(out.pop("_models")))
            res[key] = out
            done(key)

    if "isolated" not in res:
        out = run_isolated(eras, steps, per_era, cfg, seed)
        experts = out.pop("_models")
        score("Separate experts + mixture", lambda t: [lineage_forward([m], t)[0] for m in experts])
        res["isolated"] = out
        done("isolated")

    if benchmark == "skills" and "transfer" not in res:
        res["transfer"] = transfer_probe(rings, ours["n_rings"][3], eras[4], cfg, seed)
        done("transfer")

    if "compaction" not in res:
        student = train_on_dreams(rings, ours["params"] // 2, sum(steps) // 2, cfg, seed)
        score("Compacted by dreaming (no data)", lineage([student]))
        keep = ("mix_nll", "mix_acc")
        res["compaction"] = {"params": count_params(student), "teacher_params": ours["params"],
                             "final": [{k: v for k, v in evaluate(lineage([student]), e, cfg).items() if k in keep} for e in eras]}
        done("compaction")

    res["complete"] = not skipped
    write(res, path)
    final = ours["matrix"][-1]
    acc = [c.get("mix_acc") for c in final]
    print(f"COMPLETE {path} | total {res['seconds']:.0f}s | tree-ring: rings {ours['n_rings'][-1]}, params {ours['params']}, "
          f"final loss {[round(c['mix_nll'], 3) for c in final]}, acc {[None if a is None else round(a, 3) for a in acc]}, "
          f"exact diff {ours['exact_max_diff']}, rollback diff {ours['rollback_max_diff']}")


if __name__ == "__main__":
    app()
