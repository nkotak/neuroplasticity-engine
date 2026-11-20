"""Aggregate results/<benchmark>_seed*.json into plain-text tables.

Purpose
    Reads every seed of one stream and prints, per method: final average loss and
    accuracy, forgetting, parameter count, plus the growth log and the guarantee checks.

Docs
    NumPy: https://numpy.org/doc/stable/

Sample input
    python report.py --benchmark skills

Expected output
    Markdown tables on stdout and in results/summary_<benchmark>.md.
"""
import glob
import json
import math
import os
from typing import Dict, List, Optional

import numpy as np
import typer

app = typer.Typer(add_completion=False)
SEQ = {"naive": "Plain fine-tuning", "selfdream": "Fixed size + own dreams", "selfdream_splice": "Fixed size + own spliced dreams", "isolated": "Separate experts + mixture"}


def ms(vals: List[float], digits: int = 3) -> str:
    vals = [v for v in vals if v is not None]
    if not vals:
        return "n/a"
    return f"{np.mean(vals):.{digits}f}" + (f" ± {np.std(vals):.{digits}f}" if len(vals) > 1 else "")


def summarise(matrix: List[List[Dict]], key_nll: str = "mix_nll", key_acc: str = "mix_acc", last_self: bool = False) -> Dict:
    def val(cell: Dict, key: str) -> Optional[float]:
        if last_self:
            v = cell.get("self_nll" if key == key_nll else "self_acc")
            return None if v is None else v[-1]
        return cell.get(key)

    final = matrix[-1]
    n = len(final)
    out = {"nll": float(np.mean([val(c, key_nll) for c in final])),
           "nll_forget": float(np.mean([val(final[j], key_nll) - val(matrix[j][j], key_nll) for j in range(n - 1)]))}
    if val(final[0], key_acc) is not None:
        out["acc"] = float(np.mean([val(c, key_acc) for c in final]))
        out["acc_forget"] = float(np.mean([val(matrix[j][j], key_acc) - val(final[j], key_acc) for j in range(n - 1)]))
        out["per_era_acc"] = [val(c, key_acc) for c in final]
    out["per_era_nll"] = [val(c, key_nll) for c in final]
    return out


@app.command()
def main(benchmark: str = typer.Option("skills"), results: str = typer.Option("results")) -> None:
    runs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(results, f"{benchmark}_seed[0-9].json")))]
    if not runs:
        raise SystemExit(f"no results for {benchmark} in {results}")
    eras = runs[0]["eras"]
    has_acc = "mix_acc" in runs[0]["treering"]["matrix"][-1][0]
    lines = [f"# {benchmark} stream: {len(runs)} seed(s), eras: {', '.join(eras)}", ""]

    rows: Dict[str, List[Dict]] = {}
    params: Dict[str, List[int]] = {}

    def add(name: str, summary: Dict, p: int) -> None:
        rows.setdefault(name, []).append(summary)
        params.setdefault(name, []).append(p)

    for r in runs:
        add("Tree-ring (mixture)", summarise(r["treering"]["matrix"]), r["treering"]["params"])
        add("Tree-ring, newest self only", summarise(r["treering"]["matrix"], last_self=True), r["treering"]["params"])
        if "treering_dreams" in r:
            add("Tree-ring + dreams (mixture)", summarise(r["treering_dreams"]["matrix"]), r["treering_dreams"]["params"])
            add("Tree-ring + dreams, newest self only", summarise(r["treering_dreams"]["matrix"], last_self=True), r["treering_dreams"]["params"])
        if "treering_residual" in r:
            add("Tree-ring corrections + dreams (mixture)", summarise(r["treering_residual"]["matrix"]), r["treering_residual"]["params"])
            add("Tree-ring corrections + dreams, newest self only", summarise(r["treering_residual"]["matrix"], last_self=True), r["treering_residual"]["params"])
        if "treering_splice" in r:
            add("Tree-ring corrections + spliced dreams (mixture)", summarise(r["treering_splice"]["matrix"]), r["treering_splice"]["params"])
            add("Tree-ring corrections + spliced dreams, newest self only", summarise(r["treering_splice"]["matrix"], last_self=True), r["treering_splice"]["params"])
        for key, label in SEQ.items():
            if key in r:
                add(label, summarise(r[key]["matrix"]), r[key]["params"])
        ewc = [k for k in r if k.startswith("ewc_")]
        if ewc:
            best = min(ewc, key=lambda k: np.mean([c["mix_nll"] for c in r[k]["matrix"][-1]]))
            add("EWC (best of two strengths)", summarise(r[best]["matrix"]), r[best]["params"])
        if "joint" in r:
            f = r["joint"]["final"]
            add("Joint training (upper bound)", {"nll": float(np.mean([c["mix_nll"] for c in f])), "nll_forget": None,
                                                 "acc": float(np.mean([c["mix_acc"] for c in f])) if has_acc else None,
                                                 "acc_forget": None, "per_era_nll": [c["mix_nll"] for c in f],
                                                 "per_era_acc": [c.get("mix_acc") for c in f]}, r["joint"]["params"])
        if "compaction" in r:
            f = r["compaction"]["final"]
            add("Compacted by dreaming (no data)", {"nll": float(np.mean([c["mix_nll"] for c in f])), "nll_forget": None,
                                                    "acc": float(np.mean([c["mix_acc"] for c in f])) if has_acc else None,
                                                    "acc_forget": None, "per_era_nll": [c["mix_nll"] for c in f],
                                                    "per_era_acc": [c.get("mix_acc") for c in f]}, r["compaction"]["params"])

    head = "| Method | Seeds | Parameters | Final loss (nats/token) | Loss forgotten |" + (" Final accuracy | Accuracy forgotten |" if has_acc else "")
    lines += [head, "|" + "---|" * (head.count("|") - 1)]
    for name, ss in rows.items():
        row = f"| {name} | {len(ss)} | {int(np.mean(params[name])):,} | {ms([s['nll'] for s in ss])} | {ms([s['nll_forget'] for s in ss])} |"
        if has_acc:
            row += f" {ms([s.get('acc') for s in ss])} | {ms([s.get('acc_forget') for s in ss])} |"
        lines.append(row)

    key = "per_era_acc" if has_acc else "per_era_nll"
    lines += ["", f"## Final {'accuracy' if has_acc else 'loss'} per era", "| Method | " + " | ".join(eras) + " |", "|" + "---|" * (len(eras) + 1)]
    for name, ss in rows.items():
        lines.append(f"| {name} | " + " | ".join(ms([s[key][j] for s in ss]) for j in range(len(eras))) + " |")

    lines += ["", "## Growth of the tree-ring model", "| Era | Loss before | Loss after | Rings tried | Rings kept | Parameters added |", "|---|---|---|---|---|---|"]
    for j, era in enumerate(eras):
        infos = [r["treering"]["infos"][j] for r in runs]
        before = [i["loss_before"] for i in infos if i["loss_before"] < 1e9]
        lines.append(f"| {era} | {ms(before) if before else 'n/a'} | {ms([i['loss_after'] for i in infos])} | {ms([i['rings_tried'] for i in infos], 1)} | "
                     f"{ms([i['rings_kept'] for i in infos], 1)} | {int(np.mean([i['params_added'] for i in infos])):,} |")

    lines += ["", "## Guarantee checks (tree-ring, all seeds)"]
    for label, k in (("Tree-ring", "treering"), ("Tree-ring + dreams", "treering_dreams"), ("Tree-ring corrections + dreams", "treering_residual"), ("Tree-ring corrections + spliced dreams", "treering_splice")):
        rs = [r[k] for r in runs if k in r]
        if not rs:
            continue
        regret = max(c["max_regret"] for r in rs for row in r["matrix"] for c in row)
        bound = max(math.log(3 * r["n_rings"][-1]) for r in rs)
        lines.append(f"- {label}: largest change in any past self's output after later growth = {max(r['exact_max_diff'] for r in rs)}; "
                     f"largest change after rollback = {max(r['rollback_max_diff'] for r in rs)}; "
                     f"worst per-sequence gap to the best self = {regret:.3f} nats (bound log(3K) = {bound:.3f}); rings at end: {[r['n_rings'][-1] for r in rs]}")

    tr_runs = [r for r in runs if "transfer" in r]
    if tr_runs:
        lines += ["", "## Forward transfer: one new ring on the frozen model vs a stand-alone model of the same size (era: " + eras[4] + ")",
                  "| Training steps | Ring loss | Stand-alone loss | Ring accuracy | Stand-alone accuracy |", "|---|---|---|---|---|"]
        for i in range(len(tr_runs[0]["transfer"])):
            t = [r["transfer"][i] for r in tr_runs]
            lines.append(f"| {t[0]['steps']} | {ms([x['ring_nll'] for x in t])} | {ms([x['alone_nll'] for x in t])} | {ms([x['ring_acc'] for x in t])} | {ms([x['alone_acc'] for x in t])} |")

    names = []
    for r in runs:
        names += [n for n in r.get("mixed", {}) if n not in names]
    if names:
        lines += ["", "## Sequences that mix skills from every era (never seen in training)", "| Method | Seeds | Accuracy | Loss |", "|---|---|---|---|"]
        cells = {n: [r["mixed"][n] for r in runs if n in r.get("mixed", {})] for n in names}
        for n in sorted(names, key=lambda n: -np.mean([c["acc"] for c in cells[n]])):
            lines.append(f"| {n} | {len(cells[n])} | {ms([c['acc'] for c in cells[n]])} | {ms([c['nll'] for c in cells[n]])} |")

    lines += ["", f"Seconds per seed: {ms([r['seconds'] for r in runs], 0)}"]
    text = "\n".join(lines)
    with open(os.path.join(results, f"summary_{benchmark}.md"), "w") as f:
        f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    app()
