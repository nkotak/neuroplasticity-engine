"""Tree-ring transformer: a network that grows in immutable rings.

Purpose
    Each `Ring` is a block of new residual dimensions, attention heads and feed-forward
    neurons. A ring READS every dimension up to and including its own and WRITES only its
    own dimensions. Nothing ever writes into an older ring's dimensions, so once a ring is
    frozen, the model made of rings 1..k (a "self") computes exactly the same function
    forever, no matter how many rings are added later. One forward pass of the full model
    therefore yields the predictions of every past self.

Docs
    MLX: https://ml-explore.github.io/mlx/build/html/index.html

Sample input
    rings = [Ring(vocab=128, d_prev=0, delta=64, n_layers=2, n_heads=4, d_ff=128, max_len=32)]
    logits = lineage_forward(rings, tokens)          # tokens: (batch, time) int array

Expected output
    A list with one (batch, time, vocab) array of next-token logits per requested self.
"""
import json
import math
from typing import Dict, List, Optional, Sequence

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map, tree_unflatten

import safety  # noqa: F401  (caps MLX memory for every process that uses this module)


class Ring(nn.Module):
    """One growth ring. Reads dims [0, d_prev + delta), writes dims [d_prev, d_prev + delta)."""

    def __init__(self, vocab: int, d_prev: int, delta: int, n_layers: int, n_heads: int,
                 d_ff: int, max_len: int, lateral: int = 48, residual: bool = False) -> None:
        super().__init__()
        self.vocab, self.d_prev, self.delta = vocab, d_prev, delta
        self.residual = bool(residual and d_prev > 0)
        self.n_layers, self.n_heads, self.max_len = n_layers, n_heads, max_len
        self.d_head = max(8, delta // n_heads)
        self.lat = min(lateral, d_prev)
        d_read, hd = self.lat + delta, n_heads * self.d_head
        if d_prev > 0:  # one small linear "lateral adapter" through which the ring reads all older rings
            self.adapter = nn.Linear(d_prev, self.lat, bias=False)
        self.tok = nn.Embedding(vocab, delta)
        self.pos = nn.Embedding(max_len, delta)
        self.ln1 = [nn.LayerNorm(delta) for _ in range(n_layers)]
        self.ln2 = [nn.LayerNorm(delta) for _ in range(n_layers)]
        self.q = [nn.Linear(d_read, hd, bias=False) for _ in range(n_layers)]
        self.k = [nn.Linear(d_read, hd, bias=False) for _ in range(n_layers)]
        self.v = [nn.Linear(d_read, hd, bias=False) for _ in range(n_layers)]
        self.o = [nn.Linear(hd, delta) for _ in range(n_layers)]
        self.f1 = [nn.Linear(d_read, d_ff) for _ in range(n_layers)]
        self.f2 = [nn.Linear(d_ff, delta) for _ in range(n_layers)]
        self.lnf = nn.LayerNorm(delta)
        self.head = nn.Linear(d_read, vocab)
        if self.residual:  # start as an exact copy of the previous self: the correction begins at zero
            self.head.weight = mx.zeros(self.head.weight.shape)
            self.head.bias = mx.zeros(self.head.bias.shape)

    def d_ff_per_layer(self) -> List[int]:
        return [int(f.weight.shape[0]) for f in self.f1]


def count_params(module: nn.Module) -> int:
    return int(sum(v.size for _, v in tree_flatten(module.parameters())))


def clone_ring(ring: Ring) -> Ring:
    """Deep copy of a ring (used for frozen teachers in baselines)."""
    new = Ring(ring.vocab, ring.d_prev, ring.delta, ring.n_layers, ring.n_heads, 1, ring.max_len, ring.lat, ring.residual)
    new.d_head = ring.d_head
    new.update(tree_map(lambda a: mx.array(a), ring.parameters()))
    return new


def save_rings(path: str, rings: Sequence[Ring]) -> None:
    """Write all rings to `path` (.npz) plus a small JSON file describing their shapes."""
    arrays, meta = {}, []
    for i, r in enumerate(rings):
        meta.append({"vocab": r.vocab, "d_prev": r.d_prev, "delta": r.delta, "n_layers": r.n_layers,
                     "n_heads": r.n_heads, "max_len": r.max_len, "lat": r.lat, "residual": r.residual})
        for name, value in tree_flatten(r.parameters()):
            arrays[f"{i}|{name}"] = value
    mx.savez(path, **arrays)
    with open(path + ".json", "w") as f:
        json.dump(meta, f)


def load_rings(path: str) -> List[Ring]:
    with open(path + ".json") as f:
        meta = json.load(f)
    data = mx.load(path)
    rings = []
    for i, m in enumerate(meta):
        ring = Ring(m["vocab"], m["d_prev"], m["delta"], m["n_layers"], m["n_heads"], 1, m["max_len"], max(m["lat"], 1),
                    m.get("residual", False))
        ring.update(tree_unflatten([(k.split("|", 1)[1], v) for k, v in data.items() if k.startswith(f"{i}|")]))
        ring.freeze()
        rings.append(ring)
    return rings


def _attend(r: Ring, layer: int, inp: mx.array, mask: mx.array) -> mx.array:
    batch, time = inp.shape[0], inp.shape[1]
    shape = (batch, time, r.n_heads, r.d_head)
    q = r.q[layer](inp).reshape(shape).transpose(0, 2, 1, 3)
    k = r.k[layer](inp).reshape(shape).transpose(0, 2, 3, 1)
    v = r.v[layer](inp).reshape(shape).transpose(0, 2, 1, 3)
    att = mx.softmax((q @ k) / math.sqrt(r.d_head) + mask, axis=-1)
    return r.o[layer]((att @ v).transpose(0, 2, 1, 3).reshape(batch, time, r.n_heads * r.d_head))


def ring_forward(ring: Ring, tokens: mx.array, trace: Optional[List[mx.array]]) -> mx.array:
    """Logits of the newest self, given the cached activations of all older (frozen) rings.

    `trace` is what lineage_forward(..., return_trace=True) returns for the older rings:
    [ln1 output per layer..., ln2 output per layer..., final-norm output, newest older self's logits].
    None for a first ring.
    """
    time, layers = tokens.shape[1], ring.n_layers
    x = ring.tok(tokens) + ring.pos(mx.arange(time))
    mask = nn.MultiHeadAttention.create_additive_causal_mask(time).astype(x.dtype)
    join = lambda adapter, old, new: mx.concatenate([adapter(old), new], axis=-1)
    for layer in range(layers):
        own = ring.ln1[layer](x)
        x = x + _attend(ring, layer, join(ring.adapter, trace[layer], own) if trace else own, mask)
        own = ring.ln2[layer](x)
        h = nn.gelu(ring.f1[layer](join(ring.adapter, trace[layers + layer], own) if trace else own))
        x = x + ring.f2[layer](h)
    own = ring.lnf(x)
    logits = ring.head(join(ring.adapter, trace[2 * layers], own) if trace else own)
    return logits + mx.stop_gradient(trace[2 * layers + 1]) if (ring.residual and trace) else logits


def lineage_forward(rings: Sequence[Ring], tokens: mx.array, which: Optional[Sequence[int]] = None,
                    ffn_stats: Optional[Dict[int, List[mx.array]]] = None, return_trace: bool = False):
    """Run all rings once; return next-token logits for the selves listed in `which`.

    Self k is the model made of rings 0..k. `ffn_stats`, if given as {ring_index: []}, is
    filled with the mean absolute feed-forward activation per neuron and layer.
    With return_trace=True, returns (logits, trace) where trace feeds ring_forward.
    """
    n = len(rings)
    which = list(range(n)) if which is None else [w % n for w in which]
    ln1_trace, ln2_trace = [], []
    batch, time = tokens.shape
    pos = mx.arange(time)
    xs = [r.tok(tokens) + r.pos(pos) for r in rings]
    mask = nn.MultiHeadAttention.create_additive_causal_mask(time).astype(xs[0].dtype)
    starts = []
    total = 0
    for r in rings:
        starts.append(total)
        total += r.delta
    read = lambda j, adapter, blocks, full: blocks[j] if j == 0 else mx.concatenate([adapter(full[..., : starts[j]]), blocks[j]], axis=-1)
    for layer in range(rings[0].n_layers):
        blocks = [r.ln1[layer](x) for r, x in zip(rings, xs)]
        full = mx.concatenate(blocks, axis=-1)
        ln1_trace.append(full)
        xs = [xs[j] + _attend(r, layer, read(j, r.adapter if j else None, blocks, full), mask) for j, r in enumerate(rings)]
        blocks = [r.ln2[layer](x) for r, x in zip(rings, xs)]
        full = mx.concatenate(blocks, axis=-1)
        ln2_trace.append(full)
        nxt = []
        for j, r in enumerate(rings):
            h = nn.gelu(r.f1[layer](read(j, r.adapter if j else None, blocks, full)))
            if ffn_stats is not None and j in ffn_stats:
                ffn_stats[j].append(mx.mean(mx.abs(h), axis=(0, 1)))
            nxt.append(xs[j] + r.f2[layer](h))
        xs = nxt
    blocks = [r.lnf(x) for r, x in zip(rings, xs)]
    final = mx.concatenate(blocks, axis=-1)
    upto = n - 1 if return_trace else max(which, default=-1)
    every, prev = [], None
    for j in range(upto + 1):
        raw = rings[j].head(read(j, rings[j].adapter if j else None, blocks, final))
        prev = raw + mx.stop_gradient(prev) if rings[j].residual else raw
        every.append(prev)
    logits = [every[j] for j in which]
    return (logits, ln1_trace + ln2_trace + [final, every[-1]]) if return_trace else logits


if __name__ == "__main__":
    import sys

    import numpy as np

    failures, total_tests = [], 0
    mx.random.seed(0)
    toks = mx.random.randint(0, 128, (4, 24))

    total_tests += 1  # shapes
    r0 = Ring(128, 0, 48, 2, 4, 96, 32)
    out = lineage_forward([r0], toks)
    if len(out) != 1 or out[0].shape != (4, 24, 128):
        failures.append(f"shape: got {[o.shape for o in out]}")

    total_tests += 1  # adding rings must not change an older self, bit for bit
    before = np.array(lineage_forward([r0], toks)[0])
    r1 = Ring(128, 48, 16, 2, 2, 32, 32)
    r2 = Ring(128, 64, 16, 2, 2, 32, 32)
    after = np.array(lineage_forward([r0, r1, r2], toks, which=[0])[0])
    if np.abs(before - after).max() != 0.0:
        failures.append(f"old self changed after growth: max diff {np.abs(before - after).max()}")

    total_tests += 1  # causality: a later token must not affect earlier predictions
    toks2 = np.array(toks)
    toks2[:, -1] = (toks2[:, -1] + 1) % 128
    a = np.array(lineage_forward([r0, r1], toks)[1])[:, :-1]
    b = np.array(lineage_forward([r0, r1], mx.array(toks2))[1])[:, :-1]
    if np.abs(a - b).max() > 1e-5:
        failures.append(f"causality broken: {np.abs(a - b).max()}")

    total_tests += 1  # the cached-trace path gives the same newest-self logits as the full pass
    _, tr = lineage_forward([r0, r1], toks, which=[], return_trace=True)
    fast = np.array(ring_forward(r2, toks, tr))
    slow = np.array(lineage_forward([r0, r1, r2], toks, which=[2])[0])
    first = np.array(ring_forward(r0, toks, None))
    if np.abs(fast - slow).max() > 1e-5 or np.abs(first - before).max() > 1e-5:
        failures.append(f"trace path differs: {np.abs(fast - slow).max()} / {np.abs(first - before).max()}")

    total_tests += 1  # a new residual ring starts as an exact copy of the previous self, and both paths agree
    rr = Ring(128, 80, 16, 2, 2, 32, 32, residual=True)
    prev_self = np.array(lineage_forward([r0, r1, r2], toks, which=[2])[0])
    as_new = np.array(lineage_forward([r0, r1, r2, rr], toks, which=[3])[0])
    _, tr3 = lineage_forward([r0, r1, r2], toks, which=[], return_trace=True)
    via_trace = np.array(ring_forward(rr, toks, tr3))
    rr.head.bias = mx.ones(rr.head.bias.shape) * 0.5
    shifted = np.array(lineage_forward([r0, r1, r2, rr], toks, which=[3])[0])
    if np.abs(as_new - prev_self).max() != 0.0 or np.abs(via_trace - prev_self).max() > 1e-5 or abs(float((shifted - prev_self).mean()) - 0.5) > 1e-4:
        failures.append("residual ring does not start as a copy of the previous self")

    total_tests += 1  # rings survive a save / load round trip exactly
    import os
    import tempfile
    tmp = os.path.join(tempfile.mkdtemp(), "rings.npz")
    save_rings(tmp, [r0, r1, r2])
    back = np.array(lineage_forward(load_rings(tmp), toks, which=[2])[0])
    if np.abs(back - slow).max() != 0.0:
        failures.append(f"save/load changed the model: {np.abs(back - slow).max()}")

    total_tests += 1  # clone equals original
    c = np.array(lineage_forward([clone_ring(r0)], toks)[0])
    if np.abs(c - before).max() != 0.0:
        failures.append("clone differs from original")

    if failures:
        print(f"VALIDATION FAILED - {len(failures)} of {total_tests} tests failed:")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print(f"VALIDATION PASSED - all {total_tests} tests produced expected results")
