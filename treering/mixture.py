"""Bayesian mixture over a model's past selves, and dreaming from that mixture.

Purpose
    Given the next-token logits of K selves on the same sequence, predict with a weighted
    average of the selves, where each weight is the posterior probability that this self is
    the right one given how well it has predicted the sequence so far.

    Two kinds of mixture are combined:
      alpha = 0   plain Bayes: one self is assumed right for the whole sequence;
      alpha > 0   "fixed share": at every step a fraction alpha of the weight is handed back
                  to all selves, so the mixture can change its mind mid-sequence.
    An outer Bayesian mixture over a few alpha values picks whichever works on the sequence.

    Guarantee, for every sequence x and every self k (A = number of alpha values):
        -log p_mix(x) <= -log p_k(x) + log K + log A.
    Whatever is learned later, the mixture is never worse than any past self by more than
    log(K A) nats per sequence.

Docs
    MLX: https://ml-explore.github.io/mlx/build/html/index.html
    Fixed share: Herbster & Warmuth (1998), "Tracking the best expert".
    Bayesian mixtures: Cesa-Bianchi & Lugosi (2006), "Prediction, Learning, and Games".

Sample input
    mix_lp, self_lp, weights = switching_logprobs([logits_a, logits_b], targets)

Expected output
    mix_lp:  (batch, time, vocab) log-probabilities of the mixture;
    self_lp: (K, batch, time) log-probability each self gave the true next token;
    weights: (K, batch) final posterior weight of each self.
"""
import math
from typing import List, Sequence, Tuple

import mlx.core as mx

import safety
from model import Ring, lineage_forward

ALPHAS: Tuple[float, ...] = (0.0, 0.02, 0.1)


def log_softmax(x: mx.array) -> mx.array:
    return x - mx.logsumexp(x, axis=-1, keepdims=True)


class MixState:
    """Running posterior over selves for each alpha, plus the posterior over alphas."""

    def __init__(self, k: int, batch: int, alphas: Sequence[float] = ALPHAS) -> None:
        self.k, self.alphas = k, tuple(alphas)
        self.log_prior = -math.log(k)
        self.log_w = [mx.full((k, batch), self.log_prior) for _ in alphas]
        self.log_om = [mx.full((batch,), -math.log(len(alphas))) for _ in alphas]

    def _omega(self) -> mx.array:
        om = mx.stack(self.log_om)
        return om - mx.logsumexp(om, axis=0, keepdims=True)

    def predict(self, lp_t: mx.array) -> mx.array:
        """lp_t: (K, B, V) log-probs of each self for the next token -> (B, V) mixture."""
        mixes = mx.stack([mx.logsumexp(w[..., None] + lp_t, axis=0) for w in self.log_w])
        return mx.logsumexp(self._omega()[..., None] + mixes, axis=0)

    def update(self, s_t: mx.array) -> None:
        """s_t: (K, B) log-prob each self gave the token that actually came next."""
        for i, a in enumerate(self.alphas):
            ll = mx.logsumexp(self.log_w[i] + s_t, axis=0)
            post = self.log_w[i] + s_t - ll
            self.log_w[i] = post if a == 0 else mx.logaddexp(post + math.log(1 - a), mx.array(math.log(a) + self.log_prior))
            self.log_om[i] = self.log_om[i] + ll

    def weights(self) -> mx.array:
        """(K, B) posterior weight of each self, averaged over alphas."""
        om = self._omega()
        return mx.exp(mx.logsumexp(om[:, None, :] + mx.stack(self.log_w), axis=0))


def switching_logprobs(logits: Sequence[mx.array], targets: mx.array,
                       alphas: Sequence[float] = ALPHAS) -> Tuple[mx.array, mx.array, mx.array]:
    k = len(logits)
    if k == 1:
        lp = log_softmax(logits[0])
        return lp, mx.take_along_axis(lp, targets[..., None], axis=-1)[..., 0][None], mx.ones((1, targets.shape[0]))
    lp = mx.stack([log_softmax(l) for l in logits])                       # (K, B, T, V)
    idx = mx.broadcast_to(targets[None, :, :, None], (k,) + tuple(targets.shape) + (1,))
    self_lp = mx.take_along_axis(lp, idx, axis=-1)[..., 0]                 # (K, B, T)
    state = MixState(k, targets.shape[0], alphas)
    outs = []
    for t in range(targets.shape[1]):
        outs.append(state.predict(lp[:, :, t, :]))
        state.update(self_lp[:, :, t])
    return mx.stack(outs, axis=1), self_lp, state.weights()


def dream(rings: Sequence[Ring], n: int, length: int, start_token: int, temperature: float = 1.0,
          alphas: Sequence[float] = (0.0,), chunk: int = 128) -> mx.array:
    """Sample n sequences from the mixture of all selves. No stored data is used.

    Every array keeps the same shape at every step: the model always runs on a full-length
    buffer and only position t is read (later positions cannot influence it, because the
    model is causal). An earlier version grew the sequence one token at a time; the changing
    shapes filled MLX's buffer cache to about 70 GB per process and crashed the machine.

    The default alpha of 0 draws each dream from one self, chosen as the dream unfolds.
    """
    k = len(rings)
    rings = list(rings)
    forward = mx.compile(lambda toks: mx.stack([log_softmax(l) for l in lineage_forward(rings, toks)]))
    columns = mx.arange(length)[None, :]
    out = []
    for _ in range(-(-n // chunk)):
        toks = mx.full((chunk, length), start_token, dtype=mx.int32)
        state = MixState(k, chunk, alphas)
        for t in range(length - 1):
            lp_t = forward(toks)[:, :, t, :]                                   # (K, chunk, V)
            nxt = mx.random.categorical(state.predict(lp_t) / temperature).astype(mx.int32)
            state.update(mx.take_along_axis(lp_t, mx.broadcast_to(nxt[None, :, None], (k, chunk, 1)), axis=-1)[..., 0])
            toks = mx.where(columns == t + 1, nxt[:, None], toks)
            mx.eval(toks, state.log_w, state.log_om)
        out.append(toks)
        safety.check("while dreaming")
    return mx.concatenate(out)[:n]


if __name__ == "__main__":
    import sys

    import numpy as np

    failures, total_tests = [], 0
    mx.random.seed(1)
    rings: List[Ring] = [Ring(64, 0, 32, 2, 2, 64, 24)]
    rings.append(Ring(64, 32, 16, 2, 2, 32, 24))
    rings.append(Ring(64, 48, 16, 2, 2, 32, 24))
    toks = mx.random.randint(0, 64, (16, 20))
    logits = lineage_forward(rings, toks[:, :-1])
    tgt = toks[:, 1:]

    total_tests += 1  # the guarantee: mixture <= best self + log K + log A, on every sequence
    mix_lp, self_lp, w = switching_logprobs(logits, tgt)
    mix_nll = -np.array(mx.take_along_axis(mix_lp, tgt[..., None], axis=-1)[..., 0]).sum(-1)
    best = (-np.array(self_lp).sum(-1)).min(0)
    bound = math.log(3) + math.log(len(ALPHAS))
    if (mix_nll - best).max() > bound + 1e-4:
        failures.append(f"guarantee violated: gap {(mix_nll - best).max():.4f} > {bound:.4f}")

    total_tests += 1  # plain Bayes alone obeys the tighter log K bound
    mix0, _, _ = switching_logprobs(logits, tgt, alphas=(0.0,))
    nll0 = -np.array(mx.take_along_axis(mix0, tgt[..., None], axis=-1)[..., 0]).sum(-1)
    if (nll0 - best).max() > math.log(3) + 1e-4:
        failures.append(f"plain Bayes bound violated: {(nll0 - best).max():.4f}")

    total_tests += 1  # mixture probabilities sum to one and weights sum to one
    if abs(np.exp(np.array(mix_lp)).sum(-1) - 1).max() > 1e-4 or abs(np.array(w).sum(0) - 1).max() > 1e-4:
        failures.append("mixture or weights not normalised")

    total_tests += 1  # a sequence that switches source mid-way is tracked by the switching mixture
    v, t_len = 8, 40
    seq = np.concatenate([np.zeros(t_len // 2, int), np.ones(t_len // 2, int)])[None]
    sure = lambda tok: mx.array(np.where(np.arange(v) == tok, 6.0, 0.0)[None, None].repeat(t_len, 1).astype(np.float32))
    sw, _, _ = switching_logprobs([sure(0), sure(1)], mx.array(seq))
    plain, _, _ = switching_logprobs([sure(0), sure(1)], mx.array(seq), alphas=(0.0,))
    nll = lambda lp: float(-np.array(mx.take_along_axis(lp, mx.array(seq)[..., None], axis=-1)).sum())
    if not nll(sw) < nll(plain) - 20:
        failures.append(f"switching did not help on a switching sequence: {nll(sw):.1f} vs {nll(plain):.1f}")

    total_tests += 1
    d = dream(rings, 8, 12, start_token=1, chunk=4)
    if d.shape != (8, 12) or int(d[:, 0].max()) != 1:
        failures.append(f"dream shape/start wrong: {d.shape}")

    total_tests += 1  # dreaming must not grow memory: this is what crashed the machine
    mx.clear_cache()
    base = safety.used_gb()
    dream(rings, 256, 24, start_token=1, chunk=64)
    grew = safety.used_gb() - base
    if grew > safety.CACHE_LIMIT_GB + 0.2:
        failures.append(f"dreaming grew MLX memory by {grew:.2f} GB")

    if failures:
        print(f"VALIDATION FAILED - {len(failures)} of {total_tests} tests failed:")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print(f"VALIDATION PASSED - all {total_tests} tests produced expected results")
