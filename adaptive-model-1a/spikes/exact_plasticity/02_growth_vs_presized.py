"""Throwaway follow-up: is growth worth anything over pre-allocation once both use exact ridge? And how fragile is the certificate?"""
import numpy as np
from importlib import import_module as _im
_core = _im("01_core_checks")
from types import SimpleNamespace as _ns
globals().update({k: getattr(_core, k) for k in dir(_core) if not k.startswith("_")})
# originally: from exact_plasticity_spike import World, ExactLayer, run_stream, forgetting, acc, NCLS

S16 = [16] * 16; ORDER = list(range(16))

def prune_until(L, test, floor=0.995):
    """OBS-prune one neuron at a time while every task stays >= floor (accuracy read from test only to REPORT the frontier)."""
    sal, P = L.obs_saliency(); n0 = L.K.shape[0]
    while L.K.shape[0] > 1:
        sal = (L.W ** 2).sum(0) / np.diag(P); j = int(np.argmin(sal))
        saved = (L.K.copy(), L.theta.copy(), L.gain.copy(), L.birth.copy(), L.C.copy(), L.S.copy(), L.W.copy())
        P2 = L.obs_remove(j, P)
        if min(acc(L, *test[s]) for s in test) < floor:
            L.K, L.theta, L.gain, L.birth, L.C, L.S, L.W = saved; break
        P = P2
    return n0, L.K.shape[0]

print("=== A. Smallest network that still holds every task at >= 99.5% (256 facts), 3 seeds ===")
for seed in range(3):
    w = World(S16, seed)
    Lg = ExactLayer(128, seed); rg, _, _, _, tg = run_stream(w, ORDER, Lg, np.random.default_rng(200 + seed), verify=False)
    g0, g1 = prune_until(Lg, tg)
    Lp = ExactLayer(g0, 7 + seed, grow=False); rp, _, _, _, tp = run_stream(w, ORDER, Lp, np.random.default_rng(200 + seed), verify=False)
    p0, p1 = prune_until(Lp, tp)
    Lz = ExactLayer(0, seed); rz, _, _, _, tz = run_stream(w, ORDER, Lz, np.random.default_rng(200 + seed), verify=False)
    z0, z1 = prune_until(Lz, tz)
    print(f"seed {seed}: grow-on-128-base {g0}->{g1} (base left {int((Lg.birth<0).sum())}, grown left {int((Lg.birth>=0).sum())}) | "
          f"pre-allocated random {p0}->{p1} | pure growth {z0}->{z1}")

print("\n=== B. Pre-allocated random features, exact ridge, at smaller sizes (seed 0) ===")
w = World(S16, 0)
for M in (128, 192, 256, 331):
    best = max(np.mean(run_stream(w, ORDER, ExactLayer(M, 7, grow=False, base_thr=t), np.random.default_rng(200), verify=False)[0][-1]) for t in (0.0, 1.0))
    print(f"M={M}: final avg acc {best:.4f}")

print("\n=== C. Noisier inputs: does the method degrade honestly? (pure growth, 8 tasks) ===")
SIZES = [32, 32, 8, 32, 16, 32, 8, 24]
for noise in (0.2, 0.4, 0.6, 0.8):
    w = World(SIZES, 1, noise=noise); L = ExactLayer(0, 1)
    rows, added, exact, train, test = run_stream(w, list(range(8)), L, np.random.default_rng(5))
    tv = 0
    for era, t in enumerate(range(8)):
        tv += int((L.feats(test[t][0])[:, L.birth > era] > 0).sum())
    cos_same = float(np.mean([ (train[t][0][:20] @ train[t][0][0]).min() for t in range(8)]))
    print(f"noise {noise}: min cos within a fact ~{cos_same:.2f}  final avg acc {np.mean(rows[-1]):.4f}  forgetting {forgetting(rows):+.4f}  neurons {L.K.shape[0]} for {sum(SIZES)} facts  "
          f"centers {len(L.centers)}  cert fails {L.cert_fail}  train silence violations {sum(e[2] for e in exact)}  TEST-input activations of later neurons {tv}  max|W-Wjoint| {max(e[1] for e in exact):.1e}")

print("\n=== D. Storage accounting, run 2 seed 0 ===")
L = ExactLayer(128, 0); run_stream(World(S16, 0), ORDER, L, np.random.default_rng(200), verify=False)
M = L.K.shape[0]
print(f"neurons {M}: keys+thresholds {M*65}, readout {M*NCLS}, statistic C {M*(M+1)//2} (training-time only), support sketch {len(L.centers)} centres x 64 = {len(L.centers)*64} (training-time only; {int((L.c_neuron>=0).sum())} coincide with neuron keys)")
