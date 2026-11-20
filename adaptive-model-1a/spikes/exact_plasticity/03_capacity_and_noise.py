"""Throwaway follow-up 2: (E) what growth is actually for, (F) a statistical silence rule for noisier inputs."""
import numpy as np
from importlib import import_module as _im
_core = _im("01_core_checks")
globals().update({k: getattr(_core, k) for k in dir(_core) if not k.startswith("_")})
# originally: from exact_plasticity_spike import World, ExactLayer, run_stream, forgetting, acc, D

print("=== E. Knowledge exceeds a fixed network: 40 tasks x 16 facts = 640 facts (seed 0) ===")
S = [16] * 40; ORDER = list(range(40)); w = World(S, 0)
for M in (128, 256, 512):
    best = None
    for t in (0.0, 1.0):
        L = ExactLayer(M, 7, grow=False, base_thr=t); rows = run_stream(w, ORDER, L, np.random.default_rng(1), verify=False)[0]
        cur = (np.mean(rows[-1]), min(rows[-1]), forgetting(rows), [round(float(np.mean(rows[k])), 3) for k in (9, 19, 29, 39)])
        best = cur if best is None or cur[0] > best[0] else best
    print(f"fixed {M} dense features, exact ridge: final avg acc {best[0]:.4f} (worst task {best[1]:.3f})  forgetting {best[2]:+.4f}  avg acc after 10/20/30/40 tasks {best[3]}")
L = ExactLayer(128, 0); rows, added, exact, _, _ = run_stream(w, ORDER, L, np.random.default_rng(1), verify=True)
print(f"128 dense + silent growth when blocked: final avg acc {np.mean(rows[-1]):.4f} (worst task {min(rows[-1]):.3f})  forgetting {forgetting(rows):+.4f}  "
      f"neurons {L.K.shape[0]}  added per task: first 5 {added[:5]} last 5 {added[-5:]}  max|W-Wjoint| {max(e[1] for e in exact):.1e}  cert fails {L.cert_fail}")


class StatLayer(ExactLayer):
    """Silence threshold from a noise estimate instead of worst-case geometry (a probabilistic, not hard, guarantee)."""
    z = 5.0

    def _try_allocate(self, j, era):
        if self.c_neuron[j] >= 0:
            self.conflicts += 1; return False
        k = self.centers[j]; old = np.where(self.c_era < era)[0]; req = -1.0
        if len(old):
            s = np.sqrt(max(1.0 - self.rho.min(), 0.0))  # per-copy off-centre norm, from observed within-region cosine
            req = (self.centers[old] @ k).max() + self.z * np.sqrt(2.0) * s / np.sqrt(D)
        theta = max(min(self.theta0, self.rho[j] - 0.05), req)
        if theta >= self.rho[j] - 0.01:
            self.cert_fail += 1; return False
        self.c_neuron[j] = self._add_neuron(k, theta, era); return True


print("\n=== F. Noisier inputs, hard certificate vs statistical rule (pure growth, 8 tasks, 184 facts, 3 seeds) ===")
SIZES = [32, 32, 8, 32, 16, 32, 8, 24]
for noise, rho_a in ((0.2, 0.9), (0.3, 0.85), (0.4, 0.78), (0.5, 0.7)):
    for name, cls in (("hard", ExactLayer), ("stat", StatLayer)):
        res = []
        for seed in range(3):
            w = World(SIZES, seed, noise=noise); L = cls(0, seed, rho_assign=rho_a, theta0=min(0.8, rho_a - 0.05))
            rows, added, exact, train, test = run_stream(w, list(range(8)), L, np.random.default_rng(50 + seed))
            res.append((np.mean(rows[-1]), forgetting(rows), L.K.shape[0], L.cert_fail, sum(e[2] for e in exact), max(e[1] for e in exact)))
        r = np.array(res)
        print(f"noise {noise} {name}: final avg acc {r[:,0].mean():.4f}  forgetting {r[:,1].mean():+.4f}  neurons {r[:,2].mean():.0f}  not-certifiable {r[:,3].mean():.0f}  "
              f"activations on pre-birth data {int(r[:,4].sum())}  max|W-Wjoint| {r[:,5].max():.1e}")
