"""Throwaway verification spike: exact structural plasticity from one sufficient statistic.

Claim under test: if every neuron added to a memory layer is silent on all data seen
before its birth, the layer's accumulated second-moment matrix C = sum(h h^T) stays an
exact summary of all past data, so (a) learning is a ridge solve, (b) growing is
zero-padding C, (c) pruning is an Optimal Brain Surgeon downdate, and the continually
trained layer equals the layer trained on all data at once.

Not product code. numpy only. Run: python exact_plasticity_spike.py
"""
import numpy as np

D, NSYM, NCLS = 64, 32, 32


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


class World:
    """Stand-in for a frozen transformer body: layer input = tag + symbol + noise, normalised."""

    def __init__(self, sizes, seed, noise=0.2):
        r = np.random.default_rng(seed)
        self.noise = noise
        self.E_sym = unit(r.standard_normal((NSYM, D)))
        self.E_tag = unit(r.standard_normal((len(sizes), D)))
        self.tables = [(r.choice(NSYM, n, replace=False), r.integers(0, NCLS, n)) for n in sizes]

    def sample(self, t, per_fact, r):
        syms, labels = self.tables[t]
        x, y = np.repeat(syms, per_fact), np.repeat(labels, per_fact)
        u = self.E_tag[t] + self.E_sym[x] + self.noise * r.standard_normal((len(x), D)) / np.sqrt(D)
        return unit(u), y


def margins(out, y):
    n = np.arange(len(y))
    corr = out[n, y]
    o = out.copy()
    o[n, y] = -np.inf
    return corr - o.max(1)


class ExactLayer:
    def __init__(self, m0, seed, lam=1e-3, theta0=0.8, rho_assign=0.9, margin=0.3,
                 silent=True, theta_loose=0.2, grow=True, base_thr=1.0, tau_old=1e-3):
        r = np.random.default_rng(seed)
        self.K = unit(r.standard_normal((m0, D))) if m0 else np.zeros((0, D))
        self.theta = np.full(m0, base_thr / np.sqrt(D))
        self.gain = np.full(m0, np.sqrt(D))
        self.birth = np.full(m0, -1)
        self.C = np.zeros((m0, m0)); self.S = np.zeros((NCLS, m0)); self.W = np.zeros((NCLS, m0))
        self.N = 0
        self.lam, self.theta0, self.rho_assign, self.margin = lam, theta0, rho_assign, margin
        self.silent, self.theta_loose, self.grow, self.tau_old = silent, theta_loose, grow, tau_old
        self.centers = np.zeros((0, D)); self.rho = np.zeros(0)
        self.c_era = np.zeros(0, int); self.c_neuron = np.zeros(0, int)
        self.cert_fail = 0; self.conflicts = 0; self.ball_ok = 0; self.ball_total = 0
        self.era_ball = []  # (mu, R) per era, for the cheap-sketch check

    def feats(self, U):
        return self.gain * np.maximum(U @ self.K.T - self.theta, 0.0)

    def _solve(self, C, S):
        return np.linalg.solve(C + self.lam * np.eye(len(C)), S.T).T

    def predict(self, U):
        return self.feats(U) @ self.W.T

    def sse_old(self, W):
        """Squared error of W on ALL past data, computed from statistics only."""
        m = self.C.shape[0]
        Wo = W[:, :m]
        return self.N - 2 * np.sum(Wo * self.S) + np.sum((Wo @ self.C) * Wo)

    def _add_neuron(self, k, theta, era):
        self.K = np.vstack([self.K, k]); self.theta = np.append(self.theta, theta)
        self.gain = np.append(self.gain, 1.0 / (1.0 - theta)); self.birth = np.append(self.birth, era)
        m = len(self.C)
        C = np.zeros((m + 1, m + 1)); C[:m, :m] = self.C; self.C = C  # zero padding: valid iff silent on the past
        self.S = np.hstack([self.S, np.zeros((NCLS, 1))]); self.W = np.hstack([self.W, np.zeros((NCLS, 1))])
        return m

    def _try_allocate(self, j, era):
        if self.c_neuron[j] >= 0:
            self.conflicts += 1
            return False
        k = self.centers[j]
        if self.silent:
            old = np.where(self.c_era < era)[0]
            theta_req = -1.0
            if len(old):
                ang = np.arccos(np.clip(self.centers[old] @ k, -1, 1))
                alpha = np.arccos(np.clip(self.rho[old], -1, 1))
                theta_req = np.cos(np.maximum(ang - alpha, 0.0)).max() + 0.02
            theta = max(self.theta0, theta_req)
            if theta >= self.rho_assign - 0.01:
                self.cert_fail += 1
                return False
            if self.era_ball:  # would a (mean, radius) ball per era have been enough?
                self.ball_total += 1
                self.ball_ok += max(k @ mu + R for mu, R in self.era_ball) + 0.02 < self.rho_assign - 0.01
        else:
            theta = self.theta_loose
        self.c_neuron[j] = self._add_neuron(k, theta, era)
        return True

    def learn(self, U, Y, era):
        Yoh = np.eye(NCLS)[Y]
        assign = np.empty(len(U), int)
        for i, u in enumerate(U):  # support sketch: leader clustering on the sphere
            if len(self.centers):
                s = self.centers @ u; j = int(np.argmax(s))
                if s[j] >= self.rho_assign:
                    assign[i] = j; self.rho[j] = min(self.rho[j], s[j]); continue
            self.centers = np.vstack([self.centers, u]); self.rho = np.append(self.rho, 1.0)
            self.c_era = np.append(self.c_era, era); self.c_neuron = np.append(self.c_neuron, -1)
            assign[i] = len(self.centers) - 1
        added = 0
        if self.grow:
            sse_before = self.sse_old(self.W) if self.N else 0.0
            for _ in range(8):
                H = self.feats(U)
                if H.shape[1]:
                    Wt = self._solve(self.C + H.T @ H, self.S + Yoh.T @ H)
                    marg = margins(H @ Wt.T, Y)
                    hurt = (self.sse_old(Wt) - sse_before) / max(self.N, 1) if self.N else 0.0
                else:
                    marg = np.full(len(U), -1.0); hurt = 0.0
                cand = sorted(set(assign[marg < self.margin]))
                if not cand and hurt > self.tau_old:  # blocked by old knowledge: absorbing would damage it
                    fresh = [j for j in set(assign) if self.c_neuron[j] < 0]
                    order = sorted(fresh, key=lambda j: marg[assign == j].min())
                    cand = order[:max(1, len(order) // 2)]
                if not cand:
                    break
                got = sum(self._try_allocate(j, era) for j in cand)
                added += got
                if not got:
                    break
        H = self.feats(U)
        self.C += H.T @ H; self.S += Yoh.T @ H; self.N += len(U)
        self.W = self._solve(self.C, self.S)
        self.era_ball.append((U.mean(0), np.linalg.norm(U - U.mean(0), axis=1).max()))
        return added

    def obs_saliency(self):
        P = np.linalg.inv(self.C + self.lam * np.eye(len(self.C)))
        return (self.W ** 2).sum(0) / np.diag(P), P

    def obs_remove(self, j, P):
        keep = np.arange(len(self.C)) != j
        self.W = self.W[:, keep] - np.outer(self.W[:, j], P[j, keep]) / P[j, j]
        P2 = P[np.ix_(keep, keep)] - np.outer(P[keep, j], P[j, keep]) / P[j, j]
        self.K, self.theta, self.gain, self.birth = self.K[keep], self.theta[keep], self.gain[keep], self.birth[keep]
        self.C, self.S = self.C[np.ix_(keep, keep)], self.S[:, keep]
        return P2


def acc(layer, U, Y):
    return float((layer.predict(U).argmax(1) == Y).mean()) if layer.K.shape[0] else 0.0


def run_stream(world, order, layer, r, per_fact=20, verify=True):
    train, rows, added, exact = {}, [], [], []
    test = {t: world.sample(t, 10, r) for t in order}
    for era, t in enumerate(order):
        U, Y = world.sample(t, per_fact, r)
        train[t] = (U, Y, era)
        added.append(layer.learn(U, Y, era))
        rows.append([acc(layer, *test[s]) for s in order[:era + 1]])
        if verify:
            Ua = np.vstack([train[s][0] for s in order[:era + 1]]); Ya = np.concatenate([train[s][1] for s in order[:era + 1]])
            Ha = layer.feats(Ua)
            Cj = Ha.T @ Ha; Wj = layer._solve(Cj, np.eye(NCLS)[Ya].T @ Ha)
            viol = 0
            for s in order[:era + 1]:
                Us, _, es = train[s]
                viol += int((layer.feats(Us)[:, layer.birth > es] > 0).sum())
            exact.append((np.abs(Cj - layer.C).max(), np.abs(Wj - layer.W).max(), viol) if layer.C.size else (0.0, 0.0, viol))
    return rows, added, exact, train, test


def forgetting(rows):
    T = len(rows)
    return float(np.mean([max(rows[t][j] for t in range(j, T - 1)) - rows[-1][j] for j in range(T - 1)]))


def sgd_seq(world, order, M, r, per_fact=20, epochs=150, lr=3e-3):
    W1 = r.standard_normal((M, D)) * np.sqrt(2 / D); b1 = np.zeros(M)
    W2 = r.standard_normal((NCLS, M)) / np.sqrt(M); b2 = np.zeros(NCLS)
    P = [W1, b1, W2, b2]; m = [np.zeros_like(p) for p in P]; v = [np.zeros_like(p) for p in P]; step = 0
    test = {t: world.sample(t, 10, r) for t in order}

    def fwd(U):
        H = np.maximum(U * np.sqrt(D) @ P[0].T + P[1], 0)
        return H, H @ P[2].T + P[3]

    rows = []
    for era, t in enumerate(order):
        U, Y = world.sample(t, per_fact, r)
        for _ in range(epochs):
            perm = r.permutation(len(U))
            for i in range(0, len(U), 64):
                idx = perm[i:i + 64]; Ub, Yb = U[idx], Y[idx]
                H, Z = fwd(Ub)
                Z = Z - Z.max(1, keepdims=True); p = np.exp(Z); p /= p.sum(1, keepdims=True)
                G = p; G[np.arange(len(Yb)), Yb] -= 1; G /= len(Yb)
                dH = (G @ P[2]) * (H > 0)
                grads = [dH.T @ (Ub * np.sqrt(D)), dH.sum(0), G.T @ H, G.sum(0)]
                step += 1
                for k in range(4):
                    m[k] = 0.9 * m[k] + 0.1 * grads[k]; v[k] = 0.999 * v[k] + 0.001 * grads[k] ** 2
                    P[k] -= lr * (m[k] / (1 - 0.9 ** step)) / (np.sqrt(v[k] / (1 - 0.999 ** step)) + 1e-8)
            if (fwd(U)[1].argmax(1) == Y).mean() >= 0.995:
                break
        rows.append([float((fwd(test[s][0])[1].argmax(1) == test[s][1]).mean()) for s in order[:era + 1]])
    return rows


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True, linewidth=160)
    failures = []
    SIZES = [32, 32, 8, 32, 16, 32, 8, 24]
    print("=== 1. Pure growth (no base features), 8-task stream, 5 seeds ===")
    for seed in range(5):
        w = World(SIZES, seed); r = np.random.default_rng(100 + seed); L = ExactLayer(0, seed)
        rows, added, exact, train, test = run_stream(w, list(range(8)), L, r)
        cmax, wmax, viol = max(e[0] for e in exact), max(e[1] for e in exact), sum(e[2] for e in exact)
        print(f"seed {seed}: final avg acc {np.mean(rows[-1]):.4f}  forgetting {forgetting(rows):+.4f}  neurons/task {added}  "
              f"facts/task {SIZES}  max|C-Cjoint| {cmax:.1e}  max|W-Wjoint| {wmax:.1e}  silence violations {viol}  cert fails {L.cert_fail}")
        if viol or cmax > 1e-9 or wmax > 1e-6: failures.append(f"pure growth seed {seed} not exact")
        if np.mean(rows[-1]) < 0.99: failures.append(f"pure growth seed {seed} accuracy {np.mean(rows[-1]):.3f}")

    print("\n=== 2. Mixed: 128 shared random base features + silent growth, 16 tasks x 16 facts ===")
    S16 = [16] * 16
    for seed in range(3):
        w = World(S16, seed); r = np.random.default_rng(200 + seed); L = ExactLayer(128, seed)
        rows, added, exact, train, test = run_stream(w, list(range(16)), L, r)
        cmax, wmax, viol = max(e[0] for e in exact), max(e[1] for e in exact), sum(e[2] for e in exact)
        print(f"seed {seed}: final avg acc {np.mean(rows[-1]):.4f}  forgetting {forgetting(rows):+.4f}  neurons added/task {added}  total {128 + sum(added)}")
        print(f"         max|C-Cjoint| {cmax:.1e}  max|W-Wjoint| {wmax:.1e}  silence violations {viol}  cert fails {L.cert_fail}  ball-sketch enough {L.ball_ok}/{L.ball_total}")
        if viol or cmax > 1e-9 or wmax > 1e-6: failures.append(f"mixed seed {seed} not exact")
        if seed == 0: keepL, keep_train, keep_test, keep_rows = L, train, test, rows

    print("\n=== 3. Ablation: same as 2 but new neurons NOT silent (threshold 0.2) ===")
    w = World(S16, 0); r = np.random.default_rng(200); L = ExactLayer(128, 0, silent=False)
    rows, added, exact, _, _ = run_stream(w, list(range(16)), L, r)
    print(f"final avg acc {np.mean(rows[-1]):.4f}  forgetting {forgetting(rows):+.4f}  first-4-task acc now {np.round(rows[-1][:4], 3)}  "
          f"max|C-Cjoint| {max(e[0] for e in exact):.2f}  max|W-Wjoint| {max(e[1] for e in exact):.2f}  silence violations {sum(e[2] for e in exact)}")
    if max(e[0] for e in exact) < 1e-6: failures.append("ablation unexpectedly exact")

    print("\n=== 4. Baselines at the same final neuron count as run 2 seed 0 ===")
    M = keepL.K.shape[0]; w = World(S16, 0)
    for thr in (0.0, 1.0, 2.0):
        Lp = ExactLayer(M, 7, grow=False, base_thr=thr)
        rp, _, _, _, _ = run_stream(w, list(range(16)), Lp, np.random.default_rng(200), verify=False)
        print(f"pre-allocated random features, exact ridge, M={M}, bias {thr}: final avg acc {np.mean(rp[-1]):.4f}  forgetting {forgetting(rp):+.4f}")
    rs = sgd_seq(w, list(range(16)), M, np.random.default_rng(200))
    print(f"SGD (Adam) sequential MLP, M={M}: final avg acc {np.mean(rs[-1]):.4f}  forgetting {forgetting(rs):+.4f}  learning acc {np.mean([rs[t][t] for t in range(16)]):.4f}")
    print(f"exact growth (run 2 seed 0), M={M}: final avg acc {np.mean(keep_rows[-1]):.4f}  forgetting {forgetting(keep_rows):+.4f}")

    print("\n=== 5. Data volume is not knowledge: replay task 0 with fresh samples ===")
    before = keepL.K.shape[0]
    add = keepL.learn(*World(S16, 0).sample(0, 20, np.random.default_rng(999)), era=16)
    print(f"neurons added on re-exposure: {add} (before {before}, after {keepL.K.shape[0]})")
    if add: failures.append("re-exposure caused growth")

    print("\n=== 6. Pruning from statistics only (Optimal Brain Surgeon on accumulated C) ===")
    Ua = np.vstack([keep_train[s][0] for s in range(16)]); Ya = np.concatenate([keep_train[s][1] for s in range(16)])
    Ua = np.vstack([Ua, World(S16, 0).sample(0, 20, np.random.default_rng(999))[0]]); Ya = np.concatenate([Ya, World(S16, 0).sample(0, 20, np.random.default_rng(999))[1]])
    Yoh = np.eye(NCLS)[Ya]

    def J_data(L_):
        return float(((Yoh - L_.predict(Ua)) ** 2).sum() + L_.lam * (L_.W ** 2).sum())

    sal, P = keepL.obs_saliency(); j0 = int(np.argmin(sal)); J0 = J_data(keepL)
    pred = sal[j0]; P = keepL.obs_remove(j0, P); dJ = J_data(keepL) - J0
    Wre = keepL._solve(keepL.C, keepL.S)
    print(f"single removal: predicted objective increase {pred:.6e}, measured on all data {dJ:.6e}, downdate vs re-solve max diff {np.abs(Wre - keepL.W).max():.1e}")
    if abs(pred - dJ) > 1e-6 * max(1, abs(dJ)) + 1e-8: failures.append("OBS prediction mismatch")
    removed, budget, spent = 1, 0.01 * len(Ua), pred
    while True:
        sal = (keepL.W ** 2).sum(0) / np.diag(P); j = int(np.argmin(sal))
        if spent + sal[j] > budget: break
        spent += sal[j]; P = keepL.obs_remove(j, P); removed += 1
    accs = [acc(keepL, *keep_test[s]) for s in range(16)]
    print(f"pruned {removed} of {M} neurons within a budget of 0.01 squared error per example; base features left {int((keepL.birth < 0).sum())}/128, grown left {int((keepL.birth >= 0).sum())}; "
          f"avg acc after {np.mean(accs):.4f} (min task {min(accs):.3f})")

    print("\n=== 7. Order independence: run 2 seed 0 with tasks reversed ===")
    w = World(S16, 0); L = ExactLayer(128, 0)
    rr, ar, er, _, _ = run_stream(w, list(range(16))[::-1], L, np.random.default_rng(321))
    print(f"reversed: final avg acc {np.mean(rr[-1]):.4f}  forgetting {forgetting(rr):+.4f}  total neurons {L.K.shape[0]}  max|W-Wjoint| {max(e[1] for e in er):.1e}")

    print("\n=== 8. Long stream: 2 base tasks + 50 tasks of 8 facts, no base features, d=64 ===")
    SL = [32, 32] + [8] * 50
    w = World(SL, 3); L = ExactLayer(0, 3)
    rl, al, el, _, _ = run_stream(w, list(range(52)), L, np.random.default_rng(55), verify=True)
    print(f"final avg acc {np.mean(rl[-1]):.4f}  forgetting {forgetting(rl):+.4f}  neurons per 8-fact task: first ten {al[2:12]} last ten {al[-10:]}")
    print(f"cert fails {L.cert_fail}  silence violations {sum(e[2] for e in el)}  max|W-Wjoint| {max(e[1] for e in el):.1e}  ball-sketch enough {L.ball_ok}/{L.ball_total}")
    worst = min(rl[-1]); print(f"worst single task at end {worst:.3f}")

    print()
    if failures:
        print(f"VERIFICATION FAILED - {len(failures)} check(s):"); [print("  -", f) for f in failures]; raise SystemExit(1)
    print("VERIFICATION PASSED - all exactness checks held (see numbers above for the claims that are measurements, not checks)")
