# Loss of Plasticity in Neural Networks (research 2024 through November 2026)

Scope and verification notes (read first):
- Research date: 2026-11-19. Searches surfaced nothing dated September to November 2026; the newest primary sources found are from June to August 2026.
- Numbers below were extracted from arXiv abstract/HTML pages and the PMC full text of the Nature paper via an automated page reader. Several HTML pages were truncated at 100k characters, and many papers report results only in figures. Anything I could not confirm from the primary text is marked "(unverified)" or listed under Gaps.
- "LoP" = loss of plasticity. "CBP" = continual backpropagation. "S&P" = shrink and perturb. "LN" = layer normalization. "WD" = weight decay.
- Background (pre-2024) papers are labelled "Background".

## 1. What are the leading mechanistic explanations as of 2024-2026, and which are supported or contested?

### Takeaway
The field has moved from single-cause stories (dead units, rank collapse, weight growth) to a multi-mechanism view: LoP decomposes into several independent mechanisms, and no single measured statistic explains it in all settings. The newest theory (2025-2026) tries to unify the symptoms as one geometric fact, either trapping on invariant manifolds of frozen and duplicated units, or spectral degeneration of the Hessian / NTK at the start of a new task; dormant units and low rank are increasingly described as symptoms rather than root causes.

### Cited Findings

Multi-mechanism view
- Lyle et al. (arXiv:2402.18762, submitted 2024-02-29, published CoLLAs 2025) argue LoP "can be decomposed into multiple independent mechanisms"; intervening on one is not enough, combining interventions is more robust. Mechanisms identified: preactivation distribution shift (producing dead units, degraded signal propagation, and near-linear "zombie" units), parameter norm growth, regression target scale, and abruptness of distribution shift. — [Lyle et al. 2024, abstract](https://arxiv.org/abs/2402.18762); [full text](https://arxiv.org/html/2402.18762); [CoLLAs proceedings](https://proceedings.mlr.press/v274/lyle25a.html)
- Same paper: re-randomizing 1% of labels barely hurts, while 10% or more causes rapid decline; dead units spike right after a label re-randomization and partially recover; norm growth is not monotonic with LoP (fully saturated networks stop growing their norm). — [Lyle et al. 2024](https://arxiv.org/html/2402.18762)
- Same paper: networks that lost plasticity through different mechanisms showed a similar empirical-NTK pathology (off-diagonal entries close to a low-rank matrix). — [Lyle et al. 2024](https://arxiv.org/html/2402.18762)

Dead / dormant / saturated units
- Dohare et al. (Nature 632(8026):768-774, published 2024-08-21; preprint arXiv:2306.13812) report up to 25% of units dead after 800 tasks in Online Permuted MNIST with backprop (step size 0.01); with Adam the dead-unit fraction plateaued near 60%; CBP kept almost zero dead units. LoP correlated with dead units, large weights, and falling effective rank. — [Dohare et al. 2024, PMC full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC11338828/)
- Klein et al. survey (arXiv:2411.04832, v1 2024-11-07, v3 2026-04-18): saturated and dormant neurons are "often seen as symptoms of reduced capacity rather than root causes"; no common definition of saturation exists. — [Klein et al. survey](https://arxiv.org/html/2411.04832)
- He (arXiv:2603.21173, 2026-03-22) argues dormancy and effective rank describe symptoms; claims neuron dormancy is equivalent to zero-gradient states, and that networks with high dormancy on one task can perform on par with randomly initialized networks on a substantially different task (abstract-level claims only). — [He 2026](https://arxiv.org/abs/2603.21173)
- In GPT-style LLMs, dormant MLP units did not consistently track LoP: the 12M model had the most severe LoP with no clear dormant-unit trend, while the 53M model had more than 95% dormant units in layer 8. — [Hernandez-Garcia, Figliolia, Millidge 2026](https://arxiv.org/html/2606.24752)

Frozen and cloned units; rank collapse
- Joudaki et al. (arXiv:2510.00304, v1 2025-09-30, ICLR 2026) define LoP formally: a manifold induces LoP if the loss gradient is tangent to it everywhere, so gradient flow cannot leave. Two trap types: frozen-unit manifolds (saturated units with zero derivative on all inputs) and cloned-unit manifolds (blocks of units with equal row/column sums that share activations and backward errors). Adam also stays trapped on cloned manifolds. — [Joudaki et al. 2025](https://arxiv.org/html/2510.00304v2); [ICLR 2026 proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/511c7fd69db9f1ce7492a57285975849-Abstract-Conference.html)
- Same paper, "rank-plasticity tension": properties that help static generalization (low-rank features, simplicity bias, neural collapse) push networks toward these traps; as the decorrelation strength of the nonlinearity approaches 1, the probability a unit is frozen approaches 1. — [Joudaki et al. 2025](https://arxiv.org/html/2510.00304v2)
- Klein et al.: feature rank collapse is often correlated with performance decline, but the link is "not as straightforward" in offline RL, and the causal link to dead neurons is unclear. — [Klein et al. survey](https://arxiv.org/html/2411.04832)

Weight norm growth and effective learning rate
- Dohare et al.: weight magnitude grew steadily under backprop and Adam; L2, S&P and CBP kept it low or constant. — [Dohare et al. 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11338828/)
- Lyle et al. (arXiv:2407.01800, 2024-07-01, NeurIPS 2024): with normalization layers, parameter-norm growth is equivalent to decay of the effective learning rate, which can decay toward zero too quickly in continual settings. — [Lyle et al. NaP](https://arxiv.org/abs/2407.01800)
- Klein et al. list four proposed mechanisms for norm growth harming plasticity: sharper loss landscapes, saturated nonlinearities, collinear gradients, and altered effective learning rates. — [Klein et al. survey](https://arxiv.org/html/2411.04832)
- Juliani and Ash (arXiv:2405.19153, NeurIPS 2024): in on-policy RL, weight magnitude was significantly correlated with normalized reward (training GLM coefficient -1.83, p=0.004); dead units were significant in the test GLM (coefficient -21.63, p=0.003) but not the training GLM; authors note correlation is not causation. — [Juliani and Ash 2024](https://arxiv.org/html/2405.19153)
- Contested in LLMs: average absolute parameter magnitude generally rose but did not consistently track LoP (the 106M model's magnitude decreased between cycles 8 and 20 while performance deteriorated). — [Hernandez-Garcia et al. 2026](https://arxiv.org/html/2606.24752)

Curvature / Hessian / sharpness
- Background: Lyle et al. (arXiv:2303.01486, 2023-03-02, ICML 2023) tied LoP to changes in loss-landscape curvature, often without saturated units. — [Lyle et al. 2023](https://arxiv.org/abs/2303.01486)
- Lewandowski et al. (arXiv:2312.00246, v1 2023-11-30, v4 2024-10-05): networks lose directions of curvature during training and this coincides with LoP on MNIST, CIFAR-10 and ImageNet; "previous explanations are insufficient to explain loss of plasticity in all settings". — [Lewandowski et al.](https://arxiv.org/abs/2312.00246)
- Prakash et al. (arXiv:2509.22335, v1 2025-09-26, v3 2026-05-29): LoP is preceded by "Hessian spectral collapse at new-task initialization"; they derive an epsilon-rank condition (number of "fast" eigenvalues of the Gram matrix) necessary for successful training, and argue dormant units, low feature rank and norm growth are "different facets of the same phenomenon". — [Prakash et al. 2025](https://arxiv.org/html/2509.22335)
- Contested: Klein et al. state "sharpness cannot be the sole mechanism" because resets outperform SAM even with higher curvature. — [Klein et al. survey](https://arxiv.org/html/2411.04832)

NTK conditioning, gradient diversity, churn
- Rosseau, Müller, Nowé (arXiv:2606.09762, 2026-06-08, ICML 2026): first-order progress is governed by the Rayleigh quotient of the empirical NTK; an anisotropic NTK stalls small-eigenvalue directions; layer-wise dynamical isometry (Jacobian singular values near one) is proposed as the property to preserve. — [Rosseau et al. 2026](https://arxiv.org/html/2606.09762)
- Wu et al. (arXiv:2604.01913, 2026-04-02): theory for online RL attributes LoP to (1) rank collapse of the NTK Gram matrix and (2) gradient magnitude decaying at a Theta(1/k) rate (abstract-level). — [Wu et al. 2026](https://arxiv.org/abs/2604.01913)
- Tang et al. (arXiv:2506.00592, 2025-05-31, ICML 2025): LoP coincides with rising "churn" (output variability on out-of-batch data), linked to a gradual drop in NTK rank. — [Tang et al. 2025](https://arxiv.org/abs/2506.00592)
- Lewandowski et al. (arXiv:2406.06811, 2024-06-10): keeping each layer's maximum singular value near one is claimed to maintain gradient diversity and trainability. — [Lewandowski et al. spectral regularization](https://arxiv.org/abs/2406.06811)

Target scale and optimization-trap views
- Lyle et al. 2024: large regression-target offsets during pretraining reduce later fine-tuning in a dose-response pattern (offset gets encoded in one feature direction and ill-conditions the features); LN reduces but does not eliminate this. — [Lyle et al. 2024](https://arxiv.org/html/2402.18762)
- He 2026 "OCP hypothesis": optima from earlier tasks become poor local optima for new tasks and trap parameters at task transitions (abstract-level). — [He 2026](https://arxiv.org/abs/2603.21173)
- Kim (arXiv:2604.04154, 2026-04-05, cond-mat.stat-mech): models the EWC penalty as an energy barrier growing linearly with task count, giving exponential collapse of the Kramers escape rate; this is a toy double-well model of the stability-plasticity dilemma, not of unregularized LoP. — [Kim 2026](https://arxiv.org/abs/2604.04154)

Is it intrinsic or an artifact of abrupt shift?
- Liu and Mou (arXiv:2602.09234, v1 2026-02-09, v2 2026-06-16): severity of LoP is tied to abruptness of task transitions; abrupt changes cause accuracy drops after about 20 tasks on Random MNIST and 12 tasks on Bigram Cipher, while an interpolation step size of 0.02 keeps near-perfect accuracy. — [Liu and Mou 2026](https://arxiv.org/html/2602.09234v2)
- Counter-evidence: LoP also appears under stationary multilingual LLM training with no task switches (5M, 12M, 27M models). — [Hernandez-Garcia et al. 2026](https://arxiv.org/html/2606.24752)
- Dohare et al. also observed collapse under stationary conditions in RL: PPO on Ant with fixed friction improved for about 3 million steps and then degraded. — [Dohare et al. 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11338828/)

### Inferences
- Supported by multiple independent sources: unbounded weight growth, unit saturation/death, duplicated units and rank loss, and spectral degeneration of the NTK/Hessian all co-occur with LoP. Contested: whether any one is causal. The strongest position the evidence supports is "several partially independent failure modes with a shared geometric signature (few usable gradient/curvature directions at the start of a new task)".
- For a growing/pruning FFN experiment, Joudaki et al.'s cloned-unit manifold is directly relevant: neurons added by function-preserving duplication start on a manifold plain SGD and Adam do not leave; added neurons need symmetry breaking (random init, noise).
- Abrupt task switches in a small task sequence will likely exaggerate LoP relative to gradual drift, so the experiment's task-transition schedule is itself an experimental variable.

### Gaps
- No causal intervention study isolating one mechanism in transformers was found; the LLM study is explicitly correlational and reports no "smoking gun".
- The experiments section of Prakash et al. (spectral collapse) was truncated in the fetched page, so its head-to-head evidence that spectral collapse precedes other diagnostics could not be verified.
- He 2026, Wu 2026 and Tang 2025 were read at abstract level only.

## 2. What do the main mitigation methods do, and how well do they work?

### Takeaway
Every method family works somewhere and fails somewhere; the most consistently effective recipes in independent comparisons are norm control combined with something that restores unit activity (LN + weight decay; L2 Init; S&P; reset plus L2). Unit-reset methods (CBP, ReDo) are the most setting-sensitive: they are reported unstable with layer norm, in small networks, and in transformers, where regularizing attention weights matters more than resetting MLP units.

### Cited Findings

Cross-method table with exact numbers (Farias and Jozefiak, SNR paper; arXiv:2410.20098, first posted October 2024, ICLR 2025). Average accuracy on the last 10% of tasks, 5 seeds; PM = Permuted MNIST, RM = Random Label MNIST, RC = Random Label CIFAR, CI = Continual ImageNet:

| Algorithm | SGD: PM / RM / RC / CI | Adam: PM / RM / RC / CI |
|---|---|---|
| No intervention | 0.71 / 0.11 / 0.18 / 0.78 | 0.64 / 0.11 / 0.15 / 0.58 |
| SNR | 0.85 / 0.97 / 0.99 / 0.89 | 0.88 / 0.98 / 0.98 / 0.85 |
| ReDo | 0.83 / 0.72 / 0.98 / 0.87 | 0.85 / 0.67 / 0.74 / 0.80 |
| L2 | 0.82 / 0.80 / 0.95 / 0.83 | 0.88 / 0.95 / 0.97 / 0.80 |
| L2 Init | 0.83 / 0.91 / 0.97 / 0.83 | 0.88 / 0.96 / 0.98 / 0.83 |
| Shrink and Perturb | 0.83 / 0.92 / 0.97 / 0.85 | 0.88 / 0.96 / 0.97 / 0.81 |
| Layer Norm (alone) | 0.69 / 0.14 / 0.96 / 0.82 | 0.66 / 0.11 / 0.96 / 0.58 |

- Source for the table; the CBP row was garbled in extraction and is omitted (the paper states SGD CBP trails SNR on every problem). — [Farias and Jozefiak, SNR](https://arxiv.org/html/2410.20098v3)

Continual backpropagation (CBP)
- Mechanism: continually reinitialize a small fraction of low-utility units; utility is a running average (decay 0.99) of a unit's activation times the magnitude of its outgoing weights; new units get zero outgoing weights and are protected by a maturity threshold. Settings: CIFAR-100 replacement rate 1e-5, maturity threshold 1,000; Permuted MNIST replacement rate 1e-6 (about one unit per layer every 500 examples in a 2,000-unit network); PPO replacement rate 1e-4, maturity threshold 1e4, with weight decay 1e-4. — [Dohare et al. 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11338828/)
- Results: on Continual ImageNet early tasks reached up to 88% test accuracy with backprop and plasticity was substantially lost by task 2,000 at every step size; CBP, L2 and S&P maintained plasticity; on class-incremental CIFAR-100 (ResNet-18) the base system ended about 5% below a from-scratch network, S&P reduced the gap and CBP eliminated it (76.13% on all 100 classes). — [Dohare et al. 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11338828/)
- Preprint abstract figure: accuracy fell from 89% on an early task to 77% (about linear-network level) on the 2000th task. — [Dohare et al. preprint](https://arxiv.org/abs/2306.13812)
- Same paper: Adam, dropout and normalization made LoP worse in their supervised tests (dropout at 0.03 was nearly identical to backprop; higher was worse); L2 helped but was hyperparameter-sensitive; in RL, L2 avoided degradation only at weight decay 1e-3, which stalled learning at suboptimal policies. — [Dohare et al. 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11338828/)
- Failure reports: CBP did not maintain plasticity with layer norm in a large network and declined in a 10-unit-per-layer network; in a ViT on class-incremental CIFAR-100, CBP and ReDo showed learning instabilities. — [Hernandez-Garcia, Dohare, Luo, Sutton 2025](https://arxiv.org/html/2508.00212v1)
- On ViT-B/16, CBP was roughly at baseline: average accuracy 57.43 / 66.18 / 67.09 vs vanilla 57.21 / 66.53 / 70.93 on CIFAR-100 with 10 / 20 / 25 tasks. — [Sun et al. 2026, ARROW](https://arxiv.org/html/2603.07787v1)

ReDo and gradient-based reset criteria
- Background: ReDo (Sokar et al., arXiv:2302.12902, 2023-02-24) periodically recycles dormant neurons identified by normalized activation. — [Sokar et al. 2023](https://arxiv.org/abs/2302.12902)
- ReDo addresses dormant units but can slow single-task convergence and does not fix initialization-time signal propagation in large networks. — [Lyle et al. 2024](https://arxiv.org/html/2402.18762)
- In on-policy RL, ReDo was the weakest regularizer and underperformed even warm-start under one shift condition. — [Juliani and Ash 2024](https://arxiv.org/html/2405.19153)
- GraMa / ReGraMa (Liu et al., arXiv:2505.24061, May 2025, NeurIPS 2025): select neurons for reset by gradient magnitude instead of activation, arguing learning capacity matters more than expressive activity; reported to improve several deep RL algorithms on MuJoCo and DMC (search-summary level). — [Liu et al. 2025](https://arxiv.org/abs/2505.24061)

Self-Normalized Resets (SNR)
- Mechanism: reset a neuron when the time since it last fired is improbable (below rejection percentile eta) under its own inter-firing-time distribution; one hyperparameter. — [Farias and Jozefiak](https://arxiv.org/html/2410.20098v3)
- Sensitivity: SNR accuracy changes by at most about 2 percentage points across its optimal eta range (exception: RM with SGD); for L2 Init, S&P and CBP, increasing hyperparameter strength by one order of magnitude from optimum reduces accuracy by at least 72 percentage points. — [Farias and Jozefiak](https://arxiv.org/html/2410.20098v3)
- Theory: for a single ReLU neuron with adversarial targets, L2-regularized gradient descent has average regret bounded below by a positive constant, while an idealized reset oracle achieves O(d^2 ln T / T). — [Farias and Jozefiak](https://arxiv.org/html/2410.20098v3)

Selective weight reinitialization (SWR)
- Hernandez-Garcia et al. (arXiv:2508.00212, 2025-07-31): periodically reinitialize the lowest-utility weights (not units); best variant uses gradient utility |w * g_w|, threshold pruning, and resampling from the init distribution. SWR stayed stable in small networks and with layer norm where CBP/ReDo did not; it matched or beat L2 in three of four Permuted MNIST settings. Magnitude utility was largely ineffective. — [Hernandez-Garcia et al. 2025](https://arxiv.org/html/2508.00212v1)

Shrink and perturb, full resets, FIRE, DASH
- Background: shrink and perturb originates in Ash and Adams' warm-starting work (arXiv ID recalled as 1910.08475; not re-verified this session).
- Lee et al. (arXiv:2406.02596, 2024-06-01, ICML 2024) warm-start test (10% subset, 50% label noise): S&P combined with L2 and augmentation raised test accuracy from 0.704 to 0.910 on CIFAR-10, 0.442 to 0.536 on CIFAR-100 (ViT-Tiny), and 0.115 to 0.376 on Tiny ImageNet (VGG-16). — [Lee et al. 2024](https://arxiv.org/html/2406.02596v1)
- In on-policy RL, final-layer reset failed in all three shift conditions, while S&P, soft S&P, L2, and regenerative regularization mitigated LoP in all three; soft S&P + LayerNorm was best overall. — [Juliani and Ash 2024](https://arxiv.org/html/2405.19153)
- FIRE (Han et al., arXiv:2602.08040, 2026-02-08, ICLR 2026 oral): reinitialize by projecting weights to the nearest isometry, W(W^T W)^(-1/2), approximated with Newton-Schulz; under 1% added training time; little or no post-reset accuracy drop where full reset and DASH drop sharply (no exact accuracies in text). — [Han et al. 2026, FIRE](https://arxiv.org/html/2602.08040v1); [ICLR 2026 listing](https://proceedings.iclr.cc/paper_files/paper/2026/hash/8a47999fdfecfd0ab2730487d87b0b47-Abstract-Conference.html)

L2, L2 Init (regenerative regularization), weight decay
- L2 Init (Kumar, Marklund, Van Roy; arXiv:2308.11958, v3 2024-10-24, CoLLAs 2024): penalize lambda * ||theta - theta_0||^2. It performs as well as or better than CBP on four of five problems and is top-three in total average online accuracy on every environment; plain L2 still shows large LoP on Random Label MNIST, Random Label CIFAR, and 5+1 CIFAR. Results are in figures only. — [Kumar et al.](https://arxiv.org/html/2308.11958v3)
- Han et al. (arXiv:2602.11137, February 2026, ICML 2026): in LLM pretraining, the weight decay best for downstream fine-tuning was 1.0 for Llama-2-0.5B/1B and OLMo-2-1B at 20 tokens-per-parameter, versus 0.5 to 0.6 best for pretraining loss; the common default is 0.1. — [Han et al. 2026](https://arxiv.org/html/2602.11137v2)

Layer norm + weight decay; Normalize-and-Project
- LN + L2 weight decay was the most robust configuration across the synthetic nonstationary classification tasks (CIFAR-10 based; MLP/CNN/ResNet-18; 20 tasks), with modest gains on 57 Atari games (C51, Rainbow); cost is slower convergence. — [Lyle et al. 2024](https://arxiv.org/html/2402.18762)
- LN alone is insufficient: see table above (0.69 / 0.14 on PM / RM with SGD); "layer norm alone still loses plasticity over time". — [Farias and Jozefiak](https://arxiv.org/html/2410.20098v3); [Hernandez-Garcia et al. 2025](https://arxiv.org/html/2508.00212v1)
- In on-policy RL, LayerNorm fixed training-set LoP but hurt generalization under the permute shift. — [Juliani and Ash 2024](https://arxiv.org/html/2405.19153)
- NaP: add normalization before nonlinearities and project weights to a fixed-norm sphere after each update; mitigates LoP on random-label continual classification and sequential ALE, but Rainbow under a constant effective learning rate under-performs unless a decay schedule is restored. — [Lyle et al. NaP](https://arxiv.org/abs/2407.01800)

Weight clipping
- Elsayed et al. (arXiv:2407.01704, 2024-07-01, RLJ vol. 5 2024): clip every weight to [-kappa * s_l, kappa * s_l] where s_l is the init bound (kappa tuned in 1 to 5 for streaming/Atari, up to 10 or 20 for warm-starting). On label-permuted EMNIST and mini-ImageNet only weight clipping kept improving, versus L2 Init, S&P and Madam; no exact numbers in the text; CBP was not compared. — [Elsayed et al. 2024](https://arxiv.org/html/2407.01704v1)

Spectral, Parseval and isometry regularizers
- Spectral regularization keeps each layer's top singular value near one (arXiv:2406.06811). — [Lewandowski et al.](https://arxiv.org/abs/2406.06811)
- Parseval regularization (Chung et al., arXiv:2412.07224, NeurIPS 2024) keeps weight matrices orthogonal; benefits on gridworld, CARL and MetaWorld continual RL. — [Chung et al. 2024](https://arxiv.org/abs/2412.07224v1)
- AdamO (arXiv:2606.09762): decoupled penalty on ||W^T W - I||_F^2 with lambda = 1e-3; overhead a few percent wall-clock and about 4 to 5% memory; claimed to outperform ReDo, L2 Init and spectral regularization on three supervised benchmarks (figures only). The paper reinterprets NaP, ReDo, L2 Init and spectral regularization as enforcing only partial isometry, and notes weight decay drives singular values toward zero rather than one. — [Rosseau et al. 2026](https://arxiv.org/html/2606.09762)
- L2-ER (arXiv:2509.22335): combines an effective-feature-rank term with L2, derived from a Kronecker-factored Hessian approximation; claimed to match or exceed CBP, S&P and ReDo (numbers not verified). — [Prakash et al. 2025](https://arxiv.org/html/2509.22335)

Plasticity injection
- Background: Nikishin et al. (arXiv:2305.15555, 2023-05-24, NeurIPS 2023) add a fresh head without changing predictions or trainable parameter count. — [Nikishin et al. 2023](https://arxiv.org/abs/2305.15555)
- 2024-2026 evidence is negative outside its original setting: it underperformed the warm-start baseline in all three on-policy shift conditions; and "performs poorly" in FIRE's RL comparison. — [Juliani and Ash 2024](https://arxiv.org/html/2405.19153); [Han et al. 2026, FIRE](https://arxiv.org/html/2602.08040v1)

CReLU and activation-function design
- Background: Abbas et al. (arXiv:2303.07507, 2023-03-13) proposed CReLU for continual Atari. — [Abbas et al. 2023](https://arxiv.org/abs/2303.07507)
- CReLU did not address LoP under the permute shift in on-policy RL; on ViT it scored 0.617 vs 0.638 for vanilla on the first 25 tasks; in L2 Init's study it lost plasticity on 5+1 CIFAR. — [Juliani and Ash 2024](https://arxiv.org/html/2405.19153); [Sun et al. 2026](https://arxiv.org/html/2603.07787v1); [Kumar et al.](https://arxiv.org/html/2308.11958v3)
- Lillo and Cheney (arXiv:2509.22562, 2025-09-26) total average online accuracy (%), Perm. MNIST / Rand. Label MNIST / Rand. Label CIFAR / CIFAR 5+1 / Cont. ImageNet: ReLU 78.85 / 20.03 / 25.79 / 4.76 / 73.71; CReLU 82.66 / 89.47 / 92.90 / 20.56 / 84.85; Leaky-ReLU 84.14 / 91.53 / 98.34 / 48.86 / 85.28; Randomized Smooth-Leaky 84.26 / 93.33 / 98.42 / 57.01 / 86.23. Best negative slopes were roughly 0.6 to 0.9; dead-band width correlated with non-recovery rate (r = 0.84). — [Lillo and Cheney 2025](https://arxiv.org/html/2509.22562v1)
- Deep Fourier features (arXiv:2410.20634, 2024-10-27): sine and cosine in every layer; motivated by a proof that linear function approximation does not suffer LoP. — [Lewandowski et al.](https://arxiv.org/abs/2410.20634)
- AdaLin (arXiv:2505.09486, May 2025, CoLLAs 2025): per-neuron learnable linearity injection gated by gradient flow (search-summary level). — [AdaLin](https://arxiv.org/abs/2505.09486v1)

Hare and Tortoise
- A fast "Hare" network is periodically reset to the weights of its EMA "Tortoise" (momentum 0.999, reset every 10 epochs in supervised; 0.995 and every 4,000 steps in RL). Atari-100k: DrQ IQM 0.243 to 0.287; BBF IQM 0.826 to 0.891. Weaker on Tiny ImageNet with VGG-16. — [Lee et al. 2024](https://arxiv.org/html/2406.02596v1)

Growth plus pruning
- Neuroplastic Expansion (Liu et al., arXiv:2410.07994, October 2024, ICLR 2025): start at 25% of capacity (MuJoCo), grow connections with the largest gradient magnitude every delta-T steps on a cosine-decayed schedule, prune fully dormant neurons (threshold 0) back into the candidate pool with a per-layer cap, and review early replay data when dormancy rises. Random growth alone beat static TD3; gradient-based growth beat random growth; adding pruning helped further. Outperformed Reset, ReDo, LN and Plasticity Injection on MuJoCo/DMC (figures only). — [Liu et al. 2024, NE](https://arxiv.org/html/2410.07994v1)

Other 2025-2026 methods (abstract or summary level)
- C-CHAIN reduces churn to prevent NTK rank collapse (ICML 2025). — [Tang et al. 2025](https://arxiv.org/abs/2506.00592)
- ARROW, a windowed gradient-covariance preconditioner applied to the last attention block of a ViT: 64.21 / 73.52 / 73.89 vs vanilla 57.21 / 66.53 / 70.93 on CIFAR-100 (10 / 20 / 25 tasks). — [Sun et al. 2026](https://arxiv.org/html/2603.07787v1)
- UPGD (arXiv:2404.00781, ICLR 2024): utility-scaled perturbation, small noise on useful units and larger on less useful ones. — [Elsayed and Mahmood 2024](https://arxiv.org/abs/2404.00781)
- Sample Weight Decay for replay-based RL (arXiv:2604.01913). — [Wu et al. 2026](https://arxiv.org/abs/2604.01913)
- NeuMoSync (arXiv:2608.04358, 2026-08-05, TMLR): learned neuromodulatory signals per neuron. — [Razavi Rohani et al. 2026](https://arxiv.org/abs/2608.04358)
- Simulating gradual change (output/input interpolation, task sampling) matched or exceeded L2, S&P, spectral regularization and ReDo without heavy tuning. — [Liu and Mou 2026](https://arxiv.org/html/2602.09234v2)
- SPHERE (ICML 2026) applies a Parseval penalty to mixture-of-experts policies; reported 133% and 50% average-success gains over an unregularized MoE baseline on MetaWorld and HumanoidBench (unverified, search summary only). — [SPHERE listing](https://icml.cc/virtual/2026/poster/62337)

### Inferences
- Ranking by independent replication rather than by originating paper: (1) norm control toward init or a fixed scale (L2 Init, WD with LN, clipping, isometry penalties) and S&P are the most reproducible; (2) unit resets work well in MLP/CNN ReLU settings but are fragile with LN, small widths, and transformers; (3) plasticity injection and CReLU have mostly negative results outside their original settings.
- CBP's design for newly created units (zero outgoing weights, maturity protection before they can be pruned) and NE's (gradient-based choice of what to grow, capped pruning, recycled candidates) are the two documented templates for "add and prune neurons on demand".
- For a tiny pre-norm transformer, the Hernandez-Garcia 2025 result (unit resets unstable with LN and with 10-unit layers) predicts that naive unit pruning/reinit of FFN neurons may destabilize training; weight-level reinit or reset combined with L2 are the safer comparators.

### Gaps
- Exact later-task accuracies for the Nature paper's Continual ImageNet comparison exist only in figures.
- The CBP row of the SNR comparison table could not be read.
- No source found that evaluates "periodic layer resets" as a standalone method in 2024-2026 beyond the negative final-layer-reset result and reset baselines in RL; the original primacy-bias reset paper is background and was not re-verified.
- No paper found that tests on-demand FFN neuron growth plus pruning in a transformer for plasticity; NE is RL with MLP/CNN actors and critics.

## 3. Does loss of plasticity occur in transformers and language models, and what is the small-scale evidence?

### Takeaway
Yes. As of mid-2026 there is direct evidence in GPT-style language models from 5M to 314M non-embedding parameters, in ViTs, and in one-block and single-layer transformers on synthetic sequences; scale delays the onset sublinearly but does not remove it, and the smallest models lose plasticity soonest. The transformer-specific findings are that the feed-forward module is where rank loss and dormancy concentrate, while attention weights also need regularization and MLP-unit resets alone do little.

### Cited Findings

Language models
- Hernandez-Garcia, Figliolia, Millidge (Zyphra; arXiv:2606.24752, 2026-06-23): pre-norm GPT-style decoders (GeLU, MLP expansion 4, AdamW with weight decay 0.1) trained on a repeating cycle of eight languages (5B tokens per task) and probed on held-out Vietnamese; plasticity measured as area under the probe validation-loss curve relative to cycle 1. All eight sizes eventually show LoP. — [Hernandez-Garcia et al. 2026](https://arxiv.org/html/2606.24752)
- Onset (task instance) by non-embedding size: 5M: 6; 12M: 6; 27M: 16; 39M: 18; 53M: 54; 83M: 54; 106M: 62; 314M: 118. Fitted law T = 1.3e-5 * P^0.8269 (R^2 = 0.8969, eight points, outliers at 12M and 53M); extrapolation to 1B parameters is about 360 task instances (about 1.8T tokens). — [Hernandez-Garcia et al. 2026](https://arxiv.org/html/2606.24752)
- Smallest configuration: 4 layers, hidden 320, 5 heads (4,932,480 non-embedding parameters); the 5M and 12M models show rising probe AUC immediately after the first cycle. — [Hernandez-Garcia et al. 2026](https://arxiv.org/html/2606.24752)
- Stationary training on a mixture of all eight languages also produces LoP (5M, 12M, 27M); the 27M model crosses the zero line after 475B tokens stationary versus 320B continual. — [Hernandez-Garcia et al. 2026](https://arxiv.org/html/2606.24752)
- Transformer-specific pathologies observed: dormant MLP units (threshold 0.01; more than 95% of layer 8 in the 53M model, nearly 80% of layer 10 in the 106M model), collapsed attention heads (entropy below 0.1 * H_max) and "lazy" heads (above 0.9 * H_max); none tracks LoP consistently. No mitigation was tested. — [Hernandez-Garcia et al. 2026](https://arxiv.org/html/2606.24752)
- FIRE: for a GPT-0.1B pretrained on WikiText-103 and then continued on OpenWebText + WikiText-103, the continued model's validation loss is worse the longer the initial pretraining (best-checkpoint, 30k and 60k iteration restarts), and the gap between the base model and a full reset narrows; FIRE beats S&P; full reset does not beat the base model (figures only, no numeric losses in text). — [Han et al. 2026, FIRE](https://arxiv.org/html/2602.08040v1)
- Han et al.: models with similar pretraining loss can differ in plasticity; higher weight decay lowers the pseudo-rank of attention matrices (at 1.0, W_QK rank roughly halves in OLMo-2-1B) and improves linear separability; correlational. — [Han et al. 2026](https://arxiv.org/html/2602.11137v2)

Small-scale transformer evidence
- Permuted Shakespeare, single-layer transformer decoder, 500 tasks of 32,768 tokens, average loss over last 50 tasks: no intervention 3.0147; SNR alone 3.0242; CBP 2.9410; ReDo 2.8119; L2 Init 1.5262; L2 0.3101; SNR+L2 0.2551; L2* 0.1092; SNR+L2* 0.0909 (the paper's definition of the starred L2 variant was not captured in extraction and is unverified). The authors conclude resets alone barely help and regularizing attention weights is needed. — [Farias and Jozefiak](https://arxiv.org/html/2410.20098v3)
- Scaled Permuted Shakespeare: at scale 16, L2 loss 0.377 vs SNR+L2 0.324; dead-neuron rate 32.9% for L2 and about 0 for SNR+L2. — [Farias and Jozefiak](https://arxiv.org/html/2410.20098v3)
- Single-block pretrained T5-small: on Bigram Cipher, unregularized test BLEU-2 drops to about 20% after 50 tasks; abrupt switches cause drops after about 12 tasks. — [Liu and Mou 2026](https://arxiv.org/html/2602.09234v2)
- 4-block, 4-head transformer on continual CIFAR-10 pixel permutation: vanilla, LN, and LN + weight decay degrade at learning rates where AdamO stays stable; described by the authors as "intentionally small-scale". — [Rosseau et al. 2026](https://arxiv.org/html/2606.09762)

Vision transformers
- ViT-B/16 on 200-task CIFAR-100: the paper's AAT metric is 0.590, a 14.9% drop from the reinitialization upper bound; the FFN is the "structural bottleneck" (lower effective rank, more aggressive weight growth, rising dormant units); deeper blocks contract faster; the value matrix is the least stable attention projection. — [Sun et al. 2026](https://arxiv.org/html/2603.07787v1)
- ViT on class-incremental CIFAR-100 (20 tasks): gap to from-scratch on the last task close to 10% with standard LN and about 4% with a reparameterized LN; standard LN's scale parameter shrank and hurt learning. — [Hernandez-Garcia et al. 2025](https://arxiv.org/html/2508.00212v1)
- CI-ViT training accuracy (final epoch, last 50 tasks): SNR+L2* 0.93372; L2 Init 0.91955; SNR+L2 0.87326; L2 0.81399; CBP 0.70365; SNR 0.65547; no intervention 0.64315; ReDo 0.63802; S&P 0.54589. — [Farias and Jozefiak](https://arxiv.org/html/2410.20098v3)
- Joudaki et al. observe rising dead and duplicate unit fractions in a ViT over 40 five-class Tiny ImageNet tasks (figures only). — [Joudaki et al. 2025](https://arxiv.org/html/2510.00304v2)

Contrary or qualifying evidence
- Wang, Chandra, Zhang (arXiv:2503.20018, 2025-03-25): a Transformer processing experience-replay data showed no LoP across regression, classification and policy evaluation; the authors conjecture in-context learning as the reason. — [Wang et al. 2025](https://arxiv.org/abs/2503.20018)
- Commentary (not a primary study) argues that normalization, weight decay, pretraining and scale have substantially reduced LoP in LLMs without eliminating it (unverified beyond search summary). — [Infinite Faculty essay](https://infinitefaculty.substack.com/p/what-are-the-real-problems-of-continual)

### Inferences
- A tiny transformer is the regime where LoP should appear fastest: the 4-layer, width-320 LLM degraded right after its first task cycle despite pre-norm and weight decay 0.1, and a single-layer decoder lost essentially all trainability on permuted character tasks without regularization.
- Because the FFN is where rank loss and dormancy concentrate in ViTs, FFN-neuron growth/pruning targets a plausible bottleneck; but the Permuted Shakespeare result implies an FFN-only intervention can leave attention-side LoP untouched, so an L2-on-attention control arm is needed to interpret results.
- Synthetic permuted or cipher sequence tasks give strong LoP in few tasks (tens), which suits a small experiment; natural-language task cycles show a smaller, noisier effect measured in percent of probe AUC.

### Gaps
- No mitigation has been tested at LLM scale in the continual multilingual setting; the authors call this "an open empirical question".
- No study found of dormant-unit or rank dynamics in attention-only or MLP-free small transformers, and none on decoder-only models below roughly 5M non-embedding parameters trained on natural language.
- The Wang et al. replay result uses a different setting (in-context processing of replayed data), so it does not contradict weight-space LoP in standard training; I could not verify its architecture details.

## 4. Which diagnostics are considered reliable?

### Takeaway
None of the cheap internal statistics (effective rank, dormant fraction, weight norm) is reliable on its own; each has documented cases where it fails to predict trainability. The only diagnostic treated as ground truth is a direct trainability probe: train on a new or held-out task and compare against a freshly initialized network. Gradient-based measures are the best-performing cheap proxies in 2025-2026 work.

### Cited Findings
- Ground-truth definition: plasticity loss is the gap between the final-task loss of a sequentially trained network and that of the same optimizer from fresh initialization. — [Klein et al. survey](https://arxiv.org/html/2411.04832)
- Probe design at LLM scale: area under the validation-loss curve on a held-out language during a fixed-budget probe, as percent change relative to the first cycle. — [Hernandez-Garcia et al. 2026](https://arxiv.org/html/2606.24752)
- Trainability and generalizability are separate: methods that restored training-loss convergence (Spectral Decoupling, regenerative regularization, ReDo, CReLU) left CIFAR-10 warm-start test accuracy at 0.705, 0.701, 0.682, 0.678 versus 0.704 baseline. — [Lee et al. 2024](https://arxiv.org/html/2406.02596v1)
- Wang et al. (arXiv:2605.09044, 2026-05-09) construct checkpoints with high representation rank and high empirical-NTK rank where gradient descent makes zero progress (two-layer ReLU, zero output layer), for both regression and classification. — [Wang et al. 2026](https://arxiv.org/html/2605.09044)
- Same paper, pairwise checkpoint-ranking accuracy at k=1 (Slowly-Changing Regression / Permuted MNIST): optimization readiness 0.987 / 0.906; representation effective rank 0.698 / 0.858; representation 99% energy rank 0.689 / 0.828; eNTK 99% energy rank 0.765 / omitted; Hessian 99% energy rank 0.734 / 0.821; active-neuron fraction 0.671 / 0.346. — [Wang et al. 2026](https://arxiv.org/html/2605.09044)
- Optimization readiness = gradient strength (||g||^2 / loss) times gradient reliability (||g||^2 / (||g||^2 + mini-batch gradient variance)); it lower-bounds one-step gain under smoothness. Its accuracy on Permuted MNIST degrades with few samples. — [Wang et al. 2026](https://arxiv.org/html/2605.09044)
- Lyle et al. cite prior negative results that no single quantity (parameter norm, dead-unit count) explains all LoP and propose the eNTK structure as a diagnostic. — [Lyle et al. 2024](https://arxiv.org/html/2402.18762)
- Klein et al. on metrics: effective rank, dormancy ratio, gradient dormancy, parameter norms, Hessian top eigenvalue and gradient covariance/eNTK are in common use; L2 gradient norms "can be misleading"; the rank-performance association is not reliable in general; the empirical Fisher is more accurate than Gauss-Newton for curvature. — [Klein et al. survey](https://arxiv.org/html/2411.04832)
- L2 Init study: CBP and CReLU reached surprisingly large weight magnitudes while retaining plasticity, so the authors drew no clear takeaway from weight magnitude. — [Kumar et al.](https://arxiv.org/html/2308.11958v3)
- Activation study: final accuracy correlated with dead-unit fraction at r = -0.59, more than with gradient norm (r = -0.28). — [Lillo and Cheney 2025](https://arxiv.org/html/2509.22562v1)
- GraMa argues activation-based dormancy misses neurons that are active but no longer learning, and proposes per-neuron gradient magnitude instead. — [Liu et al. 2025](https://arxiv.org/abs/2505.24061)
- Diagnostics added by recent theory: cloning R^2 score and duplicate-unit fraction; frozen probability P(|f'| < 0.01); Deviation from Isometry ||W^T W - I||_F^2; Hessian epsilon-rank at new-task initialization; attention-head entropy. — [Joudaki et al. 2025](https://arxiv.org/html/2510.00304v2); [Han et al. 2026, FIRE](https://arxiv.org/html/2602.08040v1); [Prakash et al. 2025](https://arxiv.org/html/2509.22335); [Hernandez-Garcia et al. 2026](https://arxiv.org/html/2606.24752)
- Plasticine (arXiv:2504.17490, April 2025, revised February 2026) packages 13+ methods and 10+ metrics for deep RL benchmarking (search-summary level). — [Plasticine](https://arxiv.org/abs/2504.17490)

### Inferences
- Recommended measurement stack for a small experiment, in order of trust: (1) fixed-budget probe on a held-out task versus a fresh-init control of the same current architecture; (2) gradient-based readiness or per-neuron gradient magnitude; (3) dormant fraction, effective rank, weight norm, and duplicate-unit fraction as descriptive covariates only.
- When width changes during training, the fresh-init control must be re-sized to match, otherwise capacity change is confounded with plasticity change.
- Report both training-loss speed and held-out performance on new tasks; a method can restore the first without the second.

### Gaps
- Optimization readiness has been validated only on a 2x5 MLP regression task and a 3x100 MLP on Permuted MNIST; no transformer validation was found.
- No agreed threshold for "dormant" exists across papers (0, 0.01, 0.1 and percentile rules all appear), so dormant fractions are not comparable between studies.
- I found no study that benchmarks Hessian-rank diagnostics in transformers.

## 5. What do the newest (2025-2026) theory papers say, and what are the negative results or critiques?

### Takeaway
New theory converges on spectral/geometric accounts (invariant manifolds; Hessian, NTK and Jacobian conditioning) and on the claim that generalization-friendly low-rank bias is in tension with plasticity. The main critiques are that popular diagnostics do not predict trainability, that standard benchmarks exaggerate LoP by using abrupt task switches, that simple regularizers often match specialized methods, and that several headline methods (CBP, ReDo, plasticity injection, CReLU) fail to transfer across architectures and settings.

### Cited Findings

Theory
- Invariant-manifold theory with frozen-unit and cloned-unit traps; normalization raises effective rank but does not break established clones; noisy SGD or dropout lets clones escape (a single step of gradient noise at 0.01 relative magnitude started escape); dropout's effect in continual bit-flipping was "mixed or even detrimental"; limits: theory covers linear/affine manifolds only. — [Joudaki et al. 2025](https://arxiv.org/html/2510.00304v2)
- Hessian spectral collapse with an epsilon-rank necessary condition for trainability. — [Prakash et al. 2025](https://arxiv.org/html/2509.22335)
- Plasticity as NTK isotropy in expectation; dynamical isometry as tractable surrogate; prior methods as partial isometry. — [Rosseau et al. 2026](https://arxiv.org/html/2606.09762)
- Trainability lower bound via optimization readiness; counterexamples to rank diagnostics. — [Wang et al. 2026](https://arxiv.org/html/2605.09044)
- Gradual change: under smoothness and local strong convexity, gradient descent tracks the old basin's minimizer into the new task when each micro-step is near-identity (Theorem 4.7); assumptions are strong for deep nets. — [Liu and Mou 2026](https://arxiv.org/html/2602.09234v2)
- NTK rank collapse plus Theta(1/k) gradient decay in online RL. — [Wu et al. 2026](https://arxiv.org/abs/2604.01913)
- Regularization-only schemes can provably fail where resets succeed (single ReLU neuron, adversarial targets). — [Farias and Jozefiak](https://arxiv.org/html/2410.20098v3)
- Linear function approximation and a special case of deep linear networks do not suffer LoP. — [Lewandowski et al. deep Fourier features](https://arxiv.org/abs/2410.20634)

Negative results and critiques
- Simple baselines match complex methods: the deep RL survey of 50+ mitigation strategies concludes "general regularization techniques often outperform domain-specific interventions". — [Klein et al. survey](https://arxiv.org/abs/2411.04832)
- LN + weight decay beat or matched gradient-norm penalties, InFeR and Shampoo in Lyle et al.'s tests. — [Lyle et al. 2024](https://arxiv.org/html/2402.18762)
- L2 Init matched or beat CBP on four of five problems with one hyperparameter. — [Kumar et al.](https://arxiv.org/html/2308.11958v3)
- Methods "fail, sometimes even performing worse than applying no intervention at all" under on-policy domain shift (plasticity injection, CReLU, final-layer reset). — [Juliani and Ash 2024](https://arxiv.org/abs/2405.19153); [full text](https://arxiv.org/html/2405.19153)
- Trainability-oriented methods give no generalization benefit in warm-starting. — [Lee et al. 2024](https://arxiv.org/html/2406.02596v1)
- Unit reinitialization is unreliable with layer norm and small networks. — [Hernandez-Garcia et al. 2025](https://arxiv.org/html/2508.00212v1)
- Reset methods are highly hyperparameter-sensitive (at least 72 percentage-point drops one order of magnitude from optimum for L2 Init, S&P, CBP). — [Farias and Jozefiak](https://arxiv.org/html/2410.20098v3)
- LoP severity is largely a function of benchmark abruptness. — [Liu and Mou 2026](https://arxiv.org/abs/2602.09234)
- Scale does not prevent LoP, and no tracked correlate explains it in LLMs. — [Hernandez-Garcia et al. 2026](https://arxiv.org/abs/2606.24752)
- Constant effective learning rate is not a free win: NaP-constrained Rainbow under-performs without a decay schedule. — [Lyle et al. NaP](https://arxiv.org/abs/2407.01800)
- Conflicting claims on normalization: Dohare et al. found normalization better early and worse later; Lyle et al. found LN + WD the most robust; LN alone fails on Permuted and Random Label MNIST in the SNR table. — [Dohare et al. 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11338828/); [Lyle et al. 2024](https://arxiv.org/html/2402.18762); [Farias and Jozefiak](https://arxiv.org/html/2410.20098v3)
- Conflicting claims on dropout: harmful in Dohare et al.; lets cloned units escape but is mixed or detrimental in continual bit-flipping in Joudaki et al. — [Dohare et al. 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11338828/); [Joudaki et al. 2025](https://arxiv.org/html/2510.00304v2)
- Survey-level evaluation critique: current evaluation practices have gaps, and understanding mechanisms should be prioritized over new mitigations. — [Klein et al. survey](https://arxiv.org/abs/2411.04832)

### Inferences
- The normalization disagreement resolves if LN is read as fixing preactivation shift while leaving norm growth (hence effective-LR decay) unaddressed; every positive LN result pairs it with weight decay, projection, or regularization toward init.
- A credible small experiment must include cheap baselines (L2 Init, LN + tuned weight decay, S&P) with tuned hyperparameters, because the literature's recurring failure mode is a new method beating only untuned or absent simple baselines.
- Given the abruptness critique, results should be reported for at least two transition schedules (abrupt and interpolated/mixed) to show that any benefit of growth/pruning is not specific to hard switches.

### Gaps
- No peer-reviewed independent replication was found of the 2026 theory claims (dynamical isometry, spectral collapse, optimization readiness) outside the originating groups.
- No meta-analysis exists comparing all methods under matched tuning budgets in supervised continual learning; the closest are the SNR table, the on-policy RL study, and the RL survey.
- Nothing dated September to November 2026 was found; newer work may exist that the searches did not surface.
