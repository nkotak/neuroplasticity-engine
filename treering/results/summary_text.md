# text stream: 2 seed(s), eras: python, manual-pages, word-list, c-headers, qa-json, exact repeat: python

| Method | Seeds | Parameters | Final loss (nats/token) | Loss forgotten |
|---|---|---|---|---|
| Tree-ring (mixture) | 2 | 1,034,368 | 1.015 ± 0.000 | 0.004 ± 0.000 |
| Tree-ring, newest self only | 2 | 1,034,368 | 5.595 ± 0.149 | 4.704 ± 0.168 |
| Tree-ring corrections + dreams (mixture) | 1 | 1,034,368 | 1.043 | 0.001 |
| Tree-ring corrections + dreams, newest self only | 1 | 1,034,368 | 2.808 | 1.811 |
| Plain fine-tuning | 1 | 988,288 | 2.185 | 1.508 |
| Fixed size + own dreams | 1 | 988,288 | 1.488 | 0.672 |
| Separate experts + mixture | 1 | 1,068,672 | 1.010 | 0.008 |
| EWC (best of two strengths) | 1 | 988,288 | 1.591 | 0.250 |
| Joint training (upper bound) | 1 | 988,288 | 0.935 | n/a |

## Final loss per era
| Method | python | manual-pages | word-list | c-headers | qa-json | exact repeat: python |
|---|---|---|---|---|---|---|
| Tree-ring (mixture) | 1.071 ± 0.002 | 1.504 ± 0.001 | 0.897 ± 0.008 | 1.383 ± 0.006 | 0.165 ± 0.001 | 1.071 ± 0.002 |
| Tree-ring, newest self only | 5.041 ± 0.056 | 6.715 ± 0.053 | 9.133 ± 0.607 | 7.497 ± 0.123 | 0.145 ± 0.001 | 5.041 ± 0.056 |
| Tree-ring corrections + dreams (mixture) | 1.065 | 1.538 | 0.957 | 1.461 | 0.171 | 1.065 |
| Tree-ring corrections + dreams, newest self only | 2.573 | 4.829 | 2.683 | 4.036 | 0.151 | 2.573 |
| Plain fine-tuning | 1.022 | 2.639 | 3.949 | 2.554 | 1.923 | 1.022 |
| Fixed size + own dreams | 1.027 | 2.187 | 1.330 | 2.001 | 1.358 | 1.027 |
| Separate experts + mixture | 1.069 | 1.505 | 0.868 | 1.387 | 0.165 | 1.069 |
| EWC (best of two strengths) | 1.063 | 2.060 | 2.037 | 1.968 | 1.355 | 1.063 |
| Joint training (upper bound) | 1.053 | 1.347 | 0.843 | 1.152 | 0.162 | 1.053 |

## Growth of the tree-ring model
| Era | Loss before | Loss after | Rings tried | Rings kept | Parameters added |
|---|---|---|---|---|---|
| python | n/a | 1.056 ± 0.002 | 1.0 ± 0.0 | 1.0 ± 0.0 | 371,840 |
| manual-pages | 2.766 ± 0.007 | 1.512 ± 0.001 | 1.0 ± 0.0 | 1.0 ± 0.0 | 162,176 |
| word-list | 3.790 ± 0.009 | 0.892 ± 0.008 | 1.0 ± 0.0 | 1.0 ± 0.0 | 164,480 |
| c-headers | 2.407 ± 0.000 | 1.375 ± 0.006 | 1.0 ± 0.0 | 1.0 ± 0.0 | 166,784 |
| qa-json | 2.259 ± 0.001 | 0.165 ± 0.001 | 1.0 ± 0.0 | 1.0 ± 0.0 | 169,088 |
| exact repeat: python | 1.071 ± 0.002 | 1.071 ± 0.002 | 1.0 ± 0.0 | 0.0 ± 0.0 | 0 |

## Guarantee checks (tree-ring, all seeds)
- Tree-ring: largest change in any past self's output after later growth = 0.0; largest change after rollback = 0.0; worst per-sequence gap to the best self = 2.585 nats (bound log(3K) = 2.708); rings at end: [5, 5]
- Tree-ring corrections + dreams: largest change in any past self's output after later growth = 0.0; largest change after rollback = 0.0; worst per-sequence gap to the best self = 2.584 nats (bound log(3K) = 2.708); rings at end: [5]

Seconds per seed: 2551 ± 1237
