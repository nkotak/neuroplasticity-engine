# skills stream: 3 seed(s), eras: copy+reverse, sort, shift+double, addition, compositions, mixed revisit: reverse+sort, exact repeat: copy+reverse

| Method | Seeds | Parameters | Final loss (nats/token) | Loss forgotten | Final accuracy | Accuracy forgotten |
|---|---|---|---|---|---|---|
| Tree-ring (mixture) | 3 | 784,656 | 1.014 ± 0.001 | 0.009 ± 0.000 | 0.976 ± 0.004 | 0.000 ± 0.000 |
| Tree-ring, newest self only | 3 | 784,656 | 4.288 ± 0.002 | 3.721 ± 0.011 | 0.523 ± 0.024 | 0.504 ± 0.011 |
| Tree-ring + dreams (mixture) | 1 | 819,456 | 1.017 | 0.005 | 0.954 | 0.000 |
| Tree-ring + dreams, newest self only | 1 | 819,456 | 1.040 | 0.047 | 0.892 | 0.072 |
| Tree-ring corrections + dreams (mixture) | 2 | 850,776 | 1.018 ± 0.003 | 0.005 ± 0.000 | 0.950 ± 0.015 | -0.000 ± 0.000 |
| Tree-ring corrections + dreams, newest self only | 2 | 850,776 | 1.036 ± 0.003 | 0.039 ± 0.000 | 0.917 ± 0.017 | 0.038 ± 0.002 |
| Tree-ring corrections + spliced dreams (mixture) | 1 | 902,976 | 1.036 | 0.007 | 0.919 | 0.002 |
| Tree-ring corrections + spliced dreams, newest self only | 1 | 902,976 | 1.124 | 0.116 | 0.740 | 0.202 |
| Plain fine-tuning | 2 | 801,920 | 5.198 ± 0.058 | 4.922 ± 0.068 | 0.374 ± 0.000 | 0.730 ± 0.000 |
| Fixed size + own dreams | 2 | 801,920 | 1.027 ± 0.001 | 0.050 ± 0.001 | 0.929 ± 0.003 | 0.083 ± 0.003 |
| Fixed size + own spliced dreams | 1 | 801,920 | 1.083 | 0.108 | 0.800 | 0.232 |
| Separate experts + mixture | 1 | 820,992 | 1.010 | 0.009 | 0.991 | -0.001 |
| EWC (best of two strengths) | 2 | 801,920 | 3.556 ± 0.068 | 3.007 ± 0.079 | 0.423 ± 0.022 | 0.672 ± 0.025 |
| Joint training (upper bound) | 1 | 801,920 | 0.997 | n/a | 1.000 | n/a |
| Compacted by dreaming (no data) | 1 | 429,440 | 1.124 | n/a | 0.640 | n/a |

## Final accuracy per era
| Method | copy+reverse | sort | shift+double | addition | compositions | mixed revisit: reverse+sort | exact repeat: copy+reverse |
|---|---|---|---|---|---|---|---|
| Tree-ring (mixture) | 0.996 ± 0.001 | 0.980 ± 0.004 | 0.990 ± 0.006 | 0.947 ± 0.015 | 0.948 ± 0.008 | 0.976 ± 0.003 | 0.996 ± 0.001 |
| Tree-ring, newest self only | 0.854 ± 0.082 | 0.976 ± 0.002 | 0.001 ± 0.000 | 0.000 ± 0.000 | 0.004 ± 0.005 | 0.972 ± 0.003 | 0.854 ± 0.082 |
| Tree-ring + dreams (mixture) | 0.997 | 0.969 | 0.991 | 0.831 | 0.947 | 0.948 | 0.997 |
| Tree-ring + dreams, newest self only | 0.994 | 0.956 | 0.838 | 0.817 | 0.699 | 0.947 | 0.994 |
| Tree-ring corrections + dreams (mixture) | 0.996 ± 0.001 | 0.970 ± 0.014 | 0.991 ± 0.001 | 0.827 ± 0.078 | 0.933 ± 0.001 | 0.937 ± 0.009 | 0.996 ± 0.001 |
| Tree-ring corrections + dreams, newest self only | 0.994 ± 0.001 | 0.968 ± 0.011 | 0.929 ± 0.002 | 0.818 ± 0.081 | 0.779 ± 0.016 | 0.936 ± 0.011 | 0.994 ± 0.001 |
| Tree-ring corrections + spliced dreams (mixture) | 0.990 | 0.915 | 0.945 | 0.806 | 0.879 | 0.912 | 0.990 |
| Tree-ring corrections + spliced dreams, newest self only | 0.930 | 0.872 | 0.528 | 0.783 | 0.234 | 0.901 | 0.930 |
| Plain fine-tuning | 1.000 ± 0.000 | 0.057 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.027 ± 0.000 | 0.534 ± 0.000 | 1.000 ± 0.000 |
| Fixed size + own dreams | 1.000 ± 0.000 | 0.973 ± 0.005 | 0.851 ± 0.009 | 0.799 ± 0.031 | 0.891 ± 0.007 | 0.987 ± 0.004 | 1.000 ± 0.000 |
| Fixed size + own spliced dreams | 0.999 | 0.911 | 0.455 | 0.614 | 0.668 | 0.951 | 0.999 |
| Separate experts + mixture | 1.000 | 0.998 | 1.000 | 0.967 | 0.999 | 0.971 | 1.000 |
| EWC (best of two strengths) | 0.998 ± 0.000 | 0.291 ± 0.103 | 0.014 ± 0.004 | 0.000 ± 0.000 | 0.025 ± 0.001 | 0.634 ± 0.043 | 0.998 ± 0.000 |
| Joint training (upper bound) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Compacted by dreaming (no data) | 0.714 | 0.806 | 0.727 | 0.266 | 0.571 | 0.682 | 0.714 |

## Growth of the tree-ring model
| Era | Loss before | Loss after | Rings tried | Rings kept | Parameters added |
|---|---|---|---|---|---|
| copy+reverse | n/a | 1.068 ± 0.000 | 1.0 ± 0.0 | 1.0 ± 0.0 | 202,016 |
| sort | 5.759 ± 0.314 | 1.041 ± 0.001 | 1.0 ± 0.0 | 1.0 ± 0.0 | 98,000 |
| shift+double | 6.177 ± 0.152 | 0.911 ± 0.001 | 1.0 ± 0.0 | 1.0 ± 0.0 | 107,264 |
| addition | 10.923 ± 0.205 | 0.722 ± 0.003 | 1.0 ± 0.0 | 1.0 ± 0.0 | 144,368 |
| compositions | 4.251 ± 0.025 | 1.104 ± 0.001 | 1.0 ± 0.0 | 1.0 ± 0.0 | 125,792 |
| mixed revisit: reverse+sort | 1.216 ± 0.000 | 1.100 ± 0.000 | 1.0 ± 0.0 | 1.0 ± 0.0 | 107,216 |
| exact repeat: copy+reverse | 1.096 ± 0.000 | 1.096 ± 0.000 | 1.0 ± 0.0 | 0.0 ± 0.0 | 0 |

## Guarantee checks (tree-ring, all seeds)
- Tree-ring: largest change in any past self's output after later growth = 0.0; largest change after rollback = 0.0; worst per-sequence gap to the best self = 2.703 nats (bound log(3K) = 2.890); rings at end: [6, 6, 6]
- Tree-ring + dreams: largest change in any past self's output after later growth = 0.0; largest change after rollback = 0.0; worst per-sequence gap to the best self = 2.697 nats (bound log(3K) = 2.890); rings at end: [6]
- Tree-ring corrections + dreams: largest change in any past self's output after later growth = 0.0; largest change after rollback = 0.0; worst per-sequence gap to the best self = 2.693 nats (bound log(3K) = 2.890); rings at end: [6, 6]
- Tree-ring corrections + spliced dreams: largest change in any past self's output after later growth = 0.0; largest change after rollback = 0.0; worst per-sequence gap to the best self = 2.699 nats (bound log(3K) = 2.890); rings at end: [6]

## Forward transfer: one new ring on the frozen model vs a stand-alone model of the same size (era: compositions)
| Training steps | Ring loss | Stand-alone loss | Ring accuracy | Stand-alone accuracy |
|---|---|---|---|---|
| 200 | 1.346 | 1.772 | 0.289 | 0.000 |
| 500 | 1.133 | 1.241 | 0.716 | 0.399 |
| 1200 | 1.078 | 1.069 | 0.939 | 0.994 |

## Sequences that mix skills from every era (never seen in training)
| Method | Seeds | Accuracy | Loss |
|---|---|---|---|
| Joint training (upper bound) | 1 | 0.990 | 1.631 |
| Tree-ring (switching mixture) | 3 | 0.825 ± 0.003 | 1.344 ± 0.027 |
| Separate experts + mixture | 1 | 0.797 | 1.423 |
| Fixed size + own dreams | 2 | 0.755 ± 0.036 | 2.126 ± 0.194 |
| Fixed size + own spliced dreams | 1 | 0.623 | 2.389 |
| Tree-ring corrections + spliced dreams (switching mixture) | 1 | 0.608 | 1.777 |
| Tree-ring + dreams (switching mixture) | 1 | 0.521 | 2.245 |
| Tree-ring corrections + dreams (switching mixture) | 2 | 0.498 ± 0.004 | 2.437 ± 0.063 |
| Tree-ring corrections + spliced dreams, newest self only | 1 | 0.457 | 2.116 |
| Compacted by dreaming (no data) | 1 | 0.448 | 1.798 |
| Tree-ring corrections + dreams, newest self only | 2 | 0.447 ± 0.011 | 2.770 ± 0.126 |
| Tree-ring (plain Bayes mixture) | 3 | 0.443 ± 0.012 | 4.223 ± 0.078 |
| Tree-ring + dreams, newest self only | 1 | 0.422 | 2.500 |
| Tree-ring, newest self only | 3 | 0.271 ± 0.032 | 5.012 ± 0.028 |
| EWC (strength 100000) | 2 | 0.257 ± 0.011 | 4.194 ± 0.066 |
| Plain fine-tuning | 2 | 0.230 ± 0.001 | 5.747 ± 0.144 |
| EWC (strength 1000) | 1 | 0.224 | 5.723 |

Seconds per seed: 3865 ± 2234
