# Backends

`mlx`
- primary backend
- supports training, inference, and benchmarking
- only promoted training backend in v1

`mps`
- secondary backend
- inference and benchmarking in v1
- training only if an explicit mirrored trainer exists

`ane`
- inference and judge backend only in v1
- never use for training

If a requested backend capability is unavailable, fail loudly instead of falling back silently.
