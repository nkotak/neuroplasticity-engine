# Code Research

The code-research loop is an outer-loop experiment system modeled on `autoresearch`.

Rules:
- operate in a dedicated sandbox, not the main working tree
- edit only allowlisted files from `program.md`
- run a baseline if no baseline exists for the sandbox
- benchmark every experiment
- keep or discard by promotion gate only

Default research profile:
- total steps: `12000`
- warmup steps: `2000`
- eval every: `250`
- checkpoint every: `2000`
- wall-clock cap: `20 minutes`

Status values for `results.tsv`:
- `keep`
- `discard`
- `crash`
