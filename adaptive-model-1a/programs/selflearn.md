# Self-Learning

Self-learning is cycle-based and verified.

Allowed v1 task families:
- deterministic code tasks
- deterministic math tasks
- grounded QA with explicit support context
- benchmark-repair tasks derived from failure buckets

Required cycle:
1. benchmark current best
2. bucket failures
3. generate candidate tasks
4. solve with student and optional teacher
5. verify
6. write accepted JSONL
7. fine-tune
8. benchmark candidate
9. promote or reject

Never train on:
- raw unverified generations
- held-out benchmark gold items
- open-ended prose/fantasy/research-writing in v1
