# Research roadmap

[← Overview](../README.md)

Status here describes the published source snapshot, not a live view of server processes.

| Stage | Deliverable | Snapshot status |
|---|---|---|
| Reproduction | Image/video RL execution, official model evaluations, weight-update audits | Implemented; selected results published |
| Core extensions | Final numeric parsing, source quotas, bounded tool rewards, worker cleanup | Implemented and short-run tested |
| Optimization setting | PPO epochs=1 versus epochs=2 | Six core/PPO2 benchmark evaluations published |
| Attribution | Same-start original controls and component factorial experiments | Scripts implemented; results pending in this snapshot |
| Robustness | Multiple seeds, longer learning curves, matched compute comparisons | Further work |
| Evaluation validity | Resolve anomalous video results | Further work |
| Spatial consistency | Train against verified cross-question spatial relations | Proposed; not implemented |
| Tool-repair credit | Pair failed prefixes with useful recovery continuations | Proposed; not implemented |
| Adaptive rollout allocation | Balance task utility, trajectory cost, and sampling budget | Proposed; not implemented |

For each proposed mechanism, first define the failure it addresses, the training signal, a minimal baseline, and a falsifiable experiment. A project name, a reward heuristic, or an extra hyperparameter sweep is not evidence of algorithmic novelty.
