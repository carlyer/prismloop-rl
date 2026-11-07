# Research roadmap

[← Overview](../README.md)

Status here describes archived evidence synchronized on **2026-11-07**, not a live view of server processes. The latest readable long-training state was observed on 2026-10-27.

| Stage | Deliverable | Snapshot status |
|---|---|---|
| Reproduction | Image/video RL execution, official model evaluations, weight-update audits | Implemented; selected results published |
| Core extensions | Final numeric parsing, source quotas, bounded tool rewards, worker cleanup | Implemented and short-run tested |
| Optimization setting | PPO epochs=1 versus epochs=2 | Eight core/PPO2 benchmark evaluations published, including HRBench-8K |
| Attribution | Same-start original controls and component factorial experiments | Original control evaluations pending; remaining short component cells deferred for long runs |
| Long continuations | Four core/PPO2 branches toward total step 700 | Archived progress at steps 311/401; no completed milestone evaluation scores in the available snapshot |
| Robustness | Multiple seeds and matched compute comparisons | Further work; longer training alone does not establish robustness |
| Evaluation validity | Resolve anomalous video results | Further work |
| Spatial consistency | Train against verified cross-question spatial relations | Proposed; not implemented |
| Tool-repair credit | Pair failed prefixes with useful recovery continuations | Proposed; not implemented |
| Adaptive rollout allocation | Balance task utility, trajectory cost, and sampling budget | Proposed; not implemented |

For each proposed mechanism, first define the failure it addresses, the training signal, a minimal baseline, and a falsifiable experiment. A project name, a reward heuristic, or an extra hyperparameter sweep is not evidence of algorithmic novelty.

[Measured scores](../results/benchmark-results.csv), [component status](../results/controlled-ablation-status.csv) and [continuation evaluation status](../results/continuation-evaluation-status.csv) are separate tables. Empty scores are pending, never zero. Historical 700-step planning forecasts remain separate from all measurements.
