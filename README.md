# Multimodal Agentic RL

A personal reproduction and extension of **PyVision-RL** for image and video agents that reason, execute Python tools, inspect observations, and produce answers. Built on the upstream PyVision-RL / verl training stack; this repository does not claim authorship of that stack.

## What this project adds

- Strict extraction of final numeric answers for video reward calculation, with malformed-answer guards.
- Source-aware selection of rollout groups, with within-source reward-variance ranking and quota diagnostics.
- Bounded tool rewards for successful, nonempty, nonduplicate executions. This is a quality heuristic, not a proof that a tool call helped solve the task.
- Spawn-based tool workers and bounded cleanup to reduce stalled execution.
- Explicit feature switches for original-algorithm controls, individual components, and factorial ablations.
- Checkpoint weight/optimizer-update audits, model-export validation, full-record evaluation audits, and protocol comparisons.
- Separate PPO epochs=1 / epochs=2 experiments and image/video evaluation queues.

## Measured results

Percentages under this project's server evaluation protocol, including a local 32B judge. These are not copied paper results. VStar uses avg@32; VSI uses the released eight-task macro average.

| Model | VStar | HRBench-4K | VSI macro |
|---|---:|---:|---:|
| Official SFT | 68.80 | 63.38 | 35.71 |
| Official released RL | 83.97 | 72.38 | 43.18 |
| Server reproduction starting checkpoint | 70.14 | 66.00 | 35.98 |
| Core improvements, +40 steps | 71.53 | 66.50 | 36.98 |
| Core + PPO epochs=2, +40 steps | 71.65 | 68.75 | 37.52 |

Image experiments start from step 20 and end at 60; video experiments start from step 120 and end at 160. Each modality starts from its own fixed checkpoint. Gains relative to the starting checkpoint include the effect of additional training. Same-budget original-algorithm controls and component ablations were pending when this publication snapshot was prepared; causal gains have not yet been established. PPO epochs=2 performs twice as many optimizer updates per rollout batch. No 700-step performance forecast is reported as a measured result.

The complete score snapshot is in [results/benchmark-results.csv](results/benchmark-results.csv). Some MVBench and TempCompass results are anomalous and need semantic/scoring-format investigation; passing record-coverage audits does not validate those scores as capability measurements. Single-seed, short-training results do not establish statistical significance.

## Repository layout

- `verl_agents/`: complete upstream training framework with the current extensions.
- `configs/`: upstream training configurations.
- `reproduction/scripts/`: preparation, training, evaluation, auditing, and experiment orchestration scripts.
- `reproduction/requirements-runtime.txt`: runtime dependency pins.
- `docs/UPSTREAM_README.md`: original PyVision-RL documentation and attribution.
- `docs/source-provenance.json`: upstream revisions and file hashes for the source snapshot.
- `results/`: aggregate scores, without raw benchmark records or private paths.

## Setup and reproduction

This is a source publication of the current research environment, **not yet a turnkey portable release**. The historical orchestration scripts contain host-specific paths, GPU assignments, ports, and recovery behavior. Review them before use; they must not be launched unchanged on a shared server.

1. Use Linux, NVIDIA CUDA GPUs and Python 3.10. The validated environment used PyTorch 2.7, Transformers 4.54, vLLM 0.9.1 and Ray 2.47. Follow [the upstream setup](docs/UPSTREAM_README.md) and the pinned runtime requirements. Install the local package with `python -m pip install --no-deps -e ./verl_agents` after dependencies are installed.
2. Obtain official model weights and datasets from the upstream sources under their respective terms. Prepare `reproduction/models/` and `reproduction/data/prepared/` using the preparation scripts. No model or benchmark data is bundled.
3. The evaluation framework is an external dependency: obtain [PyVision-RL-Eval](https://github.com/agents-x-project/PyVision-RL-Eval), revision `9a756943702fb4b4b2cc7548f81a1e162a99d569`, in a sibling directory. Its source and datasets are not redistributed here; the checked revision has no LICENSE file.
4. Copy `reproduction/judge-config.example.json` to `reproduction/judge-config.json`, verify the required schema against the judge client, and configure your own judge service. Keep credentials outside Git.
5. Adapt interpreter paths and storage paths in `reproduction/scripts/run_training.py` and other launchers to your machine. The publication copy replaces private workspace paths with `/path/to/workspace` placeholders. Review `--help`, prepare assets, then run a smoke experiment before full training.

Feature selection is explicit via `PYVISION_ABLATION_FEATURES`: an empty value selects the original algorithm with shared runtime fixes; comma-separated `numeric,quota,tool` enables the full core. Quota selection additionally requires the source-weight configuration supplied by the launcher. `run_training.py` exposes the corresponding variant choices. Inspect the resolved configuration before interpreting an ablation.

The existing factorial controller was written for an eight-GPU shared server and fixed initial checkpoints. Its scratch-retention code removes only newly generated optimizer states after export and update validation; these compacted checkpoints cannot resume with the original optimizer. Do not use that controller without adapting its paths and retention policy.

## Attribution and scope

This repository builds on [PyVision-RL](https://github.com/agents-x-project/PyVision-RL) and its bundled verl/RAGEN-derived components. The original MIT license and source copyright notices are retained. See [THIRD_PARTY_NOTICES.txt](THIRD_PARTY_NOTICES.txt). Please credit and cite the upstream work when using this project.

Proposed spatial-consistency, tool-repair-credit, and adaptive-rollout research directions are **not implemented** in this snapshot. Current contributions are reproducibility engineering and experimentally tested extensions; no novel-algorithm or state-of-the-art claim is made.
