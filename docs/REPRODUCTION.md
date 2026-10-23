# Reproduction guide

This is a source publication of the current research environment, **not yet a turnkey portable release**. The historical orchestration scripts contain host-specific paths, GPU assignments, ports, and recovery behavior. Review them before use; they must not be launched unchanged on a shared server.

1. Use Linux, NVIDIA CUDA GPUs and Python 3.10. The validated environment used PyTorch 2.7, Transformers 4.54, vLLM 0.9.1 and Ray 2.47. Follow [the upstream setup](UPSTREAM_README.md) and the pinned runtime requirements. Install the local package with `python -m pip install --no-deps -e ./verl_agents` after dependencies are installed.
2. Obtain official model weights and datasets from the upstream sources under their respective terms. Prepare `reproduction/models/` and `reproduction/data/prepared/` using the preparation scripts. No model or benchmark data is bundled.
3. The evaluation framework is an external dependency: obtain [PyVision-RL-Eval](https://github.com/agents-x-project/PyVision-RL-Eval), revision `9a756943702fb4b4b2cc7548f81a1e162a99d569`, in a sibling directory. Its source and datasets are not redistributed here; the checked revision has no LICENSE file.
4. Copy `reproduction/judge-config.example.json` to `reproduction/judge-config.json`, verify the required schema against the judge client, and configure your own judge service. Keep credentials outside Git.
5. Adapt interpreter paths and storage paths in `reproduction/scripts/run_training.py` and other launchers to your machine. The publication copy replaces private workspace paths with `/path/to/workspace` placeholders. Review `--help`, prepare assets, then run a smoke experiment before full training.

Feature selection is explicit via `PYVISION_ABLATION_FEATURES`: an empty value selects the original algorithm with shared runtime fixes; comma-separated `numeric,quota,tool` enables the full core. Quota selection additionally requires the source-weight configuration supplied by the launcher. `run_training.py` exposes the corresponding variant choices. Inspect the resolved configuration before interpreting an ablation.

The existing factorial controller was written for an eight-GPU shared server and fixed initial checkpoints. Its scratch-retention code removes only newly generated optimizer states after export and update validation; these compacted checkpoints cannot resume with the original optimizer. Do not use that controller without adapting its paths and retention policy.


For the architecture and measured results, return to the [project overview](../README.md).
