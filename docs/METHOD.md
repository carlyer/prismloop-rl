# Method and implementation

[← Overview](../README.md)

PrismLoop-RL retains the upstream GRPO training algorithm. Its implemented extensions address three places where the training signal can be distorted: numeric answer extraction, mixed-source rollout selection, and tool execution bonuses.

## Agent interaction

A question and its image or video enter a multi-turn rollout. The policy emits reasoning and optionally Python code. The executor returns text, images, or errors. The policy continues from that feedback and eventually emits a final answer. Several sibling trajectories are sampled for each question.

The experimental configuration uses eight trajectories per question and selects 128 trajectories per global step. Group and trajectory filters can remove candidates and trigger replenishment. Selected trajectories are therefore not necessarily 16 intact groups of eight.

## Final numeric answer extraction

[research_answer.py](../verl_agents/verl/utils/reward_score/research_answer.py) extracts an explicit final answer instead of reading the first number in a complete reasoning trace. [rule_match_verify.py](../verl_agents/verl/utils/reward_score/rule_match_verify.py) applies the existing MRA scoring function to that value. Malformed, ambiguous, and nonfinite answers receive no numeric reward; unit conversion is not inferred.

The `numeric` feature affects video numeric-reward samples. The image training sources in this experiment use LLM judging, so numeric extraction is not an image ablation factor.

## Source-aware group selection

[research_selection.py](../verl_agents/verl/trainer/ppo/research_selection.py) allocates best-effort quotas from source proportions. Eligible question groups are ranked by reward variance within their own source, rather than comparing raw variances across unlike sources. Surviving sibling sets are retained whole where possible. At the final capacity boundary, a group can be partially selected. Source shortages and partial groups are logged.

This changes selection after filtering. It does not introduce a new advantage estimator or renormalize advantages after selection.

## Bounded tool bonuses

[research_tool_quality.py](../verl_agents/verl/workers/agent/envs/agents_x/research_tool_quality.py) requires successful execution and nonempty feedback, hashes normalized Python ASTs and outputs, rejects duplicates, and caps credits at two executions per episode. The current configuration uses 0.1 per credited call, for a maximum bonus of 0.2. The existing answer-correctness gate remains in effect; video numeric samples use the original threshold-based gate, not an exact-correctness requirement.

These checks reject obvious low-value execution patterns. They do not measure semantic usefulness or assign causal credit to a tool call.

## Algorithm switches

| Launcher variant | Numeric | Quota | Tool | PPO epochs |
|---|:---:|:---:|:---:|:---:|
| `original` | — | — | — | 1 |
| `numeric_only` | ✓ | — | — | 1 |
| `quota_only` | — | ✓ | — | 1 |
| `tool_only` | — | — | ✓ | 1 |
| `without_numeric` | — | ✓ | ✓ | 1 |
| `without_quota` | ✓ | — | ✓ | 1 |
| `without_tool` | ✓ | ✓ | — | 1 |
| `full` | ✓ | ✓ | ✓ | 1 |
| `ppo2` | ✓ | ✓ | ✓ | 2 |

The launcher forwards `PYVISION_ABLATION_FEATURES` through Ray to workers. Direct module use defaults to all three features when the environment variable is absent. An explicit empty value disables them. Quota selection also requires source weights in the resolved configuration.

All new controlled cells share the same engineering fixes: spawn execution, bounded cleanup, 16 tool workers, and invalid-numeric guards. Thus `original` means the original algorithm under common runtime fixes, not a byte-identical recreation of the paper environment.

## Optimization and accounting

The base configuration uses GRPO without reward-standard-deviation normalization, learning rate 1e-6, and no KL loss. With the current selected batch and minibatch configuration, epochs=1 makes one optimizer update per global step; epochs=2 makes two. FSDP shards model state across training GPUs.

A 40-step comparison matches rollout steps. It does not match optimizer updates, generated tokens, or GPU time. Those budgets must be reported separately before making efficiency claims.

## Verification

The scripts audit actual weight changes and optimizer counters, validate merged BF16 weights, and compare evaluation protocols and record coverage. These checks validate execution evidence. They do not establish scientific novelty, judge correctness, or significance by themselves.
