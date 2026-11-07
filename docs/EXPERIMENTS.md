# Experiments and evidence

[← Overview](../README.md)

## Published snapshot

The aggregate table in [benchmark-results.csv](../results/benchmark-results.csv) contains **40 measured result rows**, synchronized from available archives on **2026-11-07**. The previous 31 rows and their anomaly notes are retained. This update adds two image core/PPO2 HRBench-8K evaluations and seven official Image SFT evaluations. There are now eight extension evaluations: image core/PPO2 on VStar, HRBench-4K and HRBench-8K, and video core/PPO2 on VSI.

This is an evidence synchronization, not a new training or inference run. For the nine new results, unique question/repeat coverage, invocation fingerprints, finite scores, failure counts and aggregate/category means were rechecked against archived per-question records. Original input membership relies on the archived passed full-input audits; the prepared inputs were not re-hashed during this sync. The old 31 rows were retained from the published aggregate, rather than described as newly re-audited. File hashes and scope are in [snapshot-provenance.json](../results/snapshot-provenance.json).

| Protocol | Definition |
|---|---|
| VStar | 6,112 evaluated trajectories; avg@32 |
| HRBench-4K | 800 examples |
| HRBench-8K | 800 examples; no measured starting checkpoint in this snapshot |
| VSI | 5,130 examples; released eight-task macro average in the overview |
| Judge | Local Qwen2.5-VL-32B-Instruct |
| Image initial state | Server original-method checkpoint, step 20 |
| Video initial state | Server original-method checkpoint, step 120 |
| Extension budget | 40 additional rollout global steps per run |
| Seeds | One training seed; no significance claim |

Image and video are separate model branches. Official weights are evaluated with the server's agent scaffold, so these numbers must not be presented as the paper's results or as a direct-answer baseline.

## Observed changes, in percentage points

| Comparison to starting checkpoint | VStar | HRBench-4K | VSI macro |
|---|---:|---:|---:|
| Core, +40 steps | +1.39 | +0.50 | +1.00 |
| Core + PPO epochs=2, +40 steps | +1.51 | +2.75 | +1.54 |

These are before/after observations, not causal estimates of the extensions. Continued training itself can improve the model. PPO2 versus core changes the number of optimizer updates as well as repeated use of rollout samples.

## Supplementary HRBench-8K evaluation

| Model | Score (%) |
|---|---:|
| Official Image SFT | 58.125 |
| Image core, step 60 | 60.875 |
| Image PPO2, step 60 | 60.250 |

PPO2 is **0.625 percentage points below core** on this benchmark. SFT is a reference, not the step-20 starting checkpoint. Without a measured step-20 HRBench-8K score, do not describe the SFT-to-extension difference as the gain of the 40-step continuation.

## Video scoring protocols

Both columns below use the full 5,130-question VSI benchmark, but apply different grading protocols. The overview uses the released PyVision eight-task macro; the native rule score must be reported separately.

| Video model | Released PyVision eight-task macro (%) | Native VSI (%) |
|---|---:|---:|
| Official SFT | 35.71 | 14.70 |
| Official released RL | 43.18 | 20.13 |
| Original-method step 40 | 34.95 | 14.81 |
| Starting checkpoint, step 120 | 35.98 | 15.01 |
| Core, step 160 | 36.98 | 17.06 |
| PPO2, step 160 | 37.52 | 17.18 |

The generic `category_macro_mean` in a raw summary averages the original categories and is not automatically the published eight-task macro. Keep the aggregation conventions and native scoring distinct. The two protocols differ substantially here; that discrepancy is not evidence that either number can substitute for the other.

## Official image-weight references

These scores use the server's evaluation protocol. Newly synchronized SFT-only rows do not establish protocol matching to an unpublished RL/extension evaluation; their `protocol_matches_sft` cell is intentionally empty.

| Benchmark | Records | Official Image SFT (%) | Official Image RL (%) |
|---|---:|---:|---:|
| DynaMath | 5,010 | 58.82 | — |
| MathVerse | 3,940 | 42.13 | — |
| MathVision full | 3,040 | 21.25 | — |
| MathVision mini | 304 | 22.70 | 22.70 |
| MathVista | 1,000 | 54.20 | — |
| MMMU | 900 | 50.89 | — |
| WeMath | 1,740 | 61.44 | — |
| TIR-Bench | 1,215 | 18.77 | — |
| CharXiv reasoning | 1,000 | 36.30 | — |
| CharXiv descriptive | 4,000 | 65.40 | — |

Scores are mean answer scores under the archived adapters, not a claim to every benchmark's official composite metric. MathVision full and mini are separate datasets and denominators. A dash means no available measured score in this snapshot.

## Extended video evaluations and anomalies

| Benchmark | Records | Official Video SFT (%) | Official Video RL (%) | Interpretation |
|---|---:|---:|---:|---|
| TempCompass caption matching | 1,503 | 72.06 | 74.52 | Archived mean answer score |
| TempCompass yes/no | 2,453 | 67.71 | 63.02 | Archived mean answer score |
| TempCompass multiple choice | 1,580 | 35.89 | 6.58 | RL score flagged as anomalous |
| MVBench public-19 | 3,800 | 0.16 | 0.13 | Both scores flagged as anomalous |

Keep the flagged values visible for transparency, but do not interpret them as reliable ability comparisons until formatting, adapters and scoring semantics are resolved. MVBench public-19 does not include the missing twentieth task.

## Controlled design

The planned design has eight video cells (`numeric × quota × tool`) and four image cells (`quota × tool`). All cells start from the same checkpoint within their modality and run 40 additional steps with PPO epochs=1. The two existing full-core cells can be reused; ten additional training runs complete the design. Runtime fixes are held constant. The factorial scripts are published, but completed original-control scores are not included in this snapshot. The archived [controlled-ablation-status.csv](../results/controlled-ablation-status.csv) records original cells as `evaluation_queued` and the remaining component cells as `queued`. The short component queue was deferred while long continuations took priority; these are historical states, not live process claims.

Report each component's effect only after its corresponding controls complete. A complete single-seed factorial design can describe interactions; it does not replace uncertainty estimates across training seeds.

## Long-continuation evidence

Four implemented core/PPO2 runs continue from Image step 60 and Video step 160 toward total step 700, with milestones at 200/400/700. The latest available [training snapshot](../results/continuation-training-snapshot.json) was observed at **2026-10-27 12:29:43 UTC** (20:29:43 Asia/Shanghai):

| Run | Last completed step | Latest saved step | Historical state |
|---|---:|---:|---|
| Image PPO2 | 311 | 301 | training |
| Image core | 401 | 401 | queued |
| Video PPO2 | 401 | 401 | queued |
| Video core | 401 | 401 | queued |

Temporary runtime links were unavailable during synchronization. Saved versus completed steps are distinct, particularly after a restart. None of these counters validates current process liveness, export availability or benchmark completion.

The available [milestone evaluation table](../results/continuation-evaluation-status.csv) contains 13 queued rows at steps 200 and 400 with empty scores. No measured 700-step result is available. Empty values are pending, not zero. The historical forecast below remains separate from measured data and is not a replacement for missing long-run results.

## Known limitations

- Some MVBench scores are close to zero and the official RL TempCompass multiple-choice score is anomalously low. Investigate answer formatting, adapters, and scoring semantics before interpreting these as ability changes.
- Full-record audits check coverage, record integrity, and protocol consistency. They do not prove that a judge or answer parser is semantically correct.
- Official released RL weights have unmatched training budgets and are reference scores rather than equal-budget controls.
- Longer training can plateau or regress. The repository contains no measured 700-step extension results.
- Weights, raw benchmark questions, media, per-example outputs, and operational logs are not included in this public source release.

## Rebuilding the figure

With matplotlib and numpy installed, run:

```bash
python scripts/render_project_assets.py
```

The plotting script reads the published CSV and calculates deltas from the matching starting checkpoints. No extrapolated or pending score is plotted.

## 700-step planning forecast

The overview retains previously published subjective forecast centers for the original algorithm (VStar 81%, HRBench-4K 71%, VSI macro 41%) and the implemented core extensions with PPO epochs=2 (82%, 73%, 43%). The extension ranges are respectively 77–86%, 69–76%, and 39–46%. These historical estimates were not updated or validated in the 2026-11-07 synchronization. They are planning judgments, not measurements or calibrated probability intervals. The original-algorithm forecast has no calibrated interval.

The target is **total global step 700**, including prior training. The historical starting checkpoints are image step 20 and video step 120. A continuation from those checkpoints therefore requires 680 and 580 additional rollout steps respectively; from the short-run endpoints at steps 60/160, 640 and 540 remain. These are historical budget calculations, not the remaining work in a live run. PPO2 is applied during extension training, while the starting checkpoints retain their original training history.

Assumptions are stable training, unchanged evaluation, and diminishing returns. The estimates are informed qualitatively by the short-run observations and official released weights; there is no fitted extrapolation model or empirical uncertainty calibration. Unknown original-algorithm control outcomes, reward overfitting, plateauing, and regression can all invalidate the centers and ranges. The official released weights are not equal-budget controls. Revise the forecast after matched controls and intermediate checkpoints are evaluated.

Forecasts are stored separately in [700-step-forecasts.csv](../results/700-step-forecasts.csv). They are excluded from the measured-results CSV and result figure, and do not authorize or launch longer training. Proposed spatial consistency, tool repair, and adaptive allocation methods are excluded.
