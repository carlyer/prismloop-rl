# Experiments and evidence

[← Overview](../README.md)

## Published snapshot

The aggregate table in [benchmark-results.csv](../results/benchmark-results.csv) is the measured score snapshot from the initial publication. Branding changes do not refresh running experiments or add uncompleted scores. The README highlights six extension evaluations: image core/PPO2 on VStar and HRBench-4K, and video core/PPO2 on VSI.

| Protocol | Definition |
|---|---|
| VStar | 6,112 evaluated trajectories; avg@32 |
| HRBench-4K | 800 examples |
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

## Controlled design

The planned design has eight video cells (`numeric × quota × tool`) and four image cells (`quota × tool`). All cells start from the same checkpoint within their modality and run 40 additional steps with PPO epochs=1. The two existing full-core cells can be reused; ten additional training runs complete the design. Runtime fixes are held constant. The factorial scripts are published, but completed control scores are not included in this snapshot.

Report each component's effect only after its corresponding controls complete. A complete single-seed factorial design can describe interactions; it does not replace uncertainty estimates across training seeds.

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

The overview includes subjective forecast centers for the original algorithm (VStar 81%, HRBench-4K 71%, VSI macro 41%) and the implemented core extensions with PPO epochs=2 (82%, 73%, 43%). The extension ranges are respectively 77–86%, 69–76%, and 39–46%. These are planning judgments, not measurements or calibrated probability intervals. The original-algorithm forecast has no calibrated interval.

The target is **total global step 700**, including prior training. The historical starting checkpoints are image step 20 and video step 120. A continuation from those checkpoints therefore requires 680 and 580 additional rollout steps respectively; from the current short-run endpoints, 640 and 540 remain. PPO2 is applied during extension training, while the starting checkpoints retain their original training history.

Assumptions are stable training, unchanged evaluation, and diminishing returns. The estimates are informed qualitatively by the short-run observations and official released weights; there is no fitted extrapolation model or empirical uncertainty calibration. Unknown original-algorithm control outcomes, reward overfitting, plateauing, and regression can all invalidate the centers and ranges. The official released weights are not equal-budget controls. Revise the forecast after matched controls and intermediate checkpoints are evaluated.

Forecasts are stored separately in [700-step-forecasts.csv](../results/700-step-forecasts.csv). They are excluded from the measured-results CSV and result figure, and do not authorize or launch longer training. Proposed spatial consistency, tool repair, and adaptive allocation methods are excluded.
