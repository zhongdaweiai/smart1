# smart1

This repository is the code and documentation home for the intraday index / ETF research project.

Current focus:

- minute-level market breadth research,
- `510300.XSHG` short-horizon prediction,
- clean walk-forward filtering,
- single-factor and strategy-lab experiments.

## Repository Layout

- [research/strategy_lab](/Users/daweizhong/Documents/projects/smart1/research/strategy_lab)
  Main intraday ETF/index research code, experiment scripts, and strategy specs.
- [scripts](/Users/daweizhong/Documents/projects/smart1/scripts)
  Supporting factor-testing and legacy research scripts kept for reuse.
- [results/510300_breadth_regime](/Users/daweizhong/Documents/projects/smart1/results/510300_breadth_regime)
  Curated small summary outputs and trade logs, split by version:
  `research_v1`, `fixedthreshold_v1`, `walkforward_v1`.
- [AGENTS.md](/Users/daweizhong/Documents/projects/smart1/AGENTS.md)
  Repo-specific coding and research notes for future agent work.

## Current Best Clean Strategy

The current reference strategy description is:

- [510300_breadth_regime_strategy_spec.md](/Users/daweizhong/Documents/projects/smart1/research/strategy_lab/510300_breadth_regime_strategy_spec.md)

The current clean fixed-parameter rolling-threshold result is summarized in:

- [report.json](/Users/daweizhong/Documents/projects/smart1/results/510300_breadth_regime/fixedthreshold_v1/report.json)

The stricter daily walk-forward parameter-selection version is included in:

- [breadth_confidence_510300_regime_walkforward_v1.py](/Users/daweizhong/Documents/projects/smart1/research/strategy_lab/breadth_confidence_510300_regime_walkforward_v1.py)

## Data Policy

Large local datasets are intentionally not committed:

- `stock_data`
- `ETF data`
- `ETF data core7`
- `fundamentals_data`
- large artifacts and parquet outputs

The code currently assumes these datasets exist locally under the historical research workspace path:

- `/Users/daweizhong/Documents/projects`

That path assumption is preserved for reproducibility of the current research code.

## Suggested Next Cleanup

- convert hard-coded absolute paths to repo-relative config,
- separate reusable library code from one-off experiment scripts,
- add a strict next-bar-open execution version of the current 510300 strategy.
