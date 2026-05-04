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

The current best research line is the HS300 downside propagation strategy:

- [HS300 Downside Propagation V1](research/strategy_lab/hs300_downside_propagation_v1/README.md)
- [Agent Skill](research/strategy_lab/hs300_downside_propagation_v1/SKILL.md)
- [Canonical Parameters](research/strategy_lab/hs300_downside_propagation_v1/BEST_STRATEGY.json)

This strategy is short-only. It detects `EMERGING_DOWN` constituent
propagation, requires `IPG_10` to show internal pressure still ahead of ETF
price, and holds the short expression for about 30 minutes. It supersedes the
older broad five-minute direction strategy, whose apparent edge was materially
affected by ETF price precision issues.

The current implementation is:

- [run_hs300_downside_walkforward.py](research/strategy_lab/run_hs300_downside_walkforward.py)

Curated result summaries are in:

- [hs300_downside_wf_static_precise_v1](results/510300_breadth_regime/hs300_downside_wf_static_precise_v1/README.md)

## Previous Clean Strategy

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
