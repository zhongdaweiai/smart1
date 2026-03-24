# AGENTS

## Scope

This repository is for managing research code and documentation for intraday ETF/index prediction work.

The current primary strategy line is:

- `510300` breadth-regime continuation

## Working Rules

- Do not commit local market datasets or large parquet outputs by default.
- Treat files in [results/510300_breadth_regime](/Users/daweizhong/Documents/projects/smart1/results/510300_breadth_regime) as curated small outputs that are safe to version.
- Keep new research scripts under [research/strategy_lab](/Users/daweizhong/Documents/projects/smart1/research/strategy_lab) unless they are clearly reusable utilities.
- Keep supporting standalone scripts under [scripts](/Users/daweizhong/Documents/projects/smart1/scripts).
- Preserve experiment lineage. If a script is superseded, do not silently overwrite the old one; create a new version or add a clear note.

## Reproducibility Notes

- Much of the current code still uses absolute paths under `/Users/daweizhong/Documents/projects`.
- If refactoring, prefer introducing config-based paths rather than changing logic.
- The strategy spec in [510300_breadth_regime_strategy_spec.md](/Users/daweizhong/Documents/projects/smart1/research/strategy_lab/510300_breadth_regime_strategy_spec.md) is the reference for exact reproduction of the current clean strategy.

## Immediate Priorities

- keep the clean no-lookahead versions separate from exploratory scripts,
- move toward next-bar-open execution assumptions,
- reduce overlap between experiment scripts once the current strategy stabilizes.
