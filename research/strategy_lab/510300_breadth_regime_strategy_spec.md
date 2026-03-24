# 510300 Breadth Regime Strategy Spec

## 1. Strategy Identity

This is the current best **clean** version of the short-horizon ETF prediction strategy for `510300.XSHG`.

It is not the earlier research version that used full-sample quantiles.

This spec corresponds to the **fixed-parameter, rolling-threshold** version whose result is stored in:

- [/Users/daweizhong/Documents/projects/artifacts/breadth_confidence_510300_regime_fixedthreshold_v1/report.json](/Users/daweizhong/Documents/projects/artifacts/breadth_confidence_510300_regime_fixedthreshold_v1/report.json)

The purpose of this document is exact reproducibility.

## 2. Objective

At each minute, predict whether `510300.XSHG` is likely to continue moving **up** over the next `5` minutes.

Only trade when:

- the model prediction is extremely confident,
- the market-wide breadth impulse agrees with the prediction,
- the day has a favorable early-session regime,
- the current breadth-vs-ETF gap is still large enough.

This version is **long-only**.

## 3. Data Inputs

### 3.1 Stock-side data

Use:

- [/Users/daweizhong/Documents/projects/stock_data](/Users/daweizhong/Documents/projects/stock_data)

Per trading day, each parquet contains all stocks at 1-minute frequency.

Required stock columns:

- `datetime`
- `open`
- `close`

### 3.2 ETF-side data

Use:

- [/Users/daweizhong/Documents/projects/ETF data core7](/Users/daweizhong/Documents/projects/ETF%20data%20core7)

Only use ETF code:

- `510300.XSHG`

Required ETF columns:

- `code`
- `datetime`
- `open`
- `close`

### 3.3 Derived panel artifact

The stock/ETF merged research panel was previously built into:

- [/Users/daweizhong/Documents/projects/artifacts/market_breadth_vs_etf_2023_2026/panel.parquet](/Users/daweizhong/Documents/projects/artifacts/market_breadth_vs_etf_2023_2026/panel.parquet)

Its construction logic comes from:

- [/Users/daweizhong/Documents/projects/Research/strategy_lab/analyze_market_breadth_vs_etf.py](/Users/daweizhong/Documents/projects/Research/strategy_lab/analyze_market_breadth_vs_etf.py)

### 3.4 Model prediction artifact

The out-of-sample minute predictions were previously built into:

- [/Users/daweizhong/Documents/projects/artifacts/breadth_confidence_model_v1/predictions.parquet](/Users/daweizhong/Documents/projects/artifacts/breadth_confidence_model_v1/predictions.parquet)

Its construction logic comes from:

- [/Users/daweizhong/Documents/projects/Research/strategy_lab/breadth_confidence_model.py](/Users/daweizhong/Documents/projects/Research/strategy_lab/breadth_confidence_model.py)

## 4. Core Market Breadth Definitions

All definitions are minute-level and computed separately for each trading day.

### 4.1 Per-stock minute sign

For each stock `i` at minute `t`:

```text
sign_i,t =
  +1 if close_i,t - open_i,t > 0
  -1 if close_i,t - open_i,t < 0
   0 otherwise
```

### 4.2 Market breadth sum

```text
breadth_sum_t = Σ_i sign_i,t
```

### 4.3 Market breadth ratio

Let `N_t` be the number of stocks present at minute `t`.

```text
breadth_ratio_t = breadth_sum_t / N_t
```

### 4.4 One-minute breadth impulse

```text
breadth_diff_1_t = breadth_sum_t - breadth_sum_(t-1)
```

### 4.5 One-minute breadth-ratio change

```text
breadth_ratio_diff_1_t = breadth_ratio_t - breadth_ratio_(t-1)
```

### 4.6 Short rolling breadth means

Within the current day:

```text
breadth_ratio_ma3_t = mean(breadth_ratio_(t-2:t))
breadth_ratio_ma5_t = mean(breadth_ratio_(t-4:t))
```

## 5. ETF Return Definitions

For ETF minute `t`:

```text
ret_oc_t = log(close_t / open_t)
ret_cc_1_t = log(close_t / close_(t-1))
```

Forward target for this strategy:

```text
fwd_ret_5m_t = log(close_(t+5) / close_t)
```

## 6. Prediction Model Layer

This strategy uses the out-of-sample predictions produced by the base model.

### 6.1 Model type

Ridge regression with `alpha = 10.0`.

### 6.2 Training window

For each test day `D`:

- train on the previous `60` trading days,
- predict on day `D`.

### 6.3 Base model features

Use exactly these 11 features:

- `breadth_ratio`
- `breadth_diff_1`
- `breadth_ratio_diff_1`
- `breadth_ratio_ma3`
- `breadth_ratio_ma5`
- `ret_oc`
- `ret_cc_1`
- `breadth_ret_gap = breadth_ratio - ret_oc`
- `breadth_impulse_x_gap = breadth_ratio_diff_1 * breadth_ret_gap`
- `breadth_level_x_ret = breadth_ratio * ret_oc`
- `minute_frac = minute_idx / max(minute_idx of day)`

### 6.4 Target

```text
y_t = fwd_ret_5m_t
```

### 6.5 Standardization

Within each training window:

- z-score each feature using training-window mean/std,
- z-score the target using training-window mean/std.

### 6.6 Prediction output

For each minute:

- `pred_ret_t`: model-predicted future 5-minute log return
- `pred_sign_t = sign(pred_ret_t)`

### 6.7 Confidence score

To reproduce the current implementation exactly, use the same confidence definition as the code:

```text
confidence_t = abs(pred_ret_t) / resid_std_train
```

Where `resid_std_train` is the standard deviation of training residuals in the model-fitting routine.

Important:

- this matches the current implementation exactly,
- even though the scaling choice is not ideal,
- do not change it if exact replication is required.

## 7. Strategy Filter Variables

These are computed from the minute panel and used as post-model filters.

### 7.1 Breadth alignment

```text
breadth_sign_t = sign(breadth_diff_1_t)

breadth_align_t = 1
  if pred_sign_t != 0
  and breadth_sign_t != 0
  and pred_sign_t == breadth_sign_t
  else 0
```

### 7.2 Current breadth/ETF gap

```text
gap_abs_t = abs(breadth_ratio_t - ret_oc_t)
```

### 7.3 Open30 flip rate

For a given day `D`, only use the first 30 intraday minutes, indexed `0..29`.

Define:

```text
breadth_flip_t = 1
  if breadth_sign_t != 0
  and breadth_sign_(t-1) != 0
  and breadth_sign_t != breadth_sign_(t-1)
  else 0
```

Then:

```text
open30_flip_rate_D = mean(breadth_flip_t for t in first 30 minutes)
```

### 7.4 Open30 absolute breadth impulse mean

```text
open30_abs_bdiff_mean_D = mean(abs(breadth_diff_1_t) for t in first 30 minutes)
```

### 7.5 Open30 trendness

For ETF one-minute close-to-close returns during the first 30 minutes:

```text
open30_trendness_D =
  abs(Σ ret_cc_1_t) / Σ abs(ret_cc_1_t)
```

If the denominator is zero, set as missing and the day should fail the filter.

## 8. Final Strategy Logic

This is the current best clean version.

### 8.1 Strategy universe

- instrument: `510300.XSHG`
- direction: `LONG` only
- prediction horizon: `5` minutes

### 8.2 Base candidate set for each test day

For each minute on day `D`, start from the model predictions for that day and keep only rows satisfying:

1. confidence is above a rolling threshold
2. `breadth_align_t = 1`
3. `minute_idx >= 30`
4. `pred_sign_t > 0`
5. `abs(breadth_diff_1_t)` is below a rolling shock threshold
6. day-level regime is acceptable
7. `gap_abs_t` is above a rolling gap threshold

### 8.3 Fixed parameters

Use these fixed parameter settings:

```text
conf_q   = 0.99
shock_q  = 0.90
flip_q   = 0.70
bdiff_q  = 0.90
trend_min = 0.10
gap_q    = 0.50
```

### 8.4 Rolling threshold estimation window

For each test day `D`, use only the previous `120` trading days.

Call this trailing set `Hist(D)`.

### 8.5 Threshold estimation order

This order matters and must be reproduced exactly.

For each test day `D`:

1. Start from `Hist(D)`.
2. Compute confidence threshold:

```text
conf_thr_D = quantile(confidence on Hist(D), 0.99)
```

3. Filter both `Hist(D)` and day `D` rows to `confidence >= conf_thr_D`.
4. Filter both sides to `breadth_align = 1`.
5. Filter both sides to `minute_idx >= 30`.
6. Filter both sides to `pred_sign > 0`.
7. Compute shock threshold on the filtered history:

```text
shock_thr_D = quantile(abs(breadth_diff_1) on filtered Hist(D), 0.90)
```

8. Filter both sides to `abs(breadth_diff_1) <= shock_thr_D`.
9. Compute regime thresholds on the filtered history:

```text
flip_thr_D  = quantile(open30_flip_rate on filtered Hist(D), 0.70)
bdiff_thr_D = quantile(open30_abs_bdiff_mean on filtered Hist(D), 0.90)
```

10. Filter both sides to:

```text
open30_flip_rate <= flip_thr_D
open30_abs_bdiff_mean <= bdiff_thr_D
open30_trendness >= 0.10
```

11. Compute gap threshold on the doubly filtered history:

```text
gap_thr_D = quantile(gap_abs on filtered Hist(D), 0.50)
```

12. Final day-`D` trades are rows satisfying:

```text
gap_abs >= gap_thr_D
```

### 8.6 Output trade set

Every minute row surviving the full chain is treated as one trade signal.

Important:

- multiple signals can occur on the same day,
- the current research evaluation allows overlapping 5-minute holdings,
- this is a signal-level study, not a portfolio-level execution engine.

## 9. Execution Assumption

Current backtest assumption:

- signal is formed at minute `t` after the minute bar is observed,
- entry is effectively evaluated from that minute’s `close`,
- exit is at minute `t+5` `close`.

Return used:

```text
signed_ret_bps_t = pred_sign_t * fwd_ret_5m_t * 10000
net_signed_ret_bps_t = signed_ret_bps_t - 6
```

Round-trip cost is fixed at:

- `6 bps`

Important:

- this is cleaner than using future information in filters,
- but it is still a bar-close research fill,
- it is not yet the stricter `next-bar open` execution version.

## 10. Current Clean Performance

This refers to the fixed-parameter, rolling-threshold version.

From:

- [/Users/daweizhong/Documents/projects/artifacts/breadth_confidence_510300_regime_fixedthreshold_v1/report.json](/Users/daweizhong/Documents/projects/artifacts/breadth_confidence_510300_regime_fixedthreshold_v1/report.json)

Summary:

- `n_preds = 59`
- `n_days = 39`
- `avg_signed_bps_gross = 17.9162`
- `avg_signed_bps_net = 11.9162`
- `hit_rate = 54.24%`
- `zero_rate = 37.29%`
- `nonzero_hit_rate = 86.49%`
- `worst_day_net_bps = -186.43`

## 11. Reproducibility Requirements

If coding from scratch, the implementation must preserve:

1. same breadth definitions,
2. same ridge model feature set,
3. same rolling train/test split,
4. same confidence formula,
5. same filter order,
6. same rolling threshold estimation window,
7. same fixed parameters,
8. same trade return definition,
9. same `6 bps` cost assumption,
10. same overlapping-signal evaluation convention.

If any of these change, the result is no longer the same strategy.

## 12. What This Strategy Is Not

This spec does **not** describe:

- an options execution strategy,
- a next-bar-open execution engine,
- a capital-constrained portfolio simulation,
- a non-overlapping position manager,
- a multi-ETF model.

It is specifically:

- one ETF,
- one horizon,
- one prediction model,
- one rolling-threshold long-only signal strategy.
