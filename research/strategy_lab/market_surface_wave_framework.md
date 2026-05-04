# Market Surface Wave Framework

## 1. Goal

This document defines a research framework for intraday index prediction by
viewing the full A-share market as a dynamic surface made of thousands of
price columns.

The central idea is:

> Index prediction is not mainly about the current breadth level.
> It is about whether the market surface is forming a directional wave that is:
> coherent, expanding, propagating into index-relevant names, not yet fully
> priced by the ETF/futures, and not yet exhausted.

This is a framework, not a single factor.


## 2. Core Representation

For each stock `i` and minute `t`, define a height field:

`H(i, t)`

This height should not default to raw price. For short-horizon prediction, more
useful choices are:

- `1m` or `k`-minute return
- volatility-scaled return
- signed turnover impulse
- signed amount-normalized move
- residual move after removing market beta

The market at time `t` becomes a cross-sectional surface:

`S_t = { H(i, t) } for all active stocks i`

Over time, `{S_t}` forms a surface movie.


## 3. Why This Matters for Index Prediction

The ETF or IF price is not the market surface itself.

It is a weighted projection of the market surface:

`Index_t ~= Projection( S_t ; index weights )`

Therefore, the useful question is not:

- "How many stocks are up?"

The useful questions are:

- Is the surface becoming more directional?
- Is the directional wave spreading?
- Is the wave reaching high-weight names?
- Is the ETF/IF price lagging that internal wave?
- Is the wave healthy or already over-concentrated and exhausted?


## 4. Three Surface Families

We should never work with only one surface.

### 4.1 Whole-Market Equal-Weight Surface

Purpose:

- detect global wave formation
- detect broad risk-on / risk-off pressure
- detect early diffusion

Examples:

- full-market return surface
- full-market signed amount surface

### 4.2 Index-Weighted Surface

Purpose:

- detect whether the wave is relevant to the target index
- measure whether high-weight names are confirming the move

Examples:

- HS300-weighted return surface
- top-weight bucket surface

### 4.3 Sector Sub-Surfaces

Purpose:

- detect whether the wave is concentrated or diffusing
- measure sector relay and sector confirmation

Examples:

- bank surface
- broker surface
- consumer surface
- semis/electronics surface


## 5. State Variables of the Surface

We should organize research around geometric or physical objects, not ad hoc
factor names.

### 5.1 Surface Level

What is the current directional elevation of the surface?

Examples:

- equal-weight signed breadth
- weighted signed breadth
- cross-sectional mean return

Interpretation:

- tells us the current state
- usually better for gating than direct forecasting

### 5.2 Surface Velocity

How fast is the surface changing?

Examples:

- `dBreadth`
- `dWeightedBreadth`
- change in sector participation

Interpretation:

- often more predictive than level
- captures whether the wave is strengthening now

### 5.3 Surface Acceleration

Is the strengthening itself accelerating or decelerating?

Examples:

- second difference of breadth
- second difference of weighted breadth
- acceleration of active participation

Interpretation:

- positive acceleration: wave build-up
- negative acceleration: strong but fading

### 5.4 Surface Jerk

Is there a sudden change in acceleration?

Interpretation:

- useful for detecting snapbacks, failed breakouts, panic bursts


## 6. Wave Propagation Objects

These are the most important objects for index prediction.

### 6.1 Wave Origin

Where did the wave start?

Possible origins:

- high-weight leaders first
- small/mid cap first
- one sector first
- broad synchronous launch

Why it matters:

- different origins imply different future transmission paths

### 6.2 Wave Front

How many new stocks or sectors are joining the directional move right now?

Examples:

- new-up-stock ratio
- new-down-stock ratio
- new-active-sector ratio
- new-high-weight participant ratio

Interpretation:

- a live wave keeps recruiting new participants
- a dying wave stops expanding

### 6.3 Wave Speed

How quickly is the move traveling from one group to another?

Examples:

- leader group at `t-1` vs follower group at `t`
- sector A impulse at `t-1` vs sector B confirmation at `t`

Interpretation:

- measures propagation efficiency
- especially important for `5m` to `30m` prediction

### 6.4 Wave Penetration

How deeply has the equal-weight wave penetrated into the index-weighted core?

Examples:

- weighted breadth minus equal-weight breadth
- high-weight bucket confirmation score
- top-30 vs bottom-200 relay

Interpretation:

- full market excitement without core penetration is not enough for HS300/IF

### 6.5 Wave Relay

Is the market passing the move forward in sequence?

Examples:

- top weights move first, broader basket follows
- sector leaders move first, secondary sectors catch up

Interpretation:

- relay is healthier than isolated explosion


## 7. Surface Shape Objects

These distinguish healthy continuation from fragile spikes.

### 7.1 Surface Slope

Measure the slope of directional participation along some ordering axis:

- index weight
- liquidity
- market cap
- sector rank

Interpretation:

- if slope steepens in the index-relevant direction, the move becomes more
  tradeable for the target ETF/futures

### 7.2 Surface Curvature

Measure whether the surface is smooth or dominated by sharp spikes.

Examples:

- top-K contribution share
- weighted participation concentration
- curvature proxy from upper-tail contribution dominance

Interpretation:

- low curvature: healthy, diffused wave
- high curvature: fragile, concentrated push

### 7.3 Surface Roughness

Measure local irregularity or noise.

Examples:

- sign flip rate across minutes
- cross-sectional disorder
- sector participation instability

Interpretation:

- trend days tend to be smoother
- chop days are rough and jagged

### 7.4 Surface Entropy

Measure how dispersed the move is across sectors or weight buckets.

Interpretation:

- higher entropy often means broader support
- but extremely high entropy with no core confirmation may not help HS300


## 8. Surface Energy and Flow

Price alone is not enough. Volume/turnover must be treated as energy input.

### 8.1 Directional Energy

Examples:

- `sum( |w_i * r_i| )`
- `sum( signed_amount_i )`
- abnormal signed turnover

Interpretation:

- wave continuation requires energy input

### 8.2 Energy Diffusion

Does the energy remain concentrated in a few names or spread to many names?

Interpretation:

- broad energy diffusion is healthier
- concentrated blow-off often marks late-stage behavior

### 8.3 Follow Flow

Do followers add turnover in the same direction after leaders move?

Interpretation:

- this is a direct behavioral proxy for greed/fear propagation


## 9. Internal Surface vs ETF/IF Price

This is the bridge from internal market state to tradable prediction.

### 9.1 Surface-Price Gap

Define an internal pressure score from the surface:

- breadth
- weighted breadth
- acceleration
- diffusion
- relay
- sector confirmation

Then compare it with current ETF/IF price move.

Interpretation:

- internal surface stronger than price: underpriced wave
- price stronger than internal surface: overextended move

This is a central idea.

### 9.2 Projection Lag

The target ETF/IF may lag the internal wave by a few minutes.

This is where alpha lives:

- internal pressure exists
- wave is transmitting
- the traded instrument has not fully caught up


## 10. Four-Layer Prediction Stack

Every factor should map into one of these layers.

### Layer A: Wave Existence

Question:

- Is there a real directional surface wave, or only noise?

Typical objects:

- surface roughness
- level
- velocity
- entropy

### Layer B: Wave Direction

Question:

- If there is a wave, is it up or down?

Typical objects:

- weighted breadth sign
- sector sign balance
- leader sign
- energy sign

### Layer C: Wave Transmission

Question:

- Is the wave reaching the target index core and the ETF/IF price?

Typical objects:

- penetration
- relay
- projection gap
- weighted confirmation

### Layer D: Wave Health / Exhaustion

Question:

- Is the wave still healthy, or already near failure?

Typical objects:

- curvature
- concentration
- diffusion slowdown
- acceleration decay
- price leading internal pressure too much


## 11. Candidate Factor Families

These are the main factor families we should build next.

### 11.1 Surface Velocity Family

Purpose:

- capture whether the market wave is strengthening now

Candidate members:

- `EQ_Breadth_Velocity`
- `W_Breadth_Velocity`
- `SectorParticipation_Velocity`
- `NewJoiner_Velocity`

### 11.2 Surface Acceleration Family

Purpose:

- capture build-up vs late-stage slowdown

Candidate members:

- `Breadth_Acceleration`
- `WeightedBreadth_Acceleration`
- `Wavefront_Acceleration`
- `Energy_Acceleration`

### 11.3 Diffusion Family

Purpose:

- measure how far the wave is spreading

Candidate members:

- `NewUpStockRatio`
- `NewDownStockRatio`
- `NewSectorJoinRatio`
- `NewHighWeightJoinRatio`

### 11.4 Penetration Family

Purpose:

- measure how index-relevant the wave is

Candidate members:

- `EW_to_W_Penetration`
- `TopWeightConfirm`
- `WeightBucketSlope`
- `LeaderToCoreRelay`

### 11.5 Shape Health Family

Purpose:

- distinguish healthy moves from fragile spikes

Candidate members:

- `SurfaceCurvature`
- `SurfaceRoughness`
- `TopKConcentration`
- `SectorEntropy`
- `EffectiveParticipation`

### 11.6 Price Gap Family

Purpose:

- identify underpriced or overextended internal waves

Candidate members:

- `InternalPressureGap`
- `ProjectionLag`
- `PriceAheadOfSurface`

### 11.7 Behavioral Flow Family

Purpose:

- measure greed/fear propagation through turnover

Candidate members:

- `FollowFlow`
- `FlowDiffusion`
- `FlowAcceleration`
- `LeaderFlowRelay`


## 12. Multi-Projection Principle

A useful wave should be examined in several projections:

- stock projection
- sector projection
- weight-bucket projection
- size/liquidity projection

A move that looks strong in only one projection is less trustworthy.

The best continuation candidates often look consistent across several
projections:

- equal-weight wave exists
- sector diffusion expands
- weight core confirms
- ETF/IF has not fully caught up


## 13. Regime Thinking

The same surface variables mean different things in different regimes.

We should explicitly classify:

- smooth trend day
- shock day
- high-vol reversal day
- low-energy chop day

Examples:

- high breadth velocity on a smooth day can imply continuation
- high breadth velocity right after a panic shock can imply mean reversion

So every wave factor should be tested both:

- unconditionally
- conditionally by regime


## 14. Suggested Labels

Do not only label future `1m` or `5m` return sign.

We should build several labels:

### 14.1 Direction Label

- sign of future return over `1m`, `3m`, `5m`, `10m`, `30m`

### 14.2 Continuation Label

- whether future move continues in the current wave direction

### 14.3 Magnitude Label

- whether future move exceeds a cost-adjusted threshold

### 14.4 Trendness Label

- whether the future path is smooth or jagged

### 14.5 Exhaustion Label

- whether the current move soon stalls or reverses


## 15. Research Priorities

If we only do a few things next, do these first.

### Priority 1

Build the surface kinematics family:

- velocity
- acceleration
- wavefront expansion

### Priority 2

Build the penetration family:

- equal-weight to weighted-core transmission
- high-weight confirmation

### Priority 3

Build the shape-health family:

- curvature
- roughness
- concentration

### Priority 4

Build the price-gap family:

- internal surface pressure vs ETF/IF price


## 16. A Practical Research Formula

At the highest level, define:

`WaveScore = Directionality + Diffusion + Penetration + Energy - Exhaustion`

Where:

- `Directionality` answers whether the surface has a clear sign
- `Diffusion` answers whether the wave is spreading
- `Penetration` answers whether the wave matters for the target index
- `Energy` answers whether the move has follow-through fuel
- `Exhaustion` answers whether the wave has become too concentrated or over-priced

This should not be one rigid formula.
It is the construction principle for later factors and models.


## 17. First Three Concrete Experiments

### Experiment A: Surface Kinematics

Goal:

- test whether surface velocity and acceleration predict `5m`, `10m`, `30m`
  ETF/IF continuation

Inputs:

- full-market breadth
- weighted breadth
- new-joiner counts

### Experiment B: Penetration and Projection Lag

Goal:

- test whether equal-weight waves that penetrate into the high-weight core
  while ETF price still lags have stronger forward returns

Inputs:

- equal-weight breadth
- weighted breadth
- top-weight confirmation
- ETF current move

### Experiment C: Healthy vs Fragile Waves

Goal:

- test whether low-curvature, low-roughness, high-diffusion waves continue more
  than concentrated spikes

Inputs:

- TopK concentration
- effective participation
- entropy
- roughness


## 18. Engineering Guidance

Suggested file grouping:

- `surface_core.py`
  - build active universe
  - compute height fields

- `surface_kinematics.py`
  - velocity
  - acceleration
  - jerk

- `surface_diffusion.py`
  - new joiners
  - sector spread
  - weight bucket spread

- `surface_shape.py`
  - curvature
  - roughness
  - entropy
  - concentration

- `surface_projection.py`
  - equal-weight to weighted-core penetration
  - price gap
  - projection lag

- `surface_flow.py`
  - signed turnover
  - follow flow
  - flow relay


## 19. Hard Rules to Avoid Fake Research Progress

- Never test only level variables and conclude the wave idea is weak.
- Never use only one projection.
- Never confuse extreme shock with healthy continuation.
- Never judge continuation without measuring exhaustion.
- Never optimize all thresholds globally; use rolling or walk-forward logic.
- Never claim true IF performance from spot mapping without stating basis and
  roll are omitted.


## 20. Final Principle

The key shift is:

> Stop thinking of the market as a list of stock returns.
> Start thinking of it as a time-varying surface that can form, transmit,
> strengthen, bend, diffuse, and collapse in waves.

The target index rises or falls not because breadth is high in a static sense,
but because a directional market wave is moving through the surface and being
projected into the index basket faster than price has fully absorbed.
