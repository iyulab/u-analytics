# u-analytics

[![Crates.io](https://img.shields.io/crates/v/u-analytics.svg)](https://crates.io/crates/u-analytics)
[![docs.rs](https://docs.rs/u-analytics/badge.svg)](https://docs.rs/u-analytics)
[![CI](https://github.com/iyulab/u-analytics/actions/workflows/ci.yml/badge.svg)](https://github.com/iyulab/u-analytics/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Statistical process control, process capability analysis, Weibull reliability,
change-point detection, correlation, regression, distribution analysis, and
hypothesis testing for industrial quality engineering.

## Modules

| Module | Description |
|--------|-------------|
| `spc` | Control charts (X̄-R, X̄-S, I-MR, P, NP, C, U, Laney P'/U', G, T) with selectable run tests (`RuleSet`; Nelson/WE presets), subgroups n=2..=25 |
| `capability` | Process capability indices (Cp, Cpk, Pp, Ppk, Cpm), sigma level, and Box-Cox non-normal capability |
| `weibull` | Weibull parameter estimation (MLE, MRR) and reliability analysis (R(t), MTBF, B-life) |
| `detection` | Change-point detection (CUSUM, EWMA, PELT) and one-shot anomaly scoring by spectral residual saliency (Ren et al. 2019) |
| `smoothing` | Time series smoothing (SES, Holt linear trend, Holt-Winters seasonal) |
| `seasonality` | Periodogram (zero-padded FFT) and dominant-period estimation — AutoPeriod: permutation-thresholded peaks refined on the ACF |
| `correlation` | Correlation analysis (Pearson, Spearman, Kendall, partial, correlation matrices) |
| `regression` | Regression analysis (simple OLS, multiple OLS, VIF multicollinearity) |
| `distribution` | Distribution analysis (ECDF, histogram bins — Sturges/Scott/FD/Fixed, QQ-plot, KS test) |
| `testing` | Hypothesis testing (t-tests, ANOVA, chi-squared, normality — SW/AD/JB) |

## Features

### Statistical Process Control (SPC)

Control charts for monitoring process stability:

- **Variables charts**: X̄-R, X̄-S, Individual-MR
- **Attributes charts**: P, NP, C, U
- **Overdispersion-adjusted**: Laney P' and U' (φ coefficient corrects for between-subgroup variation)
- **Rare events**: G chart (geometric distribution) and T chart (exponential) for low-defect processes
- **Run rules**: Nelson (8 rules), Western Electric (4 rules)

```rust
use u_analytics::spc::{XBarRChart, ControlChart};

let mut chart = XBarRChart::new(5).expect("5 is a supported subgroup size");
chart.add_sample(&[25.0, 26.0, 24.5, 25.5, 25.0]).unwrap();
chart.add_sample(&[25.2, 24.8, 25.1, 24.9, 25.3]).unwrap();
chart.add_sample(&[25.1, 25.0, 24.7, 25.3, 24.9]).unwrap();

if chart.is_in_control() {
    println!("Process is stable");
}
```

```rust
use u_analytics::spc::{laney_p_chart, g_chart};

// Laney P' chart for overdispersed proportion data
// samples: (defective count, subgroup size)
let samples = vec![(3u64, 100u64), (5, 120), (2, 95)];
let chart = laney_p_chart(&samples, None).unwrap(); // Some(LaneyStandard { .. }) for Phase II
println!("p̄ = {:.4}, φ = {:.4}", chart.p_bar, chart.phi);

// G chart for rare events (e.g., days between nonconformances)
let inter_event_counts = vec![12.0, 8.0, 25.0, 5.0, 18.0];
let gchart = g_chart(&inter_event_counts).unwrap();
```

### Process Capability

Capability indices quantifying process performance against specifications:

- **Short-term**: Cp, Cpk, Cpu, Cpl
- **Long-term**: Pp, Ppk, Ppu, Ppl
- **Taguchi**: Cpm
- **Sigma level**: PPM ↔ sigma conversion (1.5σ shift convention)
- **Non-normal**: Box-Cox transformation + capability on transformed scale

```rust
use u_analytics::capability::{ProcessCapability, sigma_to_ppm};

let spec = ProcessCapability::new(Some(220.0), Some(200.0)).unwrap();
let data = [210.0, 209.5, 210.2, 209.8, 210.1, 210.3, 209.7, 210.0];
let indices = spec.compute(&data, 0.15).unwrap();

println!("Cp = {:.2}, Cpk = {:.2}", indices.cp.unwrap(), indices.cpk.unwrap());
println!("6σ PPM = {:.1}", sigma_to_ppm(6.0)); // 3.4

// No within-subgroup sigma (a flat vector): long-term indices only.
let overall = spec.compute_overall(&data).unwrap();
assert!(overall.pp.is_some() && overall.cp.is_none());
```

`compute` takes the within-subgroup sigma from a control chart (`R-bar/d2`,
`S-bar/c4`, `MR-bar/d2`); `compute_overall` has none and so reports only
Pp/Ppk/Ppu/Ppl and Cpm, leaving `cp`, `cpk`, `cpu`, `cpl` and `std_dev_within`
`None`. Filling the short-term names from the overall sigma would make `cp`
equal `pp` for every input.

```rust
use u_analytics::capability::{boxcox_capability, DEFAULT_LAMBDA_RANGE};

// Non-normal data: estimate λ over [-5, 5], transform spec limits, compute Ppk
let skewed_data = vec![0.5, 1.2, 0.8, 2.1, 0.3, 1.7, 0.9, 1.4];
let result = boxcox_capability(&skewed_data, Some(5.0), Some(0.1), DEFAULT_LAMBDA_RANGE).unwrap();
if result.lambda_at_bound {
    // the likelihood was still rising at an end of the range: λ is that limit
}
let ppk = result.indices.and_then(|i| i.ppk); // `indices` is None without limits
println!("λ = {:.3}, Ppk = {:?}", result.lambda, ppk);
```

### Weibull Reliability

Parameter estimation and reliability engineering metrics:

- **MLE**: Maximum Likelihood Estimation (Newton-Raphson)
- **MRR**: Median Rank Regression (Bernard's approximation)
- **Reliability**: R(t), hazard rate, MTBF, B-life

```rust
use u_analytics::weibull::{weibull_mle, ReliabilityAnalysis};

let failure_times = [150.0, 200.0, 250.0, 300.0, 350.0, 400.0];
let fit = weibull_mle(&failure_times).unwrap();

let ra = ReliabilityAnalysis::from_mle(&fit);
println!("R(200h) = {:.1}%", ra.reliability(200.0) * 100.0);
println!("MTBF = {:.0}h", ra.mtbf());
println!("B10 life = {:.0}h", ra.b_life(0.10).unwrap());
```

### Change-Point Detection

Algorithms for detecting process mean shifts:

- **CUSUM**: Cumulative Sum chart (Page, 1954)
- **EWMA**: Exponentially Weighted Moving Average (Roberts, 1959)

```rust
use u_analytics::detection::Cusum;

let cusum = Cusum::new(10.0, 1.0).unwrap();
let data = [10.1, 9.9, 10.0, 10.2, 12.0, 12.1, 11.9, 12.3];
let signals = cusum.signal_points(&data);
```

### Spectral residual — where does the series do something its structure does not explain?

`SpectralResidual` scores every point without a trained model and without
assuming a period (Ren et al. 2019, the method behind the Microsoft anomaly
service): the log amplitude spectrum minus its moving average, transformed
back with the original phase, is the *saliency map*, large at spikes, steps
and dropouts. A point's `score` is its saliency relative to the judgement
window before it; it is an anomaly above the threshold (τ = 3) when it also
stands `min_zscore` standard deviations from the level before it. Each point
also carries an `expected` value — the series with anomalies replaced by
their neighbours, reconstructed from its low frequencies — and a
`lower`/`upper` band whose coverage is `sensitivity` percent; the band is for
the chart, the decision is the score.

```rust
use u_analytics::detection::SpectralResidual;

let mut series: Vec<f64> = (0..60).map(|t| (t as f64 * 0.3).sin()).collect();
series[40] += 4.0;
let points = SpectralResidual::new().analyze(&series).unwrap();
assert!(points[40].is_anomaly);
let flagged: Vec<usize> = points.iter().filter(|p| p.is_anomaly).map(|p| p.index).collect();
assert_eq!(flagged, vec![40]);
```

Defaults are the paper's (q = 3, z = 40, τ = 3, gate 1.5, 70% band); `with_*`
builders change them, and `with_batch_size` scores a long series in
consecutive batches against their own context. At least 12 points.

### Seasonality — which period does the series repeat on?

Holt-Winters needs the seasonal period as an input; `estimate_period` finds it
from the data. Two stages (Vlachos, Yu & Castelli 2005, *AutoPeriod*): the
linearly detrended, zero-padded periodogram is searched for peaks above what
the same values in random order produce (99th percentile over 100 seeded
permutations — deterministic), then each peak's frequency band is refined on the
autocorrelation function to the integer lag that is a local maximum above the
`1.96/√n` white-noise bound. The answer says explicitly when there is no period.

```rust
use u_analytics::seasonality::estimate_period;

let sawtooth: Vec<f64> = (0..40).map(|i| (i % 7) as f64).collect();
let r = estimate_period(&sawtooth).unwrap();
assert_eq!(r.period, Some(7));            // every validated candidate is in r.candidates

let line: Vec<f64> = (0..40).map(|i| 2.0 * i as f64).collect();
assert_eq!(estimate_period(&line).unwrap().period, None);
```

Only periods from 2 to `n/2` are admissible — a cycle has to be seen twice —
and a series shorter than 8 points is refused (`None` from the function, as
opposed to `period: None` inside a result). Series under ~16 points rarely beat
the permutation threshold: too few orderings differ from the observed one.

## JavaScript / WASM (npm)

```bash
npm install @iyulab/u-analytics
```

The npm package ships two entry points, selected automatically via the
`exports` map:

| Environment | Entry | Notes |
|---|---|---|
| Bundlers (webpack, Vite, …) | ESM + WebAssembly ESM-integration | `default` condition |
| Node.js (`require` **and** `import`) | CJS glue that loads the wasm from the filesystem | `node` condition — works in CJS TS runners (`tsx`, `ts-node`) without loader hooks |

```js
// Both work in Node — no bundler, no experimental flags:
const { anderson_darling_normality } = require("@iyulab/u-analytics");
// or: import { anderson_darling_normality } from "@iyulab/u-analytics";

anderson_darling_normality(new Float64Array([4.9, 5.1, 5.0, 5.2, 4.8, 5.05]));
// → { statistic, statistic_modified, p_value }
```

**Errors.** A refused input throws an `Error` with two extra properties:

```ts
interface AnalyticsError extends Error {
  code: string          // stable reason — branch on this, not on `message`
  index: number | null  // position of the offending element in its input array
}
// try { c_chart([1, 2, 2.5]) } catch (e) { e.code === "count_not_whole"; e.index === 2 }
// On a [defectives, sample_size] row the code says which member failed:
//   p_chart([[1, 10], [1.5, 10]]) -> count_not_whole,      index 1
//   p_chart([[1, 10], [1, 10.5]]) -> sample_size_not_whole, index 1
```

| `code` | Meaning |
|---|---|
| `count_not_whole` | a defect or defective count is not a whole number ≥ 0 |
| `sample_size_not_whole` | a sample size is not a whole number ≥ 1 — zero, negative, fractional and not-a-number alike |
| `defectives_exceed_sample` | more defectives than the sample has items |
| `units_not_positive` | units inspected that are not a positive number |
| `subgroup_length_mismatch` | a subgroup of a different length than the first |
| `subgroup_size_out_of_range` | a subgroup size the factor tables do not cover |
| `too_few_samples` | fewer samples than the chart needs (`index: null`) |
| `standard_out_of_range` | a Phase I `p_bar`/`u_bar`/`phi` outside its domain |
| `malformed_input` | not the shape the function takes — a row that is not a pair, an unknown field |
| `invalid_input` | any other refusal; the message says what |

`message` is written for people and may change between releases; `code` does not.

### `process_capability(input)`

```ts
process_capability({
  data: number[],
  usl?: number,          // at least one of usl / lsl is required
  lsl?: number,
  sigma_within?: number, // short-term sigma, e.g. R-bar / d2
  target?: number,       // Cpm target; without it `cpm` is null
}): {
  mean: number,
  sigma_source: "within" | "overall",
  std_dev_within: number | null,
  std_dev_overall: number,
  cp: number | null, cpk: number | null, cpu: number | null, cpl: number | null,
  pp: number | null, ppk: number | null, ppu: number | null, ppl: number | null,
  cpm: number | null,
}
```

**Short-term sigma has to be supplied.** Cp/Cpk are defined against the
within-subgroup standard deviation, which is estimated from a control chart
(R-bar/d2 or S-bar/c4) and is *not* recoverable from a flat measurement vector —
the subgroup structure is gone. Omit `sigma_within` and `sigma_source` comes
back as `"overall"` with `cp`, `cpk`, `cpu`, `cpl` and `std_dev_within` all
`null`: only the long-term indices (Pp/Ppk) are reported. Filling the
short-term names with the long-term sigma instead would make `cp` equal `pp`
for every input.

**Cpm needs a `target`.** It measures how closely the data cluster about the
target -- `min(T - LSL, USL - T) / (3 * sqrt(sum((x - T)^2) / (n - 1)))` -- so it
uses neither sigma and is reported with or without `sigma_within`. Without a
`target` it is `null`; pass the specification midpoint if that is the target.

`xbar_r_chart` returns `sigma_hat` (`R-bar / d2`) for exactly this purpose, so
the two compose:

```js
const chart = xbar_r_chart(subgroups);
const cap = process_capability({
  data: subgroups.flat(), usl: 11, lsl: 9, sigma_within: chart.sigma_hat,
});
// cap.sigma_source === "within", cap.cp !== cap.pp
```

One-sided specifications are supported: pass only `usl` or only `lsl`. The
indices that need both limits (`cp`, `pp`, `cpm`) come back `null`.

### Choosing which run tests apply

All eight Nelson tests run by default. A process that is known to trip one of
them for a benign reason -- a deliberately drifting tool, a bimodal fixture --
otherwise has to have that signal filtered out downstream, after it has already
been counted as out of control. `xbar_r_chart` takes an optional second
argument naming the tests to apply:

```js
// Western Electric only
xbar_r_chart(subgroups, { rules: [
  "BeyondLimits", "NineOneSide", "TwoOfThreeBeyond2Sigma", "FourOfFiveBeyond1Sigma",
]});

// Control limits and nothing else
xbar_r_chart(subgroups, { rules: [] });
```

The names are the same values each point's `violations` reports, so the set is
written in the vocabulary the output already uses: `BeyondLimits`,
`NineOneSide`, `SixTrend`, `FourteenAlternating`, `TwoOfThreeBeyond2Sigma`,
`FourOfFiveBeyond1Sigma`, `FifteenWithin1Sigma`, `EightBeyond1Sigma`.

Omitting the argument -- or passing `undefined`, `null`, or an object without
`rules` -- applies all eight, so existing calls are unaffected. In Rust the same
choice is `RuleSet`, passed to a chart with `with_rules`.

### X-bar-S and Individual-MR charts

```ts
type RuleName = "BeyondLimits" | "NineOneSide" | "SixTrend" | "FourteenAlternating"
              | "TwoOfThreeBeyond2Sigma" | "FourOfFiveBeyond1Sigma"
              | "FifteenWithin1Sigma" | "EightBeyond1Sigma";
type Point = { index: number, value: number, violations: RuleName[] };

xbar_s_chart(subgroups: number[][], options?: { rules?: RuleName[] }): {
  xbar_cl: number, xbar_ucl: number, xbar_lcl: number,
  s_cl: number, s_ucl: number, s_lcl: number,
  sigma_hat: number | null,      // S-bar / c4
  xbar_points: Point[], s_points: Point[], in_control: boolean,
}

imr_chart(values: number[], options?: { rules?: RuleName[] }): {
  i_cl: number, i_ucl: number, i_lcl: number,
  mr_cl: number, mr_ucl: number, mr_lcl: number,
  sigma_hat: number | null,      // MR-bar / d2(2)
  i_points: Point[],
  mr_points: Point[],            // starts at index 1: the first value has no moving range
  in_control: boolean,
}
```

`xbar_s_chart` takes the same subgroup matrix as `xbar_r_chart`; above about ten
values per subgroup the standard deviation is the better estimate of spread,
because the range uses only the two extremes. Both return `sigma_hat` for
`process_capability`'s `sigma_within`, like `xbar_r_chart`. Subgroups must all
have the same size; a ragged one is reported by its row rather than skipped,
since skipping it would shift the index of every point after it.

### Applying run tests to your own series

```ts
run_rules(
  values: number[],
  limits: { ucl: number, cl: number, lcl: number },   // lcl <= cl <= ucl
  options?: { rules?: RuleName[] },
): Point[]   // one entry per value, in input order
```

The same engine the charts use, on its own -- for a statistic the crate does not
chart, or for limits fixed from an earlier study:

```js
run_rules([10.1, 10.4, 9.8, 12.9], { ucl: 12, cl: 10, lcl: 8 });
// → [{ index: 0, value: 10.1, violations: [] }, ...,
//    { index: 3, value: 12.9, violations: ["BeyondLimits"] }]
```

Given a chart's own points and limits, it finds exactly the violations the chart
reported.

### Attributes charts

```ts
type AttrPoint = { index: number, value: number, ucl: number, cl: number, lcl: number,
                   out_of_control: boolean,
                   z: number | null };   // (value − cl) / σᵢ — null when σᵢ is 0

p_chart(samples: [defectives: number, size: number][], options?: { p_bar?: number }):
  { p_bar: number, points: AttrPoint[], in_control: boolean }
laney_p_chart(samples: [defectives: number, size: number][],
              options?: { p_bar?: number, phi?: number }):   // both or neither
  { p_bar: number, phi: number, points: AttrPoint[] }
np_chart(defectives: number[], sample_size: number):
  { cl: number, ucl: number, lcl: number, points: AttrPoint[], in_control: boolean }
c_chart(defects: number[]):
  { cl: number, ucl: number, lcl: number, points: AttrPoint[], in_control: boolean }
u_chart(samples: [defects: number, units: number][], options?: { u_bar?: number }):
  { u_bar: number, points: AttrPoint[], in_control: boolean }
laney_u_chart(samples: [defects: number, units: number][],
              options?: { u_bar?: number, phi?: number }):   // both or neither
  { u_bar: number, phi: number, points: AttrPoint[] }
```

**Phase I and Phase II.** Without options the centre (and φ) are estimated from
`samples` — a Phase I study. To judge later samples against it, pass that
study's `p_bar`/`u_bar` (and `phi` for the Laney charts): every limit then uses
the Phase I values with each sample's own size, and a single sample is enough.
Charting Phase I and Phase II together instead would let a Phase II shift pull
the centre toward itself and widen its own limits. A `p_bar` outside (0, 1), a
`u_bar` ≤ 0 or a negative `phi` is refused as `standard_out_of_range`; a Laney
standard with only one of its two parts, or a key the chart does not take, is
refused rather than completed or ignored.

```js
const phaseOne = laney_p_chart(firstTwelve)              // { p_bar, phi, points }
laney_p_chart(later, { p_bar: phaseOne.p_bar, phi: phaseOne.phi })
```

**Run rules when the sample size varies.** P, U and the Laney charts have a
different pair of limits at every point, so zones on the original scale do not
exist. Each point carries `z`, its distance from the centre line in its own
standard errors; on that scale every limit is ±3 (the standardized control
chart, Montgomery §7.2.2):

```js
const chart = p_chart(samples)
run_rules(chart.points.map(p => p.z), { ucl: 3, cl: 0, lcl: -3 })
```

These charts judge each point against its limits only; for an NP or C chart,
whose limits are constant, pass `value`s and the chart's limits to `run_rules`
directly.

A row the chart cannot use -- a count that is not a whole number, more
defectives than the sample size, a sample size of zero, `units` that are not
positive -- is rejected with its `code` and `index` (see **Errors** above). The
charts would otherwise drop it, and every later point would carry the index of
the wrong row.

### CUSUM and EWMA — sequential shift detection

The Shewhart charts above judge each point on its own, which makes them slow to
notice a **small but persistent** shift. These two accumulate evidence across
points instead, and are the standard answer for that case.

```ts
cusum(input: {
  data: number[],          // observations, in order; must be non-empty
  target: number,          // process target mean (mu_0)
  sigma: number,           // known process sigma; must be > 0
  k?: number,              // reference value / allowance, default 0.5
  h?: number,              // decision interval, default 5.0
}): {
  h: number,               // echoed, so the boundary can be drawn without restating the input
  points: { index: number, s_upper: number, s_lower: number, signal: boolean }[],
  signal_indices: number[],
  in_control: boolean,
}

ewma(input: {
  data: number[],
  target: number,
  sigma: number,
  lambda?: number,         // smoothing constant in (0, 1], default 0.2
  l_factor?: number,       // limit width factor, default 3.0
}): {
  points: { index: number, ewma: number, ucl: number, lcl: number, signal: boolean }[],
  signal_indices: number[],
  in_control: boolean,
}
```

```js
// A 2-sigma shift that begins at observation 10
const data = [...Array(10).fill(10), ...Array(10).fill(12)];
cusum({ data, target: 10, sigma: 1 }).signal_indices;  // → indices inside the shifted half
```

**CUSUM's limit is a single number, EWMA's is per point.** Both cumulative sums
are on the standardized scale (`z = (x - target) / sigma`) and start at zero, so
they compare against `h` directly — hence `h` on the chart rather than on each
point. EWMA's limits are the exact (not asymptotic) ones, so they **widen with
the index** and are returned per point.

The defaults are the values their sources recommend: `k = 0.5` is optimal for a
1-sigma shift and `h = 5` gives ARL_0 ≈ 465 (Page 1954); `lambda = 0.2` with
`L = 3` is sensitive across 0.5–2.0 sigma shifts (Roberts 1959). Both reject an
empty `data` and any parameter outside its domain, naming the offending one.

### Non-normal capability, and sigma level ↔ PPM

```ts
boxcox_capability(input: {
  data: number[],          // >= 4 observations, all strictly positive
  usl?: number,            // optional; each must be positive
  lsl?: number,            // with neither limit, only lambda is estimated (indices null)
  lambda_range?: [number, number],  // search range, default [-5, 5] (Minitab's)
}): {
  lambda: number,          // ML-estimated optimal Box-Cox parameter within lambda_range
  lambda_at_bound: boolean,  // true: likelihood still rising at an end of the range —
                             // lambda is that limit, not an interior optimum
  pp: number | null, ppk: number | null, ppu: number | null, ppl: number | null,
  cp: null, cpk: null, cpu: null, cpl: null,   // always null — see below
  cpm: number | null,
}

sigma_to_ppm(sigma: number): number   // 6 → ~3.4,  3 → ~66807
ppm_to_sigma(ppm: number): number     // inverse; ppm must be inside (0, 1e6)
```

**Only the long-term indices are reported.** `data` is a flat vector, so there
is no rational subgrouping and no within-subgroup sigma to estimate. Computing
`cp`/`cpk` from the overall sigma instead would make `cp` equal `pp` for every
input — a long-term number wearing a short-term name — so they come back `null`,
the same choice `process_capability` makes when `sigma_within` is omitted.

Every index is on the **transformed** scale, which is where the normal-theory
formulas hold; they are not comparable to indices computed on the raw
non-normal data. The specification limits are transformed with the same lambda.

**`sigma_to_ppm` uses the Motorola convention, including the 1.5-sigma shift**
(`PPM = 10^6 · (1 − Φ(σ − 1.5))`). Six sigma therefore reports ~3.4 PPM rather
than the ~0.002 PPM of an unshifted normal tail — if you are checking against a
table, check which convention it uses. `ppm_to_sigma` is the exact inverse on
the same convention and rejects both ends of `(0, 1e6)`, which the sigma scale
does not reach. The round trip closes to ~3·10⁻⁴ (the inverse normal CDF is a
rational approximation).

### The remaining exports, briefly

Every export takes plain JSON values and returns one; a rejected input throws
an `Error` carrying `code` and `index` (see **Errors**). The shapes below are the ones
the sections above have not already spelled out.

```ts
// Rare-event charts
g_chart(gaps: number[]):  { g_bar: number, points: AttrPoint[] }    // events between occurrences
t_chart(times: number[]): { t_bar: number, points: AttrPoint[] }    // time between occurrences

// Normality
anderson_darling_normality(data: number[]):
  { statistic: number, statistic_modified: number, p_value: number }

// Changepoints — PELT (Killick et al., 2012)
detect_changepoints(input: {
  data: number[],
  cost?: "l2" | "normal",        // default "l2": mean shift; "normal": mean + variance
  penalty?: "bic" | number,      // default "bic"
  min_segment_len?: number,      // default 2, must be >= 2
}): { changepoints: number[], n_segments: number }
detect_changepoints_multi(input: { signals: number[][], cost?, penalty?, min_segment_len? }):
  { changepoints: number[], n_segments: number }   // channels of equal length

// Seasonality — AutoPeriod (Vlachos et al., 2005); >= 8 finite values
estimate_period(input: { data: number[] }): {
  period: number | null,          // null = no periodicity passed both stages (explicit, not an error)
  candidates: { period: number, acf: number, bin: number, power: number, power_share: number }[],
  n: number, acf_threshold: number, power_threshold: number,
}

// Spectral residual anomaly scoring — Ren et al. (2019); >= 12 finite values
spectral_residual(input: {
  data: number[],
  averaging_window?: number,      // default 3  (q)
  judgement_window?: number,      // default 40 (z)
  threshold?: number,             // default 3  (τ, on the relative saliency score)
  min_zscore?: number,            // default 1.5; 0 disables the gate
  sensitivity?: number,           // default 70: coverage % of the expected-value band
  batch_size?: number,            // score in consecutive batches (>= 12)
}): {
  points: { index, value, saliency, score, expected, lower, upper: number, is_anomaly: boolean }[],
  anomalies: number[],            // indices with is_anomaly
}

// Gage R&R — measurements[part][operator][trial]
gage_rr_xbar_r(input: { measurements: number[][][], tolerance?: number }): {
  // The two charts the method plots, from the same R-bar the components use.
  // Subgroup = one operator x part cell, so n is the trial count (2 or 3).
  range_chart:   { center: number, ucl: number, lcl: number },  // R-bar, D4*R-bar, D3*R-bar
  average_chart: { center: number, ucl: number, lcl: number },  // X-double-bar +/- A2*R-bar
  ev, av, grr, pv, tv, percent_ev, percent_av, percent_grr, percent_pv: number,
  percent_tolerance: number | null, ndc: number, status: "Acceptable" | "Marginal" | "Unacceptable",
}
gage_rr_anova(input: { measurements: number[][][], tolerance?: number }): {
  anova_table: { source, df, ss, ms, f_value: number | null, p_value: number | null }[],
  variance_components: { part, operator, interaction, repeatability, reproducibility, total },
  ev, av, grr, pv, tv, percent_grr: number, percent_tolerance: number | null,
  ndc: number, status: string, interaction_significant: boolean, interaction_pooled: boolean,
}

// Non-parametric (percentile) capability — >= 20 observations, at least one limit
percentile_capability(input: { data: number[], usl?: number, lsl?: number }): {
  cp_star, cpk_star, cpu_star, cpl_star: number | null,
  median: number, percentile_lower: number, percentile_upper: number,
}
```

Points on the average chart are *expected* to fall outside its limits -- that is
how the study reads whether the parts vary enough to be told apart. The factors
come from `spc::range_chart_factors(n)` (Rust), which refuses a subgroup size the
tables do not cover rather than approximating it; `gage_rr_xbar_r` itself accepts
only 2 or 3 trials, so that refusal is unreachable through it.

## C FFI (NuGet `UAnalytics`)

The `ffi` feature builds a `cdylib` with a C ABI for hosts that cannot load
WebAssembly — .NET through the `UAnalytics` NuGet package, or anything that can
call a C function.

```bash
cargo build --release --features ffi
```

**Every entry point is JSON-in / JSON-out with one calling convention:**

```c
int32_t uanalytics_<name>(const char *request_json, char **result_ptr);
void    uanalytics_free_string(char *ptr);   // release any string the library returned
char   *uanalytics_version(void);            // crate version; free with uanalytics_free_string
```

| Status | Meaning | `*result_ptr` |
|---|---|---|
| `0` | success | the response JSON |
| `-1` | `request_json` or `result_ptr` was null | null |
| `-2` | the request did not parse into the expected shape | `{"error": "...", "code": "malformed_input", "index": null}` |
| `-3` | the request parsed, and the computation rejected it | `{"error": "...", "code": "...", "index": ...}` |
| `-4` | internal panic (caught; never unwinds across the boundary) | `{"error": "...", "code": "invalid_input", "index": null}` |

`code` and `index` are the same values the JavaScript `Error` carries (see
**Errors** in the JavaScript section).

Unknown request fields are rejected (`-2`, naming the field) rather than
ignored, so a misspelt option cannot silently change which analysis you get.
The string written to `*result_ptr` is owned by the caller and must be released
with `uanalytics_free_string`.

**The FFI and the WASM binding share one contract.** Every entry point that
both transports carry parses the same request type and serialises the same
response type (`src/wire.rs`), and a test pins that the FFI body is the
`serde_json` rendering of the value the WASM binding emits. So the schemas in
the JavaScript section above *are* the FFI schemas: the request is the WASM
function's arguments as one JSON object, the response is the same JSON.

| C entry point | WASM export | Request JSON |
|---|---|---|
| `uanalytics_xbar_r_chart` | `xbar_r_chart(subgroups, { rules? })` | `{ subgroups, rules? }` |
| `uanalytics_xbar_s_chart` | `xbar_s_chart(subgroups, { rules? })` | `{ subgroups, rules? }` |
| `uanalytics_imr_chart` | `imr_chart(values, { rules? })` | `{ values, rules? }` |
| `uanalytics_run_rules` | `run_rules(values, limits, { rules? })` | `{ values, limits: { ucl, cl, lcl }, rules? }` |
| `uanalytics_p_chart` | `p_chart(samples, { p_bar? })` | `{ samples: [[defectives, sample_size], …], p_bar? }` |
| `uanalytics_laney_p_chart` | `laney_p_chart(samples, { p_bar?, phi? })` | `{ samples: [[defectives, sample_size], …], p_bar?, phi? }` |
| `uanalytics_process_capability` | `process_capability(input)` | `input` as is |
| `uanalytics_percentile_capability` | `percentile_capability(input)` | `input` as is |
| `uanalytics_gage_rr_xbar_r` | `gage_rr_xbar_r(input)` | `input` as is |
| `uanalytics_gage_rr_anova` | `gage_rr_anova(input)` | `input` as is |
| `uanalytics_detect_changepoints` | `detect_changepoints(input)` | `input` as is |
| `uanalytics_estimate_period` | `estimate_period(input)` | `input` as is |
| `uanalytics_spectral_residual` | `spectral_residual(input)` | `input` as is |

The composition documented for JavaScript holds here too: `imr_chart` (or
`xbar_s_chart`) returns `sigma_hat`, which is what `process_capability` needs as
`sigma_within` to report the short-term indices — the FFI does not guess a
within sigma from a flat vector any more than the WASM binding does.

Four entry points exist only on the FFI:

```ts
uanalytics_weibull_mle        { failure_times: number[] }  → { shape: number, scale: number }
uanalytics_correlation_matrix { variables: number[][] }    → { rows: number, cols: number, data: number[] } // row-major
uanalytics_simple_regression  { x: number[], y: number[] } → { slope, intercept, r_squared, adjusted_r_squared,
                                                                slope_se, intercept_se: number }
uanalytics_fit_best           { data: number[] }           → { distribution: string, parameters: [name: string, value: number][],
                                                                log_likelihood, aic, bic: number }[] // ascending AIC
```

The .NET client (`bindings/csharp/UAnalytics`, package `UAnalytics`) wraps each
entry point as a method on `AnalyticsClient` — `XbarRChart`, `ImrChart`, `PChart(samples, pBar)`, `LaneyPChart(samples, pBar, phi)`,
`ProcessCapability`, `DetectChangepoints`, … — serialising the arguments to the
request above and returning the response as a `JsonElement`. A non-zero status
surfaces as `AnalyticsException` carrying the status (`Code`), the `error`
message, and the body's `code` and `index` as `Reason` and `Index`. The
package follows its own version line (it is a binding, not the crate), noted
in the CHANGELOG entry that changes it.

## Test Status

```text
560 lib tests (576 with `ffi`, 604 with `wasm`) + 88 doc-tests
0 clippy warnings
```

## Dependencies

- [`u-numflow`](https://crates.io/crates/u-numflow) -- statistics, special functions, probability distributions

## References

- Montgomery, D.C. (2019). *Introduction to Statistical Quality Control*, 8th ed.
- Nelson, L.S. (1984). "The Shewhart Control Chart -- Tests for Special Causes"
- Abernethy, R.B. (2006). *The New Weibull Handbook*, 5th ed.
- Page, E.S. (1954). "Continuous Inspection Schemes", *Biometrika*
- Roberts, S.W. (1959). "Control Chart Tests Based on Geometric Moving Averages"
- Laney, D.B. (2002). "Improved Control Charts for Attributes", *Quality Engineering* 14(4), 531–537
- Stephens, M.A. (1974). "EDF Statistics for Goodness of Fit", *JASA* 69(347), 730–737
- Box, G.E.P. & Cox, D.R. (1964). "An Analysis of Transformations", *JRSS-B* 26(2), 211–252

## Related

- [u-numflow](https://crates.io/crates/u-numflow) -- Mathematical primitives
- [u-insight](https://github.com/iyulab/u-insight) -- Statistical analysis engine with C FFI
- [u-metaheur](https://crates.io/crates/u-metaheur) -- Metaheuristic algorithms
- [u-geometry](https://crates.io/crates/u-geometry) -- Computational geometry
- [u-schedule](https://crates.io/crates/u-schedule) -- Scheduling framework

## License

MIT
