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
| `detection` | Change-point detection (CUSUM, EWMA) |
| `smoothing` | Time series smoothing (SES, Holt linear trend, Holt-Winters seasonal) |
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
chart.add_sample(&[25.0, 26.0, 24.5, 25.5, 25.0]);
chart.add_sample(&[25.2, 24.8, 25.1, 24.9, 25.3]);
chart.add_sample(&[25.1, 25.0, 24.7, 25.3, 24.9]);

if chart.is_in_control() {
    println!("Process is stable");
}
```

```rust
use u_analytics::spc::{laney_p_chart, g_chart};

// Laney P' chart for overdispersed proportion data
// samples: (defective count, subgroup size)
let samples = vec![(3u64, 100u64), (5, 120), (2, 95)];
let chart = laney_p_chart(&samples).unwrap();
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
```

```rust
use u_analytics::capability::boxcox_capability;

// Non-normal data: auto-estimate λ, transform spec limits, compute Ppk
let skewed_data = vec![0.5, 1.2, 0.8, 2.1, 0.3, 1.7, 0.9, 1.4];
let result = boxcox_capability(&skewed_data, Some(5.0), Some(0.1)).unwrap();
println!("λ = {:.3}, Ppk = {:.3}", result.lambda, result.indices.ppk.unwrap());
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

### `process_capability(input)`

```ts
process_capability({
  data: number[],
  usl?: number,          // at least one of usl / lsl is required
  lsl?: number,
  sigma_within?: number, // short-term sigma, e.g. R-bar / d2
  target?: number,       // Cpm target; defaults to the specification midpoint
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
back as `"overall"` with `cp`, `cpk`, `cpu`, `cpl`, `cpm` and `std_dev_within`
all `null`: only the long-term indices (Pp/Ppk) are reported. Filling the
short-term names with the long-term sigma instead would make `cp` equal `pp`
for every input.

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

## Test Status

```text
556 lib tests (567 with `ffi`, 572 with `wasm`) + 88 doc-tests
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
