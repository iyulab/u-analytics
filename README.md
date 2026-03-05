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
| `spc` | Control charts (X̄-R, X̄-S, I-MR, P, NP, C, U, Laney P'/U', G, T) with Nelson/WE run rules |
| `capability` | Process capability indices (Cp, Cpk, Pp, Ppk, Cpm), sigma level, and Box-Cox non-normal capability |
| `weibull` | Weibull parameter estimation (MLE, MRR) and reliability analysis (R(t), MTBF, B-life) |
| `detection` | Change-point detection (CUSUM, EWMA) |
| `smoothing` | Time series smoothing (SES, Holt linear trend, Holt-Winters seasonal) |
| `correlation` | Correlation analysis (Pearson, Spearman, Kendall, partial, correlation matrices) |
| `regression` | Regression analysis (simple OLS, multiple OLS, VIF multicollinearity) |
| `distribution` | Distribution analysis (ECDF, histogram bins — Sturges/Scott/FD, QQ-plot, KS test) |
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

let mut chart = XBarRChart::new(5);
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

## Test Status

```
446 lib tests + 68 doc-tests = 514 total
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
