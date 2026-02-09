# u-analytics

Statistical process control, process capability analysis, Weibull reliability,
and change-point detection for industrial quality engineering.

## Features

### Statistical Process Control (SPC)

Control charts for monitoring process stability:

- **Variables charts**: X̄-R, X̄-S, Individual-MR
- **Attributes charts**: P, NP, C, U
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

### Process Capability

Capability indices quantifying process performance against specifications:

- **Short-term**: Cp, Cpk, Cpu, Cpl
- **Long-term**: Pp, Ppk, Ppu, Ppl
- **Taguchi**: Cpm
- **Sigma level**: PPM ↔ sigma conversion (1.5σ shift convention)

```rust
use u_analytics::capability::{ProcessCapability, sigma_to_ppm};

let spec = ProcessCapability::new(Some(220.0), Some(200.0)).unwrap();
let data = [210.0, 209.5, 210.2, 209.8, 210.1, 210.3, 209.7, 210.0];
let indices = spec.compute(&data, 0.15).unwrap();

println!("Cp = {:.2}, Cpk = {:.2}", indices.cp.unwrap(), indices.cpk.unwrap());
println!("6σ PPM = {:.1}", sigma_to_ppm(6.0)); // 3.4
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

## Dependencies

- [`u-optim`](https://crates.io/crates/u-optim) -- statistics, special functions, probability distributions

## References

- Montgomery, D.C. (2019). *Introduction to Statistical Quality Control*, 8th ed.
- Nelson, L.S. (1984). "The Shewhart Control Chart -- Tests for Special Causes"
- Abernethy, R.B. (2006). *The New Weibull Handbook*, 5th ed.
- Page, E.S. (1954). "Continuous Inspection Schemes", *Biometrika*
- Roberts, S.W. (1959). "Control Chart Tests Based on Geometric Moving Averages"

## Related

- [u-optim](https://crates.io/crates/u-optim) -- Mathematical optimization primitives
- [u-metaheur](https://crates.io/crates/u-metaheur) -- Metaheuristic algorithms
- [u-geometry](https://crates.io/crates/u-geometry) -- Computational geometry
- [u-schedule](https://crates.io/crates/u-schedule) -- Scheduling framework

## License

MIT
