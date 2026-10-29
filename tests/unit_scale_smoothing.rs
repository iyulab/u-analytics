//! The smoothing constants a fit chooses do not depend on the unit of the
//! data: the sum of squared errors the search minimises scales with the square
//! of the unit, and the search must stop where it would have stopped in any
//! other unit.

use u_analytics::smoothing::{HoltLinear, SimpleExponentialSmoothing};

const SCALES: [f64; 6] = [1e-12, 1e-9, 1e-6, 1e6, 1e9, 1e12];

/// A noisy level that drifts: deterministic, so every run sees the same data.
fn level_series() -> Vec<f64> {
    let mut state = 20261029u64;
    (0..80)
        .map(|i| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = (state >> 11) as f64 / (1u64 << 53) as f64;
            30.0 + 0.04 * i as f64 + 1.6 * (u - 0.5) * 2.0
        })
        .collect()
}

fn scaled(v: &[f64], k: f64) -> Vec<f64> {
    v.iter().map(|x| x * k).collect()
}

#[test]
fn the_simple_smoothing_constant_does_not_depend_on_the_unit() {
    let data = level_series();
    let alpha = SimpleExponentialSmoothing::fit(&data)
        .expect("a fit")
        .alpha();
    for k in SCALES {
        let got = SimpleExponentialSmoothing::fit(&scaled(&data, k))
            .expect("a fit")
            .alpha();
        assert!(
            (got - alpha).abs() <= 1e-6,
            "alpha at scale {k:e}: {got} vs {alpha}"
        );
    }
}

#[test]
fn the_holt_constants_do_not_depend_on_the_unit() {
    let data = level_series();
    let base = HoltLinear::fit(&data, None, None).expect("a fit");
    for k in SCALES {
        let fit = HoltLinear::fit(&scaled(&data, k), None, None).expect("a fit");
        assert!(
            (fit.alpha() - base.alpha()).abs() <= 1e-6 && (fit.beta() - base.beta()).abs() <= 1e-6,
            "(alpha, beta) at scale {k:e}: ({}, {}) vs ({}, {})",
            fit.alpha(),
            fit.beta(),
            base.alpha(),
            base.beta()
        );
    }
}
