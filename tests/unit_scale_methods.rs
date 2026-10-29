//! Charts, change detection, capability, trend tests and regressions give the
//! same answer for the same data in another unit: what is dimensionless in
//! the answer (flags, counts, p-values, indices, shapes) does not move when
//! the data, their limits and their time axis are rescaled together.

use u_analytics::capability::ProcessCapability;
use u_analytics::correlation::{kendall_tau_b, pearson, spearman};
use u_analytics::detection::{CostFunction, Cusum, Ewma, Pelt, Penalty};
use u_analytics::point_process::{
    laplace_trend_test, mil_hdbk_189_test, power_law_process_fit, Truncation,
};
use u_analytics::regression::simple_linear_regression;

const SCALES: [f64; 6] = [1e-12, 1e-9, 1e-6, 1e6, 1e9, 1e12];

/// A level shift in noise: 40 points around 10, then 40 around 12.
fn shifted_series() -> Vec<f64> {
    let mut state = 11u64;
    (0..80)
        .map(|i| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = (state >> 11) as f64 / (1u64 << 53) as f64;
            (if i < 40 { 10.0 } else { 12.0 }) + (u - 0.5) * 1.2
        })
        .collect()
}

fn scaled(v: &[f64], k: f64) -> Vec<f64> {
    v.iter().map(|x| x * k).collect()
}

fn close(got: f64, want: f64, what: &str, k: f64) {
    assert!(
        (got - want).abs() <= 1e-7 * want.abs().max(1e-3),
        "{what} at scale {k:e}: {got} vs {want}"
    );
}

#[test]
fn charts_and_change_detection_find_the_same_points_in_any_unit() {
    let data = shifted_series();
    let cusum_base = Cusum::new(10.0, 0.35).expect("cusum").signal_points(&data);
    let ewma_base = Ewma::new(10.0, 0.35).expect("ewma").signal_points(&data);
    let pelt = |d: &[f64]| {
        Pelt::new(CostFunction::L2, Penalty::Bic)
            .expect("pelt")
            .detect(d)
            .changepoints
    };
    let pelt_base = pelt(&data);
    assert!(!cusum_base.is_empty() && !ewma_base.is_empty() && !pelt_base.is_empty());
    for k in SCALES {
        let d = scaled(&data, k);
        assert_eq!(
            Cusum::new(10.0 * k, 0.35 * k)
                .expect("cusum")
                .signal_points(&d),
            cusum_base,
            "CUSUM at scale {k:e}"
        );
        assert_eq!(
            Ewma::new(10.0 * k, 0.35 * k)
                .expect("ewma")
                .signal_points(&d),
            ewma_base,
            "EWMA at scale {k:e}"
        );
        assert_eq!(pelt(&d), pelt_base, "PELT at scale {k:e}");
    }
}

#[test]
fn capability_indices_are_the_same_in_any_unit() {
    let data = shifted_series();
    let base = ProcessCapability::new(Some(14.0), Some(8.0))
        .expect("spec")
        .compute(&data, 0.5)
        .expect("indices");
    for k in SCALES {
        let got = ProcessCapability::new(Some(14.0 * k), Some(8.0 * k))
            .expect("spec")
            .compute(&scaled(&data, k), 0.5 * k)
            .expect("indices");
        close(got.cp.expect("cp"), base.cp.expect("cp"), "Cp", k);
        close(got.cpk.expect("cpk"), base.cpk.expect("cpk"), "Cpk", k);
        close(got.ppk.expect("ppk"), base.ppk.expect("ppk"), "Ppk", k);
    }
}

#[test]
fn trend_tests_of_event_times_are_the_same_in_any_unit_of_time() {
    // Failure times of a repairable system, slowly getting closer together.
    let times: Vec<f64> = (1..=30).map(|i| (i as f64).powf(1.35) * 3.0).collect();
    let end = times[times.len() - 1] * 1.05;
    let laplace = laplace_trend_test(&times, Truncation::Time(end)).expect("laplace");
    let mil = mil_hdbk_189_test(&times, Truncation::Time(end)).expect("mil");
    let fit = power_law_process_fit(&times, Truncation::Time(end)).expect("fit");
    for k in SCALES {
        let t = scaled(&times, k);
        let l = laplace_trend_test(&t, Truncation::Time(end * k)).expect("laplace");
        close(l.statistic, laplace.statistic, "Laplace U", k);
        close(l.p_value, laplace.p_value, "Laplace p", k);
        let m = mil_hdbk_189_test(&t, Truncation::Time(end * k)).expect("mil");
        close(m.p_value, mil.p_value, "MIL-HDBK-189 p", k);
        let f = power_law_process_fit(&t, Truncation::Time(end * k)).expect("fit");
        close(f.beta, fit.beta, "power-law shape", k);
    }
}

#[test]
fn regression_and_correlation_are_the_same_in_any_unit() {
    let x: Vec<f64> = (0..40).map(|i| i as f64 * 0.5).collect();
    let y: Vec<f64> = x
        .iter()
        .enumerate()
        .map(|(i, v)| 3.0 + 0.8 * v + 1.5 * ((i * 7 % 11) as f64 - 5.0) / 5.0)
        .collect();
    let reg = simple_linear_regression(&x, &y).expect("regression");
    let (p, s, kt) = (
        pearson(&x, &y).expect("pearson"),
        spearman(&x, &y).expect("spearman"),
        kendall_tau_b(&x, &y).expect("kendall"),
    );
    for kx in SCALES {
        for ky in [1e-9, 1e9] {
            let (sx, sy) = (scaled(&x, kx), scaled(&y, ky));
            let got = simple_linear_regression(&sx, &sy).expect("regression");
            close(got.r_squared, reg.r_squared, "R2", kx);
            close(got.slope_p, reg.slope_p, "slope p", kx);
            close(got.slope * kx / ky, reg.slope, "slope", kx);
            close(
                pearson(&sx, &sy).expect("pearson").p_value,
                p.p_value,
                "Pearson p",
                kx,
            );
            close(
                spearman(&sx, &sy).expect("spearman").p_value,
                s.p_value,
                "Spearman p",
                kx,
            );
            close(
                kendall_tau_b(&sx, &sy).expect("kendall").p_value,
                kt.p_value,
                "Kendall p",
                kx,
            );
        }
    }
}
