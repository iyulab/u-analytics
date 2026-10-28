//! Accuracy of a model's predictions against observations.
//!
//! The errors are `eᵢ = observedᵢ − predictedᵢ` (positive when the model
//! predicts too little), from a regression's fitted values, a forecast, or any
//! other model.
//!
//! # Examples
//!
//! ```
//! use u_analytics::accuracy::prediction_errors;
//!
//! let observed = [10.0, 12.0, 11.0, 13.0];
//! let predicted = [10.5, 11.5, 11.0, 12.0];
//! let e = prediction_errors(&observed, &predicted, 0).unwrap();
//! assert!((e.mean_error - 0.25).abs() < 1e-12);
//! assert!((e.mean_absolute_error - 0.5).abs() < 1e-12);
//! ```

/// Error measures of predictions against observations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PredictionErrors {
    /// Number of observations n.
    pub n: usize,
    /// Mean error (bias), Σeᵢ / n.
    pub mean_error: f64,
    /// Mean absolute error, Σ|eᵢ| / n.
    pub mean_absolute_error: f64,
    /// Root mean squared error, √(Σeᵢ² / (n − p)).
    pub rmse: f64,
    /// Coefficient of variation of the RMSE, `rmse / ȳ` with ȳ the mean of
    /// the observations; `None` when ȳ is 0. A fraction (× 100 for percent).
    pub cv_rmse: Option<f64>,
    /// Normalized mean bias error, Σeᵢ / ((n − p) · ȳ); `None` when ȳ is 0.
    /// A fraction (× 100 for percent); positive when the predictions are low.
    pub nmbe: Option<f64>,
    /// The p the RMSE and NMBE denominators subtract.
    pub parameters: usize,
}

/// Error measures of `predicted` against `observed`.
///
/// `parameters` is the number of parameters p the model estimated from these
/// observations; the RMSE and the NMBE divide by n − p, the residual degrees
/// of freedom (0 gives the plain means over n). With p the regression's
/// coefficient count these are the CV(RMSE) and NMBE of ASHRAE Guideline 14.
///
/// # Returns
///
/// `None` when the slices differ in length, are empty, hold a non-finite
/// value, or `parameters` is not below n.
///
/// # References
///
/// ASHRAE Guideline 14-2014, "Measurement of Energy, Demand, and Water
/// Savings", Annex B (CV(RMSE), NMBE).
/// Hyndman & Koehler (2006). "Another look at measures of forecast
/// accuracy", *International Journal of Forecasting* 22(4), 679-688.
pub fn prediction_errors(
    observed: &[f64],
    predicted: &[f64],
    parameters: usize,
) -> Option<PredictionErrors> {
    let n = observed.len();
    if n == 0 || n != predicted.len() || parameters >= n {
        return None;
    }
    if observed.iter().chain(predicted).any(|v| !v.is_finite()) {
        return None;
    }
    let nf = n as f64;
    let df = (n - parameters) as f64;
    let errors = observed.iter().zip(predicted).map(|(o, p)| o - p);
    let sum: f64 = errors.clone().sum();
    let sum_abs: f64 = errors.clone().map(f64::abs).sum();
    let sum_sq: f64 = errors.map(|e| e * e).sum();
    let mean_observed = observed.iter().sum::<f64>() / nf;
    let rmse = (sum_sq / df).sqrt();
    let relative = |v: f64| (mean_observed != 0.0).then(|| v / mean_observed);
    Some(PredictionErrors {
        n,
        mean_error: sum / nf,
        mean_absolute_error: sum_abs / nf,
        rmse,
        cv_rmse: relative(rmse),
        nmbe: relative(sum / df),
        parameters,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_definitions_by_hand() {
        // e = (1, -1, 2, 0): Σe = 2, Σ|e| = 4, Σe² = 6, ȳ = 10.
        let observed = [11.0, 9.0, 12.0, 8.0];
        let predicted = [10.0, 10.0, 10.0, 8.0];
        let e = prediction_errors(&observed, &predicted, 0).unwrap();
        assert_eq!(e.n, 4);
        assert!((e.mean_error - 0.5).abs() < 1e-15);
        assert!((e.mean_absolute_error - 1.0).abs() < 1e-15);
        assert!((e.rmse - (6.0f64 / 4.0).sqrt()).abs() < 1e-15);
        assert!((e.cv_rmse.unwrap() - (1.5f64).sqrt() / 10.0).abs() < 1e-15);
        assert!((e.nmbe.unwrap() - 2.0 / 40.0).abs() < 1e-15);
        // Two estimated parameters: the denominators become n − p = 2.
        let e = prediction_errors(&observed, &predicted, 2).unwrap();
        assert!((e.rmse - 3.0f64.sqrt()).abs() < 1e-15);
        assert!((e.nmbe.unwrap() - 2.0 / 20.0).abs() < 1e-15);
        assert!((e.mean_error - 0.5).abs() < 1e-15, "the means stay over n");
    }

    #[test]
    fn ratios_need_a_nonzero_mean() {
        let e = prediction_errors(&[1.0, -1.0], &[0.0, 0.0], 0).unwrap();
        assert_eq!((e.cv_rmse, e.nmbe), (None, None));
        assert!((e.rmse - 1.0).abs() < 1e-15);
    }

    #[test]
    fn refusals() {
        assert!(prediction_errors(&[], &[], 0).is_none());
        assert!(prediction_errors(&[1.0], &[1.0, 2.0], 0).is_none());
        assert!(prediction_errors(&[1.0, 2.0], &[1.0, 2.0], 2).is_none());
        assert!(prediction_errors(&[1.0, f64::NAN], &[1.0, 2.0], 0).is_none());
    }

    #[test]
    fn a_change_of_unit_scales_the_absolute_measures_only() {
        let observed = [101.0, 98.5, 103.2, 99.9, 100.4];
        let predicted = [100.0, 99.0, 102.0, 100.5, 100.0];
        let a = prediction_errors(&observed, &predicted, 1).unwrap();
        for k in [1e-9, 1e9] {
            let o: Vec<f64> = observed.iter().map(|v| v * k).collect();
            let p: Vec<f64> = predicted.iter().map(|v| v * k).collect();
            let b = prediction_errors(&o, &p, 1).unwrap();
            assert!((b.rmse / k - a.rmse).abs() < 1e-9 * a.rmse);
            assert!((b.cv_rmse.unwrap() - a.cv_rmse.unwrap()).abs() < 1e-12);
            assert!((b.nmbe.unwrap() - a.nmbe.unwrap()).abs() < 1e-12);
        }
    }
}
