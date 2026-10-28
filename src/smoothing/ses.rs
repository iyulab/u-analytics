//! Simple Exponential Smoothing (SES).
//!
//! Level-only smoothing for stationary time series (no trend, no seasonality).
//!
//! # Algorithm
//!
//! ```text
//! S_1 = x_1
//! S_t = α x_t + (1 - α) S_{t-1}
//! ```
//!
//! where α ∈ (0, 1) is the smoothing constant.
//!
//! # Reference
//!
//! Brown, R.G. (1956). *Exponential Smoothing for Predicting Demand*.

use super::forecast::{one_step_mse, residuals, valid_level, with_intervals, SmoothingForecast};

/// Result of simple exponential smoothing at each time step.
#[derive(Debug, Clone)]
pub struct SesResult {
    /// Smoothed values.
    pub smoothed: Vec<f64>,
    /// One-step-ahead forecast for the next period.
    pub forecast: f64,
    /// One-step-ahead in-sample forecasts: `fitted[t] = smoothed[t − 1]`, and
    /// `fitted[0]` the initial level `data[0]`.
    pub fitted: Vec<f64>,
    /// The smoothing constant α.
    pub alpha: f64,
}

impl SesResult {
    /// The index of the first one-step error the smoother learns from: before
    /// it the fitted value is the initialisation.
    pub const FIRST_ERROR: usize = 1;

    /// `data[t] − fitted[t]`, `None` at the initialisation.
    pub fn residuals(&self, data: &[f64]) -> Vec<Option<f64>> {
        residuals(data, &self.fitted, Self::FIRST_ERROR)
    }

    /// Forecasts 1..=`horizon` steps ahead with the prediction interval of
    /// ETS(A,N,N) at two-sided `level` ([`super::forecast`]); `data` is the
    /// series smoothed (for σ). `None` when `level` is not in (0, 1).
    pub fn forecasts(
        &self,
        data: &[f64],
        horizon: usize,
        level: f64,
    ) -> Option<Vec<SmoothingForecast>> {
        if !valid_level(level) {
            return None;
        }
        let alpha = self.alpha;
        let c = move |_: usize| alpha;
        Some(with_intervals(
            horizon,
            level,
            one_step_mse(data, &self.fitted, Self::FIRST_ERROR),
            Some(&c),
            |_| self.forecast,
        ))
    }
}

/// Simple Exponential Smoothing.
pub struct SimpleExponentialSmoothing {
    alpha: f64,
}

impl SimpleExponentialSmoothing {
    /// Creates a new SES smoother.
    ///
    /// # Parameters
    /// - `alpha`: smoothing constant, must be in (0, 1)
    ///
    /// Returns `None` if alpha is out of range or non-finite.
    pub fn new(alpha: f64) -> Option<Self> {
        if !alpha.is_finite() || alpha <= 0.0 || alpha >= 1.0 {
            return None;
        }
        Some(Self { alpha })
    }

    /// Returns the smoothing constant α.
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Applies SES to the given data.
    ///
    /// Returns `None` if data is empty.
    pub fn smooth(&self, data: &[f64]) -> Option<SesResult> {
        if data.is_empty() {
            return None;
        }

        let mut smoothed = Vec::with_capacity(data.len());
        let mut fitted = Vec::with_capacity(data.len());
        let mut s = data[0];
        fitted.push(s);
        smoothed.push(s);

        for &x in &data[1..] {
            fitted.push(s);
            s = self.alpha * x + (1.0 - self.alpha) * s;
            smoothed.push(s);
        }

        let forecast = s;
        Some(SesResult {
            smoothed,
            forecast,
            fitted,
            alpha: self.alpha,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ses_basic() {
        let ses = SimpleExponentialSmoothing::new(0.3).unwrap();
        let data = [10.0, 12.0, 13.0, 11.0, 14.0];
        let result = ses.smooth(&data).unwrap();

        assert_eq!(result.smoothed.len(), 5);
        // S1 = 10
        assert!((result.smoothed[0] - 10.0).abs() < 1e-10);
        // S2 = 0.3*12 + 0.7*10 = 10.6
        assert!((result.smoothed[1] - 10.6).abs() < 1e-10);
    }

    #[test]
    fn test_ses_constant_series() {
        let ses = SimpleExponentialSmoothing::new(0.5).unwrap();
        let data = [5.0; 10];
        let result = ses.smooth(&data).unwrap();

        for &v in &result.smoothed {
            assert!((v - 5.0).abs() < 1e-10);
        }
        assert!((result.forecast - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_ses_alpha_effect() {
        // Higher alpha → more responsive to recent data
        let data = [10.0, 20.0, 10.0];

        let low = SimpleExponentialSmoothing::new(0.1).unwrap();
        let high = SimpleExponentialSmoothing::new(0.9).unwrap();

        let r_low = low.smooth(&data).unwrap();
        let r_high = high.smooth(&data).unwrap();

        // After step-up, high alpha should be closer to 20
        assert!(r_high.smoothed[1] > r_low.smoothed[1]);
    }

    #[test]
    fn test_ses_single_point() {
        let ses = SimpleExponentialSmoothing::new(0.5).unwrap();
        let result = ses.smooth(&[42.0]).unwrap();
        assert_eq!(result.smoothed.len(), 1);
        assert!((result.forecast - 42.0).abs() < 1e-10);
    }

    #[test]
    fn test_ses_empty() {
        let ses = SimpleExponentialSmoothing::new(0.5).unwrap();
        assert!(ses.smooth(&[]).is_none());
    }

    #[test]
    fn test_ses_invalid_alpha() {
        assert!(SimpleExponentialSmoothing::new(0.0).is_none());
        assert!(SimpleExponentialSmoothing::new(1.0).is_none());
        assert!(SimpleExponentialSmoothing::new(-0.1).is_none());
        assert!(SimpleExponentialSmoothing::new(f64::NAN).is_none());
    }
}
