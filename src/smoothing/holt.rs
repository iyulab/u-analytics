//! Holt's Linear (Double Exponential) Smoothing.
//!
//! Extends simple exponential smoothing with a trend component for
//! non-stationary time series that exhibit a linear trend.
//!
//! # Algorithm
//!
//! ```text
//! Level:   L_t = α x_t + (1 - α)(L_{t-1} + T_{t-1})
//! Trend:   T_t = β (L_t - L_{t-1}) + (1 - β) T_{t-1}
//! Forecast: F_{t+h} = L_t + h T_t
//! ```
//!
//! where α ∈ (0, 1) is the level smoothing constant and
//! β ∈ (0, 1) is the trend smoothing constant.
//!
//! # Reference
//!
//! Holt, C.C. (1957). "Forecasting Seasonals and Trends by
//! Exponentially Weighted Moving Averages", ONR Memo 52.

/// Result of Holt's linear smoothing at each time step.
#[derive(Debug, Clone)]
pub struct HoltResult {
    /// Level estimates.
    pub level: Vec<f64>,
    /// Trend estimates.
    pub trend: Vec<f64>,
    /// Fitted values (level + trend from previous step).
    pub fitted: Vec<f64>,
}

impl HoltResult {
    /// Returns a forecast h steps ahead from the last observation.
    pub fn forecast(&self, h: usize) -> f64 {
        let last_l = *self.level.last().expect("level must be non-empty");
        let last_t = *self.trend.last().expect("trend must be non-empty");
        last_l + h as f64 * last_t
    }
}

/// Holt's Linear Exponential Smoothing.
pub struct HoltLinear {
    alpha: f64,
    beta: f64,
}

impl HoltLinear {
    /// Creates a new Holt smoother.
    ///
    /// # Parameters
    /// - `alpha`: level smoothing constant ∈ (0, 1)
    /// - `beta`: trend smoothing constant ∈ (0, 1)
    ///
    /// Returns `None` if parameters are out of range.
    pub fn new(alpha: f64, beta: f64) -> Option<Self> {
        if !alpha.is_finite() || alpha <= 0.0 || alpha >= 1.0 {
            return None;
        }
        if !beta.is_finite() || beta <= 0.0 || beta >= 1.0 {
            return None;
        }
        Some(Self { alpha, beta })
    }

    /// Returns α.
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Returns β.
    pub fn beta(&self) -> f64 {
        self.beta
    }

    /// Applies Holt's linear smoothing to the data.
    ///
    /// Requires at least 2 data points.
    /// Returns `None` if data has fewer than 2 points.
    pub fn smooth(&self, data: &[f64]) -> Option<HoltResult> {
        if data.len() < 2 {
            return None;
        }

        let n = data.len();
        let mut level = Vec::with_capacity(n);
        let mut trend = Vec::with_capacity(n);
        let mut fitted = Vec::with_capacity(n);

        // Initialize
        let l0 = data[0];
        let t0 = data[1] - data[0];
        level.push(l0);
        trend.push(t0);
        fitted.push(l0); // first fitted = initial level

        for i in 1..n {
            let l_prev = level[i - 1];
            let t_prev = trend[i - 1];

            let l = self.alpha * data[i] + (1.0 - self.alpha) * (l_prev + t_prev);
            let t = self.beta * (l - l_prev) + (1.0 - self.beta) * t_prev;

            fitted.push(l_prev + t_prev); // one-step-ahead forecast
            level.push(l);
            trend.push(t);
        }

        Some(HoltResult {
            level,
            trend,
            fitted,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_holt_linear_trend() {
        // Linear data: 10, 12, 14, 16, 18
        let holt = HoltLinear::new(0.5, 0.5).unwrap();
        let data = [10.0, 12.0, 14.0, 16.0, 18.0];
        let result = holt.smooth(&data).unwrap();

        // Forecast should be close to 20 for h=1
        let f1 = result.forecast(1);
        assert!((f1 - 20.0).abs() < 2.0, "forecast(1) = {f1}, expected ~20");
    }

    #[test]
    fn test_holt_constant_series() {
        let holt = HoltLinear::new(0.3, 0.3).unwrap();
        let data = [5.0; 20];
        let result = holt.smooth(&data).unwrap();

        // Trend should converge to ~0
        let last_trend = *result.trend.last().unwrap();
        assert!(
            last_trend.abs() < 0.1,
            "trend should be near 0, got {last_trend}"
        );
    }

    #[test]
    fn test_holt_forecast_multi_step() {
        let holt = HoltLinear::new(0.5, 0.5).unwrap();
        let data = [10.0, 12.0, 14.0, 16.0, 18.0];
        let result = holt.smooth(&data).unwrap();

        let f1 = result.forecast(1);
        let f3 = result.forecast(3);
        let f5 = result.forecast(5);

        // Multi-step forecasts should be in order
        assert!(f1 < f3);
        assert!(f3 < f5);
    }

    #[test]
    fn test_holt_minimum_data() {
        let holt = HoltLinear::new(0.5, 0.5).unwrap();
        let result = holt.smooth(&[10.0, 15.0]).unwrap();
        assert_eq!(result.level.len(), 2);
        assert_eq!(result.fitted.len(), 2);
    }

    #[test]
    fn test_holt_insufficient_data() {
        let holt = HoltLinear::new(0.5, 0.5).unwrap();
        assert!(holt.smooth(&[10.0]).is_none());
        assert!(holt.smooth(&[]).is_none());
    }

    #[test]
    fn test_holt_invalid_params() {
        assert!(HoltLinear::new(0.0, 0.5).is_none());
        assert!(HoltLinear::new(1.0, 0.5).is_none());
        assert!(HoltLinear::new(0.5, 0.0).is_none());
        assert!(HoltLinear::new(0.5, 1.0).is_none());
        assert!(HoltLinear::new(f64::NAN, 0.5).is_none());
    }

    #[test]
    fn test_holt_fitted_values() {
        let holt = HoltLinear::new(0.5, 0.5).unwrap();
        let data = [10.0, 12.0, 14.0, 16.0];
        let result = holt.smooth(&data).unwrap();

        assert_eq!(result.fitted.len(), data.len());
        // First fitted = initial level
        assert!((result.fitted[0] - 10.0).abs() < 1e-10);
    }
}
