//! Holt-Winters Triple Exponential Smoothing.
//!
//! Extends Holt's method with a seasonal component, supporting both
//! additive and multiplicative seasonality.
//!
//! # Algorithm (Additive)
//!
//! ```text
//! Level:    L_t = α (x_t - S_{t-m}) + (1 - α)(L_{t-1} + T_{t-1})
//! Trend:    T_t = β (L_t - L_{t-1}) + (1 - β) T_{t-1}
//! Season:   S_t = γ (x_t - L_t) + (1 - γ) S_{t-m}
//! Forecast: F_{t+h} = L_t + h T_t + S_{t-m+h_m}
//! ```
//!
//! # Algorithm (Multiplicative)
//!
//! ```text
//! Level:    L_t = α (x_t / S_{t-m}) + (1 - α)(L_{t-1} + T_{t-1})
//! Trend:    T_t = β (L_t - L_{t-1}) + (1 - β) T_{t-1}
//! Season:   S_t = γ (x_t / L_t) + (1 - γ) S_{t-m}
//! Forecast: F_{t+h} = (L_t + h T_t) S_{t-m+h_m}
//! ```
//!
//! # Parameters
//!
//! - α ∈ (0, 1): level smoothing
//! - β ∈ (0, 1): trend smoothing
//! - γ ∈ (0, 1): seasonal smoothing
//! - m: seasonal period (e.g., 12 for monthly data with yearly cycle)
//!
//! # Reference
//!
//! Winters, P.R. (1960). "Forecasting Sales by Exponentially Weighted
//! Moving Averages", *Management Science* 6(3), pp. 324-342.

/// Seasonality type.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Seasonality {
    /// Additive: seasonal effect is added to the trend.
    Additive,
    /// Multiplicative: seasonal effect multiplies the trend.
    Multiplicative,
}

/// Result of Holt-Winters smoothing.
#[derive(Debug, Clone)]
pub struct HoltWintersResult {
    /// Level estimates.
    pub level: Vec<f64>,
    /// Trend estimates.
    pub trend: Vec<f64>,
    /// Seasonal factors (length = data length + initial period).
    pub seasonal: Vec<f64>,
    /// Fitted values (one-step-ahead in-sample forecasts).
    pub fitted: Vec<f64>,
}

impl HoltWintersResult {
    /// Returns a forecast h steps ahead from the last observation.
    ///
    /// # Parameters
    /// - `h`: steps ahead (1-indexed)
    /// - `period`: seasonal period m
    /// - `seasonality`: additive or multiplicative
    pub fn forecast(&self, h: usize, period: usize, seasonality: Seasonality) -> f64 {
        let last_l = *self.level.last().expect("level must be non-empty");
        let last_t = *self.trend.last().expect("trend must be non-empty");

        // Seasonal factor: use the most recent completed cycle
        let s_len = self.seasonal.len();
        let idx = s_len - period + ((h - 1) % period);
        let s = self.seasonal[idx];

        match seasonality {
            Seasonality::Additive => last_l + h as f64 * last_t + s,
            Seasonality::Multiplicative => (last_l + h as f64 * last_t) * s,
        }
    }
}

/// Holt-Winters Triple Exponential Smoothing.
pub struct HoltWinters {
    alpha: f64,
    beta: f64,
    gamma: f64,
    period: usize,
    seasonality: Seasonality,
}

impl HoltWinters {
    /// Creates a new Holt-Winters smoother.
    ///
    /// # Parameters
    /// - `alpha`: level smoothing constant ∈ (0, 1)
    /// - `beta`: trend smoothing constant ∈ (0, 1)
    /// - `gamma`: seasonal smoothing constant ∈ (0, 1)
    /// - `period`: seasonal period (must be ≥ 2)
    /// - `seasonality`: additive or multiplicative
    ///
    /// Returns `None` if parameters are invalid.
    pub fn new(
        alpha: f64,
        beta: f64,
        gamma: f64,
        period: usize,
        seasonality: Seasonality,
    ) -> Option<Self> {
        if !alpha.is_finite() || alpha <= 0.0 || alpha >= 1.0 {
            return None;
        }
        if !beta.is_finite() || beta <= 0.0 || beta >= 1.0 {
            return None;
        }
        if !gamma.is_finite() || gamma <= 0.0 || gamma >= 1.0 {
            return None;
        }
        if period < 2 {
            return None;
        }
        Some(Self {
            alpha,
            beta,
            gamma,
            period,
            seasonality,
        })
    }

    /// Returns the seasonal period.
    pub fn period(&self) -> usize {
        self.period
    }

    /// Returns the seasonality type.
    pub fn seasonality(&self) -> Seasonality {
        self.seasonality
    }

    /// Applies Holt-Winters smoothing to the data.
    ///
    /// Requires at least `2 * period` data points for initialization.
    /// Returns `None` if data is insufficient or if multiplicative
    /// seasonality is used with non-positive data.
    pub fn smooth(&self, data: &[f64]) -> Option<HoltWintersResult> {
        let m = self.period;
        let n = data.len();

        if n < 2 * m {
            return None;
        }

        // Multiplicative: all data must be positive
        if self.seasonality == Seasonality::Multiplicative && data.iter().any(|&x| x <= 0.0) {
            return None;
        }

        // --- Initialization ---
        let l0: f64 = data[..m].iter().sum::<f64>() / m as f64;
        let t0: f64 = (0..m)
            .map(|i| (data[m + i] - data[i]) / m as f64)
            .sum::<f64>()
            / m as f64;

        // seasonal[i] = seasonal factor for time i
        let mut seasonal = vec![0.0; n];
        match self.seasonality {
            Seasonality::Additive => {
                for i in 0..m {
                    seasonal[i] = data[i] - l0;
                }
            }
            Seasonality::Multiplicative => {
                for i in 0..m {
                    seasonal[i] = data[i] / l0;
                }
            }
        }

        let mut level = vec![0.0; n];
        let mut trend = vec![0.0; n];
        let mut fitted = vec![0.0; n];

        // Set initial values for times 0..m-1
        for i in 0..m {
            level[i] = l0;
            trend[i] = t0;
            fitted[i] = match self.seasonality {
                Seasonality::Additive => l0 + seasonal[i],
                Seasonality::Multiplicative => l0 * seasonal[i],
            };
        }

        // Main smoothing loop
        for t in m..n {
            let s_prev = seasonal[t - m];

            let l = match self.seasonality {
                Seasonality::Additive => {
                    self.alpha * (data[t] - s_prev)
                        + (1.0 - self.alpha) * (level[t - 1] + trend[t - 1])
                }
                Seasonality::Multiplicative => {
                    self.alpha * (data[t] / s_prev)
                        + (1.0 - self.alpha) * (level[t - 1] + trend[t - 1])
                }
            };

            let b = self.beta * (l - level[t - 1]) + (1.0 - self.beta) * trend[t - 1];

            let s = match self.seasonality {
                Seasonality::Additive => {
                    self.gamma * (data[t] - l) + (1.0 - self.gamma) * s_prev
                }
                Seasonality::Multiplicative => {
                    self.gamma * (data[t] / l) + (1.0 - self.gamma) * s_prev
                }
            };

            level[t] = l;
            trend[t] = b;
            seasonal[t] = s;

            // One-step-ahead fitted value
            fitted[t] = match self.seasonality {
                Seasonality::Additive => level[t - 1] + trend[t - 1] + s_prev,
                Seasonality::Multiplicative => (level[t - 1] + trend[t - 1]) * s_prev,
            };
        }

        Some(HoltWintersResult {
            level,
            trend,
            seasonal,
            fitted,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seasonal_additive_data() -> Vec<f64> {
        // Base trend: 100 + 2*t with seasonal pattern [10, -5, -5, 0]
        let pattern = [10.0, -5.0, -5.0, 0.0];
        (0..24)
            .map(|t| 100.0 + 2.0 * t as f64 + pattern[t % 4])
            .collect()
    }

    fn seasonal_multiplicative_data() -> Vec<f64> {
        // Base trend: 100 + 2*t with multiplicative seasonal [1.2, 0.8, 0.9, 1.1]
        let pattern = [1.2, 0.8, 0.9, 1.1];
        (0..24)
            .map(|t| (100.0 + 2.0 * t as f64) * pattern[t % 4])
            .collect()
    }

    #[test]
    fn test_hw_additive_basic() {
        let data = seasonal_additive_data();
        let hw = HoltWinters::new(0.3, 0.1, 0.3, 4, Seasonality::Additive).unwrap();
        let result = hw.smooth(&data).unwrap();

        assert_eq!(result.level.len(), 24);
        assert_eq!(result.trend.len(), 24);
        assert_eq!(result.seasonal.len(), 24);
        assert_eq!(result.fitted.len(), 24);
    }

    #[test]
    fn test_hw_additive_forecast() {
        let data = seasonal_additive_data();
        let hw = HoltWinters::new(0.3, 0.1, 0.3, 4, Seasonality::Additive).unwrap();
        let result = hw.smooth(&data).unwrap();

        // Forecast should continue the pattern
        let f1 = result.forecast(1, 4, Seasonality::Additive);
        let f4 = result.forecast(4, 4, Seasonality::Additive);

        // Both should be in reasonable range
        assert!(f1 > 100.0, "forecast(1) = {f1}");
        assert!(f4 > f1 - 20.0, "forecast(4) = {f4}");
    }

    #[test]
    fn test_hw_multiplicative_basic() {
        let data = seasonal_multiplicative_data();
        let hw = HoltWinters::new(0.3, 0.1, 0.3, 4, Seasonality::Multiplicative).unwrap();
        let result = hw.smooth(&data).unwrap();

        assert_eq!(result.level.len(), 24);
        assert_eq!(result.fitted.len(), 24);
    }

    #[test]
    fn test_hw_fitted_approximates_data() {
        let data = seasonal_additive_data();
        let hw = HoltWinters::new(0.5, 0.3, 0.5, 4, Seasonality::Additive).unwrap();
        let result = hw.smooth(&data).unwrap();

        // After warm-up, fitted should be close to data
        let mape: f64 = (8..24)
            .map(|i| ((result.fitted[i] - data[i]) / data[i]).abs())
            .sum::<f64>()
            / 16.0;

        assert!(
            mape < 0.10,
            "mean absolute percentage error = {mape}, expected < 10%"
        );
    }

    #[test]
    fn test_hw_seasonal_pattern_detected() {
        let data = seasonal_additive_data();
        let hw = HoltWinters::new(0.3, 0.1, 0.5, 4, Seasonality::Additive).unwrap();
        let result = hw.smooth(&data).unwrap();

        // Last seasonal factors should reflect the pattern [10, -5, -5, 0]
        let last_cycle: Vec<f64> = (20..24).map(|i| result.seasonal[i]).collect();

        // Highest seasonal should be at position 0 mod 4 (pattern = +10)
        let max_idx = last_cycle
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(max_idx, 0, "highest seasonal at wrong position");
    }

    #[test]
    fn test_hw_insufficient_data() {
        let hw = HoltWinters::new(0.3, 0.1, 0.3, 4, Seasonality::Additive).unwrap();
        // Need at least 2*period = 8 data points
        assert!(hw.smooth(&[1.0; 7]).is_none());
        assert!(hw.smooth(&[1.0; 8]).is_some());
    }

    #[test]
    fn test_hw_multiplicative_rejects_negative() {
        let hw = HoltWinters::new(0.3, 0.1, 0.3, 4, Seasonality::Multiplicative).unwrap();
        let data = vec![1.0, 2.0, -1.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        assert!(hw.smooth(&data).is_none());
    }

    #[test]
    fn test_hw_invalid_params() {
        assert!(HoltWinters::new(0.0, 0.5, 0.5, 4, Seasonality::Additive).is_none());
        assert!(HoltWinters::new(0.5, 1.0, 0.5, 4, Seasonality::Additive).is_none());
        assert!(HoltWinters::new(0.5, 0.5, 0.0, 4, Seasonality::Additive).is_none());
        assert!(HoltWinters::new(0.5, 0.5, 0.5, 1, Seasonality::Additive).is_none());
    }

    #[test]
    fn test_hw_trend_detected() {
        let data = seasonal_additive_data();
        let hw = HoltWinters::new(0.3, 0.3, 0.3, 4, Seasonality::Additive).unwrap();
        let result = hw.smooth(&data).unwrap();

        // True trend is +2.0 per step; estimated should be positive
        let last_trend = result.trend[23];
        assert!(
            last_trend > 1.0 && last_trend < 4.0,
            "trend = {last_trend}, expected ~2.0"
        );
    }

    #[test]
    fn test_hw_level_tracks_mean() {
        let data = seasonal_additive_data();
        let hw = HoltWinters::new(0.3, 0.1, 0.3, 4, Seasonality::Additive).unwrap();
        let result = hw.smooth(&data).unwrap();

        // At t=23, true level ≈ 100 + 2*23 = 146
        let last_level = result.level[23];
        assert!(
            (last_level - 146.0).abs() < 10.0,
            "level = {last_level}, expected ~146"
        );
    }
}
