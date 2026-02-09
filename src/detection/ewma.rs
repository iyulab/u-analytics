//! Exponentially Weighted Moving Average (EWMA) chart for detecting small shifts.
//!
//! # Algorithm
//!
//! The EWMA statistic is defined as:
//!
//! ```text
//! Z_i = lambda * x_i + (1 - lambda) * Z_{i-1},   Z_0 = mu_0
//! ```
//!
//! Time-varying (exact) control limits:
//!
//! ```text
//! UCL_i = mu_0 + L * sigma * sqrt(lambda / (2 - lambda) * (1 - (1 - lambda)^(2*i)))
//! LCL_i = mu_0 - L * sigma * sqrt(lambda / (2 - lambda) * (1 - (1 - lambda)^(2*i)))
//! ```
//!
//! Asymptotic limits (as i -> infinity):
//!
//! ```text
//! UCL = mu_0 + L * sigma * sqrt(lambda / (2 - lambda))
//! LCL = mu_0 - L * sigma * sqrt(lambda / (2 - lambda))
//! ```
//!
//! # Parameters
//!
//! - **lambda**: smoothing constant in (0, 1]. Smaller values give more weight
//!   to historical data and are better at detecting small shifts.
//!   Typical range: 0.05-0.25.
//! - **L**: control limit width factor in multiples of sigma. Typical: 2.7-3.0.
//!
//! # Reference
//!
//! Roberts, S.W. (1959). "Control Chart Tests Based on Geometric Moving Averages",
//! *Technometrics* 1(3), pp. 239-250.

/// EWMA chart parameters.
///
/// Implements the Exponentially Weighted Moving Average control chart for
/// detecting small sustained shifts in the process mean.
///
/// # Reference
///
/// Roberts, S.W. (1959). "Control Chart Tests Based on Geometric Moving Averages",
/// *Technometrics* 1(3).
pub struct Ewma {
    /// Target process mean (mu_0).
    target: f64,
    /// Known process standard deviation (sigma).
    sigma: f64,
    /// Smoothing constant (0 < lambda <= 1).
    lambda: f64,
    /// Control limit width factor (L), default 3.0.
    l_factor: f64,
}

/// Result of EWMA analysis for a single observation.
#[derive(Debug, Clone)]
pub struct EwmaResult {
    /// EWMA statistic Z_i.
    pub ewma: f64,
    /// Upper control limit at this point (time-varying).
    pub ucl: f64,
    /// Lower control limit at this point (time-varying).
    pub lcl: f64,
    /// Whether the EWMA statistic exceeds the control limits.
    pub signal: bool,
    /// Index of this observation in the data sequence.
    pub index: usize,
}

impl Ewma {
    /// Creates a new EWMA chart with the given target mean and standard deviation.
    ///
    /// Uses default parameters lambda=0.2 and L=3.0.
    ///
    /// # Returns
    ///
    /// `None` if `sigma` is not positive or finite, or if `target` is not finite.
    ///
    /// # Reference
    ///
    /// Roberts (1959), *Technometrics* 1(3). Default lambda=0.2, L=3.0 provides
    /// good sensitivity for detecting shifts of 0.5-2.0 sigma.
    pub fn new(target: f64, sigma: f64) -> Option<Self> {
        Self::with_params(target, sigma, 0.2, 3.0)
    }

    /// Creates an EWMA chart with custom parameters.
    ///
    /// # Parameters
    ///
    /// - `target`: Process target mean (mu_0, must be finite)
    /// - `sigma`: Process standard deviation (must be positive and finite)
    /// - `lambda`: Smoothing constant (must be in (0, 1])
    /// - `l_factor`: Control limit width factor (must be positive and finite)
    ///
    /// # Returns
    ///
    /// `None` if any parameter is invalid.
    pub fn with_params(target: f64, sigma: f64, lambda: f64, l_factor: f64) -> Option<Self> {
        if !target.is_finite() {
            return None;
        }
        if !sigma.is_finite() || sigma <= 0.0 {
            return None;
        }
        if !lambda.is_finite() || lambda <= 0.0 || lambda > 1.0 {
            return None;
        }
        if !l_factor.is_finite() || l_factor <= 0.0 {
            return None;
        }
        Some(Self {
            target,
            sigma,
            lambda,
            l_factor,
        })
    }

    /// Computes the time-varying control limit half-width at observation index `i`.
    ///
    /// The exact formula is:
    /// ```text
    /// L * sigma * sqrt(lambda / (2 - lambda) * (1 - (1 - lambda)^(2*i)))
    /// ```
    ///
    /// This accounts for the reduced variance of the EWMA statistic in early
    /// observations, providing tighter limits initially that widen toward the
    /// asymptotic value.
    ///
    /// # Reference
    ///
    /// Roberts (1959), equation for exact EWMA variance.
    fn control_limit_half_width(&self, i: usize) -> f64 {
        let asymptotic_var = self.lambda / (2.0 - self.lambda);
        let decay = (1.0 - self.lambda).powi(2 * i as i32);
        let time_varying_var = asymptotic_var * (1.0 - decay);
        self.l_factor * self.sigma * time_varying_var.sqrt()
    }

    /// Analyzes a sequence of observations and returns EWMA statistics for each point.
    ///
    /// The EWMA statistic is initialized to the target mean (Z_0 = mu_0).
    /// Non-finite values in the data are skipped (the previous EWMA value
    /// is carried forward).
    ///
    /// # Complexity
    ///
    /// Time: O(n), Space: O(n)
    pub fn analyze(&self, data: &[f64]) -> Vec<EwmaResult> {
        let mut results = Vec::with_capacity(data.len());
        let mut z = self.target;

        for (i, &x) in data.iter().enumerate() {
            if !x.is_finite() {
                // Non-finite observations: carry forward the previous EWMA value.
                let half_width = self.control_limit_half_width(i + 1);
                results.push(EwmaResult {
                    ewma: z,
                    ucl: self.target + half_width,
                    lcl: self.target - half_width,
                    signal: false,
                    index: i,
                });
                continue;
            }

            z = self.lambda * x + (1.0 - self.lambda) * z;

            // i is 0-based, but the control limit formula uses 1-based indexing
            let half_width = self.control_limit_half_width(i + 1);
            let ucl = self.target + half_width;
            let lcl = self.target - half_width;
            let signal = z > ucl || z < lcl;

            results.push(EwmaResult {
                ewma: z,
                ucl,
                lcl,
                signal,
                index: i,
            });
        }

        results
    }

    /// Returns the indices of observations where an EWMA signal occurred.
    ///
    /// A signal occurs when the EWMA statistic exceeds the upper or lower
    /// control limit.
    ///
    /// # Complexity
    ///
    /// Time: O(n), Space: O(k) where k is the number of signal points
    pub fn signal_points(&self, data: &[f64]) -> Vec<usize> {
        self.analyze(data)
            .into_iter()
            .filter(|r| r.signal)
            .map(|r| r.index)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ewma_in_control_stays_near_target() {
        // All data at the target — EWMA should remain exactly at the target.
        let target = 25.0;
        let sigma = 2.0;
        let ewma = Ewma::new(target, sigma).expect("valid params");

        let data: Vec<f64> = vec![target; 50];
        let results = ewma.analyze(&data);

        assert_eq!(results.len(), 50);
        for r in &results {
            assert!(
                (r.ewma - target).abs() < 1e-10,
                "EWMA should stay at target when data == target, got {} at index {}",
                r.ewma,
                r.index
            );
            assert!(!r.signal, "no signals expected for in-control data at index {}", r.index);
        }
    }

    #[test]
    fn test_ewma_in_control_with_noise() {
        // Small symmetric deviations should not trigger signals.
        let target = 100.0;
        let sigma = 5.0;
        let ewma = Ewma::new(target, sigma).expect("valid params");

        // Alternating +0.2sigma and -0.2sigma
        let data: Vec<f64> = (0..100)
            .map(|i| {
                if i % 2 == 0 {
                    target + 0.2 * sigma
                } else {
                    target - 0.2 * sigma
                }
            })
            .collect();

        let signals = ewma.signal_points(&data);
        assert!(
            signals.is_empty(),
            "symmetric noise of 0.2sigma should not trigger EWMA signals"
        );
    }

    #[test]
    fn test_ewma_gradual_drift_detected() {
        // Gradual linear drift: x_i = target + 0.1*sigma*i
        let target = 0.0;
        let sigma = 1.0;
        let ewma = Ewma::new(target, sigma).expect("valid params");

        let data: Vec<f64> = (0..100).map(|i| target + 0.1 * sigma * i as f64).collect();

        let signals = ewma.signal_points(&data);
        assert!(
            !signals.is_empty(),
            "EWMA should detect gradual linear drift"
        );
    }

    #[test]
    fn test_ewma_lambda_1_degenerates_to_shewhart() {
        // When lambda=1, Z_i = x_i (no smoothing), so EWMA degenerates to
        // individual observations plotted against fixed limits.
        let target = 50.0;
        let sigma = 5.0;
        let l_factor = 3.0;
        let ewma = Ewma::with_params(target, sigma, 1.0, l_factor).expect("valid params");

        let data = [50.0, 55.0, 45.0, 70.0, 30.0];
        let results = ewma.analyze(&data);

        for (i, r) in results.iter().enumerate() {
            assert!(
                (r.ewma - data[i]).abs() < 1e-10,
                "with lambda=1, EWMA should equal the data point: Z_{}={} vs x_{}={}",
                i,
                r.ewma,
                i,
                data[i]
            );
        }

        // With lambda=1, the asymptotic factor sqrt(lambda/(2-lambda)) = sqrt(1/1) = 1
        // Limits should be target +/- L*sigma = 50 +/- 15
        // Point 70 should signal (70 > 65), point 30 should signal (30 < 35)
        assert!(results[3].signal, "70 should exceed UCL of ~65");
        assert!(results[4].signal, "30 should exceed LCL of ~35");
    }

    #[test]
    fn test_ewma_time_varying_limits_converge() {
        // Verify that the time-varying control limits converge to asymptotic limits.
        let target = 0.0;
        let sigma = 1.0;
        let lambda = 0.2;
        let l_factor = 3.0;
        let ewma = Ewma::with_params(target, sigma, lambda, l_factor).expect("valid params");

        // Asymptotic half-width: L * sigma * sqrt(lambda / (2 - lambda))
        let asymptotic_hw = l_factor * sigma * (lambda / (2.0 - lambda)).sqrt();

        // Generate 200 points at target to get the limit evolution
        let data: Vec<f64> = vec![target; 200];
        let results = ewma.analyze(&data);

        // Early limit should be smaller than asymptotic
        let first_hw = results[0].ucl - target;
        assert!(
            first_hw < asymptotic_hw,
            "initial limit half-width {} should be less than asymptotic {}",
            first_hw,
            asymptotic_hw
        );

        // Late limit should be very close to asymptotic
        let last_hw = results[199].ucl - target;
        assert!(
            (last_hw - asymptotic_hw).abs() < 1e-6,
            "limit at i=200 should be close to asymptotic: got {}, expected {}",
            last_hw,
            asymptotic_hw
        );

        // Limits should be monotonically non-decreasing
        for i in 1..results.len() {
            assert!(
                results[i].ucl >= results[i - 1].ucl - 1e-15,
                "UCL should be non-decreasing: UCL[{}]={} < UCL[{}]={}",
                i,
                results[i].ucl,
                i - 1,
                results[i - 1].ucl
            );
        }
    }

    #[test]
    fn test_ewma_limits_symmetric() {
        // UCL and LCL should be symmetric around the target.
        let target = 42.0;
        let sigma = 3.0;
        let ewma = Ewma::new(target, sigma).expect("valid params");

        let data: Vec<f64> = vec![target; 20];
        let results = ewma.analyze(&data);

        for r in &results {
            let ucl_dist = r.ucl - target;
            let lcl_dist = target - r.lcl;
            assert!(
                (ucl_dist - lcl_dist).abs() < 1e-12,
                "limits should be symmetric: UCL-target={}, target-LCL={}",
                ucl_dist,
                lcl_dist
            );
        }
    }

    #[test]
    fn test_ewma_empty_data() {
        let ewma = Ewma::new(0.0, 1.0).expect("valid params");
        let results = ewma.analyze(&[]);
        assert!(results.is_empty(), "empty data should produce empty results");

        let signals = ewma.signal_points(&[]);
        assert!(signals.is_empty(), "empty data should produce no signals");
    }

    #[test]
    fn test_ewma_single_point() {
        let ewma = Ewma::new(0.0, 1.0).expect("valid params");

        let results = ewma.analyze(&[0.0]);
        assert_eq!(results.len(), 1);
        assert!(!results[0].signal, "single in-control point should not signal");

        // Extreme single point
        let results = ewma.analyze(&[100.0]);
        assert_eq!(results.len(), 1);
        // Z_1 = 0.2*100 + 0.8*0 = 20
        // UCL_1 = 0 + 3*1*sqrt(0.2/1.8 * (1 - 0.8^2)) = 3*sqrt(0.1111*0.36) = 3*0.2 = 0.6
        // 20 > 0.6 → signal
        assert!(results[0].signal, "extreme single point should signal");
    }

    #[test]
    fn test_ewma_invalid_params() {
        // sigma must be positive
        assert!(Ewma::new(0.0, 0.0).is_none());
        assert!(Ewma::new(0.0, -1.0).is_none());
        assert!(Ewma::new(0.0, f64::NAN).is_none());
        assert!(Ewma::new(0.0, f64::INFINITY).is_none());

        // target must be finite
        assert!(Ewma::new(f64::NAN, 1.0).is_none());
        assert!(Ewma::new(f64::INFINITY, 1.0).is_none());

        // lambda must be in (0, 1]
        assert!(Ewma::with_params(0.0, 1.0, 0.0, 3.0).is_none());
        assert!(Ewma::with_params(0.0, 1.0, -0.1, 3.0).is_none());
        assert!(Ewma::with_params(0.0, 1.0, 1.1, 3.0).is_none());

        // l_factor must be positive
        assert!(Ewma::with_params(0.0, 1.0, 0.2, 0.0).is_none());
        assert!(Ewma::with_params(0.0, 1.0, 0.2, -1.0).is_none());
    }

    #[test]
    fn test_ewma_non_finite_data_skipped() {
        let ewma = Ewma::new(0.0, 1.0).expect("valid params");
        let data = [0.0, f64::NAN, 0.0, f64::INFINITY, 0.0];
        let results = ewma.analyze(&data);
        assert_eq!(results.len(), 5);
        // Non-finite points should not produce signals
        assert!(!results[1].signal);
        assert!(!results[3].signal);
    }

    #[test]
    fn test_ewma_step_shift_detected() {
        // Step shift of +2sigma at index 20.
        let target = 0.0;
        let sigma = 1.0;
        let ewma = Ewma::new(target, sigma).expect("valid params");

        let mut data = vec![0.0; 50];
        for x in data.iter_mut().skip(20) {
            *x = 2.0; // +2sigma shift
        }

        let signals = ewma.signal_points(&data);
        assert!(!signals.is_empty(), "EWMA should detect a 2-sigma step shift");

        let first_signal = signals[0];
        assert!(
            first_signal >= 20,
            "signal should not appear before the shift, got {}",
            first_signal
        );
    }

    #[test]
    fn test_ewma_small_lambda_more_smoothing() {
        // Smaller lambda gives more weight to history, so the EWMA responds
        // more slowly. After a step shift, lambda=0.05 should accumulate
        // slower than lambda=0.25.
        let target = 0.0;
        let sigma = 1.0;

        let ewma_slow = Ewma::with_params(target, sigma, 0.05, 3.0).expect("valid params");
        let ewma_fast = Ewma::with_params(target, sigma, 0.25, 3.0).expect("valid params");

        // Step shift of +2sigma at index 10
        let mut data = vec![0.0; 30];
        for x in data.iter_mut().skip(10) {
            *x = 2.0;
        }

        let results_slow = ewma_slow.analyze(&data);
        let results_fast = ewma_fast.analyze(&data);

        // After a few points post-shift, the fast EWMA should be closer to the shifted mean
        let z_slow_15 = results_slow[15].ewma;
        let z_fast_15 = results_fast[15].ewma;

        assert!(
            z_fast_15 > z_slow_15,
            "fast EWMA (lambda=0.25) should respond faster: z_fast={} > z_slow={}",
            z_fast_15,
            z_slow_15
        );
    }
}
