//! Cumulative Sum (CUSUM) chart for detecting small persistent shifts in process mean.
//!
//! # Algorithm
//!
//! Given observations x_1, ..., x_n with known process mean mu_0 and standard
//! deviation sigma, the standardized values are:
//!
//! ```text
//! z_i = (x_i - mu_0) / sigma
//! ```
//!
//! The upper and lower CUSUM statistics are:
//!
//! ```text
//! S_H(i) = max(0, S_H(i-1) + z_i - k)
//! S_L(i) = max(0, S_L(i-1) - z_i - k)
//! ```
//!
//! A signal is generated when `S_H(i) > h` or `S_L(i) > h`.
//!
//! # Parameters
//!
//! - **k**: reference value (allowance), typically 0.5 (designed to detect a 1-sigma shift)
//! - **h**: decision interval, typically 4 or 5
//!
//! # Reference
//!
//! Page, E.S. (1954). "Continuous inspection schemes", *Biometrika* 41(1-2), pp. 100-115.

/// CUSUM chart parameters and state.
///
/// Implements the tabular (two-sided) CUSUM procedure for detecting
/// both upward and downward shifts in a process mean.
///
/// # Examples
///
/// ```
/// use u_analytics::detection::Cusum;
///
/// let cusum = Cusum::new(10.0, 1.0).unwrap();
/// // In-control data
/// let data = [10.1, 9.8, 10.2, 9.9, 10.0, 10.1, 9.7, 10.3];
/// let results = cusum.analyze(&data);
/// assert_eq!(results.len(), data.len());
/// assert!(cusum.signal_points(&data).is_empty());
/// ```
///
/// # Reference
///
/// Page, E.S. (1954). "Continuous inspection schemes", *Biometrika* 41(1-2).
pub struct Cusum {
    /// Target process mean (mu_0).
    target: f64,
    /// Known process standard deviation (sigma).
    sigma: f64,
    /// Reference value (allowance), default 0.5.
    k: f64,
    /// Decision interval, default 5.0.
    h: f64,
}

/// Result of CUSUM analysis for a single observation.
#[derive(Debug, Clone)]
pub struct CusumResult {
    /// Upper cumulative sum S_H(i).
    pub s_upper: f64,
    /// Lower cumulative sum S_L(i).
    pub s_lower: f64,
    /// Whether the decision interval was exceeded (S_H > h or S_L > h).
    pub signal: bool,
    /// Index of this observation in the data sequence.
    pub index: usize,
}

impl Cusum {
    /// Creates a new CUSUM chart with the given target mean and standard deviation.
    ///
    /// Uses default parameters k=0.5 (detects 1-sigma shift) and h=5.0.
    ///
    /// # Returns
    ///
    /// `None` if `sigma` is not positive or finite, or if `target` is not finite.
    ///
    /// # Reference
    ///
    /// Page (1954), *Biometrika* 41(1-2). Default k=0.5 is optimal for detecting
    /// a shift of 1 sigma; h=5 gives ARL_0 ~ 465.
    pub fn new(target: f64, sigma: f64) -> Option<Self> {
        Self::with_params(target, sigma, 0.5, 5.0)
    }

    /// Creates a CUSUM chart with custom k and h parameters.
    ///
    /// # Parameters
    ///
    /// - `target`: Process target mean (mu_0)
    /// - `sigma`: Process standard deviation (must be positive and finite)
    /// - `k`: Reference value / allowance (must be non-negative and finite)
    /// - `h`: Decision interval (must be positive and finite)
    ///
    /// # Returns
    ///
    /// `None` if any parameter is invalid.
    pub fn with_params(target: f64, sigma: f64, k: f64, h: f64) -> Option<Self> {
        if !target.is_finite() {
            return None;
        }
        if !sigma.is_finite() || sigma <= 0.0 {
            return None;
        }
        if !k.is_finite() || k < 0.0 {
            return None;
        }
        if !h.is_finite() || h <= 0.0 {
            return None;
        }
        Some(Self {
            target,
            sigma,
            k,
            h,
        })
    }

    /// Analyzes a sequence of observations and returns CUSUM statistics for each point.
    ///
    /// The upper CUSUM detects upward shifts; the lower CUSUM detects downward shifts.
    /// Both are initialized to zero. Non-finite values in the data are skipped
    /// (their index is still consumed).
    ///
    /// # Examples
    ///
    /// ```
    /// use u_analytics::detection::Cusum;
    ///
    /// let cusum = Cusum::new(10.0, 1.0).unwrap();
    /// // Data with upward shift
    /// let mut data: Vec<f64> = vec![10.0; 10];
    /// data.extend(vec![12.0; 10]); // shift of 2 sigma
    /// let signals = cusum.signal_points(&data);
    /// assert!(!signals.is_empty()); // shift detected
    /// ```
    ///
    /// # Complexity
    ///
    /// Time: O(n), Space: O(n)
    pub fn analyze(&self, data: &[f64]) -> Vec<CusumResult> {
        let mut results = Vec::with_capacity(data.len());
        let mut s_upper = 0.0_f64;
        let mut s_lower = 0.0_f64;

        for (i, &x) in data.iter().enumerate() {
            if !x.is_finite() {
                // Non-finite observations: carry forward previous CUSUM values, no signal.
                results.push(CusumResult {
                    s_upper,
                    s_lower,
                    signal: false,
                    index: i,
                });
                continue;
            }

            let z = (x - self.target) / self.sigma;
            s_upper = (s_upper + z - self.k).max(0.0);
            s_lower = (s_lower - z - self.k).max(0.0);

            let signal = s_upper > self.h || s_lower > self.h;

            results.push(CusumResult {
                s_upper,
                s_lower,
                signal,
                index: i,
            });
        }

        results
    }

    /// Returns the indices of observations where a CUSUM signal occurred.
    ///
    /// A signal occurs when the upper or lower cumulative sum exceeds the
    /// decision interval h.
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
    fn test_cusum_in_control_no_signals() {
        // Data centered around target with small noise — should produce no signals.
        let target = 10.0;
        let sigma = 1.0;
        let cusum = Cusum::new(target, sigma).expect("valid params");

        // 50 points right at the target: z_i = 0 for all i
        let data: Vec<f64> = vec![target; 50];
        let results = cusum.analyze(&data);

        assert_eq!(results.len(), 50);
        for r in &results {
            assert!(!r.signal, "no signals expected for in-control data at index {}", r.index);
            assert!(
                r.s_upper.abs() < 1e-10,
                "s_upper should be 0 when data == target"
            );
            assert!(
                r.s_lower.abs() < 1e-10,
                "s_lower should be 0 when data == target"
            );
        }
    }

    #[test]
    fn test_cusum_in_control_with_noise() {
        // Small symmetric deviations should not trigger signals with h=5.
        let target = 50.0;
        let sigma = 2.0;
        let cusum = Cusum::new(target, sigma).expect("valid params");

        // Alternating +0.3sigma and -0.3sigma around target
        let data: Vec<f64> = (0..100)
            .map(|i| {
                if i % 2 == 0 {
                    target + 0.3 * sigma
                } else {
                    target - 0.3 * sigma
                }
            })
            .collect();

        let signals = cusum.signal_points(&data);
        assert!(
            signals.is_empty(),
            "symmetric noise of 0.3sigma should not trigger CUSUM signals"
        );
    }

    #[test]
    fn test_cusum_step_shift_detected() {
        // Step shift of +2sigma at index 20.
        let target = 100.0;
        let sigma = 5.0;
        let cusum = Cusum::new(target, sigma).expect("valid params");

        let mut data = vec![target; 50];
        // Introduce a +2sigma shift starting at index 20
        for x in data.iter_mut().skip(20) {
            *x = target + 2.0 * sigma;
        }

        let signals = cusum.signal_points(&data);
        assert!(
            !signals.is_empty(),
            "CUSUM should detect a 2-sigma step shift"
        );

        // First signal should occur near the shift point (within a few samples)
        let first_signal = signals[0];
        assert!(
            first_signal >= 20,
            "signal should not appear before the shift at index 20, got {}",
            first_signal
        );
        assert!(
            first_signal <= 30,
            "first signal should appear soon after shift at index 20, got {}",
            first_signal
        );
    }

    #[test]
    fn test_cusum_downward_shift_detected() {
        // Step shift of -2sigma at index 15.
        let target = 50.0;
        let sigma = 3.0;
        let cusum = Cusum::new(target, sigma).expect("valid params");

        let mut data = vec![target; 40];
        for x in data.iter_mut().skip(15) {
            *x = target - 2.0 * sigma;
        }

        let results = cusum.analyze(&data);
        let signals: Vec<usize> = results.iter().filter(|r| r.signal).map(|r| r.index).collect();
        assert!(
            !signals.is_empty(),
            "CUSUM should detect a -2sigma downward shift"
        );

        // Check that the lower CUSUM accumulated (not the upper)
        let first_signal_result = results.iter().find(|r| r.signal).expect("signal exists");
        assert!(
            first_signal_result.s_lower > first_signal_result.s_upper,
            "lower CUSUM should be larger for downward shift"
        );
    }

    #[test]
    fn test_cusum_custom_params() {
        let cusum = Cusum::with_params(0.0, 1.0, 0.25, 4.0);
        assert!(cusum.is_some(), "valid custom params should succeed");

        let cusum = cusum.expect("valid params");
        // k=0.25 is more sensitive, h=4.0 is a tighter threshold
        // A 1-sigma shift should be detected faster than with default params
        let mut data = vec![0.0; 30];
        for x in data.iter_mut().skip(10) {
            *x = 1.0; // 1-sigma shift
        }

        let signals = cusum.signal_points(&data);
        assert!(
            !signals.is_empty(),
            "k=0.25, h=4.0 should detect a 1-sigma shift"
        );
    }

    #[test]
    fn test_cusum_known_arl_k05_h5() {
        // With k=0.5 and h=5, a 1-sigma shift should have ARL ~ 8-10.
        // We verify that a sustained 1-sigma shift triggers within ~15 points.
        let cusum = Cusum::new(0.0, 1.0).expect("valid params");

        // All data points shifted by +1 sigma
        let data: Vec<f64> = vec![1.0; 20];
        let signals = cusum.signal_points(&data);

        assert!(
            !signals.is_empty(),
            "1-sigma shift should trigger within 20 observations"
        );
        // The first signal should appear within about 15 observations
        // (theoretical ARL ~ 8.38 for k=0.5, h=5, delta=1)
        assert!(
            signals[0] <= 15,
            "first signal should appear within ~15 observations for ARL ~8, got {}",
            signals[0]
        );
    }

    #[test]
    fn test_cusum_empty_data() {
        let cusum = Cusum::new(0.0, 1.0).expect("valid params");
        let results = cusum.analyze(&[]);
        assert!(results.is_empty(), "empty data should produce empty results");

        let signals = cusum.signal_points(&[]);
        assert!(signals.is_empty(), "empty data should produce no signals");
    }

    #[test]
    fn test_cusum_single_point() {
        let cusum = Cusum::new(0.0, 1.0).expect("valid params");

        let results = cusum.analyze(&[0.0]);
        assert_eq!(results.len(), 1);
        assert!(!results[0].signal, "single in-control point should not signal");

        let results = cusum.analyze(&[100.0]);
        assert_eq!(results.len(), 1);
        // z = 100, s_upper = 100 - 0.5 = 99.5 > 5 → signal
        assert!(results[0].signal, "extreme single point should signal");
    }

    #[test]
    fn test_cusum_invalid_params() {
        // sigma must be positive
        assert!(Cusum::new(0.0, 0.0).is_none());
        assert!(Cusum::new(0.0, -1.0).is_none());
        assert!(Cusum::new(0.0, f64::NAN).is_none());
        assert!(Cusum::new(0.0, f64::INFINITY).is_none());

        // target must be finite
        assert!(Cusum::new(f64::NAN, 1.0).is_none());
        assert!(Cusum::new(f64::INFINITY, 1.0).is_none());

        // k must be non-negative
        assert!(Cusum::with_params(0.0, 1.0, -0.1, 5.0).is_none());

        // h must be positive
        assert!(Cusum::with_params(0.0, 1.0, 0.5, 0.0).is_none());
        assert!(Cusum::with_params(0.0, 1.0, 0.5, -1.0).is_none());
    }

    #[test]
    fn test_cusum_non_finite_data_skipped() {
        let cusum = Cusum::new(0.0, 1.0).expect("valid params");
        let data = [0.0, f64::NAN, 0.0, f64::INFINITY, 0.0];
        let results = cusum.analyze(&data);
        assert_eq!(results.len(), 5);
        // Non-finite points should not produce signals
        assert!(!results[1].signal);
        assert!(!results[3].signal);
    }

    #[test]
    fn test_cusum_s_upper_s_lower_non_negative() {
        // CUSUM statistics should always be >= 0 by construction.
        let cusum = Cusum::new(50.0, 5.0).expect("valid params");
        let data: Vec<f64> = (0..100).map(|i| 50.0 + (i as f64 * 0.1).sin() * 3.0).collect();
        let results = cusum.analyze(&data);
        for r in &results {
            assert!(r.s_upper >= 0.0, "s_upper must be non-negative at index {}", r.index);
            assert!(r.s_lower >= 0.0, "s_lower must be non-negative at index {}", r.index);
        }
    }
}
