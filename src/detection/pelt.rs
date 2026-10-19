//! Pruned Exact Linear Time (PELT) algorithm for offline changepoint detection.
//!
//! # Algorithm
//!
//! PELT solves the penalized cost minimization problem:
//!
//! ```text
//! minimize  sum_{j=1}^{m+1} C(y_{tau_{j-1}+1 : tau_j}) + m * beta
//! ```
//!
//! where `C` is a segment cost function, `m` is the number of changepoints,
//! and `beta` is a penalty per changepoint.
//!
//! The dynamic programming recurrence is:
//!
//! ```text
//! F(t) = min_{tau in R_t} [ F(tau) + C(y_{tau+1:t}) + beta ]
//! ```
//!
//! with pruning rule: remove `tau` from candidates if `F(tau) + C(y_{tau+1:t}) >= F(t)`,
//! since such `tau` can never be optimal for any future `s > t`.
//!
//! # Complexity
//!
//! Expected O(n) under mild conditions on the cost function.
//! Worst case O(n^2) (no pruning effective).
//!
//! # Cost Functions
//!
//! - [`CostFunction::L2`] — Detects changes in mean. Cost = sum of squared residuals
//!   from segment mean. One parameter per segment.
//! - [`CostFunction::Normal`] — Detects changes in mean and/or variance. Cost =
//!   n * ln(MLE variance). Two parameters per segment.
//!
//! # Penalty Selection
//!
//! - **BIC** (default): `beta = p * ln(n)` where `p` is the number of parameters
//!   per segment (1 for L2, 2 for Normal). Balances model complexity and fit.
//! - **Custom**: User-specified penalty value.
//!
//! # References
//!
//! - Killick, R., Fearnhead, P., & Eckley, I.A. (2012). "Optimal Detection of
//!   Changepoints with a Linear Computational Cost", *Journal of the American
//!   Statistical Association* 107(500), pp. 1590-1598.
//! - Schwarz, G. (1978). "Estimating the Dimension of a Model",
//!   *Annals of Statistics* 6(2), pp. 461-464.

/// Cost function for evaluating segment homogeneity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CostFunction {
    /// Detects changes in mean only.
    ///
    /// Segment cost: `sum((y_i - y_bar)^2)` where `y_bar` is the segment mean.
    /// Equivalent to minus log-likelihood of a normal model with known variance.
    /// One parameter per segment (mean).
    L2,
    /// Detects changes in mean and/or variance.
    ///
    /// Segment cost: `n * ln(sigma^2_MLE)` where `sigma^2_MLE = sum((y_i - y_bar)^2) / n`.
    /// Derived from the negative log-likelihood of a normal distribution
    /// with both mean and variance estimated by MLE.
    /// Two parameters per segment (mean, variance).
    Normal,
}

impl CostFunction {
    /// Number of estimated parameters per segment.
    fn params_per_segment(self) -> usize {
        match self {
            CostFunction::L2 => 1,
            CostFunction::Normal => 2,
        }
    }
}

/// Penalty selection for the PELT algorithm.
#[derive(Debug, Clone, Copy)]
pub enum Penalty {
    /// BIC penalty: `p * ln(n)` where `p` = parameters per segment.
    ///
    /// Reference: Schwarz (1978). Automatically scales with data length.
    Bic,
    /// User-specified penalty value (must be positive and finite).
    Custom(f64),
}

/// PELT changepoint detector.
///
/// Implements the Pruned Exact Linear Time algorithm for detecting
/// multiple changepoints in a univariate time series.
///
/// # Examples
///
/// ```
/// use u_analytics::detection::{Pelt, CostFunction, Penalty};
///
/// // Data with a mean shift at index 50
/// let mut data: Vec<f64> = vec![0.0; 50];
/// data.extend(vec![5.0; 50]);
///
/// let pelt = Pelt::new(CostFunction::L2, Penalty::Bic).unwrap();
/// let result = pelt.detect(&data);
/// assert!(!result.changepoints.is_empty());
/// // The detected changepoint should be near index 50
/// assert!((result.changepoints[0] as i64 - 50).unsigned_abs() <= 2);
/// ```
///
/// # References
///
/// Killick, R., Fearnhead, P., & Eckley, I.A. (2012). "Optimal Detection of
/// Changepoints with a Linear Computational Cost", *JASA* 107(500), pp. 1590-1598.
pub struct Pelt {
    /// Cost function for segment evaluation.
    cost: CostFunction,
    /// Penalty per changepoint.
    penalty: Penalty,
    /// Minimum segment length (must be >= 2 for variance estimation).
    min_segment_len: usize,
}

/// Result of PELT changepoint detection.
#[derive(Debug, Clone)]
pub struct PeltResult {
    /// Detected changepoint indices (0-based). Each index marks the first
    /// observation of a new segment.
    ///
    /// For example, if `changepoints = [50, 100]`, the segments are
    /// `[0..50)`, `[50..100)`, `[100..n)`.
    pub changepoints: Vec<usize>,
}

/// Result of multivariate PELT changepoint detection.
#[derive(Debug, Clone)]
pub struct MultiPeltResult {
    /// Detected changepoint indices (0-based), shared across all channels.
    pub changepoints: Vec<usize>,
}

impl Pelt {
    /// Creates a new PELT detector with the given cost function and penalty.
    ///
    /// Uses a default minimum segment length of 2.
    ///
    /// # Returns
    ///
    /// `None` if a custom penalty is not positive or not finite.
    ///
    /// # Reference
    ///
    /// Killick et al. (2012), §2.2: penalty must be positive to avoid
    /// trivial solutions (changepoint at every observation).
    pub fn new(cost: CostFunction, penalty: Penalty) -> Option<Self> {
        Self::with_min_segment_len(cost, penalty, 2)
    }

    /// Creates a PELT detector with a custom minimum segment length.
    ///
    /// # Parameters
    ///
    /// - `cost`: Cost function for segment evaluation
    /// - `penalty`: Penalty per changepoint
    /// - `min_segment_len`: Minimum number of observations per segment.
    ///   Must be >= 2 (needed for variance estimation in Normal cost).
    ///
    /// # Returns
    ///
    /// `None` if parameters are invalid.
    pub fn with_min_segment_len(
        cost: CostFunction,
        penalty: Penalty,
        min_segment_len: usize,
    ) -> Option<Self> {
        if let Penalty::Custom(p) = penalty {
            if !p.is_finite() || p <= 0.0 {
                return None;
            }
        }
        if min_segment_len < 2 {
            return None;
        }
        Some(Self {
            cost,
            penalty,
            min_segment_len,
        })
    }

    /// Detects changepoints in the given data.
    ///
    /// Returns a [`PeltResult`] containing the detected changepoint indices.
    /// If the data is too short (fewer than `2 * min_segment_len` observations),
    /// no changepoints can be detected and an empty result is returned.
    ///
    /// Non-finite values (NaN, Infinity) are **not** supported in the input.
    /// The data should be pre-cleaned; non-finite values will lead to
    /// incorrect cost computations.
    ///
    /// # Examples
    ///
    /// ```
    /// use u_analytics::detection::{Pelt, CostFunction, Penalty};
    ///
    /// // Two changepoints: shift up at 30, shift down at 70
    /// let mut data = vec![0.0; 30];
    /// data.extend(vec![3.0; 40]);
    /// data.extend(vec![0.0; 30]);
    ///
    /// let pelt = Pelt::new(CostFunction::L2, Penalty::Bic).unwrap();
    /// let result = pelt.detect(&data);
    /// assert_eq!(result.changepoints.len(), 2);
    /// ```
    ///
    /// # Complexity
    ///
    /// Expected O(n), worst case O(n^2).
    pub fn detect(&self, data: &[f64]) -> PeltResult {
        let n = data.len();

        if n < 2 * self.min_segment_len {
            return PeltResult {
                changepoints: Vec::new(),
            };
        }

        let penalty_value = self.resolve_penalty(n);

        // Precompute cumulative sums for O(1) segment cost evaluation.
        // cum_sum[i] = sum(data[0..i])
        // cum_sum_sq[i] = sum(data[0..i]^2)
        let (cum_sum, cum_sum_sq) = cumulative_sums(data);

        // F[t] = optimal cost for data[0..t]
        // F[0] = -penalty (so that the first segment cost + penalty = cost alone)
        let mut f = vec![0.0_f64; n + 1];
        f[0] = -penalty_value;

        // last_change[t] = last changepoint index for optimal segmentation of data[0..t]
        let mut last_change = vec![0_usize; n + 1];

        // Candidate set R: indices tau such that tau could be the last changepoint before t
        let mut candidates: Vec<usize> = vec![0];

        for t in self.min_segment_len..=n {
            // Find the optimal last changepoint for position t
            let mut best_cost = f64::INFINITY;
            let mut best_tau = 0;

            for &tau in &candidates {
                let seg_len = t - tau;
                if seg_len < self.min_segment_len {
                    continue;
                }

                let cost = self.segment_cost(&cum_sum, &cum_sum_sq, tau, t);
                let total = f[tau] + cost + penalty_value;

                if total < best_cost {
                    best_cost = total;
                    best_tau = tau;
                }
            }

            f[t] = best_cost;
            last_change[t] = best_tau;

            // PELT pruning: remove candidates that can never be optimal
            // Killick et al. (2012), Theorem 3.1:
            // If F(tau) + C(y_{tau+1:t}) >= F(t), then tau can be pruned.
            candidates.retain(|&tau| {
                let seg_len = t - tau;
                if seg_len < self.min_segment_len {
                    return true; // Keep — not yet evaluable
                }
                let cost = self.segment_cost(&cum_sum, &cum_sum_sq, tau, t);
                f[tau] + cost < f[t] + penalty_value
            });

            candidates.push(t);
        }

        // Backtrack to extract changepoints
        let mut changepoints = Vec::new();
        let mut t = n;
        while t > 0 {
            let tau = last_change[t];
            if tau > 0 {
                changepoints.push(tau);
            }
            t = tau;
        }

        changepoints.sort_unstable();

        PeltResult { changepoints }
    }

    /// Detects changepoints in multivariate (multi-signal) data.
    ///
    /// Each inner slice represents one signal channel. All channels must
    /// have the same length. The cost function is applied independently
    /// to each channel and summed — a single set of changepoints is
    /// returned that applies to all channels simultaneously.
    ///
    /// The penalty scales with the number of channels: `penalty * n_channels`.
    ///
    /// # Parameters
    ///
    /// - `signals`: slice of signal channels, each of length `n`
    ///
    /// # Returns
    ///
    /// `None` if channels have inconsistent lengths.
    /// Otherwise returns `Some(MultiPeltResult)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use u_analytics::detection::{Pelt, CostFunction, Penalty};
    ///
    /// let signal_a: Vec<f64> = [vec![0.0; 50], vec![5.0; 50]].concat();
    /// let signal_b: Vec<f64> = [vec![0.0; 50], vec![3.0; 50]].concat();
    ///
    /// let pelt = Pelt::new(CostFunction::L2, Penalty::Bic).unwrap();
    /// let result = pelt.detect_multi(&[&signal_a, &signal_b]).unwrap();
    /// assert!(!result.changepoints.is_empty());
    /// ```
    ///
    /// # Reference
    ///
    /// Killick, R. & Eckley, I.A. (2014). "changepoint: An R Package for
    /// Changepoint Analysis", *Journal of Statistical Software* 58(3).
    ///
    /// # Complexity
    ///
    /// Expected O(n * k), worst case O(n^2 * k) where k = number of channels.
    pub fn detect_multi(&self, signals: &[&[f64]]) -> Option<MultiPeltResult> {
        if signals.is_empty() {
            return Some(MultiPeltResult {
                changepoints: Vec::new(),
            });
        }

        let n = signals[0].len();
        if signals.iter().any(|s| s.len() != n) {
            return None;
        }

        if n < 2 * self.min_segment_len {
            return Some(MultiPeltResult {
                changepoints: Vec::new(),
            });
        }

        let n_channels = signals.len();
        let penalty_value = self.resolve_penalty(n) * n_channels as f64;

        // Precompute cumulative sums per channel for O(1) segment cost.
        let mut cum_sums: Vec<Vec<f64>> = Vec::with_capacity(n_channels);
        let mut cum_sum_sqs: Vec<Vec<f64>> = Vec::with_capacity(n_channels);

        for signal in signals {
            let (cs, css) = cumulative_sums(signal);
            cum_sums.push(cs);
            cum_sum_sqs.push(css);
        }

        let mut f = vec![0.0_f64; n + 1];
        f[0] = -penalty_value;
        let mut last_change = vec![0_usize; n + 1];
        let mut candidates: Vec<usize> = vec![0];

        for t in self.min_segment_len..=n {
            let mut best_cost = f64::INFINITY;
            let mut best_tau = 0;

            for &tau in &candidates {
                let seg_len = t - tau;
                if seg_len < self.min_segment_len {
                    continue;
                }

                let cost: f64 = (0..n_channels)
                    .map(|ch| self.segment_cost(&cum_sums[ch], &cum_sum_sqs[ch], tau, t))
                    .sum();
                let total = f[tau] + cost + penalty_value;

                if total < best_cost {
                    best_cost = total;
                    best_tau = tau;
                }
            }

            f[t] = best_cost;
            last_change[t] = best_tau;

            candidates.retain(|&tau| {
                let seg_len = t - tau;
                if seg_len < self.min_segment_len {
                    return true;
                }
                let cost: f64 = (0..n_channels)
                    .map(|ch| self.segment_cost(&cum_sums[ch], &cum_sum_sqs[ch], tau, t))
                    .sum();
                f[tau] + cost < f[t] + penalty_value
            });

            candidates.push(t);
        }

        let mut changepoints = Vec::new();
        let mut t = n;
        while t > 0 {
            let tau = last_change[t];
            if tau > 0 {
                changepoints.push(tau);
            }
            t = tau;
        }
        changepoints.sort_unstable();

        Some(MultiPeltResult { changepoints })
    }

    /// Resolves the penalty value for a dataset of length `n`.
    fn resolve_penalty(&self, n: usize) -> f64 {
        match self.penalty {
            Penalty::Bic => {
                let p = self.cost.params_per_segment() as f64;
                p * (n as f64).ln()
            }
            Penalty::Custom(val) => val,
        }
    }

    /// Computes the cost of the segment `data[start..end]` using cumulative sums.
    ///
    /// # Panics
    ///
    /// Panics if `end <= start`.
    fn segment_cost(&self, cum_sum: &[f64], cum_sum_sq: &[f64], start: usize, end: usize) -> f64 {
        let seg_len = (end - start) as f64;
        let sum = cum_sum[end] - cum_sum[start];
        let sum_sq = cum_sum_sq[end] - cum_sum_sq[start];
        let mean = sum / seg_len;

        match self.cost {
            CostFunction::L2 => {
                // sum((y_i - mean)^2) = sum(y_i^2) - n * mean^2
                sum_sq - seg_len * mean * mean
            }
            CostFunction::Normal => {
                // n * ln(variance) where variance = sum((y_i - mean)^2) / n
                let variance = (sum_sq - seg_len * mean * mean) / seg_len;
                if variance <= 0.0 {
                    // Degenerate segment (constant values): assign a large negative
                    // log-likelihood to make it favorable (perfect fit).
                    seg_len * (f64::MIN_POSITIVE).ln()
                } else {
                    seg_len * variance.ln()
                }
            }
        }
    }
}

/// The cumulative sums of `signal` and of its squares, taken about the
/// signal's mean.
///
/// A segment's cost is `sum(y^2) - n * mean^2` (a difference of two sums),
/// which is unchanged by shifting the signal. Taken about zero, a series that
/// sits far from it (a Unix timestamp in seconds, an instrument with a large
/// offset) loses the variance in that subtraction and is split at nearly every
/// point; about its own mean the two terms are the size of the variation.
fn cumulative_sums(signal: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = signal.len();
    let center = signal.iter().sum::<f64>() / n as f64;
    let mut cum_sum = vec![0.0_f64; n + 1];
    let mut cum_sum_sq = vec![0.0_f64; n + 1];
    for (i, &y) in signal.iter().enumerate() {
        let y = y - center;
        cum_sum[i + 1] = cum_sum[i] + y;
        cum_sum_sq[i + 1] = cum_sum_sq[i] + y * y;
    }
    (cum_sum, cum_sum_sq)
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Constructor validation ---

    #[test]
    fn test_pelt_valid_construction() {
        assert!(Pelt::new(CostFunction::L2, Penalty::Bic).is_some());
        assert!(Pelt::new(CostFunction::Normal, Penalty::Bic).is_some());
        assert!(Pelt::new(CostFunction::L2, Penalty::Custom(10.0)).is_some());
    }

    #[test]
    fn test_pelt_invalid_custom_penalty() {
        assert!(Pelt::new(CostFunction::L2, Penalty::Custom(0.0)).is_none());
        assert!(Pelt::new(CostFunction::L2, Penalty::Custom(-1.0)).is_none());
        assert!(Pelt::new(CostFunction::L2, Penalty::Custom(f64::NAN)).is_none());
        assert!(Pelt::new(CostFunction::L2, Penalty::Custom(f64::INFINITY)).is_none());
    }

    #[test]
    fn test_pelt_invalid_min_segment_len() {
        assert!(Pelt::with_min_segment_len(CostFunction::L2, Penalty::Bic, 0).is_none());
        assert!(Pelt::with_min_segment_len(CostFunction::L2, Penalty::Bic, 1).is_none());
        assert!(Pelt::with_min_segment_len(CostFunction::L2, Penalty::Bic, 2).is_some());
    }

    // --- Empty and short data ---

    #[test]
    fn test_pelt_empty_data() {
        let pelt = Pelt::new(CostFunction::L2, Penalty::Bic).expect("valid");
        let result = pelt.detect(&[]);
        assert!(result.changepoints.is_empty());
    }

    #[test]
    fn test_pelt_too_short_data() {
        let pelt = Pelt::new(CostFunction::L2, Penalty::Bic).expect("valid");
        // min_segment_len=2, need at least 4 points for a changepoint
        let result = pelt.detect(&[1.0, 2.0, 3.0]);
        assert!(result.changepoints.is_empty());
    }

    // --- No changepoint scenarios ---

    #[test]
    fn test_pelt_constant_data_no_changepoint() {
        let pelt = Pelt::new(CostFunction::L2, Penalty::Bic).expect("valid");
        let data = vec![5.0; 100];
        let result = pelt.detect(&data);
        assert!(
            result.changepoints.is_empty(),
            "constant data should have no changepoints, got {:?}",
            result.changepoints
        );
    }

    #[test]
    fn test_pelt_normal_cost_constant_data() {
        let pelt = Pelt::new(CostFunction::Normal, Penalty::Bic).expect("valid");
        let data = vec![5.0; 100];
        let result = pelt.detect(&data);
        assert!(
            result.changepoints.is_empty(),
            "constant data should have no changepoints with Normal cost, got {:?}",
            result.changepoints
        );
    }

    // --- Single changepoint detection ---

    #[test]
    fn test_pelt_single_mean_shift_l2() {
        let pelt = Pelt::new(CostFunction::L2, Penalty::Bic).expect("valid");

        let mut data = vec![0.0; 50];
        data.extend(vec![5.0; 50]);

        let result = pelt.detect(&data);
        assert_eq!(
            result.changepoints.len(),
            1,
            "expected 1 changepoint, got {:?}",
            result.changepoints
        );
        assert!(
            (result.changepoints[0] as i64 - 50).unsigned_abs() <= 2,
            "changepoint should be near index 50, got {}",
            result.changepoints[0]
        );
    }

    #[test]
    fn test_pelt_single_mean_shift_normal() {
        let pelt = Pelt::new(CostFunction::Normal, Penalty::Bic).expect("valid");

        let mut data = vec![0.0; 50];
        data.extend(vec![5.0; 50]);

        let result = pelt.detect(&data);
        assert_eq!(
            result.changepoints.len(),
            1,
            "expected 1 changepoint with Normal cost, got {:?}",
            result.changepoints
        );
        assert!(
            (result.changepoints[0] as i64 - 50).unsigned_abs() <= 2,
            "changepoint should be near index 50, got {}",
            result.changepoints[0]
        );
    }

    // --- Multiple changepoints ---

    #[test]
    fn test_pelt_two_changepoints() {
        let pelt = Pelt::new(CostFunction::L2, Penalty::Bic).expect("valid");

        let mut data = vec![0.0; 40];
        data.extend(vec![5.0; 40]);
        data.extend(vec![0.0; 40]);

        let result = pelt.detect(&data);
        assert_eq!(
            result.changepoints.len(),
            2,
            "expected 2 changepoints, got {:?}",
            result.changepoints
        );

        // Changepoints should be near 40 and 80
        assert!(
            (result.changepoints[0] as i64 - 40).unsigned_abs() <= 2,
            "first changepoint near 40, got {}",
            result.changepoints[0]
        );
        assert!(
            (result.changepoints[1] as i64 - 80).unsigned_abs() <= 2,
            "second changepoint near 80, got {}",
            result.changepoints[1]
        );
    }

    #[test]
    fn test_pelt_three_changepoints() {
        let pelt = Pelt::new(CostFunction::L2, Penalty::Bic).expect("valid");

        let mut data = vec![0.0; 30];
        data.extend(vec![4.0; 30]);
        data.extend(vec![-2.0; 30]);
        data.extend(vec![3.0; 30]);

        let result = pelt.detect(&data);
        assert_eq!(
            result.changepoints.len(),
            3,
            "expected 3 changepoints, got {:?}",
            result.changepoints
        );

        // Check ordering
        for i in 1..result.changepoints.len() {
            assert!(
                result.changepoints[i] > result.changepoints[i - 1],
                "changepoints should be strictly increasing"
            );
        }
    }

    // --- Variance change detection ---

    #[test]
    fn test_pelt_variance_change_normal_cost() {
        // Normal cost should detect a variance change even when mean is constant.
        let pelt = Pelt::new(CostFunction::Normal, Penalty::Bic).expect("valid");

        // Low variance segment then high variance segment
        let mut data = Vec::with_capacity(200);
        // Segment 1: mean=0, low spread (deterministic zigzag)
        for i in 0..100 {
            data.push(if i % 2 == 0 { 0.1 } else { -0.1 });
        }
        // Segment 2: mean=0, high spread
        for i in 0..100 {
            data.push(if i % 2 == 0 { 5.0 } else { -5.0 });
        }

        let result = pelt.detect(&data);
        assert!(
            !result.changepoints.is_empty(),
            "Normal cost should detect variance change"
        );
        // The changepoint should be near index 100
        let cp = result.changepoints[0];
        assert!(
            (cp as i64 - 100).unsigned_abs() <= 5,
            "variance changepoint should be near 100, got {}",
            cp
        );
    }

    // --- Penalty sensitivity ---

    #[test]
    fn test_pelt_higher_penalty_fewer_changepoints() {
        let mut data = vec![0.0; 30];
        data.extend(vec![2.0; 30]);
        data.extend(vec![0.0; 30]);

        let pelt_low = Pelt::new(CostFunction::L2, Penalty::Custom(1.0)).expect("valid");
        let pelt_high = Pelt::new(CostFunction::L2, Penalty::Custom(100.0)).expect("valid");

        let result_low = pelt_low.detect(&data);
        let result_high = pelt_high.detect(&data);

        assert!(
            result_low.changepoints.len() >= result_high.changepoints.len(),
            "higher penalty should produce fewer or equal changepoints: low={}, high={}",
            result_low.changepoints.len(),
            result_high.changepoints.len()
        );
    }

    // --- Custom minimum segment length ---

    #[test]
    fn test_pelt_custom_min_segment_len() {
        let mut data = vec![0.0; 50];
        data.extend(vec![10.0; 50]);

        let pelt = Pelt::with_min_segment_len(CostFunction::L2, Penalty::Bic, 10).expect("valid");
        let result = pelt.detect(&data);
        assert_eq!(
            result.changepoints.len(),
            1,
            "should detect changepoint with min_segment_len=10"
        );

        // All segments should respect minimum length
        let mut boundaries = vec![0];
        boundaries.extend_from_slice(&result.changepoints);
        boundaries.push(data.len());
        for i in 1..boundaries.len() {
            let seg_len = boundaries[i] - boundaries[i - 1];
            assert!(
                seg_len >= 10,
                "segment length {} is less than min_segment_len=10",
                seg_len
            );
        }
    }

    // --- Exact numeric verification ---

    /// Verifies PELT on a simple 4-point example with known optimal solution.
    ///
    /// Data: [0, 0, 10, 10], penalty = 2*ln(4) ≈ 2.77
    ///
    /// Without changepoint: cost = sum((y_i - 5)^2) = 25+25+25+25 = 100
    /// With changepoint at 2: cost = 0 + 0 + 2*ln(4) = 2.77 (L2 cost of each segment = 0)
    ///
    /// PELT should find changepoint at index 2.
    #[test]
    fn test_pelt_exact_small_example() {
        let pelt = Pelt::new(CostFunction::L2, Penalty::Bic).expect("valid");
        let data = [0.0, 0.0, 10.0, 10.0];
        let result = pelt.detect(&data);

        assert_eq!(
            result.changepoints.len(),
            1,
            "expected 1 changepoint in [0,0,10,10], got {:?}",
            result.changepoints
        );
        assert_eq!(
            result.changepoints[0], 2,
            "changepoint should be at index 2"
        );
    }

    // --- Changepoints are sorted ---

    #[test]
    fn test_pelt_changepoints_sorted() {
        let pelt = Pelt::new(CostFunction::L2, Penalty::Bic).expect("valid");

        let mut data = vec![0.0; 25];
        data.extend(vec![5.0; 25]);
        data.extend(vec![-3.0; 25]);
        data.extend(vec![7.0; 25]);

        let result = pelt.detect(&data);
        for i in 1..result.changepoints.len() {
            assert!(
                result.changepoints[i] > result.changepoints[i - 1],
                "changepoints must be strictly increasing: {:?}",
                result.changepoints
            );
        }
    }

    // --- BIC penalty scales with n ---

    #[test]
    fn test_pelt_bic_penalty_scales() {
        let pelt = Pelt::new(CostFunction::L2, Penalty::Bic).expect("valid");

        // BIC for L2: penalty = ln(n). For n=100, penalty ≈ 4.6.
        // A small shift should not be detected with BIC.
        let mut data = vec![0.0; 50];
        data.extend(vec![0.5; 50]); // Very small shift

        let result = pelt.detect(&data);
        // BIC should suppress this tiny shift
        // (0.5^2 * 50 = 12.5 for each segment reduction, but penalty is ~4.6)
        // This depends on exact cost, but a 0.5-unit shift in 100 points
        // may or may not be detected. We just verify it runs without panic.
        let _ = result;
    }

    // --- Property: segments cover entire data ---

    #[test]
    fn test_pelt_segments_cover_data() {
        let pelt = Pelt::new(CostFunction::L2, Penalty::Bic).expect("valid");

        let mut data = vec![0.0; 30];
        data.extend(vec![5.0; 30]);
        data.extend(vec![0.0; 30]);

        let result = pelt.detect(&data);

        // Verify segments cover [0, n) without gaps
        let mut boundaries = vec![0];
        boundaries.extend_from_slice(&result.changepoints);
        boundaries.push(data.len());

        for i in 1..boundaries.len() {
            assert!(
                boundaries[i] > boundaries[i - 1],
                "segments must not have zero length"
            );
        }
        assert_eq!(
            *boundaries.last().expect("non-empty boundaries"),
            data.len(),
            "segments must cover entire data"
        );
    }

    // --- Downward shift ---

    #[test]
    fn test_pelt_downward_shift() {
        let pelt = Pelt::new(CostFunction::L2, Penalty::Bic).expect("valid");

        let mut data = vec![10.0; 50];
        data.extend(vec![2.0; 50]); // Downward shift

        let result = pelt.detect(&data);
        assert_eq!(result.changepoints.len(), 1, "should detect downward shift");
        assert!(
            (result.changepoints[0] as i64 - 50).unsigned_abs() <= 2,
            "changepoint should be near index 50, got {}",
            result.changepoints[0]
        );
    }

    // --- Cost function enum properties ---

    #[test]
    fn test_cost_function_params() {
        assert_eq!(CostFunction::L2.params_per_segment(), 1);
        assert_eq!(CostFunction::Normal.params_per_segment(), 2);
    }

    // --- Multi-signal tests ---

    #[test]
    fn test_pelt_multi_single_channel_matches_univariate() {
        let pelt = Pelt::new(CostFunction::L2, Penalty::Custom(5.0)).expect("valid");
        let mut data = vec![0.0; 50];
        data.extend(vec![5.0; 50]);

        let uni = pelt.detect(&data);
        let multi = pelt.detect_multi(&[&data]).expect("valid");
        assert_eq!(uni.changepoints, multi.changepoints);
    }

    #[test]
    fn test_pelt_multi_two_channels() {
        let pelt = Pelt::new(CostFunction::L2, Penalty::Bic).expect("valid");
        let a: Vec<f64> = [vec![0.0; 50], vec![5.0; 50]].concat();
        let b: Vec<f64> = [vec![0.0; 50], vec![3.0; 50]].concat();

        let result = pelt.detect_multi(&[&a, &b]).expect("valid");
        assert_eq!(
            result.changepoints.len(),
            1,
            "expected 1 changepoint, got {:?}",
            result.changepoints
        );
        assert!(
            (result.changepoints[0] as i64 - 50).unsigned_abs() <= 2,
            "changepoint near 50, got {}",
            result.changepoints[0]
        );
    }

    #[test]
    fn test_pelt_multi_inconsistent_lengths() {
        let pelt = Pelt::new(CostFunction::L2, Penalty::Bic).expect("valid");
        let a = vec![0.0; 50];
        let b = vec![0.0; 30];
        assert!(pelt.detect_multi(&[&a[..], &b[..]]).is_none());
    }

    #[test]
    fn test_pelt_multi_empty_signals() {
        let pelt = Pelt::new(CostFunction::L2, Penalty::Bic).expect("valid");
        let result = pelt.detect_multi(&[]).expect("valid");
        assert!(result.changepoints.is_empty());
    }

    #[test]
    fn test_pelt_multi_three_channels_two_changepoints() {
        let pelt = Pelt::new(CostFunction::L2, Penalty::Bic).expect("valid");
        let a: Vec<f64> = [vec![0.0; 40], vec![5.0; 40], vec![0.0; 40]].concat();
        let b: Vec<f64> = [vec![0.0; 40], vec![3.0; 40], vec![0.0; 40]].concat();
        let c: Vec<f64> = [vec![0.0; 40], vec![4.0; 40], vec![0.0; 40]].concat();

        let result = pelt.detect_multi(&[&a, &b, &c]).expect("valid");
        assert_eq!(
            result.changepoints.len(),
            2,
            "expected 2 changepoints, got {:?}",
            result.changepoints
        );
    }

    #[test]
    fn test_pelt_multi_short_data() {
        let pelt = Pelt::new(CostFunction::L2, Penalty::Bic).expect("valid");
        let a = [1.0, 2.0];
        let b = [3.0, 4.0];
        let result = pelt.detect_multi(&[&a, &b]).expect("valid");
        assert!(result.changepoints.is_empty());
    }
}

/// PELT against the optimum of its own objective -- the exact dynamic program
/// `F(t) = min over tau of F(tau) + C(tau, t) + beta` over every allowed last
/// changepoint, with segment costs from a two-pass mean and sum of squares --
/// written in Python 3 with NumPy 2.4.6. PELT's pruning must not change the
/// answer (Killick et al. 2012, Theorem 3.1).
#[cfg(test)]
mod exact_dp_reference {
    use super::*;

    /// L2 BIC two mean shifts.
    const CASE_0: [f64; 60] = [
        1.02, -1.278, 0.209, -0.284, -0.226, -0.108, -1.01, -0.116, -0.433, 1.661, 0.113, -0.176,
        -0.141, -0.334, -0.528, -0.195, 0.241, -0.119, 0.479, -0.1, 3.012, 3.773, 3.273, 2.747,
        2.909, 3.27, 3.968, 2.865, 2.878, 3.501, 2.557, 2.854, 3.441, 3.29, 3.046, 3.335, 1.586,
        3.511, 2.52, 2.166, 3.138, 3.35, 2.778, 2.462, 3.013, 0.474, 1.203, 0.874, 0.597, 1.056,
        0.397, 0.037, 0.792, 0.791, 0.393, 0.109, 0.615, -0.747, 0.845, 0.746,
    ];

    /// Normal BIC variance + mean shift, min seg 5.
    const CASE_1: [f64; 80] = [
        -0.656, 0.025, -0.386, 0.303, -0.814, -0.366, 0.284, 0.463, -0.863, -0.199, 0.131, -0.244,
        0.636, -0.476, 0.142, -0.419, 0.562, -0.009, -0.149, -0.687, 0.673, 0.301, 0.301, 0.455,
        0.14, -0.256, -0.32, -0.32, 0.548, -0.584, -0.239, -0.128, 0.09, 0.23, -0.5, -0.692,
        -0.002, 0.485, 0.303, 0.086, 0.366, 1.586, 0.513, 2.634, -0.589, 1.268, 0.778, 2.087,
        1.449, 6.1, 3.997, 3.993, -3.079, 0.319, -0.217, 2.065, -3.558, 3.349, 3.134, -1.604,
        -0.957, -0.602, 1.087, 2.282, 5.096, 0.605, 2.535, 1.311, 4.52, 2.484, 3.737, -1.155,
        0.616, -0.628, 4.01, 2.315, 0.39, 0.095, 1.969, -0.403,
    ];

    /// L2 custom penalty 5, min seg 5.
    const CASE_2: [f64; 50] = [
        -0.931, 0.481, 2.463, -0.246, -0.556, -1.171, -1.335, 0.525, 0.851, 0.009, 0.333, 0.116,
        0.139, -1.526, -0.458, 0.111, -0.783, -0.476, -0.819, -0.333, 0.853, -0.407, -0.154, 0.814,
        0.645, 2.895, -0.89, 2.057, 0.718, 1.335, 2.038, 2.283, 2.239, 1.355, 2.81, 0.917, 1.059,
        1.999, 0.649, 3.361, 2.219, 3.376, 1.173, 0.817, 1.367, 1.935, 0.613, 1.58, 1.183, 2.816,
    ];

    /// L2 BIC large offset 1e7.
    const CASE_3: [f64; 60] = [
        9999999.337,
        10000001.046,
        9999999.356,
        9999999.039,
        9999999.29,
        9999998.81,
        10000000.146,
        10000001.031,
        10000000.164,
        10000000.624,
        10000001.632,
        10000000.27,
        10000000.195,
        9999999.725,
        9999998.392,
        10000000.76,
        9999998.244,
        10000000.653,
        9999999.986,
        10000001.127,
        9999999.932,
        9999999.177,
        10000000.358,
        9999999.44,
        9999999.819,
        10000000.042,
        9999999.865,
        9999999.811,
        9999999.168,
        9999999.811,
        10000001.862,
        10000003.843,
        10000002.802,
        10000005.12,
        10000005.27,
        10000002.049,
        10000004.145,
        10000003.874,
        10000002.953,
        10000004.532,
        10000003.538,
        10000002.232,
        10000003.733,
        10000003.852,
        10000004.106,
        10000002.769,
        10000004.616,
        10000004.735,
        10000002.854,
        10000003.341,
        10000003.92,
        10000003.434,
        10000005.744,
        10000004.208,
        10000002.989,
        10000003.212,
        10000003.943,
        10000006.296,
        10000003.822,
        10000004.127,
    ];

    fn detect(cost: CostFunction, penalty: Penalty, min_len: usize, data: &[f64]) -> Vec<usize> {
        Pelt::with_min_segment_len(cost, penalty, min_len)
            .expect("valid settings")
            .detect(data)
            .changepoints
    }

    /// A mean shift at 40 on a level of 1.7e9 (a Unix timestamp in seconds).
    const LARGE_LEVEL: [f64; 80] = [
        1699999999.198,
        1699999998.676,
        1699999999.752,
        1700000000.42,
        1700000001.136,
        1700000000.11,
        1699999999.447,
        1699999999.215,
        1700000000.749,
        1700000001.635,
        1700000000.273,
        1699999998.767,
        1699999999.042,
        1700000001.6,
        1700000000.203,
        1699999998.268,
        1699999999.916,
        1699999998.837,
        1699999999.371,
        1699999999.512,
        1699999999.287,
        1700000000.553,
        1699999999.937,
        1699999999.411,
        1700000000.41,
        1700000000.83,
        1699999998.357,
        1699999999.743,
        1699999999.019,
        1699999999.827,
        1699999998.711,
        1700000000.021,
        1699999999.962,
        1699999999.696,
        1699999998.952,
        1699999999.604,
        1699999998.909,
        1699999998.645,
        1700000000.225,
        1699999998.891,
        1700000006.17,
        1700000005.717,
        1700000003.002,
        1700000005.272,
        1700000003.898,
        1700000005.033,
        1700000005.044,
        1700000003.012,
        1700000004.767,
        1700000004.744,
        1700000005.962,
        1700000003.819,
        1700000005.738,
        1700000003.901,
        1700000004.669,
        1700000004.16,
        1700000006.449,
        1700000005.568,
        1700000007.432,
        1700000005.642,
        1700000005.845,
        1700000005.841,
        1700000004.393,
        1700000004.93,
        1700000006.35,
        1700000004.603,
        1700000005.189,
        1700000004.979,
        1700000005.609,
        1700000004.635,
        1700000004.848,
        1700000005.242,
        1700000005.103,
        1700000004.135,
        1700000005.896,
        1700000003.702,
        1700000003.799,
        1700000003.718,
        1700000005.967,
        1700000004.639,
    ];

    #[test]
    fn a_large_level_does_not_move_the_optimum() {
        // The same series minus 1.7e9: the segment costs are shift-invariant,
        // so the changepoints must be the same, [40, 56, 62] by the dynamic program.
        assert_eq!(
            detect(CostFunction::L2, Penalty::Bic, 2, &LARGE_LEVEL),
            vec![40, 56, 62]
        );
    }

    #[test]
    fn pruned_search_returns_the_exact_optimum() {
        // L2 BIC two mean shifts
        assert_eq!(
            detect(CostFunction::L2, Penalty::Bic, 2, &CASE_0),
            vec![20, 45]
        );
        // Normal BIC variance + mean shift, min seg 5
        assert_eq!(
            detect(CostFunction::Normal, Penalty::Bic, 5, &CASE_1),
            vec![41]
        );
        // L2 custom penalty 5, min seg 5
        assert_eq!(
            detect(CostFunction::L2, Penalty::Custom(5.0), 5, &CASE_2),
            vec![25]
        );
        // L2 BIC large offset 1e7
        assert_eq!(detect(CostFunction::L2, Penalty::Bic, 2, &CASE_3), vec![31]);
    }
}
