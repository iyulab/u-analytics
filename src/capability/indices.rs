//! Process capability indices (Cp, Cpk, Pp, Ppk, Cpm).
//!
//! Process capability indices quantify how well a process output fits within
//! specification limits. Short-term indices (Cp, Cpk) use within-group
//! variation, while long-term indices (Pp, Ppk) use overall variation.
//!
//! # References
//!
//! - Montgomery, D.C. (2020). *Introduction to Statistical Quality Control*, 8th ed.,
//!   §7.1–7.3, §8.
//! - Kane (1986), "Process Capability Indices", *Journal of Quality Technology*
//!   18(1), pp. 41--52.
//! - Chan, Cheng & Spiring (1988), "A New Measure of Process Capability: Cpm",
//!   *Journal of Quality Technology* 20(3), pp. 162--175.
//! - Bissell, A.F. (1990), "How Reliable is Your Capability Index?",
//!   *Applied Statistics* 39(3), pp. 331--340 -- the Cpk interval.
//! - Pearn, W.L. & Kotz, S. (2006), *Encyclopedia and Handbook of Process
//!   Capability Indices*, World Scientific -- the Cp and Cpk intervals.

use u_numflow::special::{chi_squared_quantile, inverse_normal_cdf};
use u_numflow::stats;

/// Input specification for process capability analysis.
///
/// Defines the upper and/or lower specification limits and an optional target
/// value. At least one specification limit must be provided.
///
/// # Examples
///
/// ```
/// use u_analytics::capability::ProcessCapability;
///
/// // Two-sided specification: LSL = 9.0, USL = 11.0
/// let spec = ProcessCapability::new(Some(11.0), Some(9.0)).unwrap();
///
/// let data = [9.5, 10.0, 10.2, 9.8, 10.1, 10.3, 9.9, 10.0];
/// let indices = spec.compute(&data, 0.15).unwrap();
/// assert!(indices.cp.is_some());
/// assert!(indices.cpk.is_some());
/// ```
pub struct ProcessCapability {
    usl: Option<f64>,
    lsl: Option<f64>,
    target: Option<f64>,
}

/// Computed capability indices.
///
/// Fields are `Option<f64>` because not every index can be computed for
/// every input. Cp requires both USL and LSL; the short-term quartet
/// (`cp`, `cpk`, `cpu`, `cpl`) requires a within-subgroup sigma, so it is
/// `None` from [`ProcessCapability::compute_overall`], which has none. A
/// long-term number is never reported under a short-term name.
///
/// # Index interpretation
///
/// | Index | Value | Interpretation |
/// |-------|-------|----------------|
/// | Cp/Pp | >= 1.33 | Process is capable |
/// | Cpk/Ppk | >= 1.33 | Process is capable and centered |
/// | Cpm | >= 1.33 | Process meets Taguchi loss criterion |
///
/// Reference: Montgomery (2020), Chapter 8, Table 8.5.
#[derive(Debug, Clone)]
pub struct CapabilityIndices {
    /// Cp = (USL - LSL) / (6 * sigma_within). Requires both limits.
    pub cp: Option<f64>,
    /// Cpk = min(Cpu, Cpl). Requires at least one limit.
    pub cpk: Option<f64>,
    /// Cpu = (USL - mean) / (3 * sigma_within). Requires USL.
    pub cpu: Option<f64>,
    /// Cpl = (mean - LSL) / (3 * sigma_within). Requires LSL.
    pub cpl: Option<f64>,
    /// Pp = (USL - LSL) / (6 * sigma_overall). Requires both limits.
    pub pp: Option<f64>,
    /// Ppk = min(Ppu, Ppl). Requires at least one limit.
    pub ppk: Option<f64>,
    /// Ppu = (USL - mean) / (3 * sigma_overall). Requires USL.
    pub ppu: Option<f64>,
    /// Ppl = (mean - LSL) / (3 * sigma_overall). Requires LSL.
    pub ppl: Option<f64>,
    /// Cpm = min(T - LSL, USL - T) / (3 * sigma_t), where
    /// sigma_t = sqrt(sum((x_i - T)^2) / (n - 1)) is the spread of every
    /// observation about the target T. With T at the specification midpoint
    /// the numerator is (USL - LSL) / 6, the form Chan, Cheng & Spiring (1988)
    /// give.
    ///
    /// `None` unless both limits and a target are set, the target lies within
    /// the limits, and sigma_t is positive. Cpm measures how closely the
    /// process clusters about the target, so it depends on neither sigma
    /// estimate above and is the same from [`ProcessCapability::compute`] and
    /// [`ProcessCapability::compute_overall`].
    ///
    /// References: Chan, Cheng & Spiring (1988); Minitab, *Capability Analysis
    /// (Normal) Formulas*, for the numerator when T is not the midpoint.
    pub cpm: Option<f64>,
    /// Sample mean of the data.
    pub mean: f64,
    /// Short-term (within-group) standard deviation -- the value the caller
    /// supplied to [`ProcessCapability::compute`]. `None` from
    /// [`ProcessCapability::compute_overall`], along with the four indices
    /// that would be computed from it.
    pub std_dev_within: Option<f64>,
    /// Long-term (overall) standard deviation.
    pub std_dev_overall: f64,
    /// Number of observations the indices were computed from.
    pub n: usize,
}

/// A two-sided confidence interval.
///
/// The lower limit of a two-sided interval at confidence `1 - 2a` is the
/// one-sided lower bound at `1 - a`: a customer requirement "Cpk lower bound at
/// 95 % >= 1.33" is the `lower` of the interval at `confidence = 0.90`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConfidenceInterval {
    /// Lower confidence limit.
    pub lower: f64,
    /// Upper confidence limit.
    pub upper: f64,
}

/// Confidence intervals for the capability indices -- see
/// [`CapabilityIndices::intervals`].
///
/// An index's interval is `None` when the index is, and the short-term pair
/// (`cp`, `cpk`) is `None` unless the degrees of freedom of the within sigma
/// were given: they depend on how that sigma was estimated, which the indices
/// do not record.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CapabilityIntervals {
    /// Confidence level of every interval, in (0, 1).
    pub confidence: f64,
    /// Cp, from the chi-square distribution of the within variance.
    pub cp: Option<ConfidenceInterval>,
    /// Cpk, by Bissell's normal approximation.
    pub cpk: Option<ConfidenceInterval>,
    /// Pp, from the chi-square distribution of the sample variance (n - 1 df).
    pub pp: Option<ConfidenceInterval>,
    /// Ppk, by Bissell's normal approximation (n - 1 df).
    pub ppk: Option<ConfidenceInterval>,
}

/// Why [`CapabilityIndices::intervals`] refused its arguments.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IntervalError {
    /// `confidence` is not strictly between 0 and 1.
    ConfidenceOutOfRange,
    /// The within degrees of freedom are not a positive, finite number.
    WithinDfOutOfRange,
    /// Within degrees of freedom were given, but the indices have no within sigma.
    WithinDfWithoutWithinSigma,
}

impl CapabilityIndices {
    /// Two-sided confidence intervals for Cp, Cpk, Pp and Ppk at `confidence`.
    ///
    /// For a ratio of the specification width to `k` sigma estimated with `df`
    /// degrees of freedom (Cp, Pp):
    ///
    /// ```text
    /// C * sqrt(chi2(a/2; df) / df)  <=  C  <=  C * sqrt(chi2(1 - a/2; df) / df)
    /// ```
    ///
    /// For the nearer-limit index (Cpk, Ppk), Bissell's (1990) approximation:
    ///
    /// ```text
    /// Ck -/+ z(1 - a/2) * sqrt(1 / (9 n) + Ck^2 / (2 df))
    /// ```
    ///
    /// with `a = 1 - confidence`, `n` the number of observations (the mean's
    /// share) and `df` the degrees of freedom of the sigma (its share). The
    /// overall sigma is the sample standard deviation, so Pp/Ppk use
    /// `n - 1`. The within sigma's degrees of freedom depend on how it was
    /// estimated -- `n - m` for a pooled standard deviation of `m` subgroups,
    /// fewer for R-bar/d2 or a moving range (about `0.62 (n - 1)` for MR-bar/d2,
    /// Wheeler 2004) -- so Cp/Cpk intervals are given only when `within_df` is.
    ///
    /// # Errors
    ///
    /// [`IntervalError`] when `confidence` is not in (0, 1), `within_df` is
    /// not positive and finite, or `within_df` is given for indices computed
    /// without a within sigma.
    ///
    /// # Examples
    ///
    /// ```
    /// use u_analytics::capability::ProcessCapability;
    ///
    /// let spec = ProcessCapability::new(Some(11.0), Some(9.0)).unwrap();
    /// let data = [9.5, 10.0, 10.2, 9.8, 10.1, 10.3, 9.9, 10.0, 10.4, 9.7];
    /// let indices = spec.compute_overall(&data).unwrap();
    ///
    /// // A one-sided 95 % lower bound is the lower limit of the 90 % interval.
    /// let ci = indices.intervals(0.90, None).unwrap();
    /// let ppk = ci.ppk.unwrap();
    /// assert!(ppk.lower < indices.ppk.unwrap() && indices.ppk.unwrap() < ppk.upper);
    /// assert!(ci.cpk.is_none()); // no within sigma
    /// ```
    pub fn intervals(
        &self,
        confidence: f64,
        within_df: Option<f64>,
    ) -> Result<CapabilityIntervals, IntervalError> {
        if !(confidence > 0.0 && confidence < 1.0) {
            return Err(IntervalError::ConfidenceOutOfRange);
        }
        if let Some(df) = within_df {
            if !(df.is_finite() && df > 0.0) {
                return Err(IntervalError::WithinDfOutOfRange);
            }
            if self.std_dev_within.is_none() {
                return Err(IntervalError::WithinDfWithoutWithinSigma);
            }
        }
        let alpha = 1.0 - confidence;
        let n = self.n as f64;
        let overall_df = n - 1.0;
        let ratio = |c: f64, df: f64| ConfidenceInterval {
            lower: c * (chi_squared_quantile(alpha / 2.0, df) / df).sqrt(),
            upper: c * (chi_squared_quantile(1.0 - alpha / 2.0, df) / df).sqrt(),
        };
        let z = inverse_normal_cdf(1.0 - alpha / 2.0);
        let nearer = |c: f64, df: f64| {
            let half = z * (1.0 / (9.0 * n) + c * c / (2.0 * df)).sqrt();
            ConfidenceInterval {
                lower: c - half,
                upper: c + half,
            }
        };
        Ok(CapabilityIntervals {
            confidence,
            cp: within_df.and_then(|df| self.cp.map(|c| ratio(c, df))),
            cpk: within_df.and_then(|df| self.cpk.map(|c| nearer(c, df))),
            pp: self.pp.map(|c| ratio(c, overall_df)),
            ppk: self.ppk.map(|c| nearer(c, overall_df)),
        })
    }
}

impl ProcessCapability {
    /// Creates a new process capability specification.
    ///
    /// At least one of `usl` or `lsl` must be `Some`. If both are provided,
    /// `usl` must be greater than `lsl`.
    ///
    /// # Errors
    ///
    /// Returns an error string if:
    /// - Both `usl` and `lsl` are `None`
    /// - `usl <= lsl` when both are provided
    /// - Either limit is non-finite (NaN or infinity)
    ///
    /// # Examples
    ///
    /// ```
    /// use u_analytics::capability::ProcessCapability;
    ///
    /// // Two-sided
    /// let spec = ProcessCapability::new(Some(10.0), Some(5.0)).unwrap();
    ///
    /// // Upper limit only
    /// let spec = ProcessCapability::new(Some(10.0), None).unwrap();
    ///
    /// // Lower limit only
    /// let spec = ProcessCapability::new(None, Some(5.0)).unwrap();
    ///
    /// // Error: no limits
    /// assert!(ProcessCapability::new(None, None).is_err());
    ///
    /// // Error: USL <= LSL
    /// assert!(ProcessCapability::new(Some(5.0), Some(10.0)).is_err());
    /// ```
    pub fn new(usl: Option<f64>, lsl: Option<f64>) -> Result<Self, &'static str> {
        if usl.is_none() && lsl.is_none() {
            return Err("at least one specification limit (USL or LSL) is required");
        }
        if let Some(u) = usl {
            if !u.is_finite() {
                return Err("USL must be finite");
            }
        }
        if let Some(l) = lsl {
            if !l.is_finite() {
                return Err("LSL must be finite");
            }
        }
        if let (Some(u), Some(l)) = (usl, lsl) {
            if u <= l {
                return Err("USL must be greater than LSL");
            }
        }
        Ok(Self {
            usl,
            lsl,
            target: None,
        })
    }

    /// Sets the target value for Cpm.
    ///
    /// Cpm is computed only against a declared target. Without one it is
    /// `None` rather than measured against the specification midpoint: a
    /// substituted target cannot be told apart from a declared one in the
    /// result. A process whose target is the midpoint passes it here.
    ///
    /// # Examples
    ///
    /// ```
    /// use u_analytics::capability::ProcessCapability;
    ///
    /// let spec = ProcessCapability::new(Some(11.0), Some(9.0))
    ///     .unwrap()
    ///     .with_target(10.0);
    /// ```
    pub fn with_target(mut self, target: f64) -> Self {
        self.target = Some(target);
        self
    }

    /// Computes all capability indices using the provided within-group sigma.
    ///
    /// The `sigma_within` parameter represents the short-term (within-group)
    /// standard deviation, typically estimated from a control chart as R-bar/d2
    /// or S-bar/c4.
    ///
    /// The overall (long-term) standard deviation is computed from the data
    /// using the sample standard deviation.
    ///
    /// # Returns
    ///
    /// `None` if:
    /// - `data` has fewer than 2 elements
    /// - `sigma_within` is not positive or not finite
    /// - `data` contains NaN or infinity values
    ///
    /// # Examples
    ///
    /// ```
    /// use u_analytics::capability::ProcessCapability;
    ///
    /// let spec = ProcessCapability::new(Some(11.0), Some(9.0)).unwrap();
    /// let data = [9.5, 10.0, 10.2, 9.8, 10.1, 10.3, 9.9, 10.0];
    /// let sigma_within = 0.15; // from control chart
    ///
    /// let indices = spec.compute(&data, sigma_within).unwrap();
    /// assert!(indices.cp.unwrap() > 0.0);
    /// assert!(indices.cpk.unwrap() > 0.0);
    /// assert!(indices.pp.unwrap() > 0.0);
    /// assert!(indices.ppk.unwrap() > 0.0);
    /// ```
    pub fn compute(&self, data: &[f64], sigma_within: f64) -> Option<CapabilityIndices> {
        if !sigma_within.is_finite() || sigma_within <= 0.0 {
            return None;
        }
        let x_bar = stats::mean(data)?;
        let sigma_overall = stats::std_dev(data)?;

        Some(self.compute_indices(data, x_bar, Some(sigma_within), sigma_overall))
    }

    /// Computes the long-term (performance) indices only.
    ///
    /// Use this when no within-subgroup sigma is available -- a flat vector
    /// with no rational subgrouping. Pp/Ppk/Ppu/Ppl come from the overall
    /// sample standard deviation, and Cpm from the spread about the target,
    /// as in [`ProcessCapability::compute`]. The short-term quartet
    /// (`cp`, `cpk`, `cpu`, `cpl`) and `std_dev_within` are `None`: filling
    /// them from the overall sigma would make `cp` equal `pp` for every
    /// input, a long-term number under a short-term name. To report Cp/Cpk,
    /// estimate the within sigma from a control chart (R-bar/d2, S-bar/c4 or
    /// MR-bar/d2) and call [`ProcessCapability::compute`].
    ///
    /// # Returns
    ///
    /// `None` if:
    /// - `data` has fewer than 2 elements
    /// - `data` contains NaN or infinity values
    ///
    /// # Examples
    ///
    /// ```
    /// use u_analytics::capability::ProcessCapability;
    ///
    /// let spec = ProcessCapability::new(Some(11.0), Some(9.0)).unwrap();
    /// let data = [9.5, 10.0, 10.2, 9.8, 10.1, 10.3, 9.9, 10.0];
    ///
    /// let indices = spec.compute_overall(&data).unwrap();
    /// assert!(indices.pp.unwrap() > 0.0);
    /// // No within sigma, so no short-term index is claimed.
    /// assert!(indices.cp.is_none());
    /// assert!(indices.std_dev_within.is_none());
    /// ```
    pub fn compute_overall(&self, data: &[f64]) -> Option<CapabilityIndices> {
        let x_bar = stats::mean(data)?;
        let sigma_overall = stats::std_dev(data)?;

        Some(self.compute_indices(data, x_bar, None, sigma_overall))
    }

    /// Internal computation of all indices given the data, its mean, the
    /// overall sigma and -- when the caller has one -- the within sigma.
    fn compute_indices(
        &self,
        data: &[f64],
        x_bar: f64,
        sigma_within: Option<f64>,
        sigma_overall: f64,
    ) -> CapabilityIndices {
        // Short-term indices (within-group sigma) -- only with a within sigma.
        let cpu = sigma_within.and_then(|s| self.usl.map(|u| (u - x_bar) / (3.0 * s)));
        let cpl = sigma_within.and_then(|s| self.lsl.map(|l| (x_bar - l) / (3.0 * s)));
        let cp = match (sigma_within, self.usl, self.lsl) {
            (Some(s), Some(u), Some(l)) => Some((u - l) / (6.0 * s)),
            _ => None,
        };
        let cpk = match (cpu, cpl) {
            (Some(u), Some(l)) => Some(u.min(l)),
            (Some(u), None) => Some(u),
            (None, Some(l)) => Some(l),
            (None, None) => None,
        };

        // Long-term indices (overall sigma)
        let ppu = self.usl.map(|u| (u - x_bar) / (3.0 * sigma_overall));
        let ppl = self.lsl.map(|l| (x_bar - l) / (3.0 * sigma_overall));
        let pp = match (self.usl, self.lsl) {
            (Some(u), Some(l)) => Some((u - l) / (6.0 * sigma_overall)),
            _ => None,
        };
        let ppk = match (ppu, ppl) {
            (Some(u), Some(l)) => Some(u.min(l)),
            (Some(u), None) => Some(u),
            (None, Some(l)) => Some(l),
            (None, None) => None,
        };

        let cpm = self.cpm(data);

        CapabilityIndices {
            cp,
            cpk,
            cpu,
            cpl,
            pp,
            ppk,
            ppu,
            ppl,
            cpm,
            mean: x_bar,
            std_dev_within: sigma_within,
            std_dev_overall: sigma_overall,
            n: data.len(),
        }
    }

    /// Cpm against the declared target -- see [`CapabilityIndices::cpm`].
    ///
    /// The spread is taken about T directly rather than assembled from a sigma
    /// and the offset of the mean: `sum((x_i - T)^2) / (n - 1)` is the
    /// estimator Chan, Cheng & Spiring (1988) give, and it differs from
    /// `s^2 + (mean - T)^2` by a factor of n / (n - 1) on the offset term.
    fn cpm(&self, data: &[f64]) -> Option<f64> {
        let (usl, lsl, target) = (self.usl?, self.lsl?, self.target?);
        if !target.is_finite() || target < lsl || target > usl || data.len() < 2 {
            return None;
        }
        let sum_sq: f64 = data.iter().map(|&x| (x - target).powi(2)).sum();
        let sigma_t = (sum_sq / (data.len() - 1) as f64).sqrt();
        // Every observation on the target leaves no spread to divide by.
        if sigma_t <= 0.0 {
            return None;
        }
        Some((target - lsl).min(usl - target) / (3.0 * sigma_t))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Confidence intervals
    // -----------------------------------------------------------------------

    /// Indices with chosen values, for checking the interval arithmetic
    /// against hand-worked table values.
    fn indices(n: usize, c: f64, within: bool) -> CapabilityIndices {
        CapabilityIndices {
            cp: within.then_some(c),
            cpk: within.then_some(c),
            cpu: None,
            cpl: None,
            pp: Some(c),
            ppk: Some(c),
            ppu: None,
            ppl: None,
            cpm: None,
            mean: 0.0,
            std_dev_within: within.then_some(1.0),
            std_dev_overall: 1.0,
            n,
        }
    }

    /// n = 30, an index of 1.33, 95 %: chi-square table values for 29 df
    /// (16.047, 45.722) and z = 1.960 give the published-arithmetic limits.
    #[test]
    fn intervals_match_hand_worked_values() {
        let ci = indices(30, 1.33, false).intervals(0.95, None).unwrap();
        let pp = ci.pp.unwrap();
        assert!(
            (pp.lower - 1.33 * (16.047_f64 / 29.0).sqrt()).abs() < 1e-3,
            "{pp:?}"
        );
        assert!(
            (pp.upper - 1.33 * (45.722_f64 / 29.0).sqrt()).abs() < 1e-3,
            "{pp:?}"
        );
        let ppk = ci.ppk.unwrap();
        let half = 1.959964 * (1.0 / 270.0 + 1.33_f64.powi(2) / 58.0).sqrt();
        assert!((ppk.lower - (1.33 - half)).abs() < 1e-6, "{ppk:?}");
        assert!((ppk.upper - (1.33 + half)).abs() < 1e-6, "{ppk:?}");
        assert!((ppk.lower - 0.9675).abs() < 1e-3, "{ppk:?}");
    }

    /// The short-term pair needs the within degrees of freedom; given them, it
    /// uses them in place of n - 1.
    #[test]
    fn short_term_intervals_need_the_within_df() {
        let i = indices(50, 1.5, true);
        let without = i.intervals(0.95, None).unwrap();
        assert!(without.cp.is_none() && without.cpk.is_none());
        assert!(without.pp.is_some() && without.ppk.is_some());

        let with = i.intervals(0.95, Some(30.0)).unwrap();
        let (cp, pp) = (with.cp.unwrap(), with.pp.unwrap());
        // Fewer degrees of freedom: a wider interval about the same value.
        assert!(cp.upper - cp.lower > pp.upper - pp.lower);
        assert!((cp.lower - 1.5 * (chi_squared_quantile(0.025, 30.0) / 30.0).sqrt()).abs() < 1e-12);
    }

    #[test]
    fn interval_arguments_are_checked() {
        let overall = indices(20, 1.2, false);
        for c in [0.0, 1.0, -0.5, 1.5, f64::NAN] {
            assert_eq!(
                overall.intervals(c, None),
                Err(IntervalError::ConfidenceOutOfRange)
            );
        }
        assert_eq!(
            overall.intervals(0.95, Some(10.0)),
            Err(IntervalError::WithinDfWithoutWithinSigma)
        );
        let within = indices(20, 1.2, true);
        for df in [0.0, -1.0, f64::INFINITY, f64::NAN] {
            assert_eq!(
                within.intervals(0.95, Some(df)),
                Err(IntervalError::WithinDfOutOfRange)
            );
        }
    }

    /// Higher confidence, wider interval; more data, narrower.
    #[test]
    fn intervals_widen_with_confidence_and_narrow_with_n() {
        let width = |n: usize, conf: f64| {
            let ppk = indices(n, 1.33, false)
                .intervals(conf, None)
                .unwrap()
                .ppk
                .unwrap();
            ppk.upper - ppk.lower
        };
        assert!(width(30, 0.99) > width(30, 0.95));
        assert!(width(30, 0.95) > width(30, 0.90));
        assert!(width(100, 0.95) < width(30, 0.95));
    }

    // -----------------------------------------------------------------------
    // Construction tests
    // -----------------------------------------------------------------------

    #[test]
    fn new_requires_at_least_one_limit() {
        assert!(ProcessCapability::new(None, None).is_err());
    }

    #[test]
    fn new_rejects_usl_leq_lsl() {
        assert!(ProcessCapability::new(Some(5.0), Some(10.0)).is_err());
        assert!(ProcessCapability::new(Some(5.0), Some(5.0)).is_err());
    }

    #[test]
    fn new_rejects_non_finite() {
        assert!(ProcessCapability::new(Some(f64::NAN), Some(1.0)).is_err());
        assert!(ProcessCapability::new(Some(10.0), Some(f64::INFINITY)).is_err());
    }

    #[test]
    fn new_accepts_valid_two_sided() {
        assert!(ProcessCapability::new(Some(10.0), Some(5.0)).is_ok());
    }

    #[test]
    fn new_accepts_usl_only() {
        assert!(ProcessCapability::new(Some(10.0), None).is_ok());
    }

    #[test]
    fn new_accepts_lsl_only() {
        assert!(ProcessCapability::new(None, Some(5.0)).is_ok());
    }

    // -----------------------------------------------------------------------
    // Computation -- textbook example
    // -----------------------------------------------------------------------

    /// Textbook example: Montgomery (2019), Example 8.1
    ///
    /// LSL = 200, USL = 220, no target declared
    /// Process mean ~ 210, sigma_within = 2.0
    ///
    /// Cp = (220 - 200) / (6 * 2) = 20/12 = 1.6667
    #[test]
    fn textbook_centered_process() {
        let spec = ProcessCapability::new(Some(220.0), Some(200.0)).unwrap();

        let data = [
            208.0, 209.0, 210.0, 211.0, 212.0, 208.5, 209.5, 210.5, 211.5, 210.0, 209.0, 211.0,
            210.0, 209.5, 210.5, 210.0, 210.0, 210.0, 209.0, 211.0,
        ];

        let sigma_within = 2.0;
        let indices = spec.compute(&data, sigma_within).unwrap();

        // Cp = (220 - 200) / (6 * 2) = 1.6667
        let cp = indices.cp.unwrap();
        assert!(
            (cp - 1.6667).abs() < 0.001,
            "expected Cp ~ 1.6667, got {cp}"
        );

        let cpk = indices.cpk.unwrap();
        assert!(cpk > 0.0, "Cpk should be positive");

        // No target was declared, so there is no Cpm -- the midpoint is not
        // assumed on the caller's behalf.
        assert!(indices.cpm.is_none());
    }

    /// Off-center process: mean shifted toward USL.
    ///
    /// LSL = 200, USL = 220, mean ~ 215, sigma_within = 2.0
    /// Cp = (220 - 200) / (6 * 2) = 1.6667
    /// Cpu = (220 - 215) / (3 * 2) = 0.8333
    /// Cpl = (215 - 200) / (3 * 2) = 2.5
    /// Cpk = min(0.8333, 2.5) = 0.8333
    #[test]
    fn off_center_process() {
        let spec = ProcessCapability::new(Some(220.0), Some(200.0)).unwrap();

        let data = [
            213.0, 214.0, 215.0, 216.0, 217.0, 213.5, 214.5, 215.5, 216.5, 215.0, 214.0, 216.0,
            215.0, 214.5, 215.5, 215.0, 215.0, 215.0, 214.0, 216.0,
        ];

        let sigma_within = 2.0;
        let indices = spec.compute(&data, sigma_within).unwrap();

        let cp = indices.cp.unwrap();
        assert!(
            (cp - 1.6667).abs() < 0.001,
            "expected Cp ~ 1.6667, got {cp}"
        );

        let cpu = indices.cpu.unwrap();
        assert!(
            (cpu - 0.8333).abs() < 0.05,
            "expected Cpu ~ 0.8333, got {cpu}"
        );

        let cpl = indices.cpl.unwrap();
        assert!((cpl - 2.5).abs() < 0.05, "expected Cpl ~ 2.5, got {cpl}");

        let cpk = indices.cpk.unwrap();
        assert!((cpk - cpu).abs() < 1e-15, "Cpk should equal min(Cpu, Cpl)");
    }

    // -----------------------------------------------------------------------
    // One-sided specifications
    // -----------------------------------------------------------------------

    #[test]
    fn usl_only_computes_cpu_not_cpl() {
        let spec = ProcessCapability::new(Some(10.0), None).unwrap();
        let data = [7.0, 8.0, 9.0, 7.5, 8.5, 8.0, 7.0, 9.0, 8.0, 8.5];
        let indices = spec.compute(&data, 0.5).unwrap();

        assert!(indices.cpu.is_some());
        assert!(indices.cpl.is_none());
        assert!(indices.cp.is_none(), "Cp requires both limits");
        assert!(indices.cpk.is_some(), "Cpk should be Cpu for USL-only");
        assert!(
            (indices.cpk.unwrap() - indices.cpu.unwrap()).abs() < 1e-15,
            "Cpk == Cpu when only USL is set"
        );
    }

    #[test]
    fn lsl_only_computes_cpl_not_cpu() {
        let spec = ProcessCapability::new(None, Some(5.0)).unwrap();
        let data = [7.0, 8.0, 9.0, 7.5, 8.5, 8.0, 7.0, 9.0, 8.0, 8.5];
        let indices = spec.compute(&data, 0.5).unwrap();

        assert!(indices.cpl.is_some());
        assert!(indices.cpu.is_none());
        assert!(indices.cp.is_none());
        assert!(indices.cpk.is_some(), "Cpk should be Cpl for LSL-only");
        assert!(
            (indices.cpk.unwrap() - indices.cpl.unwrap()).abs() < 1e-15,
            "Cpk == Cpl when only LSL is set"
        );
    }

    // -----------------------------------------------------------------------
    // Overall-only computation
    // -----------------------------------------------------------------------

    #[test]
    fn compute_overall_reports_no_short_term_index() {
        // The previous version of this test asserted cp == pp here, pinning
        // the overall sigma being reported under the short-term names. A
        // flat vector has no within sigma, so nothing short-term is claimed.
        let spec = ProcessCapability::new(Some(220.0), Some(200.0)).unwrap();
        let data = [
            208.0, 209.0, 210.0, 211.0, 212.0, 208.5, 209.5, 210.5, 211.5, 210.0,
        ];
        let indices = spec.compute_overall(&data).unwrap();

        assert!(indices.cp.is_none());
        assert!(indices.cpk.is_none());
        assert!(indices.cpu.is_none());
        assert!(indices.cpl.is_none());
        assert!(indices.std_dev_within.is_none());
        assert!(indices.pp.unwrap() > 0.0);
        assert!(indices.ppk.unwrap() > 0.0);
        assert!(indices.ppu.unwrap() > 0.0);
        assert!(indices.ppl.unwrap() > 0.0);

        // The long-term indices are the same ones `compute` reports.
        let with_within = spec.compute(&data, 1.0).unwrap();
        assert_eq!(indices.pp, with_within.pp);
        assert_eq!(indices.ppk, with_within.ppk);
        assert_eq!(indices.std_dev_overall, with_within.std_dev_overall);
        assert_eq!(with_within.std_dev_within, Some(1.0));
    }

    // -----------------------------------------------------------------------
    // Cpm with explicit target
    // -----------------------------------------------------------------------

    /// The case reported against 0.8.0, which derived Cpm from Cp: with no
    /// target it reported 2.8793 against the midpoint, and 0.7361 against a
    /// target of 11 -- both from the within sigma of 0.2.
    ///
    /// Hand computation, sigma_t = sqrt(sum((x - T)^2) / 5):
    /// - T = 10: sum = 0.35, sigma_t = 0.264575, Cpm = (4/6) / sigma_t = 2.519763
    /// - T = 11: sum = 4.95, sigma_t = 0.994987, Cpm = (1/3) / sigma_t = 0.335013
    #[test]
    fn cpm_is_the_spread_about_a_declared_target() {
        let data = [9.8, 10.1, 10.3, 9.9, 10.2, 10.4];
        let spec = || ProcessCapability::new(Some(12.0), Some(8.0)).unwrap();

        assert!(spec().compute(&data, 0.2).unwrap().cpm.is_none());
        assert!(spec().compute_overall(&data).unwrap().cpm.is_none());

        for (target, expected) in [(10.0, 2.519_763), (11.0, 0.335_013)] {
            let spec = spec().with_target(target);
            let within = spec.compute(&data, 0.2).unwrap().cpm.unwrap();
            let overall = spec.compute_overall(&data).unwrap().cpm.unwrap();
            assert!((within - expected).abs() < 1e-6, "T={target}: {within}");
            // No sigma enters Cpm, so the two entry points agree exactly.
            assert_eq!(within, overall);
        }
    }

    /// T off the midpoint: the numerator is the distance to the nearer limit.
    ///
    /// LSL = 200, USL = 220, T = 212: min(12, 8) / 3 = 8/3.
    /// sum((x - 212)^2) = 16 + 9 + 4 + 1 + 0 = 30, sigma_t = sqrt(7.5).
    /// Cpm = (8/3) / 2.738613 = 0.973729.
    #[test]
    fn cpm_off_midpoint_uses_the_nearer_limit() {
        let spec = ProcessCapability::new(Some(220.0), Some(200.0))
            .unwrap()
            .with_target(212.0);
        let data = [208.0, 209.0, 210.0, 211.0, 212.0];
        let cpm = spec.compute_overall(&data).unwrap().cpm.unwrap();
        assert!((cpm - 0.973_729).abs() < 1e-6, "got {cpm}");
    }

    /// With the mean on the target, sum((x - T)^2) / (n - 1) is the sample
    /// variance, so Cpm is exactly Pp.
    #[test]
    fn cpm_equals_pp_when_the_mean_is_on_target() {
        let spec = ProcessCapability::new(Some(220.0), Some(200.0))
            .unwrap()
            .with_target(210.0);
        let data = [208.0, 209.0, 210.0, 211.0, 212.0];
        let indices = spec.compute(&data, 2.0).unwrap();
        let (cpm, pp) = (indices.cpm.unwrap(), indices.pp.unwrap());
        assert!((cpm - pp).abs() < 1e-12, "Cpm {cpm} vs Pp {pp}");
    }

    #[test]
    fn cpm_is_none_where_it_is_undefined() {
        let data = [208.0, 209.0, 210.0, 211.0, 212.0];
        let two_sided = || ProcessCapability::new(Some(220.0), Some(200.0)).unwrap();

        // A target outside the specification.
        for t in [199.0, 221.0, f64::NAN] {
            let cpm = two_sided()
                .with_target(t)
                .compute_overall(&data)
                .unwrap()
                .cpm;
            assert!(cpm.is_none(), "T={t}");
        }
        // Every observation on the target: no spread to divide by.
        let on_target = two_sided().with_target(210.0).compute(&[210.0; 5], 2.0);
        assert!(on_target.unwrap().cpm.is_none());
        // One-sided specification, even with a target.
        let upper = ProcessCapability::new(Some(220.0), None)
            .unwrap()
            .with_target(210.0);
        assert!(upper.compute_overall(&data).unwrap().cpm.is_none());
    }

    // -----------------------------------------------------------------------
    // Edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn compute_returns_none_for_insufficient_data() {
        let spec = ProcessCapability::new(Some(10.0), Some(0.0)).unwrap();
        assert!(spec.compute(&[5.0], 1.0).is_none());
        assert!(spec.compute(&[], 1.0).is_none());
    }

    #[test]
    fn compute_returns_none_for_invalid_sigma() {
        let spec = ProcessCapability::new(Some(10.0), Some(0.0)).unwrap();
        let data = [4.0, 5.0, 6.0, 5.0, 5.5];
        assert!(spec.compute(&data, 0.0).is_none());
        assert!(spec.compute(&data, -1.0).is_none());
        assert!(spec.compute(&data, f64::NAN).is_none());
        assert!(spec.compute(&data, f64::INFINITY).is_none());
    }

    #[test]
    fn compute_returns_none_for_nan_data() {
        let spec = ProcessCapability::new(Some(10.0), Some(0.0)).unwrap();
        let data = [4.0, f64::NAN, 6.0];
        assert!(spec.compute(&data, 1.0).is_none());
    }

    #[test]
    fn compute_overall_returns_none_for_insufficient_data() {
        let spec = ProcessCapability::new(Some(10.0), Some(0.0)).unwrap();
        assert!(spec.compute_overall(&[5.0]).is_none());
        assert!(spec.compute_overall(&[]).is_none());
    }

    // -----------------------------------------------------------------------
    // Montgomery (2020) §7.1 reference values
    // -----------------------------------------------------------------------

    /// Exact reference from Montgomery (2020), §7.1.
    ///
    /// μ = 74.001, σ_within = 0.0099, USL = 74.05, LSL = 73.95
    ///
    /// Cp  = (74.05 - 73.95) / (6 × 0.0099) = 0.1 / 0.0594 = 1.6835
    /// Cpu = (74.05 - 74.001) / (3 × 0.0099) = 0.049 / 0.0297 = 1.6499
    /// Cpl = (74.001 - 73.95) / (3 × 0.0099) = 0.051 / 0.0297 = 1.7172
    /// Cpk = min(1.6499, 1.7172) = 1.6499
    #[test]
    fn cp_cpk_montgomery_2020_reference() {
        let usl = 74.05_f64;
        let lsl = 73.95_f64;
        let mu = 74.001_f64;
        let sigma_within = 0.0099_f64;

        let spec = ProcessCapability::new(Some(usl), Some(lsl)).unwrap();

        // Single-point data: mean == mu, sigma_overall irrelevant for Cp/Cpk.
        // Use many identical points so sample mean == mu exactly.
        let data = vec![mu; 100];
        let indices = spec.compute(&data, sigma_within).unwrap();

        let cp = indices.cp.unwrap();
        let expected_cp = (usl - lsl) / (6.0 * sigma_within); // 1.6835...
        assert!(
            (cp - expected_cp).abs() < 0.001,
            "Cp: expected {expected_cp:.4}, got {cp:.4}"
        );

        let cpu = indices.cpu.unwrap();
        let expected_cpu = (usl - mu) / (3.0 * sigma_within); // 1.6499...
        assert!(
            (cpu - expected_cpu).abs() < 0.001,
            "Cpu: expected {expected_cpu:.4}, got {cpu:.4}"
        );

        let cpl = indices.cpl.unwrap();
        let expected_cpl = (mu - lsl) / (3.0 * sigma_within); // 1.7172...
        assert!(
            (cpl - expected_cpl).abs() < 0.001,
            "Cpl: expected {expected_cpl:.4}, got {cpl:.4}"
        );

        let cpk = indices.cpk.unwrap();
        let expected_cpk = expected_cpu.min(expected_cpl); // 1.6499...
        assert!(
            (cpk - expected_cpk).abs() < 0.001,
            "Cpk: expected {expected_cpk:.4}, got {cpk:.4}"
        );

        // Verify direction: process is slightly above centre, so Cpu < Cpl
        assert!(
            cpu < cpl,
            "mean above centre → Cpu ({cpu:.4}) should be < Cpl ({cpl:.4})"
        );
    }

    // -----------------------------------------------------------------------
    // Pp/Ppk differ from Cp/Cpk when sigmas differ
    // -----------------------------------------------------------------------

    #[test]
    fn pp_differs_from_cp_when_sigmas_differ() {
        let spec = ProcessCapability::new(Some(220.0), Some(200.0)).unwrap();

        let data = [
            205.0, 207.0, 210.0, 213.0, 215.0, 206.0, 208.0, 212.0, 214.0, 210.0, 204.0, 216.0,
            209.0, 211.0, 210.0,
        ];

        let sigma_within = 1.5;
        let indices = spec.compute(&data, sigma_within).unwrap();

        let cp = indices.cp.unwrap();
        let pp = indices.pp.unwrap();

        assert!(
            pp < cp,
            "Pp ({pp}) should be < Cp ({cp}) when sigma_overall > sigma_within"
        );
    }

    // -----------------------------------------------------------------------
    // Known numerical example
    // -----------------------------------------------------------------------

    /// Exact numerical verification.
    ///
    /// USL = 10, LSL = 0, sigma_within = 1.0
    /// Cp = (10 - 0) / (6 * 1) = 1.6667
    #[test]
    fn exact_numerical_verification() {
        let spec = ProcessCapability::new(Some(10.0), Some(0.0)).unwrap();

        let data = [4.0, 4.5, 5.0, 5.5, 6.0, 4.0, 5.0, 6.0, 5.0, 5.0];
        let x_bar = stats::mean(&data).unwrap();

        let sigma_within = 1.0;
        let indices = spec.compute(&data, sigma_within).unwrap();

        let expected_cp = 10.0 / 6.0;
        let expected_cpu = (10.0 - x_bar) / 3.0;
        let expected_cpl = (x_bar - 0.0) / 3.0;

        assert!(
            (indices.cp.unwrap() - expected_cp).abs() < 1e-10,
            "Cp: expected {expected_cp}, got {}",
            indices.cp.unwrap()
        );
        assert!(
            (indices.cpu.unwrap() - expected_cpu).abs() < 1e-10,
            "Cpu: expected {expected_cpu}, got {}",
            indices.cpu.unwrap()
        );
        assert!(
            (indices.cpl.unwrap() - expected_cpl).abs() < 1e-10,
            "Cpl: expected {expected_cpl}, got {}",
            indices.cpl.unwrap()
        );
        assert!(
            (indices.cpk.unwrap() - expected_cpu.min(expected_cpl)).abs() < 1e-10,
            "Cpk: expected {}, got {}",
            expected_cpu.min(expected_cpl),
            indices.cpk.unwrap()
        );
    }
}
