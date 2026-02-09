//! Process capability indices (Cp, Cpk, Pp, Ppk, Cpm).
//!
//! Process capability indices quantify how well a process output fits within
//! specification limits. Short-term indices (Cp, Cpk) use within-group
//! variation, while long-term indices (Pp, Ppk) use overall variation.
//!
//! # References
//!
//! - Montgomery (2019), *Introduction to Statistical Quality Control*, 8th ed.,
//!   Chapter 8.
//! - Kane (1986), "Process Capability Indices", *Journal of Quality Technology*
//!   18(1), pp. 41--52.
//! - Chan, Cheng & Spiring (1988), "A New Measure of Process Capability: Cpm",
//!   *Journal of Quality Technology* 20(3), pp. 162--175.

use u_optim::stats;

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
/// Fields are `Option<f64>` because not all indices can be computed for
/// one-sided specifications. For example, Cp requires both USL and LSL.
///
/// # Index interpretation
///
/// | Index | Value | Interpretation |
/// |-------|-------|----------------|
/// | Cp/Pp | >= 1.33 | Process is capable |
/// | Cpk/Ppk | >= 1.33 | Process is capable and centered |
/// | Cpm | >= 1.33 | Process meets Taguchi loss criterion |
///
/// Reference: Montgomery (2019), Chapter 8, Table 8.5.
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
    /// Cpm = Cp / sqrt(1 + ((mean - target) / sigma_within)^2).
    /// Requires both limits and a target.
    ///
    /// Reference: Chan, Cheng & Spiring (1988).
    pub cpm: Option<f64>,
    /// Sample mean of the data.
    pub mean: f64,
    /// Short-term (within-group) standard deviation.
    pub std_dev_within: f64,
    /// Long-term (overall) standard deviation.
    pub std_dev_overall: f64,
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

    /// Sets the target value for Cpm calculation.
    ///
    /// If not set, the target defaults to the midpoint `(USL + LSL) / 2`
    /// when both limits are available. Cpm is not computed for one-sided
    /// specifications without an explicit target.
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

        Some(self.compute_indices(x_bar, sigma_within, sigma_overall))
    }

    /// Computes capability indices using overall sigma for both short-term
    /// and long-term estimates.
    ///
    /// Use this when no within-group sigma estimate is available (e.g., no
    /// rational subgrouping). Both Cp/Cpk and Pp/Ppk will use the same
    /// sigma, so Cp == Pp and Cpk == Ppk.
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
    /// // When using overall sigma for both, Cp == Pp
    /// assert!((indices.cp.unwrap() - indices.pp.unwrap()).abs() < 1e-15);
    /// ```
    pub fn compute_overall(&self, data: &[f64]) -> Option<CapabilityIndices> {
        let x_bar = stats::mean(data)?;
        let sigma_overall = stats::std_dev(data)?;

        Some(self.compute_indices(x_bar, sigma_overall, sigma_overall))
    }

    /// Internal computation of all indices given mean and sigma values.
    fn compute_indices(
        &self,
        x_bar: f64,
        sigma_within: f64,
        sigma_overall: f64,
    ) -> CapabilityIndices {
        // Short-term indices (within-group sigma)
        let cpu = self.usl.map(|u| (u - x_bar) / (3.0 * sigma_within));
        let cpl = self.lsl.map(|l| (x_bar - l) / (3.0 * sigma_within));
        let cp = match (self.usl, self.lsl) {
            (Some(u), Some(l)) => Some((u - l) / (6.0 * sigma_within)),
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

        // Taguchi Cpm index
        let cpm = cp.and_then(|cp_val| {
            let target = self.target.or_else(|| {
                match (self.usl, self.lsl) {
                    (Some(u), Some(l)) => Some((u + l) / 2.0),
                    _ => None,
                }
            })?;
            let deviation_ratio = (x_bar - target) / sigma_within;
            Some(cp_val / (1.0 + deviation_ratio * deviation_ratio).sqrt())
        });

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
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    /// LSL = 200, USL = 220, target = 210 (midpoint)
    /// Process mean ~ 210, sigma_within = 2.0
    ///
    /// Cp = (220 - 200) / (6 * 2) = 20/12 = 1.6667
    #[test]
    fn textbook_centered_process() {
        let spec = ProcessCapability::new(Some(220.0), Some(200.0)).unwrap();

        let data = [
            208.0, 209.0, 210.0, 211.0, 212.0, 208.5, 209.5, 210.5, 211.5,
            210.0, 209.0, 211.0, 210.0, 209.5, 210.5, 210.0, 210.0, 210.0,
            209.0, 211.0,
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

        let cpm = indices.cpm.unwrap();
        assert!(cpm > 0.0, "Cpm should be positive for centered process");
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
            213.0, 214.0, 215.0, 216.0, 217.0, 213.5, 214.5, 215.5, 216.5,
            215.0, 214.0, 216.0, 215.0, 214.5, 215.5, 215.0, 215.0, 215.0,
            214.0, 216.0,
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
        assert!(
            (cpl - 2.5).abs() < 0.05,
            "expected Cpl ~ 2.5, got {cpl}"
        );

        let cpk = indices.cpk.unwrap();
        assert!(
            (cpk - cpu).abs() < 1e-15,
            "Cpk should equal min(Cpu, Cpl)"
        );
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
    fn compute_overall_matches_pp_equals_cp() {
        let spec = ProcessCapability::new(Some(220.0), Some(200.0)).unwrap();
        let data = [
            208.0, 209.0, 210.0, 211.0, 212.0, 208.5, 209.5, 210.5, 211.5,
            210.0,
        ];
        let indices = spec.compute_overall(&data).unwrap();

        let cp = indices.cp.unwrap();
        let pp = indices.pp.unwrap();
        assert!(
            (cp - pp).abs() < 1e-15,
            "Cp should equal Pp in compute_overall"
        );

        let cpk = indices.cpk.unwrap();
        let ppk = indices.ppk.unwrap();
        assert!(
            (cpk - ppk).abs() < 1e-15,
            "Cpk should equal Ppk in compute_overall"
        );
    }

    // -----------------------------------------------------------------------
    // Cpm with explicit target
    // -----------------------------------------------------------------------

    #[test]
    fn cpm_with_explicit_target() {
        let spec = ProcessCapability::new(Some(220.0), Some(200.0))
            .unwrap()
            .with_target(212.0);

        let data = [
            208.0, 209.0, 210.0, 211.0, 212.0, 208.5, 209.5, 210.5, 211.5,
            210.0,
        ];

        let sigma_within = 2.0;
        let indices = spec.compute(&data, sigma_within).unwrap();

        let cpm = indices.cpm.unwrap();
        let cp = indices.cp.unwrap();

        assert!(
            cpm < cp,
            "Cpm ({cpm}) should be less than Cp ({cp}) when mean != target"
        );
    }

    #[test]
    fn cpm_equals_cp_when_on_target() {
        let spec = ProcessCapability::new(Some(220.0), Some(200.0)).unwrap();

        // Data perfectly at midpoint (target = 210)
        let data = [210.0; 20];

        let sigma_within = 2.0;
        let indices = spec.compute(&data, sigma_within).unwrap();

        let cpm = indices.cpm.unwrap();
        let cp = indices.cp.unwrap();

        assert!(
            (cpm - cp).abs() < 1e-10,
            "Cpm ({cpm}) should equal Cp ({cp}) when mean == target"
        );
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
    // Pp/Ppk differ from Cp/Cpk when sigmas differ
    // -----------------------------------------------------------------------

    #[test]
    fn pp_differs_from_cp_when_sigmas_differ() {
        let spec = ProcessCapability::new(Some(220.0), Some(200.0)).unwrap();

        let data = [
            205.0, 207.0, 210.0, 213.0, 215.0, 206.0, 208.0, 212.0, 214.0,
            210.0, 204.0, 216.0, 209.0, 211.0, 210.0,
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
