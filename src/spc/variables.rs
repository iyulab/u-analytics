//! Variables control charts: X-bar-R, X-bar-S, and Individual-MR.
//!
//! These charts monitor continuous (variables) data from a process.
//! Subgroup charts (X-bar-R, X-bar-S) track the mean and within-subgroup
//! variation of small samples; the Individual-MR chart handles single
//! observations.
//!
//! # Control Chart Factors
//!
//! All constants (A2, D3, D4, d2, A3, B3, B4, c4, E2) are sourced from
//! ASTM E2587 — Standard Practice for Use of Control Charts in Statistical
//! Process Control.
//!
//! # References
//!
//! - Montgomery, D.C. (2020). *Introduction to Statistical Quality Control*, 8th ed.
//! - ASTM E2587 — Standard Practice for Use of Control Charts
//! - Shewhart, W.A. (1931). *Economic Control of Quality of Manufactured Product*.

use super::chart::{
    ChartPoint, ControlChart, ControlChartError, ControlLimits, Violation, ViolationType,
    MAX_SUBGROUP_SIZE, MIN_SUBGROUP_SIZE,
};
use super::rules::{RuleSet, RunRule};

// ---------------------------------------------------------------------------
// Control chart factor tables, indexed by subgroup size n=2..=25.
// Index 0 corresponds to n=2.
//
// Values for n=2..=10 are the ASTM E2587 published constants. The whole range
// is computed from the definitions rather than transcribed, because no
// published table this crate could cite covers n up to 25 in every factor:
//
//   d2(n) = E[W],  d3(n) = sd[W]   for W the range of n iid standard normals
//   c4(n) = sqrt(2/(n-1)) * Gamma(n/2) / Gamma((n-1)/2)
//   A2 = 3/(d2*sqrt(n))            A3 = 3/(c4*sqrt(n))
//   D3 = max(0, 1 - 3*d3/d2)       D4 = 1 + 3*d3/d2
//   B3 = max(0, 1 - 3*sqrt(1-c4^2)/c4)   B4 = 1 + 3*sqrt(1-c4^2)/c4
//
// The computation reproduces all 72 published values for n=2..=10 across the
// eight tables, which is what licenses the extension to n=25. The test
// `factor_tables_reproduce_the_published_astm_values` pins that agreement so a
// future edit cannot quietly break it.
// ---------------------------------------------------------------------------

/// A2 factors for X-bar-R chart UCL/LCL computation.
///
/// UCL = X-double-bar + A2 * R-bar, LCL = X-double-bar - A2 * R-bar.
const A2: [f64; 24] = [
    1.880, 1.023, 0.729, 0.577, 0.483, 0.419, 0.373, 0.337, 0.308, 0.285, 0.266, 0.249, 0.235,
    0.223, 0.212, 0.203, 0.194, 0.187, 0.180, 0.173, 0.167, 0.162, 0.157, 0.153,
];

/// D3 factors for R chart lower control limit.
///
/// LCL_R = D3 * R-bar.
const D3: [f64; 24] = [
    0.000, 0.000, 0.000, 0.000, 0.000, 0.076, 0.136, 0.184, 0.223, 0.256, 0.283, 0.307, 0.328,
    0.347, 0.363, 0.378, 0.391, 0.404, 0.415, 0.425, 0.435, 0.443, 0.452, 0.459,
];

/// D4 factors for R chart upper control limit.
///
/// UCL_R = D4 * R-bar.
const D4: [f64; 24] = [
    3.267, 2.575, 2.282, 2.114, 2.004, 1.924, 1.864, 1.816, 1.777, 1.744, 1.717, 1.693, 1.672,
    1.653, 1.637, 1.622, 1.609, 1.596, 1.585, 1.575, 1.565, 1.557, 1.548, 1.541,
];

/// d2 factors (mean of the range distribution) for estimating sigma from R-bar.
///
/// sigma-hat = R-bar / d2.
///
/// Index 0 corresponds to subgroup size n = 2.
///
/// # Reference
///
/// Montgomery, D.C. (2020). *Introduction to Statistical Quality Control*, 8th ed.,
/// Appendix Table VI.
#[allow(dead_code)]
const D2: [f64; 24] = [
    // d2(2) is exactly 2/sqrt(pi): the range of two standard normals is
    // |N(0, 2)|, whose mean is sqrt(2) * sqrt(2/pi).
    std::f64::consts::FRAC_2_SQRT_PI,
    1.6926,
    2.0588,
    2.3259,
    2.5344,
    2.7044,
    2.8472,
    2.9700,
    3.0775,
    3.1729,
    3.2585,
    3.3360,
    3.4068,
    3.4718,
    3.5320,
    3.5879,
    3.6401,
    3.6890,
    3.7350,
    3.7783,
    3.8194,
    3.8583,
    3.8953,
    3.9306,
];

/// A3 factors for X-bar-S chart UCL/LCL computation.
///
/// UCL = X-double-bar + A3 * S-bar, LCL = X-double-bar - A3 * S-bar.
const A3: [f64; 24] = [
    2.659, 1.954, 1.628, 1.427, 1.287, 1.182, 1.099, 1.032, 0.975, 0.927, 0.886, 0.850, 0.817,
    0.789, 0.763, 0.739, 0.718, 0.698, 0.680, 0.663, 0.647, 0.633, 0.619, 0.606,
];

/// B3 factors for S chart lower control limit.
///
/// LCL_S = B3 * S-bar.
const B3: [f64; 24] = [
    0.000, 0.000, 0.000, 0.000, 0.030, 0.118, 0.185, 0.239, 0.284, 0.321, 0.354, 0.382, 0.406,
    0.428, 0.448, 0.466, 0.482, 0.497, 0.510, 0.523, 0.534, 0.545, 0.555, 0.565,
];

/// B4 factors for S chart upper control limit.
///
/// UCL_S = B4 * S-bar.
const B4: [f64; 24] = [
    3.267, 2.568, 2.266, 2.089, 1.970, 1.882, 1.815, 1.761, 1.716, 1.679, 1.646, 1.618, 1.594,
    1.572, 1.552, 1.534, 1.518, 1.503, 1.490, 1.477, 1.466, 1.455, 1.445, 1.435,
];

/// c4 factors for unbiased estimation of sigma from S-bar.
///
/// sigma-hat = S-bar / c4.
///
/// Index 0 corresponds to subgroup size n = 2.
///
/// # Reference
///
/// Montgomery, D.C. (2020). *Introduction to Statistical Quality Control*, 8th ed.,
/// Appendix Table VI.
#[allow(dead_code)]
const C4: [f64; 24] = [
    0.7979, 0.8862, 0.9213, 0.9400, 0.9515, 0.9594, 0.9650, 0.9693, 0.9727, 0.9754, 0.9776, 0.9794,
    0.9810, 0.9823, 0.9835, 0.9845, 0.9854, 0.9862, 0.9869, 0.9876, 0.9882, 0.9887, 0.9892, 0.9896,
];

/// E2 factor for Individual chart UCL/LCL.
///
/// UCL = X-bar + E2 * MR-bar, LCL = X-bar - E2 * MR-bar.
/// E2 = 3 / d2(n=2) = 3 / 1.128 = 2.6596...
const E2: f64 = 2.660;

/// D4 factor for MR chart (n=2 moving range).
const D4_MR: f64 = 3.267;

// ---------------------------------------------------------------------------
// X-bar-R Chart
// ---------------------------------------------------------------------------

/// X-bar and Range (X-bar-R) control chart.
///
/// Monitors the process mean (X-bar chart) and process variability (R chart)
/// using subgroup ranges. Suitable for subgroup sizes n = 2..=10.
///
/// # Algorithm
///
/// 1. For each subgroup, compute the mean (X-bar) and range (R).
/// 2. Compute the grand mean (X-double-bar) and average range (R-bar).
/// 3. X-bar chart limits: CL = X-double-bar, UCL/LCL = CL +/- A2 * R-bar.
/// 4. R chart limits: CL = R-bar, UCL = D4 * R-bar, LCL = D3 * R-bar.
///
/// # Examples
///
/// ```
/// use u_analytics::spc::{XBarRChart, ControlChart};
///
/// let mut chart = XBarRChart::new(5).expect("subgroup size is in range");
/// chart.add_sample(&[25.0, 26.0, 24.5, 25.5, 25.0]);
/// chart.add_sample(&[25.2, 24.8, 25.1, 24.9, 25.3]);
/// chart.add_sample(&[25.1, 25.0, 24.7, 25.3, 24.9]);
///
/// let limits = chart.control_limits().expect("should have limits after 3 samples");
/// assert!(limits.ucl > limits.cl);
/// assert!(limits.cl > limits.lcl);
/// ```
///
/// # Reference
///
/// Montgomery, D.C. (2019). *Introduction to Statistical Quality Control*, 8th ed.,
/// Chapter 6: Control Charts for Variables.
#[derive(Debug, Clone)]
pub struct XBarRChart {
    /// Fixed subgroup size (2..=10).
    subgroup_size: usize,
    /// Stored subgroups.
    subgroups: Vec<Vec<f64>>,
    /// Computed X-bar chart points.
    xbar_points: Vec<ChartPoint>,
    /// Computed R chart points.
    r_points: Vec<ChartPoint>,
    /// X-bar chart control limits.
    xbar_limits: Option<ControlLimits>,
    /// R chart control limits.
    r_limits: Option<ControlLimits>,
    /// Run tests applied when limits are computed.
    rules: RuleSet,
}

impl XBarRChart {
    /// Apply only the given run tests when computing limits.
    ///
    /// Defaults to [`RuleSet::nelson`], so a chart built without calling this
    /// behaves exactly as before. Existing points are re-evaluated, so the
    /// call may be made at any time.
    ///
    /// # Examples
    ///
    /// ```
    /// use u_analytics::spc::{XBarRChart, ControlChart, RuleSet, ViolationType};
    ///
    /// let chart = XBarRChart::new(5).expect("5 is within range")
    ///     .with_rules(RuleSet::western_electric().without(ViolationType::NineOneSide));
    /// ```
    #[must_use]
    pub fn with_rules(mut self, rules: RuleSet) -> Self {
        self.rules = rules;
        self.recompute();
        self
    }

    /// The run tests this chart applies.
    pub fn rules(&self) -> RuleSet {
        self.rules
    }
    /// Create a new X-bar-R chart with the given subgroup size.
    ///
    /// # Errors
    ///
    /// Returns [`ControlChartError::SubgroupSizeOutOfRange`] when
    /// `subgroup_size` is outside `MIN_SUBGROUP_SIZE..=MAX_SUBGROUP_SIZE`.
    /// The size is data in most callers -- it comes from the measurements a
    /// user supplied -- so rejecting it is an ordinary outcome rather than a
    /// contract violation, and a boundary that cannot unwind (WebAssembly, C)
    /// needs it as a value.
    pub fn new(subgroup_size: usize) -> Result<Self, ControlChartError> {
        if !(MIN_SUBGROUP_SIZE..=MAX_SUBGROUP_SIZE).contains(&subgroup_size) {
            return Err(ControlChartError::SubgroupSizeOutOfRange {
                got: subgroup_size,
                min: MIN_SUBGROUP_SIZE,
                max: MAX_SUBGROUP_SIZE,
            });
        }
        Ok(Self {
            subgroup_size,
            subgroups: Vec::new(),
            xbar_points: Vec::new(),
            r_points: Vec::new(),
            xbar_limits: None,
            r_limits: None,
            rules: RuleSet::default(),
        })
    }

    /// Short-term (within-subgroup) sigma estimated from this chart:
    /// `sigma-hat = R-bar / d2(n)`.
    ///
    /// This is the quantity a capability study needs for Cp/Cpk, and it cannot
    /// be recovered from a flat measurement vector — it depends on the subgroup
    /// structure the chart already holds. Returning it here is what lets a
    /// caller feed [`crate::capability::ProcessCapability::compute`] without
    /// reimplementing the d2 table.
    ///
    /// Returns `None` if there is not enough data for control limits.
    ///
    /// # Examples
    ///
    /// ```
    /// use u_analytics::spc::{ControlChart, XBarRChart};
    ///
    /// let mut chart = XBarRChart::new(5).expect("subgroup size is in range");
    /// for g in [[9.9, 10.1, 10.0, 9.8, 10.2], [10.3, 9.7, 10.0, 10.1, 9.9]] {
    ///     chart.add_sample(&g);
    /// }
    /// let sigma = chart.sigma_hat().expect("limits available");
    /// assert!(sigma > 0.0);
    /// ```
    pub fn sigma_hat(&self) -> Option<f64> {
        let r_bar = self.r_limits.as_ref()?.cl;
        let d2 = D2[self.subgroup_size - 2];
        (d2 > 0.0).then(|| r_bar / d2)
    }

    /// Get the R chart control limits, or `None` if insufficient data.
    pub fn r_limits(&self) -> Option<ControlLimits> {
        self.r_limits.clone()
    }

    /// Get the R chart points.
    pub fn r_points(&self) -> &[ChartPoint] {
        &self.r_points
    }

    /// Recompute limits and points from stored subgroups.
    fn recompute(&mut self) {
        if self.subgroups.is_empty() {
            self.xbar_limits = None;
            self.r_limits = None;
            self.xbar_points.clear();
            self.r_points.clear();
            return;
        }

        let idx = self.subgroup_size - 2; // Factor table index

        // Compute subgroup means and ranges
        let mut xbar_values = Vec::with_capacity(self.subgroups.len());
        let mut r_values = Vec::with_capacity(self.subgroups.len());

        for subgroup in &self.subgroups {
            let mean_val = u_numflow::stats::mean(subgroup)
                .expect("subgroup should be non-empty with finite values");
            let range = subgroup_range(subgroup);
            xbar_values.push(mean_val);
            r_values.push(range);
        }

        // Grand mean and average range
        let grand_mean = u_numflow::stats::mean(&xbar_values)
            .expect("xbar_values should be non-empty with finite values");
        let r_bar = u_numflow::stats::mean(&r_values)
            .expect("r_values should be non-empty with finite values");

        // X-bar chart limits
        let a2 = A2[idx];
        self.xbar_limits = Some(ControlLimits {
            ucl: grand_mean + a2 * r_bar,
            cl: grand_mean,
            lcl: grand_mean - a2 * r_bar,
        });

        // R chart limits
        let d3 = D3[idx];
        let d4 = D4[idx];
        self.r_limits = Some(ControlLimits {
            ucl: d4 * r_bar,
            cl: r_bar,
            lcl: d3 * r_bar,
        });

        // Build points
        self.xbar_points = xbar_values
            .iter()
            .enumerate()
            .map(|(i, &v)| ChartPoint {
                value: v,
                index: i,
                violations: Vec::new(),
            })
            .collect();

        self.r_points = r_values
            .iter()
            .enumerate()
            .map(|(i, &v)| ChartPoint {
                value: v,
                index: i,
                violations: Vec::new(),
            })
            .collect();

        // Apply Nelson rules to X-bar chart
        if let Some(ref limits) = self.xbar_limits {
            let violations = self.rules.check(&self.xbar_points, limits);
            apply_violations(&mut self.xbar_points, &violations);
        }

        // Apply Nelson rules to R chart
        if let Some(ref limits) = self.r_limits {
            let violations = self.rules.check(&self.r_points, limits);
            apply_violations(&mut self.r_points, &violations);
        }
    }
}

impl ControlChart for XBarRChart {
    /// Add a subgroup sample. The sample length must equal the chart's subgroup size.
    fn add_sample(&mut self, sample: &[f64]) {
        if sample.len() != self.subgroup_size {
            return;
        }
        if !sample.iter().all(|x| x.is_finite()) {
            return;
        }
        self.subgroups.push(sample.to_vec());
        self.recompute();
    }

    fn control_limits(&self) -> Option<ControlLimits> {
        self.xbar_limits.clone()
    }

    fn is_in_control(&self) -> bool {
        self.xbar_points.iter().all(|p| p.violations.is_empty())
            && self.r_points.iter().all(|p| p.violations.is_empty())
    }

    fn violations(&self) -> Vec<Violation> {
        collect_violations(&self.xbar_points)
            .into_iter()
            .chain(collect_violations(&self.r_points))
            .collect()
    }

    fn points(&self) -> &[ChartPoint] {
        &self.xbar_points
    }
}

// ---------------------------------------------------------------------------
// X-bar-S Chart
// ---------------------------------------------------------------------------

/// X-bar and Standard Deviation (X-bar-S) control chart.
///
/// Monitors the process mean (X-bar chart) and process variability (S chart)
/// using subgroup standard deviations. Preferred over X-bar-R for larger
/// subgroups where range is a less efficient estimator.
///
/// # Algorithm
///
/// 1. For each subgroup, compute the mean (X-bar) and sample standard deviation (S).
/// 2. Compute the grand mean (X-double-bar) and average S (S-bar).
/// 3. X-bar chart limits: CL = X-double-bar, UCL/LCL = CL +/- A3 * S-bar.
/// 4. S chart limits: CL = S-bar, UCL = B4 * S-bar, LCL = B3 * S-bar.
///
/// # Reference
///
/// Montgomery, D.C. (2019). *Introduction to Statistical Quality Control*, 8th ed.,
/// Chapter 6: Control Charts for Variables.
#[derive(Debug, Clone)]
pub struct XBarSChart {
    /// Fixed subgroup size (2..=10).
    subgroup_size: usize,
    /// Stored subgroups.
    subgroups: Vec<Vec<f64>>,
    /// Computed X-bar chart points.
    xbar_points: Vec<ChartPoint>,
    /// Computed S chart points.
    s_points: Vec<ChartPoint>,
    /// X-bar chart control limits.
    xbar_limits: Option<ControlLimits>,
    /// S chart control limits.
    s_limits: Option<ControlLimits>,
    /// Run tests applied when limits are computed.
    rules: RuleSet,
}

impl XBarSChart {
    /// Apply only the given run tests when computing limits.
    ///
    /// Defaults to [`RuleSet::nelson`], so a chart built without calling this
    /// behaves exactly as before. Existing points are re-evaluated, so the
    /// call may be made at any time.
    ///
    /// # Examples
    ///
    /// ```
    /// use u_analytics::spc::{XBarSChart, ControlChart, RuleSet, ViolationType};
    ///
    /// let chart = XBarSChart::new(5).expect("5 is within range")
    ///     .with_rules(RuleSet::western_electric().without(ViolationType::NineOneSide));
    /// ```
    #[must_use]
    pub fn with_rules(mut self, rules: RuleSet) -> Self {
        self.rules = rules;
        self.recompute();
        self
    }

    /// The run tests this chart applies.
    pub fn rules(&self) -> RuleSet {
        self.rules
    }
    /// Create a new X-bar-S chart with the given subgroup size.
    ///
    /// # Errors
    ///
    /// Returns [`ControlChartError::SubgroupSizeOutOfRange`] when
    /// `subgroup_size` is outside `MIN_SUBGROUP_SIZE..=MAX_SUBGROUP_SIZE`.
    /// The size is data in most callers -- it comes from the measurements a
    /// user supplied -- so rejecting it is an ordinary outcome rather than a
    /// contract violation, and a boundary that cannot unwind (WebAssembly, C)
    /// needs it as a value.
    pub fn new(subgroup_size: usize) -> Result<Self, ControlChartError> {
        if !(MIN_SUBGROUP_SIZE..=MAX_SUBGROUP_SIZE).contains(&subgroup_size) {
            return Err(ControlChartError::SubgroupSizeOutOfRange {
                got: subgroup_size,
                min: MIN_SUBGROUP_SIZE,
                max: MAX_SUBGROUP_SIZE,
            });
        }
        Ok(Self {
            subgroup_size,
            subgroups: Vec::new(),
            xbar_points: Vec::new(),
            s_points: Vec::new(),
            xbar_limits: None,
            s_limits: None,
            rules: RuleSet::default(),
        })
    }

    /// Short-term (within-subgroup) sigma estimated from this chart:
    /// `sigma-hat = S-bar / c4(n)`.
    ///
    /// The S-chart counterpart of [`XBarRChart::sigma_hat`], and the estimator
    /// preferred for larger subgroups. Returns `None` if there is not enough
    /// data for control limits.
    ///
    /// # Examples
    ///
    /// ```
    /// use u_analytics::spc::{ControlChart, XBarSChart};
    ///
    /// let mut chart = XBarSChart::new(5).expect("subgroup size is in range");
    /// for g in [[9.9, 10.1, 10.0, 9.8, 10.2], [10.3, 9.7, 10.0, 10.1, 9.9]] {
    ///     chart.add_sample(&g);
    /// }
    /// let sigma = chart.sigma_hat().expect("limits available");
    /// assert!(sigma > 0.0);
    /// ```
    pub fn sigma_hat(&self) -> Option<f64> {
        let s_bar = self.s_limits.as_ref()?.cl;
        let c4 = C4[self.subgroup_size - 2];
        (c4 > 0.0).then(|| s_bar / c4)
    }

    /// Get the S chart control limits, or `None` if insufficient data.
    pub fn s_limits(&self) -> Option<ControlLimits> {
        self.s_limits.clone()
    }

    /// Get the S chart points.
    pub fn s_points(&self) -> &[ChartPoint] {
        &self.s_points
    }

    /// Recompute limits and points from stored subgroups.
    fn recompute(&mut self) {
        if self.subgroups.is_empty() {
            self.xbar_limits = None;
            self.s_limits = None;
            self.xbar_points.clear();
            self.s_points.clear();
            return;
        }

        let idx = self.subgroup_size - 2;

        // Compute subgroup means and standard deviations
        let mut xbar_values = Vec::with_capacity(self.subgroups.len());
        let mut s_values = Vec::with_capacity(self.subgroups.len());

        for subgroup in &self.subgroups {
            let mean_val = u_numflow::stats::mean(subgroup)
                .expect("subgroup should be non-empty with finite values");
            let sd = u_numflow::stats::std_dev(subgroup)
                .expect("subgroup should have >= 2 elements for std_dev");
            xbar_values.push(mean_val);
            s_values.push(sd);
        }

        // Grand mean and average S
        let grand_mean = u_numflow::stats::mean(&xbar_values)
            .expect("xbar_values should be non-empty with finite values");
        let s_bar = u_numflow::stats::mean(&s_values)
            .expect("s_values should be non-empty with finite values");

        // X-bar chart limits
        let a3 = A3[idx];
        self.xbar_limits = Some(ControlLimits {
            ucl: grand_mean + a3 * s_bar,
            cl: grand_mean,
            lcl: grand_mean - a3 * s_bar,
        });

        // S chart limits
        let b3 = B3[idx];
        let b4 = B4[idx];
        self.s_limits = Some(ControlLimits {
            ucl: b4 * s_bar,
            cl: s_bar,
            lcl: b3 * s_bar,
        });

        // Build points
        self.xbar_points = xbar_values
            .iter()
            .enumerate()
            .map(|(i, &v)| ChartPoint {
                value: v,
                index: i,
                violations: Vec::new(),
            })
            .collect();

        self.s_points = s_values
            .iter()
            .enumerate()
            .map(|(i, &v)| ChartPoint {
                value: v,
                index: i,
                violations: Vec::new(),
            })
            .collect();

        // Apply Nelson rules to X-bar chart
        if let Some(ref limits) = self.xbar_limits {
            let violations = self.rules.check(&self.xbar_points, limits);
            apply_violations(&mut self.xbar_points, &violations);
        }

        // Apply Nelson rules to S chart
        if let Some(ref limits) = self.s_limits {
            let violations = self.rules.check(&self.s_points, limits);
            apply_violations(&mut self.s_points, &violations);
        }
    }
}

impl ControlChart for XBarSChart {
    /// Add a subgroup sample. The sample length must equal the chart's subgroup size.
    fn add_sample(&mut self, sample: &[f64]) {
        if sample.len() != self.subgroup_size {
            return;
        }
        if !sample.iter().all(|x| x.is_finite()) {
            return;
        }
        self.subgroups.push(sample.to_vec());
        self.recompute();
    }

    fn control_limits(&self) -> Option<ControlLimits> {
        self.xbar_limits.clone()
    }

    fn is_in_control(&self) -> bool {
        self.xbar_points.iter().all(|p| p.violations.is_empty())
            && self.s_points.iter().all(|p| p.violations.is_empty())
    }

    fn violations(&self) -> Vec<Violation> {
        collect_violations(&self.xbar_points)
            .into_iter()
            .chain(collect_violations(&self.s_points))
            .collect()
    }

    fn points(&self) -> &[ChartPoint] {
        &self.xbar_points
    }
}

// ---------------------------------------------------------------------------
// Individual-MR Chart
// ---------------------------------------------------------------------------

/// Individual and Moving Range (I-MR) control chart.
///
/// Monitors individual observations (subgroup size = 1) using the moving range
/// of consecutive observations to estimate process variability.
///
/// # Algorithm
///
/// 1. Compute moving ranges: MR_i = |x_i - x_{i-1}| for i >= 1.
/// 2. Compute the mean of individual observations (X-bar) and the average
///    moving range (MR-bar).
/// 3. I chart limits: CL = X-bar, UCL/LCL = X-bar +/- E2 * MR-bar.
/// 4. MR chart limits: CL = MR-bar, UCL = D4 * MR-bar, LCL = 0.
///
/// # Examples
///
/// ```
/// use u_analytics::spc::{IndividualMRChart, ControlChart};
///
/// let mut chart = IndividualMRChart::new();
/// for &x in &[25.0, 25.2, 24.8, 25.1, 24.9, 25.3, 25.0, 24.7] {
///     chart.add_sample(&[x]);
/// }
///
/// let limits = chart.control_limits().expect("should have limits after 2+ observations");
/// assert!(limits.ucl > limits.cl);
/// assert!(limits.cl > limits.lcl);
/// ```
///
/// # Reference
///
/// Montgomery, D.C. (2019). *Introduction to Statistical Quality Control*, 8th ed.,
/// Chapter 6: Control Charts for Variables.
#[derive(Debug, Clone)]
pub struct IndividualMRChart {
    /// Individual observations.
    observations: Vec<f64>,
    /// Computed I chart points.
    i_points: Vec<ChartPoint>,
    /// Computed MR chart points.
    mr_points: Vec<ChartPoint>,
    /// I chart control limits.
    i_limits: Option<ControlLimits>,
    /// MR chart control limits.
    mr_limits: Option<ControlLimits>,
    /// Run tests applied when limits are computed.
    rules: RuleSet,
}

impl IndividualMRChart {
    /// Apply only the given run tests when computing limits.
    ///
    /// Defaults to [`RuleSet::nelson`], so a chart built without calling this
    /// behaves exactly as before. Existing points are re-evaluated, so the
    /// call may be made at any time.
    ///
    /// # Examples
    ///
    /// ```
    /// use u_analytics::spc::{IndividualMRChart, ControlChart, RuleSet, ViolationType};
    ///
    /// let chart = IndividualMRChart::new()
    ///     .with_rules(RuleSet::western_electric().without(ViolationType::NineOneSide));
    /// ```
    #[must_use]
    pub fn with_rules(mut self, rules: RuleSet) -> Self {
        self.rules = rules;
        self.recompute();
        self
    }

    /// The run tests this chart applies.
    pub fn rules(&self) -> RuleSet {
        self.rules
    }
    /// Create a new Individual-MR chart.
    pub fn new() -> Self {
        Self {
            observations: Vec::new(),
            i_points: Vec::new(),
            mr_points: Vec::new(),
            i_limits: None,
            mr_limits: None,
            rules: RuleSet::default(),
        }
    }

    /// Get the MR chart control limits, or `None` if insufficient data.
    pub fn mr_limits(&self) -> Option<ControlLimits> {
        self.mr_limits.clone()
    }

    /// Get the MR chart points.
    pub fn mr_points(&self) -> &[ChartPoint] {
        &self.mr_points
    }

    /// Recompute limits and points from stored observations.
    fn recompute(&mut self) {
        if self.observations.len() < 2 {
            self.i_limits = None;
            self.mr_limits = None;
            self.i_points.clear();
            self.mr_points.clear();
            return;
        }

        // Compute moving ranges
        let mr_values: Vec<f64> = self
            .observations
            .windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .collect();

        // X-bar and MR-bar
        let x_bar = u_numflow::stats::mean(&self.observations)
            .expect("observations should be non-empty with finite values");
        let mr_bar = u_numflow::stats::mean(&mr_values)
            .expect("mr_values should be non-empty with finite values");

        // I chart limits
        self.i_limits = Some(ControlLimits {
            ucl: x_bar + E2 * mr_bar,
            cl: x_bar,
            lcl: x_bar - E2 * mr_bar,
        });

        // MR chart limits (LCL is always 0 for n=2)
        self.mr_limits = Some(ControlLimits {
            ucl: D4_MR * mr_bar,
            cl: mr_bar,
            lcl: 0.0,
        });

        // Build I chart points
        self.i_points = self
            .observations
            .iter()
            .enumerate()
            .map(|(i, &v)| ChartPoint {
                value: v,
                index: i,
                violations: Vec::new(),
            })
            .collect();

        // Build MR chart points (starts at index 1, since MR_0 is undefined)
        self.mr_points = mr_values
            .iter()
            .enumerate()
            .map(|(i, &v)| ChartPoint {
                value: v,
                index: i + 1,
                violations: Vec::new(),
            })
            .collect();

        // Apply Nelson rules to I chart
        if let Some(ref limits) = self.i_limits {
            let violations = self.rules.check(&self.i_points, limits);
            apply_violations(&mut self.i_points, &violations);
        }

        // Apply Nelson rules to MR chart
        if let Some(ref limits) = self.mr_limits {
            let violations = self.rules.check(&self.mr_points, limits);
            apply_violations(&mut self.mr_points, &violations);
        }
    }
}

impl Default for IndividualMRChart {
    fn default() -> Self {
        Self::new()
    }
}

impl ControlChart for IndividualMRChart {
    /// Add a single observation. The sample slice must contain exactly one element.
    fn add_sample(&mut self, sample: &[f64]) {
        if sample.len() != 1 {
            return;
        }
        if !sample[0].is_finite() {
            return;
        }
        self.observations.push(sample[0]);
        self.recompute();
    }

    fn control_limits(&self) -> Option<ControlLimits> {
        self.i_limits.clone()
    }

    fn is_in_control(&self) -> bool {
        self.i_points.iter().all(|p| p.violations.is_empty())
            && self.mr_points.iter().all(|p| p.violations.is_empty())
    }

    fn violations(&self) -> Vec<Violation> {
        collect_violations(&self.i_points)
            .into_iter()
            .chain(collect_violations(&self.mr_points))
            .collect()
    }

    fn points(&self) -> &[ChartPoint] {
        &self.i_points
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compute the range (max - min) of a subgroup.
///
/// # Panics
///
/// Uses `expect` — callers must ensure `subgroup` is non-empty with finite values.
fn subgroup_range(subgroup: &[f64]) -> f64 {
    let max_val =
        u_numflow::stats::max(subgroup).expect("subgroup should be non-empty without NaN");
    let min_val =
        u_numflow::stats::min(subgroup).expect("subgroup should be non-empty without NaN");
    max_val - min_val
}

/// Apply a list of violations to chart points, matching by index.
fn apply_violations(points: &mut [ChartPoint], violations: &[(usize, ViolationType)]) {
    for &(idx, vtype) in violations {
        if let Some(point) = points.iter_mut().find(|p| p.index == idx) {
            point.violations.push(vtype);
        }
    }
}

/// Collect all violations from chart points into a flat list.
fn collect_violations(points: &[ChartPoint]) -> Vec<Violation> {
    let mut result = Vec::new();
    for point in points {
        for &vtype in &point.violations {
            result.push(Violation {
                point_index: point.index,
                violation_type: vtype,
            });
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- XBarRChart ---

    #[test]
    fn test_xbar_r_basic_limits() {
        let mut chart = XBarRChart::new(4).expect("subgroup size is in range");
        chart.add_sample(&[72.0, 84.0, 79.0, 49.0]);
        chart.add_sample(&[56.0, 87.0, 33.0, 42.0]);
        chart.add_sample(&[55.0, 73.0, 22.0, 60.0]);
        chart.add_sample(&[44.0, 80.0, 54.0, 74.0]);
        chart.add_sample(&[97.0, 26.0, 48.0, 58.0]);

        let limits = chart.control_limits().expect("should have limits");
        // Subgroup means: 71.0, 54.5, 52.5, 63.0, 57.25
        let expected_grand_mean = (71.0 + 54.5 + 52.5 + 63.0 + 57.25) / 5.0;
        assert!(
            (limits.cl - expected_grand_mean).abs() < 0.1,
            "CL={}, expected ~{expected_grand_mean}",
            limits.cl
        );

        // Verify UCL > CL > LCL
        assert!(limits.ucl > limits.cl);
        assert!(limits.cl > limits.lcl);
    }

    #[test]
    fn test_xbar_r_rejects_wrong_size() {
        let mut chart = XBarRChart::new(5).expect("subgroup size is in range");
        chart.add_sample(&[1.0, 2.0, 3.0]); // Wrong size, should be ignored
        assert!(chart.control_limits().is_none());
    }

    #[test]
    fn test_xbar_r_rejects_nan() {
        let mut chart = XBarRChart::new(3).expect("subgroup size is in range");
        chart.add_sample(&[1.0, f64::NAN, 3.0]);
        assert!(chart.control_limits().is_none());
    }

    #[test]
    fn test_xbar_r_r_chart_limits() {
        let mut chart = XBarRChart::new(5).expect("subgroup size is in range");
        chart.add_sample(&[10.0, 12.0, 11.0, 13.0, 14.0]);
        chart.add_sample(&[11.0, 13.0, 12.0, 10.0, 15.0]);
        chart.add_sample(&[12.0, 11.0, 14.0, 13.0, 10.0]);

        let r_limits = chart.r_limits().expect("should have R limits");
        assert!(r_limits.ucl > r_limits.cl);
        assert!(r_limits.lcl >= 0.0);
    }

    #[test]
    fn test_xbar_r_constant_subgroups() {
        // All identical values: R-bar = 0, limits collapse
        let mut chart = XBarRChart::new(3).expect("subgroup size is in range");
        chart.add_sample(&[10.0, 10.0, 10.0]);
        chart.add_sample(&[10.0, 10.0, 10.0]);

        let limits = chart.control_limits().expect("should have limits");
        assert!((limits.cl - 10.0).abs() < f64::EPSILON);
        assert!((limits.ucl - 10.0).abs() < f64::EPSILON);
        assert!((limits.lcl - 10.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_xbar_r_detects_out_of_control() {
        let mut chart = XBarRChart::new(3).expect("subgroup size is in range");
        for _ in 0..5 {
            chart.add_sample(&[10.0, 10.5, 9.5]);
        }
        // Add an outlier subgroup
        chart.add_sample(&[50.0, 51.0, 49.0]);

        assert!(!chart.is_in_control());
    }

    /// The n=2..=10 half of every factor table must still be the ASTM E2587
    /// published constants.
    ///
    /// The tables are computed from the definitions rather than transcribed, so
    /// this is the check that licenses that: agreement with the published values
    /// over the range a standard covers is the evidence that the extension to
    /// n=25 is the same quantity and not a plausible-looking different one.
    #[test]
    fn factor_tables_reproduce_the_published_astm_values() {
        // ASTM E2587 / Montgomery Appendix VI, n = 2..=10.
        const PUB_A2: [f64; 9] = [
            1.880, 1.023, 0.729, 0.577, 0.483, 0.419, 0.373, 0.337, 0.308,
        ];
        const PUB_D3: [f64; 9] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.076, 0.136, 0.184, 0.223];
        const PUB_D4: [f64; 9] = [
            3.267, 2.575, 2.282, 2.114, 2.004, 1.924, 1.864, 1.816, 1.777,
        ];
        const PUB_D2: [f64; 9] = [
            1.128, 1.693, 2.059, 2.326, 2.534, 2.704, 2.847, 2.970, 3.078,
        ];
        const PUB_A3: [f64; 9] = [
            2.659, 1.954, 1.628, 1.427, 1.287, 1.182, 1.099, 1.032, 0.975,
        ];
        const PUB_B3: [f64; 9] = [0.0, 0.0, 0.0, 0.0, 0.030, 0.118, 0.185, 0.239, 0.284];
        const PUB_B4: [f64; 9] = [
            3.267, 2.568, 2.266, 2.089, 1.970, 1.882, 1.815, 1.761, 1.716,
        ];
        const PUB_C4: [f64; 9] = [
            0.7979, 0.8862, 0.9213, 0.9400, 0.9515, 0.9594, 0.9650, 0.9693, 0.9727,
        ];

        for (name, table, published, tol) in [
            ("A2", &A2[..9], &PUB_A2[..], 5e-4),
            ("D3", &D3[..9], &PUB_D3[..], 5e-4),
            ("D4", &D4[..9], &PUB_D4[..], 5e-4),
            ("d2", &D2[..9], &PUB_D2[..], 5e-4),
            ("A3", &A3[..9], &PUB_A3[..], 5e-4),
            ("B3", &B3[..9], &PUB_B3[..], 5e-4),
            ("B4", &B4[..9], &PUB_B4[..], 5e-4),
            ("c4", &C4[..9], &PUB_C4[..], 5e-5),
        ] {
            for (i, (got, want)) in table.iter().zip(published).enumerate() {
                assert!(
                    (got - want).abs() < tol,
                    "{name}[n={}] = {got}, published {want}",
                    i + 2
                );
            }
        }
    }

    /// Every table covers the whole declared range and stays monotone in the
    /// direction its definition requires.
    #[test]
    fn factor_tables_cover_the_declared_range() {
        let span = MAX_SUBGROUP_SIZE - MIN_SUBGROUP_SIZE + 1;
        for (name, len) in [
            ("A2", A2.len()),
            ("D3", D3.len()),
            ("D4", D4.len()),
            ("d2", D2.len()),
            ("A3", A3.len()),
            ("B3", B3.len()),
            ("B4", B4.len()),
            ("c4", C4.len()),
        ] {
            assert_eq!(
                len, span,
                "{name} does not cover {MIN_SUBGROUP_SIZE}..={MAX_SUBGROUP_SIZE}"
            );
        }
        // d2 and c4 rise with n; A2, A3, D4 and B4 fall; D3 and B3 rise from 0.
        for i in 1..span {
            assert!(D2[i] > D2[i - 1], "d2 not increasing at n={}", i + 2);
            assert!(C4[i] > C4[i - 1], "c4 not increasing at n={}", i + 2);
            assert!(A2[i] < A2[i - 1], "A2 not decreasing at n={}", i + 2);
            assert!(A3[i] < A3[i - 1], "A3 not decreasing at n={}", i + 2);
            assert!(D4[i] < D4[i - 1], "D4 not decreasing at n={}", i + 2);
            assert!(B4[i] < B4[i - 1], "B4 not decreasing at n={}", i + 2);
            assert!(D3[i] >= D3[i - 1], "D3 not non-decreasing at n={}", i + 2);
            assert!(B3[i] >= B3[i - 1], "B3 not non-decreasing at n={}", i + 2);
        }
        // c4 is bounded by 1 and approaches it.
        assert!(C4[span - 1] < 1.0 && C4[span - 1] > 0.98);
    }

    #[test]
    fn test_xbar_r_subgroup_size_below_range_is_an_error() {
        assert_eq!(
            XBarRChart::new(1).unwrap_err(),
            ControlChartError::SubgroupSizeOutOfRange {
                got: 1,
                min: 2,
                max: 25
            }
        );
    }

    #[test]
    fn test_xbar_r_subgroup_size_above_range_is_an_error() {
        assert_eq!(
            XBarRChart::new(26).unwrap_err(),
            ControlChartError::SubgroupSizeOutOfRange {
                got: 26,
                min: 2,
                max: 25
            }
        );
    }

    #[test]
    fn test_subgroup_size_11_is_now_accepted() {
        // The regression this guards: n=11 was rejected outright, so a study
        // running larger subgroups could not use the chart at all.
        let mut chart = XBarRChart::new(11).expect("11 is within 2..=25");
        for _ in 0..5 {
            chart.add_sample(&[10.0, 10.2, 9.8, 10.1, 9.9, 10.3, 9.7, 10.0, 10.1, 9.9, 10.0]);
        }
        let limits = chart.control_limits().expect("limits from 5 subgroups");
        assert!(limits.ucl > limits.cl && limits.cl > limits.lcl);
        assert!(chart.sigma_hat().is_some_and(|s| s > 0.0));
    }

    // --- XBarSChart ---

    #[test]
    fn test_xbar_s_basic_limits() {
        let mut chart = XBarSChart::new(4).expect("subgroup size is in range");
        chart.add_sample(&[72.0, 84.0, 79.0, 49.0]);
        chart.add_sample(&[56.0, 87.0, 33.0, 42.0]);
        chart.add_sample(&[55.0, 73.0, 22.0, 60.0]);
        chart.add_sample(&[44.0, 80.0, 54.0, 74.0]);
        chart.add_sample(&[97.0, 26.0, 48.0, 58.0]);

        let limits = chart.control_limits().expect("should have limits");
        assert!(limits.ucl > limits.cl);
        assert!(limits.cl > limits.lcl);

        let s_limits = chart.s_limits().expect("should have S limits");
        assert!(s_limits.ucl > s_limits.cl);
        assert!(s_limits.lcl >= 0.0);
    }

    #[test]
    fn test_xbar_s_rejects_wrong_size() {
        let mut chart = XBarSChart::new(5).expect("subgroup size is in range");
        chart.add_sample(&[1.0, 2.0]);
        assert!(chart.control_limits().is_none());
    }

    #[test]
    fn test_xbar_s_in_control() {
        let mut chart = XBarSChart::new(4).expect("subgroup size is in range");
        for _ in 0..10 {
            chart.add_sample(&[10.0, 10.2, 9.8, 10.1]);
        }
        assert!(chart.is_in_control());
    }

    // --- IndividualMRChart ---

    #[test]
    fn test_imr_basic_limits() {
        let mut chart = IndividualMRChart::new();
        let data = [10.0, 12.0, 11.0, 13.0, 10.0, 14.0, 11.0, 12.0, 13.0, 10.0];
        for &x in &data {
            chart.add_sample(&[x]);
        }

        let limits = chart.control_limits().expect("should have limits");
        assert!(limits.ucl > limits.cl);
        assert!(limits.cl > limits.lcl);

        let mr_limits = chart.mr_limits().expect("should have MR limits");
        assert!(mr_limits.ucl > mr_limits.cl);
        assert!((mr_limits.lcl).abs() < f64::EPSILON);
    }

    #[test]
    fn test_imr_needs_two_points() {
        let mut chart = IndividualMRChart::new();
        chart.add_sample(&[10.0]);
        assert!(chart.control_limits().is_none());
    }

    #[test]
    fn test_imr_center_line_is_mean() {
        let mut chart = IndividualMRChart::new();
        let data = [5.0, 10.0, 15.0, 20.0, 25.0];
        for &x in &data {
            chart.add_sample(&[x]);
        }
        let limits = chart.control_limits().expect("should have limits");
        assert!((limits.cl - 15.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_imr_mr_values() {
        let mut chart = IndividualMRChart::new();
        let data = [10.0, 12.0, 9.0];
        for &x in &data {
            chart.add_sample(&[x]);
        }
        // MR values: |12-10| = 2, |9-12| = 3
        let mr_pts = chart.mr_points();
        assert_eq!(mr_pts.len(), 2);
        assert!((mr_pts[0].value - 2.0).abs() < f64::EPSILON);
        assert!((mr_pts[1].value - 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_imr_rejects_multi_element_sample() {
        let mut chart = IndividualMRChart::new();
        chart.add_sample(&[1.0, 2.0]);
        assert!(chart.points().is_empty());
    }

    #[test]
    fn test_imr_detects_out_of_control() {
        let mut chart = IndividualMRChart::new();
        for i in 0..10 {
            chart.add_sample(&[50.0 + (i as f64 % 3.0) * 0.5]);
        }
        // Add a far outlier
        chart.add_sample(&[100.0]);

        assert!(!chart.is_in_control());
    }

    #[test]
    fn test_imr_default() {
        let chart = IndividualMRChart::default();
        assert!(chart.points().is_empty());
    }

    // --- Helper function tests ---

    #[test]
    fn test_subgroup_range() {
        assert!((subgroup_range(&[1.0, 5.0, 3.0]) - 4.0).abs() < f64::EPSILON);
        assert!((subgroup_range(&[10.0, 10.0, 10.0])).abs() < f64::EPSILON);
    }

    // --- Textbook verification: X-bar-R chart factors ---

    #[test]
    fn test_xbar_r_chart_factors_n5() {
        // For n=5: A2=0.577, D3=0.0, D4=2.114
        // Subgroup with mean=50, range=10
        let mut chart = XBarRChart::new(5).expect("subgroup size is in range");
        chart.add_sample(&[45.0, 47.0, 50.0, 53.0, 55.0]);

        let limits = chart.control_limits().expect("limits");
        assert!((limits.cl - 50.0).abs() < f64::EPSILON);

        let r_limits = chart.r_limits().expect("R limits");
        assert!((r_limits.cl - 10.0).abs() < f64::EPSILON);

        // UCL = 50 + 0.577 * 10 = 55.77
        assert!((limits.ucl - 55.77).abs() < 0.01);
        // LCL = 50 - 0.577 * 10 = 44.23
        assert!((limits.lcl - 44.23).abs() < 0.01);
    }

    // --- Textbook verification: I-MR chart ---

    #[test]
    fn test_imr_e2_factor() {
        // E2 = 2.660
        // Two points with X-bar = 100, MR = |105-95| = 10
        let mut chart = IndividualMRChart::new();
        chart.add_sample(&[95.0]);
        chart.add_sample(&[105.0]);

        let limits = chart.control_limits().expect("limits");
        // X-bar = 100
        assert!((limits.cl - 100.0).abs() < f64::EPSILON);
        // MR-bar = 10
        // UCL = 100 + 2.660 * 10 = 126.6
        assert!((limits.ucl - 126.6).abs() < 0.1);
        // LCL = 100 - 2.660 * 10 = 73.4
        assert!((limits.lcl - 73.4).abs() < 0.1);
    }

    // --- Montgomery Table VI: d2 constant verification ---

    /// Verify d2 constants against Montgomery (2020), Appendix Table VI.
    ///
    /// d2 is the expected value of the sample range for a standard-normal
    /// distribution with subgroup size n. Used as sigma-hat = R-bar / d2.
    ///
    /// Allowed tolerance: ±0.001 (matches 3-decimal precision in Table VI).
    #[test]
    fn test_d2_constants_montgomery_table_vi() {
        // (n, expected_d2) pairs from Montgomery (2020) Table VI, n=2..10
        let expected: [(usize, f64); 9] = [
            (2, 1.128),
            (3, 1.693),
            (4, 2.059),
            (5, 2.326),
            (6, 2.534),
            (7, 2.704),
            (8, 2.847),
            (9, 2.970),
            (10, 3.078),
        ];
        for (n, d2_ref) in expected {
            let d2 = D2[n - 2];
            assert!(
                (d2 - d2_ref).abs() < 0.001,
                "d2(n={n}): expected {d2_ref}, got {d2}"
            );
        }
    }

    // --- Montgomery Table VI: c4 constant verification ---

    /// Verify c4 constants against Montgomery (2020), Appendix Table VI.
    ///
    /// c4 is the bias-correction factor for estimating sigma from S-bar.
    /// sigma-hat = S-bar / c4.
    ///
    /// Allowed tolerance: ±0.001 (matches 4-decimal precision in Table VI).
    #[test]
    fn test_c4_constants_montgomery_table_vi() {
        // (n, expected_c4) pairs from Montgomery (2020) Table VI, n=2..6
        let expected: [(usize, f64); 6] = [
            (2, 0.7979),
            (3, 0.8862),
            (4, 0.9213),
            (5, 0.9400),
            (6, 0.9515),
            (7, 0.9594),
        ];
        for (n, c4_ref) in expected {
            let c4 = C4[n - 2];
            assert!(
                (c4 - c4_ref).abs() < 0.001,
                "c4(n={n}): expected {c4_ref}, got {c4}"
            );
        }
    }

    // --- sigma-hat from the chart (Cycle 266) --------------------------------
    //
    // Short-term sigma is what a capability study needs for Cp/Cpk, and it
    // cannot be recovered from a flat measurement vector: it depends on the
    // subgroup structure the chart holds. Without these the caller has to
    // reimplement the d2/c4 tables.

    #[test]
    fn xbar_r_sigma_hat_is_r_bar_over_d2() {
        let mut chart = XBarRChart::new(5).expect("subgroup size is in range");
        let groups = [
            [9.9, 10.1, 10.0, 9.8, 10.2],
            [10.3, 9.7, 10.0, 10.1, 9.9],
            [9.8, 10.2, 10.1, 9.9, 10.0],
            [10.5, 9.5, 10.0, 10.2, 9.8],
        ];
        for g in &groups {
            chart.add_sample(g);
        }
        let r_bar = chart.r_limits().expect("limits").cl;
        let expected = r_bar / D2[5 - 2];
        let sigma = chart.sigma_hat().expect("sigma-hat");
        assert!(
            (sigma - expected).abs() < 1e-12,
            "sigma-hat must be R-bar/d2: got {sigma}, expected {expected}"
        );
        assert!(sigma > 0.0);
    }

    #[test]
    fn xbar_s_sigma_hat_is_s_bar_over_c4() {
        let mut chart = XBarSChart::new(5).expect("subgroup size is in range");
        for g in &[
            [9.9, 10.1, 10.0, 9.8, 10.2],
            [10.3, 9.7, 10.0, 10.1, 9.9],
            [9.8, 10.2, 10.1, 9.9, 10.0],
        ] {
            chart.add_sample(g);
        }
        let s_bar = chart.s_limits().expect("limits").cl;
        let expected = s_bar / C4[5 - 2];
        let sigma = chart.sigma_hat().expect("sigma-hat");
        assert!((sigma - expected).abs() < 1e-12);
    }

    #[test]
    fn sigma_hat_is_none_without_limits() {
        assert!(XBarRChart::new(5).unwrap().sigma_hat().is_none());
        assert!(XBarSChart::new(5).unwrap().sigma_hat().is_none());
    }

    #[test]
    fn short_term_sigma_differs_from_overall_when_subgroups_drift() {
        use u_numflow::stats;

        // Six subgroups of five with real between-subgroup drift: the whole
        // point of a short-term estimate is that it is NOT the overall sigma.
        let groups = [
            [9.9, 10.1, 10.0, 9.8, 10.2],
            [10.3, 9.7, 10.0, 10.1, 9.9],
            [9.8, 10.2, 10.1, 9.9, 10.0],
            [10.5, 9.5, 10.0, 10.2, 9.8],
            [9.6, 10.4, 10.0, 9.9, 10.1],
            [10.1, 9.9, 10.0, 10.3, 9.7],
        ];
        let mut chart = XBarRChart::new(5).expect("subgroup size is in range");
        for g in &groups {
            chart.add_sample(g);
        }
        let sigma_within = chart.sigma_hat().expect("sigma-hat");
        let flat: Vec<f64> = groups.iter().flatten().copied().collect();
        let sigma_overall = stats::std_dev(&flat).expect("overall sigma");

        assert!(
            (sigma_within - sigma_overall).abs() > 1e-6,
            "short-term and long-term sigma must differ on drifting subgroups:              within={sigma_within}, overall={sigma_overall}"
        );
    }
}
