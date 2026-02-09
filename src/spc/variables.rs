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
//! - Montgomery, D.C. (2019). *Introduction to Statistical Quality Control*, 8th ed.
//! - ASTM E2587 — Standard Practice for Use of Control Charts
//! - Shewhart, W.A. (1931). *Economic Control of Quality of Manufactured Product*.

use super::chart::{ChartPoint, ControlChart, ControlLimits, Violation, ViolationType};
use super::rules::{NelsonRules, RunRule};

// ---------------------------------------------------------------------------
// Control chart factor tables (ASTM E2587), indexed by subgroup size n=2..10
// Index 0 corresponds to n=2.
// ---------------------------------------------------------------------------

/// A2 factors for X-bar-R chart UCL/LCL computation.
///
/// UCL = X-double-bar + A2 * R-bar, LCL = X-double-bar - A2 * R-bar.
const A2: [f64; 9] = [1.880, 1.023, 0.729, 0.577, 0.483, 0.419, 0.373, 0.337, 0.308];

/// D3 factors for R chart lower control limit.
///
/// LCL_R = D3 * R-bar.
const D3: [f64; 9] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.076, 0.136, 0.184, 0.223];

/// D4 factors for R chart upper control limit.
///
/// UCL_R = D4 * R-bar.
const D4: [f64; 9] = [3.267, 2.575, 2.282, 2.114, 2.004, 1.924, 1.864, 1.816, 1.777];

/// d2 factors (mean of the range distribution) for estimating sigma from R-bar.
///
/// sigma-hat = R-bar / d2.
#[allow(dead_code)]
const D2: [f64; 9] = [1.128, 1.693, 2.059, 2.326, 2.534, 2.704, 2.847, 2.970, 3.078];

/// A3 factors for X-bar-S chart UCL/LCL computation.
///
/// UCL = X-double-bar + A3 * S-bar, LCL = X-double-bar - A3 * S-bar.
const A3: [f64; 9] = [2.659, 1.954, 1.628, 1.427, 1.287, 1.182, 1.099, 1.032, 0.975];

/// B3 factors for S chart lower control limit.
///
/// LCL_S = B3 * S-bar.
const B3: [f64; 9] = [0.0, 0.0, 0.0, 0.0, 0.030, 0.118, 0.185, 0.239, 0.284];

/// B4 factors for S chart upper control limit.
///
/// UCL_S = B4 * S-bar.
const B4: [f64; 9] = [3.267, 2.568, 2.266, 2.089, 1.970, 1.882, 1.815, 1.761, 1.716];

/// c4 factors for unbiased estimation of sigma from S-bar.
///
/// sigma-hat = S-bar / c4.
#[allow(dead_code)]
const C4: [f64; 9] = [
    0.7979, 0.8862, 0.9213, 0.9400, 0.9515, 0.9594, 0.9650, 0.9693, 0.9727,
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
/// let mut chart = XBarRChart::new(5);
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
}

impl XBarRChart {
    /// Create a new X-bar-R chart with the given subgroup size.
    ///
    /// # Panics
    ///
    /// Panics if `subgroup_size` is not in the range 2..=10.
    pub fn new(subgroup_size: usize) -> Self {
        assert!(
            (2..=10).contains(&subgroup_size),
            "subgroup_size must be 2..=10, got {subgroup_size}"
        );
        Self {
            subgroup_size,
            subgroups: Vec::new(),
            xbar_points: Vec::new(),
            r_points: Vec::new(),
            xbar_limits: None,
            r_limits: None,
        }
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
            let mean_val = u_optim::stats::mean(subgroup)
                .expect("subgroup should be non-empty with finite values");
            let range = subgroup_range(subgroup);
            xbar_values.push(mean_val);
            r_values.push(range);
        }

        // Grand mean and average range
        let grand_mean = u_optim::stats::mean(&xbar_values)
            .expect("xbar_values should be non-empty with finite values");
        let r_bar = u_optim::stats::mean(&r_values)
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
            let nelson = NelsonRules;
            let violations = nelson.check(&self.xbar_points, limits);
            apply_violations(&mut self.xbar_points, &violations);
        }

        // Apply Nelson rules to R chart
        if let Some(ref limits) = self.r_limits {
            let nelson = NelsonRules;
            let violations = nelson.check(&self.r_points, limits);
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
}

impl XBarSChart {
    /// Create a new X-bar-S chart with the given subgroup size.
    ///
    /// # Panics
    ///
    /// Panics if `subgroup_size` is not in the range 2..=10.
    pub fn new(subgroup_size: usize) -> Self {
        assert!(
            (2..=10).contains(&subgroup_size),
            "subgroup_size must be 2..=10, got {subgroup_size}"
        );
        Self {
            subgroup_size,
            subgroups: Vec::new(),
            xbar_points: Vec::new(),
            s_points: Vec::new(),
            xbar_limits: None,
            s_limits: None,
        }
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
            let mean_val = u_optim::stats::mean(subgroup)
                .expect("subgroup should be non-empty with finite values");
            let sd = u_optim::stats::std_dev(subgroup)
                .expect("subgroup should have >= 2 elements for std_dev");
            xbar_values.push(mean_val);
            s_values.push(sd);
        }

        // Grand mean and average S
        let grand_mean = u_optim::stats::mean(&xbar_values)
            .expect("xbar_values should be non-empty with finite values");
        let s_bar = u_optim::stats::mean(&s_values)
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
            let nelson = NelsonRules;
            let violations = nelson.check(&self.xbar_points, limits);
            apply_violations(&mut self.xbar_points, &violations);
        }

        // Apply Nelson rules to S chart
        if let Some(ref limits) = self.s_limits {
            let nelson = NelsonRules;
            let violations = nelson.check(&self.s_points, limits);
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
}

impl IndividualMRChart {
    /// Create a new Individual-MR chart.
    pub fn new() -> Self {
        Self {
            observations: Vec::new(),
            i_points: Vec::new(),
            mr_points: Vec::new(),
            i_limits: None,
            mr_limits: None,
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
        let x_bar = u_optim::stats::mean(&self.observations)
            .expect("observations should be non-empty with finite values");
        let mr_bar = u_optim::stats::mean(&mr_values)
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
            let nelson = NelsonRules;
            let violations = nelson.check(&self.i_points, limits);
            apply_violations(&mut self.i_points, &violations);
        }

        // Apply Nelson rules to MR chart
        if let Some(ref limits) = self.mr_limits {
            let nelson = NelsonRules;
            let violations = nelson.check(&self.mr_points, limits);
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
        u_optim::stats::max(subgroup).expect("subgroup should be non-empty without NaN");
    let min_val =
        u_optim::stats::min(subgroup).expect("subgroup should be non-empty without NaN");
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
        let mut chart = XBarRChart::new(4);
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
        let mut chart = XBarRChart::new(5);
        chart.add_sample(&[1.0, 2.0, 3.0]); // Wrong size, should be ignored
        assert!(chart.control_limits().is_none());
    }

    #[test]
    fn test_xbar_r_rejects_nan() {
        let mut chart = XBarRChart::new(3);
        chart.add_sample(&[1.0, f64::NAN, 3.0]);
        assert!(chart.control_limits().is_none());
    }

    #[test]
    fn test_xbar_r_r_chart_limits() {
        let mut chart = XBarRChart::new(5);
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
        let mut chart = XBarRChart::new(3);
        chart.add_sample(&[10.0, 10.0, 10.0]);
        chart.add_sample(&[10.0, 10.0, 10.0]);

        let limits = chart.control_limits().expect("should have limits");
        assert!((limits.cl - 10.0).abs() < f64::EPSILON);
        assert!((limits.ucl - 10.0).abs() < f64::EPSILON);
        assert!((limits.lcl - 10.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_xbar_r_detects_out_of_control() {
        let mut chart = XBarRChart::new(3);
        for _ in 0..5 {
            chart.add_sample(&[10.0, 10.5, 9.5]);
        }
        // Add an outlier subgroup
        chart.add_sample(&[50.0, 51.0, 49.0]);

        assert!(!chart.is_in_control());
    }

    #[test]
    #[should_panic(expected = "subgroup_size must be 2..=10")]
    fn test_xbar_r_invalid_size_1() {
        let _ = XBarRChart::new(1);
    }

    #[test]
    #[should_panic(expected = "subgroup_size must be 2..=10")]
    fn test_xbar_r_invalid_size_11() {
        let _ = XBarRChart::new(11);
    }

    // --- XBarSChart ---

    #[test]
    fn test_xbar_s_basic_limits() {
        let mut chart = XBarSChart::new(4);
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
        let mut chart = XBarSChart::new(5);
        chart.add_sample(&[1.0, 2.0]);
        assert!(chart.control_limits().is_none());
    }

    #[test]
    fn test_xbar_s_in_control() {
        let mut chart = XBarSChart::new(4);
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
        let mut chart = XBarRChart::new(5);
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
}
