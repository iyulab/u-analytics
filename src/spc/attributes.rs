//! Attributes control charts: P, NP, C, and U charts.
//!
//! These charts monitor discrete (count/proportion) data from a process.
//! Unlike variables charts, attributes charts use the binomial or Poisson
//! distribution to compute control limits.
//!
//! # Chart Selection Guide
//!
//! | Chart | Data Type | Sample Size |
//! |-------|-----------|-------------|
//! | P     | Proportion defective | Variable |
//! | NP    | Count defective | Constant |
//! | C     | Count of defects | Constant area |
//! | U     | Defects per unit | Variable area |
//!
//! # References
//!
//! - Montgomery, D.C. (2019). *Introduction to Statistical Quality Control*, 8th ed.,
//!   Chapter 7: Control Charts for Attributes.
//! - ASTM E2587 — Standard Practice for Use of Control Charts

/// A single data point on an attributes control chart.
///
/// Contains the computed statistic, its control limits (which may vary
/// per point for charts with variable sample sizes), and an out-of-control flag.
#[derive(Debug, Clone)]
pub struct AttributeChartPoint {
    /// The zero-based index of this point.
    pub index: usize,
    /// The computed statistic value (proportion, count, or rate).
    pub value: f64,
    /// Upper control limit for this point.
    pub ucl: f64,
    /// Center line for this point.
    pub cl: f64,
    /// Lower control limit for this point.
    pub lcl: f64,
    /// Whether this point is out of control (beyond UCL or below LCL).
    pub out_of_control: bool,
}

// ---------------------------------------------------------------------------
// P Chart
// ---------------------------------------------------------------------------

/// Proportion nonconforming (P) chart.
///
/// Monitors the fraction of defective items in samples that may have
/// different sizes. Control limits vary per subgroup when sample sizes differ.
///
/// # Formulas
///
/// - CL = p-bar = total_defectives / total_inspected
/// - UCL_i = p-bar + 3 * sqrt(p-bar * (1 - p-bar) / n_i)
/// - LCL_i = max(0, p-bar - 3 * sqrt(p-bar * (1 - p-bar) / n_i))
///
/// # Reference
///
/// Montgomery, D.C. (2019). *Introduction to Statistical Quality Control*, 8th ed.,
/// Chapter 7, Section 7.3.
pub struct PChart {
    /// Stored samples as (defective_count, sample_size) pairs.
    samples: Vec<(u64, u64)>,
    /// Computed chart points.
    chart_points: Vec<AttributeChartPoint>,
    /// Overall proportion defective (p-bar).
    p_bar: Option<f64>,
}

impl PChart {
    /// Create a new P chart.
    pub fn new() -> Self {
        Self {
            samples: Vec::new(),
            chart_points: Vec::new(),
            p_bar: None,
        }
    }

    /// Add a sample with the number of defective items and the total sample size.
    ///
    /// Ignores samples where `defectives > sample_size` or `sample_size == 0`.
    pub fn add_sample(&mut self, defectives: u64, sample_size: u64) {
        if sample_size == 0 || defectives > sample_size {
            return;
        }
        self.samples.push((defectives, sample_size));
        self.recompute();
    }

    /// Get the overall proportion defective (p-bar), or `None` if no data.
    pub fn p_bar(&self) -> Option<f64> {
        self.p_bar
    }

    /// Get all chart points.
    pub fn points(&self) -> &[AttributeChartPoint] {
        &self.chart_points
    }

    /// Check if the process is in statistical control.
    pub fn is_in_control(&self) -> bool {
        self.chart_points.iter().all(|p| !p.out_of_control)
    }

    /// Recompute p-bar, control limits, and out-of-control flags.
    fn recompute(&mut self) {
        if self.samples.is_empty() {
            self.p_bar = None;
            self.chart_points.clear();
            return;
        }

        let total_defectives: u64 = self.samples.iter().map(|&(d, _)| d).sum();
        let total_inspected: u64 = self.samples.iter().map(|&(_, n)| n).sum();
        let p_bar = total_defectives as f64 / total_inspected as f64;
        self.p_bar = Some(p_bar);

        self.chart_points = self
            .samples
            .iter()
            .enumerate()
            .map(|(i, &(defectives, sample_size))| {
                let p = defectives as f64 / sample_size as f64;
                let n = sample_size as f64;
                let sigma = (p_bar * (1.0 - p_bar) / n).sqrt();
                let ucl = p_bar + 3.0 * sigma;
                let lcl = (p_bar - 3.0 * sigma).max(0.0);

                AttributeChartPoint {
                    index: i,
                    value: p,
                    ucl,
                    cl: p_bar,
                    lcl,
                    out_of_control: p > ucl || p < lcl,
                }
            })
            .collect();
    }
}

impl Default for PChart {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// NP Chart
// ---------------------------------------------------------------------------

/// Count of nonconforming items (NP) chart.
///
/// Monitors the count of defective items in samples of constant size.
/// Simpler than the P chart when sample sizes are uniform.
///
/// # Formulas
///
/// - CL = n * p-bar
/// - UCL = n * p-bar + 3 * sqrt(n * p-bar * (1 - p-bar))
/// - LCL = max(0, n * p-bar - 3 * sqrt(n * p-bar * (1 - p-bar)))
///
/// # Reference
///
/// Montgomery, D.C. (2019). *Introduction to Statistical Quality Control*, 8th ed.,
/// Chapter 7, Section 7.3.
pub struct NPChart {
    /// Constant sample size.
    sample_size: u64,
    /// Defective counts per subgroup.
    defective_counts: Vec<u64>,
    /// Computed chart points.
    chart_points: Vec<AttributeChartPoint>,
    /// Control limits (constant for NP chart).
    limits: Option<(f64, f64, f64)>, // (ucl, cl, lcl)
}

impl NPChart {
    /// Create a new NP chart with a constant sample size.
    ///
    /// # Panics
    ///
    /// Panics if `sample_size == 0`.
    pub fn new(sample_size: u64) -> Self {
        assert!(sample_size > 0, "sample_size must be > 0");
        Self {
            sample_size,
            defective_counts: Vec::new(),
            chart_points: Vec::new(),
            limits: None,
        }
    }

    /// Add a defective count for one subgroup.
    ///
    /// Ignores values where `defectives > sample_size`.
    pub fn add_sample(&mut self, defectives: u64) {
        if defectives > self.sample_size {
            return;
        }
        self.defective_counts.push(defectives);
        self.recompute();
    }

    /// Get the control limits as `(ucl, cl, lcl)`, or `None` if no data.
    pub fn control_limits(&self) -> Option<(f64, f64, f64)> {
        self.limits
    }

    /// Get all chart points.
    pub fn points(&self) -> &[AttributeChartPoint] {
        &self.chart_points
    }

    /// Check if the process is in statistical control.
    pub fn is_in_control(&self) -> bool {
        self.chart_points.iter().all(|p| !p.out_of_control)
    }

    /// Recompute limits and points.
    fn recompute(&mut self) {
        if self.defective_counts.is_empty() {
            self.limits = None;
            self.chart_points.clear();
            return;
        }

        let total_defectives: u64 = self.defective_counts.iter().sum();
        let total_inspected = self.sample_size * self.defective_counts.len() as u64;
        let p_bar = total_defectives as f64 / total_inspected as f64;
        let n = self.sample_size as f64;

        let np_bar = n * p_bar;
        let sigma = (n * p_bar * (1.0 - p_bar)).sqrt();
        let ucl = np_bar + 3.0 * sigma;
        let lcl = (np_bar - 3.0 * sigma).max(0.0);

        self.limits = Some((ucl, np_bar, lcl));

        self.chart_points = self
            .defective_counts
            .iter()
            .enumerate()
            .map(|(i, &count)| {
                let value = count as f64;
                AttributeChartPoint {
                    index: i,
                    value,
                    ucl,
                    cl: np_bar,
                    lcl,
                    out_of_control: value > ucl || value < lcl,
                }
            })
            .collect();
    }
}

// ---------------------------------------------------------------------------
// C Chart
// ---------------------------------------------------------------------------

/// Count of defects per unit (C) chart.
///
/// Monitors the total number of defects observed in a constant area of
/// opportunity (inspection unit). Based on the Poisson distribution.
///
/// # Formulas
///
/// - CL = c-bar (mean defect count)
/// - UCL = c-bar + 3 * sqrt(c-bar)
/// - LCL = max(0, c-bar - 3 * sqrt(c-bar))
///
/// # Reference
///
/// Montgomery, D.C. (2019). *Introduction to Statistical Quality Control*, 8th ed.,
/// Chapter 7, Section 7.4.
pub struct CChart {
    /// Defect counts per unit.
    defect_counts: Vec<u64>,
    /// Computed chart points.
    chart_points: Vec<AttributeChartPoint>,
    /// Control limits (constant for C chart).
    limits: Option<(f64, f64, f64)>, // (ucl, cl, lcl)
}

impl CChart {
    /// Create a new C chart.
    pub fn new() -> Self {
        Self {
            defect_counts: Vec::new(),
            chart_points: Vec::new(),
            limits: None,
        }
    }

    /// Add a defect count for one inspection unit.
    pub fn add_sample(&mut self, defects: u64) {
        self.defect_counts.push(defects);
        self.recompute();
    }

    /// Get the control limits as `(ucl, cl, lcl)`, or `None` if no data.
    pub fn control_limits(&self) -> Option<(f64, f64, f64)> {
        self.limits
    }

    /// Get all chart points.
    pub fn points(&self) -> &[AttributeChartPoint] {
        &self.chart_points
    }

    /// Check if the process is in statistical control.
    pub fn is_in_control(&self) -> bool {
        self.chart_points.iter().all(|p| !p.out_of_control)
    }

    /// Recompute limits and points.
    fn recompute(&mut self) {
        if self.defect_counts.is_empty() {
            self.limits = None;
            self.chart_points.clear();
            return;
        }

        let total: u64 = self.defect_counts.iter().sum();
        let c_bar = total as f64 / self.defect_counts.len() as f64;
        let sigma = c_bar.sqrt();
        let ucl = c_bar + 3.0 * sigma;
        let lcl = (c_bar - 3.0 * sigma).max(0.0);

        self.limits = Some((ucl, c_bar, lcl));

        self.chart_points = self
            .defect_counts
            .iter()
            .enumerate()
            .map(|(i, &count)| {
                let value = count as f64;
                AttributeChartPoint {
                    index: i,
                    value,
                    ucl,
                    cl: c_bar,
                    lcl,
                    out_of_control: value > ucl || value < lcl,
                }
            })
            .collect();
    }
}

impl Default for CChart {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// U Chart
// ---------------------------------------------------------------------------

/// Defects per unit (U) chart.
///
/// Monitors the defect rate when the area of opportunity (inspection size)
/// varies between subgroups. Control limits are computed individually for
/// each subgroup based on its inspection size.
///
/// # Formulas
///
/// - CL = u-bar = total_defects / total_units
/// - UCL_i = u-bar + 3 * sqrt(u-bar / n_i)
/// - LCL_i = max(0, u-bar - 3 * sqrt(u-bar / n_i))
///
/// # Reference
///
/// Montgomery, D.C. (2019). *Introduction to Statistical Quality Control*, 8th ed.,
/// Chapter 7, Section 7.4.
pub struct UChart {
    /// Stored samples as (defect_count, units_inspected) pairs.
    samples: Vec<(u64, f64)>,
    /// Computed chart points.
    chart_points: Vec<AttributeChartPoint>,
    /// Overall defect rate (u-bar).
    u_bar: Option<f64>,
}

impl UChart {
    /// Create a new U chart.
    pub fn new() -> Self {
        Self {
            samples: Vec::new(),
            chart_points: Vec::new(),
            u_bar: None,
        }
    }

    /// Add a sample with the number of defects and the number of units inspected.
    ///
    /// The `units_inspected` can be fractional (e.g., area or length).
    /// Ignores samples where `units_inspected <= 0` or is not finite.
    pub fn add_sample(&mut self, defects: u64, units_inspected: f64) {
        if !units_inspected.is_finite() || units_inspected <= 0.0 {
            return;
        }
        self.samples.push((defects, units_inspected));
        self.recompute();
    }

    /// Get the overall defect rate (u-bar), or `None` if no data.
    pub fn u_bar(&self) -> Option<f64> {
        self.u_bar
    }

    /// Get all chart points.
    pub fn points(&self) -> &[AttributeChartPoint] {
        &self.chart_points
    }

    /// Check if the process is in statistical control.
    pub fn is_in_control(&self) -> bool {
        self.chart_points.iter().all(|p| !p.out_of_control)
    }

    /// Recompute u-bar, control limits, and out-of-control flags.
    fn recompute(&mut self) {
        if self.samples.is_empty() {
            self.u_bar = None;
            self.chart_points.clear();
            return;
        }

        let total_defects: u64 = self.samples.iter().map(|&(d, _)| d).sum();
        let total_units: f64 = self.samples.iter().map(|&(_, n)| n).sum();
        let u_bar = total_defects as f64 / total_units;
        self.u_bar = Some(u_bar);

        self.chart_points = self
            .samples
            .iter()
            .enumerate()
            .map(|(i, &(defects, units))| {
                let u = defects as f64 / units;
                let sigma = (u_bar / units).sqrt();
                let ucl = u_bar + 3.0 * sigma;
                let lcl = (u_bar - 3.0 * sigma).max(0.0);

                AttributeChartPoint {
                    index: i,
                    value: u,
                    ucl,
                    cl: u_bar,
                    lcl,
                    out_of_control: u > ucl || u < lcl,
                }
            })
            .collect();
    }
}

impl Default for UChart {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- P Chart ---

    #[test]
    fn test_p_chart_basic() {
        // Textbook example: 10 samples of size 100
        let mut chart = PChart::new();
        let defectives = [5, 8, 3, 6, 4, 7, 2, 9, 5, 6];
        for &d in &defectives {
            chart.add_sample(d, 100);
        }

        let p_bar = chart.p_bar().expect("should have p_bar");
        // p-bar = 55/1000 = 0.055
        assert!(
            (p_bar - 0.055).abs() < 1e-10,
            "p_bar={p_bar}, expected 0.055"
        );

        // All 10 points should exist
        assert_eq!(chart.points().len(), 10);

        // Verify center line on all points
        for pt in chart.points() {
            assert!((pt.cl - 0.055).abs() < 1e-10);
        }
    }

    #[test]
    fn test_p_chart_limits() {
        let mut chart = PChart::new();
        // p-bar = 0.10, n = 100
        // sigma = sqrt(0.1 * 0.9 / 100) = 0.03
        // UCL = 0.10 + 0.09 = 0.19
        // LCL = 0.10 - 0.09 = 0.01
        chart.add_sample(10, 100);

        let pt = &chart.points()[0];
        assert!((pt.cl - 0.1).abs() < 1e-10);
        assert!((pt.ucl - 0.19).abs() < 0.001);
        assert!((pt.lcl - 0.01).abs() < 0.001);
    }

    #[test]
    fn test_p_chart_variable_sample_sizes() {
        let mut chart = PChart::new();
        chart.add_sample(5, 100);
        chart.add_sample(10, 200);
        chart.add_sample(3, 50);

        // p-bar = 18/350
        let p_bar = chart.p_bar().expect("p_bar");
        assert!((p_bar - 18.0 / 350.0).abs() < 1e-10);

        // UCL should differ per point due to variable n
        let pts = chart.points();
        // Larger sample = tighter limits
        assert!(pts[1].ucl - pts[1].cl < pts[0].ucl - pts[0].cl);
    }

    #[test]
    fn test_p_chart_rejects_invalid() {
        let mut chart = PChart::new();
        chart.add_sample(5, 0); // Zero sample size
        assert!(chart.p_bar().is_none());

        chart.add_sample(10, 5); // Defectives > sample size
        assert!(chart.p_bar().is_none());
    }

    #[test]
    fn test_p_chart_lcl_clamped_to_zero() {
        let mut chart = PChart::new();
        // Very small p with small n → LCL would be negative
        chart.add_sample(1, 10);
        let pt = &chart.points()[0];
        assert!(pt.lcl >= 0.0);
    }

    #[test]
    fn test_p_chart_out_of_control() {
        let mut chart = PChart::new();
        // Establish baseline with many normal samples
        for _ in 0..20 {
            chart.add_sample(5, 100);
        }
        // Add an outlier
        chart.add_sample(30, 100);

        assert!(!chart.is_in_control());
        let last = chart.points().last().expect("should have points");
        assert!(last.out_of_control);
    }

    #[test]
    fn test_p_chart_default() {
        let chart = PChart::default();
        assert!(chart.p_bar().is_none());
        assert!(chart.points().is_empty());
    }

    // --- NP Chart ---

    #[test]
    fn test_np_chart_basic() {
        let mut chart = NPChart::new(100);
        let defectives = [5, 8, 3, 6, 4, 7, 2, 9, 5, 6];
        for &d in &defectives {
            chart.add_sample(d);
        }

        let (ucl, cl, lcl) = chart.control_limits().expect("should have limits");
        // np-bar = 55/10 = 5.5
        assert!((cl - 5.5).abs() < 1e-10);
        assert!(ucl > cl);
        assert!(lcl < cl);
        assert!(lcl >= 0.0);
    }

    #[test]
    fn test_np_chart_rejects_invalid() {
        let mut chart = NPChart::new(100);
        chart.add_sample(101); // More defectives than sample size
        assert!(chart.control_limits().is_none());
    }

    #[test]
    #[should_panic(expected = "sample_size must be > 0")]
    fn test_np_chart_zero_sample_size() {
        let _ = NPChart::new(0);
    }

    #[test]
    fn test_np_chart_out_of_control() {
        let mut chart = NPChart::new(100);
        for _ in 0..20 {
            chart.add_sample(5);
        }
        chart.add_sample(30);

        assert!(!chart.is_in_control());
    }

    #[test]
    fn test_np_chart_limits_formula() {
        // n=200, p-bar = 0.05 → np-bar = 10
        // sigma = sqrt(200 * 0.05 * 0.95) = sqrt(9.5) ≈ 3.082
        // UCL = 10 + 3*3.082 = 19.246
        // LCL = 10 - 3*3.082 = 0.754
        let mut chart = NPChart::new(200);
        for _ in 0..10 {
            chart.add_sample(10);
        }

        let (ucl, cl, lcl) = chart.control_limits().expect("limits");
        assert!((cl - 10.0).abs() < 1e-10);
        let expected_sigma = (200.0_f64 * 0.05 * 0.95).sqrt();
        assert!((ucl - (10.0 + 3.0 * expected_sigma)).abs() < 0.01);
        assert!((lcl - (10.0 - 3.0 * expected_sigma)).abs() < 0.01);
    }

    // --- C Chart ---

    #[test]
    fn test_c_chart_basic() {
        let mut chart = CChart::new();
        let counts = [3, 5, 4, 6, 2, 7, 3, 4, 5, 6];
        for &c in &counts {
            chart.add_sample(c);
        }

        let (ucl, cl, lcl) = chart.control_limits().expect("should have limits");
        // c-bar = 45/10 = 4.5
        assert!((cl - 4.5).abs() < 1e-10);
        // UCL = 4.5 + 3*sqrt(4.5) = 4.5 + 6.364 = 10.864
        let expected_ucl = 4.5 + 3.0 * 4.5_f64.sqrt();
        assert!((ucl - expected_ucl).abs() < 0.01);
        assert!(lcl >= 0.0);
    }

    #[test]
    fn test_c_chart_out_of_control() {
        let mut chart = CChart::new();
        for _ in 0..20 {
            chart.add_sample(5);
        }
        chart.add_sample(50); // Way out of control

        assert!(!chart.is_in_control());
    }

    #[test]
    fn test_c_chart_single_sample() {
        let mut chart = CChart::new();
        chart.add_sample(10);

        let (_, cl, _) = chart.control_limits().expect("limits");
        assert!((cl - 10.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_c_chart_lcl_clamped() {
        // c-bar = 1 → LCL = 1 - 3*1 = -2 → clamped to 0
        let mut chart = CChart::new();
        chart.add_sample(1);

        let (_, _, lcl) = chart.control_limits().expect("limits");
        assert!((lcl - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_c_chart_default() {
        let chart = CChart::default();
        assert!(chart.control_limits().is_none());
        assert!(chart.points().is_empty());
    }

    // --- U Chart ---

    #[test]
    fn test_u_chart_basic() {
        let mut chart = UChart::new();
        // 5 samples, each inspecting 10 units
        chart.add_sample(3, 10.0);
        chart.add_sample(5, 10.0);
        chart.add_sample(4, 10.0);
        chart.add_sample(6, 10.0);
        chart.add_sample(2, 10.0);

        let u_bar = chart.u_bar().expect("should have u_bar");
        // u-bar = 20/50 = 0.4
        assert!((u_bar - 0.4).abs() < 1e-10);

        assert_eq!(chart.points().len(), 5);
    }

    #[test]
    fn test_u_chart_variable_units() {
        let mut chart = UChart::new();
        chart.add_sample(10, 5.0);  // u = 2.0
        chart.add_sample(20, 10.0); // u = 2.0
        chart.add_sample(5, 2.5);   // u = 2.0

        let u_bar = chart.u_bar().expect("u_bar");
        // u-bar = 35/17.5 = 2.0
        assert!((u_bar - 2.0).abs() < 1e-10);

        // Larger inspection area → tighter limits
        let pts = chart.points();
        let width_0 = pts[0].ucl - pts[0].cl; // n=5
        let width_1 = pts[1].ucl - pts[1].cl; // n=10
        assert!(width_1 < width_0, "larger n should have tighter limits");
    }

    #[test]
    fn test_u_chart_rejects_invalid() {
        let mut chart = UChart::new();
        chart.add_sample(5, 0.0);         // Zero units
        assert!(chart.u_bar().is_none());

        chart.add_sample(5, -1.0);        // Negative units
        assert!(chart.u_bar().is_none());

        chart.add_sample(5, f64::NAN);    // NaN units
        assert!(chart.u_bar().is_none());

        chart.add_sample(5, f64::INFINITY); // Infinite units
        assert!(chart.u_bar().is_none());
    }

    #[test]
    fn test_u_chart_out_of_control() {
        let mut chart = UChart::new();
        for _ in 0..20 {
            chart.add_sample(4, 10.0);
        }
        chart.add_sample(50, 10.0); // Far outlier

        assert!(!chart.is_in_control());
    }

    #[test]
    fn test_u_chart_lcl_clamped() {
        let mut chart = UChart::new();
        // Small u-bar with small n → LCL would be negative
        chart.add_sample(1, 1.0);
        let pt = &chart.points()[0];
        assert!(pt.lcl >= 0.0);
    }

    #[test]
    fn test_u_chart_default() {
        let chart = UChart::default();
        assert!(chart.u_bar().is_none());
        assert!(chart.points().is_empty());
    }

    #[test]
    fn test_u_chart_limits_formula() {
        // u-bar = 2.0, n = 4.0
        // sigma = sqrt(2.0/4.0) = sqrt(0.5) ≈ 0.7071
        // UCL = 2.0 + 3*0.7071 = 4.1213
        // LCL = max(0, 2.0 - 2.1213) = 0.0 (clamped)
        let mut chart = UChart::new();
        chart.add_sample(8, 4.0);

        let pt = &chart.points()[0];
        assert!((pt.cl - 2.0).abs() < 1e-10);
        let expected_sigma = (2.0_f64 / 4.0).sqrt();
        assert!((pt.ucl - (2.0 + 3.0 * expected_sigma)).abs() < 0.001);
    }

    // --- Cross-chart consistency ---

    #[test]
    fn test_p_and_np_consistent() {
        // P chart with constant n should give equivalent results to NP chart
        let mut p_chart = PChart::new();
        let mut np_chart = NPChart::new(100);

        let defectives = [5, 8, 3, 6, 4];
        for &d in &defectives {
            p_chart.add_sample(d, 100);
            np_chart.add_sample(d);
        }

        let p_bar = p_chart.p_bar().expect("p_bar");
        let (_, np_cl, _) = np_chart.control_limits().expect("np limits");

        // NP center line = n * p-bar
        assert!(
            (np_cl - 100.0 * p_bar).abs() < 1e-10,
            "NP CL should equal n * p_bar"
        );
    }

    #[test]
    fn test_c_and_u_consistent_equal_units() {
        // U chart with constant n=1 should give same limits as C chart
        let mut c_chart = CChart::new();
        let mut u_chart = UChart::new();

        let defects = [3, 5, 4, 6, 2];
        for &d in &defects {
            c_chart.add_sample(d);
            u_chart.add_sample(d, 1.0);
        }

        let (c_ucl, c_cl, c_lcl) = c_chart.control_limits().expect("C limits");
        let u_bar = u_chart.u_bar().expect("u_bar");

        assert!(
            (c_cl - u_bar).abs() < 1e-10,
            "C chart CL should equal U chart u-bar when n=1"
        );

        let u_pt = &u_chart.points()[0];
        assert!((u_pt.ucl - c_ucl).abs() < 1e-10);
        assert!((u_pt.lcl - c_lcl).abs() < 1e-10);
    }
}
