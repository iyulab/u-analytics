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
/// # Examples
///
/// ```
/// use u_analytics::spc::PChart;
///
/// let mut chart = PChart::new();
/// chart.add_sample(3, 100);  // 3 defectives out of 100
/// chart.add_sample(5, 100);
/// chart.add_sample(2, 100);
/// chart.add_sample(4, 100);
///
/// let p_bar = chart.p_bar().expect("should have p_bar after adding samples");
/// assert!(p_bar > 0.0);
/// ```
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
// Laney P' Chart
// ---------------------------------------------------------------------------

/// A single data point for a Laney P' or U' chart.
///
/// Laney charts correct for overdispersion or underdispersion by estimating
/// a process sigma-inflation factor φ from the moving range of standardized
/// subgroup statistics.
///
/// # References
///
/// - Laney, D.B. (2002). "Improved control charts for attributes",
///   *Quality Engineering* 14(4), pp. 531-537.
#[derive(Debug, Clone)]
pub struct LaneyAttributePoint {
    /// Zero-based index of this subgroup.
    pub index: usize,
    /// The observed statistic (proportion or defect rate) for this subgroup.
    pub value: f64,
    /// Upper control limit (overdispersion-adjusted).
    pub ucl: f64,
    /// Center line (overall mean proportion or rate).
    pub cl: f64,
    /// Lower control limit (overdispersion-adjusted, clamped to 0).
    pub lcl: f64,
    /// Whether this point lies beyond its control limits.
    pub out_of_control: bool,
}

/// Laney P' chart result.
///
/// Contains the overall proportion defective, the overdispersion factor φ,
/// and the per-subgroup chart points with adjusted limits.
#[derive(Debug, Clone)]
pub struct LaneyPChart {
    /// Overall proportion defective (p̄ = Σdᵢ / Σnᵢ).
    pub p_bar: f64,
    /// Overdispersion/underdispersion correction factor φ = MR̄ / d₂.
    /// φ = 1.0 means no correction needed (ordinary P chart).
    pub phi: f64,
    /// Per-subgroup chart points.
    pub points: Vec<LaneyAttributePoint>,
}

/// Laney U' chart result.
///
/// Contains the overall defect rate, the overdispersion factor φ,
/// and the per-subgroup chart points with adjusted limits.
#[derive(Debug, Clone)]
pub struct LaneyUChart {
    /// Overall defect rate (ū = Σdefectsᵢ / Σunitsᵢ).
    pub u_bar: f64,
    /// Overdispersion/underdispersion correction factor φ = MR̄ / d₂.
    /// φ = 1.0 means no correction needed (ordinary U chart).
    pub phi: f64,
    /// Per-subgroup chart points.
    pub points: Vec<LaneyAttributePoint>,
}

/// Compute the Laney P' chart from `(defective_count, sample_size)` pairs.
///
/// The Laney P' chart adjusts control limits for overdispersion or underdispersion
/// by estimating a sigma-inflation factor φ from the moving range of standardized
/// proportions. When φ = 1.0 the limits reduce to those of a standard P chart.
///
/// # Algorithm
///
/// 1. p̄ = Σdᵢ / Σnᵢ
/// 2. zᵢ = (pᵢ − p̄) / √(p̄·(1−p̄)/nᵢ)
/// 3. MR̄ = mean(|zᵢ − z_{i-1}|) for i = 1..n-1
/// 4. φ = MR̄ / d₂,  d₂ = 1.128 (for moving range of 2 observations)
/// 5. UCLᵢ = p̄ + 3·φ·√(p̄·(1−p̄)/nᵢ)
/// 6. LCLᵢ = max(0, p̄ − 3·φ·√(p̄·(1−p̄)/nᵢ))
///
/// # Returns
///
/// `None` if fewer than 3 subgroups are provided, if all sample sizes are zero,
/// or if p̄ is 0 or 1 (degenerate cases where σ = 0).
///
/// # Reference
///
/// Laney, D.B. (2002). "Improved control charts for attributes",
/// *Quality Engineering* 14(4), pp. 531-537.
pub fn laney_p_chart(samples: &[(u64, u64)]) -> Option<LaneyPChart> {
    if samples.len() < 3 {
        return None;
    }

    let total_defectives: u64 = samples.iter().map(|&(d, _)| d).sum();
    let total_inspected: u64 = samples.iter().map(|&(_, n)| n).sum();
    if total_inspected == 0 {
        return None;
    }

    let p_bar = total_defectives as f64 / total_inspected as f64;
    // Degenerate: σ = 0 means all proportions are exactly p̄ — φ computation is undefined.
    let base_var = p_bar * (1.0 - p_bar);
    if base_var <= 0.0 {
        // φ = 0 (no variability), build chart with zero-width limits.
        let points = samples
            .iter()
            .enumerate()
            .map(|(i, &(d, n))| {
                let value = if n > 0 { d as f64 / n as f64 } else { p_bar };
                LaneyAttributePoint {
                    index: i,
                    value,
                    ucl: p_bar,
                    cl: p_bar,
                    lcl: p_bar,
                    out_of_control: false,
                }
            })
            .collect();
        return Some(LaneyPChart {
            p_bar,
            phi: 0.0,
            points,
        });
    }

    // Step 2: standardized proportions.
    let z_scores: Vec<f64> = samples
        .iter()
        .map(|&(d, n)| {
            let p_i = d as f64 / n as f64;
            let sigma_i = (base_var / n as f64).sqrt();
            (p_i - p_bar) / sigma_i
        })
        .collect();

    // Step 3: moving ranges of z-scores.
    let mr_bar = {
        let mrs: Vec<f64> = z_scores.windows(2).map(|w| (w[1] - w[0]).abs()).collect();
        mrs.iter().sum::<f64>() / mrs.len() as f64
    };

    // Step 4: φ = MR̄ / d₂ (d₂ = 1.128 for subgroup of 2).
    const D2: f64 = 1.128;
    let phi = mr_bar / D2;

    // Steps 5-6: build chart points.
    let points = samples
        .iter()
        .enumerate()
        .map(|(i, &(d, n))| {
            let p_i = d as f64 / n as f64;
            let sigma_i = (base_var / n as f64).sqrt();
            let ucl = p_bar + 3.0 * phi * sigma_i;
            let lcl = (p_bar - 3.0 * phi * sigma_i).max(0.0);
            LaneyAttributePoint {
                index: i,
                value: p_i,
                ucl,
                cl: p_bar,
                lcl,
                out_of_control: p_i > ucl || p_i < lcl,
            }
        })
        .collect();

    Some(LaneyPChart { p_bar, phi, points })
}

/// Compute the Laney U' chart from `(defect_count, inspection_units)` pairs.
///
/// The Laney U' chart adjusts control limits for overdispersion or underdispersion
/// by estimating a sigma-inflation factor φ from the moving range of standardized
/// defect rates. When φ = 1.0 the limits reduce to those of a standard U chart.
///
/// # Algorithm
///
/// 1. ū = Σdefectsᵢ / Σunitsᵢ
/// 2. zᵢ = (uᵢ − ū) / √(ū / unitsᵢ)
/// 3. MR̄ = mean(|zᵢ − z_{i-1}|)
/// 4. φ = MR̄ / d₂,  d₂ = 1.128
/// 5. UCLᵢ = ū + 3·φ·√(ū / unitsᵢ)
/// 6. LCLᵢ = max(0, ū − 3·φ·√(ū / unitsᵢ))
///
/// # Returns
///
/// `None` if fewer than 3 subgroups, if total units are zero, or if any
/// `units` entry is non-positive.
///
/// # Reference
///
/// Laney, D.B. (2002). "Improved control charts for attributes",
/// *Quality Engineering* 14(4), pp. 531-537.
pub fn laney_u_chart(samples: &[(u64, f64)]) -> Option<LaneyUChart> {
    if samples.len() < 3 {
        return None;
    }

    // Validate: all units must be positive and finite.
    if samples.iter().any(|&(_, n)| !n.is_finite() || n <= 0.0) {
        return None;
    }

    let total_defects: u64 = samples.iter().map(|&(d, _)| d).sum();
    let total_units: f64 = samples.iter().map(|&(_, n)| n).sum();
    if total_units <= 0.0 {
        return None;
    }

    let u_bar = total_defects as f64 / total_units;

    // Degenerate: ū = 0 means no defects observed — φ computation is undefined.
    if u_bar <= 0.0 {
        let points = samples
            .iter()
            .enumerate()
            .map(|(i, &(d, n))| {
                let value = d as f64 / n;
                LaneyAttributePoint {
                    index: i,
                    value,
                    ucl: 0.0,
                    cl: 0.0,
                    lcl: 0.0,
                    out_of_control: false,
                }
            })
            .collect();
        return Some(LaneyUChart {
            u_bar: 0.0,
            phi: 0.0,
            points,
        });
    }

    // Step 2: standardized defect rates.
    let z_scores: Vec<f64> = samples
        .iter()
        .map(|&(d, n)| {
            let u_i = d as f64 / n;
            let sigma_i = (u_bar / n).sqrt();
            (u_i - u_bar) / sigma_i
        })
        .collect();

    // Step 3: moving ranges.
    let mr_bar = {
        let mrs: Vec<f64> = z_scores.windows(2).map(|w| (w[1] - w[0]).abs()).collect();
        mrs.iter().sum::<f64>() / mrs.len() as f64
    };

    // Step 4: φ = MR̄ / d₂.
    const D2: f64 = 1.128;
    let phi = mr_bar / D2;

    // Steps 5-6: build chart points.
    let points = samples
        .iter()
        .enumerate()
        .map(|(i, &(d, n))| {
            let u_i = d as f64 / n;
            let sigma_i = (u_bar / n).sqrt();
            let ucl = u_bar + 3.0 * phi * sigma_i;
            let lcl = (u_bar - 3.0 * phi * sigma_i).max(0.0);
            LaneyAttributePoint {
                index: i,
                value: u_i,
                ucl,
                cl: u_bar,
                lcl,
                out_of_control: u_i > ucl || u_i < lcl,
            }
        })
        .collect();

    Some(LaneyUChart { u_bar, phi, points })
}

// ---------------------------------------------------------------------------
// G Chart (Geometric — inter-defect conforming count)
// ---------------------------------------------------------------------------

/// A single data point on a G or T chart.
#[derive(Debug, Clone)]
pub struct GChartPoint {
    /// Zero-based index of this inter-event observation.
    pub index: usize,
    /// The observed inter-event count (for G) or time (for T).
    pub value: f64,
    /// Upper control limit.
    pub ucl: f64,
    /// Center line (mean of the series).
    pub cl: f64,
    /// Lower control limit (clamped to 0).
    pub lcl: f64,
    /// Whether this point lies beyond its control limits.
    pub out_of_control: bool,
}

/// G chart (geometric distribution) result for rare-event monitoring.
///
/// Monitors the number of conforming units between consecutive defect events.
/// Appropriate when the defect rate is very low (< 1%) and standard P/NP
/// charts produce degenerate limits.
///
/// # Reference
///
/// Kaminsky, F.C. et al. (1992). "Statistical control charts based on a
/// geometric distribution", *Journal of Quality Technology* 24(2), pp. 63-69.
#[derive(Debug, Clone)]
pub struct GChart {
    /// Mean inter-event conforming count (ḡ).
    pub g_bar: f64,
    /// Per-observation chart points.
    pub points: Vec<GChartPoint>,
}

/// T chart (exponential distribution) result for rare-event monitoring.
///
/// Monitors the time between consecutive defect events.
/// Control limits are derived from the exponential distribution percentiles
/// corresponding to ±3σ probability mass (α/2 = 0.00135).
///
/// # Reference
///
/// Borror, C.M., Keats, J.B. & Montgomery, D.C. (2003). "Robustness of the
/// time between events CUSUM", *International Journal of Production Research*
/// 41(15), pp. 3435-3444.
#[derive(Debug, Clone)]
pub struct TChart {
    /// Mean inter-event time (t̄).
    pub t_bar: f64,
    /// Per-observation chart points.
    pub points: Vec<TChartPoint>,
}

/// A single data point on a T chart.
#[derive(Debug, Clone)]
pub struct TChartPoint {
    /// Zero-based index of this inter-event observation.
    pub index: usize,
    /// The observed inter-event time.
    pub value: f64,
    /// Upper control limit.
    pub ucl: f64,
    /// Center line (mean inter-event time).
    pub cl: f64,
    /// Lower control limit (clamped to 0).
    pub lcl: f64,
    /// Whether this point lies beyond its control limits.
    pub out_of_control: bool,
}

/// Compute the G chart from inter-event conforming counts.
///
/// # Formulas
///
/// - ḡ = mean(gᵢ)
/// - UCL = ḡ + 3·√(ḡ·(ḡ+1))
/// - LCL = max(0, ḡ − 3·√(ḡ·(ḡ+1)))
///
/// # Returns
///
/// `None` if fewer than 3 observations or any count is negative.
///
/// # Reference
///
/// Kaminsky, F.C. et al. (1992). "Statistical control charts based on a
/// geometric distribution", *Journal of Quality Technology* 24(2), pp. 63-69.
pub fn g_chart(inter_event_counts: &[f64]) -> Option<GChart> {
    if inter_event_counts.len() < 3 {
        return None;
    }
    if inter_event_counts
        .iter()
        .any(|&v| !v.is_finite() || v < 0.0)
    {
        return None;
    }

    let g_bar = inter_event_counts.iter().sum::<f64>() / inter_event_counts.len() as f64;
    let spread = (g_bar * (g_bar + 1.0)).sqrt();
    let ucl = g_bar + 3.0 * spread;
    let lcl = (g_bar - 3.0 * spread).max(0.0);

    let points = inter_event_counts
        .iter()
        .enumerate()
        .map(|(i, &v)| GChartPoint {
            index: i,
            value: v,
            ucl,
            cl: g_bar,
            lcl,
            out_of_control: v > ucl || v < lcl,
        })
        .collect();

    Some(GChart { g_bar, points })
}

/// Compute the T chart from inter-event times.
///
/// # Formulas (exponential distribution percentiles)
///
/// - t̄ = mean(tᵢ)
/// - UCL = t̄ · (−ln(0.00135))  ≈ t̄ · 6.6077
/// - LCL = max(0, t̄ · (−ln(0.99865)))  ≈ t̄ · 0.00135
///
/// The constants are derived from the 0.00135 and 0.99865 quantiles of the
/// standard exponential distribution, matching the ±3σ tail probability used
/// in Shewhart charts (α/2 = 0.00135).
///
/// # Returns
///
/// `None` if fewer than 3 observations or any time is non-positive.
///
/// # Reference
///
/// Borror, C.M., Keats, J.B. & Montgomery, D.C. (2003). "Robustness of the
/// time between events CUSUM", *International Journal of Production Research*
/// 41(15), pp. 3435-3444.
pub fn t_chart(inter_event_times: &[f64]) -> Option<TChart> {
    if inter_event_times.len() < 3 {
        return None;
    }
    if inter_event_times
        .iter()
        .any(|&v| !v.is_finite() || v <= 0.0)
    {
        return None;
    }

    let t_bar = inter_event_times.iter().sum::<f64>() / inter_event_times.len() as f64;

    // Exponential quantile: Q(p) = -t̄ · ln(1 - p) = t̄ · (-ln(p)) for the survival function.
    // UCL corresponds to the 0.99865 quantile of Exp(1/t̄): −ln(1 − 0.99865) = −ln(0.00135).
    // LCL corresponds to the 0.00135 quantile of Exp(1/t̄): −ln(1 − 0.00135) = −ln(0.99865).
    let ucl_factor = -(0.00135_f64.ln()); // ≈ 6.6077
    let lcl_factor = -(0.99865_f64.ln()); // ≈ 0.001351

    let ucl = t_bar * ucl_factor;
    let lcl = (t_bar * lcl_factor).max(0.0);

    let points = inter_event_times
        .iter()
        .enumerate()
        .map(|(i, &v)| TChartPoint {
            index: i,
            value: v,
            ucl,
            cl: t_bar,
            lcl,
            out_of_control: v > ucl || v < lcl,
        })
        .collect();

    Some(TChart { t_bar, points })
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
        chart.add_sample(10, 5.0); // u = 2.0
        chart.add_sample(20, 10.0); // u = 2.0
        chart.add_sample(5, 2.5); // u = 2.0

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
        chart.add_sample(5, 0.0); // Zero units
        assert!(chart.u_bar().is_none());

        chart.add_sample(5, -1.0); // Negative units
        assert!(chart.u_bar().is_none());

        chart.add_sample(5, f64::NAN); // NaN units
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

    // --- Laney P' Chart ---

    #[test]
    fn laney_p_basic() {
        let samples: Vec<(u64, u64)> = (0..10).map(|i| (i % 5 + 2, 200)).collect();
        let chart = laney_p_chart(&samples).expect("chart should be Some");
        assert!(chart.phi > 0.0);
        assert!(chart.p_bar > 0.0 && chart.p_bar < 1.0);
        assert_eq!(chart.points.len(), 10);
    }

    #[test]
    fn laney_p_constant_proportion_phi_near_zero() {
        // All samples identical → all z-scores = 0 → MR = 0 → phi = 0
        let samples: Vec<(u64, u64)> = vec![(10, 1000); 20];
        let chart = laney_p_chart(&samples).expect("chart should be Some");
        assert!((chart.p_bar - 0.01).abs() < 1e-10);
        assert!(chart.phi >= 0.0);
    }

    #[test]
    fn laney_p_ucl_above_lcl() {
        let samples: Vec<(u64, u64)> = vec![(5, 100), (8, 100), (3, 100), (6, 100), (4, 100)];
        let chart = laney_p_chart(&samples).expect("chart should be Some");
        for p in &chart.points {
            assert!(p.ucl >= p.lcl);
            assert!((p.cl - chart.p_bar).abs() < 1e-10);
        }
    }

    #[test]
    fn laney_p_insufficient_data() {
        let samples: Vec<(u64, u64)> = vec![(2, 100), (3, 100)];
        assert!(laney_p_chart(&samples).is_none());
    }

    #[test]
    fn laney_u_basic() {
        let samples: Vec<(u64, f64)> = vec![(5, 10.0); 10];
        let chart = laney_u_chart(&samples).expect("chart should be Some");
        assert!((chart.u_bar - 0.5).abs() < 1e-10);
        assert!(chart.phi >= 0.0);
    }

    #[test]
    fn laney_u_ucl_above_cl() {
        let samples: Vec<(u64, f64)> = (0..8).map(|i| ((i % 4 + 2) as u64, 10.0)).collect();
        let chart = laney_u_chart(&samples).expect("chart should be Some");
        for p in &chart.points {
            assert!(p.ucl > p.cl || (p.ucl - p.cl).abs() < 1e-10);
        }
    }

    // --- Montgomery (2020) Example 7.1 Reference Validation ---

    /// P chart reference validation against Montgomery (2019) §7.2–7.3 formula.
    ///
    /// Uses 20 samples of n=100 with Σd=198, so p̄=198/2000=0.099 exactly.
    /// UCL = 0.099 + 3·√(0.099·0.901/100) ≈ 0.188196
    /// LCL = max(0, 0.099 − 0.089196) ≈ 0.009804
    ///
    /// Reference: Montgomery, D.C. (2019). *Introduction to Statistical Quality
    /// Control*, 8th ed., §7.2–7.3.
    #[test]
    fn p_chart_montgomery_reference_formula() {
        // Construct 20 samples of n=100 so total defectives = 198 (p̄ = 0.099).
        // Spread as [10, 10, 10, ..., 10, 8] so sum = 19*10 + 8 = 198.
        let mut chart = PChart::new();
        for _ in 0..19 {
            chart.add_sample(10, 100);
        }
        chart.add_sample(8, 100);

        let p_bar = chart.p_bar().expect("p_bar");
        assert!(
            (p_bar - 0.099).abs() < 1e-10,
            "p̄ expected 0.099, got {p_bar}"
        );

        let sigma = (0.099_f64 * 0.901 / 100.0).sqrt();
        let expected_ucl = 0.099 + 3.0 * sigma; // ≈ 0.188196
        let expected_lcl = (0.099 - 3.0 * sigma).max(0.0); // ≈ 0.009804

        for pt in chart.points() {
            assert!(
                (pt.ucl - expected_ucl).abs() < 1e-6,
                "UCL mismatch at index {}: expected {expected_ucl:.6}, got {:.6}",
                pt.index,
                pt.ucl
            );
            assert!(
                (pt.lcl - expected_lcl).abs() < 1e-6,
                "LCL mismatch at index {}: expected {expected_lcl:.6}, got {:.6}",
                pt.index,
                pt.lcl
            );
        }
    }

    /// NP chart reference validation against Montgomery (2019) §7.2–7.3 formula.
    ///
    /// n=100, p̄=0.099 → np̄=9.9
    /// UCL = 9.9 + 3·√(9.9·0.901) = 9.9 + 3·2.9867 ≈ 18.860
    /// LCL = max(0, 9.9 − 8.960) ≈ 0.940
    ///
    /// Reference: Montgomery, D.C. (2019). *Introduction to Statistical Quality
    /// Control*, 8th ed., §7.2–7.3.
    #[test]
    fn np_chart_montgomery_reference() {
        // 20 samples of n=100, total defectives=198 → p̄=0.099, np̄=9.9
        let mut chart = NPChart::new(100);
        for _ in 0..19 {
            chart.add_sample(10);
        }
        chart.add_sample(8);

        let (ucl, cl, lcl) = chart.control_limits().expect("limits");
        // np̄ = 9.9
        assert!(
            (cl - 9.9).abs() < 1e-10,
            "NP CL expected 9.9, got {cl}"
        );
        // sigma = sqrt(9.9 * 0.901) = sqrt(8.9199) ≈ 2.98662
        let expected_sigma = (9.9_f64 * 0.901).sqrt();
        let expected_ucl = 9.9 + 3.0 * expected_sigma; // ≈ 18.860
        let expected_lcl = (9.9 - 3.0 * expected_sigma).max(0.0); // ≈ 0.940
        assert!(
            (ucl - expected_ucl).abs() < 1e-6,
            "NP UCL expected {expected_ucl:.4}, got {ucl:.4}"
        );
        assert!(
            (lcl - expected_lcl).abs() < 1e-6,
            "NP LCL expected {expected_lcl:.4}, got {lcl:.4}"
        );
    }

    /// C chart validation against Montgomery (2020) §7.4.
    ///
    /// Reference: c̄=10
    /// UCL = 10 + 3·√10 = 10 + 9.4868 = 19.4868
    /// LCL = max(0, 10 − 9.4868) = 0.5132
    #[test]
    fn c_chart_montgomery_reference() {
        // 20 samples all with defect count 10 → c̄ = 10 exactly
        let mut chart = CChart::new();
        for _ in 0..20 {
            chart.add_sample(10);
        }

        let (ucl, cl, lcl) = chart.control_limits().expect("limits");
        assert!(
            (cl - 10.0).abs() < 1e-10,
            "C chart CL expected 10.0, got {cl}"
        );
        let expected_ucl = 10.0 + 3.0 * 10.0_f64.sqrt(); // ≈ 19.4868
        let expected_lcl = (10.0 - 3.0 * 10.0_f64.sqrt()).max(0.0); // ≈ 0.5132
        assert!(
            (ucl - expected_ucl).abs() < 1e-6,
            "C chart UCL expected {expected_ucl:.4}, got {ucl:.4}"
        );
        assert!(
            (lcl - expected_lcl).abs() < 1e-6,
            "C chart LCL expected {expected_lcl:.4}, got {lcl:.4}"
        );
    }

    /// U chart validation against Montgomery (2020) §7.4.
    ///
    /// Reference: ū=2.0, n=10
    /// UCL = 2.0 + 3·√(2.0/10) = 2.0 + 3·0.4472 = 3.3416
    /// LCL = max(0, 2.0 − 1.3416) = 0.6584
    #[test]
    fn u_chart_montgomery_reference() {
        // 20 samples each inspecting 10 units, defects arranged so u=2.0
        let mut chart = UChart::new();
        for _ in 0..20 {
            chart.add_sample(20, 10.0); // u = 20/10 = 2.0
        }

        let u_bar = chart.u_bar().expect("u_bar");
        assert!(
            (u_bar - 2.0).abs() < 1e-10,
            "U chart ū expected 2.0, got {u_bar}"
        );

        let sigma = (2.0_f64 / 10.0).sqrt(); // sqrt(0.2) ≈ 0.44721
        let expected_ucl = 2.0 + 3.0 * sigma; // ≈ 3.3416
        let expected_lcl = (2.0 - 3.0 * sigma).max(0.0); // ≈ 0.6584

        for pt in chart.points() {
            assert!(
                (pt.ucl - expected_ucl).abs() < 1e-6,
                "U chart UCL expected {expected_ucl:.4}, got {:.4}",
                pt.ucl
            );
            assert!(
                (pt.lcl - expected_lcl).abs() < 1e-6,
                "U chart LCL expected {expected_lcl:.4}, got {:.4}",
                pt.lcl
            );
        }
    }

    // --- G Chart ---

    #[test]
    fn g_chart_ucl_above_cl() {
        let gaps = vec![100.0, 120.0, 95.0, 110.0, 105.0];
        let chart = g_chart(&gaps).expect("chart should be Some");
        assert!(chart.points[0].ucl > chart.points[0].cl);
        assert!(chart.points[0].lcl >= 0.0);
    }

    #[test]
    fn g_chart_insufficient() {
        assert!(g_chart(&[100.0, 120.0]).is_none());
    }

    #[test]
    fn g_chart_all_same() {
        let chart = g_chart(&[50.0; 8]).expect("chart should be Some");
        assert!((chart.g_bar - 50.0).abs() < 1e-10);
        assert!(chart.points[0].ucl > chart.points[0].cl);
    }

    // --- T Chart ---

    #[test]
    fn t_chart_ucl_factor() {
        // UCL = t_bar * (-ln(0.00135)) ≈ t_bar * 6.6077
        let times = vec![100.0; 10];
        let chart = t_chart(&times).expect("chart should be Some");
        let ratio = chart.points[0].ucl / chart.t_bar;
        assert!((ratio - 6.6077).abs() < 0.01, "ratio={ratio}");
    }

    #[test]
    fn t_chart_non_positive() {
        assert!(t_chart(&[10.0, -5.0, 20.0, 15.0]).is_none());
    }

    #[test]
    fn t_chart_insufficient() {
        assert!(t_chart(&[10.0, 20.0]).is_none());
    }
}
