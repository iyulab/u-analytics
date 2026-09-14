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

use super::chart::{ChartInputError, ControlChartError};

/// Fewest samples a Laney chart takes: φ is a mean of moving ranges, and one
/// moving range is not a spread estimate.
const LANEY_MIN_SAMPLES: usize = 3;

/// A sample with a proportion: at least one item, at most that many defectives.
fn check_proportion(defectives: u64, sample_size: u64) -> Result<(), ControlChartError> {
    if sample_size == 0 {
        return Err(ControlChartError::ZeroSampleSize);
    }
    if defectives > sample_size {
        return Err(ControlChartError::DefectivesExceedSampleSize {
            defectives,
            sample_size,
        });
    }
    Ok(())
}

/// An area of opportunity: a positive, finite number of units.
fn check_units(units: f64) -> Result<(), ControlChartError> {
    if units.is_finite() && units > 0.0 {
        Ok(())
    } else {
        Err(ControlChartError::NonPositiveUnits)
    }
}

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
    /// The point on the standardized scale, `(value − cl) / σᵢ`, where `σᵢ` is
    /// this point's own standard error (the limits are `cl ± 3σᵢ`, before the
    /// lower one is clamped at 0). `None` when `σᵢ` is 0.
    ///
    /// When sample sizes vary the limits vary with them, so zone-based run
    /// rules have no single set of zones on the original scale. On this scale
    /// the limits are ±3 for every point: pass these values to
    /// [`RunRule`](crate::spc::RunRule)s with limits `(3, 0, −3)` — the
    /// standardized control chart (Montgomery 2019, §7.2.2).
    pub z: Option<f64>,
}

/// A point whose limits are `cl ± 3·sigma` (lower one clamped at 0).
fn attribute_point(index: usize, value: f64, cl: f64, sigma: f64) -> AttributeChartPoint {
    let ucl = cl + 3.0 * sigma;
    let lcl = (cl - 3.0 * sigma).max(0.0);
    AttributeChartPoint {
        index,
        value,
        ucl,
        cl,
        lcl,
        out_of_control: value > ucl || value < lcl,
        z: standardized(value, cl, sigma),
    }
}

fn standardized(value: f64, cl: f64, sigma: f64) -> Option<f64> {
    (sigma > 0.0 && sigma.is_finite()).then(|| (value - cl) / sigma)
}

/// A known proportion: finite and strictly inside (0, 1).
fn check_proportion_standard(p: f64, parameter: &'static str) -> Result<f64, ControlChartError> {
    if p.is_finite() && p > 0.0 && p < 1.0 {
        Ok(p)
    } else {
        Err(ControlChartError::InvalidStandard { parameter })
    }
}

/// A known rate: finite and strictly positive.
fn check_rate_standard(u: f64, parameter: &'static str) -> Result<f64, ControlChartError> {
    if u.is_finite() && u > 0.0 {
        Ok(u)
    } else {
        Err(ControlChartError::InvalidStandard { parameter })
    }
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
/// # Phase I and Phase II
///
/// [`PChart::new`] estimates p-bar from the samples it is given (Phase I).
/// To judge later samples against an established process, build the chart
/// with [`PChart::with_center`]: every limit then uses that p-bar with each
/// sample's own n. Re-estimating from Phase I and Phase II samples together
/// would let a shift in Phase II pull the centre toward itself and widen its
/// own limits.
///
/// # Examples
///
/// ```
/// use u_analytics::spc::PChart;
///
/// let mut chart = PChart::new();
/// chart.add_sample(3, 100).unwrap();  // 3 defectives out of 100
/// chart.add_sample(5, 100).unwrap();
/// chart.add_sample(2, 100).unwrap();
/// chart.add_sample(4, 100).unwrap();
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
    /// A known centre line; `None` estimates it from the samples.
    center: Option<f64>,
}

impl PChart {
    /// Create a new P chart that estimates p-bar from its samples (Phase I).
    pub fn new() -> Self {
        Self {
            samples: Vec::new(),
            chart_points: Vec::new(),
            p_bar: None,
            center: None,
        }
    }

    /// Create a P chart whose centre line is a known p-bar, typically from a
    /// Phase I study (Phase II).
    ///
    /// # Errors
    ///
    /// [`ControlChartError::InvalidStandard`] unless `p_bar` is finite and
    /// strictly between 0 and 1 — at 0 or 1 every limit collapses onto the
    /// centre line.
    pub fn with_center(p_bar: f64) -> Result<Self, ControlChartError> {
        Ok(Self {
            center: Some(check_proportion_standard(p_bar, "p_bar")?),
            ..Self::new()
        })
    }

    /// Add a sample with the number of defective items and the total sample size.
    ///
    /// # Errors
    ///
    /// [`ControlChartError::ZeroSampleSize`](crate::spc::ControlChartError::ZeroSampleSize)
    /// if `sample_size` is 0, or
    /// [`ControlChartError::DefectivesExceedSampleSize`](crate::spc::ControlChartError::DefectivesExceedSampleSize)
    /// if `defectives > sample_size`. The chart is left unchanged; such a
    /// sample used to be dropped silently, shifting every later point.
    pub fn add_sample(
        &mut self,
        defectives: u64,
        sample_size: u64,
    ) -> Result<(), ControlChartError> {
        check_proportion(defectives, sample_size)?;
        self.samples.push((defectives, sample_size));
        self.recompute();
        Ok(())
    }

    /// The centre line in use — the known p-bar for a chart built
    /// [`with_center`](Self::with_center), otherwise the estimate — or `None`
    /// if no data.
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

        let p_bar = self.center.unwrap_or_else(|| {
            let total_defectives: u64 = self.samples.iter().map(|&(d, _)| d).sum();
            let total_inspected: u64 = self.samples.iter().map(|&(_, n)| n).sum();
            total_defectives as f64 / total_inspected as f64
        });
        self.p_bar = Some(p_bar);

        self.chart_points = self
            .samples
            .iter()
            .enumerate()
            .map(|(i, &(defectives, sample_size))| {
                let p = defectives as f64 / sample_size as f64;
                let sigma = (p_bar * (1.0 - p_bar) / sample_size as f64).sqrt();
                attribute_point(i, p, p_bar, sigma)
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
    /// # Errors
    ///
    /// Returns [`ControlChartError::ZeroSampleSize`](crate::spc::ControlChartError::ZeroSampleSize)
    /// when `sample_size == 0`. The size usually comes from data, so this is an
    /// ordinary outcome rather than a contract violation -- and a boundary that
    /// cannot unwind, such as the WebAssembly entry points, needs it as a value.
    pub fn new(sample_size: u64) -> Result<Self, ControlChartError> {
        if sample_size == 0 {
            return Err(ControlChartError::ZeroSampleSize);
        }
        Ok(Self {
            sample_size,
            defective_counts: Vec::new(),
            chart_points: Vec::new(),
            limits: None,
        })
    }

    /// Add a defective count for one subgroup.
    ///
    /// # Errors
    ///
    /// [`ControlChartError::DefectivesExceedSampleSize`](crate::spc::ControlChartError::DefectivesExceedSampleSize)
    /// if `defectives` exceeds the chart's sample size. The chart is left
    /// unchanged.
    pub fn add_sample(&mut self, defectives: u64) -> Result<(), ControlChartError> {
        if defectives > self.sample_size {
            return Err(ControlChartError::DefectivesExceedSampleSize {
                defectives,
                sample_size: self.sample_size,
            });
        }
        self.defective_counts.push(defectives);
        self.recompute();
        Ok(())
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

        self.chart_points = self
            .defective_counts
            .iter()
            .enumerate()
            .map(|(i, &count)| attribute_point(i, count as f64, np_bar, sigma))
            .collect();
        let first = &self.chart_points[0];
        self.limits = Some((first.ucl, np_bar, first.lcl));
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

        self.chart_points = self
            .defect_counts
            .iter()
            .enumerate()
            .map(|(i, &count)| attribute_point(i, count as f64, c_bar, sigma))
            .collect();
        let first = &self.chart_points[0];
        self.limits = Some((first.ucl, c_bar, first.lcl));
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
/// A known u-bar from a Phase I study is given with [`UChart::with_center`];
/// see [`PChart`] for why it is not re-estimated from later samples.
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
    /// A known centre line; `None` estimates it from the samples.
    center: Option<f64>,
}

impl UChart {
    /// Create a new U chart that estimates u-bar from its samples (Phase I).
    pub fn new() -> Self {
        Self {
            samples: Vec::new(),
            chart_points: Vec::new(),
            u_bar: None,
            center: None,
        }
    }

    /// Create a U chart whose centre line is a known u-bar (Phase II).
    ///
    /// # Errors
    ///
    /// [`ControlChartError::InvalidStandard`] unless `u_bar` is finite and
    /// positive.
    pub fn with_center(u_bar: f64) -> Result<Self, ControlChartError> {
        Ok(Self {
            center: Some(check_rate_standard(u_bar, "u_bar")?),
            ..Self::new()
        })
    }

    /// Add a sample with the number of defects and the number of units inspected.
    ///
    /// The `units_inspected` can be fractional (e.g., area or length).
    ///
    /// # Errors
    ///
    /// [`ControlChartError::NonPositiveUnits`](crate::spc::ControlChartError::NonPositiveUnits)
    /// if `units_inspected` is not a positive, finite number. The chart is left
    /// unchanged.
    pub fn add_sample(
        &mut self,
        defects: u64,
        units_inspected: f64,
    ) -> Result<(), ControlChartError> {
        check_units(units_inspected)?;
        self.samples.push((defects, units_inspected));
        self.recompute();
        Ok(())
    }

    /// The centre line in use — the known u-bar for a chart built
    /// [`with_center`](Self::with_center), otherwise the estimate — or `None`
    /// if no data.
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

        let u_bar = self.center.unwrap_or_else(|| {
            let total_defects: u64 = self.samples.iter().map(|&(d, _)| d).sum();
            let total_units: f64 = self.samples.iter().map(|&(_, n)| n).sum();
            total_defects as f64 / total_units
        });
        self.u_bar = Some(u_bar);

        self.chart_points = self
            .samples
            .iter()
            .enumerate()
            .map(|(i, &(defects, units))| {
                attribute_point(i, defects as f64 / units, u_bar, (u_bar / units).sqrt())
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
    /// The point on the standardized scale, `(value − cl) / (φ·σᵢ)` — the
    /// scale on which every limit is ±3. `None` when `φ·σᵢ` is 0. See
    /// [`AttributeChartPoint::z`] for using it with zone-based run rules.
    pub z: Option<f64>,
}

/// A centre line and φ established by a Phase I Laney study, to judge later
/// samples against (Phase II).
///
/// Both come from the same study: φ scales the standard error about *that*
/// centre, so re-estimating either one from Phase II data would let a shift
/// pull the chart toward itself.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LaneyStandard {
    /// p̄ for a P′ chart (strictly between 0 and 1), ū for a U′ chart
    /// (positive).
    pub center: f64,
    /// The sigma-inflation factor (finite, ≥ 0).
    pub phi: f64,
}

/// Laney P' chart result.
///
/// Contains the overall proportion defective, the overdispersion factor φ,
/// and the per-subgroup chart points with adjusted limits.
#[derive(Debug, Clone)]
pub struct LaneyPChart {
    /// Proportion defective at the centre line: estimated (p̄ = Σdᵢ / Σnᵢ), or
    /// the [`LaneyStandard::center`] given.
    pub p_bar: f64,
    /// Overdispersion/underdispersion correction factor φ = MR̄ / d₂, or the
    /// [`LaneyStandard::phi`] given. φ = 1.0 means no correction (ordinary P
    /// chart).
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
    /// Defect rate at the centre line: estimated (ū = Σdefectsᵢ / Σunitsᵢ), or
    /// the [`LaneyStandard::center`] given.
    pub u_bar: f64,
    /// Overdispersion/underdispersion correction factor φ = MR̄ / d₂, or the
    /// [`LaneyStandard::phi`] given. φ = 1.0 means no correction (ordinary U
    /// chart).
    pub phi: f64,
    /// Per-subgroup chart points.
    pub points: Vec<LaneyAttributePoint>,
}

/// d₂ for a moving range of two observations.
const LANEY_D2: f64 = 1.128;

/// φ = MR̄ / d₂ over the standardized statistics `(value − center) / σᵢ`.
/// `rows` are `(value, σᵢ)`; φ is 0 when any σᵢ is 0 (no variation to scale).
fn laney_phi(rows: &[(f64, f64)], center: f64) -> f64 {
    if rows.iter().any(|&(_, sigma)| sigma <= 0.0) {
        return 0.0;
    }
    let z: Vec<f64> = rows.iter().map(|&(v, s)| (v - center) / s).collect();
    let mr_bar = z.windows(2).map(|w| (w[1] - w[0]).abs()).sum::<f64>() / (z.len() - 1) as f64;
    mr_bar / LANEY_D2
}

/// Points with limits `center ± 3·φ·σᵢ` (lower clamped at 0).
fn laney_points(rows: &[(f64, f64)], center: f64, phi: f64) -> Vec<LaneyAttributePoint> {
    rows.iter()
        .enumerate()
        .map(|(index, &(value, sigma_i))| {
            let sigma = phi * sigma_i;
            let ucl = center + 3.0 * sigma;
            let lcl = (center - 3.0 * sigma).max(0.0);
            LaneyAttributePoint {
                index,
                value,
                ucl,
                cl: center,
                lcl,
                out_of_control: value > ucl || value < lcl,
                z: standardized(value, center, sigma),
            }
        })
        .collect()
}

/// Validates a Laney standard and the sample count it implies: estimating φ
/// needs `LANEY_MIN_SAMPLES`, a given φ needs one sample.
fn laney_preconditions(
    n: usize,
    standard: Option<LaneyStandard>,
    check_center: fn(f64, &'static str) -> Result<f64, ControlChartError>,
    center_name: &'static str,
) -> Result<(), ChartInputError> {
    if let Some(s) = standard {
        check_center(s.center, center_name).map_err(ChartInputError::Standard)?;
        if !(s.phi.is_finite() && s.phi >= 0.0) {
            return Err(ChartInputError::Standard(
                ControlChartError::InvalidStandard { parameter: "phi" },
            ));
        }
    }
    let min = if standard.is_some() {
        1
    } else {
        LANEY_MIN_SAMPLES
    };
    if n < min {
        return Err(ChartInputError::TooFewSamples { got: n, min });
    }
    Ok(())
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
/// With `standard` given (Phase II), steps 1–4 are skipped: p̄ and φ are the
/// Phase I values and every limit uses them with the sample's own nᵢ.
///
/// When p̄ is 0 or 1 there is no variation to scale (σ = 0): the chart comes
/// back with φ = 0 and every limit on the centre line.
///
/// # Errors
///
/// - [`ChartInputError::TooFewSamples`] for fewer than 3 samples when φ is
///   estimated (it needs at least two moving ranges), or none at all with a
///   standard.
/// - [`ChartInputError::Sample`] with
///   [`ControlChartError::ZeroSampleSize`] or
///   [`ControlChartError::DefectivesExceedSampleSize`] for the first sample
///   that has no proportion, with its position.
/// - [`ChartInputError::Standard`] for a standard p̄ outside (0, 1) or a φ that
///   is negative or not finite.
///
/// # Reference
///
/// Laney, D.B. (2002). "Improved control charts for attributes",
/// *Quality Engineering* 14(4), pp. 531-537.
pub fn laney_p_chart(
    samples: &[(u64, u64)],
    standard: Option<LaneyStandard>,
) -> Result<LaneyPChart, ChartInputError> {
    laney_preconditions(samples.len(), standard, check_proportion_standard, "p_bar")?;
    // Checked per sample, not only in total: one sample with nothing inspected
    // makes its z-score NaN, and a NaN phi puts every limit at NaN -- where no
    // point compares as out of control and the chart reads as in control.
    for (index, &(defectives, sample_size)) in samples.iter().enumerate() {
        if let Err(error) = check_proportion(defectives, sample_size) {
            return Err(ChartInputError::Sample { index, error });
        }
    }

    let p_bar = standard.map_or_else(
        || {
            let total_defectives: u64 = samples.iter().map(|&(d, _)| d).sum();
            let total_inspected: u64 = samples.iter().map(|&(_, n)| n).sum();
            total_defectives as f64 / total_inspected as f64
        },
        |s| s.center,
    );
    let base_var = p_bar * (1.0 - p_bar);
    let rows: Vec<(f64, f64)> = samples
        .iter()
        .map(|&(d, n)| (d as f64 / n as f64, (base_var / n as f64).sqrt()))
        .collect();
    let phi = standard.map_or_else(|| laney_phi(&rows, p_bar), |s| s.phi);

    Ok(LaneyPChart {
        p_bar,
        phi,
        points: laney_points(&rows, p_bar, phi),
    })
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
/// With `standard` given (Phase II), steps 1–4 are skipped. When ū is 0 there
/// is no variation to scale: φ = 0 and every limit is 0.
///
/// # Errors
///
/// - [`ChartInputError::TooFewSamples`] for fewer than 3 samples when φ is
///   estimated, or none at all with a standard.
/// - [`ChartInputError::Sample`] with [`ControlChartError::NonPositiveUnits`]
///   for the first sample whose units are not a positive, finite number, with
///   its position.
/// - [`ChartInputError::Standard`] for a standard ū that is not positive or a
///   φ that is negative or not finite.
///
/// # Reference
///
/// Laney, D.B. (2002). "Improved control charts for attributes",
/// *Quality Engineering* 14(4), pp. 531-537.
pub fn laney_u_chart(
    samples: &[(u64, f64)],
    standard: Option<LaneyStandard>,
) -> Result<LaneyUChart, ChartInputError> {
    laney_preconditions(samples.len(), standard, check_rate_standard, "u_bar")?;
    for (index, &(_, units)) in samples.iter().enumerate() {
        if let Err(error) = check_units(units) {
            return Err(ChartInputError::Sample { index, error });
        }
    }

    let u_bar = standard.map_or_else(
        || {
            let total_defects: u64 = samples.iter().map(|&(d, _)| d).sum();
            // Every term is positive and finite, so the total is too.
            let total_units: f64 = samples.iter().map(|&(_, n)| n).sum();
            total_defects as f64 / total_units
        },
        |s| s.center,
    );
    let rows: Vec<(f64, f64)> = samples
        .iter()
        .map(|&(d, n)| (d as f64 / n, (u_bar / n).sqrt()))
        .collect();
    let phi = standard.map_or_else(|| laney_phi(&rows, u_bar), |s| s.phi);

    Ok(LaneyUChart {
        u_bar,
        phi,
        points: laney_points(&rows, u_bar, phi),
    })
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
/// # References
///
/// - Kaminsky, F.C. et al. (1992). "Statistical control charts based on a
///   geometric distribution", *Journal of Quality Technology* 24(2), pp. 63-69.
/// - Benneyan, J.C. (2001). "Number-Between g-Type Statistical Quality Control
///   Charts for Healthcare Applications", *Health Care Management Science* 4(4),
///   pp. 305-318.
/// - Woodall, W.H. (2006). "The Use of Control Charts in Health-Care and
///   Public-Health Surveillance", *Journal of Quality Technology* 38(2),
///   pp. 89-104.
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
/// The spread term √(ḡ·(ḡ+1)) follows directly from the variance of the
/// geometric distribution: Var(G) = (1−p)/p² ≈ ḡ·(ḡ+1) when p is small.
///
/// # Returns
///
/// `None` if fewer than 3 observations or any count is negative.
///
/// # References
///
/// - Kaminsky, F.C. et al. (1992). "Statistical control charts based on a
///   geometric distribution", *Journal of Quality Technology* 24(2), pp. 63-69.
/// - Benneyan, J.C. (2001). "Number-Between g-Type Statistical Quality Control
///   Charts for Healthcare Applications", *Health Care Management Science* 4(4),
///   pp. 305-318.
/// - Woodall, W.H. (2006). "The Use of Control Charts in Health-Care and
///   Public-Health Surveillance", *Journal of Quality Technology* 38(2),
///   pp. 89-104.
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
            chart.add_sample(d, 100).unwrap();
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
        chart.add_sample(10, 100).unwrap();

        let pt = &chart.points()[0];
        assert!((pt.cl - 0.1).abs() < 1e-10);
        assert!((pt.ucl - 0.19).abs() < 0.001);
        assert!((pt.lcl - 0.01).abs() < 0.001);
    }

    #[test]
    fn test_p_chart_variable_sample_sizes() {
        let mut chart = PChart::new();
        chart.add_sample(5, 100).unwrap();
        chart.add_sample(10, 200).unwrap();
        chart.add_sample(3, 50).unwrap();

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
        use crate::spc::ControlChartError;
        let mut chart = PChart::new();
        assert_eq!(
            chart.add_sample(5, 0),
            Err(ControlChartError::ZeroSampleSize)
        );
        assert_eq!(
            chart.add_sample(10, 5),
            Err(ControlChartError::DefectivesExceedSampleSize {
                defectives: 10,
                sample_size: 5
            })
        );
        assert!(chart.p_bar().is_none());
    }

    #[test]
    fn test_p_chart_lcl_clamped_to_zero() {
        let mut chart = PChart::new();
        // Very small p with small n → LCL would be negative
        chart.add_sample(1, 10).unwrap();
        let pt = &chart.points()[0];
        assert!(pt.lcl >= 0.0);
    }

    #[test]
    fn test_p_chart_out_of_control() {
        let mut chart = PChart::new();
        // Establish baseline with many normal samples
        for _ in 0..20 {
            chart.add_sample(5, 100).unwrap();
        }
        // Add an outlier
        chart.add_sample(30, 100).unwrap();

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
        let mut chart = NPChart::new(100).expect("100 is a valid sample size");
        let defectives = [5, 8, 3, 6, 4, 7, 2, 9, 5, 6];
        for &d in &defectives {
            chart.add_sample(d).unwrap();
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
        let mut chart = NPChart::new(100).expect("100 is a valid sample size");
        assert_eq!(
            chart.add_sample(101),
            Err(crate::spc::ControlChartError::DefectivesExceedSampleSize {
                defectives: 101,
                sample_size: 100
            })
        );
        assert!(chart.control_limits().is_none());
    }

    #[test]
    fn test_np_chart_zero_sample_size() {
        assert!(matches!(
            NPChart::new(0),
            Err(crate::spc::ControlChartError::ZeroSampleSize)
        ));
    }

    #[test]
    fn test_np_chart_out_of_control() {
        let mut chart = NPChart::new(100).expect("100 is a valid sample size");
        for _ in 0..20 {
            chart.add_sample(5).unwrap();
        }
        chart.add_sample(30).unwrap();

        assert!(!chart.is_in_control());
    }

    #[test]
    fn test_np_chart_limits_formula() {
        // n=200, p-bar = 0.05 → np-bar = 10
        // sigma = sqrt(200 * 0.05 * 0.95) = sqrt(9.5) ≈ 3.082
        // UCL = 10 + 3*3.082 = 19.246
        // LCL = 10 - 3*3.082 = 0.754
        let mut chart = NPChart::new(200).expect("200 is a valid sample size");
        for _ in 0..10 {
            chart.add_sample(10).unwrap();
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
        chart.add_sample(3, 10.0).unwrap();
        chart.add_sample(5, 10.0).unwrap();
        chart.add_sample(4, 10.0).unwrap();
        chart.add_sample(6, 10.0).unwrap();
        chart.add_sample(2, 10.0).unwrap();

        let u_bar = chart.u_bar().expect("should have u_bar");
        // u-bar = 20/50 = 0.4
        assert!((u_bar - 0.4).abs() < 1e-10);

        assert_eq!(chart.points().len(), 5);
    }

    #[test]
    fn test_u_chart_variable_units() {
        let mut chart = UChart::new();
        chart.add_sample(10, 5.0).unwrap(); // u = 2.0
        chart.add_sample(20, 10.0).unwrap(); // u = 2.0
        chart.add_sample(5, 2.5).unwrap(); // u = 2.0

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
        for units in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert_eq!(
                chart.add_sample(5, units),
                Err(crate::spc::ControlChartError::NonPositiveUnits),
                "units = {units}"
            );
        }
        assert!(chart.u_bar().is_none());
    }

    #[test]
    fn test_u_chart_out_of_control() {
        let mut chart = UChart::new();
        for _ in 0..20 {
            chart.add_sample(4, 10.0).unwrap();
        }
        chart.add_sample(50, 10.0).unwrap(); // Far outlier

        assert!(!chart.is_in_control());
    }

    #[test]
    fn test_u_chart_lcl_clamped() {
        let mut chart = UChart::new();
        // Small u-bar with small n → LCL would be negative
        chart.add_sample(1, 1.0).unwrap();
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
        chart.add_sample(8, 4.0).unwrap();

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
        let mut np_chart = NPChart::new(100).expect("100 is a valid sample size");

        let defectives = [5, 8, 3, 6, 4];
        for &d in &defectives {
            p_chart.add_sample(d, 100).unwrap();
            np_chart.add_sample(d).unwrap();
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
            u_chart.add_sample(d, 1.0).unwrap();
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
        let chart = laney_p_chart(&samples, None).expect("valid samples");
        assert!(chart.phi > 0.0);
        assert!(chart.p_bar > 0.0 && chart.p_bar < 1.0);
        assert_eq!(chart.points.len(), 10);
    }

    #[test]
    fn laney_p_constant_proportion_phi_near_zero() {
        // All samples identical → all z-scores = 0 → MR = 0 → phi = 0
        let samples: Vec<(u64, u64)> = vec![(10, 1000); 20];
        let chart = laney_p_chart(&samples, None).expect("valid samples");
        assert!((chart.p_bar - 0.01).abs() < 1e-10);
        assert!(chart.phi >= 0.0);
    }

    #[test]
    fn laney_p_ucl_above_lcl() {
        let samples: Vec<(u64, u64)> = vec![(5, 100), (8, 100), (3, 100), (6, 100), (4, 100)];
        let chart = laney_p_chart(&samples, None).expect("valid samples");
        for p in &chart.points {
            assert!(p.ucl >= p.lcl);
            assert!((p.cl - chart.p_bar).abs() < 1e-10);
        }
    }

    #[test]
    fn laney_p_insufficient_data() {
        let samples: Vec<(u64, u64)> = vec![(2, 100), (3, 100)];
        assert_eq!(
            laney_p_chart(&samples, None).unwrap_err(),
            ChartInputError::TooFewSamples { got: 2, min: 3 }
        );
    }

    #[test]
    fn laney_p_rejects_a_sample_with_no_proportion() {
        // One sample with nothing inspected divides by zero: its z-score is not
        // a number, so phi and every limit become NaN, no point compares as out
        // of control, and the chart reads as in control.
        assert_eq!(
            laney_p_chart(&[(3, 100), (0, 0), (4, 100), (2, 100)], None).unwrap_err(),
            ChartInputError::Sample {
                index: 1,
                error: ControlChartError::ZeroSampleSize
            }
        );
        // More defectives than inspected is not a proportion either — and the
        // position reported is that sample's, not the first sample's.
        assert_eq!(
            laney_p_chart(&[(3, 100), (4, 100), (12, 10), (2, 100)], None).unwrap_err(),
            ChartInputError::Sample {
                index: 2,
                error: ControlChartError::DefectivesExceedSampleSize {
                    defectives: 12,
                    sample_size: 10
                }
            }
        );
    }

    // --- Phase I / Phase II ---

    #[test]
    fn laney_p_phase_two_judges_a_point_against_phase_one() {
        // Phase I: p̄ = 0.0514, φ = 0.236. A Phase II sample of 11 / 170
        // (p = 0.0647) sits outside UCL = p̄ + 3·φ·√(p̄(1−p̄)/170) ≈ 0.0634.
        let standard = LaneyStandard {
            center: 0.0514,
            phi: 0.236,
        };
        let chart = laney_p_chart(&[(11, 170)], Some(standard)).expect("one sample suffices");
        assert_eq!(
            (chart.p_bar, chart.phi),
            (0.0514, 0.236),
            "the standard, not an estimate"
        );
        let p = &chart.points[0];
        let expected_ucl = 0.0514 + 3.0 * 0.236 * (0.0514 * (1.0 - 0.0514) / 170.0_f64).sqrt();
        assert!((p.ucl - expected_ucl).abs() < 1e-15);
        assert!((p.ucl - 0.0634).abs() < 5e-5, "ucl = {}", p.ucl);
        assert!(p.out_of_control);
    }

    #[test]
    fn re_estimating_across_phases_lets_a_shift_widen_its_own_limits() {
        // Negative control for the test above: pooling a shifted Phase II with
        // Phase I moves the centre toward the shift and loosens the verdict.
        let phase_one: Vec<(u64, u64)> = [(8, 160), (9, 170), (7, 150), (10, 180), (8, 165)].into();
        let pooled_input: Vec<(u64, u64)> = phase_one
            .iter()
            .copied()
            .chain([(30, 170), (32, 175)])
            .collect();
        let p1 = laney_p_chart(&phase_one, None).expect("phase I");
        let pooled = laney_p_chart(&pooled_input, None).expect("pooled");
        let phase_two = laney_p_chart(
            &[(30, 170), (32, 175)],
            Some(LaneyStandard {
                center: p1.p_bar,
                phi: p1.phi,
            }),
        )
        .expect("phase II");
        assert!(
            pooled.p_bar > p1.p_bar,
            "the shift pulls the pooled centre up"
        );
        assert!(phase_two.points.iter().all(|p| p.cl == p1.p_bar));
        assert!(pooled.points[5].ucl > phase_two.points[0].ucl);
    }

    #[test]
    fn p_and_u_charts_take_a_known_center() {
        let mut p = PChart::with_center(0.05).expect("valid p̄");
        p.add_sample(2, 100).unwrap();
        p.add_sample(20, 100).unwrap();
        assert_eq!(p.p_bar(), Some(0.05), "not re-estimated from the samples");
        let sigma = (0.05_f64 * 0.95 / 100.0).sqrt();
        assert!((p.points()[1].ucl - (0.05 + 3.0 * sigma)).abs() < 1e-15);
        assert!(p.points()[1].out_of_control);

        let mut u = UChart::with_center(2.0).expect("valid ū");
        u.add_sample(9, 2.0).unwrap();
        assert_eq!(u.u_bar(), Some(2.0));
        assert!((u.points()[0].ucl - (2.0 + 3.0 * (2.0_f64 / 2.0).sqrt())).abs() < 1e-15);
    }

    #[test]
    fn a_standard_outside_its_domain_is_refused() {
        for p in [0.0, 1.0, -0.1, f64::NAN] {
            assert_eq!(
                PChart::with_center(p).err(),
                Some(ControlChartError::InvalidStandard { parameter: "p_bar" }),
                "p̄ = {p}"
            );
        }
        assert!(UChart::with_center(0.0).is_err());
        let bad_phi = LaneyStandard {
            center: 0.1,
            phi: -1.0,
        };
        assert_eq!(
            laney_p_chart(&[(1, 10)], Some(bad_phi)).unwrap_err(),
            ChartInputError::Standard(ControlChartError::InvalidStandard { parameter: "phi" })
        );
        let good = LaneyStandard {
            center: 0.1,
            phi: 1.0,
        };
        assert_eq!(
            laney_u_chart(&[], Some(good)).unwrap_err(),
            ChartInputError::TooFewSamples { got: 0, min: 1 }
        );
    }

    #[test]
    fn z_is_the_point_on_the_scale_where_every_limit_is_three() {
        // Varying n gives varying limits; on the standardized scale they are
        // all ±3, so run rules have one set of zones.
        let mut p = PChart::new();
        for (d, n) in [(3, 100), (12, 180), (1, 60), (9, 150), (4, 90)] {
            p.add_sample(d, n).unwrap();
        }
        for pt in p.points() {
            let sigma = (pt.ucl - pt.cl) / 3.0;
            let z = pt.z.expect("p̄ strictly inside (0, 1)");
            assert!((pt.cl + z * sigma - pt.value).abs() < 1e-12);
            if pt.lcl > 0.0 {
                assert_eq!(pt.out_of_control, z.abs() > 3.0);
            }
        }
        let laney =
            laney_p_chart(&[(3, 100), (12, 180), (1, 60), (9, 150), (4, 90)], None).expect("valid");
        for pt in &laney.points {
            let z = pt.z.expect("φ > 0");
            assert!((pt.cl + z * (pt.ucl - pt.cl) / 3.0 - pt.value).abs() < 1e-12);
        }
        // No variation to scale: no standardized value.
        let flat = laney_p_chart(&[(0, 10), (0, 12), (0, 9)], None).expect("valid");
        assert!(flat.points.iter().all(|p| p.z.is_none()));
    }

    #[test]
    fn laney_u_rejects_units_by_position() {
        assert_eq!(
            laney_u_chart(&[(3, 10.0), (4, 10.0), (2, 10.0), (5, 0.0)], None).unwrap_err(),
            ChartInputError::Sample {
                index: 3,
                error: ControlChartError::NonPositiveUnits
            }
        );
        assert_eq!(
            laney_u_chart(&[(3, 10.0)], None).unwrap_err(),
            ChartInputError::TooFewSamples { got: 1, min: 3 }
        );
    }
    #[test]
    fn laney_u_basic() {
        let samples: Vec<(u64, f64)> = vec![(5, 10.0); 10];
        let chart = laney_u_chart(&samples, None).expect("valid samples");
        assert!((chart.u_bar - 0.5).abs() < 1e-10);
        assert!(chart.phi >= 0.0);
    }

    #[test]
    fn laney_u_ucl_above_cl() {
        let samples: Vec<(u64, f64)> = (0..8).map(|i| ((i % 4 + 2) as u64, 10.0)).collect();
        let chart = laney_u_chart(&samples, None).expect("valid samples");
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
            chart.add_sample(10, 100).unwrap();
        }
        chart.add_sample(8, 100).unwrap();

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
        let mut chart = NPChart::new(100).expect("100 is a valid sample size");
        for _ in 0..19 {
            chart.add_sample(10).unwrap();
        }
        chart.add_sample(8).unwrap();

        let (ucl, cl, lcl) = chart.control_limits().expect("limits");
        // np̄ = 9.9
        assert!((cl - 9.9).abs() < 1e-10, "NP CL expected 9.9, got {cl}");
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
            chart.add_sample(20, 10.0).unwrap(); // u = 20/10 = 2.0
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
        let chart = g_chart(&gaps).expect("valid samples");
        assert!(chart.points[0].ucl > chart.points[0].cl);
        assert!(chart.points[0].lcl >= 0.0);
    }

    #[test]
    fn g_chart_insufficient() {
        assert!(g_chart(&[100.0, 120.0]).is_none());
    }

    #[test]
    fn g_chart_all_same() {
        let chart = g_chart(&[50.0; 8]).expect("valid samples");
        assert!((chart.g_bar - 50.0).abs() < 1e-10);
        assert!(chart.points[0].ucl > chart.points[0].cl);
    }

    // --- T Chart ---

    #[test]
    fn t_chart_ucl_factor() {
        // UCL = t_bar * (-ln(0.00135)) ≈ t_bar * 6.6077
        let times = vec![100.0; 10];
        let chart = t_chart(&times).expect("valid samples");
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

    // --- Laney P' invariant: UCL = p̄ + 3·φ·σᵢ for every point ---
    //
    // The Laney P' UCL formula is exactly: UCL_i = p̄ + 3·φ·√(p̄·(1−p̄)/nᵢ).
    // This invariant must hold for every subgroup regardless of φ.
    //
    // Reference: Laney (2002) §3, steps 5–6.
    #[test]
    fn laney_p_ucl_formula_invariant() {
        let samples: Vec<(u64, u64)> = vec![
            (3, 100),
            (7, 100),
            (2, 100),
            (8, 100),
            (4, 100),
            (5, 150),
            (9, 150),
            (3, 150),
            (6, 150),
            (4, 150),
        ];
        let chart = laney_p_chart(&samples, None).expect("valid samples");

        for (i, (&(d, n), pt)) in samples.iter().zip(&chart.points).enumerate() {
            let p_i = d as f64 / n as f64;
            let sigma_i = (chart.p_bar * (1.0 - chart.p_bar) / n as f64).sqrt();
            let expected_ucl = chart.p_bar + 3.0 * chart.phi * sigma_i;
            let expected_lcl = (chart.p_bar - 3.0 * chart.phi * sigma_i).max(0.0);

            assert!(
                (pt.value - p_i).abs() < 1e-10,
                "point {i}: value expected {p_i:.6}, got {:.6}",
                pt.value
            );
            assert!(
                (pt.ucl - expected_ucl).abs() < 1e-10,
                "point {i}: UCL expected {expected_ucl:.6}, got {:.6}",
                pt.ucl
            );
            assert!(
                (pt.lcl - expected_lcl).abs() < 1e-10,
                "point {i}: LCL expected {expected_lcl:.6}, got {:.6}",
                pt.lcl
            );
        }
    }

    // --- Laney P' invariant: φ=1 means Laney limits = standard P chart limits ---
    //
    // When all z-scores are identical, MR=0 → φ=0, not φ=1.
    // To get φ=1 we need to engineer data where MR̄=d₂=1.128.
    // Strategy: use 4 samples with exactly two z-values (+δ, -δ, +δ, -δ)
    // so that |Δz|=2δ for all 3 consecutive pairs.
    // MR̄ = 2δ → φ = 2δ/1.128.  For φ=1: δ = 0.564.
    //
    // We need integer counts (d, n) such that z = (d/n - p̄)/σ = ±0.564 exactly.
    // Choose p̄ = 0.5, n = 10000: σ = √(0.25/10000) = 0.005.
    // p_high = 0.5 + 0.564·0.005 = 0.5028  → d_high = 5028
    // p_low  = 0.5 − 0.564·0.005 = 0.4972  → d_low  = 4972
    // Verify: z_high = (0.5028-0.5)/0.005 = 0.56 (but MR̄=2·0.56=1.12, φ=0.9929)
    // The integer representation does not produce φ=1 exactly; we therefore
    // verify only that φ is close to 1 and that the UCL/LCL formula is self-consistent.
    //
    // Reference: Laney (2002) §3.
    #[test]
    fn laney_p_phi_near_one_limits_close_to_standard() {
        // δ = 0.564 → p_high/low with n=10000 and p̄=0.5 produce z = ±0.56
        // (integer rounding: 5028/10000, 4972/10000)
        // MR̄ = 2·0.56 = 1.12, φ = 1.12/1.128 ≈ 0.993 (very close to 1.0)
        let n: u64 = 10_000;
        let d_high: u64 = 5028; // z ≈ +0.56
        let d_low: u64 = 4972; // z ≈ -0.56
                               // 6 samples alternating: φ = MR̄/1.128 with MR̄ from 5 identical MRs of 1.12
        let samples: Vec<(u64, u64)> = vec![
            (d_high, n),
            (d_low, n),
            (d_high, n),
            (d_low, n),
            (d_high, n),
            (d_low, n),
        ];

        let laney = laney_p_chart(&samples, None).expect("valid samples");

        // φ should be close to 1 (within 1%).
        assert!(
            (laney.phi - 1.0).abs() < 0.01,
            "φ expected ≈1.0, got {}",
            laney.phi
        );

        // Standard P chart limits for same data at n=10000.
        let p_bar = laney.p_bar;
        let sigma_std = (p_bar * (1.0 - p_bar) / n as f64).sqrt();
        let std_ucl = p_bar + 3.0 * sigma_std;
        let std_lcl = (p_bar - 3.0 * sigma_std).max(0.0);

        // Laney UCL must equal standard UCL scaled by φ; since φ≈1, the
        // difference is bounded by 3·σ·|φ−1|.
        let max_deviation = 3.0 * sigma_std * (laney.phi - 1.0).abs();
        assert!(
            (laney.points[0].ucl - std_ucl).abs() <= max_deviation + 1e-12,
            "Laney UCL={:.6} vs std UCL={std_ucl:.6}, deviation bound={max_deviation:.2e}",
            laney.points[0].ucl
        );
        assert!(
            (laney.points[0].lcl - std_lcl).abs() <= max_deviation + 1e-12,
            "Laney LCL={:.6} vs std LCL={std_lcl:.6}, deviation bound={max_deviation:.2e}",
            laney.points[0].lcl
        );
    }

    // --- Laney P' invariant: φ > 1 → wider limits than standard P chart ---
    //
    // Overdispersed data (variance greater than binomial prediction) forces
    // φ > 1.  The Laney UCL must then exceed the standard P chart UCL.
    //
    // Reference: Laney (2002) §2–3.
    #[test]
    fn laney_p_phi_gt_one_wider_than_standard() {
        // Strongly overdispersed: proportions swing far from p̄.
        let samples: Vec<(u64, u64)> = vec![
            (1, 100),
            (20, 100),
            (2, 100),
            (18, 100),
            (1, 100),
            (22, 100),
            (3, 100),
            (19, 100),
            (2, 100),
            (20, 100),
        ];

        let laney = laney_p_chart(&samples, None).expect("valid samples");
        assert!(
            laney.phi > 1.0,
            "φ expected > 1 for overdispersed data, got {}",
            laney.phi
        );

        // Standard P chart UCL for the same data.
        let total_d: u64 = samples.iter().map(|&(d, _)| d).sum();
        let total_n: u64 = samples.iter().map(|&(_, n)| n).sum();
        let p_bar = total_d as f64 / total_n as f64;
        let std_ucl_first = p_bar + 3.0 * (p_bar * (1.0 - p_bar) / 100.0_f64).sqrt();

        assert!(
            laney.points[0].ucl > std_ucl_first,
            "Laney UCL ({:.4}) must exceed standard P chart UCL ({std_ucl_first:.4}) when φ>1",
            laney.points[0].ucl
        );
    }

    // --- Laney P' numerical reference test ---
    //
    // Hand-computed example:
    //   5 samples of n=50, defectives = [3, 7, 2, 8, 4]
    //   p̄ = 24/250 = 0.096
    //   p_i = [0.06, 0.14, 0.04, 0.16, 0.08]
    //   σ_i = √(0.096·0.904/50) = √(0.001737...) ≈ 0.041681
    //   z_i = (p_i − 0.096) / 0.041681
    //        = [−0.8636, +1.0557, −1.3467, +1.5393, −0.3832]
    //   MR = [|z1−z0|, |z2−z1|, |z3−z2|, |z4−z3|]
    //       = [1.9193, 2.4024, 2.8860, 1.9225]
    //   MR̄ = (1.9193+2.4024+2.8860+1.9225)/4 = 9.1302/4 = 2.28255
    //   φ = 2.28255 / 1.128 = 2.0235
    //   UCL_0 = 0.096 + 3·2.0235·0.041681 = 0.096 + 0.25296 = 0.34896
    //   LCL_0 = max(0, 0.096 − 0.25296) = 0  (negative → 0)
    //
    // Reference: Laney (2002) §3 algorithm steps 1–6.
    #[test]
    fn laney_p_numerical_reference() {
        let samples: Vec<(u64, u64)> = vec![(3, 50), (7, 50), (2, 50), (8, 50), (4, 50)];
        let chart = laney_p_chart(&samples, None).expect("valid samples");

        let p_bar = 24.0_f64 / 250.0;
        assert!(
            (chart.p_bar - p_bar).abs() < 1e-10,
            "p̄ expected {p_bar:.6}, got {:.6}",
            chart.p_bar
        );

        // Recompute φ from first principles to validate implementation.
        let sigma_i = (p_bar * (1.0 - p_bar) / 50.0_f64).sqrt();
        let p_vals = [0.06_f64, 0.14, 0.04, 0.16, 0.08];
        let z: Vec<f64> = p_vals.iter().map(|&p| (p - p_bar) / sigma_i).collect();
        let mr_bar = z.windows(2).map(|w| (w[1] - w[0]).abs()).sum::<f64>() / 4.0;
        let phi_expected = mr_bar / 1.128;

        assert!(
            (chart.phi - phi_expected).abs() < 1e-10,
            "φ expected {phi_expected:.6}, got {:.6}",
            chart.phi
        );

        // UCL for point 0 (n=50).
        let ucl_0 = p_bar + 3.0 * phi_expected * sigma_i;
        let lcl_0 = (p_bar - 3.0 * phi_expected * sigma_i).max(0.0);
        assert!(
            (chart.points[0].ucl - ucl_0).abs() < 1e-10,
            "UCL[0] expected {ucl_0:.6}, got {:.6}",
            chart.points[0].ucl
        );
        assert!(
            (chart.points[0].lcl - lcl_0).abs() < 1e-10,
            "LCL[0] expected {lcl_0:.6}, got {:.6}",
            chart.points[0].lcl
        );
    }

    // --- Laney U' invariant: φ > 1 → wider limits than standard U chart ---
    //
    // Reference: Laney (2002) §4 (U' extension).
    #[test]
    fn laney_u_phi_gt_one_wider_than_standard() {
        // Overdispersed U data: defect rates swing from low to high.
        let samples: Vec<(u64, f64)> = vec![
            (1, 10.0),
            (15, 10.0),
            (2, 10.0),
            (14, 10.0),
            (1, 10.0),
            (16, 10.0),
            (2, 10.0),
            (13, 10.0),
        ];

        let laney = laney_u_chart(&samples, None).expect("valid samples");
        assert!(
            laney.phi > 1.0,
            "φ expected > 1 for overdispersed data, got {}",
            laney.phi
        );

        let total_d: u64 = samples.iter().map(|&(d, _)| d).sum();
        let total_n: f64 = samples.iter().map(|&(_, n)| n).sum();
        let u_bar = total_d as f64 / total_n;
        let std_ucl = u_bar + 3.0 * (u_bar / 10.0_f64).sqrt();

        assert!(
            laney.points[0].ucl > std_ucl,
            "Laney U' UCL ({:.4}) must exceed standard U chart UCL ({std_ucl:.4}) when φ>1",
            laney.points[0].ucl
        );
    }

    // --- G chart formula verification ---
    //
    // Kaminsky (1992) formula: UCL = ḡ + 3·√(ḡ·(ḡ+1)), LCL = max(0, ḡ − 3·√(ḡ·(ḡ+1)))
    // The spread √(ḡ·(ḡ+1)) comes from the geometric distribution variance
    // Var(G) = (1−p)/p² where p = 1/(ḡ+1), giving √((ḡ+1)·ḡ) = √(ḡ·(ḡ+1)).
    //
    // Numerical check with ḡ = 50.0:
    //   spread = √(50·51) = √2550 ≈ 50.4975
    //   UCL = 50 + 3·50.4975 ≈ 201.4925
    //   LCL = max(0, 50 − 151.4925) = 0
    //
    // Reference: Kaminsky et al. (1992) §3; Benneyan (2001) §2.
    #[test]
    fn g_chart_formula_verification() {
        // 8 identical observations so ḡ = 50.0 exactly.
        let gaps = vec![50.0_f64; 8];
        let chart = g_chart(&gaps).expect("valid samples");

        let g_bar = 50.0_f64;
        assert!(
            (chart.g_bar - g_bar).abs() < 1e-10,
            "ḡ expected 50.0, got {}",
            chart.g_bar
        );

        let spread = (g_bar * (g_bar + 1.0)).sqrt(); // √(50·51) = √2550
        let expected_ucl = g_bar + 3.0 * spread;
        let expected_lcl = (g_bar - 3.0 * spread).max(0.0);

        assert!(
            (chart.points[0].ucl - expected_ucl).abs() < 1e-10,
            "UCL expected {expected_ucl:.6}, got {:.6}",
            chart.points[0].ucl
        );
        assert!(
            (chart.points[0].lcl - expected_lcl).abs() < 1e-10,
            "LCL expected {expected_lcl:.6}, got {:.6}",
            chart.points[0].lcl
        );
        // LCL is 0 because ḡ - 3·spread < 0 for ḡ = 50.
        assert!(
            chart.points[0].lcl >= 0.0,
            "LCL must be clamped to 0, got {}",
            chart.points[0].lcl
        );
    }

    // --- G chart spread > ḡ for any positive ḡ ---
    //
    // Invariant: √(ḡ·(ḡ+1)) > ḡ  ⟺  ḡ+1 > ḡ  ⟺  true.
    // Therefore LCL is always 0 for any ḡ > 0 (the lower 3σ band is negative).
    #[test]
    fn g_chart_lcl_always_zero() {
        for &g in &[1.0_f64, 5.0, 20.0, 100.0, 500.0] {
            let gaps = vec![g; 5];
            let chart = g_chart(&gaps).expect("valid samples");
            assert!(
                (chart.points[0].lcl - 0.0).abs() < 1e-10,
                "LCL must be 0 for ḡ={g}, got {}",
                chart.points[0].lcl
            );
        }
    }

    // --- T chart LCL factor verification ---
    //
    // The T chart LCL factor is −ln(0.99865) ≈ 0.001351 (not exactly 0.00135).
    // This test pins the exact numeric value.
    //
    // Exponential quantile: Q(p; θ) = θ·(−ln(1−p)) where θ = t̄.
    //   LCL factor = −ln(1 − 0.00135) = −ln(0.99865) ≈ 0.0013509
    //   UCL factor = −ln(1 − 0.99865) = −ln(0.00135) ≈ 6.6077
    //
    // Reference: Borror, Keats & Montgomery (2003) §2.
    #[test]
    fn t_chart_lcl_factor_verification() {
        let times = vec![100.0_f64; 10];
        let chart = t_chart(&times).expect("valid samples");

        let ucl_factor = chart.points[0].ucl / chart.t_bar;
        let lcl_factor = chart.points[0].lcl / chart.t_bar;

        let expected_ucl_factor = -(0.00135_f64.ln()); // ≈ 6.6077
        let expected_lcl_factor = -(0.99865_f64.ln()); // ≈ 0.001351

        assert!(
            (ucl_factor - expected_ucl_factor).abs() < 1e-10,
            "UCL factor expected {expected_ucl_factor:.6}, got {ucl_factor:.6}"
        );
        assert!(
            (lcl_factor - expected_lcl_factor).abs() < 1e-10,
            "LCL factor expected {expected_lcl_factor:.6}, got {lcl_factor:.6}"
        );
    }

    // --- T chart invariant: UCL / LCL ratio is constant ---
    //
    // UCL = t̄ · k_U,  LCL = t̄ · k_L  →  UCL/LCL = k_U/k_L = constant.
    // k_U/k_L = −ln(0.00135) / −ln(0.99865) ≈ 4893.
    // This is a scale-invariant property of the exponential distribution.
    #[test]
    fn t_chart_ucl_lcl_ratio_scale_invariant() {
        let k_u = -(0.00135_f64.ln());
        let k_l = -(0.99865_f64.ln());
        let expected_ratio = k_u / k_l;

        for &t_bar in &[10.0_f64, 100.0, 1000.0] {
            let times = vec![t_bar; 10];
            let chart = t_chart(&times).expect("valid samples");
            let ratio = chart.points[0].ucl / chart.points[0].lcl;
            assert!(
                (ratio - expected_ratio).abs() < 0.01,
                "UCL/LCL ratio expected {expected_ratio:.2}, got {ratio:.2} at t̄={t_bar}"
            );
        }
    }
}
