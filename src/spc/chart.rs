//! Core control chart types and trait.
//!
//! Defines the fundamental building blocks for all control charts: control limits,
//! chart points with violation annotations, and the [`ControlChart`] trait that
//! all variables charts implement.
//!
//! # References
//!
//! - Montgomery, D.C. (2019). *Introduction to Statistical Quality Control*, 8th ed.
//! - ASTM E2587 — Standard Practice for Use of Control Charts

use std::fmt;

/// Control limits for a chart.
///
/// Represents the upper control limit (UCL), center line (CL), and lower
/// control limit (LCL) computed from the process data.
///
/// # Invariants
///
/// - `lcl <= cl <= ucl`
/// - All values are finite
#[derive(Debug, Clone, PartialEq)]
pub struct ControlLimits {
    /// Upper control limit (UCL = CL + 3 sigma).
    pub ucl: f64,
    /// Center line (process mean or target).
    pub cl: f64,
    /// Lower control limit (LCL = CL - 3 sigma).
    pub lcl: f64,
}

/// A single point on a control chart.
///
/// Each point corresponds to a statistic computed from one subgroup or
/// individual observation, along with any run-rule violations detected
/// at that point.
#[derive(Debug, Clone)]
pub struct ChartPoint {
    /// The computed statistic value (e.g., subgroup mean, range, proportion).
    pub value: f64,
    /// The zero-based index of this point in the sequence.
    pub index: usize,
    /// List of violations detected at this point.
    pub violations: Vec<ViolationType>,
}

/// Types of control chart violations: Nelson's eight tests, and the Western
/// Electric run rule where it differs from Nelson's.
///
/// Each variant is one test for special causes of variation, and its
/// [`name`](ViolationType::name) is the name it goes by on the wire.
///
/// # Reference
///
/// Nelson, L.S. (1984). "The Shewhart Control Chart — Tests for Special Causes",
/// *Journal of Quality Technology* 16(4), pp. 237-239.
/// Western Electric (1956). *Statistical Quality Control Handbook*.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViolationType {
    /// Point beyond control limits (Nelson Rule 1).
    ///
    /// A single point falls outside the 3-sigma control limits.
    BeyondLimits,

    /// 9 points in a row on same side of center line (Nelson Rule 2).
    ///
    /// Indicates a sustained shift in the process mean.
    NineOneSide,

    /// 6 points in a row steadily increasing or decreasing (Nelson Rule 3).
    ///
    /// Indicates a trend in the process.
    SixTrend,

    /// 14 points in a row alternating up and down (Nelson Rule 4).
    ///
    /// Indicates systematic variation (e.g., two alternating streams).
    FourteenAlternating,

    /// 2 out of 3 points beyond 2 sigma on same side (Nelson Rule 5).
    ///
    /// An early warning of a potential shift.
    TwoOfThreeBeyond2Sigma,

    /// 4 out of 5 points beyond 1 sigma on same side (Nelson Rule 6).
    ///
    /// Indicates a small sustained shift.
    FourOfFiveBeyond1Sigma,

    /// 15 points in a row within 1 sigma of center line (Nelson Rule 7).
    ///
    /// Indicates stratification — reduced variation suggesting mixed streams.
    FifteenWithin1Sigma,

    /// 8 points in a row beyond 1 sigma on either side (Nelson Rule 8).
    ///
    /// Indicates a mixture pattern — points avoid the center zone.
    EightBeyond1Sigma,

    /// 8 points in a row on same side of center line (Western Electric rule 4).
    ///
    /// The Western Electric handbook's run rule; Nelson's Rule 2
    /// ([`NineOneSide`](ViolationType::NineOneSide)) asks for nine.
    EightOneSide,
}

impl ViolationType {
    /// Every test: Nelson's eight in his numbering, then Western Electric's run rule.
    pub const ALL: [ViolationType; 9] = [
        ViolationType::BeyondLimits,
        ViolationType::NineOneSide,
        ViolationType::SixTrend,
        ViolationType::FourteenAlternating,
        ViolationType::TwoOfThreeBeyond2Sigma,
        ViolationType::FourOfFiveBeyond1Sigma,
        ViolationType::FifteenWithin1Sigma,
        ViolationType::EightBeyond1Sigma,
        ViolationType::EightOneSide,
    ];

    /// The name a point's violations and a request's rules use for this test.
    pub fn name(self) -> &'static str {
        match self {
            ViolationType::BeyondLimits => "BeyondLimits",
            ViolationType::NineOneSide => "NineOneSide",
            ViolationType::SixTrend => "SixTrend",
            ViolationType::FourteenAlternating => "FourteenAlternating",
            ViolationType::TwoOfThreeBeyond2Sigma => "TwoOfThreeBeyond2Sigma",
            ViolationType::FourOfFiveBeyond1Sigma => "FourOfFiveBeyond1Sigma",
            ViolationType::FifteenWithin1Sigma => "FifteenWithin1Sigma",
            ViolationType::EightBeyond1Sigma => "EightBeyond1Sigma",
            ViolationType::EightOneSide => "EightOneSide",
        }
    }

    /// The test a [`name`](ViolationType::name) stands for.
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|v| v.name() == name)
    }
}

/// A violation detected on the chart.
///
/// Associates a specific point index with the type of violation observed.
#[derive(Debug, Clone)]
pub struct Violation {
    /// The index of the point where the violation was detected.
    pub point_index: usize,
    /// The type of violation.
    pub violation_type: ViolationType,
}

/// Trait for control charts that process subgroup or individual data.
///
/// Implementors accumulate sample data, compute control limits from the
/// accumulated data, and detect violations using run rules.
pub trait ControlChart {
    /// Add a sample (subgroup) to the chart.
    ///
    /// For subgroup charts (X-bar-R, X-bar-S), the slice contains the
    /// individual measurements within one subgroup.
    ///
    /// For individual charts (I-MR), the slice contains exactly one element.
    ///
    /// # Errors
    ///
    /// [`ControlChartError::SampleLengthMismatch`] if the slice is not the
    /// chart's sample length, or [`ControlChartError::NonFiniteValue`] if it
    /// holds a NaN or infinity. The chart is left unchanged. A rejected sample
    /// used to be dropped without a word, so every later point sat one
    /// position off the input it came from.
    fn add_sample(&mut self, sample: &[f64]) -> Result<(), ControlChartError>;

    /// Get the computed control limits, or `None` if insufficient data.
    fn control_limits(&self) -> Option<ControlLimits>;

    /// Check if the process is in statistical control.
    ///
    /// Returns `true` if no violations have been detected across all points.
    fn is_in_control(&self) -> bool;

    /// Get all violations detected across all chart points.
    fn violations(&self) -> Vec<Violation>;

    /// Get all chart points.
    fn points(&self) -> &[ChartPoint];
}

/// Smallest subgroup size the variables-chart factor tables cover.
pub const MIN_SUBGROUP_SIZE: usize = 2;

/// Largest subgroup size the variables-chart factor tables cover.
///
/// The factor tables (A2, A3, D3, D4, B3, B4, d2, c4) are indexed by subgroup
/// size, so this is the highest `n` for which control limits can be computed.
pub const MAX_SUBGROUP_SIZE: usize = 25;

/// Errors returned when a control chart cannot be constructed or cannot take
/// a sample.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum ControlChartError {
    /// A sample is not the length the chart takes: a subgroup of another size,
    /// or more than one value for an individuals chart.
    SampleLengthMismatch {
        /// The chart's sample length.
        expected: usize,
        /// The length that was given.
        got: usize,
    },
    /// A sample holds a NaN or an infinity.
    NonFiniteValue,
    /// A sample reports more defectives than it has items.
    DefectivesExceedSampleSize {
        /// Defectives reported.
        defectives: u64,
        /// Items in the sample.
        sample_size: u64,
    },
    /// Units inspected that is not a positive, finite number.
    NonPositiveUnits,
    /// The requested subgroup size is outside the range the factor tables cover.
    ///
    /// Carries the rejected size so a caller can report it without re-deriving
    /// it, and so a boundary that cannot unwind -- a WebAssembly or C entry
    /// point -- can turn it into a value rather than a panic.
    SubgroupSizeOutOfRange {
        /// The size that was asked for.
        got: usize,
        /// Smallest supported size.
        min: usize,
        /// Largest supported size.
        max: usize,
    },
    /// A sample size of zero: there is nothing to take a proportion of.
    ZeroSampleSize,
    /// A known (Phase I) parameter outside its domain: `p_bar` in (0, 1),
    /// `u_bar` > 0, `phi` >= 0, each finite.
    ///
    /// Carries the value given and the domain, so a caller can say what the
    /// parameter has to be without restating the rule.
    InvalidStandard {
        /// The parameter's name (`p_bar`, `u_bar`, `phi`).
        parameter: &'static str,
        /// What the parameter has to satisfy, for a message.
        requirement: &'static str,
        /// The value given.
        got: f64,
        /// Inclusive lower bound.
        min: Option<f64>,
        /// Exclusive lower bound.
        exclusive_min: Option<f64>,
        /// Inclusive upper bound.
        max: Option<f64>,
        /// Exclusive upper bound.
        exclusive_max: Option<f64>,
    },
}

impl ControlChartError {
    /// A known proportion outside (0, 1).
    pub(crate) fn proportion_standard(parameter: &'static str, got: f64) -> Self {
        Self::InvalidStandard {
            parameter,
            requirement: "in (0, 1)",
            got,
            min: None,
            exclusive_min: Some(0.0),
            max: None,
            exclusive_max: Some(1.0),
        }
    }

    /// A known rate that is not > 0.
    pub(crate) fn rate_standard(parameter: &'static str, got: f64) -> Self {
        Self::InvalidStandard {
            parameter,
            requirement: "> 0",
            got,
            min: None,
            exclusive_min: Some(0.0),
            max: None,
            exclusive_max: None,
        }
    }

    /// A known dispersion factor below 0.
    pub(crate) fn dispersion_standard(parameter: &'static str, got: f64) -> Self {
        Self::InvalidStandard {
            parameter,
            requirement: ">= 0",
            got,
            min: Some(0.0),
            exclusive_min: None,
            max: None,
            exclusive_max: None,
        }
    }
}

impl fmt::Display for ControlChartError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ControlChartError::SubgroupSizeOutOfRange { got, min, max } => {
                write!(f, "subgroup size must be {min}..={max}, got {got}")
            }
            ControlChartError::ZeroSampleSize => write!(f, "sample size must be at least 1"),
            ControlChartError::InvalidStandard {
                parameter,
                requirement,
                got,
                ..
            } => write!(
                f,
                "{parameter} must be a finite number {requirement}, got {got}"
            ),
            ControlChartError::SampleLengthMismatch { expected, got } => {
                write!(f, "sample has {got} values, expected {expected}")
            }
            ControlChartError::NonFiniteValue => {
                write!(f, "sample contains a value that is not a finite number")
            }
            ControlChartError::DefectivesExceedSampleSize {
                defectives,
                sample_size,
            } => write!(
                f,
                "{defectives} defectives out of a sample of {sample_size}"
            ),
            ControlChartError::NonPositiveUnits => {
                write!(f, "units inspected must be a positive, finite number")
            }
        }
    }
}

impl std::error::Error for ControlChartError {}

/// Why a chart computed from a whole slice of samples refused that slice.
///
/// Carries the position of the refused sample, which a sample-by-sample
/// [`ControlChartError`] cannot: the chart functions that take every sample
/// at once are the only place that position is known.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum ChartInputError {
    /// The sample at `index` cannot be charted.
    Sample {
        /// Zero-based position of the sample in the input slice.
        index: usize,
        /// What is wrong with it.
        error: ControlChartError,
    },
    /// Fewer samples than the chart needs.
    TooFewSamples {
        /// Samples given.
        got: usize,
        /// Samples required.
        min: usize,
    },
    /// A known (Phase I) parameter given with the samples is invalid
    /// ([`ControlChartError::InvalidStandard`]).
    Standard(ControlChartError),
}

impl fmt::Display for ChartInputError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ChartInputError::Sample { index, error } => write!(f, "sample {index}: {error}"),
            ChartInputError::TooFewSamples { got, min } => {
                write!(f, "at least {min} samples are needed, got {got}")
            }
            ChartInputError::Standard(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for ChartInputError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ChartInputError::Sample { error, .. } | ChartInputError::Standard(error) => Some(error),
            ChartInputError::TooFewSamples { .. } => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_control_limits_construction() {
        let limits = ControlLimits {
            ucl: 30.0,
            cl: 25.0,
            lcl: 20.0,
        };
        assert!((limits.ucl - 30.0).abs() < f64::EPSILON);
        assert!((limits.cl - 25.0).abs() < f64::EPSILON);
        assert!((limits.lcl - 20.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_chart_point_construction() {
        let point = ChartPoint {
            value: 25.5,
            index: 0,
            violations: vec![ViolationType::BeyondLimits],
        };
        assert!((point.value - 25.5).abs() < f64::EPSILON);
        assert_eq!(point.index, 0);
        assert_eq!(point.violations.len(), 1);
        assert_eq!(point.violations[0], ViolationType::BeyondLimits);
    }

    #[test]
    fn test_violation_type_equality() {
        assert_eq!(ViolationType::BeyondLimits, ViolationType::BeyondLimits);
        assert_ne!(ViolationType::BeyondLimits, ViolationType::NineOneSide);
    }

    #[test]
    fn test_violation_construction() {
        let v = Violation {
            point_index: 5,
            violation_type: ViolationType::SixTrend,
        };
        assert_eq!(v.point_index, 5);
        assert_eq!(v.violation_type, ViolationType::SixTrend);
    }

    #[test]
    fn test_control_limits_clone() {
        let limits = ControlLimits {
            ucl: 30.0,
            cl: 25.0,
            lcl: 20.0,
        };
        let cloned = limits.clone();
        assert_eq!(limits, cloned);
    }
}
