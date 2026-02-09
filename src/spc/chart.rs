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

/// Types of control chart violations based on Nelson's eight rules.
///
/// Each variant corresponds to one of Nelson's tests for special causes
/// of variation.
///
/// # Reference
///
/// Nelson, L.S. (1984). "The Shewhart Control Chart — Tests for Special Causes",
/// *Journal of Quality Technology* 16(4), pp. 237-239.
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
    /// For individual charts (I-MR), the slice should contain exactly
    /// one element.
    fn add_sample(&mut self, sample: &[f64]);

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
