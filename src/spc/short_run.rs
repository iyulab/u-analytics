//! Short-run control charts: one chart for a process that runs in groups.
//!
//! A process that makes several kinds of item in short runs -- a part number,
//! a recipe, a grade -- has a target and a spread of its own for each kind.
//! Plotting the raw values on one chart mixes the populations: the limits
//! widen to cover the distance between the targets, and a shift inside one
//! kind hides among the others. The remedy is to put every observation on a
//! scale common to all groups first, and chart *that*:
//!
//! | Chart | Plotted value | Assumes |
//! |-------|---------------|---------|
//! | [`deviation_chart`] (variable aim, "DNOM") | `x − target` | one spread for all groups |
//! | [`standardized_chart`] (Z-MR) | `(x − target) / spread` | a spread per group |
//! | [`standardized_p_chart`] | `(p − p̄) / sqrt(p̄(1 − p̄)/n)` | a centre per group |
//! | [`standardized_u_chart`] | `(u − ū) / sqrt(ū/a)` | a centre per group |
//!
//! The variables charts are individuals-and-moving-range charts built on
//! [`IndividualMRChart`] with the centre line fixed at the aim (0 on the
//! transformed scale), so the limits are the ones every I-MR chart has --
//! there is no second set of limit formulas to keep in step. The attributes
//! charts reuse the points of the P and U charts, each against the centre of
//! *its own* group.
//!
//! # Groups
//!
//! A group is an index `g` into the `targets` (and `spreads`, `centers`)
//! slice, so the slice length is the number of groups; every observation names
//! the group it belongs to. A group no observation names is allowed and
//! ignored.
//!
//! # Estimating a spread
//!
//! A spread the caller does not know is estimated per group from the moving
//! ranges of *that group's own* successive observations (in the order given,
//! however the groups interleave), `MR-bar / d2`. A moving range that spans
//! two groups is never used: it measures the distance between targets, not
//! the spread. A group needs at least [`MIN_GROUP_OBSERVATIONS`] observations
//! for this -- two moving ranges, below which an average of ranges is not a
//! spread estimate. Estimating from the data being charted makes the limits
//! approximate (the estimate is itself uncertain); fix the spread from an
//! earlier study with `GroupStandard::Known` when one exists.
//!
//! # References
//!
//! - Bothe, D.R. (1989). "A powerful new control chart for job shop
//!   manufacturers", *Quality Progress* 22(12).
//! - Wheeler, D.J. (1991). *Short Run SPC*. SPC Press.

use std::fmt;

use super::attributes::{
    attribute_point, check_proportion, check_proportion_standard, check_rate_standard, check_units,
    AttributeChartPoint,
};
use super::chart::{ChartPoint, ControlChart, ControlChartError, ControlLimits};
use super::rules::RuleSet;
use super::variables::{IndividualMRChart, D2};

/// Fewest observations a group needs for its spread to be estimated: two
/// moving ranges.
pub const MIN_GROUP_OBSERVATIONS: usize = 3;

/// Fewest samples a group needs for its centre line to be estimated: with one
/// sample a group's own centre equals the sample and every standardized value
/// is 0.
pub const MIN_GROUP_SAMPLES: usize = 2;

/// A per-group quantity that is either known in advance or estimated.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GroupStandard<'a> {
    /// Known per group, indexed by group: a spread from an earlier study, or a
    /// proportion or rate the group is run to.
    Known(&'a [f64]),
    /// Estimated from the observations of each group.
    Estimated,
}

/// Why a short-run chart could not be built.
#[derive(Debug, Clone, PartialEq)]
pub enum ShortRunError {
    /// Two slices that must be the same length are not.
    LengthMismatch {
        /// Name of the slice that is the wrong length.
        parameter: &'static str,
        /// The length it has to be.
        expected: usize,
        /// The length it is.
        got: usize,
    },
    /// An observation or a target that is NaN or infinite.
    NonFinite {
        /// Name of the slice.
        parameter: &'static str,
        /// Position within it.
        index: usize,
    },
    /// An observation names a group beyond the targets.
    GroupOutOfRange {
        /// Position of the observation.
        index: usize,
        /// The group it names.
        group: usize,
        /// Number of groups there are.
        groups: usize,
    },
    /// A known spread, proportion, rate or sigma outside its domain.
    InvalidStandard {
        /// The group it belongs to; `None` for the one `sigma` of a deviation
        /// chart.
        group: Option<usize>,
        /// The domain error, naming the parameter and what it has to satisfy.
        error: ControlChartError,
    },
    /// An observation or sample the chart refuses (zero sample size, more
    /// defectives than items, units not positive).
    Sample {
        /// Position of the sample.
        index: usize,
        /// Why.
        error: ControlChartError,
    },
    /// Fewer observations than the chart needs, overall or in one group.
    TooFew {
        /// The group, or `None` for the input as a whole.
        group: Option<usize>,
        /// How many are needed.
        min: usize,
        /// How many there are.
        got: usize,
    },
    /// A group whose spread (or centre) the data cannot give: the successive
    /// values of the group are all equal.
    Degenerate {
        /// The group.
        group: usize,
    },
}

impl fmt::Display for ShortRunError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LengthMismatch {
                parameter,
                expected,
                got,
            } => write!(f, "{parameter} has {got} entries, expected {expected}"),
            Self::NonFinite { parameter, index } => {
                write!(f, "{parameter}[{index}] is not a finite number")
            }
            Self::GroupOutOfRange {
                index,
                group,
                groups,
            } => write!(
                f,
                "observation {index} names group {group}, but there are {groups} groups"
            ),
            Self::InvalidStandard {
                group: Some(group),
                error,
            } => write!(f, "group {group}: {error}"),
            Self::InvalidStandard { group: None, error } => write!(f, "{error}"),
            Self::Sample { index, error } => write!(f, "sample {index}: {error}"),
            Self::TooFew {
                group: Some(group),
                min,
                got,
            } => write!(
                f,
                "group {group} has {got} observations, at least {min} are needed"
            ),
            Self::TooFew {
                group: None,
                min,
                got,
            } => write!(f, "{got} observations, at least {min} are needed"),
            Self::Degenerate { group } => write!(
                f,
                "group {group}: successive values are all equal, so no spread can be estimated"
            ),
        }
    }
}

impl std::error::Error for ShortRunError {}

/// The scale a variables short-run chart plots.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShortRunScale {
    /// `x − target`, in the units of the measurement.
    Deviation,
    /// `(x − target) / spread`, a pure number: the limits are ±3.
    Standardized,
}

/// A variables short-run chart: an individuals-and-moving-range chart of the
/// transformed values.
#[derive(Debug, Clone)]
pub struct ShortRunChart {
    /// What the points are.
    pub scale: ShortRunScale,
    /// The group of each observation, in the order given.
    pub groups: Vec<usize>,
    /// Limits of the individuals chart: centre 0, the aim on the plotted scale.
    pub limits: ControlLimits,
    /// Limits of the moving-range chart.
    pub mr_limits: ControlLimits,
    /// The plotted values with the run tests' violations.
    pub points: Vec<ChartPoint>,
    /// Moving ranges of the plotted values in the order given (the first
    /// point has none), so they start at index 1. Successive points of
    /// different groups are included: on the plotted scale the groups share
    /// one centre and one spread.
    pub mr_points: Vec<ChartPoint>,
    /// The sigma of the plotted values the limits use: 1 for a standardized
    /// chart, the given or moving-range estimate for a deviation chart.
    pub sigma_hat: f64,
    /// Spread used for each group (`None` for a group no observation names).
    /// Empty for a deviation chart, whose spread is shared.
    pub group_spreads: Vec<Option<f64>>,
}

impl ShortRunChart {
    /// Whether no point of the individuals chart or the moving-range chart
    /// carries a violation.
    pub fn is_in_control(&self) -> bool {
        self.points
            .iter()
            .chain(&self.mr_points)
            .all(|p| p.violations.is_empty())
    }
}

/// An attributes short-run chart: every point against the centre of its own
/// group, the standardized value `z` on the common scale where the limits are
/// ±3.
#[derive(Debug, Clone)]
pub struct ShortRunAttributeChart {
    /// The group of each sample, in the order given.
    pub groups: Vec<usize>,
    /// Centre line used for each group (`None` for a group no sample names).
    pub group_centers: Vec<Option<f64>>,
    /// Points on the original scale (each with its group's centre and limits)
    /// and their standardized value [`AttributeChartPoint::z`].
    pub points: Vec<AttributeChartPoint>,
}

impl ShortRunAttributeChart {
    /// Whether no point is beyond its limits.
    pub fn is_in_control(&self) -> bool {
        self.points.iter().all(|p| !p.out_of_control)
    }
}

/// Checks the pieces every variables chart shares: lengths, finiteness and
/// group indices. Returns the number of groups.
fn check_input(values: &[f64], groups: &[usize], targets: &[f64]) -> Result<usize, ShortRunError> {
    if groups.len() != values.len() {
        return Err(ShortRunError::LengthMismatch {
            parameter: "groups",
            expected: values.len(),
            got: groups.len(),
        });
    }
    if let Some(index) = values.iter().position(|v| !v.is_finite()) {
        return Err(ShortRunError::NonFinite {
            parameter: "values",
            index,
        });
    }
    if let Some(index) = targets.iter().position(|v| !v.is_finite()) {
        return Err(ShortRunError::NonFinite {
            parameter: "targets",
            index,
        });
    }
    check_groups(groups, targets.len())?;
    Ok(targets.len())
}

fn check_groups(groups: &[usize], n_groups: usize) -> Result<(), ShortRunError> {
    match groups.iter().position(|&g| g >= n_groups) {
        Some(index) => Err(ShortRunError::GroupOutOfRange {
            index,
            group: groups[index],
            groups: n_groups,
        }),
        None => Ok(()),
    }
}

/// Spread of each group from the moving ranges of its own successive values.
fn estimate_spreads(
    values: &[f64],
    groups: &[usize],
    n_groups: usize,
) -> Result<Vec<Option<f64>>, ShortRunError> {
    let mut last: Vec<Option<f64>> = vec![None; n_groups];
    let mut range_sum = vec![0.0_f64; n_groups];
    let mut count = vec![0_usize; n_groups];
    for (&x, &g) in values.iter().zip(groups) {
        if let Some(prev) = last[g] {
            range_sum[g] += (x - prev).abs();
        }
        last[g] = Some(x);
        count[g] += 1;
    }
    (0..n_groups)
        .map(|g| match count[g] {
            0 => Ok(None),
            n if n < MIN_GROUP_OBSERVATIONS => Err(ShortRunError::TooFew {
                group: Some(g),
                min: MIN_GROUP_OBSERVATIONS,
                got: n,
            }),
            n => {
                let spread = range_sum[g] / (n - 1) as f64 / D2[0];
                if spread > 0.0 {
                    Ok(Some(spread))
                } else {
                    Err(ShortRunError::Degenerate { group: g })
                }
            }
        })
        .collect()
}

/// Builds the I-MR chart of already transformed values, centred at 0.
fn imr_of(
    transformed: &[f64],
    rules: RuleSet,
    sigma: Option<f64>,
    scale: ShortRunScale,
    groups: &[usize],
    group_spreads: Vec<Option<f64>>,
) -> Result<ShortRunChart, ShortRunError> {
    if transformed.len() < 2 {
        return Err(ShortRunError::TooFew {
            group: None,
            min: 2,
            got: transformed.len(),
        });
    }
    let mut chart = IndividualMRChart::new()
        .with_rules(rules)
        .with_center(0.0)
        .expect("0 is a finite centre");
    if let Some(sigma) = sigma {
        chart = chart
            .with_sigma(sigma)
            .map_err(|error| ShortRunError::InvalidStandard { group: None, error })?;
    }
    for x in transformed {
        chart
            .add_sample(std::slice::from_ref(x))
            .expect("transformed values are finite");
    }
    Ok(ShortRunChart {
        scale,
        groups: groups.to_vec(),
        limits: chart
            .control_limits()
            .expect("two observations give limits"),
        mr_limits: chart.mr_limits().expect("two observations give limits"),
        points: chart.points().to_vec(),
        mr_points: chart.mr_points().to_vec(),
        sigma_hat: chart.sigma_hat().expect("two observations give limits"),
        group_spreads,
    })
}

/// Variable-aim ("deviation from nominal") chart: `x − target` of each
/// observation on one I-MR chart centred at 0.
///
/// Every group is assumed to vary alike -- the chart pools the moving ranges
/// of the deviations, so a group with a markedly different spread belongs on
/// a [`standardized_chart`] instead. `sigma` fixes the sigma of the
/// deviations (for example from an earlier study); otherwise it is estimated
/// from their moving ranges.
///
/// # Errors
///
/// [`ShortRunError`] for slices of unequal length, a value or target that is
/// not finite, a group index beyond `targets`, fewer than two observations, or
/// a `sigma` that is not finite and positive.
///
/// # Examples
///
/// ```
/// use u_analytics::spc::{deviation_chart, RuleSet};
///
/// // Two part numbers, aimed at 10 and 20.
/// let values = [10.2, 20.1, 9.9, 19.8, 10.1, 20.2];
/// let groups = [0, 1, 0, 1, 0, 1];
/// let chart = deviation_chart(&values, &groups, &[10.0, 20.0], RuleSet::default(), None)
///     .expect("six observations");
/// assert_eq!(chart.limits.cl, 0.0);
/// assert!(chart.is_in_control());
/// ```
pub fn deviation_chart(
    values: &[f64],
    groups: &[usize],
    targets: &[f64],
    rules: RuleSet,
    sigma: Option<f64>,
) -> Result<ShortRunChart, ShortRunError> {
    check_input(values, groups, targets)?;
    let deviations: Vec<f64> = values
        .iter()
        .zip(groups)
        .map(|(&x, &g)| x - targets[g])
        .collect();
    imr_of(
        &deviations,
        rules,
        sigma,
        ShortRunScale::Deviation,
        groups,
        Vec::new(),
    )
}

/// Standardized (Z-MR) chart: `(x − target) / spread` of each observation on
/// one I-MR chart centred at 0 with sigma 1, so the individuals limits are
/// ±3.
///
/// `spreads` is [`GroupStandard::Known`] (a spread per group) or
/// [`GroupStandard::Estimated`] as `MR-bar / d2` over the moving ranges of
/// *that group's own* successive values in the order given (a range across two
/// groups measures the distance between targets and is never used; a group
/// needs at least [`MIN_GROUP_OBSERVATIONS`] of them). Estimating from the data
/// being charted makes the limits approximate. The chart's sigma is 1 by
/// construction.
///
/// # Errors
///
/// [`ShortRunError`] for the conditions of [`deviation_chart`], a known spread
/// that is not finite and positive, a group with fewer than
/// [`MIN_GROUP_OBSERVATIONS`] observations when spreads are estimated, or a
/// group whose successive values are all equal.
///
/// # Examples
///
/// ```
/// use u_analytics::spc::{standardized_chart, GroupStandard, RuleSet};
///
/// // Part 0 is aimed at 10 and varies by about 0.2; part 1 is aimed at 20
/// // and varies by about 0.1.
/// let values = [10.2, 20.1, 10.0, 19.9, 10.1, 20.1];
/// let groups = [0, 1, 0, 1, 0, 1];
/// let chart = standardized_chart(
///     &values,
///     &groups,
///     &[10.0, 20.0],
///     GroupStandard::Known(&[0.2, 0.1]),
///     RuleSet::default(),
/// )
/// .expect("six observations");
/// assert_eq!(chart.limits.ucl, 3.0);
/// assert!((chart.points[0].value - 1.0).abs() < 1e-12);
/// ```
pub fn standardized_chart(
    values: &[f64],
    groups: &[usize],
    targets: &[f64],
    spreads: GroupStandard<'_>,
    rules: RuleSet,
) -> Result<ShortRunChart, ShortRunError> {
    let n_groups = check_input(values, groups, targets)?;
    let group_spreads: Vec<Option<f64>> = match spreads {
        GroupStandard::Known(known) => {
            if known.len() != n_groups {
                return Err(ShortRunError::LengthMismatch {
                    parameter: "spreads",
                    expected: n_groups,
                    got: known.len(),
                });
            }
            for (group, &s) in known.iter().enumerate() {
                if !(s.is_finite() && s > 0.0) {
                    return Err(ShortRunError::InvalidStandard {
                        group: Some(group),
                        error: ControlChartError::rate_standard("spreads", s),
                    });
                }
            }
            let mut used = vec![false; n_groups];
            for &g in groups {
                used[g] = true;
            }
            known
                .iter()
                .zip(used)
                .map(|(&s, used)| used.then_some(s))
                .collect()
        }
        GroupStandard::Estimated => estimate_spreads(values, groups, n_groups)?,
    };
    let z: Vec<f64> = values
        .iter()
        .zip(groups)
        .map(|(&x, &g)| {
            let spread = group_spreads[g].expect("a named group has a spread");
            (x - targets[g]) / spread
        })
        .collect();
    imr_of(
        &z,
        rules,
        Some(1.0),
        ShortRunScale::Standardized,
        groups,
        group_spreads,
    )
}

/// Resolves the per-group centres of an attributes chart. `observed` is each
/// group's `(events, exposure)` totals.
fn resolve_centers(
    centers: GroupStandard<'_>,
    groups: &[usize],
    n_groups: usize,
    totals: &[(f64, f64)],
    counts: &[usize],
    check: fn(f64, &'static str) -> Result<f64, ControlChartError>,
    parameter: &'static str,
) -> Result<Vec<Option<f64>>, ShortRunError> {
    let mut used = vec![false; n_groups];
    for &g in groups {
        used[g] = true;
    }
    match centers {
        GroupStandard::Known(known) => {
            if known.len() != n_groups {
                return Err(ShortRunError::LengthMismatch {
                    parameter: "centers",
                    expected: n_groups,
                    got: known.len(),
                });
            }
            for (group, &c) in known.iter().enumerate() {
                check(c, parameter).map_err(|error| ShortRunError::InvalidStandard {
                    group: Some(group),
                    error,
                })?;
            }
            Ok(known
                .iter()
                .zip(used)
                .map(|(&c, used)| used.then_some(c))
                .collect())
        }
        GroupStandard::Estimated => (0..n_groups)
            .map(|g| {
                if !used[g] {
                    return Ok(None);
                }
                if counts[g] < MIN_GROUP_SAMPLES {
                    return Err(ShortRunError::TooFew {
                        group: Some(g),
                        min: MIN_GROUP_SAMPLES,
                        got: counts[g],
                    });
                }
                let (events, exposure) = totals[g];
                let center = events / exposure;
                // A group with no events at all (or, for a proportion, all
                // events) has centre 0 (or 1) and so no standard error.
                if center > 0.0 {
                    Ok(Some(center))
                } else {
                    Err(ShortRunError::Degenerate { group: g })
                }
            })
            .collect(),
    }
}

/// Standardized P chart: each sample's proportion against the proportion
/// defective of its own group, `z = (p − p̄ᵍ) / sqrt(p̄ᵍ (1 − p̄ᵍ) / n)`.
///
/// `defectives[i]` of `sample_sizes[i]` belong to group `groups[i]`. Centres
/// are [`GroupStandard::Known`] (each strictly between 0 and 1) or
/// [`GroupStandard::Estimated`] as the group's total defectives over its total
/// inspected (at least [`MIN_GROUP_SAMPLES`] samples). Limits on the original
/// scale are `p̄ᵍ ± 3 σ` (lower clamped at 0), on the `z` scale ±3.
///
/// # Errors
///
/// [`ShortRunError`] for slices of unequal length, a group index beyond the
/// centres, a sample the P chart refuses, a known centre outside (0, 1), a
/// group with too few samples to estimate its centre, or an estimated centre
/// of 0 or 1.
///
/// # Examples
///
/// ```
/// use u_analytics::spc::{standardized_p_chart, GroupStandard};
///
/// let chart = standardized_p_chart(
///     &[10, 25, 9],
///     &[100, 50, 100],
///     &[0, 1, 0],
///     GroupStandard::Known(&[0.1, 0.5]),
/// )
/// .expect("three samples");
/// // 10 of 100 is exactly group 0's proportion.
/// assert_eq!(chart.points[0].z, Some(0.0));
/// assert!(chart.is_in_control());
/// ```
pub fn standardized_p_chart(
    defectives: &[u64],
    sample_sizes: &[u64],
    groups: &[usize],
    centers: GroupStandard<'_>,
) -> Result<ShortRunAttributeChart, ShortRunError> {
    let n_groups = match centers {
        GroupStandard::Known(known) => known.len(),
        GroupStandard::Estimated => groups.iter().map(|&g| g + 1).max().unwrap_or(0),
    };
    for (parameter, len) in [
        ("sample_sizes", sample_sizes.len()),
        ("groups", groups.len()),
    ] {
        if len != defectives.len() {
            return Err(ShortRunError::LengthMismatch {
                parameter,
                expected: defectives.len(),
                got: len,
            });
        }
    }
    check_groups(groups, n_groups)?;
    let mut totals = vec![(0.0_f64, 0.0_f64); n_groups];
    let mut counts = vec![0_usize; n_groups];
    for (index, ((&d, &n), &g)) in defectives.iter().zip(sample_sizes).zip(groups).enumerate() {
        check_proportion(d, n).map_err(|error| ShortRunError::Sample { index, error })?;
        totals[g].0 += d as f64;
        totals[g].1 += n as f64;
        counts[g] += 1;
    }
    let group_centers = resolve_centers(
        centers,
        groups,
        n_groups,
        &totals,
        &counts,
        check_proportion_standard,
        "p_bar",
    )?;
    // An estimated centre of 1 has no standard error either.
    if let Some(group) = group_centers
        .iter()
        .position(|c| matches!(c, Some(c) if *c >= 1.0))
    {
        return Err(ShortRunError::Degenerate { group });
    }
    let points = defectives
        .iter()
        .zip(sample_sizes)
        .zip(groups)
        .enumerate()
        .map(|(index, ((&d, &n), &g))| {
            let center = group_centers[g].expect("a named group has a centre");
            let sigma = (center * (1.0 - center) / n as f64).sqrt();
            attribute_point(index, d as f64 / n as f64, center, sigma)
        })
        .collect();
    Ok(ShortRunAttributeChart {
        groups: groups.to_vec(),
        group_centers,
        points,
    })
}

/// Standardized U chart: each sample's defect rate against the rate of its own
/// group, `z = (u − ūᵍ) / sqrt(ūᵍ / a)` for `a` units inspected.
///
/// `defects[i]` over `units[i]` belong to group `groups[i]`. Centres are
/// [`GroupStandard::Known`] (each positive) or [`GroupStandard::Estimated`] as
/// the group's total defects over its total units (at least
/// [`MIN_GROUP_SAMPLES`] samples). Limits on the original scale are
/// `ūᵍ ± 3 σ` (lower clamped at 0), on the `z` scale ±3.
///
/// # Errors
///
/// [`ShortRunError`] for slices of unequal length, a group index beyond the
/// centres, units that are not positive and finite, a known centre that is not
/// positive and finite, a group with too few samples to estimate its centre,
/// or an estimated centre of 0.
///
/// # Examples
///
/// ```
/// use u_analytics::spc::{standardized_u_chart, GroupStandard};
///
/// let chart = standardized_u_chart(
///     &[4, 30],
///     &[2.0, 3.0],
///     &[0, 1],
///     GroupStandard::Known(&[2.0, 10.0]),
/// )
/// .expect("two samples");
/// assert_eq!(chart.points[0].z, Some(0.0));
/// assert_eq!(chart.points[1].z, Some(0.0));
/// ```
pub fn standardized_u_chart(
    defects: &[u64],
    units: &[f64],
    groups: &[usize],
    centers: GroupStandard<'_>,
) -> Result<ShortRunAttributeChart, ShortRunError> {
    let n_groups = match centers {
        GroupStandard::Known(known) => known.len(),
        GroupStandard::Estimated => groups.iter().map(|&g| g + 1).max().unwrap_or(0),
    };
    for (parameter, len) in [("units", units.len()), ("groups", groups.len())] {
        if len != defects.len() {
            return Err(ShortRunError::LengthMismatch {
                parameter,
                expected: defects.len(),
                got: len,
            });
        }
    }
    check_groups(groups, n_groups)?;
    let mut totals = vec![(0.0_f64, 0.0_f64); n_groups];
    let mut counts = vec![0_usize; n_groups];
    for (index, ((&d, &a), &g)) in defects.iter().zip(units).zip(groups).enumerate() {
        check_units(a).map_err(|error| ShortRunError::Sample { index, error })?;
        totals[g].0 += d as f64;
        totals[g].1 += a;
        counts[g] += 1;
    }
    let group_centers = resolve_centers(
        centers,
        groups,
        n_groups,
        &totals,
        &counts,
        check_rate_standard,
        "u_bar",
    )?;
    let points = defects
        .iter()
        .zip(units)
        .zip(groups)
        .enumerate()
        .map(|(index, ((&d, &a), &g))| {
            let center = group_centers[g].expect("a named group has a centre");
            let sigma = (center / a).sqrt();
            attribute_point(index, d as f64 / a, center, sigma)
        })
        .collect();
    Ok(ShortRunAttributeChart {
        groups: groups.to_vec(),
        group_centers,
        points,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const E: f64 = 1e-12;

    fn close(a: f64, b: f64) {
        assert!((a - b).abs() < E, "{a} != {b}");
    }

    #[test]
    fn deviation_is_x_minus_target_on_one_centred_chart() {
        // Deviations: 0.2, 0.1, -0.1, -0.2, 0.1, 0.2.
        let values = [10.2, 20.1, 9.9, 19.8, 10.1, 20.2];
        let groups = [0, 1, 0, 1, 0, 1];
        let c = deviation_chart(&values, &groups, &[10.0, 20.0], RuleSet::default(), None)
            .expect("chart");
        let devs = [0.2, 0.1, -0.1, -0.2, 0.1, 0.2];
        for (p, d) in c.points.iter().zip(devs) {
            close(p.value, d);
        }
        assert_eq!(c.scale, ShortRunScale::Deviation);
        assert_eq!(c.limits.cl, 0.0);
        // Moving ranges 0.1, 0.2, 0.1, 0.3, 0.1 -> mean 0.16.
        close(c.mr_limits.cl, 0.16);
        close(c.sigma_hat, 0.16 / D2[0]);
        // The same limits an I-MR chart of the deviations has about 0.
        close(c.limits.ucl, -c.limits.lcl);
        assert!(c.group_spreads.is_empty());
        assert!(c.is_in_control());
    }

    #[test]
    fn deviation_chart_keeps_a_shift_the_raw_chart_hides() {
        // Group 1 sits 0.5 above its aim for the last five of ten runs.
        let mut values = Vec::new();
        let mut groups = Vec::new();
        for i in 0..10 {
            let g = i % 2;
            let aim = if g == 0 { 10.0 } else { 20.0 };
            let wobble = if (i / 2) % 2 == 0 { 0.05 } else { -0.05 };
            let shift = if g == 1 && i >= 5 { 0.5 } else { 0.0 };
            values.push(aim + wobble + shift);
            groups.push(g);
        }
        let c = deviation_chart(
            &values,
            &groups,
            &[10.0, 20.0],
            RuleSet::default(),
            Some(0.1),
        )
        .expect("chart");
        // 0.5 / 0.1 = 5 sigma > 3 sigma.
        assert!(c.points.iter().any(|p| !p.violations.is_empty()));
    }

    #[test]
    fn standardized_known_spreads_are_hand_computed() {
        let values = [10.2, 20.1, 10.0, 19.9, 10.1, 20.1];
        let groups = [0, 1, 0, 1, 0, 1];
        let c = standardized_chart(
            &values,
            &groups,
            &[10.0, 20.0],
            GroupStandard::Known(&[0.2, 0.1]),
            RuleSet::default(),
        )
        .expect("chart");
        let z = [1.0, 1.0, 0.0, -1.0, 0.5, 1.0];
        for (p, z) in c.points.iter().zip(z) {
            close(p.value, z);
        }
        assert_eq!(c.scale, ShortRunScale::Standardized);
        close(c.limits.ucl, 3.0);
        close(c.limits.cl, 0.0);
        close(c.limits.lcl, -3.0);
        close(c.sigma_hat, 1.0);
        close(c.mr_limits.cl, D2[0]);
        assert_eq!(c.group_spreads, vec![Some(0.2), Some(0.1)]);
        assert_eq!(c.groups, groups);
    }

    #[test]
    fn standardized_estimated_spreads_use_each_groups_own_ranges() {
        // Group 0: 10.0, 10.2, 10.1 -> ranges 0.2, 0.1 (mean 0.15).
        // Group 1: 20.0, 20.4, 20.2 -> ranges 0.4, 0.2 (mean 0.30).
        // The runs interleave, so ranges across groups (about 10) are not used.
        let values = [10.0, 20.0, 10.2, 20.4, 10.1, 20.2];
        let groups = [0, 1, 0, 1, 0, 1];
        let c = standardized_chart(
            &values,
            &groups,
            &[10.0, 20.0],
            GroupStandard::Estimated,
            RuleSet::default(),
        )
        .expect("chart");
        let s0 = 0.15 / D2[0];
        let s1 = 0.30 / D2[0];
        let spreads = c.group_spreads.clone();
        close(spreads[0].expect("group 0"), s0);
        close(spreads[1].expect("group 1"), s1);
        close(c.points[2].value, 0.2 / s0);
        close(c.points[3].value, 0.4 / s1);
    }

    #[test]
    fn estimating_needs_three_observations_per_group() {
        let e = standardized_chart(
            &[10.0, 20.0, 10.2, 20.4, 10.1],
            &[0, 1, 0, 1, 0],
            &[10.0, 20.0],
            GroupStandard::Estimated,
            RuleSet::default(),
        )
        .expect_err("group 1 has two");
        assert_eq!(
            e,
            ShortRunError::TooFew {
                group: Some(1),
                min: MIN_GROUP_OBSERVATIONS,
                got: 2
            }
        );
    }

    #[test]
    fn constant_group_has_no_spread_to_estimate() {
        let e = standardized_chart(
            &[10.0, 10.0, 10.0, 20.0, 20.5, 20.2],
            &[0, 0, 0, 1, 1, 1],
            &[10.0, 20.0],
            GroupStandard::Estimated,
            RuleSet::default(),
        )
        .expect_err("group 0 never moves");
        assert_eq!(e, ShortRunError::Degenerate { group: 0 });
    }

    #[test]
    fn unused_groups_are_ignored() {
        let c = standardized_chart(
            &[10.2, 10.0, 10.1],
            &[0, 0, 0],
            &[10.0, 20.0, 30.0],
            GroupStandard::Estimated,
            RuleSet::default(),
        )
        .expect("only group 0 is named");
        assert_eq!(c.group_spreads[1], None);
        assert_eq!(c.group_spreads[2], None);
    }

    #[test]
    fn refusals_name_what_is_wrong() {
        let r = RuleSet::default();
        let t = [0.0, 1.0];
        assert_eq!(
            deviation_chart(&[1.0, 2.0], &[0], &t, r, None).expect_err("lengths"),
            ShortRunError::LengthMismatch {
                parameter: "groups",
                expected: 2,
                got: 1
            }
        );
        assert_eq!(
            deviation_chart(&[1.0, f64::NAN], &[0, 1], &t, r, None).expect_err("nan"),
            ShortRunError::NonFinite {
                parameter: "values",
                index: 1
            }
        );
        assert_eq!(
            deviation_chart(&[1.0, 2.0], &[0, 2], &t, r, None).expect_err("group"),
            ShortRunError::GroupOutOfRange {
                index: 1,
                group: 2,
                groups: 2
            }
        );
        assert_eq!(
            deviation_chart(&[1.0], &[0], &t, r, None).expect_err("one value"),
            ShortRunError::TooFew {
                group: None,
                min: 2,
                got: 1
            }
        );
        assert!(matches!(
            standardized_chart(
                &[1.0, 2.0],
                &[0, 1],
                &t,
                GroupStandard::Known(&[1.0, 0.0]),
                r
            ),
            Err(ShortRunError::InvalidStandard { group: Some(1), .. })
        ));
        assert_eq!(
            standardized_chart(&[1.0, 2.0], &[0, 1], &t, GroupStandard::Known(&[1.0]), r)
                .expect_err("spreads length"),
            ShortRunError::LengthMismatch {
                parameter: "spreads",
                expected: 2,
                got: 1
            }
        );
    }

    #[test]
    fn standardized_p_known_and_estimated() {
        // Group 0 runs at 10 %: 10/100 -> z 0; 19/100 -> sigma 0.03 -> z 3.
        // Group 1 runs at 50 %: 25/50 -> z 0.
        let c = standardized_p_chart(
            &[10, 25, 19, 20],
            &[100, 50, 100, 100],
            &[0, 1, 0, 0],
            GroupStandard::Known(&[0.1, 0.5]),
        )
        .expect("chart");
        close(c.points[0].z.expect("z"), 0.0);
        close(c.points[1].z.expect("z"), 0.0);
        close(c.points[2].z.expect("z"), 3.0);
        close(c.points[3].z.expect("z"), 10.0 / 3.0);
        assert!(c.points[3].out_of_control);
        assert!(!c.is_in_control());
        close(c.points[3].ucl, 0.1 + 3.0 * 0.03);

        // Estimated: group 0 has 5/100 and 15/100 -> centre 20/200 = 0.1.
        let e = standardized_p_chart(
            &[5, 15, 3, 4],
            &[100, 100, 10, 10],
            &[0, 0, 1, 1],
            GroupStandard::Estimated,
        )
        .expect("chart");
        close(e.group_centers[0].expect("c0"), 0.1);
        close(e.group_centers[1].expect("c1"), 0.35);
        close(e.points[0].z.expect("z"), (0.05 - 0.1) / 0.03);
        close(e.points[1].z.expect("z"), (0.15 - 0.1) / 0.03);
    }

    #[test]
    fn standardized_p_refusals() {
        assert_eq!(
            standardized_p_chart(&[1], &[10], &[0], GroupStandard::Estimated)
                .expect_err("one sample"),
            ShortRunError::TooFew {
                group: Some(0),
                min: MIN_GROUP_SAMPLES,
                got: 1
            }
        );
        assert_eq!(
            standardized_p_chart(&[0, 0], &[10, 10], &[0, 0], GroupStandard::Estimated)
                .expect_err("no defectives"),
            ShortRunError::Degenerate { group: 0 }
        );
        assert_eq!(
            standardized_p_chart(&[10, 10], &[10, 10], &[0, 0], GroupStandard::Estimated)
                .expect_err("all defective"),
            ShortRunError::Degenerate { group: 0 }
        );
        assert!(matches!(
            standardized_p_chart(&[11], &[10], &[0], GroupStandard::Known(&[0.1])),
            Err(ShortRunError::Sample { index: 0, .. })
        ));
        assert!(matches!(
            standardized_p_chart(&[1], &[10], &[0], GroupStandard::Known(&[1.0])),
            Err(ShortRunError::InvalidStandard { group: Some(0), .. })
        ));
        assert_eq!(
            standardized_p_chart(&[1], &[10], &[1], GroupStandard::Known(&[0.1]))
                .expect_err("group"),
            ShortRunError::GroupOutOfRange {
                index: 0,
                group: 1,
                groups: 1
            }
        );
    }

    #[test]
    fn standardized_u_known_and_estimated() {
        // Group 0 centre 2.0, 2 units -> sigma 1.0; 8 defects = 4.0 -> z 2.
        let c = standardized_u_chart(
            &[8, 30],
            &[2.0, 3.0],
            &[0, 1],
            GroupStandard::Known(&[2.0, 10.0]),
        )
        .expect("chart");
        close(c.points[0].z.expect("z"), 2.0);
        close(c.points[1].z.expect("z"), 0.0);

        // Estimated: group 0 has 4/2 units and 8/2 units -> 12/4 = 3.0.
        let e = standardized_u_chart(&[4, 8], &[2.0, 2.0], &[0, 0], GroupStandard::Estimated)
            .expect("chart");
        close(e.group_centers[0].expect("c0"), 3.0);
        close(
            e.points[0].z.expect("z"),
            (2.0 - 3.0) / (3.0_f64 / 2.0).sqrt(),
        );
    }

    #[test]
    fn standardized_u_refusals() {
        assert!(matches!(
            standardized_u_chart(&[1], &[0.0], &[0], GroupStandard::Known(&[1.0])),
            Err(ShortRunError::Sample { index: 0, .. })
        ));
        assert!(matches!(
            standardized_u_chart(&[1], &[1.0], &[0], GroupStandard::Known(&[0.0])),
            Err(ShortRunError::InvalidStandard { group: Some(0), .. })
        ));
        assert_eq!(
            standardized_u_chart(&[0, 0], &[1.0, 1.0], &[0, 0], GroupStandard::Estimated)
                .expect_err("no defects"),
            ShortRunError::Degenerate { group: 0 }
        );
    }

    #[test]
    fn a_single_group_matches_the_ordinary_charts() {
        // One group's standardized P points are the P chart's own z.
        let d = [3_u64, 5, 2, 4, 6];
        let n = [100_u64, 120, 80, 100, 110];
        let ours = standardized_p_chart(&d, &n, &[0; 5], GroupStandard::Estimated).expect("chart");
        let mut theirs = super::super::attributes::PChart::new();
        for (&d, &n) in d.iter().zip(&n) {
            theirs.add_sample(d, n).expect("sample");
        }
        for (a, b) in ours.points.iter().zip(theirs.points()) {
            assert_eq!(a.z, b.z);
            assert_eq!(a.ucl, b.ucl);
            assert_eq!(a.lcl, b.lcl);
        }
    }

    #[test]
    fn a_single_group_matches_the_ordinary_individuals_chart() {
        // With one group, a deviation chart is an I-MR chart centred on the aim.
        let values = [5.1, 4.9, 5.3, 4.8, 5.0, 5.2];
        let c = deviation_chart(&values, &[0; 6], &[5.0], RuleSet::default(), None).expect("chart");
        let mut imr = IndividualMRChart::new().with_center(0.0).expect("centre");
        for v in values {
            imr.add_sample(&[v - 5.0]).expect("sample");
        }
        assert_eq!(Some(c.limits), imr.control_limits());
        assert_eq!(Some(c.mr_limits), imr.mr_limits());
    }

    #[test]
    fn with_center_fixes_the_centre_line_and_nothing_else() {
        let obs = [10.5, 10.7, 10.4, 10.6, 10.8];
        let mut free = IndividualMRChart::new();
        let mut fixed = IndividualMRChart::new().with_center(10.0).expect("centre");
        for x in obs {
            free.add_sample(&[x]).expect("x");
            fixed.add_sample(&[x]).expect("x");
        }
        let (f, g) = (
            free.control_limits().expect("limits"),
            fixed.control_limits().expect("limits"),
        );
        assert_eq!(g.cl, 10.0);
        close(g.ucl - g.cl, f.ucl - f.cl);
        assert_eq!(free.mr_limits(), fixed.mr_limits());
        assert_eq!(
            IndividualMRChart::new().with_center(f64::NAN).err(),
            Some(ControlChartError::NonFiniteValue)
        );
    }
}
