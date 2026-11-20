//! Hotelling's T² control chart for individual multivariate observations.
//!
//! A process measured on `p` correlated variables is watched through one statistic, the
//! squared Mahalanobis distance of an observation from the process mean,
//!
//! ```text
//! T² = (x − μ)ᵀ S⁻¹ (x − μ)
//! ```
//!
//! Two phases use different limits because the mean and covariance are estimated differently
//! (Tracy, Young & Mason 1992; Montgomery 2019, ch. 11):
//!
//! - **Phase I** — `m` historical observations judge themselves. The mean and covariance `S`
//!   come from the same `m` rows, so `T²ᵢ` is not an F-statistic but a scaled Beta one:
//!   `UCL = (m − 1)² / m · B⁻¹(1 − α; p/2, (m − p − 1)/2)`, which needs `m > p + 1`.
//! - **Phase II** — new observations, independent of the `m` that gave the estimates, are
//!   monitored against them: `UCL = p (m + 1)(m − 1) / (m (m − p)) · F⁻¹(1 − α; p, m − p)`,
//!   which needs `m > p`.
//!
//! Observations taken in rational subgroups of `n >= 2` are charted by their means: [`subgroup_phase_i`]
//! and [`subgroup_phase_ii`] give `T² = n (x̄ − μ)ᵀ S⁻¹ (x̄ − μ)` with `S` the average of the
//! within-subgroup covariance matrices, against the F limits of Alt (1985). Variation between
//! subgroups does not enter `S`, so a drifting mean shows as a signal instead of being absorbed.
//!
//! The limits are exact distribution quantiles, not asymptotic chi-squared ones, so they hold
//! for small `m`. A significance `α` is the per-point false-alarm probability when the process
//! is in control and normal.
//!
//! # References
//!
//! - Tracy, N.D., Young, J.C. & Mason, R.L. (1992). "Multivariate control charts for
//!   individual observations", *Journal of Quality Technology* 24(2), pp. 88-95.
//! - Alt, F.B. (1985). "Multivariate quality control", in *The Encyclopedia of Statistical
//!   Sciences*, Wiley (the limits for subgroups).
//! - Montgomery, D.C. (2019). *Introduction to Statistical Quality Control*, 8th ed., ch. 11.
//! - Hotelling, H. (1947). "Multivariate quality control", in *Techniques of Statistical
//!   Analysis*, McGraw-Hill.

use u_numflow::matrix::Matrix;
use u_numflow::special::{f_distribution_quantile, inverse_regularized_incomplete_beta};

/// Why a Hotelling T² chart could not be built.
#[derive(Debug, Clone, PartialEq)]
pub enum HotellingError {
    /// The significance level is not in `(0, 1)`.
    InvalidAlpha {
        /// The value given.
        got: f64,
    },
    /// Fewer observations than the limit's distribution needs.
    TooFew {
        /// How many are needed: `p + 2` for Phase I, `p + 1` for the estimates of Phase II.
        min: usize,
        /// How many there are.
        got: usize,
    },
    /// No variables, or a row whose length differs from the first row's (or from the mean's).
    DimensionMismatch {
        /// Name of the argument.
        parameter: &'static str,
        /// The length it has to be.
        expected: usize,
        /// The length it is.
        got: usize,
        /// The row, when the argument is a table of rows.
        index: Option<usize>,
    },
    /// A NaN or an infinity in an observation, the mean or the covariance.
    NonFinite {
        /// Name of the argument.
        parameter: &'static str,
        /// The row, when the argument is a table of rows.
        index: Option<usize>,
    },
    /// The covariance matrix cannot be inverted: a variable is constant, or one is a linear
    /// combination of others, so the data hold no information in that direction.
    Degenerate,
}

/// The mean and covariance a Phase II chart monitors against, estimated from `m` observations.
#[derive(Debug, Clone, PartialEq)]
pub struct HotellingEstimates {
    /// The mean of each variable.
    pub mean: Vec<f64>,
    /// The sample covariance matrix (denominator `m − 1`), row-major, `p × p`.
    pub covariance: Vec<Vec<f64>>,
    /// The number of observations the estimates come from.
    pub m: usize,
}

/// A Hotelling T² chart: one statistic per observation against one upper limit.
#[derive(Debug, Clone, PartialEq)]
pub struct HotellingChart {
    /// `T²` of each observation, in the order given.
    pub t2: Vec<f64>,
    /// The upper control limit (the lower limit is 0).
    pub ucl: f64,
    /// Positions (from 0) whose `T²` exceeds the limit.
    pub out_of_control: Vec<usize>,
    /// The estimates the statistics are measured against.
    pub estimates: HotellingEstimates,
}

impl HotellingChart {
    /// Whether every observation is within the limit.
    pub fn is_in_control(&self) -> bool {
        self.out_of_control.is_empty()
    }
}

fn check_alpha(alpha: f64) -> Result<(), HotellingError> {
    if alpha.is_finite() && alpha > 0.0 && alpha < 1.0 {
        Ok(())
    } else {
        Err(HotellingError::InvalidAlpha { got: alpha })
    }
}

/// The rows as a `p`-wide table, every value finite: the width is that of the first row.
fn check_table(rows: &[Vec<f64>], parameter: &'static str) -> Result<usize, HotellingError> {
    let p = rows.first().map_or(0, Vec::len);
    if p == 0 {
        return Err(HotellingError::DimensionMismatch {
            parameter,
            expected: 1,
            got: 0,
            index: Some(0),
        });
    }
    for (i, row) in rows.iter().enumerate() {
        if row.len() != p {
            return Err(HotellingError::DimensionMismatch {
                parameter,
                expected: p,
                got: row.len(),
                index: Some(i),
            });
        }
        if row.iter().any(|v| !v.is_finite()) {
            return Err(HotellingError::NonFinite {
                parameter,
                index: Some(i),
            });
        }
    }
    Ok(p)
}

/// The mean and sample covariance (denominator `m − 1`) of the rows.
fn estimate(rows: &[Vec<f64>], p: usize) -> HotellingEstimates {
    let m = rows.len();
    let mut mean = vec![0.0; p];
    for row in rows {
        for (acc, v) in mean.iter_mut().zip(row) {
            *acc += v;
        }
    }
    for acc in &mut mean {
        *acc /= m as f64;
    }
    let mut covariance = vec![vec![0.0; p]; p];
    for row in rows {
        for a in 0..p {
            let da = row[a] - mean[a];
            for b in a..p {
                covariance[a][b] += da * (row[b] - mean[b]);
            }
        }
    }
    let denominator = (m - 1) as f64;
    for a in 0..p {
        for b in a..p {
            covariance[a][b] /= denominator;
            covariance[b][a] = covariance[a][b];
        }
    }
    HotellingEstimates {
        mean,
        covariance,
        m,
    }
}

/// The lower Cholesky factor `L` of the `p × p` covariance (`S = L Lᵀ`). A covariance that is
/// not positive definite is `Degenerate`.
fn factor_of(p: usize, covariance: &[Vec<f64>]) -> Result<Matrix, HotellingError> {
    let flat: Vec<f64> = covariance.iter().flatten().copied().collect();
    let s = Matrix::new(p, p, flat).map_err(|_| HotellingError::Degenerate)?;
    let factor = s.cholesky().map_err(|_| HotellingError::Degenerate)?;
    // Positive definite to rounding is not enough: a pivot lost to cancellation (a constant or
    // a duplicated variable) gives a factor with a (near) zero diagonal. The scale of S is its
    // largest diagonal entry.
    let scale = (0..p).map(|i| s.get(i, i)).fold(0.0_f64, f64::max);
    for i in 0..p {
        let d = factor.get(i, i);
        if !(d.is_finite() && d * d > 1e-13 * scale) {
            return Err(HotellingError::Degenerate);
        }
    }
    Ok(factor)
}

/// `(x − μ)ᵀ S⁻¹ (x − μ)` for every row, by a Cholesky factorisation of `S` (the factor is
/// built once; each row is one pair of triangular solves). A covariance that is not positive
/// definite is `Degenerate`.
fn t_squared(
    rows: &[Vec<f64>],
    mean: &[f64],
    covariance: &[Vec<f64>],
) -> Result<Vec<f64>, HotellingError> {
    let p = mean.len();
    let factor = factor_of(p, covariance)?;
    let mut out = Vec::with_capacity(rows.len());
    for row in rows {
        let centred: Vec<f64> = row.iter().zip(mean).map(|(x, m)| x - m).collect();
        // Forward solve L z = centred: T² = zᵀz.
        let mut z = vec![0.0; p];
        for i in 0..p {
            let mut sum = centred[i];
            for (j, zj) in z.iter().enumerate().take(i) {
                sum -= factor.get(i, j) * zj;
            }
            z[i] = sum / factor.get(i, i);
        }
        out.push(z.iter().map(|v| v * v).sum());
    }
    Ok(out)
}

/// The Phase I upper control limit for `m` observations of `p` variables: the exact Beta limit
/// of Tracy, Young & Mason (1992). Needs `m > p + 1`.
pub fn phase_i_limit(m: usize, p: usize, alpha: f64) -> Result<f64, HotellingError> {
    check_alpha(alpha)?;
    if p == 0 {
        return Err(HotellingError::DimensionMismatch {
            parameter: "variables",
            expected: 1,
            got: 0,
            index: None,
        });
    }
    if m < p + 2 {
        return Err(HotellingError::TooFew { min: p + 2, got: m });
    }
    let (m_f, p_f) = (m as f64, p as f64);
    let quantile =
        inverse_regularized_incomplete_beta(1.0 - alpha, p_f / 2.0, (m_f - p_f - 1.0) / 2.0);
    Ok((m_f - 1.0) * (m_f - 1.0) / m_f * quantile)
}

/// The Phase II upper control limit for new observations against estimates from `m` observations
/// of `p` variables: `p (m + 1)(m − 1) / (m (m − p)) · F⁻¹(1 − α; p, m − p)`. Needs `m > p`.
pub fn phase_ii_limit(m: usize, p: usize, alpha: f64) -> Result<f64, HotellingError> {
    check_alpha(alpha)?;
    if p == 0 {
        return Err(HotellingError::DimensionMismatch {
            parameter: "variables",
            expected: 1,
            got: 0,
            index: None,
        });
    }
    if m < p + 1 {
        return Err(HotellingError::TooFew { min: p + 1, got: m });
    }
    let (m_f, p_f) = (m as f64, p as f64);
    let f = f_distribution_quantile(1.0 - alpha, p_f, m_f - p_f);
    Ok(p_f * (m_f + 1.0) * (m_f - 1.0) / (m_f * (m_f - p_f)) * f)
}

/// The Phase I chart of `data` (rows are observations, columns the `p` variables): `T²` of each
/// row against the mean and covariance of all rows, and the exact Beta limit at significance
/// `alpha`. Rows above the limit are `out_of_control` — in practice they are investigated and,
/// once their cause is found, removed and the chart rebuilt before the estimates are used for
/// [`phase_ii`].
///
/// # Errors
///
/// `alpha` outside `(0, 1)`; fewer than `p + 2` rows; rows of different lengths or with a NaN or
/// an infinity; a singular covariance ([`HotellingError::Degenerate`]).
///
/// # Examples
///
/// ```
/// use u_analytics::spc::hotelling::phase_i;
///
/// // Two correlated variables: the last point breaks the correlation, not either variable's range.
/// let data = vec![
///     vec![1.0, 1.1], vec![2.0, 1.9], vec![3.0, 3.2], vec![4.0, 3.9], vec![5.0, 5.1],
///     vec![6.0, 5.8], vec![7.0, 7.2], vec![8.0, 7.9], vec![4.5, 1.5],
/// ];
/// let chart = phase_i(&data, 0.0027).unwrap();
/// assert_eq!(chart.out_of_control, vec![8]);
/// ```
pub fn phase_i(data: &[Vec<f64>], alpha: f64) -> Result<HotellingChart, HotellingError> {
    check_alpha(alpha)?;
    let p = check_table(data, "data")?;
    let ucl = phase_i_limit(data.len(), p, alpha)?;
    let estimates = estimate(data, p);
    let t2 = t_squared(data, &estimates.mean, &estimates.covariance)?;
    let out_of_control = (0..t2.len()).filter(|&i| t2[i] > ucl).collect();
    Ok(HotellingChart {
        t2,
        ucl,
        out_of_control,
        estimates,
    })
}

/// The mean and covariance of a reference, checked: non-empty, finite, a square `p × p` matrix.
/// Returns `p`.
fn check_reference(mean: &[f64], covariance: &[Vec<f64>]) -> Result<usize, HotellingError> {
    let p = mean.len();
    if p == 0 {
        return Err(HotellingError::DimensionMismatch {
            parameter: "mean",
            expected: 1,
            got: 0,
            index: None,
        });
    }
    if mean.iter().any(|v| !v.is_finite()) {
        return Err(HotellingError::NonFinite {
            parameter: "mean",
            index: None,
        });
    }
    if covariance.len() != p {
        return Err(HotellingError::DimensionMismatch {
            parameter: "covariance",
            expected: p,
            got: covariance.len(),
            index: None,
        });
    }
    for (i, row) in covariance.iter().enumerate() {
        if row.len() != p {
            return Err(HotellingError::DimensionMismatch {
                parameter: "covariance",
                expected: p,
                got: row.len(),
                index: Some(i),
            });
        }
        if row.iter().any(|v| !v.is_finite()) {
            return Err(HotellingError::NonFinite {
                parameter: "covariance",
                index: Some(i),
            });
        }
    }
    Ok(p)
}

/// The Phase II chart of `observations` against `estimates` (from [`phase_i`] after the causes
/// of its signals were removed, or from [`estimate_from`]): `T²` of each new observation and the
/// exact F limit at significance `alpha`.
///
/// # Errors
///
/// `alpha` outside `(0, 1)`; fewer than `p + 1` observations behind the estimates; an
/// observation or an estimate of the wrong length or not finite; a singular covariance.
pub fn phase_ii(
    observations: &[Vec<f64>],
    estimates: &HotellingEstimates,
    alpha: f64,
) -> Result<HotellingChart, HotellingError> {
    check_alpha(alpha)?;
    let p = check_reference(&estimates.mean, &estimates.covariance)?;
    if !observations.is_empty() {
        let width = check_table(observations, "observations")?;
        if width != p {
            return Err(HotellingError::DimensionMismatch {
                parameter: "observations",
                expected: p,
                got: width,
                index: Some(0),
            });
        }
    }
    let ucl = phase_ii_limit(estimates.m, p, alpha)?;
    let t2 = t_squared(observations, &estimates.mean, &estimates.covariance)?;
    let out_of_control = (0..t2.len()).filter(|&i| t2[i] > ucl).collect();
    Ok(HotellingChart {
        t2,
        ucl,
        out_of_control,
        estimates: estimates.clone(),
    })
}

/// The mean and covariance of `data` as the estimates a Phase II chart monitors against.
///
/// # Errors
///
/// Rows of different lengths or with a NaN or an infinity, or fewer than 2 rows.
pub fn estimate_from(data: &[Vec<f64>]) -> Result<HotellingEstimates, HotellingError> {
    let p = check_table(data, "data")?;
    if data.len() < 2 {
        return Err(HotellingError::TooFew {
            min: 2,
            got: data.len(),
        });
    }
    Ok(estimate(data, p))
}

// ---------------------------------------------------------------------------------------------
// Subgrouped observations
// ---------------------------------------------------------------------------------------------

/// What a subgroup Phase II chart monitors against: the grand mean, the pooled within-subgroup
/// covariance, and the number of subgroups `m` and their common size `n` they come from.
#[derive(Debug, Clone, PartialEq)]
pub struct SubgroupEstimates {
    /// The mean of each variable (the mean of the subgroup means).
    pub mean: Vec<f64>,
    /// The average of the within-subgroup sample covariance matrices (each with denominator
    /// `n − 1`), row-major, `p × p`.
    pub covariance: Vec<Vec<f64>>,
    /// The number of subgroups the estimates come from.
    pub m: usize,
    /// The number of observations in each subgroup.
    pub n: usize,
}

/// A Hotelling T² chart of subgroups: one statistic per subgroup against one upper limit.
#[derive(Debug, Clone, PartialEq)]
pub struct SubgroupHotellingChart {
    /// `T² = n (x̄ − μ)ᵀ S⁻¹ (x̄ − μ)` of each subgroup mean, in the order given.
    pub t2: Vec<f64>,
    /// The upper control limit (the lower limit is 0).
    pub ucl: f64,
    /// Positions (from 0) of the subgroups whose `T²` exceeds the limit.
    pub out_of_control: Vec<usize>,
    /// The estimates the statistics are measured against.
    pub estimates: SubgroupEstimates,
}

impl SubgroupHotellingChart {
    /// Whether every subgroup is within the limit.
    pub fn is_in_control(&self) -> bool {
        self.out_of_control.is_empty()
    }
}

/// Subgroup size and width of a table of subgroups, each a table of `n` rows of `p` finite
/// values: all the same `n >= 2` and `p`. A refusal names the subgroup as `index`.
fn check_subgroups(
    subgroups: &[Vec<Vec<f64>>],
    parameter: &'static str,
) -> Result<(usize, usize), HotellingError> {
    let n = subgroups.first().map_or(0, Vec::len);
    if n < 2 {
        return Err(HotellingError::TooFew { min: 2, got: n });
    }
    let mut p = 0;
    for (i, subgroup) in subgroups.iter().enumerate() {
        if subgroup.len() != n {
            return Err(HotellingError::DimensionMismatch {
                parameter,
                expected: n,
                got: subgroup.len(),
                index: Some(i),
            });
        }
        let width = check_table(subgroup, parameter).map_err(|e| match e {
            HotellingError::DimensionMismatch {
                parameter,
                expected,
                got,
                ..
            } => HotellingError::DimensionMismatch {
                parameter,
                expected,
                got,
                index: Some(i),
            },
            HotellingError::NonFinite { parameter, .. } => HotellingError::NonFinite {
                parameter,
                index: Some(i),
            },
            other => other,
        })?;
        if i == 0 {
            p = width;
        } else if width != p {
            return Err(HotellingError::DimensionMismatch {
                parameter,
                expected: p,
                got: width,
                index: Some(i),
            });
        }
    }
    Ok((n, p))
}

/// The mean of each subgroup, and the covariance matrices of the subgroups averaged.
fn subgroup_summary(subgroups: &[Vec<Vec<f64>>], p: usize) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let means: Vec<Vec<f64>> = subgroups.iter().map(|g| estimate(g, p).mean).collect();
    let mut pooled = vec![vec![0.0; p]; p];
    for g in subgroups {
        let s = estimate(g, p).covariance;
        for (row, srow) in pooled.iter_mut().zip(&s) {
            for (acc, v) in row.iter_mut().zip(srow) {
                *acc += v;
            }
        }
    }
    let m = subgroups.len() as f64;
    for row in &mut pooled {
        for v in row {
            *v /= m;
        }
    }
    (means, pooled)
}

/// The degrees of freedom of the subgroup limits' F distribution, `m n − m − p + 1`, and the
/// refusal when there are too few observations for the pooled covariance (`m (n − 1) < p`).
fn subgroup_df(m: usize, n: usize, p: usize) -> Result<f64, HotellingError> {
    if m * (n - 1) < p {
        return Err(HotellingError::TooFew {
            min: m + p,
            got: m * n,
        });
    }
    Ok((m * n - m - p + 1) as f64)
}

/// The Phase I upper control limit for `m` subgroups of `n` observations of `p` variables
/// (Alt 1985; Montgomery 2019, ch. 11): `p (m − 1)(n − 1) / (m n − m − p + 1) ·
/// F⁻¹(1 − α; p, m n − m − p + 1)`. Needs `n >= 2` and `m (n − 1) >= p`.
///
/// # Errors
///
/// `alpha` outside `(0, 1)`; `n < 2`; fewer than `m + p` observations in all.
pub fn subgroup_phase_i_limit(
    m: usize,
    n: usize,
    p: usize,
    alpha: f64,
) -> Result<f64, HotellingError> {
    subgroup_limit(m, n, p, alpha, m as f64 - 1.0)
}

/// The Phase II upper control limit for new subgroups of `n` against estimates from `m`
/// subgroups of `n` observations of `p` variables: `p (m + 1)(n − 1) / (m n − m − p + 1) ·
/// F⁻¹(1 − α; p, m n − m − p + 1)`. Needs `n >= 2` and `m (n − 1) >= p`.
///
/// # Errors
///
/// As [`subgroup_phase_i_limit`].
pub fn subgroup_phase_ii_limit(
    m: usize,
    n: usize,
    p: usize,
    alpha: f64,
) -> Result<f64, HotellingError> {
    subgroup_limit(m, n, p, alpha, m as f64 + 1.0)
}

/// `p · factor · (n − 1) / df · F⁻¹(1 − α; p, df)`, the two limits differing in `factor` only
/// (`m − 1` for the subgroups that gave the estimates, `m + 1` for new ones).
fn subgroup_limit(
    m: usize,
    n: usize,
    p: usize,
    alpha: f64,
    factor: f64,
) -> Result<f64, HotellingError> {
    check_alpha(alpha)?;
    if p == 0 {
        return Err(HotellingError::DimensionMismatch {
            parameter: "variables",
            expected: 1,
            got: 0,
            index: None,
        });
    }
    if n < 2 {
        return Err(HotellingError::TooFew { min: 2, got: n });
    }
    let df = subgroup_df(m, n, p)?;
    let f = f_distribution_quantile(1.0 - alpha, p as f64, df);
    Ok(p as f64 * factor * (n as f64 - 1.0) / df * f)
}

/// The Phase I chart of `subgroups` — each a table of `n` rows of `p` variables, all the same
/// `n >= 2` — judging each subgroup mean against the grand mean with the covariance pooled
/// within the subgroups (the average of their sample covariances): `T² = n (x̄ᵢ − x̿)ᵀ S⁻¹
/// (x̄ᵢ − x̿)` against the limit of [`subgroup_phase_i_limit`]. A subgroup above the limit is
/// `out_of_control`; remove the ones with a known cause and rebuild before using the estimates
/// for [`subgroup_phase_ii`].
///
/// The pooled covariance measures variation *within* subgroups, so a shift between subgroups
/// cannot inflate it: the chart sees a drifting mean that the chart of individual observations
/// would absorb into its own covariance.
///
/// # Errors
///
/// `alpha` outside `(0, 1)`; subgroups of different sizes (the subgroup as `index`), of fewer
/// than 2 rows, of rows of different lengths or not finite; fewer than `m + p` observations;
/// a pooled covariance that cannot be inverted.
///
/// # Examples
///
/// ```
/// use u_analytics::spc::hotelling::subgroup_phase_i;
///
/// // Eight subgroups of three observations of two variables that move together.
/// let subgroups: Vec<Vec<Vec<f64>>> = (0..8)
///     .map(|i| {
///         let b = i as f64;
///         vec![
///             vec![b, b + 0.1],
///             vec![b + 0.2, b + 0.1],
///             vec![b + 0.1, b + 0.3],
///         ]
///     })
///     .collect();
/// let chart = subgroup_phase_i(&subgroups, 0.0027).unwrap();
/// assert_eq!(chart.t2.len(), 8);
/// assert_eq!(chart.estimates.n, 3);
/// ```
pub fn subgroup_phase_i(
    subgroups: &[Vec<Vec<f64>>],
    alpha: f64,
) -> Result<SubgroupHotellingChart, HotellingError> {
    check_alpha(alpha)?;
    if subgroups.is_empty() {
        return Err(HotellingError::DimensionMismatch {
            parameter: "subgroups",
            expected: 1,
            got: 0,
            index: None,
        });
    }
    let (n, p) = check_subgroups(subgroups, "subgroups")?;
    let m = subgroups.len();
    let ucl = subgroup_phase_i_limit(m, n, p, alpha)?;
    let (means, covariance) = subgroup_summary(subgroups, p);
    let grand = estimate(&means, p).mean;
    let t2 = subgroup_t_squared(&means, &grand, &covariance, n)?;
    let out_of_control = (0..t2.len()).filter(|&i| t2[i] > ucl).collect();
    Ok(SubgroupHotellingChart {
        t2,
        ucl,
        out_of_control,
        estimates: SubgroupEstimates {
            mean: grand,
            covariance,
            m,
            n,
        },
    })
}

/// `n (x̄ − μ)ᵀ S⁻¹ (x̄ − μ)` for each subgroup mean.
fn subgroup_t_squared(
    means: &[Vec<f64>],
    mean: &[f64],
    covariance: &[Vec<f64>],
    n: usize,
) -> Result<Vec<f64>, HotellingError> {
    let t2 = t_squared(means, mean, covariance)?;
    Ok(t2.into_iter().map(|v| v * n as f64).collect())
}

/// The Phase II chart of new `subgroups` of `estimates.n` observations against `estimates` (from
/// [`subgroup_phase_i`] after the causes of its signals were removed): `T²` of each subgroup mean
/// and the F limit of [`subgroup_phase_ii_limit`] at significance `alpha`. `subgroups` may be empty.
///
/// # Errors
///
/// `alpha` outside `(0, 1)`; fewer than `estimates.m + p` observations behind the estimates; a
/// subgroup of a size other than `estimates.n` (the subgroup as `index`), of the wrong width or
/// not finite; a singular covariance.
pub fn subgroup_phase_ii(
    subgroups: &[Vec<Vec<f64>>],
    estimates: &SubgroupEstimates,
    alpha: f64,
) -> Result<SubgroupHotellingChart, HotellingError> {
    check_alpha(alpha)?;
    let p = check_reference(&estimates.mean, &estimates.covariance)?;
    let ucl = subgroup_phase_ii_limit(estimates.m, estimates.n, p, alpha)?;
    let means = if subgroups.is_empty() {
        Vec::new()
    } else {
        for (i, subgroup) in subgroups.iter().enumerate() {
            if subgroup.len() != estimates.n {
                return Err(HotellingError::DimensionMismatch {
                    parameter: "subgroups",
                    expected: estimates.n,
                    got: subgroup.len(),
                    index: Some(i),
                });
            }
        }
        let (_, width) = check_subgroups(subgroups, "subgroups")?;
        if width != p {
            return Err(HotellingError::DimensionMismatch {
                parameter: "subgroups",
                expected: p,
                got: width,
                index: Some(0),
            });
        }
        subgroups.iter().map(|g| estimate(g, p).mean).collect()
    };
    let t2 = subgroup_t_squared(&means, &estimates.mean, &estimates.covariance, estimates.n)?;
    let out_of_control = (0..t2.len()).filter(|&i| t2[i] > ucl).collect();
    Ok(SubgroupHotellingChart {
        t2,
        ucl,
        out_of_control,
        estimates: estimates.clone(),
    })
}

// ---------------------------------------------------------------------------------------------
// Which variable moved
// ---------------------------------------------------------------------------------------------

/// What each variable contributes to the `T²` of one observation: a ranking to read after a
/// signal, not a test. The three vectors have one entry per variable, in the order of the
/// observation.
///
/// The entries do **not** add up to `t2` (they are not shares of it), and none has a limit of its
/// own. With strongly correlated variables a variable that did not move still gets a large
/// `drop_one`, because the others predict it (smearing): read `drop_one` next to `unconditional`.
#[derive(Debug, Clone, PartialEq)]
pub struct HotellingContributions {
    /// The `T²` of the observation (`n (x̄ − μ)ᵀ S⁻¹ (x̄ − μ)` for a subgroup mean).
    pub t2: f64,
    /// `T² − T²₋ⱼ`: how much `T²` falls when variable `j` is left out, `T²₋ⱼ` being the statistic
    /// of the other `p − 1` variables. It is the squared standardised deviation of variable `j`
    /// from what the other variables predict (`z_j²`), whatever the order the variables are in.
    pub drop_one: Vec<f64>,
    /// `z_j`: that deviation in standard deviations of variable `j` given the others, with the
    /// sign of its direction. `z_j² = drop_one_j`.
    pub z: Vec<f64>,
    /// The variable on its own: `n (x_j − μ_j)² / S_jj`, the squared standardised distance from
    /// its mean whatever the others did (`n = 1` for an individual observation).
    pub unconditional: Vec<f64>,
}

/// The contributions of `x − μ` to `scale · rᵀ S⁻¹ r`, by the Cholesky factor of `S` alone.
///
/// With `w = S⁻¹ r`, `z_j = √scale · w_j / √(S⁻¹)_jj` and `(S⁻¹)_jj = ‖L⁻¹ e_j‖²`: one pair of
/// triangular solves for `w` and one more per variable, `O(p³)` in all.
fn contributions_of(
    observation: &[f64],
    mean: &[f64],
    covariance: &[Vec<f64>],
    scale: f64,
) -> Result<HotellingContributions, HotellingError> {
    let p = check_reference(mean, covariance)?;
    if observation.len() != p {
        return Err(HotellingError::DimensionMismatch {
            parameter: "observation",
            expected: p,
            got: observation.len(),
            index: None,
        });
    }
    if observation.iter().any(|v| !v.is_finite()) {
        return Err(HotellingError::NonFinite {
            parameter: "observation",
            index: None,
        });
    }
    let factor = factor_of(p, covariance)?;
    let r: Vec<f64> = observation.iter().zip(mean).map(|(x, m)| x - m).collect();
    // L u = r, then Lᵀ w = u: w = S⁻¹ r and T² = scale · uᵀu.
    let mut u = vec![0.0; p];
    for i in 0..p {
        let mut sum = r[i];
        for (j, uj) in u.iter().enumerate().take(i) {
            sum -= factor.get(i, j) * uj;
        }
        u[i] = sum / factor.get(i, i);
    }
    let mut w = vec![0.0; p];
    for i in (0..p).rev() {
        let mut sum = u[i];
        for (j, wj) in w.iter().enumerate().skip(i + 1) {
            sum -= factor.get(j, i) * wj;
        }
        w[i] = sum / factor.get(i, i);
    }
    let t2 = scale * u.iter().map(|v| v * v).sum::<f64>();
    let root = scale.sqrt();
    let mut z = Vec::with_capacity(p);
    let mut column = vec![0.0; p];
    for (j, wj) in w.iter().enumerate() {
        // x = L⁻¹ e_j (zero above row j): (S⁻¹)_jj = ‖x‖².
        let mut norm = 0.0;
        for i in j..p {
            let mut sum = if i == j { 1.0 } else { 0.0 };
            for (k, xk) in column.iter().enumerate().take(i).skip(j) {
                sum -= factor.get(i, k) * xk;
            }
            column[i] = sum / factor.get(i, i);
            norm += column[i] * column[i];
        }
        z.push(root * wj / norm.sqrt());
    }
    let drop_one = z.iter().map(|v| v * v).collect();
    let unconditional = (0..p)
        .map(|j| scale * r[j] * r[j] / covariance[j][j])
        .collect();
    Ok(HotellingContributions {
        t2,
        drop_one,
        z,
        unconditional,
    })
}

/// Which variable moved: the contributions of one `observation` to its `T²` against `estimates`
/// (from [`phase_i`] or [`estimate_from`], the same ones [`phase_ii`] takes).
///
/// `drop_one` is `T²` minus the `T²` of the other variables, equal to the squared standardised
/// deviation of the variable from what the others predict; `z` is that deviation with its sign;
/// `unconditional` is the variable's own squared standardised distance. See
/// [`HotellingContributions`] for how to read them — a ranking, not a test: the entries do not
/// add up to `T²` and have no limit.
///
/// The identity `T² − T²₋ⱼ = (S⁻¹r)ⱼ² / (S⁻¹)ⱼⱼ` follows from the block inverse of `S`; it is
/// checked in the tests against the definition (the two statistics by Gaussian elimination), not
/// taken from the literature, which decomposes `T²` the same way (Mason, Tracy & Young 1995;
/// Runger, Alt & Montgomery 1996).
///
/// # Errors
///
/// An observation or an estimate of the wrong length or not finite; a singular covariance.
///
/// # Examples
///
/// ```
/// use u_analytics::spc::hotelling::{contributions, HotellingEstimates};
///
/// // Two variables, correlation 0.9; the first rises while the second falls.
/// let estimates = HotellingEstimates {
///     mean: vec![0.0, 0.0],
///     covariance: vec![vec![1.0, 0.9], vec![0.9, 1.0]],
///     m: 50,
/// };
/// let c = contributions(&[1.0, -1.0], &estimates).unwrap();
/// assert!((c.t2 - 20.0).abs() < 1e-9);
/// // Each is 19 against T² = 20: the entries are not shares of it.
/// assert!((c.drop_one[0] - 19.0).abs() < 1e-9);
/// // The first variable's own distance is 1; given the second it is far larger, and the sign
/// // says the first is above what the second predicts.
/// assert!((c.unconditional[0] - 1.0).abs() < 1e-12);
/// assert!(c.z[0] > 0.0 && c.z[1] < 0.0);
/// ```
pub fn contributions(
    observation: &[f64],
    estimates: &HotellingEstimates,
) -> Result<HotellingContributions, HotellingError> {
    contributions_of(observation, &estimates.mean, &estimates.covariance, 1.0)
}

/// [`contributions`] for the mean of a subgroup of `estimates.n` observations: the entries are
/// those of `T² = n (x̄ − μ)ᵀ S⁻¹ (x̄ − μ)` (`drop_one` and `unconditional` carry the factor `n`,
/// `z` its square root).
///
/// # Errors
///
/// A subgroup size `n` below 2; a mean or an estimate of the wrong length or not finite; a
/// singular covariance.
pub fn subgroup_contributions(
    subgroup_mean: &[f64],
    estimates: &SubgroupEstimates,
) -> Result<HotellingContributions, HotellingError> {
    if estimates.n < 2 {
        return Err(HotellingError::TooFew {
            min: 2,
            got: estimates.n,
        });
    }
    contributions_of(
        subgroup_mean,
        &estimates.mean,
        &estimates.covariance,
        estimates.n as f64,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use u_numflow::special::{f_distribution_cdf, regularized_incomplete_beta};

    /// A seeded source of numbers (no dependencies).
    struct Lcg(u64);

    impl Lcg {
        fn unit(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
        /// Sum of twelve uniforms minus six: a bell, close enough to a normal.
        fn bell(&mut self) -> f64 {
            (0..12).map(|_| self.unit()).sum::<f64>() - 6.0
        }
    }

    fn correlated(rng: &mut Lcg, m: usize, p: usize) -> Vec<Vec<f64>> {
        (0..m)
            .map(|_| {
                let common = rng.bell();
                (0..p)
                    .map(|j| 10.0 + j as f64 + common * (1.0 + j as f64 * 0.3) + rng.bell() * 0.5)
                    .collect()
            })
            .collect()
    }

    /// The squared Mahalanobis distance from the definition: solve `S y = x − μ` by Gaussian
    /// elimination with partial pivoting and take `(x − μ)ᵀ y` — nothing shared with the
    /// Cholesky path under test.
    fn mahalanobis_by_elimination(x: &[f64], mean: &[f64], s: &[Vec<f64>]) -> f64 {
        let p = mean.len();
        let mut a: Vec<Vec<f64>> = (0..p)
            .map(|i| {
                let mut row = s[i].clone();
                row.push(x[i] - mean[i]);
                row
            })
            .collect();
        for col in 0..p {
            let pivot = (col..p)
                .max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))
                .unwrap();
            a.swap(col, pivot);
            for row in (col + 1)..p {
                let factor = a[row][col] / a[col][col];
                for k in col..=p {
                    a[row][k] -= factor * a[col][k];
                }
            }
        }
        let mut y = vec![0.0; p];
        for i in (0..p).rev() {
            let mut sum = a[i][p];
            for j in (i + 1)..p {
                sum -= a[i][j] * y[j];
            }
            y[i] = sum / a[i][i];
        }
        (0..p).map(|i| (x[i] - mean[i]) * y[i]).sum()
    }

    #[test]
    fn t_squared_is_the_squared_mahalanobis_distance_of_the_definition() {
        let mut rng = Lcg(3);
        for p in [1usize, 2, 3, 5] {
            let data = correlated(&mut rng, 40, p);
            let chart = phase_i(&data, 0.01).unwrap();
            for (row, &t2) in data.iter().zip(&chart.t2) {
                let want = mahalanobis_by_elimination(
                    row,
                    &chart.estimates.mean,
                    &chart.estimates.covariance,
                );
                assert!(
                    (t2 - want).abs() <= 1e-9 * want.max(1.0),
                    "p {p}: {t2} vs {want}"
                );
            }
        }
    }

    #[test]
    fn one_variable_is_the_squared_standardised_value() {
        let data: Vec<Vec<f64>> = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
            .iter()
            .map(|&v| vec![v])
            .collect();
        let chart = phase_i(&data, 0.05).unwrap();
        let mean = 5.0;
        let var = data.iter().map(|r| (r[0] - mean).powi(2)).sum::<f64>() / 7.0;
        for (row, &t2) in data.iter().zip(&chart.t2) {
            assert!((t2 - (row[0] - mean).powi(2) / var).abs() < 1e-12);
        }
    }

    /// T² does not change when the variables are rescaled, shifted or mixed: an invertible affine
    /// map of the data (units, offsets, a rotation) leaves every statistic and the limit alone.
    #[test]
    fn t_squared_is_invariant_under_an_affine_map_of_the_variables() {
        let mut rng = Lcg(8);
        let data = correlated(&mut rng, 30, 3);
        let map = [[2.0, 0.5, -1.0], [0.0, 3.0, 0.7], [1.0, 0.0, 0.25]];
        let moved: Vec<Vec<f64>> = data
            .iter()
            .map(|r| {
                (0..3)
                    .map(|i| {
                        map[i].iter().zip(r).map(|(a, x)| a * x).sum::<f64>() + 100.0 * i as f64
                            - 7.0
                    })
                    .collect()
            })
            .collect();
        let (a, b) = (
            phase_i(&data, 0.01).unwrap(),
            phase_i(&moved, 0.01).unwrap(),
        );
        assert_eq!(a.ucl, b.ucl);
        for (x, y) in a.t2.iter().zip(&b.t2) {
            assert!((x - y).abs() <= 1e-8 * x.max(1.0), "{x} vs {y}");
        }
    }

    /// The limits are quantiles of their distributions: the CDF of the Beta (Phase I) and of the F
    /// (Phase II) at the scaled limit gives back `1 − α`.
    #[test]
    fn the_limits_are_the_one_minus_alpha_quantiles_of_their_distributions() {
        for (m, p, alpha) in [
            (20usize, 2usize, 0.0027),
            (30, 3, 0.01),
            (100, 5, 0.05),
            (8, 2, 0.1),
        ] {
            let ucl = phase_i_limit(m, p, alpha).unwrap();
            let (mf, pf) = (m as f64, p as f64);
            let beta = regularized_incomplete_beta(
                ucl * mf / ((mf - 1.0) * (mf - 1.0)),
                pf / 2.0,
                (mf - pf - 1.0) / 2.0,
            );
            assert!(
                (beta - (1.0 - alpha)).abs() < 1e-10,
                "phase I m {m} p {p}: {beta}"
            );
            let ucl = phase_ii_limit(m, p, alpha).unwrap();
            let f = f_distribution_cdf(
                ucl * mf * (mf - pf) / (pf * (mf + 1.0) * (mf - 1.0)),
                pf,
                mf - pf,
            );
            assert!(
                (f - (1.0 - alpha)).abs() < 1e-10,
                "phase II m {m} p {p}: {f}"
            );
        }
    }

    /// With `p = 1` the Phase I limit is the maximum squared z of `m` points under a normal
    /// with estimated mean and sd: `(m − 1)²/m · B(1/2, (m − 2)/2)`; at `m = 3` the Beta is
    /// `B(1/2, 1/2)` (arcsine) whose `1 − α` quantile is `sin²(π(1 − α)/2)`.
    #[test]
    fn the_phase_i_limit_at_three_points_of_one_variable_is_the_arcsine_quantile() {
        for alpha in [0.5, 0.1, 0.0027] {
            let ucl = phase_i_limit(3, 1, alpha).unwrap();
            let want = 4.0 / 3.0 * (std::f64::consts::FRAC_PI_2 * (1.0 - alpha)).sin().powi(2);
            assert!((ucl - want).abs() < 1e-12, "{ucl} vs {want}");
        }
    }

    #[test]
    fn a_point_that_breaks_the_correlation_is_the_one_flagged() {
        let mut rng = Lcg(21);
        let mut data = correlated(&mut rng, 60, 2);
        // Each variable stays inside its own range, but against the strong common factor it is far off.
        let (a, b) = (data[0][0], data[0][1]);
        data.push(vec![a + 4.0, b - 4.0]);
        let chart = phase_i(&data, 0.0027).unwrap();
        assert!(
            chart.out_of_control.contains(&60),
            "{:?} T² {}",
            chart.out_of_control,
            chart.t2[60]
        );
    }

    #[test]
    fn in_control_normal_data_signals_about_alpha_of_the_time() {
        let mut rng = Lcg(5);
        let (alpha, trials, m) = (0.05, 400, 40);
        let mut signals = 0usize;
        let mut points = 0usize;
        for _ in 0..trials {
            let data = correlated(&mut rng, m, 3);
            let chart = phase_i(&data, alpha).unwrap();
            signals += chart.out_of_control.len();
            points += m;
        }
        let rate = signals as f64 / points as f64;
        // The sum of twelve uniforms is not exactly normal; the rate stays near alpha.
        assert!((rate - alpha).abs() < 0.015, "signal rate {rate}");
    }

    #[test]
    fn phase_ii_judges_new_points_against_the_estimates() {
        let mut rng = Lcg(9);
        let history = correlated(&mut rng, 50, 3);
        let estimates = estimate_from(&history).unwrap();
        let fresh = correlated(&mut rng, 20, 3);
        let chart = phase_ii(&fresh, &estimates, 0.0027).unwrap();
        assert_eq!(chart.t2.len(), 20);
        for (row, &t2) in fresh.iter().zip(&chart.t2) {
            let want = mahalanobis_by_elimination(row, &estimates.mean, &estimates.covariance);
            assert!((t2 - want).abs() <= 1e-9 * want.max(1.0));
        }
        let shifted = vec![vec![
            estimates.mean[0] + 30.0,
            estimates.mean[1],
            estimates.mean[2],
        ]];
        assert_eq!(
            phase_ii(&shifted, &estimates, 0.0027)
                .unwrap()
                .out_of_control,
            vec![0]
        );
    }

    #[test]
    fn refusals_name_what_is_wrong() {
        let rows = |n: usize, p: usize| -> Vec<Vec<f64>> {
            (0..n)
                .map(|i| {
                    (0..p)
                        .map(|j| (i * (j + 2)) as f64 + (i * i) as f64 * 0.1 * j as f64)
                        .collect()
                })
                .collect()
        };
        assert_eq!(
            phase_i(&rows(10, 2), 0.0).unwrap_err(),
            HotellingError::InvalidAlpha { got: 0.0 }
        );
        assert_eq!(
            phase_i(&rows(10, 2), 1.0).unwrap_err(),
            HotellingError::InvalidAlpha { got: 1.0 }
        );
        assert!(matches!(
            phase_i(&rows(10, 2), f64::NAN).unwrap_err(),
            HotellingError::InvalidAlpha { .. }
        ));
        assert_eq!(
            phase_i(&rows(3, 2), 0.05).unwrap_err(),
            HotellingError::TooFew { min: 4, got: 3 }
        );
        let mut ragged = rows(10, 2);
        ragged[4].pop();
        assert_eq!(
            phase_i(&ragged, 0.05).unwrap_err(),
            HotellingError::DimensionMismatch {
                parameter: "data",
                expected: 2,
                got: 1,
                index: Some(4)
            }
        );
        let mut bad = rows(10, 2);
        bad[2][1] = f64::NAN;
        assert_eq!(
            phase_i(&bad, 0.05).unwrap_err(),
            HotellingError::NonFinite {
                parameter: "data",
                index: Some(2)
            }
        );
        // A constant variable, and a variable that is twice another: singular covariance.
        let constant: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64, 5.0]).collect();
        assert_eq!(
            phase_i(&constant, 0.05).unwrap_err(),
            HotellingError::Degenerate
        );
        let doubled: Vec<Vec<f64>> = (0..10)
            .map(|i| {
                vec![
                    i as f64 + (i * i) as f64 * 0.01,
                    2.0 * (i as f64 + (i * i) as f64 * 0.01),
                ]
            })
            .collect();
        assert_eq!(
            phase_i(&doubled, 0.05).unwrap_err(),
            HotellingError::Degenerate
        );
        // Collinear to rounding (a variable that is twice another plus a 2e-6 flicker): the
        // factorisation succeeds but its last pivot is noise, so T² would be meaningless.
        let flicker = |size: f64| -> Vec<Vec<f64>> {
            (0..12)
                .map(|i| {
                    let x = i as f64 + (i * i) as f64 * 0.01;
                    vec![x, 2.0 * x + size * (i % 3) as f64]
                })
                .collect()
        };
        assert_eq!(
            phase_i(&flicker(2e-6), 0.05).unwrap_err(),
            HotellingError::Degenerate
        );
        // ... while the same shape with a real spread is a chart.
        assert!(phase_i(&flicker(1e-3), 0.05).is_ok());
        // Phase II: estimates of the wrong width, too few behind them.
        let estimates = estimate_from(&rows(10, 2)).unwrap();
        assert!(matches!(
            phase_ii(&[vec![1.0, 2.0, 3.0]], &estimates, 0.05).unwrap_err(),
            HotellingError::DimensionMismatch {
                parameter: "observations",
                ..
            }
        ));
        let thin = HotellingEstimates {
            mean: vec![0.0, 0.0],
            covariance: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            m: 2,
        };
        assert_eq!(
            phase_ii(&[vec![1.0, 1.0]], &thin, 0.05).unwrap_err(),
            HotellingError::TooFew { min: 3, got: 2 }
        );
    }

    // ----- subgroups -----

    /// `m` subgroups of `n` observations of `p` correlated variables: a common mean path, a
    /// common within-subgroup noise.
    fn subgroups(rng: &mut Lcg, m: usize, n: usize, p: usize) -> Vec<Vec<Vec<f64>>> {
        (0..m)
            .map(|_| {
                (0..n)
                    .map(|_| {
                        let common = rng.bell();
                        (0..p)
                            .map(|j| {
                                10.0 + j as f64 + common * (1.0 + j as f64 * 0.3) + rng.bell() * 0.5
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn subgroup_t_squared_is_n_times_the_mahalanobis_distance_of_the_mean() {
        let mut rng = Lcg(21);
        let data = subgroups(&mut rng, 12, 4, 3);
        let chart = subgroup_phase_i(&data, 0.01).expect("a chart");
        let p = 3;
        // The definition, from scratch: the subgroup means, the grand mean, the average of the
        // within-subgroup covariances, and an elimination solve for the distance.
        let means: Vec<Vec<f64>> = data
            .iter()
            .map(|g| {
                (0..p)
                    .map(|j| g.iter().map(|r| r[j]).sum::<f64>() / g.len() as f64)
                    .collect()
            })
            .collect();
        let grand: Vec<f64> = (0..p)
            .map(|j| means.iter().map(|x| x[j]).sum::<f64>() / means.len() as f64)
            .collect();
        let mut pooled = vec![vec![0.0; p]; p];
        for (g, mean) in data.iter().zip(&means) {
            for a in 0..p {
                for b in 0..p {
                    let s: f64 = g.iter().map(|r| (r[a] - mean[a]) * (r[b] - mean[b])).sum();
                    pooled[a][b] += s / (g.len() - 1) as f64 / data.len() as f64;
                }
            }
        }
        for (i, x) in means.iter().enumerate() {
            let want = 4.0 * mahalanobis_by_elimination(x, &grand, &pooled);
            assert!(
                (chart.t2[i] - want).abs() <= 1e-9 * want.max(1.0),
                "subgroup {i}: {} vs {want}",
                chart.t2[i]
            );
        }
        for (a, b) in chart.estimates.mean.iter().zip(&grand) {
            assert!((a - b).abs() < 1e-12);
        }
        assert_eq!((chart.estimates.m, chart.estimates.n), (12, 4));
    }

    #[test]
    fn the_subgroup_limits_are_the_one_minus_alpha_quantiles_of_their_distribution() {
        for (m, n, p) in [(8, 3, 2), (20, 5, 3), (50, 4, 6)] {
            for alpha in [0.0027, 0.05] {
                let df = (m * n - m - p + 1) as f64;
                for (limit, factor) in [
                    (subgroup_phase_i_limit(m, n, p, alpha), m as f64 - 1.0),
                    (subgroup_phase_ii_limit(m, n, p, alpha), m as f64 + 1.0),
                ] {
                    let ucl = limit.expect("a limit");
                    let f = ucl * df / (p as f64 * factor * (n as f64 - 1.0));
                    let cdf = f_distribution_cdf(f, p as f64, df);
                    assert!(
                        (cdf - (1.0 - alpha)).abs() < 1e-9,
                        "m {m} n {n} p {p} alpha {alpha}: cdf {cdf}"
                    );
                }
            }
        }
        // New subgroups are farther from estimates than the subgroups that made them.
        assert!(
            subgroup_phase_ii_limit(10, 4, 2, 0.01).unwrap()
                > subgroup_phase_i_limit(10, 4, 2, 0.01).unwrap()
        );
    }

    #[test]
    fn a_subgroup_with_a_shifted_mean_is_flagged_and_the_rest_are_not() {
        let mut rng = Lcg(33);
        let mut data = subgroups(&mut rng, 25, 5, 2);
        for row in &mut data[17] {
            row[0] += 4.0;
            row[1] -= 4.0;
        }
        let chart = subgroup_phase_i(&data, 0.0027).expect("a chart");
        assert!(
            chart.out_of_control.contains(&17),
            "{:?}",
            chart.out_of_control
        );
        assert!(
            chart.out_of_control.len() <= 2,
            "{:?}",
            chart.out_of_control
        );
        // The same estimates, a Phase II chart: the shifted subgroup is far out, a typical one is in.
        let clean: Vec<Vec<Vec<f64>>> = data
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != 17)
            .map(|(_, g)| g.clone())
            .collect();
        let base = subgroup_phase_i(&clean, 0.0027).expect("cleaned");
        let next = subgroup_phase_ii(
            &[data[17].clone(), clean[0].clone()],
            &base.estimates,
            0.0027,
        )
        .expect("a chart");
        assert_eq!(next.out_of_control, vec![0]);
        assert!(subgroup_phase_ii(&[], &base.estimates, 0.0027)
            .unwrap()
            .t2
            .is_empty());
    }

    #[test]
    fn in_control_subgroups_signal_about_alpha_of_the_time() {
        let mut rng = Lcg(77);
        let (alpha, mut signals, mut total) = (0.05, 0, 0);
        for _ in 0..60 {
            let data = subgroups(&mut rng, 30, 4, 3);
            let chart = subgroup_phase_i(&data, alpha).expect("a chart");
            signals += chart.out_of_control.len();
            total += data.len();
        }
        let rate = signals as f64 / total as f64;
        // 1800 points: the standard error is about 0.005.
        assert!((rate - alpha).abs() < 0.02, "signal rate {rate}");
    }

    #[test]
    fn subgroup_refusals_name_the_subgroup() {
        let mut rng = Lcg(5);
        let good = subgroups(&mut rng, 6, 3, 2);
        assert_eq!(
            subgroup_phase_i(&good, 1.0).unwrap_err(),
            HotellingError::InvalidAlpha { got: 1.0 }
        );
        assert!(matches!(
            subgroup_phase_i(&[], 0.01),
            Err(HotellingError::DimensionMismatch {
                parameter: "subgroups",
                ..
            })
        ));
        // One observation per subgroup has no within-subgroup covariance.
        let single: Vec<Vec<Vec<f64>>> = good.iter().map(|g| vec![g[0].clone()]).collect();
        assert_eq!(
            subgroup_phase_i(&single, 0.01).unwrap_err(),
            HotellingError::TooFew { min: 2, got: 1 }
        );
        // A subgroup of another size, width or with a NaN is named.
        let mut odd = good.clone();
        odd[4].pop();
        assert_eq!(
            subgroup_phase_i(&odd, 0.01).unwrap_err(),
            HotellingError::DimensionMismatch {
                parameter: "subgroups",
                expected: 3,
                got: 2,
                index: Some(4)
            }
        );
        let mut narrow = good.clone();
        narrow[2][1].pop();
        assert_eq!(
            subgroup_phase_i(&narrow, 0.01).unwrap_err(),
            HotellingError::DimensionMismatch {
                parameter: "subgroups",
                expected: 2,
                got: 1,
                index: Some(2)
            }
        );
        let mut nan = good.clone();
        nan[5][0][1] = f64::NAN;
        assert_eq!(
            subgroup_phase_i(&nan, 0.01).unwrap_err(),
            HotellingError::NonFinite {
                parameter: "subgroups",
                index: Some(5)
            }
        );
        // Too few observations for the pooled covariance of 4 variables: m (n - 1) = 3 < 4.
        let thin = subgroups(&mut rng, 3, 2, 4);
        assert_eq!(
            subgroup_phase_i(&thin, 0.01).unwrap_err(),
            HotellingError::TooFew { min: 7, got: 6 }
        );
        // A constant variable has no within-subgroup variation.
        let flat: Vec<Vec<Vec<f64>>> = good
            .iter()
            .map(|g| g.iter().map(|r| vec![r[0], 5.0]).collect())
            .collect();
        assert_eq!(
            subgroup_phase_i(&flat, 0.01).unwrap_err(),
            HotellingError::Degenerate
        );
        // Phase II: a new subgroup of the wrong size, and of the wrong width.
        let base = subgroup_phase_i(&good, 0.01).expect("a chart");
        let mut short = good[..2].to_vec();
        short[1].pop();
        assert_eq!(
            subgroup_phase_ii(&short, &base.estimates, 0.01).unwrap_err(),
            HotellingError::DimensionMismatch {
                parameter: "subgroups",
                expected: 3,
                got: 2,
                index: Some(1)
            }
        );
        let wide: Vec<Vec<Vec<f64>>> = vec![vec![vec![1.0, 2.0, 3.0]; 3]];
        assert!(matches!(
            subgroup_phase_ii(&wide, &base.estimates, 0.01),
            Err(HotellingError::DimensionMismatch {
                parameter: "subgroups",
                expected: 2,
                got: 3,
                ..
            })
        ));
    }

    // -----------------------------------------------------------------------------------------
    // Which variable moved
    // -----------------------------------------------------------------------------------------

    /// A positive definite covariance `B Bᵀ + ½ I` and a mean, from the seeded source.
    fn random_estimates(rng: &mut Lcg, p: usize) -> HotellingEstimates {
        let b: Vec<Vec<f64>> = (0..p)
            .map(|_| (0..p).map(|_| rng.bell()).collect())
            .collect();
        let covariance = (0..p)
            .map(|i| {
                (0..p)
                    .map(|j| {
                        let dot: f64 = (0..p).map(|k| b[i][k] * b[j][k]).sum();
                        dot + if i == j { 0.5 } else { 0.0 }
                    })
                    .collect()
            })
            .collect();
        HotellingEstimates {
            mean: (0..p).map(|_| rng.bell()).collect(),
            covariance,
            m: 50,
        }
    }

    /// `T²` of the observation on the variables but `j`, from the definition, by elimination.
    fn t_squared_without(x: &[f64], e: &HotellingEstimates, j: usize) -> f64 {
        let keep: Vec<usize> = (0..x.len()).filter(|&k| k != j).collect();
        let sub_x: Vec<f64> = keep.iter().map(|&k| x[k]).collect();
        let sub_mean: Vec<f64> = keep.iter().map(|&k| e.mean[k]).collect();
        let sub_s: Vec<Vec<f64>> = keep
            .iter()
            .map(|&a| keep.iter().map(|&b| e.covariance[a][b]).collect())
            .collect();
        mahalanobis_by_elimination(&sub_x, &sub_mean, &sub_s)
    }

    #[test]
    fn each_contribution_is_what_the_definition_says() {
        let mut rng = Lcg(11);
        let mut checked = 0;
        for p in 2..=7usize {
            for _ in 0..30 {
                let e = random_estimates(&mut rng, p);
                let x: Vec<f64> = e.mean.iter().map(|m| m + 3.0 * rng.bell()).collect();
                let c = contributions(&x, &e).unwrap();
                let full = mahalanobis_by_elimination(&x, &e.mean, &e.covariance);
                assert!((c.t2 - full).abs() <= 1e-9 * full.max(1.0), "t2 p {p}");
                for j in 0..p {
                    let want = full - t_squared_without(&x, &e, j);
                    assert!(
                        (c.drop_one[j] - want).abs() <= 1e-8 * full.max(1.0),
                        "p {p} j {j}: {} vs {want}",
                        c.drop_one[j]
                    );
                    let r = x[j] - e.mean[j];
                    let own = r * r / e.covariance[j][j];
                    assert!((c.unconditional[j] - own).abs() <= 1e-12 * own.max(1.0));
                    assert_eq!(c.drop_one[j], c.z[j] * c.z[j]);
                    checked += 1;
                }
            }
        }
        assert_eq!(checked, 30 * (2 + 3 + 4 + 5 + 6 + 7));
    }

    #[test]
    fn the_sign_is_the_direction_the_variable_departs_from_what_the_others_predict() {
        // Correlation 0.9: with the second variable at its mean the first predicts 0 for it, so
        // a first variable above its mean puts the second *below* what it predicts.
        let e = HotellingEstimates {
            mean: vec![0.0, 0.0],
            covariance: vec![vec![1.0, 0.9], vec![0.9, 1.0]],
            m: 50,
        };
        let c = contributions(&[1.0, 0.0], &e).unwrap();
        assert!(c.z[0] > 0.0 && c.z[1] < 0.0, "{:?}", c.z);
        let c = contributions(&[-1.0, 0.0], &e).unwrap();
        assert!(c.z[0] < 0.0 && c.z[1] > 0.0, "{:?}", c.z);
    }

    #[test]
    fn the_entries_are_not_shares_of_t_squared() {
        // Worked by hand at correlation 0.9 (S⁻¹ = [[1, −.9], [−.9, 1]] / 0.19).
        let e = HotellingEstimates {
            mean: vec![0.0, 0.0],
            covariance: vec![vec![1.0, 0.9], vec![0.9, 1.0]],
            m: 50,
        };
        // Both up together: they predict each other, T² = 0.2 / 0.19.
        let c = contributions(&[1.0, 1.0], &e).unwrap();
        assert!((c.t2 - 0.2 / 0.19).abs() < 1e-12);
        for v in &c.drop_one {
            assert!((v - (0.2 / 0.19 - 1.0)).abs() < 1e-12, "{v}");
        }
        assert!(c.drop_one.iter().sum::<f64>() < c.t2);
        // One up, one down: T² = 3.8 / 0.19 = 20, each entry 19, the sum 38 is twice T².
        let c = contributions(&[1.0, -1.0], &e).unwrap();
        assert!((c.t2 - 20.0).abs() < 1e-12);
        for v in &c.drop_one {
            assert!((v - 19.0).abs() < 1e-12, "{v}");
        }
        assert!(c.drop_one.iter().sum::<f64>() > c.t2);
    }

    #[test]
    fn a_variable_that_did_not_move_gets_rho_squared_of_the_one_that_did() {
        // The smearing the documentation warns of: only the first variable moved.
        for rho in [0.5, 0.9, 0.99] {
            let e = HotellingEstimates {
                mean: vec![0.0, 0.0],
                covariance: vec![vec![1.0, rho], vec![rho, 1.0]],
                m: 50,
            };
            let c = contributions(&[2.0, 0.0], &e).unwrap();
            assert!(
                (c.drop_one[1] / c.drop_one[0] - rho * rho).abs() < 1e-12,
                "rho {rho}"
            );
            // On its own the second variable shows nothing.
            assert_eq!(c.unconditional[1], 0.0);
        }
    }

    #[test]
    fn uncorrelated_variables_contribute_what_they_do_on_their_own() {
        let e = HotellingEstimates {
            mean: vec![1.0, -2.0, 0.5],
            covariance: vec![
                vec![4.0, 0.0, 0.0],
                vec![0.0, 0.25, 0.0],
                vec![0.0, 0.0, 9.0],
            ],
            m: 30,
        };
        let c = contributions(&[3.0, -1.0, 0.0], &e).unwrap();
        for j in 0..3 {
            assert!((c.drop_one[j] - c.unconditional[j]).abs() < 1e-12);
        }
        // One variable: its whole T².
        let one = HotellingEstimates {
            mean: vec![2.0],
            covariance: vec![vec![0.5]],
            m: 10,
        };
        let c = contributions(&[3.0], &one).unwrap();
        assert!((c.t2 - 2.0).abs() < 1e-12 && (c.drop_one[0] - 2.0).abs() < 1e-12);
        assert!((c.z[0] - 2f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn contributions_do_not_change_when_a_variable_is_rescaled_or_shifted() {
        let mut rng = Lcg(23);
        for p in [2usize, 4, 6] {
            let e = random_estimates(&mut rng, p);
            let x: Vec<f64> = e.mean.iter().map(|m| m + 2.0 * rng.bell()).collect();
            let c = contributions(&x, &e).unwrap();
            // Variable `k` in other units (x ↦ a x + b).
            let (k, a, b) = (p / 2, 37.0, -1_000.0);
            let mut e2 = e.clone();
            let mut x2 = x.clone();
            e2.mean[k] = a * e.mean[k] + b;
            x2[k] = a * x[k] + b;
            for i in 0..p {
                if i != k {
                    e2.covariance[i][k] *= a;
                    e2.covariance[k][i] *= a;
                }
            }
            e2.covariance[k][k] *= a * a;
            let c2 = contributions(&x2, &e2).unwrap();
            assert!((c.t2 - c2.t2).abs() <= 1e-8 * c.t2.max(1.0));
            for j in 0..p {
                assert!((c.drop_one[j] - c2.drop_one[j]).abs() <= 1e-8 * c.t2.max(1.0));
                assert!((c.z[j].abs() - c2.z[j].abs()).abs() <= 1e-8 * c.z[j].abs().max(1.0));
                assert!((c.unconditional[j] - c2.unconditional[j]).abs() <= 1e-8 * c.t2.max(1.0));
            }
        }
    }

    #[test]
    fn the_individual_contributions_belong_to_the_t_squared_of_the_charts() {
        let mut rng = Lcg(5);
        let data = correlated(&mut rng, 40, 4);
        let chart = phase_i(&data, 0.01).unwrap();
        let c = contributions(&data[7], &chart.estimates).unwrap();
        assert!((c.t2 - chart.t2[7]).abs() <= 1e-9 * chart.t2[7].max(1.0));
        let two = phase_ii(&[data[7].clone()], &chart.estimates, 0.01).unwrap();
        assert!((c.t2 - two.t2[0]).abs() <= 1e-9 * c.t2.max(1.0));
    }

    #[test]
    fn a_subgroup_carries_the_size_of_the_subgroup() {
        let mut rng = Lcg(9);
        let e = random_estimates(&mut rng, 3);
        let x: Vec<f64> = e.mean.iter().map(|m| m + rng.bell()).collect();
        let single = contributions(&x, &e).unwrap();
        let n = 5usize;
        let sub = SubgroupEstimates {
            mean: e.mean.clone(),
            covariance: e.covariance.clone(),
            m: 20,
            n,
        };
        let many = subgroup_contributions(&x, &sub).unwrap();
        let nf = n as f64;
        assert!((many.t2 - nf * single.t2).abs() <= 1e-9 * many.t2.max(1.0));
        for j in 0..3 {
            assert!((many.drop_one[j] - nf * single.drop_one[j]).abs() <= 1e-9 * many.t2.max(1.0));
            assert!(
                (many.z[j] - nf.sqrt() * single.z[j]).abs() <= 1e-9 * single.z[j].abs().max(1.0)
            );
            assert!(
                (many.unconditional[j] - nf * single.unconditional[j]).abs()
                    <= 1e-9 * many.t2.max(1.0)
            );
        }
        // The chart of subgroups gives the same T² for the same mean.
        let rows: Vec<Vec<f64>> = (0..n)
            .map(|i| x.iter().map(|v| v + (i as f64 - 2.0) * 0.1).collect())
            .collect();
        let mean_of_rows: Vec<f64> = (0..3)
            .map(|j| rows.iter().map(|r| r[j]).sum::<f64>() / nf)
            .collect();
        let chart = subgroup_phase_ii(&[rows], &sub, 0.01).unwrap();
        let by_mean = subgroup_contributions(&mean_of_rows, &sub).unwrap();
        assert!((chart.t2[0] - by_mean.t2).abs() <= 1e-9 * by_mean.t2.max(1.0));
    }

    #[test]
    fn contribution_refusals_name_what_is_wrong() {
        let e = HotellingEstimates {
            mean: vec![0.0, 0.0],
            covariance: vec![vec![1.0, 0.5], vec![0.5, 1.0]],
            m: 20,
        };
        assert_eq!(
            contributions(&[1.0], &e).unwrap_err(),
            HotellingError::DimensionMismatch {
                parameter: "observation",
                expected: 2,
                got: 1,
                index: None
            }
        );
        assert_eq!(
            contributions(&[1.0, f64::NAN], &e).unwrap_err(),
            HotellingError::NonFinite {
                parameter: "observation",
                index: None
            }
        );
        let singular = HotellingEstimates {
            covariance: vec![vec![1.0, 1.0], vec![1.0, 1.0]],
            ..e.clone()
        };
        assert_eq!(
            contributions(&[1.0, 2.0], &singular).unwrap_err(),
            HotellingError::Degenerate
        );
        let bad_n = SubgroupEstimates {
            mean: e.mean.clone(),
            covariance: e.covariance.clone(),
            m: 20,
            n: 1,
        };
        assert_eq!(
            subgroup_contributions(&[1.0, 2.0], &bad_n).unwrap_err(),
            HotellingError::TooFew { min: 2, got: 1 }
        );
    }
}
