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
//! The limits are exact distribution quantiles, not asymptotic chi-squared ones, so they hold
//! for small `m`. A significance `α` is the per-point false-alarm probability when the process
//! is in control and normal.
//!
//! # References
//!
//! - Tracy, N.D., Young, J.C. & Mason, R.L. (1992). "Multivariate control charts for
//!   individual observations", *Journal of Quality Technology* 24(2), pp. 88-95.
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

/// `(x − μ)ᵀ S⁻¹ (x − μ)` for every row, by a Cholesky factorisation of `S` (the factor is
/// built once; each row is one pair of triangular solves). A covariance that is not positive
/// definite is `Degenerate`.
fn t_squared(
    rows: &[Vec<f64>],
    mean: &[f64],
    covariance: &[Vec<f64>],
) -> Result<Vec<f64>, HotellingError> {
    let p = mean.len();
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
    let p = estimates.mean.len();
    if p == 0 {
        return Err(HotellingError::DimensionMismatch {
            parameter: "mean",
            expected: 1,
            got: 0,
            index: None,
        });
    }
    if estimates.mean.iter().any(|v| !v.is_finite()) {
        return Err(HotellingError::NonFinite {
            parameter: "mean",
            index: None,
        });
    }
    if estimates.covariance.len() != p {
        return Err(HotellingError::DimensionMismatch {
            parameter: "covariance",
            expected: p,
            got: estimates.covariance.len(),
            index: None,
        });
    }
    for (i, row) in estimates.covariance.iter().enumerate() {
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
}
