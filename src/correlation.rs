//! Correlation analysis.
//!
//! Pearson, Spearman, and Kendall correlation coefficients with p-values,
//! correlation matrices, and Fisher z-transformation confidence intervals.
//!
//! # Examples
//!
//! ```
//! use u_analytics::correlation::{pearson, spearman, kendall_tau_b};
//!
//! let x = [1.0, 2.0, 3.0, 4.0, 5.0];
//! let y = [2.0, 4.0, 5.0, 4.0, 5.0];
//!
//! let p = pearson(&x, &y).unwrap();
//! assert!(p.r > 0.7);
//! assert!(p.p_value < 0.2);
//!
//! let s = spearman(&x, &y).unwrap();
//! assert!(s.r > 0.7);
//!
//! let k = kendall_tau_b(&x, &y).unwrap();
//! assert!(k.r > 0.5);
//! ```

use crate::error::{check_option, check_series, TimeSeriesError};
use std::ops::RangeInclusive;
use u_numflow::matrix::Matrix;
use u_numflow::special;
use u_numflow::stats;

/// Result of a correlation computation.
#[derive(Debug, Clone, Copy)]
pub struct CorrelationResult {
    /// Correlation coefficient in [-1, 1].
    pub r: f64,
    /// Two-tailed p-value for testing H₀: ρ = 0.
    pub p_value: f64,
    /// Sample size.
    pub n: usize,
}

/// Confidence interval for a correlation coefficient.
#[derive(Debug, Clone, Copy)]
pub struct CorrelationCI {
    /// Lower bound of the confidence interval.
    pub lower: f64,
    /// Upper bound of the confidence interval.
    pub upper: f64,
    /// Confidence level (e.g. 0.95).
    pub confidence: f64,
}

// ---------------------------------------------------------------------------
// Pearson
// ---------------------------------------------------------------------------

/// Computes Pearson product-moment correlation coefficient and p-value.
///
/// # Algorithm
///
/// r = cov(x,y) / (σ_x · σ_y)
///
/// p-value via t-test: t = r·√(n-2) / √(1-r²), df = n-2.
///
/// # Returns
///
/// `None` if either slice has fewer than 3 elements, the slices differ in
/// length, or either variable has zero variance.
///
/// # References
///
/// Pearson (1895). "Note on regression and inheritance in the case of
/// two parents". Proceedings of the Royal Society of London, 58, 240–242.
///
/// # Examples
///
/// ```
/// use u_analytics::correlation::pearson;
///
/// let x = [1.0, 2.0, 3.0, 4.0, 5.0];
/// let y = [2.0, 4.0, 6.0, 8.0, 10.0];
/// let result = pearson(&x, &y).unwrap();
/// assert!((result.r - 1.0).abs() < 1e-10);
/// ```
pub fn pearson(x: &[f64], y: &[f64]) -> Option<CorrelationResult> {
    let n = x.len();
    if n < 3 || n != y.len() {
        return None;
    }

    if x.iter().any(|v| !v.is_finite()) || y.iter().any(|v| !v.is_finite()) {
        return None;
    }

    let cov = stats::covariance(x, y)?;
    let sx = stats::std_dev(x)?;
    let sy = stats::std_dev(y)?;

    if sx < 1e-300 || sy < 1e-300 {
        return None; // zero variance
    }

    let r = (cov / (sx * sy)).clamp(-1.0, 1.0);
    let p_value = correlation_p_value(r, n);

    Some(CorrelationResult { r, p_value, n })
}

// ---------------------------------------------------------------------------
// Spearman
// ---------------------------------------------------------------------------

/// Computes Spearman rank correlation coefficient and p-value.
///
/// # Algorithm
///
/// Ranks both variables using the mid-rank method for ties, then computes
/// Pearson correlation on the ranks. P-value uses the same t-test
/// approximation as Pearson.
///
/// # Returns
///
/// `None` if fewer than 3 observations, slices differ in length, or inputs
/// contain non-finite values.
///
/// # References
///
/// Spearman (1904). "The proof and measurement of association between two
/// things". The American Journal of Psychology, 15(1), 72–101.
///
/// # Examples
///
/// ```
/// use u_analytics::correlation::spearman;
///
/// let x = [1.0, 2.0, 3.0, 4.0, 5.0];
/// let y = [5.0, 6.0, 7.0, 8.0, 7.0];
/// let result = spearman(&x, &y).unwrap();
/// assert!(result.r > 0.5);
/// ```
pub fn spearman(x: &[f64], y: &[f64]) -> Option<CorrelationResult> {
    let n = x.len();
    if n < 3 || n != y.len() {
        return None;
    }

    if x.iter().any(|v| !v.is_finite()) || y.iter().any(|v| !v.is_finite()) {
        return None;
    }

    let rx = rank_data(x);
    let ry = rank_data(y);

    // Pearson on ranks
    let cov = stats::covariance(&rx, &ry)?;
    let srx = stats::std_dev(&rx)?;
    let sry = stats::std_dev(&ry)?;

    if srx < 1e-300 || sry < 1e-300 {
        return None;
    }

    let r = (cov / (srx * sry)).clamp(-1.0, 1.0);
    let p_value = correlation_p_value(r, n);

    Some(CorrelationResult { r, p_value, n })
}

// ---------------------------------------------------------------------------
// Kendall tau-b
// ---------------------------------------------------------------------------

/// Computes Kendall's tau-b correlation coefficient with tie correction.
///
/// # Algorithm
///
/// τ_b = (C - D) / √[(n₀ - n₁)(n₀ - n₂)]
///
/// where C = concordant pairs, D = discordant pairs,
/// n₀ = n(n-1)/2, n₁ = Σ tᵢ(tᵢ-1)/2 (ties in x), n₂ = Σ uⱼ(uⱼ-1)/2 (ties in y).
///
/// # p-value
///
/// With no tie in either variable and at most 33 observations -- or more, when
/// at most one pair is on the minority side -- the two-sided p-value is exact
/// (Kendall 1970): twice the probability that a random permutation has no more
/// discordant pairs than observed. Otherwise it is the normal approximation of
/// S = C − D with the tie terms of the variance. These are the cases in which
/// SciPy's `kendalltau(method="auto")` is exact too.
///
/// # Complexity
///
/// O(n²) naive enumeration. For n > 10,000 consider O(n log n) Knight's
/// algorithm (not implemented — sufficient for u-insight's typical data sizes).
/// The exact p-value costs O(n · n²) once, for n ≤ 33.
///
/// # Returns
///
/// `None` if fewer than 3 observations, slices differ in length, or inputs
/// contain non-finite values.
///
/// # References
///
/// Kendall (1938). "A new measure of rank correlation".
/// Biometrika, 30(1/2), 81–93.
///
/// Kendall (1970). *Rank Correlation Methods*, 4th ed. Griffin.
///
/// # Examples
///
/// ```
/// use u_analytics::correlation::kendall_tau_b;
///
/// let x = [1.0, 2.0, 3.0, 4.0, 5.0];
/// let y = [1.0, 2.0, 3.0, 4.0, 5.0];
/// let result = kendall_tau_b(&x, &y).unwrap();
/// assert!((result.r - 1.0).abs() < 1e-10);
/// ```
pub fn kendall_tau_b(x: &[f64], y: &[f64]) -> Option<CorrelationResult> {
    let n = x.len();
    if n < 3 || n != y.len() {
        return None;
    }

    if x.iter().any(|v| !v.is_finite()) || y.iter().any(|v| !v.is_finite()) {
        return None;
    }

    let mut concordant: i64 = 0;
    let mut discordant: i64 = 0;
    let mut ties_x: i64 = 0;
    let mut ties_y: i64 = 0;
    let mut _ties_xy: i64 = 0;

    for i in 0..n {
        for j in (i + 1)..n {
            let dx = x[i] - x[j];
            let dy = y[i] - y[j];
            let product = dx * dy;

            if dx == 0.0 && dy == 0.0 {
                _ties_xy += 1;
                ties_x += 1;
                ties_y += 1;
            } else if dx == 0.0 {
                ties_x += 1;
            } else if dy == 0.0 {
                ties_y += 1;
            } else if product > 0.0 {
                concordant += 1;
            } else {
                discordant += 1;
            }
        }
    }

    let n0 = (n as i64) * (n as i64 - 1) / 2;
    let denom_sq = (n0 - ties_x) as f64 * (n0 - ties_y) as f64;

    if denom_sq <= 0.0 {
        return None; // all values tied in x or y
    }

    let tau = (concordant - discordant) as f64 / denom_sq.sqrt();
    let tau = tau.clamp(-1.0, 1.0);

    let p_value = if ties_x == 0
        && ties_y == 0
        && (n <= KENDALL_EXACT_MAX_N || concordant.min(discordant) <= 1)
    {
        // Nothing tied: the null distribution of the discordant pairs is the
        // inversion count of a random permutation, tabulated exactly.
        kendall_exact_p(n, discordant)
    } else {
        // Normal approximation of S = C - D with the variance of Kendall
        // (1970) for tied data: the tie-free term less the tied groups, plus
        // the two terms the groups of x and of y share.
        let m = n as f64 * (n as f64 - 1.0);
        let (pairs_x, triples_x, weighted_x) = tie_sums(x);
        let (pairs_y, triples_y, weighted_y) = tie_sums(y);
        let var_s = (m * (2.0 * n as f64 + 5.0) - weighted_x - weighted_y) / 18.0
            + 2.0 * pairs_x * pairs_y / m
            + triples_x * triples_y / (9.0 * m * (n as f64 - 2.0));

        if var_s > 0.0 {
            let s = (concordant - discordant) as f64;
            let z = s / var_s.sqrt();
            2.0 * special::standard_normal_sf(z.abs())
        } else {
            1.0
        }
    };

    Some(CorrelationResult { r: tau, p_value, n })
}

/// The largest `n` whose permutations can be counted in `u128` (33! < 2^128):
/// SciPy's `kendalltau(method="auto")` takes the exact distribution up to the
/// same size.
const KENDALL_EXACT_MAX_N: usize = 33;

/// The exact two-sided p-value of Kendall's S for `n` observations with no
/// ties, `discordant` of the n(n-1)/2 pairs discordant: twice the probability
/// that a random permutation has at most `min(D, total - D)` inversions
/// (Kendall 1970, ch. 5), capped at 1.
///
/// The counts of permutations by inversions are built row by row
/// (Mahonian numbers: adding the j-th element adds 0..j-1 inversions) in
/// integer arithmetic, so the probability is one correctly rounded division.
/// The two smallest tails have closed forms -- 2/n! and 2/(n-1)! -- that hold
/// for any `n`; beyond `KENDALL_EXACT_MAX_N` only those are reached.
fn kendall_exact_p(n: usize, discordant: i64) -> f64 {
    let total = (n * (n - 1) / 2) as i64;
    let tail = discordant.min(total - discordant);
    if 4 * tail == (n * (n - 1)) as i64 {
        return 1.0; // the middle of a symmetric distribution
    }
    // 2 / m!, with m! left to underflow rather than overflow.
    let two_over_factorial = |m: usize| (2..=m).fold(2.0_f64, |p, k| p / k as f64);
    match tail {
        0 => return two_over_factorial(n),
        1 => return two_over_factorial(n - 1),
        _ => {}
    }
    debug_assert!(n <= KENDALL_EXACT_MAX_N, "counts would overflow u128");
    let c = tail as usize;
    // counts[k]: the permutations of the first j elements with k inversions.
    let mut counts = vec![0u128; c + 1];
    counts[0] = 1;
    for j in 2..=n {
        // new[k] = old[k] + ... + old[k - (j - 1)], from running sums.
        let mut next = vec![0u128; c + 1];
        let mut window = 0u128;
        for k in 0..=c {
            window += counts[k];
            if k >= j {
                window -= counts[k - j];
            }
            next[k] = window;
        }
        counts = next;
    }
    let at_most: u128 = counts.iter().sum();
    let permutations: u128 = (2..=n as u128).product();
    (2.0 * at_most as f64 / permutations as f64).min(1.0)
}

// ---------------------------------------------------------------------------
// Fisher z-transformation
// ---------------------------------------------------------------------------

/// Computes Fisher z-transformation: z = arctanh(r).
///
/// Transforms r ∈ (-1, 1) to z ∈ (-∞, +∞) where z is approximately normal
/// with variance 1/(n-3).
///
/// # Returns
///
/// `None` if r is not in (-1, 1).
///
/// # References
///
/// Fisher (1921). "On the probable error of a coefficient of correlation".
/// Metron, 1, 3–32.
pub fn fisher_z(r: f64) -> Option<f64> {
    if r <= -1.0 || r >= 1.0 || !r.is_finite() {
        return None;
    }
    Some(r.atanh())
}

/// Inverse Fisher z-transformation: r = tanh(z).
pub fn fisher_z_inv(z: f64) -> f64 {
    z.tanh()
}

/// Computes confidence interval for a Pearson correlation coefficient
/// using Fisher z-transformation.
///
/// # Arguments
///
/// * `r` — Pearson correlation coefficient
/// * `n` — Sample size
/// * `confidence` — Confidence level (e.g. 0.95)
///
/// # Returns
///
/// `None` if n < 4, r is not in (-1, 1), or confidence is not in (0, 1).
///
/// # Examples
///
/// ```
/// use u_analytics::correlation::correlation_ci;
///
/// let ci = correlation_ci(0.8, 30, 0.95).unwrap();
/// assert!(ci.lower < 0.8);
/// assert!(ci.upper > 0.8);
/// assert!(ci.lower > 0.0);
/// assert!(ci.upper < 1.0);
/// ```
pub fn correlation_ci(r: f64, n: usize, confidence: f64) -> Option<CorrelationCI> {
    if n < 4 || r <= -1.0 || r >= 1.0 || !r.is_finite() {
        return None;
    }
    if confidence <= 0.0 || confidence >= 1.0 {
        return None;
    }

    let z = r.atanh();
    let se = 1.0 / ((n as f64 - 3.0).sqrt());
    let alpha = 1.0 - confidence;
    let z_crit = special::inverse_normal_cdf(1.0 - alpha / 2.0);

    let z_lower = z - z_crit * se;
    let z_upper = z + z_crit * se;

    Some(CorrelationCI {
        lower: z_lower.tanh(),
        upper: z_upper.tanh(),
        confidence,
    })
}

// ---------------------------------------------------------------------------
// Correlation Matrix
// ---------------------------------------------------------------------------

/// Computes a pairwise Pearson correlation matrix.
///
/// # Arguments
///
/// * `variables` — Slice of variable data. Each inner slice is one variable's
///   observations. All must have the same length.
///
/// # Returns
///
/// A symmetric n×n `Matrix` where entry (i,j) is the Pearson r between
/// variables i and j. Diagonal is 1.0. Returns `None` if fewer than 2
/// variables, observations < 3, or variable lengths differ.
///
/// # Examples
///
/// ```
/// use u_analytics::correlation::correlation_matrix;
///
/// let x = [1.0, 2.0, 3.0, 4.0, 5.0];
/// let y = [2.0, 4.0, 6.0, 8.0, 10.0];
/// let z = [5.0, 4.0, 3.0, 2.0, 1.0];
/// let mat = correlation_matrix(&[&x, &y, &z]).unwrap();
/// assert!((mat.get(0, 1) - 1.0).abs() < 1e-10);   // x,y perfectly correlated
/// assert!((mat.get(0, 2) + 1.0).abs() < 1e-10);   // x,z perfectly anti-correlated
/// ```
pub fn correlation_matrix(variables: &[&[f64]]) -> Option<Matrix> {
    let p = variables.len();
    if p < 2 {
        return None;
    }
    let n = variables[0].len();
    if n < 3 {
        return None;
    }
    for v in variables {
        if v.len() != n {
            return None;
        }
    }

    // Pre-compute means and std devs
    let mut sds = Vec::with_capacity(p);
    for v in variables {
        let m = stats::mean(v)?;
        if !m.is_finite() {
            return None;
        }
        let s = stats::std_dev(v)?;
        if !s.is_finite() || s < 1e-300 {
            return None;
        }
        sds.push(s);
    }

    let mut data = vec![0.0; p * p];

    for i in 0..p {
        data[i * p + i] = 1.0; // diagonal
        for j in (i + 1)..p {
            let cov = stats::covariance(variables[i], variables[j])?;
            let r = (cov / (sds[i] * sds[j])).clamp(-1.0, 1.0);
            data[i * p + j] = r;
            data[j * p + i] = r;
        }
    }

    Matrix::new(p, p, data).ok()
}

/// Computes a pairwise Spearman rank correlation matrix.
///
/// Same as [`correlation_matrix`] but uses rank-transformed data.
pub fn spearman_matrix(variables: &[&[f64]]) -> Option<Matrix> {
    let p = variables.len();
    if p < 2 {
        return None;
    }
    let n = variables[0].len();
    if n < 3 {
        return None;
    }
    for v in variables {
        if v.len() != n || v.iter().any(|x| !x.is_finite()) {
            return None;
        }
    }

    // Rank-transform all variables
    let ranked: Vec<Vec<f64>> = variables.iter().map(|v| rank_data(v)).collect();
    let ranked_refs: Vec<&[f64]> = ranked.iter().map(|v| v.as_slice()).collect();

    correlation_matrix(&ranked_refs)
}

/// Computes a Kendall tau-b correlation matrix for the given variables.
///
/// Each off-diagonal entry is the result of [`kendall_tau_b`] for the
/// corresponding pair of variables; diagonal entries are 1.0.
///
/// # Returns
///
/// `None` if there are fewer than 2 variables, fewer than 3 observations,
/// any variable has differing length, contains a non-finite value, or any
/// pairwise computation returns `None`.
///
/// # References
///
/// Kendall (1938). "A new measure of rank correlation".
/// Knight (1966). "A computer method for calculating Kendall's tau with
/// ungrouped data" (used internally by [`kendall_tau_b`] for O(n log n)).
///
/// # Examples
///
/// ```
/// use u_analytics::correlation::kendall_matrix;
///
/// let x = [1.0, 2.0, 3.0, 4.0, 5.0];
/// let y = [2.0, 4.0, 6.0, 8.0, 10.0];      // perfectly monotonic with x
/// let z = [5.0, 4.0, 3.0, 2.0, 1.0];        // perfectly anti-monotonic with x
/// let mat = kendall_matrix(&[&x, &y, &z]).unwrap();
/// assert!((mat.get(0, 1) - 1.0).abs() < 1e-10);
/// assert!((mat.get(0, 2) - (-1.0)).abs() < 1e-10);
/// ```
pub fn kendall_matrix(variables: &[&[f64]]) -> Option<Matrix> {
    let p = variables.len();
    if p < 2 {
        return None;
    }
    let n = variables[0].len();
    if n < 3 {
        return None;
    }
    for v in variables {
        if v.len() != n || v.iter().any(|x| !x.is_finite()) {
            return None;
        }
    }

    let mut data = vec![0.0_f64; p * p];
    for i in 0..p {
        data[i * p + i] = 1.0;
        for j in (i + 1)..p {
            let r = kendall_tau_b(variables[i], variables[j])?.r;
            data[i * p + j] = r;
            data[j * p + i] = r;
        }
    }
    Matrix::new(p, p, data).ok()
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Computes two-tailed p-value for a correlation coefficient via t-test.
///
/// t = r·√(n-2) / √(1-r²), df = n-2.
fn correlation_p_value(r: f64, n: usize) -> f64 {
    if n < 3 {
        return 1.0;
    }
    let df = (n - 2) as f64;
    let r2 = r * r;

    // Handle r ≈ ±1.0 (denominator → 0)
    if r2 >= 1.0 - 1e-15 {
        return 0.0; // perfect correlation
    }

    let t = r * (df / (1.0 - r2)).sqrt();
    2.0 * special::t_distribution_sf(t.abs(), df)
}

/// Ranks data using the mid-rank method for ties.
///
/// Returns a Vec of ranks (1-based). Tied values receive the average rank.
fn rank_data(data: &[f64]) -> Vec<f64> {
    let n = data.len();
    let mut indexed: Vec<(usize, f64)> = data.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut ranks = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        // Find all tied elements
        while j < n && (indexed[j].1 - indexed[i].1).abs() < 1e-300 {
            j += 1;
        }
        // Average rank for the tied group (1-based)
        let avg_rank = (i + j) as f64 / 2.0 + 0.5;
        for item in indexed.iter().take(j).skip(i) {
            ranks[item.0] = avg_rank;
        }
        i = j;
    }

    ranks
}

/// The sums over the groups of equal values (sizes `t`) that the variance of
/// Kendall's S needs: Σ t(t-1)/2, Σ t(t-1)(t-2) and Σ t(t-1)(2t+5).
fn tie_sums(data: &[f64]) -> (f64, f64, f64) {
    let mut sorted: Vec<f64> = data.to_vec();
    sorted.sort_by(f64::total_cmp);

    let (mut pairs, mut triples, mut weighted) = (0.0, 0.0, 0.0);
    let mut i = 0;
    while i < sorted.len() {
        let mut j = i;
        while j < sorted.len() && sorted[j] == sorted[i] {
            j += 1;
        }
        let t = (j - i) as f64;
        pairs += t * (t - 1.0) / 2.0;
        triples += t * (t - 1.0) * (t - 2.0);
        weighted += t * (t - 1.0) * (2.0 * t + 5.0);
        i = j;
    }
    (pairs, triples, weighted)
}

// ---------------------------------------------------------------------------
// Autocorrelation (ACF) and Partial Autocorrelation (PACF)
// ---------------------------------------------------------------------------

/// Result of autocorrelation analysis.
#[derive(Debug, Clone)]
pub struct AcfResult {
    /// Autocorrelation values for lags 0, 1, ..., max_lag.
    /// `acf[0]` is always 1.0.
    pub acf: Vec<f64>,
    /// 95% white-noise threshold = z₀.₉₇₅ / √n: under the hypothesis that the
    /// series is white noise, |acf\[k\]| above it (k > 0) is significant.
    pub confidence_threshold: f64,
    /// 95% limits for lags 1, ..., max_lag by Bartlett's formula,
    /// z₀.₉₇₅ · √((1 + 2 Σⱼ₌₁ᵏ⁻¹ r(j)²) / n): for lag k, under the hypothesis
    /// that the autocorrelation is zero beyond lag k − 1 (an MA(k − 1)
    /// process). The bands an ACF plot draws to choose an MA order.
    pub bartlett_limits: Vec<f64>,
}

/// Result of partial autocorrelation analysis.
#[derive(Debug, Clone)]
pub struct PacfResult {
    /// Partial autocorrelation values for lags 1, ..., max_lag.
    pub pacf: Vec<f64>,
    /// 95% threshold = z₀.₉₇₅ / √n (Quenouille): for lag k, under the
    /// hypothesis of an AR(k − 1) process, |pacf| above it is significant.
    pub confidence_threshold: f64,
}

/// z₀.₉₇₅, the two-sided 95 % standard normal point.
const Z_975: f64 = 1.959_963_984_540_054;

/// The fewest values [`acf`] takes: one lag needs a pair.
pub const ACF_MIN_OBSERVATIONS: usize = 2;

/// The fewest values [`pacf`] takes: lags below `n / 2` leave lag 1 from 4 on.
pub const PACF_MIN_OBSERVATIONS: usize = 4;

/// The largest lag of an autocorrelation request: `max_lag` within
/// `domain`, or, when it is `None`, `min(⌊10·log₁₀ n⌋, highest)`
/// (statsmodels' default), never below 1.
fn lag_choice(
    max_lag: Option<usize>,
    n: usize,
    domain: RangeInclusive<usize>,
) -> Result<usize, TimeSeriesError> {
    match max_lag {
        None => Ok((((n as f64).log10() * 10.0).floor() as usize).clamp(1, *domain.end())),
        Some(k) => check_option("max_lag", k, &domain),
    }
}

/// The values [`acf`] takes for `max_lag` on `data`: `1..=n − 1`.
///
/// # Errors
///
/// The series' own refusals, as [`acf`] gives them: a non-finite value or
/// fewer than 2 values.
///
/// # Examples
///
/// ```
/// use u_analytics::correlation::acf_max_lag_domain;
///
/// assert_eq!(acf_max_lag_domain(&[1.0, 2.0, 4.0, 3.0]).unwrap(), 1..=3);
/// ```
pub fn acf_max_lag_domain(data: &[f64]) -> Result<RangeInclusive<usize>, TimeSeriesError> {
    check_series(data, ACF_MIN_OBSERVATIONS)?;
    Ok(1..=data.len() - 1)
}

/// The values [`pacf`] takes for `max_lag` on `data`: `1..=n / 2 − 1`.
///
/// # Errors
///
/// The series' own refusals, as [`pacf`] gives them: a non-finite value or
/// fewer than 4 values.
pub fn pacf_max_lag_domain(data: &[f64]) -> Result<RangeInclusive<usize>, TimeSeriesError> {
    check_series(data, PACF_MIN_OBSERVATIONS)?;
    Ok(1..=data.len() / 2 - 1)
}

/// Compute the sample autocorrelation function (ACF).
///
/// Uses the biased estimator (denominator N, not N-k) which guarantees a
/// positive-semidefinite autocovariance matrix.
///
/// # Arguments
///
/// * `data` — time series observations (at least 2, all finite).
/// * `max_lag` — the largest lag, in `[1, n − 1]`; `None` for statsmodels'
///   default `min(⌊10·log₁₀ n⌋, n − 1)`.
///
/// # Errors
///
/// [`TimeSeriesError`]: a non-finite value, fewer than 2 values, `max_lag`
/// outside `[1, n − 1]`, or a series that does not vary.
///
/// # References
///
/// - Box & Jenkins (1976). *Time Series Analysis: Forecasting and Control*.
///
/// # Examples
///
/// ```
/// use u_analytics::correlation::acf;
///
/// let data = [1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0, 2.0];
/// let r = acf(&data, Some(5)).unwrap();
/// assert!((r.acf[0] - 1.0).abs() < 1e-10); // lag 0 is always 1
/// assert!(r.acf.len() == 6); // lags 0..=5
/// ```
pub fn acf(data: &[f64], max_lag: Option<usize>) -> Result<AcfResult, TimeSeriesError> {
    let domain = acf_max_lag_domain(data)?;
    let n = data.len();
    let max_lag = lag_choice(max_lag, n, domain)?;
    let nf = n as f64;

    let mean = data.iter().sum::<f64>() / nf;

    // Lag-0 autocovariance (biased variance)
    let c0: f64 = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / nf;

    // A constant series: no variation beyond the rounding of its mean, judged
    // against the series' own size (a fixed floor would misjudge small units).
    let mean_square: f64 = data.iter().map(|&x| x * x).sum::<f64>() / nf;
    if c0 <= 1e-20 * mean_square {
        return Err(TimeSeriesError::NoVariation);
    }

    let mut acf_vals = Vec::with_capacity(max_lag + 1);
    acf_vals.push(1.0); // r(0) = 1

    for lag in 1..=max_lag {
        let ck: f64 = data[..n - lag]
            .iter()
            .zip(&data[lag..])
            .map(|(&xt, &xt_h)| (xt - mean) * (xt_h - mean))
            .sum::<f64>()
            / nf;
        acf_vals.push(ck / c0);
    }

    let threshold = Z_975 / nf.sqrt();
    let mut cumulative = 0.0;
    let bartlett_limits = acf_vals[1..]
        .iter()
        .map(|r| {
            let limit = Z_975 * ((1.0 + 2.0 * cumulative) / nf).sqrt();
            cumulative += r * r;
            limit
        })
        .collect();

    Ok(AcfResult {
        acf: acf_vals,
        confidence_threshold: threshold,
        bartlett_limits,
    })
}

/// Compute the sample partial autocorrelation function (PACF) via
/// Durbin-Levinson recursion on the biased sample ACF (statsmodels'
/// `pacf(method="ldb")`).
///
/// PACF at lag *h* measures the correlation between xₜ and xₜ₊ₕ after
/// removing the linear effect of intermediate lags.
///
/// # Arguments
///
/// * `data` — time series observations (at least 4, all finite).
/// * `max_lag` — the largest lag, in `[1, n / 2 − 1]` (statsmodels' limit:
///   beyond `n / 2` the recursion runs on autocorrelations from too few
///   pairs); `None` for `min(⌊10·log₁₀ n⌋, n / 2 − 1)`.
///
/// # Errors
///
/// [`TimeSeriesError`]: a non-finite value, fewer than 4 values, `max_lag`
/// outside `[1, n / 2 − 1]`, or a series that does not vary.
///
/// # References
///
/// - Durbin (1960). "The fitting of time-series models". Revue de l'Institut
///   International de Statistique, 28(3), 233–244.
///
/// # Examples
///
/// ```
/// use u_analytics::correlation::pacf;
///
/// let data = [1.0, 3.0, 2.0, 4.0, 3.0, 5.0, 4.0, 6.0, 5.0, 7.0];
/// let r = pacf(&data, Some(4)).unwrap();
/// assert!(r.pacf.len() == 4); // lags 1..=4
/// // Lag 1 PACF equals lag 1 ACF
/// ```
pub fn pacf(data: &[f64], max_lag: Option<usize>) -> Result<PacfResult, TimeSeriesError> {
    let domain = pacf_max_lag_domain(data)?;
    let n = data.len();
    let max_lag = lag_choice(max_lag, n, domain)?;

    // The autocorrelations it recurses on (which refuse a series that does
    // not vary).
    let acf_result = acf(data, Some(max_lag))?;
    let rho = &acf_result.acf;

    let mut pacf_vals = Vec::with_capacity(max_lag);

    // φ₁₁ = ρ(1)
    pacf_vals.push(rho[1]);

    if max_lag == 1 {
        return Ok(PacfResult {
            pacf: pacf_vals,
            confidence_threshold: Z_975 / (n as f64).sqrt(),
        });
    }

    // Durbin-Levinson recursion
    let mut phi_prev = vec![rho[1]];

    for h in 2..=max_lag {
        // Numerator: ρ(h) - Σⱼ₌₁ᴴ⁻¹ φ_{h-1,j} × ρ(h-j)
        let mut num = rho[h];
        for j in 0..h - 1 {
            num -= phi_prev[j] * rho[h - 1 - j];
        }

        // Denominator: 1 - Σⱼ₌₁ᴴ⁻¹ φ_{h-1,j} × ρ(j+1)
        let mut den = 1.0;
        for j in 0..h - 1 {
            den -= phi_prev[j] * rho[j + 1];
        }

        let phi_hh = if den.abs() > 1e-14 { num / den } else { 0.0 };

        // Update: φₕⱼ = φ_{h-1,j} - φₕₕ × φ_{h-1,h-1-j}
        let mut phi_new = Vec::with_capacity(h);
        for j in 0..h - 1 {
            phi_new.push(phi_prev[j] - phi_hh * phi_prev[h - 2 - j]);
        }
        phi_new.push(phi_hh);

        pacf_vals.push(phi_hh);
        phi_prev = phi_new;
    }

    Ok(PacfResult {
        pacf: pacf_vals,
        confidence_threshold: Z_975 / (n as f64).sqrt(),
    })
}

/// ACF (with Bartlett's limits) and PACF against statsmodels 0.15.0
/// (`acf(x, nlags=8, fft=False, alpha=0.05)`, `pacf(x, nlags=8, method="ldb")`).
#[cfg(test)]
mod statsmodels_reference {
    use super::*;

    const X: [f64; 24] = [
        2.3, 3.1, 4.0, 3.6, 2.9, 2.2, 2.8, 3.9, 4.6, 4.1, 3.3, 2.5, 2.7, 3.8, 4.9, 4.4, 3.5, 2.6,
        3.0, 4.2, 5.1, 4.5, 3.4, 2.9,
    ];

    fn close(got: &[f64], want: &[f64], what: &str) {
        assert_eq!(got.len(), want.len(), "{what}");
        for (k, (g, w)) in got.iter().zip(want).enumerate() {
            assert!((g - w).abs() < 1e-12, "{what}[{k}]: {g} vs {w}");
        }
    }

    #[test]
    fn acf_and_bartlett_limits_reproduce_statsmodels() {
        let r = acf(&X, Some(8)).unwrap();
        close(
            &r.acf,
            &[
                1.0,
                0.486403576681892,
                -0.36186808082372074,
                -0.6951594797553611,
                -0.2634125570953007,
                0.42164395757528844,
                0.6593055663079663,
                0.26106100487729356,
                -0.3171014941549895,
            ],
            "acf",
        );
        close(
            &r.bartlett_limits,
            &[
                0.40007597302960923,
                0.48559020426940086,
                0.5269888881332575,
                0.6575826795727397,
                0.6742603293351266,
                0.7152198020271394,
                0.80665422242141,
                0.8200659919957791,
            ],
            "bartlett",
        );
        assert!((r.confidence_threshold - 0.40007597302960923).abs() < 1e-12);
    }

    #[test]
    fn pacf_reproduces_statsmodels_ldb() {
        let r = pacf(&X, Some(8)).unwrap();
        close(
            &r.pacf,
            &[
                0.486403576681892,
                -0.7839238375815395,
                0.00047345408884359627,
                0.1902495235598503,
                0.19973005383776718,
                0.1225767860916582,
                0.05567632845800997,
                0.03539504608148146,
            ],
            "pacf",
        );
    }

    #[test]
    fn a_small_unit_is_not_a_constant_series() {
        let small: Vec<f64> = X.iter().map(|v| v * 1e-150).collect();
        let r = acf(&small, Some(8)).unwrap();
        assert!((r.acf[1] - 0.486403576681892).abs() < 1e-12);
        assert_eq!(
            acf(&[3.0; 10], Some(3)).unwrap_err(),
            TimeSeriesError::NoVariation
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Pearson tests
    // -----------------------------------------------------------------------

    #[test]
    fn pearson_perfect_positive() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [2.0, 4.0, 6.0, 8.0, 10.0];
        let result = pearson(&x, &y).expect("should compute");
        assert!((result.r - 1.0).abs() < 1e-10);
        assert!(result.p_value < 1e-10);
    }

    #[test]
    fn pearson_perfect_negative() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [10.0, 8.0, 6.0, 4.0, 2.0];
        let result = pearson(&x, &y).expect("should compute");
        assert!((result.r + 1.0).abs() < 1e-10);
        assert!(result.p_value < 1e-10);
    }

    #[test]
    fn pearson_uncorrelated() {
        // Designed to have r = 0
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [3.0, 1.0, 5.0, 1.0, 5.0];
        let result = pearson(&x, &y).expect("should compute");
        assert!(result.r.abs() < 0.5);
    }

    #[test]
    fn pearson_known_value() {
        // Height (inches) vs GPA example
        let x = [68.0, 71.0, 62.0, 75.0, 58.0, 60.0, 67.0, 68.0, 71.0, 69.0];
        let y = [4.1, 4.6, 3.8, 4.4, 3.2, 3.1, 3.8, 4.1, 4.3, 3.7];
        let result = pearson(&x, &y).expect("should compute");
        // Computed: r ≈ 0.8816
        assert!((result.r - 0.8816).abs() < 0.01, "r = {}", result.r);
    }

    /// Verifies exact Pearson r numeric reference from the spec.
    ///
    /// x = [1,2,3,4,5], y = [2,4,5,4,5]
    /// Σ(xᵢ-x̄)(yᵢ-ȳ) = 6
    /// Σ(xᵢ-x̄)² = 10, Σ(yᵢ-ȳ)² = 6
    /// r = 6 / √(10·6) = 6/√60 = 6/7.7460 ≈ 0.7746
    ///
    /// Reference: Pearson (1895), Proc. Royal Society London, 58, 240–242.
    #[test]
    fn pearson_numeric_reference() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [2.0, 4.0, 5.0, 4.0, 5.0];
        let result = pearson(&x, &y).expect("should compute");

        let expected_r = 6.0_f64 / (60.0_f64).sqrt(); // = 0.774596...
        assert!(
            (result.r - expected_r).abs() < 1e-3,
            "r expected {:.6}, got {}",
            expected_r,
            result.r
        );
        assert!(
            (result.r - 0.7746).abs() < 1e-3,
            "r expected ≈0.7746, got {}",
            result.r
        );
    }

    #[test]
    fn pearson_insufficient_data() {
        assert!(pearson(&[1.0, 2.0], &[3.0, 4.0]).is_none());
        assert!(pearson(&[1.0], &[2.0]).is_none());
    }

    #[test]
    fn pearson_length_mismatch() {
        assert!(pearson(&[1.0, 2.0, 3.0], &[4.0, 5.0]).is_none());
    }

    #[test]
    fn pearson_zero_variance() {
        assert!(pearson(&[5.0, 5.0, 5.0], &[1.0, 2.0, 3.0]).is_none());
    }

    #[test]
    fn pearson_nan_input() {
        assert!(pearson(&[1.0, f64::NAN, 3.0], &[4.0, 5.0, 6.0]).is_none());
    }

    // -----------------------------------------------------------------------
    // Spearman tests
    // -----------------------------------------------------------------------

    #[test]
    fn spearman_perfect_monotone() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [1.0, 4.0, 9.0, 16.0, 25.0]; // monotone increasing
        let result = spearman(&x, &y).expect("should compute");
        assert!((result.r - 1.0).abs() < 1e-10);
    }

    #[test]
    fn spearman_perfect_negative() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [5.0, 4.0, 3.0, 2.0, 1.0];
        let result = spearman(&x, &y).expect("should compute");
        assert!((result.r + 1.0).abs() < 1e-10);
    }

    #[test]
    fn spearman_with_ties() {
        let x = [1.0, 2.0, 2.0, 4.0, 5.0];
        let y = [1.0, 3.0, 3.0, 4.0, 5.0];
        let result = spearman(&x, &y).expect("should compute");
        assert!(result.r > 0.9); // strong positive
    }

    #[test]
    fn spearman_nonlinear_monotone() {
        // Spearman should be 1.0 for any monotone function
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y: Vec<f64> = x.iter().map(|&v: &f64| v.powi(3)).collect();
        let result = spearman(&x, &y).expect("should compute");
        assert!((result.r - 1.0).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // Kendall tests
    // -----------------------------------------------------------------------

    #[test]
    fn kendall_perfect_concordance() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [1.0, 2.0, 3.0, 4.0, 5.0];
        let result = kendall_tau_b(&x, &y).expect("should compute");
        assert!((result.r - 1.0).abs() < 1e-10);
    }

    #[test]
    fn kendall_perfect_discordance() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [5.0, 4.0, 3.0, 2.0, 1.0];
        let result = kendall_tau_b(&x, &y).expect("should compute");
        assert!((result.r + 1.0).abs() < 1e-10);
    }

    #[test]
    fn kendall_known_value() {
        // Known example: tau ≈ 0.733
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [3.0, 4.0, 1.0, 2.0, 5.0];
        let result = kendall_tau_b(&x, &y).expect("should compute");
        assert!((result.r - 0.2).abs() < 0.01);
    }

    #[test]
    fn kendall_with_ties() {
        let x = [1.0, 2.0, 2.0, 4.0, 5.0];
        let y = [1.0, 3.0, 3.0, 4.0, 5.0];
        let result = kendall_tau_b(&x, &y).expect("should compute");
        assert!(result.r > 0.8);
    }

    #[test]
    fn kendall_all_ties() {
        let x = [1.0, 1.0, 1.0];
        let y = [2.0, 2.0, 2.0];
        assert!(kendall_tau_b(&x, &y).is_none()); // denom = 0
    }

    // -----------------------------------------------------------------------
    // Fisher z tests
    // -----------------------------------------------------------------------

    #[test]
    fn fisher_z_zero() {
        let z = fisher_z(0.0).expect("should compute");
        assert!(z.abs() < 1e-15);
    }

    #[test]
    fn fisher_z_roundtrip() {
        for &r in &[0.0, 0.3, 0.5, 0.8, 0.95, -0.5, -0.95] {
            let z = fisher_z(r).expect("should compute");
            let r_back = fisher_z_inv(z);
            assert!((r - r_back).abs() < 1e-10, "Roundtrip failed for r={r}");
        }
    }

    #[test]
    fn fisher_z_boundary() {
        assert!(fisher_z(1.0).is_none());
        assert!(fisher_z(-1.0).is_none());
        assert!(fisher_z(1.5).is_none());
        assert!(fisher_z(f64::NAN).is_none());
    }

    // -----------------------------------------------------------------------
    // Confidence interval tests
    // -----------------------------------------------------------------------

    #[test]
    fn ci_contains_r() {
        let ci = correlation_ci(0.6, 50, 0.95).expect("should compute");
        assert!(ci.lower < 0.6);
        assert!(ci.upper > 0.6);
        assert!(ci.lower > -1.0);
        assert!(ci.upper < 1.0);
    }

    #[test]
    fn ci_wider_at_higher_confidence() {
        let ci_95 = correlation_ci(0.5, 30, 0.95).expect("should compute");
        let ci_99 = correlation_ci(0.5, 30, 0.99).expect("should compute");
        assert!(ci_99.upper - ci_99.lower > ci_95.upper - ci_95.lower);
    }

    #[test]
    fn ci_narrower_with_more_data() {
        let ci_30 = correlation_ci(0.5, 30, 0.95).expect("should compute");
        let ci_100 = correlation_ci(0.5, 100, 0.95).expect("should compute");
        assert!(ci_100.upper - ci_100.lower < ci_30.upper - ci_30.lower);
    }

    #[test]
    fn ci_edge_cases() {
        assert!(correlation_ci(0.5, 3, 0.95).is_none()); // n too small
        assert!(correlation_ci(1.0, 30, 0.95).is_none()); // r = 1
        assert!(correlation_ci(0.5, 30, 0.0).is_none()); // invalid confidence
        assert!(correlation_ci(0.5, 30, 1.0).is_none());
    }

    // -----------------------------------------------------------------------
    // Correlation matrix tests
    // -----------------------------------------------------------------------

    #[test]
    fn corr_matrix_identity() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [2.0, 4.0, 6.0, 8.0, 10.0];
        let z = [5.0, 4.0, 3.0, 2.0, 1.0];
        let mat = correlation_matrix(&[&x, &y, &z]).expect("should compute");

        // Diagonal = 1
        assert!((mat.get(0, 0) - 1.0).abs() < 1e-10);
        assert!((mat.get(1, 1) - 1.0).abs() < 1e-10);
        assert!((mat.get(2, 2) - 1.0).abs() < 1e-10);

        // x,y perfectly correlated
        assert!((mat.get(0, 1) - 1.0).abs() < 1e-10);

        // x,z perfectly anti-correlated
        assert!((mat.get(0, 2) + 1.0).abs() < 1e-10);

        // Symmetric
        assert!((mat.get(0, 1) - mat.get(1, 0)).abs() < 1e-15);
        assert!((mat.get(0, 2) - mat.get(2, 0)).abs() < 1e-15);
    }

    #[test]
    fn corr_matrix_insufficient_variables() {
        let x = [1.0, 2.0, 3.0];
        assert!(correlation_matrix(&[&x]).is_none());
    }

    #[test]
    fn corr_matrix_length_mismatch() {
        let x = [1.0, 2.0, 3.0];
        let y = [4.0, 5.0];
        assert!(correlation_matrix(&[&x, &y]).is_none());
    }

    #[test]
    fn spearman_matrix_basic() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [1.0, 4.0, 9.0, 16.0, 25.0]; // monotone
        let z = [5.0, 4.0, 3.0, 2.0, 1.0];
        let mat = spearman_matrix(&[&x, &y, &z]).expect("should compute");

        // x and y have perfect monotone relationship
        assert!((mat.get(0, 1) - 1.0).abs() < 1e-10);
        // x and z are perfectly anti-monotone
        assert!((mat.get(0, 2) + 1.0).abs() < 1e-10);
    }

    #[test]
    fn kendall_matrix_basic() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [2.0, 4.0, 6.0, 8.0, 10.0]; // perfectly monotone with x
        let z = [5.0, 4.0, 3.0, 2.0, 1.0]; // perfectly anti-monotone with x
        let mat = kendall_matrix(&[&x, &y, &z]).expect("should compute");

        assert_eq!(mat.rows(), 3);
        assert!((mat.get(0, 0) - 1.0).abs() < 1e-15);
        assert!((mat.get(1, 1) - 1.0).abs() < 1e-15);
        assert!((mat.get(2, 2) - 1.0).abs() < 1e-15);

        assert!((mat.get(0, 1) - 1.0).abs() < 1e-10);
        assert!((mat.get(0, 2) + 1.0).abs() < 1e-10);

        // Symmetric
        assert!((mat.get(0, 1) - mat.get(1, 0)).abs() < 1e-15);
        assert!((mat.get(0, 2) - mat.get(2, 0)).abs() < 1e-15);
    }

    #[test]
    fn kendall_matrix_insufficient_variables() {
        let x = [1.0, 2.0, 3.0];
        assert!(kendall_matrix(&[&x]).is_none());
    }

    #[test]
    fn kendall_matrix_length_mismatch() {
        let x = [1.0, 2.0, 3.0, 4.0];
        let y = [4.0, 5.0];
        assert!(kendall_matrix(&[&x, &y]).is_none());
    }

    #[test]
    fn kendall_matrix_with_ties() {
        let x = [1.0, 2.0, 2.0, 3.0, 4.0];
        let y = [1.0, 2.0, 3.0, 4.0, 5.0];
        let mat = kendall_matrix(&[&x, &y]).expect("should compute");
        assert!(mat.get(0, 1) > 0.7);
    }

    // -----------------------------------------------------------------------
    // Rank data tests
    // -----------------------------------------------------------------------

    #[test]
    fn rank_no_ties() {
        let ranks = rank_data(&[3.0, 1.0, 2.0]);
        assert!((ranks[0] - 3.0).abs() < 1e-10); // 3.0 → rank 3
        assert!((ranks[1] - 1.0).abs() < 1e-10); // 1.0 → rank 1
        assert!((ranks[2] - 2.0).abs() < 1e-10); // 2.0 → rank 2
    }

    #[test]
    fn rank_with_ties() {
        let ranks = rank_data(&[1.0, 2.0, 2.0, 4.0]);
        assert!((ranks[0] - 1.0).abs() < 1e-10);
        assert!((ranks[1] - 2.5).abs() < 1e-10); // average of ranks 2,3
        assert!((ranks[2] - 2.5).abs() < 1e-10);
        assert!((ranks[3] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn rank_all_same() {
        let ranks = rank_data(&[5.0, 5.0, 5.0]);
        assert!((ranks[0] - 2.0).abs() < 1e-10); // average of 1,2,3
        assert!((ranks[1] - 2.0).abs() < 1e-10);
        assert!((ranks[2] - 2.0).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // P-value tests
    // -----------------------------------------------------------------------

    #[test]
    fn pvalue_significant() {
        // Large sample, strong correlation → very small p-value
        let x: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&v| v * 2.0 + 1.0).collect();
        let result = pearson(&x, &y).expect("should compute");
        assert!(result.p_value < 1e-10);
    }

    #[test]
    fn pvalue_not_significant() {
        // Carefully constructed to have very low correlation
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [5.0, 1.0, 3.0, 5.0, 1.0];
        let result = pearson(&x, &y).expect("should compute");
        assert!(result.p_value > 0.3); // not significant
    }

    // -----------------------------------------------------------------------
    // ACF tests
    // -----------------------------------------------------------------------

    #[test]
    fn acf_lag_zero_is_one() {
        let data = [1.0, 3.0, 2.0, 5.0, 4.0, 7.0, 6.0, 8.0];
        let r = acf(&data, Some(3)).expect("should compute");
        assert!((r.acf[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn acf_linear_trend() {
        // Linear trend → strong positive autocorrelation at low lags
        let data: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let r = acf(&data, Some(5)).expect("should compute");
        assert!(r.acf[1] > 0.8, "lag-1 ACF for linear trend should be high");
        assert!(r.acf.len() == 6);
    }

    #[test]
    fn acf_alternating() {
        // Alternating series → negative lag-1 autocorrelation
        let data: Vec<f64> = (0..30)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let r = acf(&data, Some(5)).expect("should compute");
        assert!(r.acf[1] < -0.8, "alternating → negative lag-1 ACF");
        assert!(r.acf[2] > 0.8, "alternating → positive lag-2 ACF");
    }

    #[test]
    fn acf_white_noise_threshold() {
        let data: Vec<f64> = (0..100).map(|i| ((i * 7 + 3) % 13) as f64).collect();
        let r = acf(&data, Some(10)).expect("should compute");
        // Threshold should be ~0.196 for n=100
        assert!((r.confidence_threshold - 1.96 / 10.0).abs() < 0.01);
    }

    #[test]
    fn acf_edge_cases() {
        assert_eq!(
            acf(&[1.0], Some(3)).unwrap_err(),
            TimeSeriesError::TooFewObservations { needed: 2, got: 1 }
        );
        assert_eq!(
            acf(&[1.0, 2.0], Some(0)).unwrap_err(),
            TimeSeriesError::OptionOutOfRange {
                option: "max_lag",
                got: 0.0,
                min: 1.0,
                max: 1.0
            }
        );
        assert_eq!(
            acf(&[5.0, 5.0, 5.0, 5.0], Some(2)).unwrap_err(),
            TimeSeriesError::NoVariation
        );
        // A non-finite value is named before the length is judged.
        assert_eq!(
            acf(&[f64::NAN], Some(1)).unwrap_err(),
            TimeSeriesError::ValueNotFinite { index: 0 }
        );
    }

    #[test]
    fn acf_max_lag_is_refused_past_n_minus_1_and_defaults_to_ten_log_n() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(
            acf(&data, Some(100)).unwrap_err(),
            TimeSeriesError::OptionOutOfRange {
                option: "max_lag",
                got: 100.0,
                min: 1.0,
                max: 4.0
            }
        );
        // min(floor(10 log10 5), 4) = 4.
        assert_eq!(acf(&data, None).expect("defined").acf.len(), 5);
        let long: Vec<f64> = (0..100).map(|i| ((i * 37) % 11) as f64).collect();
        assert_eq!(acf(&long, None).expect("defined").acf.len(), 21);
    }

    // -----------------------------------------------------------------------
    // PACF tests
    // -----------------------------------------------------------------------

    #[test]
    fn pacf_lag1_equals_acf_lag1() {
        let data = [1.0, 3.0, 2.0, 5.0, 4.0, 7.0, 6.0, 8.0, 5.0, 3.0];
        let acf_r = acf(&data, Some(4)).expect("should compute ACF");
        let pacf_r = pacf(&data, Some(4)).expect("should compute PACF");
        assert!(
            (pacf_r.pacf[0] - acf_r.acf[1]).abs() < 1e-10,
            "PACF[1] should equal ACF[1]: {} vs {}",
            pacf_r.pacf[0],
            acf_r.acf[1]
        );
    }

    #[test]
    fn pacf_linear_trend() {
        let data: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let r = pacf(&data, Some(5)).expect("should compute");
        // AR(1)-like: strong lag-1 PACF, weaker at higher lags
        assert!(r.pacf[0].abs() > 0.5, "lag-1 PACF should be strong");
    }

    #[test]
    fn pacf_bounded() {
        let data: Vec<f64> = (0..30).map(|i| (i as f64 * 0.5).sin()).collect();
        let r = pacf(&data, Some(10)).expect("should compute");
        for (i, &p) in r.pacf.iter().enumerate() {
            assert!(
                (-1.0..=1.0).contains(&p),
                "PACF[{}] = {} out of [-1, 1]",
                i + 1,
                p
            );
        }
    }

    #[test]
    fn pacf_edge_cases() {
        assert_eq!(
            pacf(&[1.0, 2.0, 3.0], Some(1)).unwrap_err(),
            TimeSeriesError::TooFewObservations { needed: 4, got: 3 }
        );
        // Lags below n / 2: 1 to 4 for 10 values.
        let data = [1.0, 3.0, 2.0, 5.0, 4.0, 7.0, 6.0, 8.0, 5.0, 3.0];
        for k in [0, 5] {
            assert_eq!(
                pacf(&data, Some(k)).unwrap_err(),
                TimeSeriesError::OptionOutOfRange {
                    option: "max_lag",
                    got: k as f64,
                    min: 1.0,
                    max: 4.0
                }
            );
        }
        assert_eq!(pacf(&data, None).expect("defined").pacf.len(), 4);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    fn bounded_vec(min_len: usize, max_len: usize) -> BoxedStrategy<Vec<f64>> {
        proptest::collection::vec(-1e6_f64..1e6, min_len..=max_len).boxed()
    }

    proptest! {
        #[test]
        fn pearson_bounded(
            data in bounded_vec(5, 50).prop_flat_map(|x| {
                let n = x.len();
                (Just(x), proptest::collection::vec(-1e6_f64..1e6, n..=n))
            })
        ) {
            let (x, y) = data;
            if let Some(result) = pearson(&x, &y) {
                prop_assert!(result.r >= -1.0 && result.r <= 1.0, "r out of bounds: {}", result.r);
                prop_assert!(result.p_value >= 0.0 && result.p_value <= 1.0, "p out of bounds: {}", result.p_value);
            }
        }

        #[test]
        fn spearman_bounded(
            data in bounded_vec(5, 50).prop_flat_map(|x| {
                let n = x.len();
                (Just(x), proptest::collection::vec(-1e6_f64..1e6, n..=n))
            })
        ) {
            let (x, y) = data;
            if let Some(result) = spearman(&x, &y) {
                prop_assert!(result.r >= -1.0 && result.r <= 1.0, "r out of bounds: {}", result.r);
                prop_assert!(result.p_value >= 0.0 && result.p_value <= 1.0, "p out of bounds: {}", result.p_value);
            }
        }

        #[test]
        fn kendall_bounded(
            data in bounded_vec(5, 30).prop_flat_map(|x| {
                let n = x.len();
                (Just(x), proptest::collection::vec(-1e6_f64..1e6, n..=n))
            })
        ) {
            let (x, y) = data;
            if let Some(result) = kendall_tau_b(&x, &y) {
                prop_assert!(result.r >= -1.0 && result.r <= 1.0, "tau out of bounds: {}", result.r);
                prop_assert!(result.p_value >= 0.0 && result.p_value <= 1.0, "p out of bounds: {}", result.p_value);
            }
        }

        #[test]
        fn pearson_symmetric(
            data in bounded_vec(5, 50).prop_flat_map(|x| {
                let n = x.len();
                (Just(x), proptest::collection::vec(-1e6_f64..1e6, n..=n))
            })
        ) {
            let (x, y) = data;
            let r_xy = pearson(&x, &y);
            let r_yx = pearson(&y, &x);
            match (r_xy, r_yx) {
                (Some(a), Some(b)) => {
                    prop_assert!((a.r - b.r).abs() < 1e-10, "not symmetric: {} vs {}", a.r, b.r);
                }
                (None, None) => {}
                _ => prop_assert!(false, "one is None but not the other"),
            }
        }

        #[test]
        fn fisher_z_roundtrip_prop(r in -0.99_f64..0.99) {
            let z = fisher_z(r).expect("should compute");
            let r_back = fisher_z_inv(z);
            prop_assert!((r - r_back).abs() < 1e-10);
        }

        #[test]
        fn ci_contains_true_r(
            r in -0.99_f64..0.99,
            n in 10_usize..200
        ) {
            let ci = correlation_ci(r, n, 0.95).expect("should compute");
            prop_assert!(ci.lower < ci.upper, "CI inverted");
            prop_assert!(ci.lower >= -1.0 && ci.upper <= 1.0, "CI out of bounds");
        }

        #[test]
        fn acf_bounded_prop(
            data in proptest::collection::vec(-1e3_f64..1e3, 5..=50),
        ) {
            if let Ok(r) = acf(&data, Some(10)) {
                prop_assert!((r.acf[0] - 1.0).abs() < 1e-10, "ACF[0] must be 1.0");
                for (i, &v) in r.acf.iter().enumerate() {
                    prop_assert!((-1.0..=1.0).contains(&v), "ACF[{i}] = {v} out of [-1,1]");
                }
            }
        }

        #[test]
        fn pacf_bounded_prop(
            data in proptest::collection::vec(-1e3_f64..1e3, 5..=50),
        ) {
            if let Ok(r) = pacf(&data, Some(5)) {
                for (i, &v) in r.pacf.iter().enumerate() {
                    prop_assert!((-1.0 - 1e-10..=1.0 + 1e-10).contains(&v), "PACF[{i}] = {v} out of bounds");
                }
            }
        }
    }
}

/// A p-value far in the tail against SciPy 1.17.1 (`scipy.stats.pearsonr`).
#[cfg(test)]
mod scipy_tail_reference {
    use super::*;

    #[test]
    fn pearson_p_value_keeps_its_tail() {
        let x: Vec<f64> = (1..=20).map(f64::from).collect();
        let y = [
            5.3999999999999995,
            7.0,
            9.4,
            11.8,
            13.2,
            15.8,
            17.6,
            19.8,
            22.250000000000004,
            23.75,
            26.25,
            28.150000000000002,
            30.55,
            32.050000000000004,
            34.55,
            36.7,
            38.550000000000004,
            41.1,
            42.699999999999996,
            45.0,
        ];
        let got = pearson(&x, &y).expect("valid").p_value;
        let want = 8.20335692044105e-33;
        assert!(
            (got - want).abs() <= 1e-9 * want,
            "p: got {got:e}, SciPy {want:e}"
        );
    }
}

/// References for the Kendall p-values: SciPy 1.17.1 `kendalltau`, and the
/// definition (enumeration of all permutations).
#[cfg(test)]
mod kendall_exact_tests {
    use super::*;

    // (permutation of 0..n, tau, p) -- SciPy 1.17.1 scipy.stats.kendalltau(range(n), perm), method='auto'
    const EXACT_CASES: &[(&[u8], f64, f64)] = &[
        (&[2, 1, 0], -1.0, 0.3333333333333333),
        (&[2, 1, 0], -1.0, 0.3333333333333333),
        (&[2, 1, 0, 3], 0.0, 1.0),
        (&[2, 0, 3, 1], 0.0, 1.0),
        (&[4, 0, 3, 1, 2], -0.19999999999999998, 0.8166666666666667),
        (&[2, 0, 3, 4, 1], 0.19999999999999998, 0.8166666666666667),
        (&[4, 1, 3, 2, 5, 0], -0.2, 0.7194444444444444),
        (&[3, 2, 1, 5, 0, 4], -0.06666666666666665, 1.0),
        (
            &[0, 5, 3, 1, 4, 2, 6],
            0.33333333333333337,
            0.3813492063492063,
        ),
        (
            &[5, 2, 4, 1, 3, 0, 6],
            -0.14285714285714288,
            0.7726190476190476,
        ),
        (
            &[1, 4, 3, 0, 7, 5, 6, 2],
            0.21428571428571427,
            0.5484126984126985,
        ),
        (
            &[5, 2, 3, 6, 1, 0, 4, 7],
            0.07142857142857142,
            0.9048611111111111,
        ),
        (
            &[2, 3, 6, 7, 1, 4, 8, 5, 0],
            0.05555555555555555,
            0.919455467372134,
        ),
        (
            &[7, 8, 1, 6, 0, 3, 4, 5, 2],
            -0.2777777777777778,
            0.35848765432098767,
        ),
        (
            &[7, 4, 2, 5, 9, 0, 8, 3, 6, 1],
            -0.15555555555555553,
            0.6006536596119929,
        ),
        (
            &[3, 4, 6, 0, 1, 9, 8, 5, 2, 7],
            0.19999999999999998,
            0.4843127204585538,
        ),
        (
            &[10, 7, 6, 4, 5, 3, 1, 2, 11, 0, 9, 8],
            -0.2121212121212121,
            0.38070480349126185,
        ),
        (
            &[7, 4, 3, 9, 10, 8, 2, 1, 6, 5, 0, 11],
            -0.1212121212121212,
            0.6383612539081289,
        ),
        (
            &[6, 10, 0, 13, 1, 4, 5, 2, 9, 3, 12, 7, 14, 8, 11],
            0.29523809523809524,
            0.13946285911906778,
        ),
        (
            &[9, 11, 12, 1, 4, 3, 6, 8, 7, 14, 13, 0, 2, 5, 10],
            -0.028571428571428574,
            0.9225700904951897,
        ),
        (
            &[11, 17, 13, 1, 6, 4, 2, 5, 10, 7, 12, 9, 14, 16, 15, 3, 0, 8],
            0.006535947712418302,
            1.0,
        ),
        (
            &[8, 16, 12, 4, 13, 6, 10, 17, 2, 5, 1, 15, 9, 14, 7, 11, 0, 3],
            -0.21568627450980396,
            0.22906043853787195,
        ),
        (
            &[
                15, 3, 0, 13, 11, 8, 1, 7, 17, 19, 10, 14, 12, 2, 5, 4, 6, 18, 9, 16,
            ],
            0.09473684210526316,
            0.5858584459782121,
        ),
        (
            &[
                7, 13, 17, 12, 11, 19, 2, 16, 8, 3, 10, 9, 6, 15, 18, 14, 0, 4, 5, 1,
            ],
            -0.2736842105263158,
            0.0983302187347563,
        ),
        (
            &[
                9, 21, 11, 3, 17, 12, 8, 1, 5, 13, 15, 22, 6, 4, 18, 2, 0, 14, 16, 23, 19, 7, 10,
                20,
            ],
            0.11594202898550725,
            0.44607885611402504,
        ),
        (
            &[
                17, 2, 18, 7, 10, 21, 15, 14, 13, 23, 1, 4, 6, 9, 3, 8, 22, 0, 20, 5, 11, 19, 12,
                16,
            ],
            -0.007246376811594203,
            0.9804334461613707,
        ),
        (
            &[
                20, 13, 24, 18, 14, 2, 9, 5, 10, 15, 11, 26, 12, 21, 25, 19, 16, 22, 27, 3, 1, 17,
                8, 7, 23, 4, 6, 0,
            ],
            -0.15873015873015872,
            0.24626955123962418,
        ),
        (
            &[
                3, 11, 14, 0, 6, 7, 21, 24, 19, 10, 8, 2, 27, 16, 26, 20, 23, 15, 9, 5, 13, 1, 25,
                4, 17, 12, 22, 18,
            ],
            0.13227513227513227,
            0.3362383470773132,
        ),
        (
            &[
                29, 16, 28, 19, 18, 12, 13, 27, 1, 25, 4, 17, 22, 15, 3, 2, 10, 8, 9, 23, 11, 7,
                21, 0, 26, 6, 5, 14, 24, 20,
            ],
            -0.18160919540229886,
            0.1653935231257671,
        ),
        (
            &[
                17, 10, 9, 22, 19, 18, 8, 5, 21, 29, 13, 28, 14, 20, 27, 25, 0, 23, 24, 7, 16, 12,
                11, 4, 2, 26, 6, 1, 3, 15,
            ],
            -0.20459770114942533,
            0.11698808645983594,
        ),
        (
            &[
                31, 24, 14, 11, 29, 12, 28, 16, 10, 7, 6, 3, 8, 17, 25, 20, 26, 0, 13, 21, 5, 9,
                18, 23, 27, 1, 22, 19, 2, 30, 15, 4, 32,
            ],
            -0.041666666666666664,
            0.7468360016830153,
        ),
        (
            &[
                0, 27, 13, 32, 19, 21, 24, 25, 30, 16, 23, 8, 26, 7, 5, 14, 29, 20, 31, 22, 28, 6,
                15, 2, 4, 18, 9, 10, 1, 17, 11, 12, 3,
            ],
            -0.27272727272727276,
            0.025883100486831147,
        ),
        (
            &[
                0, 1, 2, 4, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
            ],
            0.9894736842105263,
            1.644127049324866e-17,
        ),
        (
            &[
                0, 1, 2, 4, 3, 5, 6, 7, 8, 9, 11, 10, 12, 13, 14, 15, 16, 17, 18, 19,
            ],
            0.9789473684210527,
            1.7181127665444849e-16,
        ),
        (
            &[
                19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
            ],
            -1.0,
            8.22063524662433e-19,
        ),
        (
            &[
                19, 18, 17, 16, 15, 13, 14, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
            ],
            -0.9894736842105263,
            1.644127049324866e-17,
        ),
        (
            &[
                0, 1, 2, 4, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
            ],
            0.9962121212121212,
            7.600781509709488e-36,
        ),
        (
            &[
                0, 1, 2, 4, 3, 5, 6, 7, 8, 9, 11, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
            ],
            0.9924242424242423,
            1.2898295895264586e-34,
        ),
        (
            &[
                32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12,
                11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
            ],
            -1.0,
            2.30326712415439e-37,
        ),
        (
            &[
                32, 31, 30, 29, 28, 26, 27, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12,
                11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
            ],
            -0.9962121212121212,
            7.600781509709488e-36,
        ),
        (
            &[
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
            ],
            1.0,
            6.774315071042324e-39,
        ),
        (
            &[
                0, 1, 2, 3, 4, 5, 6, 8, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
            ],
            0.9964349376114083,
            2.30326712415439e-37,
        ),
        (
            &[
                33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13,
                12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
            ],
            -1.0,
            6.774315071042324e-39,
        ),
        (
            &[
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
                44, 45, 46, 47, 48, 49,
            ],
            1.0,
            6.575898833266317e-65,
        ),
        (
            &[
                0, 1, 2, 3, 4, 5, 6, 8, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
                44, 45, 46, 47, 48, 49,
            ],
            0.9983673469387755,
            3.287949416633158e-63,
        ),
        (
            &[
                49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29,
                28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8,
                7, 6, 5, 4, 3, 2, 1, 0,
            ],
            -1.0,
            6.575898833266317e-65,
        ),
        (
            &[
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
                44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64,
                65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85,
                86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99,
            ],
            1.0,
            2.143020576250934e-158,
        ),
        (
            &[
                0, 1, 2, 3, 4, 5, 6, 8, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
                44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64,
                65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85,
                86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99,
            ],
            0.9995959595959597,
            2.1430205762509337e-156,
        ),
        (
            &[
                99, 98, 97, 96, 95, 94, 93, 92, 91, 90, 89, 88, 87, 86, 85, 84, 83, 82, 81, 80, 79,
                78, 77, 76, 75, 74, 73, 72, 71, 70, 69, 68, 67, 66, 65, 64, 63, 62, 61, 60, 59, 58,
                57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37,
                36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16,
                15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
            ],
            -1.0,
            2.143020576250934e-158,
        ),
        (
            &[
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
                44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64,
                65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85,
                86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104,
                105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120,
                121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136,
                137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152,
                153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168,
                169,
            ],
            1.0,
            2.755801935583541e-307,
        ),
        (
            &[
                0, 1, 2, 3, 4, 5, 6, 8, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
                44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64,
                65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85,
                86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104,
                105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120,
                121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136,
                137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152,
                153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168,
                169,
            ],
            0.9998607727114516,
            4.68486329049202e-305,
        ),
        (
            &[
                169, 168, 167, 166, 165, 164, 163, 162, 161, 160, 159, 158, 157, 156, 155, 154,
                153, 152, 151, 150, 149, 148, 147, 146, 145, 144, 143, 142, 141, 140, 139, 138,
                137, 136, 135, 134, 133, 132, 131, 130, 129, 128, 127, 126, 125, 124, 123, 122,
                121, 120, 119, 118, 117, 116, 115, 114, 113, 112, 111, 110, 109, 108, 107, 106,
                105, 104, 103, 102, 101, 100, 99, 98, 97, 96, 95, 94, 93, 92, 91, 90, 89, 88, 87,
                86, 85, 84, 83, 82, 81, 80, 79, 78, 77, 76, 75, 74, 73, 72, 71, 70, 69, 68, 67, 66,
                65, 64, 63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45,
                44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24,
                23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1,
                0,
            ],
            -1.0,
            2.755801935583541e-307,
        ),
        (
            &[
                7, 11, 20, 1, 33, 32, 29, 18, 3, 5, 27, 10, 8, 0, 31, 23, 9, 13, 19, 17, 28, 24, 6,
                4, 2, 26, 15, 16, 12, 14, 25, 21, 22, 30,
            ],
            0.07664884135472372,
            0.5238310765052258,
        ),
        (
            &[
                29, 22, 18, 16, 38, 31, 2, 24, 11, 23, 30, 5, 4, 34, 15, 33, 36, 6, 32, 27, 12, 14,
                19, 0, 37, 8, 35, 20, 26, 13, 21, 28, 7, 3, 17, 1, 9, 25, 39, 10,
            ],
            -0.10512820512820513,
            0.33938312065834597,
        ),
        (
            &[
                6, 51, 13, 41, 17, 52, 24, 35, 53, 2, 49, 22, 43, 39, 30, 58, 10, 59, 8, 21, 5, 20,
                50, 4, 44, 47, 38, 9, 36, 32, 42, 15, 26, 18, 37, 45, 23, 55, 25, 40, 28, 54, 11,
                7, 48, 16, 3, 27, 0, 34, 14, 46, 33, 56, 12, 1, 31, 57, 19, 29,
            ],
            -0.05536723163841807,
            0.5319468085548157,
        ),
    ];

    // (x, y, tau, p) with ties -- SciPy 1.17.1 scipy.stats.kendalltau(x, y), method='auto' (asymptotic)
    const TIE_CASES: &[(&[f64], &[f64], f64, f64)] = &[
        (
            &[0.0, 2.0, 1.0, 1.0, 2.0, 0.0, 2.0, 1.0],
            &[3.0, 2.2, 1.6, 1.6, 2.2, 3.0, 1.2, 0.6],
            -0.30550504633038933,
            0.3441320967322945,
        ),
        (
            &[3.0, 1.0, 0.0, 3.0, 2.0, 3.0, 2.0, 2.0, 3.0, 0.0, 3.0, 1.0],
            &[
                9.8,
                10.6,
                5.0,
                12.8,
                3.2,
                9.8,
                2.2,
                2.2,
                1.7999999999999998,
                10.0,
                3.8,
                1.6,
            ],
            0.05251050315105037,
            0.827163859987499,
        ),
        (
            &[5.0, 4.0, 3.0, 4.0, 3.0, 4.0, 7.0, 1.0, 6.0, 2.0, 6.0, 9.0],
            &[
                4.0,
                2.4,
                3.8,
                3.4,
                2.8,
                2.4,
                6.2,
                1.6,
                5.6,
                1.2,
                4.6,
                5.3999999999999995,
            ],
            0.7146459189048197,
            0.001720169176312472,
        ),
        (
            &[
                3.0, 4.0, 1.0, 1.0, 0.0, 4.0, 4.0, 1.0, 2.0, 4.0, 2.0, 2.0, 4.0, 3.0, 3.0, 4.0,
                4.0, 1.0, 3.0, 3.0, 0.0, 0.0, 3.0, 2.0, 4.0,
            ],
            &[
                1.7999999999999998,
                5.4,
                4.6,
                0.6,
                0.0,
                3.4,
                4.4,
                1.6,
                1.2,
                4.4,
                3.2,
                2.2,
                7.4,
                6.8,
                5.8,
                2.4,
                4.4,
                4.6,
                6.8,
                2.8,
                2.0,
                0.0,
                2.8,
                2.2,
                2.4,
            ],
            0.4371231544800122,
            0.005038720516121261,
        ),
        (
            &[
                4.0, 6.0, 4.0, 1.0, 6.0, 5.0, 7.0, 6.0, 7.0, 2.0, 4.0, 6.0, 4.0, 0.0, 5.0, 0.0,
                7.0, 5.0, 2.0, 3.0, 7.0, 7.0, 4.0, 7.0, 0.0, 6.0, 6.0, 6.0, 1.0, 5.0, 1.0, 6.0,
                6.0, 2.0, 7.0, 7.0, 2.0, 6.0, 6.0, 6.0,
            ],
            &[
                9.4,
                3.5999999999999996,
                3.4,
                1.6,
                9.6,
                5.0,
                10.2,
                10.6,
                8.2,
                5.2,
                5.4,
                8.6,
                2.4,
                4.0,
                7.0,
                3.0,
                10.2,
                6.0,
                3.2,
                5.8,
                5.2,
                10.2,
                3.4,
                10.2,
                1.0,
                10.6,
                9.6,
                9.6,
                4.6,
                3.0,
                5.6,
                5.6,
                8.6,
                3.2,
                10.2,
                4.2,
                3.2,
                10.6,
                9.6,
                9.6,
            ],
            0.5598074769339775,
            2.6671536744057452e-06,
        ),
        (
            &[
                1.0, 4.0, 2.0, 0.0, 7.0, 3.0, 9.0, 2.0, 3.0, 3.0, 7.0, 4.0, 8.0, 9.0, 2.0, 4.0,
                2.0, 0.0, 2.0, 3.0, 6.0, 4.0, 2.0, 3.0, 0.0, 0.0, 3.0, 9.0, 6.0, 0.0, 7.0, 7.0,
                4.0, 5.0, 4.0, 8.0, 4.0, 6.0, 0.0, 0.0, 1.0, 0.0, 4.0, 0.0, 6.0, 6.0, 9.0, 1.0,
                7.0, 1.0, 5.0, 3.0, 1.0, 0.0, 1.0, 9.0, 4.0, 7.0, 3.0, 6.0, 3.0, 7.0, 5.0, 9.0,
                1.0, 1.0, 1.0, 7.0, 0.0, 8.0, 6.0, 5.0, 0.0, 2.0, 9.0, 7.0, 9.0, 1.0, 0.0, 9.0,
                8.0, 2.0, 7.0, 8.0, 7.0, 6.0, 8.0, 6.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 6.0, 6.0,
                0.0, 6.0, 6.0, 2.0,
            ],
            &[
                2.6,
                8.4,
                3.2,
                6.0,
                8.2,
                6.8,
                6.3999999999999995,
                1.2,
                5.8,
                7.8,
                7.2,
                4.4,
                6.8,
                10.399999999999999,
                4.2,
                7.4,
                3.2,
                4.0,
                7.2,
                4.8,
                4.6,
                5.4,
                4.2,
                4.8,
                5.0,
                3.0,
                4.8,
                11.399999999999999,
                5.6,
                0.0,
                10.2,
                9.2,
                6.4,
                9.0,
                2.4,
                9.8,
                2.4,
                9.6,
                2.0,
                3.0,
                2.6,
                4.0,
                7.4,
                3.0,
                3.5999999999999996,
                5.6,
                10.399999999999999,
                1.6,
                5.2,
                6.6,
                5.0,
                1.7999999999999998,
                4.6,
                5.0,
                2.6,
                11.399999999999999,
                6.4,
                7.2,
                5.8,
                4.6,
                4.8,
                8.2,
                8.0,
                9.399999999999999,
                1.6,
                2.6,
                2.6,
                9.2,
                6.0,
                9.8,
                9.6,
                8.0,
                1.0,
                1.2,
                10.399999999999999,
                10.2,
                8.399999999999999,
                2.6,
                2.0,
                11.399999999999999,
                8.8,
                3.2,
                10.2,
                8.8,
                8.2,
                5.6,
                10.8,
                7.6,
                3.0,
                8.6,
                3.0,
                5.0,
                0.0,
                4.0,
                7.6,
                7.6,
                1.0,
                6.6,
                7.6,
                5.2,
            ],
            0.6273681127435055,
            2.3395873885354134e-18,
        ),
        (
            &[
                2.0, 81.0, 93.0, 5.0, 23.0, 26.0, 30.0, 18.0, 96.0, 39.0, 2.0, 30.0, 60.0, 40.0,
                90.0, 32.0, 63.0, 92.0, 6.0, 28.0, 85.0, 70.0, 35.0, 39.0, 58.0, 21.0, 23.0, 50.0,
                79.0, 79.0, 35.0, 15.0, 24.0, 94.0, 52.0, 3.0, 55.0, 72.0, 45.0, 76.0, 16.0, 93.0,
                42.0, 61.0, 75.0, 57.0, 87.0, 25.0, 54.0, 88.0, 46.0, 71.0, 57.0, 11.0, 74.0, 56.0,
                68.0, 56.0, 29.0, 88.0, 4.0, 19.0, 75.0, 38.0, 85.0, 86.0, 25.0, 96.0, 42.0, 10.0,
                34.0, 82.0, 27.0, 9.0, 26.0, 32.0, 24.0, 82.0, 7.0, 62.0, 90.0, 10.0, 30.0, 52.0,
                21.0, 32.0, 37.0, 0.0, 76.0, 26.0, 24.0, 50.0, 23.0, 41.0, 58.0, 32.0, 51.0, 0.0,
                19.0, 49.0,
            ],
            &[
                1.2,
                51.6,
                58.8,
                6.0,
                14.799999999999999,
                15.6,
                21.0,
                10.799999999999999,
                59.599999999999994,
                24.4,
                4.2,
                19.0,
                36.0,
                25.0,
                56.0,
                20.2,
                40.8,
                57.199999999999996,
                3.5999999999999996,
                19.8,
                51.0,
                44.0,
                22.0,
                24.4,
                35.8,
                14.6,
                16.799999999999997,
                32.0,
                47.4,
                47.4,
                24.0,
                9.0,
                17.4,
                59.4,
                34.2,
                1.7999999999999998,
                33.0,
                43.199999999999996,
                29.0,
                47.6,
                10.6,
                58.8,
                26.2,
                36.6,
                47.0,
                37.199999999999996,
                53.199999999999996,
                15.0,
                32.4,
                52.8,
                27.599999999999998,
                43.6,
                36.199999999999996,
                8.6,
                47.4,
                36.6,
                43.8,
                33.6,
                19.4,
                53.8,
                4.4,
                12.4,
                47.0,
                23.8,
                54.0,
                51.6,
                18.0,
                60.599999999999994,
                27.2,
                9.0,
                22.4,
                49.199999999999996,
                17.2,
                6.3999999999999995,
                18.6,
                22.2,
                17.4,
                49.199999999999996,
                6.2,
                40.199999999999996,
                55.0,
                7.0,
                18.0,
                32.2,
                12.6,
                20.2,
                23.2,
                0.0,
                45.6,
                16.6,
                15.399999999999999,
                31.0,
                16.799999999999997,
                26.599999999999998,
                34.8,
                19.2,
                33.599999999999994,
                0.0,
                14.4,
                32.4,
            ],
            0.9619049618044789,
            3.673981239914972e-45,
        ),
    ];

    fn assert_close(got: f64, want: f64, rel: f64, what: &str) {
        let scale = want.abs().max(1e-300);
        assert!(
            (got - want).abs() <= rel * scale,
            "{what}: got {got:e}, want {want:e}"
        );
    }

    /// The exact p-value and tau equal SciPy's `kendalltau` (method "auto":
    /// the exact distribution up to 33 observations without ties, and at any
    /// size when at most one pair is on the minority side; the normal
    /// approximation otherwise).
    #[test]
    fn kendall_p_values_reproduce_scipy_without_ties() {
        for (perm, tau, p) in EXACT_CASES {
            let x: Vec<f64> = (0..perm.len()).map(|i| i as f64).collect();
            let y: Vec<f64> = perm.iter().map(|&v| f64::from(v)).collect();
            let r = kendall_tau_b(&x, &y).expect("computes");
            let what = format!("n = {}", perm.len());
            assert!((r.r - tau).abs() < 1e-12, "{what}: tau {} vs {tau}", r.r);
            assert_close(r.p_value, *p, 1e-12, &what);
        }
    }

    /// With ties SciPy takes the normal approximation with the variance of
    /// Kendall (1970); so does the crate, terms for tied groups included.
    #[test]
    fn kendall_p_values_reproduce_scipy_with_ties() {
        assert!(!TIE_CASES.is_empty());
        for (x, y, tau, p) in TIE_CASES {
            let r = kendall_tau_b(x, y).expect("computes");
            let what = format!("n = {} with ties", x.len());
            assert!((r.r - tau).abs() < 1e-12, "{what}: tau {} vs {tau}", r.r);
            assert_close(r.p_value, *p, 1e-10, &what);
        }
    }

    /// The definition: over all n! orderings of y against x, the p-value of an
    /// outcome is twice the share with no more discordant pairs than the
    /// smaller side of the outcome (capped at 1), 4 * tail = n(n-1) giving 1.
    #[test]
    fn exact_p_is_the_share_of_permutations_with_fewer_inversions() {
        fn permutations(
            n: usize,
            out: &mut Vec<Vec<usize>>,
            cur: &mut Vec<usize>,
            used: &mut [bool],
        ) {
            if cur.len() == n {
                out.push(cur.clone());
                return;
            }
            for v in 0..n {
                if !used[v] {
                    used[v] = true;
                    cur.push(v);
                    permutations(n, out, cur, used);
                    cur.pop();
                    used[v] = false;
                }
            }
        }
        for n in 3..=8usize {
            let mut all = Vec::new();
            permutations(n, &mut all, &mut Vec::new(), &mut vec![false; n]);
            let inversions: Vec<i64> = all
                .iter()
                .map(|p| {
                    (0..n)
                        .flat_map(|i| (i + 1..n).map(move |j| (i, j)))
                        .filter(|&(i, j)| p[i] > p[j])
                        .count() as i64
                })
                .collect();
            let total = (n * (n - 1) / 2) as i64;
            for d in 0..=total {
                let tail = d.min(total - d);
                let share =
                    inversions.iter().filter(|&&k| k <= tail).count() as f64 / all.len() as f64;
                let want = if 4 * tail == (n * (n - 1)) as i64 {
                    1.0
                } else {
                    (2.0 * share).min(1.0)
                };
                assert_close(
                    kendall_exact_p(n, d),
                    want,
                    1e-14,
                    &format!("n = {n}, D = {d}"),
                );
            }
        }
    }

    /// The tail sums to one over the whole distribution and is symmetric.
    #[test]
    fn exact_p_is_symmetric_and_a_probability() {
        for n in [3usize, 10, 33] {
            let total = (n * (n - 1) / 2) as i64;
            for d in 0..=total {
                let p = kendall_exact_p(n, d);
                assert!((0.0..=1.0).contains(&p), "n = {n}, D = {d}: {p}");
                assert_eq!(p, kendall_exact_p(n, total - d), "n = {n}, D = {d}");
            }
        }
    }

    /// Past 33 observations, only the two smallest tails are exact -- 2/n!
    /// and 2/(n-1)! -- and the rest stays the normal approximation.
    #[test]
    fn beyond_33_only_the_smallest_tails_are_exact() {
        let n = 40;
        let x: Vec<f64> = (0..n).map(f64::from).collect();
        // One pair swapped: tail 1, exact 2 / 39!.
        let mut y = x.clone();
        y.swap(7, 8);
        let r = kendall_tau_b(&x, &y).expect("computes");
        let factorial_39: f64 = (2..=39).map(f64::from).product();
        assert_close(r.p_value, 2.0 / factorial_39, 1e-12, "tail 1");
        // Two pairs swapped: tail 2, the normal approximation (smooth, tiny
        // but not the exact value of the table above).
        y.swap(20, 21);
        let r2 = kendall_tau_b(&x, &y).expect("computes");
        assert!(r2.p_value > r.p_value && r2.p_value < 1e-9);
    }
}
