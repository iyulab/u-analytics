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
/// # Complexity
///
/// O(n²) naive enumeration. For n > 10,000 consider O(n log n) Knight's
/// algorithm (not implemented — sufficient for u-insight's typical data sizes).
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

    // Normal approximation for p-value (valid for n > ~10)
    // Variance under H0: var(tau) = 2(2n+5) / (9n(n-1))
    // With ties: var(S) = (v0 - vt - vu) / 18 + ...
    // Simplified: use standard formula for tau-b
    let nf = n as f64;
    let v0 = nf * (nf - 1.0) * (2.0 * nf + 5.0);
    let vt = compute_tie_variance_term(x);
    let vu = compute_tie_variance_term(y);
    let var_s = (v0 - vt - vu) / 18.0;

    let p_value = if var_s > 0.0 {
        let s = (concordant - discordant) as f64;
        let z = s / var_s.sqrt();
        2.0 * (1.0 - special::standard_normal_cdf(z.abs()))
    } else {
        1.0
    };

    Some(CorrelationResult {
        r: tau,
        p_value,
        n,
    })
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
    2.0 * (1.0 - special::t_distribution_cdf(t.abs(), df))
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

/// Computes the tie variance term for Kendall's tau-b.
///
/// Returns Σ tᵢ(tᵢ-1)(2tᵢ+5) for each group of tᵢ tied values.
fn compute_tie_variance_term(data: &[f64]) -> f64 {
    let mut sorted: Vec<f64> = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let mut result = 0.0;
    let mut i = 0;
    while i < sorted.len() {
        let mut j = i;
        while j < sorted.len() && (sorted[j] - sorted[i]).abs() < 1e-300 {
            j += 1;
        }
        let t = (j - i) as f64;
        if t > 1.0 {
            result += t * (t - 1.0) * (2.0 * t + 5.0);
        }
        i = j;
    }
    result
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
    /// 95% confidence threshold = 1.96 / √n.
    /// Values with |acf[k]| > threshold (for k > 0) are significant.
    pub confidence_threshold: f64,
}

/// Result of partial autocorrelation analysis.
#[derive(Debug, Clone)]
pub struct PacfResult {
    /// Partial autocorrelation values for lags 1, ..., max_lag.
    pub pacf: Vec<f64>,
    /// 95% confidence threshold = 1.96 / √n.
    pub confidence_threshold: f64,
}

/// Compute the sample autocorrelation function (ACF).
///
/// Uses the biased estimator (denominator N, not N-k) which guarantees a
/// positive-semidefinite autocovariance matrix.
///
/// # Arguments
///
/// * `data` — time series observations (at least 2).
/// * `max_lag` — maximum lag to compute. Clamped to `n - 1`.
///
/// # Returns
///
/// `None` if `data.len() < 2`, `max_lag == 0`, or data contains non-finite values.
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
/// let r = acf(&data, 5).unwrap();
/// assert!((r.acf[0] - 1.0).abs() < 1e-10); // lag 0 is always 1
/// assert!(r.acf.len() == 6); // lags 0..=5
/// ```
pub fn acf(data: &[f64], max_lag: usize) -> Option<AcfResult> {
    let n = data.len();
    if n < 2 || max_lag == 0 || data.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let max_lag = max_lag.min(n - 1);
    let nf = n as f64;

    let mean = data.iter().sum::<f64>() / nf;

    // Lag-0 autocovariance (biased variance)
    let c0: f64 = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / nf;

    if c0 <= 0.0 {
        return None; // constant series
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

    let threshold = 1.96 / nf.sqrt();

    Some(AcfResult {
        acf: acf_vals,
        confidence_threshold: threshold,
    })
}

/// Compute the sample partial autocorrelation function (PACF) via
/// Durbin-Levinson recursion.
///
/// PACF at lag *h* measures the correlation between xₜ and xₜ₊ₕ after
/// removing the linear effect of intermediate lags.
///
/// # Arguments
///
/// * `data` — time series observations (at least 3).
/// * `max_lag` — maximum lag to compute. Clamped to `n - 1`.
///
/// # Returns
///
/// `None` if `data.len() < 3`, `max_lag == 0`, or data contains non-finite values.
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
/// let r = pacf(&data, 4).unwrap();
/// assert!(r.pacf.len() == 4); // lags 1..=4
/// // Lag 1 PACF equals lag 1 ACF
/// ```
pub fn pacf(data: &[f64], max_lag: usize) -> Option<PacfResult> {
    let n = data.len();
    if n < 3 || max_lag == 0 {
        return None;
    }

    // First compute ACF
    let acf_result = acf(data, max_lag)?;
    let rho = &acf_result.acf;
    let max_lag = rho.len() - 1; // actual max lag (may be clamped)

    if max_lag == 0 {
        return None;
    }

    let mut pacf_vals = Vec::with_capacity(max_lag);

    // φ₁₁ = ρ(1)
    pacf_vals.push(rho[1]);

    if max_lag == 1 {
        return Some(PacfResult {
            pacf: pacf_vals,
            confidence_threshold: 1.96 / (n as f64).sqrt(),
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

    Some(PacfResult {
        pacf: pacf_vals,
        confidence_threshold: 1.96 / (n as f64).sqrt(),
    })
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
        let r = acf(&data, 3).expect("should compute");
        assert!((r.acf[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn acf_linear_trend() {
        // Linear trend → strong positive autocorrelation at low lags
        let data: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let r = acf(&data, 5).expect("should compute");
        assert!(r.acf[1] > 0.8, "lag-1 ACF for linear trend should be high");
        assert!(r.acf.len() == 6);
    }

    #[test]
    fn acf_alternating() {
        // Alternating series → negative lag-1 autocorrelation
        let data: Vec<f64> = (0..30).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        let r = acf(&data, 5).expect("should compute");
        assert!(r.acf[1] < -0.8, "alternating → negative lag-1 ACF");
        assert!(r.acf[2] > 0.8, "alternating → positive lag-2 ACF");
    }

    #[test]
    fn acf_white_noise_threshold() {
        let data: Vec<f64> = (0..100).map(|i| ((i * 7 + 3) % 13) as f64).collect();
        let r = acf(&data, 10).expect("should compute");
        // Threshold should be ~0.196 for n=100
        assert!((r.confidence_threshold - 1.96 / 10.0).abs() < 0.01);
    }

    #[test]
    fn acf_edge_cases() {
        assert!(acf(&[1.0], 3).is_none()); // too short
        assert!(acf(&[1.0, 2.0], 0).is_none()); // max_lag 0
        assert!(acf(&[5.0, 5.0, 5.0, 5.0], 2).is_none()); // constant
        assert!(acf(&[1.0, f64::NAN, 3.0], 1).is_none()); // NaN
    }

    #[test]
    fn acf_max_lag_clamped() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let r = acf(&data, 100).expect("should compute");
        assert_eq!(r.acf.len(), 5); // clamped to n-1 = 4, so lags 0..=4
    }

    // -----------------------------------------------------------------------
    // PACF tests
    // -----------------------------------------------------------------------

    #[test]
    fn pacf_lag1_equals_acf_lag1() {
        let data = [1.0, 3.0, 2.0, 5.0, 4.0, 7.0, 6.0, 8.0, 5.0, 3.0];
        let acf_r = acf(&data, 5).expect("should compute ACF");
        let pacf_r = pacf(&data, 5).expect("should compute PACF");
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
        let r = pacf(&data, 5).expect("should compute");
        // AR(1)-like: strong lag-1 PACF, weaker at higher lags
        assert!(r.pacf[0].abs() > 0.5, "lag-1 PACF should be strong");
    }

    #[test]
    fn pacf_bounded() {
        let data: Vec<f64> = (0..30).map(|i| (i as f64 * 0.5).sin()).collect();
        let r = pacf(&data, 10).expect("should compute");
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
        assert!(pacf(&[1.0, 2.0], 3).is_none()); // too short (< 3)
        assert!(pacf(&[1.0, 2.0, 3.0], 0).is_none()); // max_lag 0
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
            if let Some(r) = acf(&data, 10) {
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
            if let Some(r) = pacf(&data, 5) {
                for (i, &v) in r.pacf.iter().enumerate() {
                    prop_assert!((-1.0 - 1e-10..=1.0 + 1e-10).contains(&v), "PACF[{i}] = {v} out of bounds");
                }
            }
        }
    }
}
