//! Hypothesis testing.
//!
//! Parametric and non-parametric statistical tests: t-tests, ANOVA,
//! chi-squared tests, and normality tests.
//!
//! # Examples
//!
//! ```
//! use u_analytics::testing::{one_sample_t_test, TestResult};
//!
//! let data = [5.1, 4.9, 5.2, 5.0, 4.8, 5.3, 5.1, 4.9];
//! let result = one_sample_t_test(&data, 5.0).unwrap();
//! assert!(result.p_value > 0.05); // cannot reject H₀: μ = 5.0
//! ```

use u_numflow::special;
use u_numflow::stats;

/// Result of a hypothesis test.
#[derive(Debug, Clone, Copy)]
pub struct TestResult {
    /// Test statistic (t, F, χ², or z depending on test).
    pub statistic: f64,
    /// Degrees of freedom (may be fractional for Welch).
    pub df: f64,
    /// Two-tailed p-value.
    pub p_value: f64,
}

// ---------------------------------------------------------------------------
// t-tests
// ---------------------------------------------------------------------------

/// One-sample t-test: H₀: μ = μ₀.
///
/// # Algorithm
///
/// t = (x̄ - μ₀) / (s / √n), df = n-1.
///
/// # Returns
///
/// `None` if fewer than 2 observations or non-finite values.
///
/// # Examples
///
/// ```
/// use u_analytics::testing::one_sample_t_test;
///
/// let data = [2.0, 4.0, 6.0, 8.0, 10.0];
/// let r = one_sample_t_test(&data, 6.0).unwrap();
/// assert!(r.p_value > 0.5); // mean is 6.0
/// ```
pub fn one_sample_t_test(data: &[f64], mu0: f64) -> Option<TestResult> {
    let n = data.len();
    if n < 2 {
        return None;
    }
    if data.iter().any(|v| !v.is_finite()) || !mu0.is_finite() {
        return None;
    }

    let mean = stats::mean(data)?;
    let sd = stats::std_dev(data)?;

    if sd < 1e-300 {
        return None; // zero variance
    }

    let t = (mean - mu0) / (sd / (n as f64).sqrt());
    let df = (n - 1) as f64;
    let p_value = 2.0 * (1.0 - special::t_distribution_cdf(t.abs(), df));

    Some(TestResult {
        statistic: t,
        df,
        p_value,
    })
}

/// Two-sample Welch t-test: H₀: μ₁ = μ₂ (unequal variances).
///
/// # Algorithm
///
/// t = (x̄₁ - x̄₂) / √(s₁²/n₁ + s₂²/n₂)
/// df = Welch-Satterthwaite approximation.
///
/// # Returns
///
/// `None` if either sample has fewer than 2 observations.
///
/// # References
///
/// Welch (1947). "The generalization of Student's problem when several
/// different population variances are involved". Biometrika, 34, 28–35.
///
/// # Examples
///
/// ```
/// use u_analytics::testing::two_sample_t_test;
///
/// let a = [5.1, 4.9, 5.2, 5.0, 4.8];
/// let b = [7.1, 6.9, 7.2, 7.0, 6.8];
/// let r = two_sample_t_test(&a, &b).unwrap();
/// assert!(r.p_value < 0.01); // means clearly differ
/// ```
pub fn two_sample_t_test(a: &[f64], b: &[f64]) -> Option<TestResult> {
    let n1 = a.len();
    let n2 = b.len();
    if n1 < 2 || n2 < 2 {
        return None;
    }
    if a.iter().any(|v| !v.is_finite()) || b.iter().any(|v| !v.is_finite()) {
        return None;
    }

    let mean1 = stats::mean(a)?;
    let mean2 = stats::mean(b)?;
    let var1 = stats::variance(a)?;
    let var2 = stats::variance(b)?;

    let n1f = n1 as f64;
    let n2f = n2 as f64;

    let se_sq = var1 / n1f + var2 / n2f;
    if se_sq < 1e-300 {
        return None;
    }

    let t = (mean1 - mean2) / se_sq.sqrt();

    // Welch-Satterthwaite degrees of freedom
    let v1 = var1 / n1f;
    let v2 = var2 / n2f;
    let df = (v1 + v2).powi(2) / (v1 * v1 / (n1f - 1.0) + v2 * v2 / (n2f - 1.0));

    let p_value = 2.0 * (1.0 - special::t_distribution_cdf(t.abs(), df));

    Some(TestResult {
        statistic: t,
        df,
        p_value,
    })
}

/// Paired t-test: H₀: mean difference = 0.
///
/// # Algorithm
///
/// Computes differences dᵢ = xᵢ - yᵢ, then applies one-sample t-test
/// with μ₀ = 0.
///
/// # Returns
///
/// `None` if fewer than 2 pairs, slices differ in length, or non-finite values.
///
/// # Examples
///
/// ```
/// use u_analytics::testing::paired_t_test;
///
/// let before = [5.0, 6.0, 7.0, 8.0, 9.0];
/// let after  = [5.5, 6.2, 7.1, 8.3, 9.4];
/// let r = paired_t_test(&before, &after).unwrap();
/// assert!(r.statistic < 0.0); // after > before
/// ```
pub fn paired_t_test(x: &[f64], y: &[f64]) -> Option<TestResult> {
    if x.len() != y.len() || x.len() < 2 {
        return None;
    }

    let diffs: Vec<f64> = x.iter().zip(y.iter()).map(|(&a, &b)| a - b).collect();
    one_sample_t_test(&diffs, 0.0)
}

// ---------------------------------------------------------------------------
// ANOVA
// ---------------------------------------------------------------------------

/// Result of one-way ANOVA.
#[derive(Debug, Clone)]
pub struct AnovaResult {
    /// F-statistic.
    pub f_statistic: f64,
    /// Degrees of freedom between groups.
    pub df_between: usize,
    /// Degrees of freedom within groups.
    pub df_within: usize,
    /// p-value.
    pub p_value: f64,
    /// Sum of squares between groups.
    pub ss_between: f64,
    /// Sum of squares within groups.
    pub ss_within: f64,
    /// Mean square between.
    pub ms_between: f64,
    /// Mean square within.
    pub ms_within: f64,
    /// Group means.
    pub group_means: Vec<f64>,
    /// Grand mean.
    pub grand_mean: f64,
}

/// One-way ANOVA: H₀: all group means are equal.
///
/// # Algorithm
///
/// F = MS_between / MS_within where
/// MS_between = SS_between / (k-1),
/// MS_within = SS_within / (N-k).
///
/// # Returns
///
/// `None` if fewer than 2 groups, any group has fewer than 2 observations,
/// or non-finite values.
///
/// # References
///
/// Fisher (1925). "Statistical Methods for Research Workers".
///
/// # Examples
///
/// ```
/// use u_analytics::testing::one_way_anova;
///
/// let group1 = [5.0, 6.0, 7.0, 5.5, 6.5];
/// let group2 = [8.0, 9.0, 8.5, 9.5, 8.0];
/// let group3 = [4.0, 3.0, 3.5, 4.5, 4.0];
/// let r = one_way_anova(&[&group1, &group2, &group3]).unwrap();
/// assert!(r.p_value < 0.01); // means clearly differ
/// ```
pub fn one_way_anova(groups: &[&[f64]]) -> Option<AnovaResult> {
    let k = groups.len();
    if k < 2 {
        return None;
    }

    for g in groups {
        if g.len() < 2 || g.iter().any(|v| !v.is_finite()) {
            return None;
        }
    }

    let total_n: usize = groups.iter().map(|g| g.len()).sum();

    // Grand mean
    let grand_sum: f64 = groups.iter().flat_map(|g| g.iter()).sum();
    let grand_mean = grand_sum / total_n as f64;

    // Group means
    let group_means: Vec<f64> = groups
        .iter()
        .map(|g| g.iter().sum::<f64>() / g.len() as f64)
        .collect();

    // Sum of squares
    let ss_between: f64 = groups
        .iter()
        .zip(group_means.iter())
        .map(|(g, &gm)| g.len() as f64 * (gm - grand_mean).powi(2))
        .sum();

    let ss_within: f64 = groups
        .iter()
        .zip(group_means.iter())
        .map(|(g, &gm)| g.iter().map(|&x| (x - gm).powi(2)).sum::<f64>())
        .sum();

    let df_between = k - 1;
    let df_within = total_n - k;

    if df_within == 0 {
        return None;
    }

    let ms_between = ss_between / df_between as f64;
    let ms_within = ss_within / df_within as f64;

    let f_statistic = if ms_within > 1e-300 {
        ms_between / ms_within
    } else {
        f64::INFINITY
    };

    let p_value = if f_statistic.is_infinite() {
        0.0
    } else {
        1.0 - special::f_distribution_cdf(f_statistic, df_between as f64, df_within as f64)
    };

    Some(AnovaResult {
        f_statistic,
        df_between,
        df_within,
        p_value,
        ss_between,
        ss_within,
        ms_between,
        ms_within,
        group_means,
        grand_mean,
    })
}

// ---------------------------------------------------------------------------
// Chi-squared tests
// ---------------------------------------------------------------------------

/// Chi-squared goodness-of-fit test: H₀: observed matches expected distribution.
///
/// # Algorithm
///
/// χ² = Σ (Oᵢ - Eᵢ)² / Eᵢ, df = k-1.
///
/// # Returns
///
/// `None` if fewer than 2 categories, any expected frequency ≤ 0, or
/// slices differ in length.
///
/// # Examples
///
/// ```
/// use u_analytics::testing::chi_squared_goodness_of_fit;
///
/// let observed = [50.0, 30.0, 20.0];
/// let expected = [40.0, 35.0, 25.0];
/// let r = chi_squared_goodness_of_fit(&observed, &expected).unwrap();
/// assert!(r.statistic > 0.0);
/// ```
pub fn chi_squared_goodness_of_fit(observed: &[f64], expected: &[f64]) -> Option<TestResult> {
    let k = observed.len();
    if k < 2 || k != expected.len() {
        return None;
    }

    for &e in expected {
        if e <= 0.0 || !e.is_finite() {
            return None;
        }
    }
    for &o in observed {
        if o < 0.0 || !o.is_finite() {
            return None;
        }
    }

    let chi2: f64 = observed
        .iter()
        .zip(expected.iter())
        .map(|(&o, &e)| (o - e).powi(2) / e)
        .sum();

    let df = (k - 1) as f64;
    let p_value = 1.0 - special::chi_squared_cdf(chi2, df);

    Some(TestResult {
        statistic: chi2,
        df,
        p_value,
    })
}

/// Chi-squared test of independence on a contingency table.
///
/// # Arguments
///
/// * `table` — Flat row-major contingency table (rows × cols observed frequencies).
/// * `n_rows` — Number of rows.
/// * `n_cols` — Number of columns.
///
/// # Algorithm
///
/// Expected: Eᵢⱼ = (row_sumᵢ × col_sumⱼ) / N.
/// χ² = Σᵢⱼ (Oᵢⱼ - Eᵢⱼ)² / Eᵢⱼ, df = (r-1)(c-1).
///
/// # Returns
///
/// `None` if fewer than 2 rows or columns, any cell is negative, or
/// any marginal is zero.
///
/// # Examples
///
/// ```
/// use u_analytics::testing::chi_squared_independence;
///
/// // 2×2 contingency table
/// let table = [30.0, 10.0, 20.0, 40.0];
/// let r = chi_squared_independence(&table, 2, 2).unwrap();
/// assert!(r.p_value < 0.01);
/// ```
pub fn chi_squared_independence(table: &[f64], n_rows: usize, n_cols: usize) -> Option<TestResult> {
    if n_rows < 2 || n_cols < 2 || table.len() != n_rows * n_cols {
        return None;
    }

    for &v in table {
        if v < 0.0 || !v.is_finite() {
            return None;
        }
    }

    // Row sums and column sums
    let mut row_sums = vec![0.0; n_rows];
    let mut col_sums = vec![0.0; n_cols];
    let mut total = 0.0;

    for i in 0..n_rows {
        for j in 0..n_cols {
            let val = table[i * n_cols + j];
            row_sums[i] += val;
            col_sums[j] += val;
            total += val;
        }
    }

    if total <= 0.0 {
        return None;
    }

    // Check no zero marginals
    for &r in &row_sums {
        if r <= 0.0 {
            return None;
        }
    }
    for &c in &col_sums {
        if c <= 0.0 {
            return None;
        }
    }

    // Compute chi-squared statistic
    let mut chi2 = 0.0;
    for i in 0..n_rows {
        for j in 0..n_cols {
            let observed = table[i * n_cols + j];
            let expected = row_sums[i] * col_sums[j] / total;
            chi2 += (observed - expected).powi(2) / expected;
        }
    }

    let df = ((n_rows - 1) * (n_cols - 1)) as f64;
    let p_value = 1.0 - special::chi_squared_cdf(chi2, df);

    Some(TestResult {
        statistic: chi2,
        df,
        p_value,
    })
}

// ---------------------------------------------------------------------------
// Normality tests
// ---------------------------------------------------------------------------

/// Jarque-Bera normality test: H₀: data is normally distributed.
///
/// # Algorithm
///
/// JB = (n/6) · [S² + (K²/4)]
///
/// where S = skewness, K = excess kurtosis. JB ~ χ²(2) under H₀.
///
/// # Returns
///
/// `None` if fewer than 8 observations or non-finite values.
///
/// # References
///
/// Jarque & Bera (1987). "A test for normality of observations and
/// regression residuals". International Statistical Review, 55(2), 163–172.
///
/// # Examples
///
/// ```
/// use u_analytics::testing::jarque_bera_test;
///
/// // Near-normal data
/// let data = [-1.5, -1.0, -0.5, 0.0, 0.0, 0.5, 1.0, 1.5];
/// let r = jarque_bera_test(&data).unwrap();
/// assert!(r.p_value > 0.05); // cannot reject normality
/// ```
pub fn jarque_bera_test(data: &[f64]) -> Option<TestResult> {
    let n = data.len();
    if n < 8 {
        return None;
    }
    if data.iter().any(|v| !v.is_finite()) {
        return None;
    }

    let s = stats::skewness(data)?;
    let k = stats::kurtosis(data)?;

    let nf = n as f64;
    let jb = (nf / 6.0) * (s * s + k * k / 4.0);
    let p_value = 1.0 - special::chi_squared_cdf(jb, 2.0);

    Some(TestResult {
        statistic: jb,
        df: 2.0,
        p_value,
    })
}

/// Result of the Anderson-Darling normality test.
#[derive(Debug, Clone, Copy)]
pub struct AndersonDarlingResult {
    /// The A² test statistic (raw, before sample-size correction).
    pub statistic: f64,
    /// The modified statistic A*² = A² × (1 + 0.75/n + 2.25/n²).
    pub statistic_star: f64,
    /// The p-value. Small values reject the null hypothesis of normality.
    pub p_value: f64,
}

/// Anderson-Darling normality test: H₀: data is normally distributed.
///
/// More sensitive to tail deviations than Kolmogorov-Smirnov.
///
/// # Algorithm
///
/// 1. Standardize sorted data: zᵢ = (x₍ᵢ₎ - x̄) / s
/// 2. Compute A² = -n - (1/n) Σᵢ (2i-1) [ln Φ(zᵢ) + ln(1 - Φ(z_{n+1-i}))]
/// 3. Apply Stephens (1986) correction: A*² = A² (1 + 0.75/n + 2.25/n²)
/// 4. Compute p-value from piecewise exponential approximation
///
/// # Returns
///
/// `None` if n < 8, all values identical, or non-finite values.
///
/// # References
///
/// - Anderson & Darling (1952). "Asymptotic theory of certain goodness of
///   fit criteria based on stochastic processes". Annals of Mathematical
///   Statistics, 23(2), 193–212.
/// - Stephens (1986). "Tests based on EDF statistics". In D'Agostino &
///   Stephens (Eds.), Goodness-of-Fit Techniques. Marcel Dekker.
///
/// # Examples
///
/// ```
/// use u_analytics::testing::anderson_darling_test;
///
/// let data = [-1.5, -1.0, -0.5, 0.0, 0.0, 0.5, 1.0, 1.5];
/// let r = anderson_darling_test(&data).unwrap();
/// assert!(r.p_value > 0.05); // cannot reject normality
/// ```
pub fn anderson_darling_test(data: &[f64]) -> Option<AndersonDarlingResult> {
    let n = data.len();
    if n < 8 {
        return None;
    }
    if data.iter().any(|v| !v.is_finite()) {
        return None;
    }

    let mean = stats::mean(data)?;
    let sd = stats::std_dev(data)?;

    if sd < 1e-300 {
        return None; // zero variance
    }

    // Sort data
    let mut x: Vec<f64> = data.to_vec();
    x.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let nf = n as f64;

    // Compute A² statistic
    let mut s = 0.0;
    for i in 0..n {
        let z = (x[i] - mean) / sd;
        let phi = special::standard_normal_cdf(z);
        // Clamp to avoid ln(0) or ln(negative)
        let phi = phi.clamp(1e-15, 1.0 - 1e-15);

        let z_rev = (x[n - 1 - i] - mean) / sd;
        let phi_rev = special::standard_normal_cdf(z_rev);
        let phi_rev = phi_rev.clamp(1e-15, 1.0 - 1e-15);

        let coeff = (2 * (i + 1) - 1) as f64;
        s += coeff * (phi.ln() + (1.0 - phi_rev).ln());
    }

    let a2 = -nf - s / nf;

    // Stephens (1986) correction for sample size
    let a2_star = a2 * (1.0 + 0.75 / nf + 2.25 / (nf * nf));

    // P-value from piecewise approximation (D'Agostino & Stephens 1986)
    let p = if a2_star >= 0.6 {
        (1.2937 - 5.709 * a2_star + 0.0186 * a2_star * a2_star).exp()
    } else if a2_star > 0.34 {
        (0.9177 - 4.279 * a2_star - 1.38 * a2_star * a2_star).exp()
    } else if a2_star > 0.2 {
        1.0 - (-8.318 + 42.796 * a2_star - 59.938 * a2_star * a2_star).exp()
    } else {
        1.0 - (-13.436 + 101.14 * a2_star - 223.73 * a2_star * a2_star).exp()
    };

    Some(AndersonDarlingResult {
        statistic: a2,
        statistic_star: a2_star,
        p_value: p.clamp(0.0, 1.0),
    })
}

/// Result of the Shapiro-Wilk normality test.
#[derive(Debug, Clone, Copy)]
pub struct ShapiroWilkResult {
    /// The W statistic (0 < W ≤ 1). Values close to 1 suggest normality.
    pub w: f64,
    /// The p-value. Small values reject the null hypothesis of normality.
    pub p_value: f64,
}

/// Shapiro-Wilk normality test: H₀: data is normally distributed.
///
/// The most powerful general normality test for small to moderate samples.
///
/// # Algorithm
///
/// Uses the Royston (1992, 1995) algorithm (AS R94):
/// 1. Compute coefficients from normal order statistics (Blom approximation)
/// 2. Calculate W = (Σ aᵢ x₍ᵢ₎)² / Σ (xᵢ - x̄)²
/// 3. Transform W to z-score via log-normal approximation
/// 4. Compute p-value from standard normal distribution
///
/// # Supported range
///
/// n = 3..5000. Returns `None` outside this range.
///
/// # Returns
///
/// `None` if n < 3, n > 5000, all values identical, or non-finite values.
///
/// # References
///
/// - Shapiro & Wilk (1965). "An analysis of variance test for normality".
///   Biometrika, 52(3–4), 591–611.
/// - Royston (1992). "Approximating the Shapiro-Wilk W-test for
///   non-normality". Statistics and Computing, 2, 117–119.
/// - Royston (1995). "Remark AS R94: A remark on Algorithm AS 181".
///   Applied Statistics, 44(4), 547–551.
///
/// # Examples
///
/// ```
/// use u_analytics::testing::shapiro_wilk_test;
///
/// let data = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5];
/// let r = shapiro_wilk_test(&data).unwrap();
/// assert!(r.w > 0.9);
/// assert!(r.p_value > 0.05); // cannot reject normality
/// ```
pub fn shapiro_wilk_test(data: &[f64]) -> Option<ShapiroWilkResult> {
    let n = data.len();
    if !(3..=5000).contains(&n) {
        return None;
    }
    if data.iter().any(|v| !v.is_finite()) {
        return None;
    }

    // Sort data
    let mut x: Vec<f64> = data.to_vec();
    x.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    if x[n - 1] - x[0] < 1e-300 {
        return None; // all values identical
    }

    let nn2 = n / 2;

    // Special case n = 3
    if n == 3 {
        return shapiro_wilk_n3(&x);
    }

    // Compute coefficients via Royston algorithm
    let a = sw_coefficients(n, nn2)?;

    // Compute W statistic
    let w = sw_statistic(&x, &a, n, nn2);

    if !(0.0..=1.0 + 1e-10).contains(&w) {
        return None;
    }
    let w = w.min(1.0);

    // Compute p-value
    let p_value = sw_p_value(w, n);

    Some(ShapiroWilkResult {
        w,
        p_value: p_value.clamp(0.0, 1.0),
    })
}

// Shapiro-Wilk: n=3 exact formula
fn shapiro_wilk_n3(x: &[f64]) -> Option<ShapiroWilkResult> {
    // For n=3: a = [sqrt(1/2), 0, -sqrt(1/2)]
    let a1 = std::f64::consts::FRAC_1_SQRT_2; // 0.7071...
    let mean = (x[0] + x[1] + x[2]) / 3.0;
    let ss = x.iter().map(|&v| (v - mean).powi(2)).sum::<f64>();
    if ss < 1e-300 {
        return None;
    }

    let numerator = a1 * (x[2] - x[0]);
    let w = (numerator * numerator) / ss;
    let w = w.clamp(0.75, 1.0);

    // Exact p-value for n=3: p = 1 - (6/pi) * arccos(sqrt(w))
    let p = 1.0 - (6.0 / std::f64::consts::PI) * w.sqrt().acos();
    let p = p.clamp(0.0, 1.0);

    Some(ShapiroWilkResult { w, p_value: p })
}

// Royston polynomial coefficients (AS R94)
const SW_C1: [f64; 6] = [0.0, 0.221157, -0.147981, -2.07119, 4.434685, -2.706056];
const SW_C2: [f64; 6] = [0.0, 0.042981, -0.293762, -1.752461, 5.682633, -3.582633];
const SW_C3: [f64; 4] = [0.544, -0.39978, 0.025054, -6.714e-4];
const SW_C4: [f64; 4] = [1.3822, -0.77857, 0.062767, -0.0020322];
const SW_C5: [f64; 4] = [-1.5861, -0.31082, -0.083751, 0.0038915];
const SW_C6: [f64; 3] = [-0.4803, -0.082676, 0.0030302];
const SW_G: [f64; 2] = [-2.273, 0.459];

// Evaluate polynomial: c[0] + c[1]*x + c[2]*x^2 + ... (Horner's method)
fn sw_poly(c: &[f64], x: f64) -> f64 {
    let mut result = c[c.len() - 1];
    for i in (0..c.len() - 1).rev() {
        result = result * x + c[i];
    }
    result
}

// Compute Shapiro-Wilk coefficients using Royston's algorithm
fn sw_coefficients(n: usize, nn2: usize) -> Option<Vec<f64>> {
    let mut a = vec![0.0; nn2];

    // Blom's approximation for expected normal order statistics
    let mut m = vec![0.0; nn2];
    let mut summ2 = 0.0;
    for (i, mi) in m.iter_mut().enumerate() {
        // m[i] corresponds to the (i+1)-th order statistic expectation
        let p = (i as f64 + 1.0 - 0.375) / (n as f64 + 0.25);
        *mi = special::inverse_normal_cdf(p);
        summ2 += *mi * *mi;
    }
    summ2 *= 2.0;
    let ssumm2 = summ2.sqrt();
    let rsn = 1.0 / (n as f64).sqrt();

    // First coefficient: polynomial correction
    let a1 = sw_poly(&SW_C1, rsn) - m[0] / ssumm2;

    if n <= 5 {
        // For n=4,5: only a[0] is corrected
        let fac_sq = summ2 - 2.0 * m[0] * m[0];
        let one_minus = 1.0 - 2.0 * a1 * a1;
        if fac_sq <= 0.0 || one_minus <= 0.0 {
            return None;
        }
        let fac = (fac_sq / one_minus).sqrt();
        a[0] = a1;
        for i in 1..nn2 {
            a[i] = -m[i] / fac;
        }
    } else {
        // For n>5: a[0] and a[1] are corrected
        let a2 = -m[1] / ssumm2 + sw_poly(&SW_C2, rsn);
        let fac_sq = summ2 - 2.0 * m[0] * m[0] - 2.0 * m[1] * m[1];
        let one_minus = 1.0 - 2.0 * a1 * a1 - 2.0 * a2 * a2;
        if fac_sq <= 0.0 || one_minus <= 0.0 {
            return None;
        }
        let fac = (fac_sq / one_minus).sqrt();
        a[0] = a1;
        a[1] = a2;
        for i in 2..nn2 {
            a[i] = -m[i] / fac;
        }
    }

    Some(a)
}

// Compute W statistic from sorted data, coefficients, and range
fn sw_statistic(x: &[f64], a: &[f64], n: usize, nn2: usize) -> f64 {
    // Numerator: (sum a_i * (x_{n+1-i} - x_i))^2
    let mut sa = 0.0;
    for i in 0..nn2 {
        sa += a[i] * (x[n - 1 - i] - x[i]);
    }

    // Denominator: sum of squares about the mean
    let mean = x.iter().sum::<f64>() / n as f64;
    let ss: f64 = x.iter().map(|&v| (v - mean).powi(2)).sum();

    if ss < 1e-300 {
        return 1.0; // degenerate
    }

    (sa * sa) / ss
}

// Compute p-value from W statistic using Royston's transformation
fn sw_p_value(w: f64, n: usize) -> f64 {
    let nf = n as f64;

    if n == 3 {
        // Should not reach here (handled separately), but just in case
        let p = 1.0 - (6.0 / std::f64::consts::PI) * w.sqrt().acos();
        return p.clamp(0.0, 1.0);
    }

    let w1 = 1.0 - w;
    if w1 <= 0.0 {
        return 1.0; // perfectly normal
    }

    let y = w1.ln();

    if n <= 11 {
        // Small sample: gamma + log transformation
        let gamma = sw_poly(&SW_G, nf);
        if y >= gamma {
            return 0.0; // extremely non-normal
        }
        let y2 = -(gamma - y).ln();
        let m = sw_poly(&SW_C3, nf);
        let s = sw_poly(&SW_C4, nf).exp();
        if s < 1e-300 {
            return 0.0;
        }
        let z = (y2 - m) / s;
        1.0 - special::standard_normal_cdf(z)
    } else {
        // Large sample: log-normal transformation
        let xx = nf.ln();
        let m = sw_poly(&SW_C5, xx);
        let s = sw_poly(&SW_C6, xx).exp();
        if s < 1e-300 {
            return 0.0;
        }
        let z = (y - m) / s;
        1.0 - special::standard_normal_cdf(z)
    }
}

// ---------------------------------------------------------------------------
// Non-parametric tests
// ---------------------------------------------------------------------------

/// Mann-Whitney U test: H₀: the two populations have the same distribution.
///
/// Non-parametric alternative to the two-sample t-test. Does not assume
/// normality.
///
/// # Algorithm
///
/// 1. Combine samples, rank all observations (average ranks for ties)
/// 2. U₁ = R₁ - n₁(n₁+1)/2 where R₁ = sum of ranks in sample 1
/// 3. Normal approximation: z = (U₁ - μ) / σ
///    where μ = n₁n₂/2, σ² includes tie correction
///
/// # Returns
///
/// `None` if either sample has fewer than 2 observations or non-finite values.
///
/// # References
///
/// - Mann & Whitney (1947). "On a test of whether one of two random
///   variables is stochastically larger than the other". Annals of
///   Mathematical Statistics, 18(1), 50–60.
///
/// # Examples
///
/// ```
/// use u_analytics::testing::mann_whitney_u_test;
///
/// let a = [1.0, 2.0, 3.0, 4.0, 5.0];
/// let b = [6.0, 7.0, 8.0, 9.0, 10.0];
/// let r = mann_whitney_u_test(&a, &b).unwrap();
/// assert!(r.p_value < 0.05);
/// ```
pub fn mann_whitney_u_test(a: &[f64], b: &[f64]) -> Option<TestResult> {
    let n1 = a.len();
    let n2 = b.len();
    if n1 < 2 || n2 < 2 {
        return None;
    }
    if a.iter().any(|v| !v.is_finite()) || b.iter().any(|v| !v.is_finite()) {
        return None;
    }

    let n = n1 + n2;
    let n1f = n1 as f64;
    let n2f = n2 as f64;
    let nf = n as f64;

    // Combine and rank
    let mut combined: Vec<(f64, usize)> = Vec::with_capacity(n);
    for &v in a {
        combined.push((v, 0)); // group 0 = sample a
    }
    for &v in b {
        combined.push((v, 1)); // group 1 = sample b
    }
    combined.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(std::cmp::Ordering::Equal));

    // Assign average ranks and track ties
    let ranks = average_ranks(&combined);

    // Sum of ranks for sample a
    let r1: f64 = combined
        .iter()
        .zip(ranks.iter())
        .filter(|((_, g), _)| *g == 0)
        .map(|(_, &r)| r)
        .sum();

    // U statistic
    let u1 = r1 - n1f * (n1f + 1.0) / 2.0;

    // Tie correction
    let tie_correction = compute_tie_correction(&combined);

    // Normal approximation
    let mu = n1f * n2f / 2.0;
    let sigma_sq = n1f * n2f / 12.0 * (nf + 1.0 - tie_correction / (nf * (nf - 1.0)));

    if sigma_sq <= 0.0 {
        return None;
    }

    let z = (u1 - mu) / sigma_sq.sqrt();
    let p_value = 2.0 * (1.0 - special::standard_normal_cdf(z.abs()));

    Some(TestResult {
        statistic: u1,
        df: 0.0, // not applicable for non-parametric
        p_value,
    })
}

/// Wilcoxon signed-rank test: H₀: median of differences = 0.
///
/// Non-parametric alternative to the paired t-test. Does not assume
/// normality of differences.
///
/// # Algorithm
///
/// 1. Compute differences dᵢ = xᵢ - yᵢ, discard zeros
/// 2. Rank |dᵢ| (average ranks for ties)
/// 3. T⁺ = sum of ranks where dᵢ > 0
/// 4. Normal approximation: z = (T⁺ - μ) / σ
///    where μ = n(n+1)/4, σ² includes tie correction
///
/// # Returns
///
/// `None` if fewer than 2 non-zero differences, slices differ in length,
/// or non-finite values.
///
/// # References
///
/// - Wilcoxon (1945). "Individual comparisons by ranking methods".
///   Biometrics Bulletin, 1(6), 80–83.
///
/// # Examples
///
/// ```
/// use u_analytics::testing::wilcoxon_signed_rank_test;
///
/// let before = [5.0, 6.0, 7.0, 8.0, 9.0];
/// let after  = [6.0, 7.5, 8.0, 9.5, 11.0];
/// let r = wilcoxon_signed_rank_test(&after, &before).unwrap();
/// assert!(r.statistic > 0.0); // T+ sum of positive ranks
/// ```
pub fn wilcoxon_signed_rank_test(x: &[f64], y: &[f64]) -> Option<TestResult> {
    if x.len() != y.len() || x.len() < 2 {
        return None;
    }
    if x.iter().any(|v| !v.is_finite()) || y.iter().any(|v| !v.is_finite()) {
        return None;
    }

    // Compute differences and discard zeros
    let diffs: Vec<f64> = x
        .iter()
        .zip(y.iter())
        .map(|(&a, &b)| a - b)
        .filter(|&d| d.abs() > 1e-300)
        .collect();

    let nr = diffs.len();
    if nr < 2 {
        return None;
    }

    let nf = nr as f64;

    // Sort by absolute difference and rank
    let mut abs_diffs: Vec<(f64, usize)> = diffs
        .iter()
        .enumerate()
        .map(|(i, &d)| (d.abs(), i))
        .collect();
    abs_diffs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // Assign average ranks
    let ranks = average_ranks(&abs_diffs);

    // T+ = sum of ranks where the original difference is positive
    let t_plus: f64 = abs_diffs
        .iter()
        .zip(ranks.iter())
        .filter(|((_, orig_idx), _)| diffs[*orig_idx] > 0.0)
        .map(|(_, &r)| r)
        .sum();

    // Tie correction for variance
    let tie_correction_val = compute_tie_correction(&abs_diffs);

    // Normal approximation
    let mu = nf * (nf + 1.0) / 4.0;
    let sigma_sq = nf * (nf + 1.0) * (2.0 * nf + 1.0) / 24.0 - tie_correction_val / 48.0;

    if sigma_sq <= 0.0 {
        return None;
    }

    let z = (t_plus - mu) / sigma_sq.sqrt();
    let p_value = 2.0 * (1.0 - special::standard_normal_cdf(z.abs()));

    Some(TestResult {
        statistic: t_plus,
        df: 0.0,
        p_value,
    })
}

// Assign average ranks to sorted (value, group_or_index) pairs.
// Handles ties by assigning the average of the tied ranks.
fn average_ranks(sorted: &[(f64, usize)]) -> Vec<f64> {
    let n = sorted.len();
    let mut ranks = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n && (sorted[j].0 - sorted[i].0).abs() < 1e-12 {
            j += 1;
        }
        // Positions i..j are tied; average rank = (i+1 + j) / 2
        let avg_rank = (i + 1 + j) as f64 / 2.0;
        for rank in ranks.iter_mut().take(j).skip(i) {
            *rank = avg_rank;
        }
        i = j;
    }
    ranks
}

// Compute tie correction factor: Σ tₖ(tₖ² - 1) for all tie groups
fn compute_tie_correction(sorted: &[(f64, usize)]) -> f64 {
    let n = sorted.len();
    let mut correction = 0.0;
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n && (sorted[j].0 - sorted[i].0).abs() < 1e-12 {
            j += 1;
        }
        let t = (j - i) as f64;
        if t > 1.0 {
            correction += t * (t * t - 1.0);
        }
        i = j;
    }
    correction
}

/// Kruskal-Wallis test: H₀: all groups have the same distribution.
///
/// Non-parametric alternative to one-way ANOVA. Does not assume normality.
///
/// # Algorithm
///
/// 1. Combine all groups, rank observations (average ranks for ties)
/// 2. H = (12 / N(N+1)) Σ nᵢ (R̄ᵢ - R̄)² with tie correction
/// 3. H ~ χ²(k-1) under H₀
///
/// # Returns
///
/// `None` if fewer than 2 groups, any group has fewer than 2 observations,
/// or non-finite values.
///
/// # References
///
/// - Kruskal & Wallis (1952). "Use of ranks in one-criterion variance
///   analysis". JASA, 47(260), 583–621.
///
/// # Examples
///
/// ```
/// use u_analytics::testing::kruskal_wallis_test;
///
/// let g1 = [1.0, 2.0, 3.0, 4.0, 5.0];
/// let g2 = [6.0, 7.0, 8.0, 9.0, 10.0];
/// let g3 = [11.0, 12.0, 13.0, 14.0, 15.0];
/// let r = kruskal_wallis_test(&[&g1, &g2, &g3]).unwrap();
/// assert!(r.p_value < 0.01);
/// ```
pub fn kruskal_wallis_test(groups: &[&[f64]]) -> Option<TestResult> {
    let k = groups.len();
    if k < 2 {
        return None;
    }
    for g in groups {
        if g.len() < 2 || g.iter().any(|v| !v.is_finite()) {
            return None;
        }
    }

    let total_n: usize = groups.iter().map(|g| g.len()).sum();
    let nf = total_n as f64;

    // Combine all observations with group labels
    let mut combined: Vec<(f64, usize)> = Vec::with_capacity(total_n);
    for (gi, g) in groups.iter().enumerate() {
        for &v in *g {
            combined.push((v, gi));
        }
    }
    combined.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let ranks = average_ranks(&combined);

    // Sum of ranks per group
    let mut rank_sums = vec![0.0; k];
    for ((_, gi), &r) in combined.iter().zip(ranks.iter()) {
        rank_sums[*gi] += r;
    }

    // H statistic: H = (12 / N(N+1)) * Σ Rᵢ²/nᵢ - 3(N+1)
    let mean_rank = (nf + 1.0) / 2.0;
    let mut h = 0.0;
    for (gi, g) in groups.iter().enumerate() {
        let ni = g.len() as f64;
        let mean_rank_i = rank_sums[gi] / ni;
        h += ni * (mean_rank_i - mean_rank).powi(2);
    }
    h *= 12.0 / (nf * (nf + 1.0));

    // Tie correction: divide by 1 - Σtₖ(tₖ²-1) / (N³ - N)
    let tie_corr = compute_tie_correction(&combined);
    let denom = 1.0 - tie_corr / (nf * nf * nf - nf);
    if denom > 1e-15 {
        h /= denom;
    }

    let df = (k - 1) as f64;
    let p_value = 1.0 - special::chi_squared_cdf(h, df);

    Some(TestResult {
        statistic: h,
        df,
        p_value,
    })
}

// ---------------------------------------------------------------------------
// Variance tests
// ---------------------------------------------------------------------------

/// Levene test for equality of variances: H₀: all groups have equal variance.
///
/// Robust to non-normality. Uses the **median** variant (Brown-Forsythe),
/// which is recommended for non-normal data.
///
/// # Algorithm
///
/// 1. Compute zᵢⱼ = |xᵢⱼ - median(groupᵢ)|
/// 2. Apply one-way ANOVA on the zᵢⱼ values
///
/// # Returns
///
/// `None` if fewer than 2 groups, any group < 2 observations, or non-finite values.
///
/// # References
///
/// - Levene (1960). "Robust tests for equality of variances". In
///   Olkin (Ed.), Contributions to Probability and Statistics.
/// - Brown & Forsythe (1974). "Robust tests for the equality of variances".
///   JASA, 69(346), 364–367.
///
/// # Examples
///
/// ```
/// use u_analytics::testing::levene_test;
///
/// let g1 = [4.9, 5.0, 5.0, 5.1, 5.0]; // tight cluster (low variance)
/// let g2 = [0.0, 3.0, 5.0, 7.0, 10.0]; // wide spread (high variance)
/// let r = levene_test(&[&g1, &g2]).unwrap();
/// assert!(r.p_value < 0.05); // clear variance difference
/// ```
pub fn levene_test(groups: &[&[f64]]) -> Option<TestResult> {
    let k = groups.len();
    if k < 2 {
        return None;
    }
    for g in groups {
        if g.len() < 2 || g.iter().any(|v| !v.is_finite()) {
            return None;
        }
    }

    // Compute z-values: |x - median(group)| (Brown-Forsythe variant)
    let z_groups: Vec<Vec<f64>> = groups
        .iter()
        .map(|g| {
            let median = stats::median(g).unwrap_or(0.0);
            g.iter().map(|&x| (x - median).abs()).collect()
        })
        .collect();

    // Apply ANOVA on the z-values
    let z_refs: Vec<&[f64]> = z_groups.iter().map(|v| v.as_slice()).collect();
    let anova = one_way_anova(&z_refs)?;

    Some(TestResult {
        statistic: anova.f_statistic,
        df: anova.df_between as f64,
        p_value: anova.p_value,
    })
}

// ---------------------------------------------------------------------------
// Multiple comparison correction
// ---------------------------------------------------------------------------

/// Bonferroni correction: adjusts p-values for multiple comparisons.
///
/// adjusted_pᵢ = min(pᵢ × m, 1.0) where m = number of tests.
///
/// # Returns
///
/// `None` if the slice is empty or contains non-finite values.
pub fn bonferroni_correction(p_values: &[f64]) -> Option<Vec<f64>> {
    if p_values.is_empty() || p_values.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let m = p_values.len() as f64;
    Some(p_values.iter().map(|&p| (p * m).min(1.0)).collect())
}

/// Benjamini-Hochberg FDR correction.
///
/// Controls false discovery rate at level α.
///
/// # Algorithm
///
/// 1. Sort p-values.
/// 2. For rank i (1-indexed): adjusted_pᵢ = pᵢ × m / i.
/// 3. Enforce monotonicity (cumulative minimum from right).
///
/// # Returns
///
/// `None` if the slice is empty or contains non-finite values.
///
/// # References
///
/// Benjamini & Hochberg (1995). "Controlling the false discovery rate".
/// JRSS-B, 57(1), 289–300.
pub fn benjamini_hochberg(p_values: &[f64]) -> Option<Vec<f64>> {
    let m = p_values.len();
    if m == 0 || p_values.iter().any(|v| !v.is_finite()) {
        return None;
    }

    // Sort indices by p-value
    let mut indices: Vec<usize> = (0..m).collect();
    indices.sort_by(|&a, &b| {
        p_values[a]
            .partial_cmp(&p_values[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mf = m as f64;
    let mut adjusted = vec![0.0; m];

    // Compute adjusted p-values
    let mut cummin = f64::INFINITY;
    for (rank_rev, &orig_idx) in indices.iter().enumerate().rev() {
        let rank = rank_rev + 1; // 1-indexed
        let adj = (p_values[orig_idx] * mf / rank as f64).min(1.0);
        cummin = cummin.min(adj);
        adjusted[orig_idx] = cummin;
    }

    Some(adjusted)
}

// ---------------------------------------------------------------------------
// Bartlett test for equality of variances
// ---------------------------------------------------------------------------

/// Bartlett test for equality of variances: H₀: all groups have equal variance.
///
/// Assumes data are from **normal** distributions. For non-normal data, prefer
/// [`levene_test`] (Brown-Forsythe variant).
///
/// # Algorithm
///
/// 1. Compute pooled variance: s²ₚ = Σ(nᵢ-1)s²ᵢ / (N-k)
/// 2. Numerator: (N-k) ln(s²ₚ) - Σ(nᵢ-1) ln(s²ᵢ)
/// 3. Correction factor: C = 1 + [1/(3(k-1))] × [Σ 1/(nᵢ-1) - 1/(N-k)]
/// 4. Statistic: T = numerator / C ~ χ²(k-1)
///
/// # Returns
///
/// `None` if fewer than 2 groups, any group < 2 observations, any group has
/// zero variance, or non-finite values.
///
/// # References
///
/// - Bartlett (1937). "Properties of sufficiency and statistical tests".
///   Proceedings of the Royal Society A, 160(901), 268–282.
///
/// # Examples
///
/// ```
/// use u_analytics::testing::bartlett_test;
///
/// let g1 = [2.0, 3.0, 4.0, 5.0, 6.0]; // variance ~2.5
/// let g2 = [10.0, 20.0, 30.0, 40.0, 50.0]; // variance ~250
/// let r = bartlett_test(&[&g1, &g2]).unwrap();
/// assert!(r.p_value < 0.01); // strongly different variances
/// ```
pub fn bartlett_test(groups: &[&[f64]]) -> Option<TestResult> {
    let k = groups.len();
    if k < 2 {
        return None;
    }

    let mut sizes = Vec::with_capacity(k);
    let mut vars = Vec::with_capacity(k);
    let mut n_total: usize = 0;

    for g in groups {
        if g.len() < 2 || g.iter().any(|v| !v.is_finite()) {
            return None;
        }
        let n = g.len();
        let v = stats::variance(g)?;
        if v <= 0.0 {
            return None; // zero variance → ln undefined
        }
        sizes.push(n);
        vars.push(v);
        n_total += n;
    }

    let nk = n_total - k; // N - k
    if nk == 0 {
        return None;
    }
    let nk_f = nk as f64;

    // Pooled variance
    let s2_pooled: f64 = sizes
        .iter()
        .zip(vars.iter())
        .map(|(&n, &v)| (n as f64 - 1.0) * v)
        .sum::<f64>()
        / nk_f;

    if s2_pooled <= 0.0 {
        return None;
    }

    // Numerator: (N-k) ln(s²ₚ) - Σ(nᵢ-1) ln(s²ᵢ)
    let num = nk_f * s2_pooled.ln()
        - sizes
            .iter()
            .zip(vars.iter())
            .map(|(&n, &v)| (n as f64 - 1.0) * v.ln())
            .sum::<f64>();

    // Correction factor C
    let sum_recip: f64 = sizes.iter().map(|&n| 1.0 / (n as f64 - 1.0)).sum();
    let c = 1.0 + (sum_recip - 1.0 / nk_f) / (3.0 * (k as f64 - 1.0));

    let statistic = num / c;
    let df = (k - 1) as f64;
    let p_value = 1.0 - special::chi_squared_cdf(statistic, df);

    Some(TestResult {
        statistic,
        df,
        p_value,
    })
}

// ---------------------------------------------------------------------------
// Fisher exact test (2×2)
// ---------------------------------------------------------------------------

/// Fisher exact test for a 2×2 contingency table.
///
/// Tests H₀: the two categorical variables are independent.
/// Unlike the chi-squared test, this is exact and valid for small samples.
///
/// # Arguments
///
/// The 2×2 table is specified as four cell counts:
///
/// ```text
///          Col1   Col2
///   Row1 |  a   |  b  |
///   Row2 |  c   |  d  |
/// ```
///
/// # Algorithm
///
/// 1. Compute probability of observed table via hypergeometric distribution
///    (using log-factorials for numerical stability).
/// 2. Enumerate all tables with the same marginals.
/// 3. Two-tailed p-value = sum of probabilities ≤ P(observed).
///
/// # Returns
///
/// `None` if any marginal total is zero (degenerate table).
///
/// # References
///
/// - Fisher (1922). "On the interpretation of χ² from contingency tables,
///   and the calculation of P". JRSS, 85(1), 87–94.
///
/// # Examples
///
/// ```
/// use u_analytics::testing::fisher_exact_test;
///
/// // Tea-tasting experiment
/// let r = fisher_exact_test(3, 1, 1, 3).unwrap();
/// assert!(r.p_value > 0.05); // not significant at 5%
/// ```
pub fn fisher_exact_test(a: u64, b: u64, c: u64, d: u64) -> Option<TestResult> {
    let row1 = a + b;
    let row2 = c + d;
    let col1 = a + c;
    let col2 = b + d;
    let n = a + b + c + d;

    // Degenerate if any marginal is zero
    if row1 == 0 || row2 == 0 || col1 == 0 || col2 == 0 {
        return None;
    }

    // Log-probability of a specific table given marginals
    let log_prob = |a_i: u64| -> f64 {
        let b_i = row1 - a_i;
        let c_i = col1 - a_i;
        let d_i = row2 - c_i;
        ln_factorial(row1) + ln_factorial(row2) + ln_factorial(col1) + ln_factorial(col2)
            - ln_factorial(a_i)
            - ln_factorial(b_i)
            - ln_factorial(c_i)
            - ln_factorial(d_i)
            - ln_factorial(n)
    };

    // Range of valid values for cell a
    let a_min = col1.saturating_sub(row2);
    let a_max = row1.min(col1);

    let log_p_obs = log_prob(a);

    // Two-tailed: sum probabilities ≤ P(observed)
    let mut p_value = 0.0;
    for a_i in a_min..=a_max {
        let lp = log_prob(a_i);
        // Use small tolerance for floating-point comparison
        if lp <= log_p_obs + 1e-10 {
            p_value += lp.exp();
        }
    }

    // Clamp to [0, 1]
    let p_value = p_value.min(1.0);

    // Odds ratio: (a*d) / (b*c)
    let odds_ratio = if b > 0 && c > 0 {
        (a as f64 * d as f64) / (b as f64 * c as f64)
    } else {
        f64::INFINITY
    };

    Some(TestResult {
        statistic: odds_ratio,
        df: 1.0,
        p_value,
    })
}

// ---------------------------------------------------------------------------
// Mann-Kendall trend test
// ---------------------------------------------------------------------------

/// Result of the Mann-Kendall trend test.
#[derive(Debug, Clone, Copy)]
pub struct MannKendallResult {
    /// Mann-Kendall S statistic: Σ sign(xⱼ - xᵢ) for all i < j.
    pub s_statistic: i64,
    /// Variance of S (with tie correction).
    pub variance: f64,
    /// Z statistic (with continuity correction).
    pub z_statistic: f64,
    /// Two-tailed p-value.
    pub p_value: f64,
    /// Kendall's tau: S / [n(n-1)/2]. Range [-1, 1].
    pub kendall_tau: f64,
    /// Sen's slope estimator: median of (xⱼ - xᵢ)/(j - i) for all i < j.
    pub sen_slope: f64,
}

/// Mann-Kendall non-parametric trend test with Sen's slope estimator.
///
/// Tests H₀: no monotonic trend vs H₁: monotonic trend exists.
/// Assumes serially independent observations (no autocorrelation).
///
/// # Algorithm
///
/// 1. S = Σᵢ<ⱼ sign(xⱼ - xᵢ)
/// 2. Var(S) = \[n(n-1)(2n+5) - Σ tₖ(tₖ-1)(2tₖ+5)\] / 18 (tie-corrected)
/// 3. Z = (S-1)/√Var(S) if S>0, 0 if S=0, (S+1)/√Var(S) if S<0
/// 4. Sen's slope = median of all pairwise slopes (xⱼ - xᵢ)/(j - i)
///
/// References:
/// - Mann (1945), "Nonparametric tests against trend"
/// - Kendall (1975), "Rank Correlation Methods"
/// - Sen (1968), "Estimates of the regression coefficient based on Kendall's tau"
///
/// # Complexity
///
/// O(n² log n) — pairwise comparisons O(n²) plus median finding O(n² log n²).
///
/// # Returns
///
/// `None` if fewer than 4 data points, non-finite values, or zero variance.
///
/// # Examples
///
/// ```
/// use u_analytics::testing::mann_kendall_test;
///
/// // Clear upward trend
/// let data = [1.0, 2.3, 3.1, 4.5, 5.2, 6.8, 7.1, 8.9, 9.5, 10.2];
/// let r = mann_kendall_test(&data).unwrap();
/// assert!(r.p_value < 0.01);
/// assert!(r.kendall_tau > 0.8);
/// assert!(r.sen_slope > 0.0);
/// ```
pub fn mann_kendall_test(data: &[f64]) -> Option<MannKendallResult> {
    let n = data.len();
    if n < 4 || data.iter().any(|v| !v.is_finite()) {
        return None;
    }

    // Step 1: Compute S statistic
    let mut s: i64 = 0;
    for i in 0..n - 1 {
        for j in (i + 1)..n {
            let diff = data[j] - data[i];
            if diff > 0.0 {
                s += 1;
            } else if diff < 0.0 {
                s -= 1;
            }
        }
    }

    // Step 2: Compute tie groups
    let mut sorted: Vec<f64> = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("finite values"));

    let mut tie_correction: f64 = 0.0;
    let mut current_count: usize = 1;
    for i in 1..sorted.len() {
        if (sorted[i] - sorted[i - 1]).abs() < 1e-10 {
            current_count += 1;
        } else {
            if current_count > 1 {
                let t = current_count as f64;
                tie_correction += t * (t - 1.0) * (2.0 * t + 5.0);
            }
            current_count = 1;
        }
    }
    if current_count > 1 {
        let t = current_count as f64;
        tie_correction += t * (t - 1.0) * (2.0 * t + 5.0);
    }

    // Step 3: Variance with tie correction
    let nf = n as f64;
    let variance = (nf * (nf - 1.0) * (2.0 * nf + 5.0) - tie_correction) / 18.0;

    if variance < 1e-300 {
        return None; // All values identical
    }

    // Step 4: Z with continuity correction
    let sigma = variance.sqrt();
    let z_statistic = if s > 0 {
        (s as f64 - 1.0) / sigma
    } else if s < 0 {
        (s as f64 + 1.0) / sigma
    } else {
        0.0
    };

    // Step 5: Two-tailed p-value
    let p_value = 2.0 * (1.0 - special::standard_normal_cdf(z_statistic.abs()));
    let p_value = p_value.clamp(0.0, 1.0);

    // Step 6: Kendall's tau
    let kendall_tau = (2 * s) as f64 / (nf * (nf - 1.0));

    // Step 7: Sen's slope = median of pairwise slopes
    let mut slopes = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n - 1 {
        for j in (i + 1)..n {
            let dx = (j - i) as f64;
            slopes.push((data[j] - data[i]) / dx);
        }
    }
    slopes.sort_by(|a, b| a.partial_cmp(b).expect("finite values"));
    let m = slopes.len();
    let sen_slope = if m % 2 == 0 {
        (slopes[m / 2 - 1] + slopes[m / 2]) / 2.0
    } else {
        slopes[m / 2]
    };

    Some(MannKendallResult {
        s_statistic: s,
        variance,
        z_statistic,
        p_value,
        kendall_tau,
        sen_slope,
    })
}

/// Natural log of n! using Stirling/ln_gamma for large values.
fn ln_factorial(n: u64) -> f64 {
    if n <= 1 {
        return 0.0;
    }
    // ln(n!) = ln_gamma(n+1)
    special::ln_gamma(n as f64 + 1.0)
}

// ---------------------------------------------------------------------------
// Augmented Dickey-Fuller test
// ---------------------------------------------------------------------------

/// Result of the Augmented Dickey-Fuller (ADF) unit root test.
#[derive(Debug, Clone)]
pub struct AdfResult {
    /// ADF test statistic (t-ratio for γ̂).
    pub statistic: f64,
    /// Number of lags used.
    pub n_lags: usize,
    /// Number of observations used in the regression.
    pub n_obs: usize,
    /// Critical values at 1%, 5%, 10% significance levels.
    pub critical_values: [f64; 3],
    /// Whether the null hypothesis (unit root) is rejected at each level.
    pub rejected: [bool; 3],
}

/// Model specification for the ADF test.
#[derive(Debug, Clone, Copy)]
pub enum AdfModel {
    /// No constant, no trend: Δyₜ = γyₜ₋₁ + Σδᵢ·Δyₜ₋ᵢ + εₜ
    None,
    /// Constant only (default): Δyₜ = α + γyₜ₋₁ + Σδᵢ·Δyₜ₋ᵢ + εₜ
    Constant,
    /// Constant + linear trend: Δyₜ = α + βt + γyₜ₋₁ + Σδᵢ·Δyₜ₋ᵢ + εₜ
    ConstantTrend,
}

/// Augmented Dickey-Fuller (ADF) unit root test for stationarity.
///
/// Tests H₀: unit root (non-stationary) vs H₁: stationary.
///
/// # Algorithm
///
/// 1. Constructs Δyₜ = α + γyₜ₋₁ + Σδᵢ·Δyₜ₋ᵢ + εₜ
/// 2. Estimates via OLS
/// 3. Tests t-ratio for γ against Dickey-Fuller critical values
///
/// When `max_lags` is `None`, lag length is selected by AIC (Schwert rule
/// for maximum). When `Some(p)`, exactly `p` lags are used.
///
/// Reference: Dickey & Fuller (1979), "Distribution of the Estimators for
/// Autoregressive Time Series with a Unit Root"
///
/// # Returns
///
/// `None` if fewer than 10 data points, non-finite values, or OLS fails.
///
/// # Examples
///
/// ```
/// use u_analytics::testing::{adf_test, AdfModel};
///
/// // Stationary series: strong mean-reversion
/// let mut data = vec![0.0_f64; 40];
/// for i in 1..40 {
///     data[i] = 0.3 * data[i - 1] + [0.5, -0.8, 0.3, -0.6, 0.9,
///         -0.4, 0.7, -0.2, 0.1, -0.5][i % 10];
/// }
/// let r = adf_test(&data, AdfModel::Constant, None).unwrap();
/// assert!(r.statistic.is_finite());
/// assert_eq!(r.critical_values.len(), 3);
/// ```
pub fn adf_test(data: &[f64], model: AdfModel, max_lags: Option<usize>) -> Option<AdfResult> {
    let n = data.len();
    if n < 10 || data.iter().any(|v| !v.is_finite()) {
        return None;
    }

    // Compute differences
    let dy: Vec<f64> = data.windows(2).map(|w| w[1] - w[0]).collect();

    // Determine lag count
    let best_lag = match max_lags {
        Some(p) => p, // Use exact lag count when specified
        None => {
            // Schwert (1989) rule for maximum lag
            let schwert = (12.0 * (n as f64 / 100.0).powf(0.25)).floor() as usize;
            let p_max = schwert.min(n / 3);
            // Select optimal lag by AIC
            select_adf_lag(data, &dy, model, p_max)
        }
    };

    // Run OLS regression with selected lag
    adf_ols(data, &dy, model, best_lag)
}

/// Selects optimal lag for ADF by minimizing AIC.
fn select_adf_lag(data: &[f64], dy: &[f64], model: AdfModel, p_max: usize) -> usize {
    let mut best_aic = f64::INFINITY;
    let mut best_p = 0;

    for p in 0..=p_max {
        if let Some((aic, _)) = adf_ols_aic(data, dy, model, p) {
            if aic < best_aic {
                best_aic = aic;
                best_p = p;
            }
        }
    }

    best_p
}

/// Builds the ADF design matrix and dependent variable.
///
/// Returns (design_matrix_row_major, y_dep, n_rows, n_cols, gamma_col_index).
#[allow(clippy::type_complexity)]
fn adf_build_matrix(
    data: &[f64],
    dy: &[f64],
    model: AdfModel,
    p: usize,
) -> Option<(Vec<f64>, Vec<f64>, usize, usize, usize)> {
    let start = p + 1;
    if start >= dy.len() || dy.len() - start < 5 {
        return None;
    }
    let m = dy.len() - start;

    let y_dep: Vec<f64> = dy[start..].to_vec();

    // Count columns: intercept + y_{t-1} + [trend] + p lags
    let has_intercept = !matches!(model, AdfModel::None);
    let has_trend = matches!(model, AdfModel::ConstantTrend);
    let ncols = has_intercept as usize + 1 + has_trend as usize + p;

    // Build row-major design matrix
    let mut x_data = Vec::with_capacity(m * ncols);
    let mut gamma_col = 0;

    for i in 0..m {
        let t = start + i;
        if has_intercept {
            x_data.push(1.0); // intercept
            gamma_col = 1;
        }
        x_data.push(data[t]); // y_{t-1}
        if has_trend {
            x_data.push((t + 1) as f64); // trend
        }
        for lag in 1..=p {
            x_data.push(dy[t - lag]);
        }
    }

    Some((x_data, y_dep, m, ncols, gamma_col))
}

/// Lightweight OLS for ADF: returns (gamma_t_stat, rss, k).
///
/// Solves X'Xβ = X'y using Cholesky-like decomposition (Gaussian elimination).
fn adf_ols_core(
    x_data: &[f64],
    y: &[f64],
    m: usize,
    ncols: usize,
    gamma_col: usize,
) -> Option<(f64, f64, usize)> {
    // Compute X'X (ncols × ncols, symmetric)
    let mut xtx = vec![0.0_f64; ncols * ncols];
    for i in 0..m {
        let row = &x_data[i * ncols..(i + 1) * ncols];
        for j in 0..ncols {
            for k in j..ncols {
                xtx[j * ncols + k] += row[j] * row[k];
            }
        }
    }
    // Mirror upper to lower
    for j in 0..ncols {
        for k in (j + 1)..ncols {
            xtx[k * ncols + j] = xtx[j * ncols + k];
        }
    }

    // Compute X'y (ncols × 1)
    let mut xty = vec![0.0_f64; ncols];
    for i in 0..m {
        let row = &x_data[i * ncols..(i + 1) * ncols];
        for j in 0..ncols {
            xty[j] += row[j] * y[i];
        }
    }

    // Solve via Gaussian elimination with partial pivoting
    let mut augmented = vec![0.0_f64; ncols * (ncols + 1)];
    for i in 0..ncols {
        for j in 0..ncols {
            augmented[i * (ncols + 1) + j] = xtx[i * ncols + j];
        }
        augmented[i * (ncols + 1) + ncols] = xty[i];
    }

    for col in 0..ncols {
        // Partial pivoting
        let mut max_row = col;
        let mut max_val = augmented[col * (ncols + 1) + col].abs();
        for row in (col + 1)..ncols {
            let val = augmented[row * (ncols + 1) + col].abs();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }
        if max_val < 1e-15 {
            return None; // Singular
        }
        if max_row != col {
            for j in 0..=ncols {
                let a = col * (ncols + 1) + j;
                let b = max_row * (ncols + 1) + j;
                augmented.swap(a, b);
            }
        }

        let pivot = augmented[col * (ncols + 1) + col];
        for row in (col + 1)..ncols {
            let factor = augmented[row * (ncols + 1) + col] / pivot;
            for j in col..=ncols {
                let above = augmented[col * (ncols + 1) + j];
                augmented[row * (ncols + 1) + j] -= factor * above;
            }
        }
    }

    // Back-substitution
    let mut beta = vec![0.0_f64; ncols];
    for i in (0..ncols).rev() {
        let mut sum = augmented[i * (ncols + 1) + ncols];
        for j in (i + 1)..ncols {
            sum -= augmented[i * (ncols + 1) + j] * beta[j];
        }
        beta[i] = sum / augmented[i * (ncols + 1) + i];
    }

    // Compute residuals and RSS
    let mut rss = 0.0;
    for i in 0..m {
        let row = &x_data[i * ncols..(i + 1) * ncols];
        let y_hat: f64 = row.iter().zip(beta.iter()).map(|(&x, &b)| x * b).sum();
        let resid = y[i] - y_hat;
        rss += resid * resid;
    }

    // Standard error of coefficients
    let df = m - ncols;
    if df == 0 {
        return None;
    }
    let mse = rss / df as f64;

    // Compute (X'X)^{-1} via Gauss-Jordan elimination to get variance of γ̂
    let mut xtx_aug = vec![0.0_f64; ncols * ncols * 2]; // xtx | I
    for i in 0..ncols {
        for j in 0..ncols {
            xtx_aug[i * 2 * ncols + j] = xtx[i * ncols + j];
        }
        xtx_aug[i * 2 * ncols + ncols + i] = 1.0;
    }

    // Gauss-Jordan elimination
    for col in 0..ncols {
        let mut max_row = col;
        let mut max_val = xtx_aug[col * 2 * ncols + col].abs();
        for row in (col + 1)..ncols {
            let val = xtx_aug[row * 2 * ncols + col].abs();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }
        if max_val < 1e-15 {
            return None;
        }
        if max_row != col {
            for j in 0..(2 * ncols) {
                let a = col * 2 * ncols + j;
                let b = max_row * 2 * ncols + j;
                xtx_aug.swap(a, b);
            }
        }

        let pivot = xtx_aug[col * 2 * ncols + col];
        for j in 0..(2 * ncols) {
            xtx_aug[col * 2 * ncols + j] /= pivot;
        }
        for row in 0..ncols {
            if row == col {
                continue;
            }
            let factor = xtx_aug[row * 2 * ncols + col];
            for j in 0..(2 * ncols) {
                let above = xtx_aug[col * 2 * ncols + j];
                xtx_aug[row * 2 * ncols + j] -= factor * above;
            }
        }
    }

    // Extract diagonal element for gamma column
    let var_gamma = mse * xtx_aug[gamma_col * 2 * ncols + ncols + gamma_col];
    if var_gamma <= 0.0 {
        return None;
    }
    let se_gamma = var_gamma.sqrt();
    let t_gamma = beta[gamma_col] / se_gamma;

    Some((t_gamma, rss, ncols))
}

/// Runs ADF OLS and returns AIC + number of observations.
fn adf_ols_aic(data: &[f64], dy: &[f64], model: AdfModel, p: usize) -> Option<(f64, usize)> {
    let (x_data, y_dep, m, ncols, gamma_col) = adf_build_matrix(data, dy, model, p)?;
    let (_t_stat, rss, k) = adf_ols_core(&x_data, &y_dep, m, ncols, gamma_col)?;
    let aic = 2.0 * k as f64 + m as f64 * (rss / m as f64).ln();
    Some((aic, m))
}

/// Runs the actual ADF OLS and returns the test result.
fn adf_ols(data: &[f64], dy: &[f64], model: AdfModel, p: usize) -> Option<AdfResult> {
    let (x_data, y_dep, m, ncols, gamma_col) = adf_build_matrix(data, dy, model, p)?;
    let (gamma_t, _rss, _k) = adf_ols_core(&x_data, &y_dep, m, ncols, gamma_col)?;

    let critical_values = adf_critical_values(model, m);

    let rejected = [
        gamma_t <= critical_values[0],
        gamma_t <= critical_values[1],
        gamma_t <= critical_values[2],
    ];

    Some(AdfResult {
        statistic: gamma_t,
        n_lags: p,
        n_obs: m,
        critical_values,
        rejected,
    })
}

/// MacKinnon (1994) critical values for ADF test.
///
/// Returns [1%, 5%, 10%] critical values based on sample size.
fn adf_critical_values(model: AdfModel, n: usize) -> [f64; 3] {
    // MacKinnon (1994) regression-based approximation:
    // cv(n) = τ_∞ + τ₁/n + τ₂/n²
    //
    // Coefficients from MacKinnon (2010), Table 1.
    let (tau_inf, tau1, tau2): ([f64; 3], [f64; 3], [f64; 3]) = match model {
        AdfModel::None => (
            [-2.5658, -1.9393, -1.6156],
            [-1.960, -0.398, -0.181],
            [-10.04, 0.0, 0.0],
        ),
        AdfModel::Constant => (
            [-3.4336, -2.8621, -2.5671],
            [-5.999, -2.738, -1.438],
            [-29.25, -8.36, -4.48],
        ),
        AdfModel::ConstantTrend => (
            [-3.9638, -3.4126, -3.1279],
            [-8.353, -4.039, -2.418],
            [-47.44, -17.83, -7.58],
        ),
    };

    let nf = n as f64;
    let inv_n = 1.0 / nf;
    let inv_n2 = inv_n * inv_n;

    [
        tau_inf[0] + tau1[0] * inv_n + tau2[0] * inv_n2,
        tau_inf[1] + tau1[1] * inv_n + tau2[1] * inv_n2,
        tau_inf[2] + tau1[2] * inv_n + tau2[2] * inv_n2,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // One-sample t-test
    // -----------------------------------------------------------------------

    #[test]
    fn one_sample_null_true() {
        let data = [5.0, 5.1, 4.9, 5.0, 5.1, 4.9, 5.0, 5.0];
        let r = one_sample_t_test(&data, 5.0).expect("should compute");
        assert!(r.p_value > 0.3, "p = {}", r.p_value);
    }

    #[test]
    fn one_sample_null_false() {
        let data = [5.0, 5.1, 4.9, 5.0, 5.1, 4.9, 5.0, 5.0];
        let r = one_sample_t_test(&data, 10.0).expect("should compute");
        assert!(r.p_value < 0.001, "p = {}", r.p_value);
    }

    #[test]
    fn one_sample_edge_cases() {
        assert!(one_sample_t_test(&[1.0], 0.0).is_none()); // n < 2
        assert!(one_sample_t_test(&[5.0, 5.0, 5.0], 5.0).is_none()); // zero var
        assert!(one_sample_t_test(&[1.0, f64::NAN, 3.0], 2.0).is_none());
    }

    // -----------------------------------------------------------------------
    // Two-sample t-test
    // -----------------------------------------------------------------------

    #[test]
    fn two_sample_same_mean() {
        let a = [5.0, 5.1, 4.9, 5.0, 5.1, 4.9, 5.0, 5.0];
        let b = [5.0, 5.2, 4.8, 5.1, 4.9, 5.0, 5.1, 4.9];
        let r = two_sample_t_test(&a, &b).expect("should compute");
        assert!(r.p_value > 0.3, "p = {}", r.p_value);
    }

    #[test]
    fn two_sample_different_means() {
        let a = [1.0, 2.0, 3.0, 2.0, 1.5, 2.5];
        let b = [10.0, 11.0, 12.0, 10.5, 11.5, 10.5];
        let r = two_sample_t_test(&a, &b).expect("should compute");
        assert!(r.p_value < 0.001, "p = {}", r.p_value);
    }

    #[test]
    fn two_sample_different_sizes() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0, 7.0, 8.0];
        let r = two_sample_t_test(&a, &b).expect("should compute");
        assert!(r.p_value < 0.05);
    }

    #[test]
    fn two_sample_edge_cases() {
        assert!(two_sample_t_test(&[1.0], &[2.0, 3.0]).is_none());
        assert!(two_sample_t_test(&[1.0, 2.0], &[3.0]).is_none());
    }

    // -----------------------------------------------------------------------
    // Paired t-test
    // -----------------------------------------------------------------------

    #[test]
    fn paired_no_difference() {
        let x = [5.0, 6.0, 7.0, 8.0, 9.0];
        let y = [5.1, 5.9, 7.1, 7.9, 9.1];
        let r = paired_t_test(&x, &y).expect("should compute");
        assert!(r.p_value > 0.3, "p = {}", r.p_value);
    }

    #[test]
    fn paired_significant_difference() {
        // Differences have non-zero variance
        let before = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let after = [6.2, 7.1, 8.3, 9.0, 10.4, 11.1, 12.2, 13.3];
        let r = paired_t_test(&before, &after).expect("should compute");
        assert!(r.p_value < 0.001, "p = {}", r.p_value);
        assert!(r.statistic < 0.0); // after > before
    }

    #[test]
    fn paired_edge_cases() {
        assert!(paired_t_test(&[1.0, 2.0], &[3.0]).is_none()); // length mismatch
        assert!(paired_t_test(&[1.0], &[2.0]).is_none()); // n < 2
    }

    // -----------------------------------------------------------------------
    // ANOVA
    // -----------------------------------------------------------------------

    #[test]
    fn anova_same_means() {
        let g1 = [5.0, 5.1, 4.9, 5.0, 5.1];
        let g2 = [5.0, 5.2, 4.8, 5.1, 4.9];
        let g3 = [5.1, 4.9, 5.0, 5.0, 5.1];
        let r = one_way_anova(&[&g1, &g2, &g3]).expect("should compute");
        assert!(r.p_value > 0.3, "p = {}", r.p_value);
    }

    #[test]
    fn anova_different_means() {
        let g1 = [1.0, 2.0, 3.0, 2.0, 1.5];
        let g2 = [5.0, 6.0, 7.0, 6.0, 5.5];
        let g3 = [10.0, 11.0, 12.0, 11.0, 10.5];
        let r = one_way_anova(&[&g1, &g2, &g3]).expect("should compute");
        assert!(r.p_value < 0.001, "p = {}", r.p_value);
        assert!(r.df_between == 2);
        assert!(r.df_within == 12);
    }

    #[test]
    fn anova_ss_decomposition() {
        let g1 = [1.0, 2.0, 3.0, 2.0, 1.5];
        let g2 = [5.0, 6.0, 7.0, 6.0, 5.5];
        let r = one_way_anova(&[&g1, &g2]).expect("should compute");
        // SS_total = SS_between + SS_within
        let all_data: Vec<f64> = g1.iter().chain(g2.iter()).copied().collect();
        let ss_total: f64 = all_data.iter().map(|&x| (x - r.grand_mean).powi(2)).sum();
        assert!(
            (ss_total - (r.ss_between + r.ss_within)).abs() < 1e-10,
            "SS decomposition: {ss_total} vs {} + {}",
            r.ss_between,
            r.ss_within
        );
    }

    #[test]
    fn anova_edge_cases() {
        let g1 = [1.0, 2.0, 3.0];
        assert!(one_way_anova(&[&g1]).is_none()); // < 2 groups
    }

    // -----------------------------------------------------------------------
    // Chi-squared goodness of fit
    // -----------------------------------------------------------------------

    #[test]
    fn chi2_gof_uniform() {
        // Perfect uniform distribution
        let observed = [25.0, 25.0, 25.0, 25.0];
        let expected = [25.0, 25.0, 25.0, 25.0];
        let r = chi_squared_goodness_of_fit(&observed, &expected).expect("should compute");
        assert!((r.statistic).abs() < 1e-15);
        assert!((r.p_value - 1.0).abs() < 0.01);
    }

    #[test]
    fn chi2_gof_significant() {
        let observed = [90.0, 10.0];
        let expected = [50.0, 50.0];
        let r = chi_squared_goodness_of_fit(&observed, &expected).expect("should compute");
        assert!(r.p_value < 0.001, "p = {}", r.p_value);
    }

    #[test]
    fn chi2_gof_edge_cases() {
        assert!(chi_squared_goodness_of_fit(&[10.0], &[10.0]).is_none()); // < 2
        assert!(chi_squared_goodness_of_fit(&[10.0, 20.0], &[10.0, 0.0]).is_none()); // expected 0
        assert!(chi_squared_goodness_of_fit(&[10.0, 20.0], &[10.0]).is_none()); // mismatch
    }

    // -----------------------------------------------------------------------
    // Chi-squared independence
    // -----------------------------------------------------------------------

    #[test]
    fn chi2_independence_significant() {
        // Strong association
        let table = [30.0, 10.0, 10.0, 50.0];
        let r = chi_squared_independence(&table, 2, 2).expect("should compute");
        assert!(r.p_value < 0.001, "p = {}", r.p_value);
        assert!((r.df - 1.0).abs() < 1e-10);
    }

    #[test]
    fn chi2_independence_not_significant() {
        // No association
        let table = [25.0, 25.0, 25.0, 25.0];
        let r = chi_squared_independence(&table, 2, 2).expect("should compute");
        assert!(r.p_value > 0.3, "p = {}", r.p_value);
    }

    #[test]
    fn chi2_independence_3x3() {
        let table = [10.0, 20.0, 30.0, 40.0, 30.0, 20.0, 20.0, 25.0, 25.0];
        let r = chi_squared_independence(&table, 3, 3).expect("should compute");
        assert!((r.df - 4.0).abs() < 1e-10);
        assert!(r.p_value < 0.05);
    }

    #[test]
    fn chi2_independence_edge_cases() {
        assert!(chi_squared_independence(&[10.0, 20.0], 1, 2).is_none()); // 1 row
        assert!(chi_squared_independence(&[10.0, 20.0], 2, 1).is_none()); // 1 col
        assert!(chi_squared_independence(&[10.0], 2, 2).is_none()); // wrong size
    }

    // -----------------------------------------------------------------------
    // Jarque-Bera
    // -----------------------------------------------------------------------

    #[test]
    fn jb_normal_data() {
        // Symmetric, light-tailed data
        let data = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.0, 0.5, 1.0, 1.5, 2.0];
        let r = jarque_bera_test(&data).expect("should compute");
        assert!(r.p_value > 0.05, "p = {}", r.p_value);
    }

    #[test]
    fn jb_skewed_data() {
        // Highly right-skewed
        let data = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10.0, 20.0, 50.0];
        let r = jarque_bera_test(&data).expect("should compute");
        assert!(r.p_value < 0.05, "p = {}", r.p_value);
    }

    #[test]
    fn jb_edge_cases() {
        assert!(jarque_bera_test(&[1.0, 2.0, 3.0, 4.0]).is_none()); // n < 8
    }

    // -----------------------------------------------------------------------
    // Anderson-Darling
    // -----------------------------------------------------------------------

    #[test]
    fn ad_normal_data() {
        let data = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.0, 0.5, 1.0, 1.5, 2.0];
        let r = anderson_darling_test(&data).expect("should compute");
        assert!(r.p_value > 0.05, "p = {}", r.p_value);
        assert!(r.statistic > 0.0, "A2 = {}", r.statistic);
        assert!(r.statistic_star > r.statistic, "A*2 should be > A2");
    }

    #[test]
    fn ad_skewed_data() {
        // Exponential-like data — not normal
        let data = [0.1, 0.2, 0.3, 0.5, 0.8, 1.3, 2.1, 3.4, 5.5, 8.9, 14.4, 23.3];
        let r = anderson_darling_test(&data).expect("should compute");
        assert!(
            r.p_value < 0.05,
            "p = {} (should reject normality)",
            r.p_value
        );
    }

    #[test]
    fn ad_bimodal_data() {
        // Bimodal data — clearly not normal
        let mut data = vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5];
        data.extend_from_slice(&[9.5, 9.6, 9.7, 9.8, 9.9, 10.0]);
        let r = anderson_darling_test(&data).expect("should compute");
        assert!(
            r.p_value < 0.01,
            "p = {} (bimodal should reject normality)",
            r.p_value
        );
    }

    #[test]
    fn ad_large_normal_sample() {
        let n = 100;
        let data: Vec<f64> = (1..=n)
            .map(|i| {
                let p = (i as f64 - 0.5) / n as f64;
                special::inverse_normal_cdf(p)
            })
            .collect();
        let r = anderson_darling_test(&data).expect("should compute");
        assert!(r.p_value > 0.05, "p = {} for normal quantiles", r.p_value);
    }

    #[test]
    fn ad_edge_cases() {
        assert!(anderson_darling_test(&[1.0, 2.0, 3.0, 4.0]).is_none()); // n < 8
        assert!(anderson_darling_test(&[5.0; 10]).is_none()); // constant
        assert!(anderson_darling_test(&[1.0, f64::NAN, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).is_none());
    }

    #[test]
    fn ad_p_value_ranges() {
        // Test different A*² ranges for p-value formula
        // Use datasets that produce different A*² magnitudes

        // Near-normal → small A*² (< 0.2 range)
        let near_normal = [-1.5, -1.0, -0.5, 0.0, 0.0, 0.5, 1.0, 1.5];
        let r = anderson_darling_test(&near_normal).expect("should compute");
        assert!(r.p_value >= 0.0 && r.p_value <= 1.0);

        // Heavy-tailed → large A*² (>= 0.6 range)
        let heavy_tail = [0.1, 0.2, 0.3, 0.5, 0.8, 1.3, 2.1, 3.4, 5.5, 8.9, 14.4, 23.3];
        let r = anderson_darling_test(&heavy_tail).expect("should compute");
        assert!(r.p_value >= 0.0 && r.p_value <= 1.0);
    }

    // -----------------------------------------------------------------------
    // Shapiro-Wilk
    // -----------------------------------------------------------------------

    #[test]
    fn sw_normal_data() {
        // Approximately normal data (symmetric, bell-shaped)
        let data = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.0, 0.5, 1.0, 1.5, 2.0];
        let r = shapiro_wilk_test(&data).expect("should compute");
        assert!(r.w > 0.9, "W = {}", r.w);
        assert!(r.p_value > 0.05, "p = {}", r.p_value);
    }

    #[test]
    fn sw_bimodal_data() {
        // Bimodal data — clearly not normal
        let mut data = vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5];
        data.extend_from_slice(&[9.5, 9.6, 9.7, 9.8, 9.9, 10.0]);
        let r = shapiro_wilk_test(&data).expect("should compute");
        assert!(
            r.p_value < 0.01,
            "p = {} (bimodal should reject normality)",
            r.p_value
        );
    }

    #[test]
    fn sw_n3() {
        let data = [1.0, 2.0, 3.0];
        let r = shapiro_wilk_test(&data).expect("n=3 should work");
        assert!(r.w > 0.0 && r.w <= 1.0, "W = {}", r.w);
        assert!(r.p_value >= 0.0 && r.p_value <= 1.0, "p = {}", r.p_value);
    }

    #[test]
    fn sw_n4() {
        let data = [1.0, 2.0, 3.0, 4.0];
        let r = shapiro_wilk_test(&data).expect("n=4 should work");
        assert!(r.w > 0.0 && r.w <= 1.0, "W = {}", r.w);
        assert!(r.p_value >= 0.0 && r.p_value <= 1.0, "p = {}", r.p_value);
    }

    #[test]
    fn sw_n5() {
        let data = [-1.0, -0.5, 0.0, 0.5, 1.0];
        let r = shapiro_wilk_test(&data).expect("n=5 should work");
        assert!(r.w > 0.9, "W = {}", r.w);
        assert!(r.p_value > 0.05, "p = {}", r.p_value);
    }

    #[test]
    fn sw_skewed_data() {
        // Exponential-like data — not normal
        let data = [0.1, 0.2, 0.3, 0.5, 0.8, 1.3, 2.1, 3.4, 5.5, 8.9, 14.4, 23.3];
        let r = shapiro_wilk_test(&data).expect("should compute");
        assert!(
            r.p_value < 0.05,
            "p = {} (skewed data should reject normality)",
            r.p_value
        );
    }

    #[test]
    fn sw_large_normal_sample() {
        // Generate pseudo-normal data via Box-Muller-like approach
        // Use linearly spaced quantiles from standard normal
        let n = 100;
        let data: Vec<f64> = (1..=n)
            .map(|i| {
                let p = (i as f64 - 0.5) / n as f64;
                special::inverse_normal_cdf(p)
            })
            .collect();
        let r = shapiro_wilk_test(&data).expect("should compute");
        assert!(r.w > 0.99, "W = {} for normal quantiles", r.w);
        assert!(r.p_value > 0.05, "p = {}", r.p_value);
    }

    #[test]
    fn sw_w_bounded() {
        // W should be in (0, 1] for any valid data
        let datasets: Vec<Vec<f64>> = vec![
            vec![1.0, 2.0, 3.0],
            vec![1.0, 1.0, 2.0, 3.0, 3.0],
            vec![0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0],
            (0..20).map(|i| (i as f64).powi(2)).collect(),
        ];
        for (idx, data) in datasets.iter().enumerate() {
            let r = shapiro_wilk_test(data).unwrap_or_else(|| panic!("dataset {idx} should work"));
            assert!(r.w > 0.0 && r.w <= 1.0, "dataset {idx}: W = {}", r.w);
            assert!(
                r.p_value >= 0.0 && r.p_value <= 1.0,
                "dataset {idx}: p = {}",
                r.p_value
            );
        }
    }

    #[test]
    fn sw_edge_cases() {
        assert!(shapiro_wilk_test(&[1.0, 2.0]).is_none()); // n < 3
        assert!(shapiro_wilk_test(&[]).is_none()); // empty
        assert!(shapiro_wilk_test(&[5.0, 5.0, 5.0]).is_none()); // constant
        assert!(shapiro_wilk_test(&[1.0, f64::NAN, 3.0]).is_none()); // NaN
        assert!(shapiro_wilk_test(&[1.0, f64::INFINITY, 3.0]).is_none()); // Inf
    }

    #[test]
    fn sw_n5001_rejected() {
        let data: Vec<f64> = (0..5001).map(|i| i as f64).collect();
        assert!(shapiro_wilk_test(&data).is_none()); // n > 5000
    }

    // -----------------------------------------------------------------------
    // Mann-Whitney U
    // -----------------------------------------------------------------------

    #[test]
    fn mw_clearly_different() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0];
        let b = [6.0, 7.0, 8.0, 9.0, 10.0];
        let r = mann_whitney_u_test(&a, &b).expect("should compute");
        assert!(r.p_value < 0.05, "p = {}", r.p_value);
    }

    #[test]
    fn mw_same_distribution() {
        let a = [1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0];
        let b = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0];
        let r = mann_whitney_u_test(&a, &b).expect("should compute");
        assert!(
            r.p_value > 0.3,
            "p = {} (interleaved, same dist)",
            r.p_value
        );
    }

    #[test]
    fn mw_with_ties() {
        let a = [1.0, 2.0, 2.0, 3.0, 3.0];
        let b = [3.0, 4.0, 4.0, 5.0, 5.0];
        let r = mann_whitney_u_test(&a, &b).expect("should compute");
        assert!(r.p_value < 0.05, "p = {} (shifted with ties)", r.p_value);
    }

    #[test]
    fn mw_different_sizes() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0, 7.0, 8.0];
        let r = mann_whitney_u_test(&a, &b).expect("should compute");
        assert!(r.p_value < 0.05);
    }

    #[test]
    fn mw_edge_cases() {
        assert!(mann_whitney_u_test(&[1.0], &[2.0, 3.0]).is_none()); // n1 < 2
        assert!(mann_whitney_u_test(&[1.0, 2.0], &[3.0]).is_none()); // n2 < 2
        assert!(mann_whitney_u_test(&[1.0, f64::NAN], &[2.0, 3.0]).is_none());
    }

    // -----------------------------------------------------------------------
    // Wilcoxon signed-rank
    // -----------------------------------------------------------------------

    #[test]
    fn wilcoxon_significant_increase() {
        let before = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let after = [6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5];
        let r = wilcoxon_signed_rank_test(&before, &after).expect("should compute");
        assert!(r.p_value < 0.05, "p = {} (consistent increase)", r.p_value);
    }

    #[test]
    fn wilcoxon_no_difference() {
        let x = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let y = [5.1, 5.9, 7.1, 7.9, 9.1, 9.9];
        let r = wilcoxon_signed_rank_test(&x, &y).expect("should compute");
        assert!(r.p_value > 0.3, "p = {} (small random diffs)", r.p_value);
    }

    #[test]
    fn wilcoxon_with_ties() {
        // Some differences are equal in magnitude
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [0.0, 1.0, 2.0, 3.0, 4.0]; // constant difference = 1.0
        let r = wilcoxon_signed_rank_test(&x, &y).expect("should compute");
        // All differences positive with ties in magnitude
        assert!(r.statistic > 0.0);
    }

    #[test]
    fn wilcoxon_with_zero_diffs() {
        // Some pairs are equal → zero differences discarded
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [1.0, 2.0, 3.0, 3.0, 4.0]; // first 3 are zero diffs
        let r = wilcoxon_signed_rank_test(&x, &y).expect("should compute");
        assert!(r.p_value >= 0.0 && r.p_value <= 1.0);
    }

    #[test]
    fn wilcoxon_edge_cases() {
        assert!(wilcoxon_signed_rank_test(&[1.0, 2.0], &[3.0]).is_none()); // mismatch
        assert!(wilcoxon_signed_rank_test(&[1.0], &[2.0]).is_none()); // n < 2
                                                                      // All zero differences → fewer than 2 non-zero diffs
        assert!(wilcoxon_signed_rank_test(&[5.0, 5.0], &[5.0, 5.0]).is_none());
    }

    // -----------------------------------------------------------------------
    // Kruskal-Wallis
    // -----------------------------------------------------------------------

    #[test]
    fn kw_clearly_different() {
        let g1 = [1.0, 2.0, 3.0, 4.0, 5.0];
        let g2 = [6.0, 7.0, 8.0, 9.0, 10.0];
        let g3 = [11.0, 12.0, 13.0, 14.0, 15.0];
        let r = kruskal_wallis_test(&[&g1, &g2, &g3]).expect("should compute");
        assert!(r.p_value < 0.01, "p = {}", r.p_value);
        assert!((r.df - 2.0).abs() < 1e-10);
    }

    #[test]
    fn kw_same_distribution() {
        let g1 = [1.0, 3.0, 5.0, 7.0, 9.0];
        let g2 = [2.0, 4.0, 6.0, 8.0, 10.0];
        let r = kruskal_wallis_test(&[&g1, &g2]).expect("should compute");
        assert!(r.p_value > 0.3, "p = {} (interleaved)", r.p_value);
    }

    #[test]
    fn kw_with_ties() {
        let g1 = [1.0, 2.0, 2.0, 3.0];
        let g2 = [3.0, 4.0, 4.0, 5.0];
        let g3 = [5.0, 6.0, 6.0, 7.0];
        let r = kruskal_wallis_test(&[&g1, &g2, &g3]).expect("should compute");
        assert!(r.statistic > 0.0);
    }

    #[test]
    fn kw_edge_cases() {
        let g1 = [1.0, 2.0, 3.0];
        assert!(kruskal_wallis_test(&[&g1]).is_none()); // < 2 groups
    }

    // -----------------------------------------------------------------------
    // Levene (Brown-Forsythe)
    // -----------------------------------------------------------------------

    #[test]
    fn levene_equal_variance() {
        let g1 = [1.0, 2.0, 3.0, 4.0, 5.0];
        let g2 = [6.0, 7.0, 8.0, 9.0, 10.0];
        let r = levene_test(&[&g1, &g2]).expect("should compute");
        // Both have same spread, different means → equal variance
        assert!(r.p_value > 0.3, "p = {} (equal variance)", r.p_value);
    }

    #[test]
    fn levene_unequal_variance() {
        let g1 = [4.5, 4.8, 5.0, 5.2, 5.5]; // small spread
        let g2 = [0.0, 2.0, 5.0, 8.0, 10.0]; // large spread
        let r = levene_test(&[&g1, &g2]).expect("should compute");
        assert!(r.p_value < 0.05, "p = {} (unequal variance)", r.p_value);
    }

    #[test]
    fn levene_three_groups() {
        let g1 = [1.0, 2.0, 3.0, 4.0, 5.0];
        let g2 = [0.0, 3.0, 5.0, 7.0, 10.0];
        let g3 = [-5.0, 0.0, 5.0, 10.0, 15.0];
        let r = levene_test(&[&g1, &g2, &g3]).expect("should compute");
        assert!(r.df >= 2.0);
    }

    #[test]
    fn levene_edge_cases() {
        let g1 = [1.0, 2.0, 3.0];
        assert!(levene_test(&[&g1]).is_none()); // < 2 groups
    }

    // -----------------------------------------------------------------------
    // Multiple comparison correction
    // -----------------------------------------------------------------------

    #[test]
    fn bonferroni_basic() {
        let ps = [0.01, 0.04, 0.03, 0.005];
        let adj = bonferroni_correction(&ps).expect("should compute");
        assert!((adj[0] - 0.04).abs() < 1e-10);
        assert!((adj[1] - 0.16).abs() < 1e-10);
        assert!((adj[2] - 0.12).abs() < 1e-10);
        assert!((adj[3] - 0.02).abs() < 1e-10);
    }

    #[test]
    fn bonferroni_capped_at_one() {
        let ps = [0.5, 0.6];
        let adj = bonferroni_correction(&ps).expect("should compute");
        assert!((adj[0] - 1.0).abs() < 1e-10);
        assert!((adj[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn bh_basic() {
        let ps = [0.01, 0.04, 0.03, 0.005];
        let adj = benjamini_hochberg(&ps).expect("should compute");
        // All adjusted p-values should be >= original
        for (i, (&orig, &adjusted)) in ps.iter().zip(adj.iter()).enumerate() {
            assert!(
                adjusted >= orig - 1e-15,
                "adj[{i}] = {adjusted} < original {orig}"
            );
        }
        // Adjusted p-values should still be ordered (weakly) by original order
        // after reordering by original p-value
    }

    #[test]
    fn bh_all_significant() {
        let ps = [0.001, 0.002, 0.003];
        let adj = benjamini_hochberg(&ps).expect("should compute");
        for &a in &adj {
            assert!(a < 0.05);
        }
    }

    #[test]
    fn correction_edge_cases() {
        assert!(bonferroni_correction(&[]).is_none());
        assert!(benjamini_hochberg(&[]).is_none());
        assert!(bonferroni_correction(&[f64::NAN]).is_none());
    }

    // -----------------------------------------------------------------------
    // Bartlett test
    // -----------------------------------------------------------------------

    #[test]
    fn bartlett_equal_variances() {
        let g1 = [2.0, 3.0, 4.0, 5.0, 6.0];
        let g2 = [12.0, 13.0, 14.0, 15.0, 16.0];
        let r = bartlett_test(&[&g1, &g2]).expect("should compute");
        assert!(
            r.p_value > 0.9,
            "equal variance → p high, got {}",
            r.p_value
        );
    }

    #[test]
    fn bartlett_unequal_variances() {
        let g1 = [2.0, 3.0, 4.0, 5.0, 6.0]; // var ≈ 2.5
        let g2 = [10.0, 20.0, 30.0, 40.0, 50.0]; // var ≈ 250
        let r = bartlett_test(&[&g1, &g2]).expect("should compute");
        assert!(
            r.p_value < 0.01,
            "very different variances → p < 0.01, got {}",
            r.p_value
        );
        assert!((r.df - 1.0).abs() < 1e-10); // k-1 = 1
    }

    #[test]
    fn bartlett_three_groups() {
        let g1 = [1.0, 2.0, 3.0, 4.0, 5.0];
        let g2 = [1.5, 2.5, 3.5, 4.5, 5.5];
        let g3 = [10.0, 30.0, 50.0, 70.0, 90.0]; // much higher variance
        let r = bartlett_test(&[&g1, &g2, &g3]).expect("should compute");
        assert!(r.p_value < 0.05, "one group with high variance");
        assert!((r.df - 2.0).abs() < 1e-10); // k-1 = 2
    }

    #[test]
    fn bartlett_edge_cases() {
        let g1 = [1.0, 2.0, 3.0];
        assert!(bartlett_test(&[&g1]).is_none()); // < 2 groups

        let g2 = [5.0, 5.0, 5.0]; // zero variance
        assert!(bartlett_test(&[&g1, &g2]).is_none());

        let g3 = [1.0]; // group too small
        assert!(bartlett_test(&[&g1, &g3]).is_none());
    }

    // -----------------------------------------------------------------------
    // Fisher exact test
    // -----------------------------------------------------------------------

    #[test]
    fn fisher_tea_tasting() {
        // Classic Fisher tea-tasting: [[3,1],[1,3]]
        let r = fisher_exact_test(3, 1, 1, 3).expect("should compute");
        // Known two-tailed p ≈ 0.4857
        assert!(
            (r.p_value - 0.4857).abs() < 0.01,
            "p ≈ 0.4857, got {}",
            r.p_value
        );
    }

    #[test]
    fn fisher_significant() {
        // Strong association: [[10, 0], [0, 10]]
        let r = fisher_exact_test(10, 0, 0, 10).expect("should compute");
        assert!(r.p_value < 0.001, "perfect association → p very small");
    }

    #[test]
    fn fisher_no_association() {
        // Proportional table: [[5, 5], [5, 5]]
        let r = fisher_exact_test(5, 5, 5, 5).expect("should compute");
        assert!(
            r.p_value > 0.9,
            "no association → p ≈ 1.0, got {}",
            r.p_value
        );
    }

    #[test]
    fn fisher_small_table() {
        // [[1, 0], [0, 1]]
        let r = fisher_exact_test(1, 0, 0, 1).expect("should compute");
        assert!(r.p_value > 0.0 && r.p_value <= 1.0);
    }

    #[test]
    fn fisher_asymmetric() {
        // [[8, 2], [1, 5]]
        let r = fisher_exact_test(8, 2, 1, 5).expect("should compute");
        assert!(r.p_value < 0.05, "significant association");
    }

    #[test]
    fn fisher_edge_cases() {
        // Zero marginals → None
        assert!(fisher_exact_test(0, 0, 1, 2).is_none()); // row1 = 0
        assert!(fisher_exact_test(1, 2, 0, 0).is_none()); // row2 = 0
        assert!(fisher_exact_test(0, 1, 0, 2).is_none()); // col1 = 0
    }

    #[test]
    fn fisher_odds_ratio() {
        let r = fisher_exact_test(3, 1, 1, 3).expect("should compute");
        // OR = (3*3)/(1*1) = 9
        assert!(
            (r.statistic - 9.0).abs() < 1e-10,
            "OR = 9, got {}",
            r.statistic
        );
    }

    // -----------------------------------------------------------------------
    // Mann-Kendall trend test
    // -----------------------------------------------------------------------

    #[test]
    fn mk_increasing_trend() {
        let data = [1.0, 2.3, 3.1, 4.5, 5.2, 6.8, 7.1, 8.9, 9.5, 10.2];
        let r = mann_kendall_test(&data).expect("should compute");
        assert!(r.p_value < 0.01, "p = {}", r.p_value);
        assert!(r.kendall_tau > 0.8, "tau = {}", r.kendall_tau);
        assert!(r.sen_slope > 0.0, "slope = {}", r.sen_slope);
        assert!(r.s_statistic > 0);
    }

    #[test]
    fn mk_decreasing_trend() {
        let data = [10.0, 9.2, 8.5, 7.1, 6.3, 5.0, 4.2, 3.1, 2.0, 1.1];
        let r = mann_kendall_test(&data).expect("should compute");
        assert!(r.p_value < 0.01, "p = {}", r.p_value);
        assert!(r.kendall_tau < -0.8, "tau = {}", r.kendall_tau);
        assert!(r.sen_slope < 0.0, "slope = {}", r.sen_slope);
        assert!(r.s_statistic < 0);
    }

    #[test]
    fn mk_no_trend() {
        // Random-looking data with no clear trend
        let data = [5.0, 3.0, 7.0, 2.0, 8.0, 4.0, 6.0, 1.0, 9.0, 5.0];
        let r = mann_kendall_test(&data).expect("should compute");
        // Should not detect significant trend
        assert!(
            r.p_value > 0.05,
            "p = {} (should be > 0.05 for no trend)",
            r.p_value
        );
    }

    #[test]
    fn mk_perfect_monotone() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let r = mann_kendall_test(&data).expect("should compute");
        // Perfect monotone: S = n(n-1)/2 = 45, tau = 1.0
        assert_eq!(r.s_statistic, 45);
        assert!((r.kendall_tau - 1.0).abs() < 1e-10);
        assert!((r.sen_slope - 1.0).abs() < 1e-10);
    }

    #[test]
    fn mk_with_ties() {
        let data = [1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 5.0];
        let r = mann_kendall_test(&data).expect("should compute");
        assert!(r.s_statistic > 0);
        // Tie correction should reduce variance
        let n = data.len() as f64;
        let base_var = n * (n - 1.0) * (2.0 * n + 5.0) / 18.0;
        assert!(r.variance < base_var, "ties should reduce variance");
    }

    #[test]
    fn mk_edge_cases() {
        // Too few data points
        assert!(mann_kendall_test(&[1.0, 2.0, 3.0]).is_none());
        // NaN
        assert!(mann_kendall_test(&[1.0, f64::NAN, 3.0, 4.0]).is_none());
        // All identical (zero variance)
        assert!(mann_kendall_test(&[5.0, 5.0, 5.0, 5.0]).is_none());
    }

    #[test]
    fn mk_minimum_n() {
        // n = 4 should work
        let data = [1.0, 2.0, 3.0, 4.0];
        let r = mann_kendall_test(&data).expect("n=4 should work");
        assert_eq!(r.s_statistic, 6); // C(4,2) = 6 pairs, all positive
        assert!((r.kendall_tau - 1.0).abs() < 1e-10);
    }

    #[test]
    fn mk_sen_slope_robust_to_outlier() {
        // Mostly linear (slope ≈ 1) with one outlier
        let data = [1.0, 2.0, 3.0, 100.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let r = mann_kendall_test(&data).expect("should compute");
        // Sen's slope should be close to 1.0 despite the outlier at index 3
        assert!(
            (r.sen_slope - 1.0).abs() < 0.5,
            "sen slope = {}, expected ≈ 1.0",
            r.sen_slope
        );
    }

    // -----------------------------------------------------------------------
    // ADF test
    // -----------------------------------------------------------------------

    #[test]
    fn adf_stationary_mean_reverting() {
        // Strongly mean-reverting AR(1) process: y_t = 0.3*y_{t-1} + noise
        // This is clearly stationary (|ρ| < 1)
        let data = [
            0.5, 0.45, -0.2, 0.14, 0.54, -0.04, 0.39, -0.18, 0.35, 0.01, -0.3, 0.21, 0.47, -0.06,
            0.38, -0.25, 0.12, 0.44, -0.13, 0.36, -0.09, 0.27, 0.51, -0.15, 0.33, -0.22, 0.18,
            0.42, -0.08, 0.31, -0.19, 0.25, 0.48, -0.11, 0.37, -0.24, 0.15, 0.43, -0.07, 0.34,
        ];
        let r = adf_test(&data, AdfModel::Constant, None).expect("should compute");
        // Should reject H₀ (series is stationary)
        assert!(
            r.rejected[2],
            "should reject at 10%: stat={}, cv={}",
            r.statistic, r.critical_values[2]
        );
    }

    #[test]
    fn adf_nonstationary_random_walk() {
        // Cumulative sum simulates random walk (non-stationary)
        let increments = [
            0.1, -0.2, 0.15, -0.05, 0.3, -0.1, 0.2, -0.15, 0.25, -0.08, 0.12, -0.18, 0.22, -0.07,
            0.16, -0.11, 0.19, -0.14, 0.21, -0.09, 0.13, -0.17, 0.24, -0.06, 0.18, -0.12, 0.2,
            -0.13, 0.15, -0.1,
        ];
        let mut walk = Vec::with_capacity(increments.len());
        let mut cum = 0.0;
        for &inc in &increments {
            cum += inc;
            walk.push(cum);
        }
        let r = adf_test(&walk, AdfModel::Constant, None).expect("should compute");
        // Random walk should NOT reject at 1%
        assert!(
            !r.rejected[0],
            "should NOT reject at 1%: stat={}, cv={}",
            r.statistic, r.critical_values[0]
        );
    }

    #[test]
    fn adf_with_fixed_lags() {
        // Use wider oscillation to avoid near-singular design matrix
        let data: Vec<f64> = (0..50)
            .map(|i| (i as f64 * 0.5).sin() + 0.02 * i as f64)
            .collect();
        let r = adf_test(&data, AdfModel::Constant, Some(2)).expect("should compute");
        assert_eq!(r.n_lags, 2);
        assert!(r.statistic.is_finite());
    }

    #[test]
    fn adf_constant_trend_model() {
        // Linear trend is unit-root-like under "constant" model
        // but with "constant+trend" model, it should be recognized
        let data: Vec<f64> = (0..30)
            .map(|i| i as f64 + (i as f64 * 0.3).sin() * 0.5)
            .collect();
        let r = adf_test(&data, AdfModel::ConstantTrend, None).expect("should compute");
        assert!(r.statistic.is_finite());
        assert_eq!(r.critical_values.len(), 3);
    }

    #[test]
    fn adf_edge_cases() {
        // Too few data points
        assert!(adf_test(&[1.0; 9], AdfModel::Constant, None).is_none());
        // NaN
        let mut data = vec![0.0; 20];
        data[5] = f64::NAN;
        assert!(adf_test(&data, AdfModel::Constant, None).is_none());
    }

    #[test]
    fn adf_critical_values_constant() {
        // Verify critical values are reasonable for n=100
        let cv = adf_critical_values(AdfModel::Constant, 100);
        // At n=100: approximately -3.51, -2.89, -2.58
        assert!(cv[0] < -3.4 && cv[0] > -3.6, "1% cv = {}", cv[0]);
        assert!(cv[1] < -2.8 && cv[1] > -3.0, "5% cv = {}", cv[1]);
        assert!(cv[2] < -2.5 && cv[2] > -2.7, "10% cv = {}", cv[2]);
    }

    #[test]
    fn adf_critical_values_ordering() {
        let cv = adf_critical_values(AdfModel::Constant, 50);
        // 1% < 5% < 10% (more negative for stricter levels)
        assert!(cv[0] < cv[1], "1% ({}) should be < 5% ({})", cv[0], cv[1]);
        assert!(cv[1] < cv[2], "5% ({}) should be < 10% ({})", cv[1], cv[2]);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn one_sample_p_bounded(
            data in proptest::collection::vec(-1e3_f64..1e3, 3..=30),
            mu0 in -1e3_f64..1e3
        ) {
            if let Some(r) = one_sample_t_test(&data, mu0) {
                prop_assert!(r.p_value >= 0.0 && r.p_value <= 1.0, "p = {}", r.p_value);
            }
        }

        #[test]
        fn two_sample_p_bounded(
            a in proptest::collection::vec(-1e3_f64..1e3, 3..=20),
            b in proptest::collection::vec(-1e3_f64..1e3, 3..=20),
        ) {
            if let Some(r) = two_sample_t_test(&a, &b) {
                prop_assert!(r.p_value >= 0.0 && r.p_value <= 1.0, "p = {}", r.p_value);
            }
        }

        #[test]
        fn anova_p_bounded(
            g1 in proptest::collection::vec(-1e3_f64..1e3, 3..=15),
            g2 in proptest::collection::vec(-1e3_f64..1e3, 3..=15),
            g3 in proptest::collection::vec(-1e3_f64..1e3, 3..=15),
        ) {
            if let Some(r) = one_way_anova(&[&g1, &g2, &g3]) {
                prop_assert!(r.p_value >= 0.0 && r.p_value <= 1.0, "p = {}", r.p_value);
                prop_assert!(r.f_statistic >= 0.0, "F = {}", r.f_statistic);
            }
        }

        #[test]
        fn bonferroni_monotone(
            ps in proptest::collection::vec(0.001_f64..1.0, 2..=10)
        ) {
            let adj = bonferroni_correction(&ps).expect("should compute");
            for (i, (&orig, &adjusted)) in ps.iter().zip(adj.iter()).enumerate() {
                prop_assert!(adjusted >= orig - 1e-15,
                    "adj[{i}] = {adjusted} < orig = {orig}");
            }
        }

        #[test]
        fn shapiro_wilk_p_bounded(
            data in proptest::collection::vec(-1e3_f64..1e3, 3..=50)
        ) {
            if let Some(r) = shapiro_wilk_test(&data) {
                prop_assert!(r.w > 0.0 && r.w <= 1.0, "W = {}", r.w);
                prop_assert!(r.p_value >= 0.0 && r.p_value <= 1.0, "p = {}", r.p_value);
            }
        }

        #[test]
        fn anderson_darling_p_bounded(
            data in proptest::collection::vec(-1e3_f64..1e3, 8..=100)
        ) {
            if let Some(r) = anderson_darling_test(&data) {
                prop_assert!(r.statistic >= 0.0, "A2 = {}", r.statistic);
                prop_assert!(r.p_value >= 0.0 && r.p_value <= 1.0, "p = {}", r.p_value);
            }
        }

        #[test]
        fn mann_whitney_p_bounded(
            a in proptest::collection::vec(-1e3_f64..1e3, 3..=20),
            b in proptest::collection::vec(-1e3_f64..1e3, 3..=20),
        ) {
            if let Some(r) = mann_whitney_u_test(&a, &b) {
                prop_assert!(r.p_value >= 0.0 && r.p_value <= 1.0, "p = {}", r.p_value);
                prop_assert!(r.statistic >= 0.0, "U = {}", r.statistic);
            }
        }

        #[test]
        fn wilcoxon_p_bounded(
            diffs in proptest::collection::vec(-1e3_f64..1e3, 3..=20),
        ) {
            let zeros: Vec<f64> = vec![0.0; diffs.len()];
            if let Some(r) = wilcoxon_signed_rank_test(&diffs, &zeros) {
                prop_assert!(r.p_value >= 0.0 && r.p_value <= 1.0, "p = {}", r.p_value);
            }
        }

        #[test]
        fn bartlett_p_bounded(
            g1 in proptest::collection::vec(0.1_f64..100.0, 3..=15),
            g2 in proptest::collection::vec(0.1_f64..100.0, 3..=15),
        ) {
            if let Some(r) = bartlett_test(&[&g1, &g2]) {
                prop_assert!(r.p_value >= 0.0 && r.p_value <= 1.0, "p = {}", r.p_value);
                prop_assert!(r.statistic >= 0.0, "T = {}", r.statistic);
            }
        }

        #[test]
        fn fisher_p_bounded(
            a in 0_u64..20,
            b in 0_u64..20,
            c in 0_u64..20,
            d in 0_u64..20,
        ) {
            if let Some(r) = fisher_exact_test(a, b, c, d) {
                prop_assert!(r.p_value >= 0.0 && r.p_value <= 1.0, "p = {}", r.p_value);
            }
        }

        #[test]
        fn mann_kendall_p_bounded(
            data in proptest::collection::vec(-1e3_f64..1e3, 4..=30)
        ) {
            if let Some(r) = mann_kendall_test(&data) {
                prop_assert!(r.p_value >= 0.0 && r.p_value <= 1.0, "p = {}", r.p_value);
                prop_assert!(r.kendall_tau >= -1.0 && r.kendall_tau <= 1.0,
                    "tau = {}", r.kendall_tau);
            }
        }

        #[test]
        fn mann_kendall_monotone_increasing(n in 5_usize..=30) {
            let data: Vec<f64> = (0..n).map(|i| i as f64).collect();
            let r = mann_kendall_test(&data).expect("should compute");
            prop_assert!((r.kendall_tau - 1.0).abs() < 1e-10,
                "tau should be 1.0 for monotone, got {}", r.kendall_tau);
            prop_assert!(r.sen_slope > 0.0, "slope should be positive");
        }
    }
}
