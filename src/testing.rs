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

use crate::error::{check_option, check_series, TimeSeriesError};
use std::ops::RangeInclusive;
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
    let p_value = 2.0 * special::t_distribution_sf(t.abs(), df);

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

    let p_value = 2.0 * special::t_distribution_sf(t.abs(), df);

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
        special::f_distribution_sf(f_statistic, df_between as f64, df_within as f64)
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
    let p_value = special::chi_squared_sf(chi2, df);

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
    let p_value = special::chi_squared_sf(chi2, df);

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
/// ```text
/// JB = (n/6) · [g₁² + g₂²/4],   p = exp(−JB/2)
/// ```
///
/// `g₁` and `g₂` are the **moment** coefficients of skewness and excess
/// kurtosis (`m₃/m₂^(3/2)` and `m₄/m₂² − 3`), which is what Jarque and Bera
/// wrote the statistic in and what SciPy, R and statsmodels compute. The
/// bias-adjusted `G₁`/`G₂` that [`u_numflow::stats::skewness`] and
/// [`u_numflow::stats::kurtosis`] report -- Excel's `SKEW()`/`KURT()` -- are a
/// different pair of estimators, and substituting them moves JB by tens of
/// percent at the sample sizes this test is used at.
///
/// `JB ~ χ²(2)` under H₀, and a χ² with two degrees of freedom is an
/// exponential: its survival function is `exp(−JB/2)` exactly. Taking that
/// instead of `1 − cdf` keeps the p-value to full relative precision in the
/// tail, where `1 − cdf` rounds to zero.
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

    let g1 = stats::skewness_moment(data)?;
    let g2 = stats::kurtosis_moment(data)?;

    let nf = n as f64;
    let jb = (nf / 6.0) * (g1 * g1 + g2 * g2 / 4.0);
    // χ²(2) is Exponential(1/2): sf(x) = exp(−x/2), no cancellation in the tail.
    let p_value = (-jb / 2.0).exp();

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

/// Upper-tail p-value for the Anderson-Darling A*² statistic, using the
/// D'Agostino & Stephens (1986) piecewise approximation.
///
/// # Numerical range
///
/// The large-A*² branch `exp(1.2937 − 5.709·A*² + 0.0186·A*²²)` is a bounded-range
/// fit: its positive quadratic term turns the exponent back *upward* past the
/// vertex A*² = 5.709 / (2·0.0186) ≈ 153.47. Left unguarded, for a strongly
/// non-normal large sample (A*² in the hundreds) the exponent grows positive, the
/// exponential overflows, and after `clamp(0, 1)` the p-value flips to exactly
/// `1.0` ("perfectly normal") while A*² is simultaneously huge and still growing —
/// an internally inconsistent result. Clamping the evaluation point to the vertex
/// makes the p-value monotonically non-increasing in A*², plateauing at a tiny
/// (~5e-190) representable floor instead of jumping to 1.
///
/// # References
///
/// - D'Agostino & Stephens (1986). "Tests based on EDF statistics". In
///   *Goodness-of-Fit Techniques*. Marcel Dekker.
fn ad_upper_tail_pvalue(a2_star: f64) -> f64 {
    // Vertex of the upper-branch quadratic 1.2937 − 5.709·x + 0.0186·x²; beyond it
    // the fit is invalid and non-monotone, so evaluation is clamped here.
    const UPPER_BRANCH_VERTEX: f64 = 153.467_741_935_483_87; // 5.709 / (2 * 0.0186)

    let p = if a2_star >= 0.6 {
        let x = a2_star.min(UPPER_BRANCH_VERTEX);
        (1.2937 - 5.709 * x + 0.0186 * x * x).exp()
    } else if a2_star > 0.34 {
        (0.9177 - 4.279 * a2_star - 1.38 * a2_star * a2_star).exp()
    } else if a2_star > 0.2 {
        1.0 - (-8.318 + 42.796 * a2_star - 59.938 * a2_star * a2_star).exp()
    } else {
        1.0 - (-13.436 + 101.14 * a2_star - 223.73 * a2_star * a2_star).exp()
    };
    p.clamp(0.0, 1.0)
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
/// `None` if n < 8 (the floor below which the sample-size correction and the
/// p-value approximation are not tabulated; R's `nortest::ad.test` refuses the
/// same), all values identical, or non-finite values.
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
        let z_rev = (x[n - 1 - i] - mean) / sd;
        // Both tails evaluated directly, so an outlier contributes its true
        // ln-probability instead of a subtraction that rounds to ln(0); the
        // floor only guards arguments beyond the f64 range of the tail (|z| > 38).
        let ln_phi = special::standard_normal_cdf(z).max(f64::MIN_POSITIVE).ln();
        let ln_sf_rev = special::standard_normal_sf(z_rev)
            .max(f64::MIN_POSITIVE)
            .ln();

        let coeff = (2 * (i + 1) - 1) as f64;
        s += coeff * (ln_phi + ln_sf_rev);
    }

    let a2 = -nf - s / nf;

    // Stephens (1986) correction for sample size
    let a2_star = a2 * (1.0 + 0.75 / nf + 2.25 / (nf * nf));

    // P-value from piecewise approximation (D'Agostino & Stephens 1986), with the
    // large-A*² branch clamped to its valid range (see `ad_upper_tail_pvalue`).
    let p = ad_upper_tail_pvalue(a2_star);

    Some(AndersonDarlingResult {
        statistic: a2,
        statistic_star: a2_star,
        p_value: p,
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
        special::standard_normal_sf(z)
    } else {
        // Large sample: log-normal transformation
        let xx = nf.ln();
        let m = sw_poly(&SW_C5, xx);
        let s = sw_poly(&SW_C6, xx).exp();
        if s < 1e-300 {
            return 0.0;
        }
        let z = (y - m) / s;
        special::standard_normal_sf(z)
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
/// 3. p-value (two-sided):
///    - **no ties and both samples ≤ 50**: the exact null distribution of U,
///      by the recursion P(m, n, u) = m/(m+n)·P(m−1, n, u−n) + n/(m+n)·P(m, n−1, u)
///    - otherwise: normal approximation with continuity correction,
///      z = (|U₁ − μ| − ½) / σ, μ = n₁n₂/2, σ² with the tie correction
///
/// The normal approximation alone understates p for small samples: with
/// three observations each and complete separation it gives 0.0495 where
/// the smallest achievable p is 0.100.
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

    let p_value = if tie_correction == 0.0 && n1 <= EXACT_MAX_N && n2 <= EXACT_MAX_N {
        // Without ties U is an integer in 0..=n1·n2 and its null
        // distribution is symmetric about n1·n2/2.
        let u = u1.round() as usize;
        let tail = u.min(n1 * n2 - u);
        (2.0 * mann_whitney_exact_cdf(n1, n2, tail)).min(1.0)
    } else {
        continuity_corrected_p(u1, mu, sigma_sq)
    };

    Some(TestResult {
        statistic: u1,
        df: 0.0, // not applicable for non-parametric
        p_value,
    })
}

/// Largest sample size for which the rank tests use their exact null
/// distribution (when there are no ties). Above it the normal approximation
/// with continuity correction is accurate to well under 0.001 in p.
const EXACT_MAX_N: usize = 50;

/// Two-sided p of a rank statistic by the normal approximation with
/// continuity correction.
fn continuity_corrected_p(statistic: f64, mu: f64, sigma_sq: f64) -> f64 {
    let z = ((statistic - mu).abs() - 0.5).max(0.0) / sigma_sq.sqrt();
    (2.0 * special::standard_normal_sf(z)).min(1.0)
}

/// P(U ≤ u) for the Mann-Whitney statistic with sample sizes `m`, `n` and
/// no ties.
///
/// Every arrangement of the two samples' ranks is equally likely under H₀,
/// and the last-ranked observation belongs to the first sample with
/// probability m/(m+n), in which case it adds n to U:
/// P(m, n, u) = m/(m+n)·P(m−1, n, u−n) + n/(m+n)·P(m, n−1, u).
/// O(m·n·(m·n)) time, O(n·m·n) memory -- a few megabytes at m = n = 50.
fn mann_whitney_exact_cdf(m: usize, n: usize, u: usize) -> f64 {
    let max_u = m * n;
    // row[j] = distribution of U for (i, j), indices 0..=i·j.
    let mut prev: Vec<Vec<f64>> = (0..=n).map(|_| vec![1.0]).collect(); // i = 0
    for i in 1..=m {
        let mut cur: Vec<Vec<f64>> = Vec::with_capacity(n + 1);
        cur.push(vec![1.0]); // j = 0: U = 0
        for j in 1..=n {
            let mut dist = vec![0.0; i * j + 1];
            let wi = i as f64 / (i + j) as f64;
            let wj = j as f64 / (i + j) as f64;
            // Last observation from sample 1: U shifts by j.
            for (k, &pk) in prev[j].iter().enumerate() {
                dist[k + j] += wi * pk;
            }
            // Last observation from sample 2: U unchanged.
            for (k, &pk) in cur[j - 1].iter().enumerate() {
                dist[k] += wj * pk;
            }
            cur.push(dist);
        }
        prev = cur;
    }
    prev[n].iter().take(u.min(max_u) + 1).sum::<f64>().min(1.0)
}

/// P(T⁺ ≤ t) for the Wilcoxon signed-rank statistic with `n` non-zero,
/// untied differences: each rank enters T⁺ with probability ½ independently.
fn wilcoxon_exact_cdf(n: usize, t: usize) -> f64 {
    let max_t = n * (n + 1) / 2;
    let mut dist = vec![0.0; max_t + 1];
    dist[0] = 1.0;
    let mut reach = 0;
    for rank in 1..=n {
        reach += rank;
        for s in (rank..=reach).rev() {
            dist[s] = 0.5 * dist[s] + 0.5 * dist[s - rank];
        }
        for d in dist.iter_mut().take(rank) {
            *d *= 0.5;
        }
    }
    dist.iter().take(t.min(max_t) + 1).sum::<f64>().min(1.0)
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
/// 4. p-value (two-sided):
///    - **no tied |dᵢ| and n ≤ 50**: the exact null distribution of T⁺
///      (each rank enters T⁺ with probability ½)
///    - otherwise: normal approximation with continuity correction,
///      z = (|T⁺ − μ| − ½) / σ, μ = n(n+1)/4, σ² with the tie correction
///
/// The normal approximation alone understates p for small samples: five
/// differences all of one sign give 0.043 where the exact p is 0.0625.
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

    let p_value = if tie_correction_val == 0.0 && nr <= EXACT_MAX_N {
        // Without ties T⁺ is an integer in 0..=n(n+1)/2, symmetric about
        // n(n+1)/4.
        let t = t_plus.round() as usize;
        let tail = t.min(nr * (nr + 1) / 2 - t);
        (2.0 * wilcoxon_exact_cdf(nr, tail)).min(1.0)
    } else {
        continuity_corrected_p(t_plus, mu, sigma_sq)
    };

    Some(TestResult {
        statistic: t_plus,
        df: 0.0,
        p_value,
    })
}

/// Whether two sample values are one rank: equal, or equal up to the rounding
/// of the arithmetic that made them (a relative 1e-12, so the same samples in
/// another unit tie the same pairs; a fixed gap would call every value of a
/// sample in picovolts tied).
fn tied(a: f64, b: f64) -> bool {
    a == b || (a - b).abs() <= 1e-12 * a.abs().max(b.abs())
}

// Assign average ranks to sorted (value, group_or_index) pairs.
// Handles ties by assigning the average of the tied ranks.
fn average_ranks(sorted: &[(f64, usize)]) -> Vec<f64> {
    let n = sorted.len();
    let mut ranks = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n && tied(sorted[j].0, sorted[i].0) {
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
        while j < n && tied(sorted[j].0, sorted[i].0) {
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
    let p_value = special::chi_squared_sf(h, df);

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
/// `None` if the slice is empty or a value is not a probability in `[0, 1]`
/// (a NaN included).
pub fn bonferroni_correction(p_values: &[f64]) -> Option<Vec<f64>> {
    if p_values.is_empty() || p_values.iter().any(|v| !(0.0..=1.0).contains(v)) {
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
/// `None` if the slice is empty or a value is not a probability in `[0, 1]`
/// (a NaN included).
///
/// # References
///
/// Benjamini & Hochberg (1995). "Controlling the false discovery rate".
/// JRSS-B, 57(1), 289–300.
pub fn benjamini_hochberg(p_values: &[f64]) -> Option<Vec<f64>> {
    let m = p_values.len();
    if m == 0 || p_values.iter().any(|v| !(0.0..=1.0).contains(v)) {
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
    let p_value = special::chi_squared_sf(statistic, df);

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

    // Tied means equal, as in S above: a tolerance in the unit of the data
    // would call a whole series tied (or none of it) depending on whether it
    // is written in volts or picovolts.
    let mut tie_correction: f64 = 0.0;
    let mut current_count: usize = 1;
    for i in 1..sorted.len() {
        if sorted[i] == sorted[i - 1] {
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
    let p_value = 2.0 * special::standard_normal_sf(z_statistic.abs());
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
// KPSS stationarity test
// ---------------------------------------------------------------------------

/// What the KPSS null hypothesis says the series is stationary around.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KpssModel {
    /// A constant level (statsmodels `regression="c"`).
    Level,
    /// A linear trend (statsmodels `regression="ct"`).
    Trend,
}

/// The truncation lag of the Newey-West long-run variance in [`kpss_test`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KpssLags {
    /// Hobijn, Franses & Ooms (1998): chosen from the residuals' own
    /// autocovariances (statsmodels' default, `nlags="auto"`).
    Hobijn,
    /// Schwert (1989): `⌈12·(n/100)^¼⌉` (statsmodels `nlags="legacy"`).
    Schwert,
    /// Exactly this many, below n.
    Fixed(usize),
}

/// Where a [`KpssResult::p_value`] stands relative to the table it is read from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KpssPValue {
    /// Interpolated inside the table (between 0.01 and 0.10).
    Interpolated,
    /// The statistic is beyond the 1 % critical value: the p-value is at most 0.01.
    AtMost,
    /// The statistic is below the 10 % critical value: the p-value is at least 0.10.
    AtLeast,
}

/// Result of the KPSS stationarity test.
#[derive(Debug, Clone)]
pub struct KpssResult {
    /// The KPSS statistic η / σ̂² (Kwiatkowski et al. 1992, eq. 11).
    pub statistic: f64,
    /// The truncation lag of the long-run variance.
    pub n_lags: usize,
    /// p-value, linearly interpolated in the table of critical values and
    /// clamped to [0.01, 0.10] — see [`KpssResult::p_value_bound`].
    pub p_value: f64,
    /// Whether `p_value` is interpolated or a bound of the table.
    pub p_value_bound: KpssPValue,
    /// Critical values at 10 %, 5 %, 2.5 % and 1 % (Kwiatkowski et al. 1992, Table 1).
    pub critical_values: [f64; 4],
    /// `statistic > critical_value` at each of those levels: stationarity is rejected.
    pub rejected: [bool; 4],
}

/// Significance levels of [`KpssResult::critical_values`], in order.
pub const KPSS_LEVELS: [f64; 4] = [0.10, 0.05, 0.025, 0.01];

/// The values [`kpss_test`] takes for a fixed truncation lag on `data`:
/// `0..=n − 1`.
///
/// # Errors
///
/// The series' own refusals, as [`kpss_test`] gives them: a non-finite value
/// or fewer than [`KPSS_MIN_OBSERVATIONS`] values.
pub fn kpss_lags_domain(data: &[f64]) -> Result<RangeInclusive<usize>, TimeSeriesError> {
    check_series(data, KPSS_MIN_OBSERVATIONS)?;
    Ok(0..=data.len() - 1)
}

/// The fewest values [`kpss_test`] takes.
pub const KPSS_MIN_OBSERVATIONS: usize = 10;

/// KPSS test of stationarity: H₀ the series is stationary around a level (or
/// a trend), H₁ it has a unit root.
///
/// The complement of [`adf_test`], whose null is the unit root: a series ADF
/// does not reject and KPSS does is non-stationary; differencing it once and
/// testing again is how the number of differences is chosen (Hyndman &
/// Khandakar 2008, `ndiffs`).
///
/// # Algorithm
///
/// Residuals eₜ of the series on a constant (or constant + trend);
/// η = Σ Sₜ² / n² with Sₜ the partial sums of eₜ; σ̂² the Newey-West long-run
/// variance with Bartlett weights 1 − i/(l + 1); statistic η / σ̂². The p-value
/// is interpolated in the table of critical values.
///
/// # Errors
///
/// [`TimeSeriesError`]: a non-finite value, fewer than 10 values, a fixed
/// lag outside `[0, n − 1]` (`lags`), or no variation about the level or
/// trend (σ̂² not positive against the series' size).
///
/// # References
///
/// - Kwiatkowski, Phillips, Schmidt & Shin (1992). "Testing the null
///   hypothesis of stationarity against the alternative of a unit root."
///   *Journal of Econometrics*, 54, 159–178.
/// - Hobijn, Franses & Ooms (1998). "Generalisations of the KPSS-test for
///   stationarity." Econometric Institute Report 9802/A.
/// - Newey & West (1994). "Automatic lag selection in covariance matrix
///   estimation." *Review of Economic Studies*, 61(4), 631–653.
///
/// # Examples
///
/// ```
/// use u_analytics::testing::{kpss_test, KpssLags, KpssModel};
///
/// // A walk that drifts: stationarity around a level is rejected at 5 %.
/// let mut walk = vec![0.0];
/// for i in 1..80 {
///     walk.push(walk[i - 1] + 1.0 + 0.1 * (((i * 7919) % 13) as f64 - 6.0));
/// }
/// let r = kpss_test(&walk, KpssModel::Level, KpssLags::Hobijn).unwrap();
/// assert!(r.rejected[1]);
/// ```
pub fn kpss_test(
    data: &[f64],
    model: KpssModel,
    lags: KpssLags,
) -> Result<KpssResult, TimeSeriesError> {
    let domain = kpss_lags_domain(data)?;
    let n = data.len();
    if let KpssLags::Fixed(l) = lags {
        check_option("lags", l, &domain)?;
    }
    let nf = n as f64;

    let residuals: Vec<f64> = match model {
        KpssModel::Level => {
            let mean = data.iter().sum::<f64>() / nf;
            data.iter().map(|x| x - mean).collect()
        }
        KpssModel::Trend => {
            // OLS on 1 and t = 1..n.
            let t_mean = (nf + 1.0) / 2.0;
            let y_mean = data.iter().sum::<f64>() / nf;
            let (mut sty, mut stt) = (0.0, 0.0);
            for (i, y) in data.iter().enumerate() {
                let dt = (i + 1) as f64 - t_mean;
                sty += dt * (y - y_mean);
                stt += dt * dt;
            }
            let slope = sty / stt;
            data.iter()
                .enumerate()
                .map(|(i, y)| y - y_mean - slope * ((i + 1) as f64 - t_mean))
                .collect()
        }
    };

    // No variation beyond the rounding of the fit, against the series' size.
    let sum_sq: f64 = residuals.iter().map(|e| e * e).sum();
    let size: f64 = data.iter().map(|x| x * x).sum();
    if sum_sq <= 1e-20 * size {
        return Err(TimeSeriesError::NoVariation);
    }

    let autocovariance = |i: usize| -> f64 {
        residuals[i..]
            .iter()
            .zip(&residuals[..n - i])
            .map(|(a, b)| a * b)
            .sum()
    };

    let n_lags = match lags {
        KpssLags::Fixed(l) => l,
        KpssLags::Schwert => ((12.0 * (nf / 100.0).powf(0.25)).ceil() as usize).min(n - 1),
        KpssLags::Hobijn => {
            // Hobijn et al. (1998), as statsmodels' `_kpss_autolag`.
            let cov_lags = nf.powf(2.0 / 9.0) as usize;
            let mut s0 = sum_sq / nf;
            let mut s1 = 0.0;
            for i in 1..=cov_lags.min(n - 1) {
                let g = autocovariance(i) / (nf / 2.0);
                s0 += g;
                s1 += i as f64 * g;
            }
            let s_hat = s1 / s0;
            let gamma = 1.1447 * (s_hat * s_hat).powf(1.0 / 3.0);
            ((gamma * nf.powf(1.0 / 3.0)) as usize).min(n - 1)
        }
    };

    let mut long_run = sum_sq;
    for i in 1..=n_lags {
        long_run += 2.0 * autocovariance(i) * (1.0 - i as f64 / (n_lags as f64 + 1.0));
    }
    let long_run = long_run / nf;
    if long_run <= 0.0 {
        return Err(TimeSeriesError::NoVariation);
    }

    let mut partial = 0.0;
    let eta = residuals
        .iter()
        .map(|e| {
            partial += e;
            partial * partial
        })
        .sum::<f64>()
        / (nf * nf);
    let statistic = eta / long_run;

    let critical_values = match model {
        KpssModel::Level => [0.347, 0.463, 0.574, 0.739],
        KpssModel::Trend => [0.119, 0.146, 0.176, 0.216],
    };
    let (p_value, p_value_bound) = if statistic <= critical_values[0] {
        (KPSS_LEVELS[0], KpssPValue::AtLeast)
    } else if statistic >= critical_values[3] {
        (KPSS_LEVELS[3], KpssPValue::AtMost)
    } else {
        let k = critical_values
            .windows(2)
            .position(|w| statistic <= w[1])
            .expect("the statistic lies inside the table");
        let (c0, c1) = (critical_values[k], critical_values[k + 1]);
        let (p0, p1) = (KPSS_LEVELS[k], KPSS_LEVELS[k + 1]);
        (
            p0 + (statistic - c0) / (c1 - c0) * (p1 - p0),
            KpssPValue::Interpolated,
        )
    };
    let rejected = critical_values.map(|c| statistic > c);

    Ok(KpssResult {
        statistic,
        n_lags,
        p_value,
        p_value_bound,
        critical_values,
        rejected,
    })
}

/// KPSS against statsmodels 0.15.0 `kpss(x, regression, nlags)` on a
/// stationary AR(1) and a random walk (seeded, rounded to 3 places).
#[cfg(test)]
mod kpss_reference {
    use super::*;

    const AR: [f64; 60] = [
        0.0, 0.299, -0.154, -0.953, -0.836, -1.326, -0.471, 1.152, -0.031, -0.633, 0.237, 0.452,
        0.286, -0.816, -0.355, 0.553, -1.123, -0.907, -2.264, -2.196, -2.72, -1.323, -1.796,
        -0.447, -0.022, -0.196, -2.595, -1.577, -0.68, -0.159, -1.594, -1.115, -1.425, -1.379,
        0.509, -0.604, -0.275, 0.774, -0.274, -0.222, 0.021, 0.073, -1.196, -0.402, 1.198, -1.068,
        0.432, 0.292, -0.524, 1.79, 1.478, -0.608, -0.168, 0.51, 0.015, 0.689, 0.209, 0.75, 1.739,
        0.02,
    ];
    const RW: [f64; 60] = [
        0.001, 0.3, 0.026, -0.865, -1.32, -2.312, -2.252, -0.912, -1.404, -2.024, -1.534, -1.177,
        -1.072, -2.002, -2.031, -1.336, -2.68, -3.138, -5.039, -6.329, -8.171, -8.406, -9.673,
        -9.402, -9.245, -9.432, -11.949, -12.488, -12.537, -12.424, -13.954, -14.432, -15.411,
        -16.22, -15.159, -15.967, -16.0, -15.116, -15.7, -15.812, -15.702, -15.638, -16.863,
        -16.787, -15.428, -16.975, -16.116, -15.997, -16.638, -14.638, -13.876, -15.075, -15.0,
        -14.423, -14.612, -13.929, -13.996, -13.329, -11.89, -12.566,
    ];

    #[test]
    fn statistic_lag_and_p_value_reproduce_statsmodels() {
        use KpssLags::{Fixed, Hobijn, Schwert};
        use KpssModel::{Level, Trend};
        type Case = (&'static [f64; 60], KpssModel, KpssLags, f64, f64, usize);
        let cases: [Case; 12] = [
            (
                &AR,
                Level,
                Hobijn,
                0.5229270606681895,
                0.036502914263921285,
                3,
            ),
            (&AR, Level, Schwert, 0.288001639308506, 0.1, 11),
            (
                &AR,
                Level,
                Fixed(4),
                0.46189194657609306,
                0.05047760923444266,
                4,
            ),
            (&AR, Trend, Hobijn, 0.2274846895383343, 0.01, 3),
            (
                &AR,
                Trend,
                Schwert,
                0.14193842821368732,
                0.05752142923391235,
                11,
            ),
            (
                &AR,
                Trend,
                Fixed(4),
                0.20574827757371436,
                0.013844395909857114,
                4,
            ),
            (&RW, Level, Hobijn, 0.9286533322383663, 0.01, 5),
            (
                &RW,
                Level,
                Schwert,
                0.5112288964656923,
                0.0391376359311504,
                11,
            ),
            (&RW, Level, Fixed(4), 1.09862199299003, 0.01, 4),
            (&RW, Trend, Hobijn, 0.26866997231516837, 0.01, 4),
            (
                &RW,
                Trend,
                Schwert,
                0.14468829763941088,
                0.05242907844553539,
                11,
            ),
            (&RW, Trend, Fixed(4), 0.26866997231516837, 0.01, 4),
        ];
        for (data, model, lags, statistic, p, n_lags) in cases {
            let r = kpss_test(data, model, lags).unwrap();
            let what = format!("{model:?} {lags:?}");
            assert_eq!(r.n_lags, n_lags, "{what}");
            assert!(
                (r.statistic - statistic).abs() < 1e-10 * statistic,
                "{what}: {} vs {statistic}",
                r.statistic
            );
            assert!(
                (r.p_value - p).abs() < 1e-12,
                "{what}: p {} vs {p}",
                r.p_value
            );
        }
    }

    #[test]
    fn bounds_rejections_and_refusals() {
        let r = kpss_test(&RW, KpssModel::Level, KpssLags::Fixed(4)).unwrap();
        assert_eq!(r.p_value_bound, KpssPValue::AtMost);
        assert_eq!(r.rejected, [true; 4]);
        let r = kpss_test(&AR, KpssModel::Level, KpssLags::Schwert).unwrap();
        assert_eq!(r.p_value_bound, KpssPValue::AtLeast);
        assert_eq!(r.rejected, [false; 4]);
        let r = kpss_test(&AR, KpssModel::Level, KpssLags::Hobijn).unwrap();
        assert_eq!(r.p_value_bound, KpssPValue::Interpolated);
        assert_eq!(r.rejected, [true, true, false, false]);

        assert_eq!(
            kpss_test(&AR[..9], KpssModel::Level, KpssLags::Hobijn).unwrap_err(),
            TimeSeriesError::TooFewObservations { needed: 10, got: 9 }
        );
        assert_eq!(
            kpss_test(&AR, KpssModel::Level, KpssLags::Fixed(60)).unwrap_err(),
            TimeSeriesError::OptionOutOfRange {
                option: "lags",
                got: 60.0,
                min: 0.0,
                max: 59.0
            }
        );
        assert_eq!(
            kpss_test(&[4.0; 20], KpssModel::Level, KpssLags::Hobijn).unwrap_err(),
            TimeSeriesError::NoVariation
        );
        let line: Vec<f64> = (0..20).map(|i| 2.0 + 0.5 * i as f64).collect();
        assert_eq!(
            kpss_test(&line, KpssModel::Trend, KpssLags::Hobijn).unwrap_err(),
            TimeSeriesError::NoVariation
        );
        let mut bad = AR;
        bad[3] = f64::INFINITY;
        assert_eq!(
            kpss_test(&bad[..5], KpssModel::Level, KpssLags::Hobijn).unwrap_err(),
            TimeSeriesError::ValueNotFinite { index: 3 }
        );
    }

    /// The statistic is a ratio of the residuals' own scale: the same in any unit.
    #[test]
    fn unit_invariant() {
        let base = kpss_test(&AR, KpssModel::Trend, KpssLags::Hobijn).unwrap();
        for f in [1e-9, 1e9] {
            let x: Vec<f64> = AR.iter().map(|v| v * f).collect();
            let r = kpss_test(&x, KpssModel::Trend, KpssLags::Hobijn).unwrap();
            assert_eq!(r.n_lags, base.n_lags);
            assert!((r.statistic - base.statistic).abs() < 1e-9 * base.statistic);
        }
    }
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

/// How [`adf_test`] chooses the number of lagged differences.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdfLags {
    /// The count from 0 to `max` with the smallest AIC, every candidate fitted
    /// on the same observations (those the largest count leaves), then the
    /// chosen count refitted on all the observations it leaves. `max` `None`
    /// is Schwert's rule, `⌊12·(n/100)^¼⌋`, capped at `n / 3`.
    Aic {
        /// The largest count considered.
        max: Option<usize>,
    },
    /// Exactly this many.
    Fixed(usize),
}

impl Default for AdfLags {
    fn default() -> Self {
        AdfLags::Aic { max: None }
    }
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
/// `lags` chooses the number of lagged differences ([`AdfLags`]): by AIC up
/// to a maximum (the default, as `statsmodels.adfuller(autolag="AIC")`), or a
/// fixed count.
///
/// Reference: Dickey & Fuller (1979), "Distribution of the Estimators for
/// Autoregressive Time Series with a Unit Root"
///
/// # Errors
///
/// [`TimeSeriesError`]: a non-finite value, fewer than 10 values, a lag
/// count (`lags`, or the AIC search's `max_lags`) outside `[0, max]` where
/// `max` leaves the regression at least 5 rows and a residual degree of
/// freedom ([`adf_lags_domain`]), or a degenerate regression.
///
/// # Examples
///
/// ```
/// use u_analytics::testing::{adf_test, AdfLags, AdfModel};
///
/// // Stationary series: strong mean-reversion
/// let mut data = vec![0.0_f64; 40];
/// for i in 1..40 {
///     data[i] = 0.3 * data[i - 1] + [0.5, -0.8, 0.3, -0.6, 0.9,
///         -0.4, 0.7, -0.2, 0.1, -0.5][i % 10];
/// }
/// let r = adf_test(&data, AdfModel::Constant, AdfLags::default()).unwrap();
/// assert!(r.statistic.is_finite());
/// assert_eq!(r.critical_values.len(), 3);
/// ```
pub fn adf_test(
    data: &[f64],
    model: AdfModel,
    lags: AdfLags,
) -> Result<AdfResult, TimeSeriesError> {
    let domain = adf_lags_domain(data, model)?;
    let n = data.len();
    match lags {
        AdfLags::Fixed(p) => {
            check_option("lags", p, &domain)?;
        }
        AdfLags::Aic { max: Some(m) } => {
            check_option("max_lags", m, &domain)?;
        }
        AdfLags::Aic { max: None } => {}
    }

    // Compute differences
    let dy: Vec<f64> = data.windows(2).map(|w| w[1] - w[0]).collect();

    let best_lag = match lags {
        AdfLags::Fixed(p) => p,
        AdfLags::Aic { max } => {
            // Schwert (1989) rule for the largest count considered.
            let schwert = (12.0 * (n as f64 / 100.0).powf(0.25)).floor() as usize;
            let p_max = max.unwrap_or_else(|| schwert.min(n / 3));
            select_adf_lag(data, &dy, model, p_max).ok_or(TimeSeriesError::DegenerateRegression)?
        }
    };

    // The chosen count, on every observation it leaves.
    adf_ols(data, &dy, model, best_lag).ok_or(TimeSeriesError::DegenerateRegression)
}

/// The fewest values [`adf_test`] takes.
pub const ADF_MIN_OBSERVATIONS: usize = 10;

/// The fewest rows the ADF test regression is fitted on.
const ADF_MIN_REGRESSION_ROWS: usize = 5;

/// The numbers of lagged differences [`adf_test`] takes on `data` under
/// `model` -- for a fixed count (`lags`) and for the bound of the AIC search
/// (`max_lags`) alike: `0..=p`, where with p lags the regression keeps
/// n − 1 − p rows (one value is lost to differencing, p to the lags) and has
/// the lagged level, the model's deterministic terms and p lag columns; it
/// needs at least 5 rows and a residual degree of freedom.
///
/// # Errors
///
/// The series' own refusals, as [`adf_test`] gives them: a non-finite value
/// or fewer than [`ADF_MIN_OBSERVATIONS`] values.
///
/// # Examples
///
/// ```
/// use u_analytics::testing::{adf_lags_domain, AdfModel};
///
/// let data: Vec<f64> = (0..20).map(|i| (i as f64).sin()).collect();
/// assert_eq!(adf_lags_domain(&data, AdfModel::Constant).unwrap(), 0..=8);
/// ```
pub fn adf_lags_domain(
    data: &[f64],
    model: AdfModel,
) -> Result<RangeInclusive<usize>, TimeSeriesError> {
    check_series(data, ADF_MIN_OBSERVATIONS)?;
    let n = data.len();
    let fixed_columns = match model {
        AdfModel::None => 1,
        AdfModel::Constant => 2,
        AdfModel::ConstantTrend => 3,
    };
    Ok(0..=(n - 1 - ADF_MIN_REGRESSION_ROWS).min((n - 2 - fixed_columns) / 2))
}

/// The lag count in `0..=p_max` with the smallest AIC. Every candidate is
/// fitted on the observations `p_max` lags leave, so the AICs compare the same
/// data — fitted on their own (longer) samples they would not be comparable.
/// `None` when no candidate can be fitted.
fn select_adf_lag(data: &[f64], dy: &[f64], model: AdfModel, p_max: usize) -> Option<usize> {
    let start = p_max;
    let mut best: Option<(f64, usize)> = None;
    for p in 0..=p_max {
        if let Some(aic) = adf_ols_aic(data, dy, model, p, start) {
            if best.is_none_or(|(b, _)| aic < b) {
                best = Some((aic, p));
            }
        }
    }
    best.map(|(_, p)| p)
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
    start: usize,
) -> Option<(Vec<f64>, Vec<f64>, usize, usize, usize)> {
    // Row t regresses dy[t] on the level data[t] and dy[t-1] ... dy[t-p].
    debug_assert!(start >= p, "the first row needs its p lagged differences");
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

    // A pivot is nothing when it is rounding against the sizes of the two
    // regressors it couples: X'X is in the squares of the data's units, so a
    // fixed floor would call a series in small units singular.
    let own_size: Vec<f64> = (0..ncols).map(|i| xtx[i * ncols + i]).collect();
    let negligible = |pivot: f64, col: usize, row: usize| {
        pivot <= 1e-15 * (own_size[col] * own_size[row]).sqrt()
    };

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
        if negligible(max_val, col, max_row) {
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

    // An exact fit leaves no residual variance, so the t-ratio divides by
    // rounding noise: a series whose differences repeat with a period the lags
    // span came out as t = -1.9e8, a "rejection" at every level. Treat a fit
    // that explains the differences to eight digits as undefined, which also
    // keeps it out of the AIC lag search (where ln(rss) -> -inf would win).
    let yy: f64 = y.iter().map(|v| v * v).sum();
    if yy == 0.0 || rss <= yy * 1e-16 {
        return None;
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
        if negligible(max_val, col, max_row) {
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
/// AIC of the regression with `p` lags on the rows from `start` on.
fn adf_ols_aic(data: &[f64], dy: &[f64], model: AdfModel, p: usize, start: usize) -> Option<f64> {
    let (x_data, y_dep, m, ncols, gamma_col) = adf_build_matrix(data, dy, model, p, start)?;
    let (_t_stat, rss, k) = adf_ols_core(&x_data, &y_dep, m, ncols, gamma_col)?;
    Some(2.0 * k as f64 + m as f64 * (rss / m as f64).ln())
}

/// Runs the actual ADF OLS and returns the test result.
fn adf_ols(data: &[f64], dy: &[f64], model: AdfModel, p: usize) -> Option<AdfResult> {
    let (x_data, y_dep, m, ncols, gamma_col) = adf_build_matrix(data, dy, model, p, p)?;
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

    #[test]
    fn corrections_refuse_what_is_not_a_probability() {
        assert!(bonferroni_correction(&[0.01, 1.5]).is_none());
        assert!(benjamini_hochberg(&[-0.1, 0.2]).is_none());
        assert_eq!(bonferroni_correction(&[0.01, 0.2]), Some(vec![0.02, 0.4]));
    }

    // -----------------------------------------------------------------------
    // Anderson-Darling large-n p-value overflow (upstream-014)
    // -----------------------------------------------------------------------

    /// Public-domain Mulberry32 PRNG (matches the upstream-014 reproduction).
    fn mulberry32(seed: u32) -> impl FnMut() -> f64 {
        let mut a = seed;
        move || {
            a = a.wrapping_add(0x6d2b_79f5);
            let mut t = a;
            t = (t ^ (t >> 15)).wrapping_mul(t | 1);
            t ^= t.wrapping_add((t ^ (t >> 7)).wrapping_mul(t | 61));
            ((t ^ (t >> 14)) as f64) / 4_294_967_296.0
        }
    }

    /// Regression (upstream-014): for a fixed, clearly non-normal distribution
    /// shape (exponential), as n grows the A*² statistic grows monotonically and
    /// the p-value must keep tracking toward 0 — it must NEVER jump to exactly 1.0
    /// ("perfectly normal") while A*² is simultaneously large and growing. Before
    /// the fix, the p-value overflowed to exactly 1 past n≈7000.
    #[test]
    fn ad_pvalue_no_overflow_large_nonnormal_n() {
        let mut rng = mulberry32(0x5350_4331);
        let full: Vec<f64> = (0..20_000)
            .map(|_| {
                let u = rng();
                -(1.0 - u).ln() // inverse-CDF exponential, rate=1
            })
            .collect();

        let mut prev_a2 = 0.0_f64;
        let mut prev_p = f64::INFINITY;
        for &n in &[3000usize, 5000, 6000, 7000, 8000, 10000, 16000, 20000] {
            let r = anderson_darling_test(&full[..n]).expect("computes");
            // A*² must keep growing for an increasingly non-normal large sample.
            assert!(
                r.statistic_star > prev_a2,
                "A*² must grow with n; n={n} A*²={} prev={prev_a2}",
                r.statistic_star
            );
            // The core defect: p must never flip to exactly 1.0 while A*² is huge.
            assert!(
                r.p_value < 0.5,
                "clearly non-normal data (n={n}, A*²={}) must not report p={} ≈ normal",
                r.statistic_star,
                r.p_value
            );
            // Monotone non-increasing: p tracks the growing statistic downward.
            assert!(
                r.p_value <= prev_p + 1e-12,
                "p must be non-increasing as A*² grows; n={n} p={} prev={prev_p}",
                r.p_value
            );
            prev_a2 = r.statistic_star;
            prev_p = r.p_value;
        }
    }

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

    /// The statistic is the one the cited paper defines, which is the one
    /// SciPy, R and statsmodels report -- moment ratios, not the bias-adjusted
    /// estimators Excel's SKEW()/KURT() return.
    ///
    /// Reference values computed from the definition, independently of the
    /// implementation: g1 = m3/m2^1.5, g2 = m4/m2^2 - 3, JB = n/6 (g1^2 + g2^2/4).
    #[test]
    fn jb_is_the_moment_form_the_paper_defines() {
        let data = [
            2.1, 3.4, 1.9, 5.6, 2.2, 3.1, 4.8, 2.9, 3.3, 7.2, 2.5, 3.0, 4.1, 2.7, 3.8, 2.4, 6.1,
            3.6, 2.8, 3.2,
        ];
        let r = jarque_bera_test(&data).expect("20 finite points");
        assert!(
            (r.statistic - 5.563_714_337_9).abs() < 1e-8,
            "JB = {}",
            r.statistic
        );

        // What it used to compute, from the adjusted estimators: 35 % higher on
        // this sample. The gap is not a rounding difference to be tolerated.
        let n = data.len() as f64;
        let big_g1 = u_numflow::stats::skewness(&data).expect("same data");
        let big_g2 = u_numflow::stats::kurtosis(&data).expect("same data");
        let adjusted = (n / 6.0) * (big_g1 * big_g1 + big_g2 * big_g2 / 4.0);
        assert!(
            adjusted > r.statistic * 1.3,
            "the two forms differ by more than a third here: {adjusted} vs {}",
            r.statistic
        );
    }

    /// chi-squared with two degrees of freedom is an exponential, so the
    /// p-value is `exp(-JB/2)` exactly -- and stays a number where `1 - cdf`
    /// rounds to zero.
    #[test]
    fn jb_p_value_keeps_its_digits_in_the_tail() {
        // A ramp with one far outlier: JB near 93, where `1 - cdf` is exactly
        // zero in f64 and the true p-value is about 7e-21.
        let mut data: Vec<f64> = (1..20).map(f64::from).collect();
        data.push(60.0);
        let r = jarque_bera_test(&data).expect("20 finite points");
        assert!(r.statistic > 90.0, "JB = {}", r.statistic);
        assert!(
            1.0 - special::chi_squared_cdf(r.statistic, 2.0) == 0.0,
            "this sample is only interesting because the complement underflows"
        );
        assert!(
            r.p_value > 0.0,
            "the complement returns 0 here; this must not"
        );
        assert!(
            (r.p_value - 6.702e-21).abs() < 1e-23,
            "p = {} (expected about 6.7e-21)",
            r.p_value
        );
        assert!(
            (r.p_value - (-r.statistic / 2.0).exp()).abs() <= f64::EPSILON * r.p_value,
            "p = {} vs exp(-JB/2) = {}",
            r.p_value,
            (-r.statistic / 2.0).exp()
        );
        // And it still agrees with the distribution it claims, where that is
        // computable at all.
        let small = jarque_bera_test(&[-2.0, -1.5, -1.0, -0.5, 0.0, 0.0, 0.5, 1.0, 1.5, 2.0])
            .expect("10 finite points");
        let via_cdf = 1.0 - special::chi_squared_cdf(small.statistic, 2.0);
        assert!(
            (small.p_value - via_cdf).abs() < 1e-12,
            "{} vs {via_cdf}",
            small.p_value
        );
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
    // Anderson-Darling normality (formerly also `anderson_darling_normality`)
    // -----------------------------------------------------------------------

    #[test]
    fn ad_normal_data_large_p() {
        // Clearly normal data → cannot reject normality (p > 0.05)
        let data = [2.1, 1.9, 2.0, 2.05, 1.95, 2.02, 1.98, 2.01, 2.03, 1.97];
        let r = anderson_darling_test(&data).unwrap();
        assert!(r.p_value > 0.05, "p={}", r.p_value);
    }

    #[test]
    fn ad_exponential_data_small_p() {
        // Clearly non-normal (exponential) → reject normality (p < 0.05)
        let data: Vec<f64> = (1..=30).map(|i| (i as f64 * 0.3).exp()).collect();
        let r = anderson_darling_test(&data).unwrap();
        assert!(r.p_value < 0.05, "p={}", r.p_value);
    }

    #[test]
    fn ad_statistic_non_negative() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let r = anderson_darling_test(&data).unwrap();
        assert!(r.statistic >= 0.0);
        assert!(r.statistic_star >= r.statistic);
    }

    #[test]
    fn ad_p_value_in_range() {
        let data = [5.0, 5.1, 4.9, 5.05, 4.95, 5.02, 4.98, 5.0];
        let r = anderson_darling_test(&data).unwrap();
        assert!(r.p_value >= 0.0 && r.p_value <= 1.0);
    }

    #[test]
    fn ad_insufficient_data() {
        assert!(anderson_darling_test(&[1.0, 2.0]).is_none()); // n < 3
    }

    #[test]
    fn ad_degenerate_data() {
        // All same value → std = 0 → None
        assert!(anderson_darling_test(&[5.0, 5.0, 5.0, 5.0]).is_none());
    }

    #[test]
    fn ad_modified_statistic_formula() {
        // A²* > A² for any finite n
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let r = anderson_darling_test(&data).unwrap();
        assert!(r.statistic_star > r.statistic - 1e-10);
        // Below 8 values the correction is not tabulated: refused, not extrapolated.
        assert!(anderson_darling_test(&data[..7]).is_none());
    }

    /// Validates A²* = A² · (1 + 0.75/n + 2.25/n²) against manual computation.
    ///
    /// Reference: Stephens (1974), "EDF statistics for goodness of fit",
    /// J. Amer. Statist. Assoc. 69(347), 730–737 (composite normal case).
    #[test]
    fn ad_correction_factor_formula() {
        let data = [2.1, 1.9, 2.0, 2.05, 1.95, 2.02, 1.98, 2.01, 2.03, 1.97];
        let n = data.len() as f64;
        let r = anderson_darling_test(&data).unwrap();

        // Verify the Stephens correction formula A²* = A² · (1 + 0.75/n + 2.25/n²)
        let expected_star = r.statistic * (1.0 + 0.75 / n + 2.25 / (n * n));
        assert!(
            (r.statistic_star - expected_star).abs() < 1e-12,
            "A²* = {}, expected = {}",
            r.statistic_star,
            expected_star
        );

        // A² must be positive (it is a sum of log terms with negative coefficient)
        assert!(
            r.statistic >= 0.0,
            "A² = {} must be non-negative",
            r.statistic
        );
    }

    /// Validates that the A² summation formula implements:
    /// A² = -n - (1/n)·Σᵢ₌₀ⁿ⁻¹ (2i+1)·[ln Φ(zᵢ) + ln(1 − Φ(z_{n−1−i}))]
    ///
    /// This tests the formula structure by checking that a perfectly symmetric
    /// dataset centered at 0 produces a smaller A² than a skewed dataset.
    #[test]
    fn ad_formula_structure_symmetric_vs_skewed() {
        // Symmetric around mean → smaller A²
        let symmetric = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 0.0];
        // Right-skewed → larger A² (heavier right tail deviates from normality)
        let skewed: Vec<f64> = (1..=10).map(|i| (i as f64 * 0.5).exp()).collect();

        let r_sym = anderson_darling_test(&symmetric).unwrap();
        let r_skew = anderson_darling_test(&skewed).unwrap();

        assert!(
            r_sym.statistic < r_skew.statistic,
            "Symmetric A²={} should be less than skewed A²={}",
            r_sym.statistic,
            r_skew.statistic
        );
    }

    /// Validates p-value invariants for the Anderson-Darling test.
    ///
    /// - Normal data: p > 0.05 (cannot reject normality)
    /// - Exponential data: p < 0.05 (reject normality)
    #[test]
    fn ad_pvalue_invariants() {
        // Near-normal data — tightly clustered, should not reject normality
        let normal_data = [2.1, 1.9, 2.0, 2.05, 1.95, 2.02, 1.98, 2.01, 2.03, 1.97];
        let r_normal = anderson_darling_test(&normal_data).unwrap();
        assert!(
            r_normal.p_value > 0.05,
            "Normal data: p = {} (expected > 0.05)",
            r_normal.p_value
        );

        // Exponential data — heavy right tail, should reject normality
        let exp_data: Vec<f64> = (1..=30).map(|i| (i as f64 * 0.3).exp()).collect();
        let r_exp = anderson_darling_test(&exp_data).unwrap();
        assert!(
            r_exp.p_value < 0.05,
            "Exponential data: p = {} (expected < 0.05)",
            r_exp.p_value
        );
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

    /// Oracle: the exact p by enumerating every arrangement of the ranks
    /// (Python, independent of the recursion). The normal approximation this
    /// replaced reported the second column -- always too small, and below
    /// 0.05 in the first and third rows where the exact p is not.
    #[test]
    fn mann_whitney_small_samples_use_the_exact_distribution() {
        // (n1, n2, U, exact p, old normal-approximation p)
        let cases = [
            (3, 3, 0, 0.1000, 0.0495),
            (4, 4, 0, 0.0286, 0.0209),
            (4, 4, 1, 0.0571, 0.0433),
            (5, 5, 2, 0.0317, 0.0283),
            (8, 8, 13, 0.0499, 0.0460),
            (10, 10, 23, 0.0433, 0.0413),
        ];
        for (n1, n2, u, exact, old) in cases {
            let (a, b) = separated_samples(n1, n2, u);
            let r = mann_whitney_u_test(&a, &b).expect("valid samples");
            assert_eq!(r.statistic, u as f64);
            assert!(
                (r.p_value - exact).abs() < 1e-4,
                "n={n1},{n2} U={u}: p={} want {exact} (old {old})",
                r.p_value
            );
        }
    }

    /// Two samples of sizes `n1`, `n2` with no ties whose U₁ is exactly `u`:
    /// sample 1 takes the lowest ranks, then its top member is moved up past
    /// `u` members of sample 2 one step at a time.
    fn separated_samples(n1: usize, n2: usize, u: usize) -> (Vec<f64>, Vec<f64>) {
        let mut ranks_a: Vec<usize> = (1..=n1).collect();
        let mut remaining = u;
        for k in (0..n1).rev() {
            let room = n2.min(remaining);
            ranks_a[k] += room;
            remaining -= room;
            if remaining == 0 {
                break;
            }
        }
        let ranks_b: Vec<usize> = (1..=n1 + n2).filter(|r| !ranks_a.contains(r)).collect();
        (
            ranks_a.iter().map(|&r| r as f64).collect(),
            ranks_b.iter().map(|&r| r as f64).collect(),
        )
    }

    /// Oracle: exact p by enumerating all 2ⁿ sign assignments (Python).
    #[test]
    fn wilcoxon_small_samples_use_the_exact_distribution() {
        // (n, T+, exact p)
        let cases = [
            (5, 0, 0.0625),
            (5, 1, 0.125),
            (6, 2, 0.09375),
            (8, 5, 0.078125),
            (10, 8, 0.048828125),
            (12, 14, 0.0522460938),
        ];
        for (n, t_plus, exact) in cases {
            // Differences 1..=n; the ranks summing to t_plus are positive.
            let positive = ranks_summing_to(n, t_plus);
            let x: Vec<f64> = (1..=n)
                .map(|r| {
                    if positive.contains(&r) {
                        r as f64
                    } else {
                        -(r as f64)
                    }
                })
                .collect();
            let y = vec![0.0; n];
            let r = wilcoxon_signed_rank_test(&x, &y).expect("valid samples");
            assert_eq!(r.statistic, t_plus as f64);
            assert!(
                (r.p_value - exact).abs() < 1e-9,
                "n={n} T+={t_plus}: p={} want {exact}",
                r.p_value
            );
        }
    }

    /// A set of distinct ranks from 1..=n summing to `target` (greedy from the top).
    fn ranks_summing_to(n: usize, target: usize) -> Vec<usize> {
        let mut out = Vec::new();
        let mut left = target;
        for r in (1..=n).rev() {
            if r <= left {
                out.push(r);
                left -= r;
            }
        }
        assert_eq!(left, 0);
        out
    }

    /// Above the exact range, or with ties, the p comes from the normal
    /// approximation with continuity correction.
    #[test]
    fn rank_tests_fall_back_to_continuity_corrected_normal() {
        let a: Vec<f64> = (0..60).map(|i| i as f64).collect();
        let b: Vec<f64> = (0..60).map(|i| i as f64 + 20.5).collect();
        let r = mann_whitney_u_test(&a, &b).expect("valid samples");
        let (n1, n2) = (60.0, 60.0);
        let mu = n1 * n2 / 2.0;
        let sigma = (n1 * n2 * (n1 + n2 + 1.0) / 12.0_f64).sqrt();
        let z = ((r.statistic - mu).abs() - 0.5) / sigma;
        let want = 2.0 * special::standard_normal_sf(z);
        assert!((r.p_value - want).abs() < 1e-12);

        // At the boundary the two methods agree closely.
        let a: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let b: Vec<f64> = (0..50).map(|i| i as f64 + 10.5).collect();
        let exact = mann_whitney_u_test(&a, &b).expect("valid samples").p_value;
        let mu = 1250.0;
        let sigma = (2500.0 * 101.0 / 12.0_f64).sqrt();
        let u = mann_whitney_u_test(&a, &b)
            .expect("valid samples")
            .statistic;
        let approx = continuity_corrected_p(u, mu, sigma * sigma);
        assert!(
            (exact - approx).abs() < 1e-3,
            "exact {exact} vs approx {approx}"
        );
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

    /// Which values are tied does not depend on the unit the series is
    /// written in: the same series in volts or in picovolts has the same
    /// ties, so the same S, variance, Z and p; only Sen's slope scales.
    #[test]
    fn mk_does_not_depend_on_the_unit() {
        let data = [1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 5.0, 4.5, 6.0, 7.0, 6.5];
        let base = mann_kendall_test(&data).expect("computes");
        for scale in [1e-15, 1e-12, 1e-6, 1e6, 1e12] {
            let scaled: Vec<f64> = data.iter().map(|v| v * scale).collect();
            let r = mann_kendall_test(&scaled)
                .unwrap_or_else(|| panic!("scale {scale:e}: the series has variation"));
            assert_eq!(r.s_statistic, base.s_statistic, "scale {scale:e}");
            assert!(
                (r.variance - base.variance).abs() <= 1e-12 * base.variance,
                "scale {scale:e}: variance {} vs {}",
                r.variance,
                base.variance
            );
            assert!(
                (r.p_value - base.p_value).abs() <= 1e-12,
                "scale {scale:e}: p {} vs {}",
                r.p_value,
                base.p_value
            );
            assert!(
                (r.sen_slope / scale - base.sen_slope).abs() <= 1e-9,
                "scale {scale:e}: slope"
            );
        }
    }

    /// The rank tests tie values that are equal, whatever the unit: the same
    /// samples in volts or in picovolts rank, tie and test the same.
    #[test]
    fn rank_tests_do_not_depend_on_the_unit() {
        let a = [1.0, 2.0, 2.0, 3.5, 4.0, 4.0, 5.5, 6.0];
        let b = [2.0, 3.0, 3.5, 5.0, 5.5, 7.0, 7.5, 8.0, 8.0];
        let c = [0.5, 1.5, 2.0, 2.5, 3.0];
        let paired_x = [1.5, 2.5, 3.0, 4.5, 5.0, 6.5, 7.0, 8.5];
        let paired_y = [1.0, 2.0, 3.5, 4.0, 5.5, 6.0, 7.5, 8.0];
        let scaled = |v: &[f64], k: f64| -> Vec<f64> { v.iter().map(|x| x * k).collect() };
        let mw = mann_whitney_u_test(&a, &b).expect("computes");
        let wx = wilcoxon_signed_rank_test(&paired_x, &paired_y).expect("computes");
        let kw = kruskal_wallis_test(&[&a, &b, &c]).expect("computes");
        for k in [1e-15, 1e-12, 1e-6, 1e6, 1e12] {
            let (sa, sb, sc) = (scaled(&a, k), scaled(&b, k), scaled(&c, k));
            let got = mann_whitney_u_test(&sa, &sb).expect("computes");
            assert!(
                (got.statistic - mw.statistic).abs() < 1e-12,
                "mann-whitney U at {k:e}"
            );
            assert!(
                (got.p_value - mw.p_value).abs() < 1e-12,
                "mann-whitney p at {k:e}"
            );
            let got = wilcoxon_signed_rank_test(&scaled(&paired_x, k), &scaled(&paired_y, k))
                .expect("computes");
            assert!(
                (got.statistic - wx.statistic).abs() < 1e-12,
                "wilcoxon T at {k:e}"
            );
            assert!(
                (got.p_value - wx.p_value).abs() < 1e-12,
                "wilcoxon p at {k:e}"
            );
            let got = kruskal_wallis_test(&[&sa, &sb, &sc]).expect("computes");
            assert!(
                (got.statistic - kw.statistic).abs() < 1e-12,
                "kruskal H at {k:e}"
            );
            assert!(
                (got.p_value - kw.p_value).abs() < 1e-12,
                "kruskal p at {k:e}"
            );
        }
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
        let r = adf_test(&data, AdfModel::Constant, AdfLags::default()).expect("should compute");
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
        let r = adf_test(&walk, AdfModel::Constant, AdfLags::default()).expect("should compute");
        // Random walk should NOT reject at 1%
        assert!(
            !r.rejected[0],
            "should NOT reject at 1%: stat={}, cv={}",
            r.statistic, r.critical_values[0]
        );
    }

    #[test]
    fn adf_with_fixed_lags() {
        // A pure sinusoid plus a line obeys an exact second-order recurrence, so
        // two lags fit its differences exactly (an undefined t-ratio); the noise
        // makes it a series the test can be run on.
        let mut rng = mulberry32(0xadf2);
        let data: Vec<f64> = (0..50)
            .map(|i| (i as f64 * 0.5).sin() + 0.02 * i as f64 + 0.1 * (rng() - 0.5))
            .collect();
        let r = adf_test(&data, AdfModel::Constant, AdfLags::Fixed(2)).expect("should compute");
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
        let r =
            adf_test(&data, AdfModel::ConstantTrend, AdfLags::default()).expect("should compute");
        assert!(r.statistic.is_finite());
        assert_eq!(r.critical_values.len(), 3);
    }

    #[test]
    fn adf_edge_cases() {
        assert_eq!(
            adf_test(&[1.0; 9], AdfModel::Constant, AdfLags::default()).unwrap_err(),
            TimeSeriesError::TooFewObservations { needed: 10, got: 9 }
        );
        let mut data = vec![0.0; 20];
        data[5] = f64::NAN;
        assert_eq!(
            adf_test(&data, AdfModel::Constant, AdfLags::default()).unwrap_err(),
            TimeSeriesError::ValueNotFinite { index: 5 }
        );
        // 20 values under a constant: room for 0 to min(14, 8) lags.
        let walk: Vec<f64> = (0..20).map(|i| ((i * 37) % 11) as f64).collect();
        assert_eq!(adf_lags_domain(&walk, AdfModel::Constant).unwrap(), 0..=8);
        assert_eq!(
            adf_lags_domain(&walk, AdfModel::ConstantTrend).unwrap(),
            0..=7
        );
        assert_eq!(
            adf_lags_domain(&walk[..9], AdfModel::Constant).unwrap_err(),
            TimeSeriesError::TooFewObservations { needed: 10, got: 9 }
        );
        for lags in [AdfLags::Fixed(9), AdfLags::Aic { max: Some(9) }] {
            let option = if matches!(lags, AdfLags::Fixed(_)) {
                "lags"
            } else {
                "max_lags"
            };
            assert_eq!(
                adf_test(&walk, AdfModel::Constant, lags).unwrap_err(),
                TimeSeriesError::OptionOutOfRange {
                    option,
                    got: 9.0,
                    min: 0.0,
                    max: 8.0
                }
            );
        }
        assert_eq!(
            adf_test(&[3.0; 20], AdfModel::Constant, AdfLags::default()).unwrap_err(),
            TimeSeriesError::DegenerateRegression
        );
    }

    /// Differences that repeat with period 10 are fitted exactly by 9 lags. The
    /// residual variance is rounding noise, and the t-ratio was -1.9e8 -- a
    /// "rejection" of the unit root at every level for what is a random walk
    /// with a periodic drift. The exact fit is undefined, and the lag search
    /// must not choose it.
    #[test]
    fn adf_exact_fit_is_undefined_and_not_chosen() {
        let shocks = [0.5, -0.8, 0.3, -0.6, 0.9, -0.4, 0.7, -0.2, 0.1, -0.5];
        let mut walk = vec![0.0_f64; 60];
        for i in 1..walk.len() {
            walk[i] = walk[i - 1] + shocks[(i * 7) % 10];
        }
        assert_eq!(
            adf_test(&walk, AdfModel::ConstantTrend, AdfLags::Fixed(9)).unwrap_err(),
            TimeSeriesError::DegenerateRegression
        );
        let r = adf_test(&walk, AdfModel::ConstantTrend, AdfLags::default())
            .expect("a lag below 9 fits");
        assert!(r.n_lags < 9, "chose the exact fit: {} lags", r.n_lags);
        assert!(r.statistic.abs() < 1e3, "t = {}", r.statistic);
    }

    /// statsmodels 0.15 `adfuller` on the same series: the AIC search fits
    /// every candidate on one sample and refits the chosen count on all the
    /// observations it leaves (the statistic, the count and `nobs` agree).
    #[test]
    fn adf_matches_statsmodels() {
        let mut x = vec![0.0_f64; 120];
        for t in 1..x.len() {
            let e = ((t * 7919) % 101) as f64 / 50.0 - 1.0;
            x[t] = 0.7 * x[t - 1] + e;
        }
        for (model, lags, stat, n_lags, n_obs) in [
            (
                AdfModel::Constant,
                AdfLags::Aic { max: Some(8) },
                -3.436_052_894_8,
                5,
                114,
            ),
            (
                AdfModel::Constant,
                AdfLags::Aic { max: None },
                -3.436_052_894_8,
                5,
                114,
            ),
            (
                AdfModel::ConstantTrend,
                AdfLags::Aic { max: Some(6) },
                -3.505_247_069_0,
                5,
                114,
            ),
            (
                AdfModel::None,
                AdfLags::Aic { max: Some(5) },
                -3.459_240_021_9,
                5,
                114,
            ),
            (
                AdfModel::Constant,
                AdfLags::Fixed(3),
                -4.652_930_484_4,
                3,
                116,
            ),
        ] {
            let r = adf_test(&x, model, lags).expect("fits");
            assert_eq!((r.n_lags, r.n_obs), (n_lags, n_obs), "{model:?} {lags:?}");
            assert!(
                (r.statistic - stat).abs() < 1e-8,
                "{model:?} {lags:?}: {}",
                r.statistic
            );
        }
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

/// The tests against values SciPy 1.17.1 computes for the same data and the
/// same variant (`scipy.stats` call named at each value).
#[cfg(test)]
mod scipy_reference {
    use super::*;

    const A: [f64; 8] = [5.1, 4.9, 5.6, 5.8, 6.0, 5.2, 4.7, 5.5];
    const B: [f64; 7] = [6.3, 5.9, 6.8, 7.1, 6.0, 6.6, 7.4];
    const G1: [f64; 5] = [23.1, 25.4, 22.8, 26.0, 24.3];
    const G2: [f64; 6] = [27.2, 28.9, 26.5, 29.4, 27.8, 28.1];
    const G3: [f64; 4] = [22.0, 21.4, 24.9, 23.3];

    /// `got` within `rel` of `want`, relative (absolute below 1).
    fn close(got: f64, want: f64, rel: f64, what: &str) {
        let scale = want.abs().max(1.0);
        assert!(
            (got - want).abs() <= rel * scale,
            "{what}: got {got}, SciPy {want} (relative tolerance {rel})"
        );
    }

    #[test]
    fn t_tests_reproduce_scipy() {
        // ttest_1samp(A, 5.0)
        let r = one_sample_t_test(&A, 5.0).expect("valid");
        close(r.statistic, 2.197950389679656, 1e-12, "one-sample t");
        close(r.p_value, 0.06392412293886501, 1e-9, "one-sample p");
        assert_eq!(r.df, 7.0);
        // ttest_ind(A, B, equal_var=False)
        let r = two_sample_t_test(&A, &B).expect("valid");
        close(r.statistic, -4.67567567567568, 1e-12, "Welch t");
        close(r.df, 11.56763357082458, 1e-12, "Welch df");
        close(r.p_value, 0.0005912510074711228, 1e-9, "Welch p");
        // ttest_rel(X, Y)
        let x = [12.1, 13.4, 11.8, 14.2, 12.9, 13.0];
        let y = [12.9, 13.9, 12.0, 15.1, 13.5, 13.2];
        let r = paired_t_test(&x, &y).expect("valid");
        close(r.statistic, -4.437601569801824, 1e-12, "paired t");
        close(r.p_value, 0.006779965572779817, 1e-9, "paired p");
        assert_eq!(r.df, 5.0);
    }

    #[test]
    fn anova_and_equal_variance_tests_reproduce_scipy() {
        let groups: [&[f64]; 3] = [&G1, &G2, &G3];
        // f_oneway(G1, G2, G3)
        let r = one_way_anova(&groups).expect("valid");
        close(r.f_statistic, 20.526701958197258, 1e-12, "ANOVA F");
        close(r.p_value, 0.00013390844134272023, 1e-9, "ANOVA p");
        // levene(G1, G2, G3, center="median")
        let r = levene_test(&groups).expect("valid");
        close(r.statistic, 0.4752743705616534, 1e-12, "Brown-Forsythe W");
        close(r.p_value, 0.6329340379295331, 1e-9, "Brown-Forsythe p");
        // bartlett(G1, G2, G3)
        let r = bartlett_test(&groups).expect("valid");
        close(r.statistic, 0.5282298427149905, 1e-12, "Bartlett T");
        close(r.p_value, 0.7678852881660961, 1e-9, "Bartlett p");
    }

    #[test]
    fn kruskal_wallis_with_ties_reproduces_scipy() {
        // kruskal([1, 2, 2, 3, 5], [4, 4, 6, 7], [5, 8, 8, 9, 10])
        let r = kruskal_wallis_test(&[
            &[1.0, 2.0, 2.0, 3.0, 5.0],
            &[4.0, 4.0, 6.0, 7.0],
            &[5.0, 8.0, 8.0, 9.0, 10.0],
        ])
        .expect("valid");
        close(r.statistic, 9.22394678492239, 1e-12, "H (tie-corrected)");
        close(r.p_value, 0.009932198829456881, 1e-9, "Kruskal-Wallis p");
    }

    #[test]
    fn chi_squared_and_fisher_tests_reproduce_scipy() {
        // chisquare([18, 22, 30, 30], [25, 25, 25, 25])
        let r = chi_squared_goodness_of_fit(&[18.0, 22.0, 30.0, 30.0], &[25.0; 4]).expect("valid");
        close(r.statistic, 4.32, 1e-12, "goodness-of-fit chi-squared");
        close(r.p_value, 0.22891886433610517, 1e-9, "goodness-of-fit p");
        // chi2_contingency([[12, 7, 9], [5, 14, 11]], correction=False)
        let r = chi_squared_independence(&[12.0, 7.0, 9.0, 5.0, 14.0, 11.0], 2, 3).expect("valid");
        close(
            r.statistic,
            5.353085901027075,
            1e-12,
            "independence chi-squared",
        );
        close(r.p_value, 0.06880059054935043, 1e-9, "independence p");
        assert_eq!(r.df, 2.0);
        // fisher_exact([[8, 2], [1, 5]]) -- two-sided
        let r = fisher_exact_test(8, 2, 1, 5).expect("valid");
        close(r.p_value, 0.034965034965034975, 1e-9, "Fisher p");
    }

    #[test]
    fn shapiro_wilk_reproduces_scipy() {
        // shapiro([148, 154, 158, 160, 161, 162, 166, 170, 182, 195, 236])
        let r = shapiro_wilk_test(&[
            148.0, 154.0, 158.0, 160.0, 161.0, 162.0, 166.0, 170.0, 182.0, 195.0, 236.0,
        ])
        .expect("valid");
        close(r.w, 0.7888146948631716, 1e-9, "W, skewed");
        close(r.p_value, 0.006703814061898823, 1e-9, "p, skewed");
        // shapiro([2.1, 3.4, 1.9, 5.6, 4.4, 3.3, 2.8, 4.0, 3.9, 3.1, 2.6, 4.8, 3.6, 3.0, 4.2])
        let r = shapiro_wilk_test(&[
            2.1, 3.4, 1.9, 5.6, 4.4, 3.3, 2.8, 4.0, 3.9, 3.1, 2.6, 4.8, 3.6, 3.0, 4.2,
        ])
        .expect("valid");
        close(r.w, 0.9864343014896592, 1e-9, "W, near normal");
        close(r.p_value, 0.9958703691266937, 1e-9, "p, near normal");
    }

    #[test]
    fn mann_kendall_tau_and_sen_slope_reproduce_scipy() {
        // kendalltau(range(12), MK) and theilslopes(MK, range(12)) -- no ties,
        // so Kendall's tau-b is the Mann-Kendall S over n(n-1)/2.
        let mk = [4.2, 4.0, 4.6, 4.9, 4.7, 5.3, 5.1, 5.8, 5.6, 6.2, 6.0, 6.7];
        let r = mann_kendall_test(&mk).expect("valid");
        close(r.kendall_tau, 0.8484848484848484, 1e-12, "Kendall tau");
        close(r.sen_slope, 0.22500000000000003, 1e-12, "Sen's slope");
    }

    #[test]
    fn benjamini_hochberg_reproduces_scipy() {
        // false_discovery_control([0.01, 0.04, 0.03, 0.005, 0.20, 0.50], method="bh")
        let adjusted = benjamini_hochberg(&[0.01, 0.04, 0.03, 0.005, 0.20, 0.50]).expect("valid");
        for (got, want) in adjusted.iter().zip([0.03, 0.06, 0.06, 0.03, 0.24, 0.5]) {
            close(*got, want, 1e-12, "BH-adjusted p");
        }
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

/// Bonferroni's adjustment is `min(p m, 1)` (Dunn 1961); the expected values
/// are that definition worked by hand, not another library's output.
#[cfg(test)]
mod definition_reference {
    use super::*;

    #[test]
    fn bonferroni_is_p_times_m_capped_at_one() {
        let adjusted = bonferroni_correction(&[0.01, 0.02, 0.04, 0.3, 0.0004]).expect("valid");
        let want = [0.05, 0.1, 0.2, 1.0, 0.002];
        assert_eq!(adjusted.len(), want.len());
        for (got, want) in adjusted.iter().zip(want) {
            assert!((got - want).abs() <= 1e-15, "got {got}, want {want}");
        }
    }
}

/// P-values far in the upper tail against SciPy 1.17.1 -- sizes at which
/// `1 - cdf` has no digits left (the smallest here is 3e-31).
#[cfg(test)]
mod scipy_tail_reference {
    use super::*;

    fn close(got: f64, want: f64, what: &str) {
        assert!(
            (got - want).abs() <= 1e-9 * want.abs(),
            "{what}: got {got:e}, SciPy {want:e}"
        );
    }

    #[test]
    fn t_tests_keep_their_tail() {
        let a = [100.2, 99.8, 100.5, 100.1, 99.9, 100.3, 100.0, 99.7];
        let b = [120.1, 119.7, 120.4, 120.0, 119.8, 120.3];
        // ttest_1samp(a, 90.0)
        let r = one_sample_t_test(&a, 90.0).expect("valid");
        close(r.p_value, 1.6824197797300066e-12, "one-sample p");
        // ttest_ind(a, b, equal_var=False)
        let r = two_sample_t_test(&a, &b).expect("valid");
        close(r.p_value, 8.838218005160137e-19, "Welch p");
    }

    #[test]
    fn f_and_chi_squared_tests_keep_their_tail() {
        // f_oneway(g1, g2, g3)
        let (g1, g2, g3) = (
            [10.1, 10.3, 9.9, 10.0, 10.2],
            [12.0, 12.2, 11.9, 12.1, 12.3],
            [14.1, 13.9, 14.0, 14.2, 13.8],
        );
        let r = one_way_anova(&[&g1, &g2, &g3]).expect("valid");
        close(r.p_value, 2.297566098988527e-13, "ANOVA p");
        // chi2_contingency([[80, 20], [30, 70]], correction=False)
        let r = chi_squared_independence(&[80.0, 20.0, 30.0, 70.0], 2, 2).expect("valid");
        close(r.p_value, 1.1885852395646684e-12, "independence p");
        // chisquare([90, 10, 5, 95], [50, 50, 50, 50])
        let r = chi_squared_goodness_of_fit(&[90.0, 10.0, 5.0, 95.0], &[50.0; 4]).expect("valid");
        close(r.p_value, 3.1567404374222543e-31, "goodness-of-fit p");
        // kruskal(1..=10, 11..=20, 21..=30)
        let low: Vec<f64> = (1..=10).map(f64::from).collect();
        let mid: Vec<f64> = (11..=20).map(f64::from).collect();
        let high: Vec<f64> = (21..=30).map(f64::from).collect();
        let r = kruskal_wallis_test(&[&low, &mid, &high]).expect("valid");
        close(r.p_value, 2.49000508218493e-06, "Kruskal-Wallis p");
    }
}
