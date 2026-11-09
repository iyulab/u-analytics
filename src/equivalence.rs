//! Equivalence testing: two one-sided tests (TOST).
//!
//! A significance test can only fail to find a difference. To show that a
//! change *did not matter* — a new supplier, a process change, a replacement
//! gauge — the question is turned around: the null hypothesis is that the
//! difference is **at least** as large as a margin the user accepts, and
//! rejecting it demonstrates equivalence (Schuirmann 1987).
//!
//! For a difference `d` with equivalence margins `(low, high)`, `low < high`:
//!
//! ```text
//! H₀₁: d ≤ low   vs  H₁₁: d > low      t_L = (d̂ − low)  / se,  p_L = P(T ≥ t_L)
//! H₀₂: d ≥ high  vs  H₁₂: d < high     t_U = (d̂ − high) / se,  p_U = P(T ≤ t_U)
//! ```
//!
//! Equivalence is declared at level `α` when **both** are rejected, that is when
//! `p = max(p_L, p_U) < α` — the same as the `1 − 2α` confidence interval of `d`
//! lying inside `(low, high)`. The p-value of the procedure is `p`; it is never
//! the product or the sum of the two.
//!
//! # Examples
//!
//! ```
//! use u_analytics::equivalence::tost_one_sample;
//!
//! // Is the mean of these fill weights within ±0.5 g of the 100 g target?
//! let weights = [99.8, 100.1, 100.3, 99.9, 100.0, 100.2, 99.7, 100.1, 100.0, 99.9];
//! let r = tost_one_sample(&weights, 100.0, -0.5, 0.5, 0.05).unwrap();
//! assert!(r.equivalent);
//! assert!(r.p_value < 0.05);
//! ```
//!
//! # References
//!
//! - Schuirmann, D. J. (1987). "A comparison of the two one-sided tests
//!   procedure and the power approach for assessing the equivalence of average
//!   bioavailability". *J. Pharmacokinetics and Biopharmaceutics* 15(6), 657–680.
//! - Lakens, D. (2017). "Equivalence tests: a practical primer for t tests,
//!   correlations, and meta-analyses". *Social Psychological and Personality
//!   Science* 8(4), 355–362.

use u_numflow::special;
use u_numflow::stats;

/// The outcome of a two one-sided tests (TOST) equivalence test.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EquivalenceResult {
    /// The observed difference `d̂`: the mean (one-sample), the mean of the
    /// pairwise differences (paired) or `mean(a) − mean(b)` (two-sample).
    pub estimate: f64,
    /// Standard error of the estimate.
    pub std_error: f64,
    /// Degrees of freedom of the t distribution (fractional for Welch).
    pub df: f64,
    /// `t_L = (d̂ − low) / se`: the statistic of the test that `d > low`.
    pub t_lower: f64,
    /// `t_U = (d̂ − high) / se`: the statistic of the test that `d < high`.
    pub t_upper: f64,
    /// `P(T ≥ t_L)`, the p-value against `d ≤ low`.
    pub p_lower: f64,
    /// `P(T ≤ t_U)`, the p-value against `d ≥ high`.
    pub p_upper: f64,
    /// The p-value of the TOST procedure, `max(p_lower, p_upper)`.
    pub p_value: f64,
    /// Lower end of the `1 − 2α` confidence interval of the difference.
    pub ci_lower: f64,
    /// Upper end of the `1 − 2α` confidence interval of the difference.
    pub ci_upper: f64,
    /// Whether equivalence is declared at level `alpha` (`p_value < alpha`).
    pub equivalent: bool,
}

/// The margins and the level, checked once: `low < high`, both finite, and
/// `0 < alpha < 0.5` (at `α ≥ ½` the `1 − 2α` interval is empty).
fn check_margins(low: f64, high: f64, alpha: f64) -> bool {
    low.is_finite() && high.is_finite() && low < high && alpha > 0.0 && alpha < 0.5
}

/// Assembles the result from the estimate, its standard error and the degrees
/// of freedom: the one place the two tests, the p-value and the interval are
/// formed, so the three kinds of TOST cannot differ in them.
fn assemble(
    estimate: f64,
    std_error: f64,
    df: f64,
    low: f64,
    high: f64,
    alpha: f64,
) -> Option<EquivalenceResult> {
    if !(std_error.is_finite() && std_error > 0.0 && df.is_finite() && df > 0.0) {
        return None;
    }
    let t_lower = (estimate - low) / std_error;
    let t_upper = (estimate - high) / std_error;
    // Both tails are formed directly (`sf` of the mirrored statistic), so a
    // p-value far below 1e-16 keeps its digits.
    let p_lower = special::t_distribution_sf(t_lower, df);
    let p_upper = special::t_distribution_sf(-t_upper, df);
    let p_value = p_lower.max(p_upper);
    let critical = special::t_distribution_quantile(1.0 - alpha, df);
    let half_width = critical * std_error;
    Some(EquivalenceResult {
        estimate,
        std_error,
        df,
        t_lower,
        t_upper,
        p_lower,
        p_upper,
        p_value,
        ci_lower: estimate - half_width,
        ci_upper: estimate + half_width,
        equivalent: p_value < alpha,
    })
}

/// TOST for one mean: is `μ − target` inside `(low, high)`?
///
/// `low` and `high` are margins on the difference from `target`, so
/// `target = 100`, `low = −0.5`, `high = 0.5` asks whether the mean is within
/// ±0.5 of 100.
///
/// # Algorithm
///
/// `d̂ = x̄ − target`, `se = s/√n`, `df = n − 1`.
///
/// # Returns
///
/// `None` if there are fewer than 2 observations, a value is not finite, the
/// sample has no variation, `low ≥ high`, or `alpha` is not in `(0, 0.5)`.
pub fn tost_one_sample(
    data: &[f64],
    target: f64,
    low: f64,
    high: f64,
    alpha: f64,
) -> Option<EquivalenceResult> {
    if data.len() < 2 || !check_margins(low, high, alpha) || !target.is_finite() {
        return None;
    }
    if data.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let n = data.len() as f64;
    let estimate = stats::mean(data)? - target;
    let sd = stats::std_dev(data)?;
    assemble(estimate, sd / n.sqrt(), n - 1.0, low, high, alpha)
}

/// TOST for paired data: is the mean of `x − y` inside `(low, high)`?
///
/// # Returns
///
/// `None` if the slices differ in length, there are fewer than 2 pairs, or for
/// the reasons of [`tost_one_sample`] applied to the differences.
pub fn tost_paired(
    x: &[f64],
    y: &[f64],
    low: f64,
    high: f64,
    alpha: f64,
) -> Option<EquivalenceResult> {
    if x.len() != y.len() || x.len() < 2 {
        return None;
    }
    let differences: Vec<f64> = x.iter().zip(y).map(|(&a, &b)| a - b).collect();
    tost_one_sample(&differences, 0.0, low, high, alpha)
}

/// TOST for two independent samples: is `mean(a) − mean(b)` inside `(low, high)`?
///
/// # Algorithm
///
/// With `equal_variance = false` (Welch, the safer default for a process
/// comparison): `se = √(s₁²/n₁ + s₂²/n₂)` with the Welch–Satterthwaite
/// degrees of freedom. With `true`: the pooled variance and `df = n₁ + n₂ − 2`.
///
/// # Returns
///
/// `None` if either sample has fewer than 2 observations, a value is not
/// finite, both samples have no variation, `low ≥ high`, or `alpha` is not in
/// `(0, 0.5)`.
pub fn tost_two_sample(
    a: &[f64],
    b: &[f64],
    low: f64,
    high: f64,
    alpha: f64,
    equal_variance: bool,
) -> Option<EquivalenceResult> {
    if a.len() < 2 || b.len() < 2 || !check_margins(low, high, alpha) {
        return None;
    }
    if a.iter().chain(b).any(|v| !v.is_finite()) {
        return None;
    }
    let (n1, n2) = (a.len() as f64, b.len() as f64);
    let estimate = stats::mean(a)? - stats::mean(b)?;
    let (var1, var2) = (stats::variance(a)?, stats::variance(b)?);
    let (std_error, df) = if equal_variance {
        let pooled = ((n1 - 1.0) * var1 + (n2 - 1.0) * var2) / (n1 + n2 - 2.0);
        ((pooled * (1.0 / n1 + 1.0 / n2)).sqrt(), n1 + n2 - 2.0)
    } else {
        let (v1, v2) = (var1 / n1, var2 / n2);
        (
            (v1 + v2).sqrt(),
            (v1 + v2).powi(2) / (v1 * v1 / (n1 - 1.0) + v2 * v2 / (n2 - 1.0)),
        )
    };
    assemble(estimate, std_error, df, low, high, alpha)
}
