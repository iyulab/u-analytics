//! Maximum Likelihood Estimation (MLE) for Weibull parameters.
//!
//! Uses the Newton-Raphson method to find the shape parameter beta that
//! maximizes the Weibull log-likelihood, then derives the scale parameter
//! analytically. Right-censored observations (units still running --
//! suspensions) enter the likelihood through their survival term.

use u_numflow::special::inverse_normal_cdf;

/// Result of Weibull MLE fitting.
#[derive(Debug, Clone)]
pub struct WeibullMleResult {
    /// Shape parameter (beta).
    pub shape: f64,
    /// Scale parameter (eta).
    pub scale: f64,
    /// Log-likelihood at the fitted parameters.
    pub log_likelihood: f64,
    /// Number of Newton-Raphson iterations used.
    pub iterations: usize,
    /// Asymptotic covariance of `(shape, scale)`: the inverse of the observed
    /// Fisher information at the fit, censored units included. Row and column
    /// order is shape, scale.
    pub covariance: [[f64; 2]; 2],
}

/// A two-sided confidence interval.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WeibullInterval {
    /// Lower confidence limit.
    pub lower: f64,
    /// Upper confidence limit.
    pub upper: f64,
}

impl WeibullMleResult {
    /// Two-sided interval for the shape at `confidence` (in (0, 1)), normal
    /// on the log scale: `β̂ · exp(∓ z · se(β̂) / β̂)` -- the usual Weibull
    /// practice, which keeps the limits positive. `None` outside (0, 1).
    pub fn shape_interval(&self, confidence: f64) -> Option<WeibullInterval> {
        log_normal_interval(self.shape, self.covariance[0][0], confidence)
    }

    /// Two-sided interval for the scale (characteristic life) at
    /// `confidence`, normal on the log scale. `None` outside (0, 1).
    pub fn scale_interval(&self, confidence: f64) -> Option<WeibullInterval> {
        log_normal_interval(self.scale, self.covariance[1][1], confidence)
    }

    /// The B-life -- the time by which `fraction_failed` of the population
    /// has failed (0.1 → B10) -- with its two-sided interval at `confidence`,
    /// by the delta method on `ln t_p = ln η + ln(−ln(1 − p)) / β`.
    ///
    /// Returns `(estimate, interval)`, or `None` when `fraction_failed` or
    /// `confidence` is outside (0, 1).
    ///
    /// # Examples
    ///
    /// ```
    /// use u_analytics::weibull::weibull_mle_censored;
    ///
    /// let fit = weibull_mle_censored(
    ///     &[55.0, 187.0, 216.0, 240.0, 244.0, 335.0],
    ///     &[361.0, 373.0, 375.0, 386.0],
    /// )
    /// .unwrap();
    /// let (b10, ci) = fit.b_life_interval(0.10, 0.90).unwrap();
    /// assert!(ci.lower < b10 && b10 < ci.upper);
    /// ```
    pub fn b_life_interval(
        &self,
        fraction_failed: f64,
        confidence: f64,
    ) -> Option<(f64, WeibullInterval)> {
        super::ReliabilityAnalysis::from_mle(self).b_life_interval(fraction_failed, confidence)
    }
}

/// The two-sided standard normal quantile for `confidence`, or `None`
/// outside (0, 1).
pub(super) fn two_sided_z(confidence: f64) -> Option<f64> {
    (confidence > 0.0 && confidence < 1.0)
        .then(|| inverse_normal_cdf(1.0 - (1.0 - confidence) / 2.0))
}

fn log_normal_interval(estimate: f64, variance: f64, confidence: f64) -> Option<WeibullInterval> {
    let z = two_sided_z(confidence)?;
    let half = z * variance.max(0.0).sqrt() / estimate;
    Some(WeibullInterval {
        lower: estimate * (-half).exp(),
        upper: estimate * half.exp(),
    })
}

/// Maximum Newton-Raphson iterations.
const MAX_ITER: usize = 100;

/// Convergence tolerance for Newton-Raphson.
const TOL: f64 = 1e-10;

/// Fits Weibull distribution parameters (shape β, scale η) via Maximum Likelihood Estimation.
///
/// Given failure times t₁, …, tₙ (all positive), finds the shape (β) and scale (η)
/// parameters that maximise the Weibull log-likelihood:
///
/// ```text
/// L(β, η) = n·ln(β) - n·β·ln(η) + (β-1)·Σln(tᵢ) - Σ(tᵢ/η)^β
/// ```
///
/// The MLE for η given β is derived analytically:
///
/// ```text
/// η_hat = (Σ tᵢ^β / n)^(1/β)
/// ```
///
/// Substituting into the profile log-likelihood yields the score equation for β:
///
/// ```text
/// f(β) = n/β + Σln(tᵢ) - n · Σ(tᵢ^β · ln(tᵢ)) / Σtᵢ^β = 0
/// ```
///
/// Newton-Raphson solves f(β) = 0. The derivative is:
///
/// ```text
/// f′(β) = -n/β² - n · [S₂·S₀ - S₁²] / S₀²
/// ```
///
/// where S₀ = Σtᵢ^β, S₁ = Σtᵢ^β·ln(tᵢ), S₂ = Σtᵢ^β·(ln(tᵢ))².
///
/// Iteration starts from β₀ = 1.2 (slightly above the exponential case β = 1).
///
/// # Reference
/// Lawless, J.F. (2003). *Statistical Models and Methods for Lifetime Data*, 2nd ed., §4.2.
/// Wiley-Interscience. ISBN 0-471-37215-3.
///
/// # Complexity
/// O(n·k) where n = sample size, k = Newton-Raphson iterations (typically < 20)
///
/// # Arguments
/// * `failure_times` - Positive failure times (must have at least 2 values)
///
/// # Returns
/// `None` if data is insufficient (< 2 values), any value is non-positive,
/// or the algorithm does not converge within 100 iterations.
///
/// # Panics
/// Does not panic. All error conditions return `None`.
///
/// # Examples
///
/// ```
/// use u_analytics::weibull::weibull_mle;
/// let data = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
/// let result = weibull_mle(&data).unwrap();
/// assert!(result.shape > 0.0);
/// assert!(result.scale > 0.0);
/// assert!(result.log_likelihood.is_finite());
/// ```
pub fn weibull_mle(failure_times: &[f64]) -> Option<WeibullMleResult> {
    weibull_mle_censored(failure_times, &[])
}

/// Weibull MLE from failure times and right-censored times (suspensions).
///
/// A suspension is a unit removed, or still running, at time `c` without
/// having failed: it contributes its survival probability `exp(-(c/η)^β)` to
/// the likelihood rather than a density. Dropping suspensions -- fitting the
/// failures alone -- biases the scale low (the survivors' running time is
/// ignored) and with it every B-life and MTBF taken from the fit. With `r`
/// failures and the sums over **every** unit (failures and suspensions):
///
/// ```text
/// L(β, η) = r·ln(β) − r·β·ln(η) + (β−1)·Σ_fail ln(tᵢ) − Σ_all (tᵢ/η)^β
/// η_hat   = (Σ_all tᵢ^β / r)^(1/β)
/// f(β)    = r/β + Σ_fail ln(tᵢ) − r · S₁/S₀ = 0
/// ```
///
/// with `S₀ = Σ_all tᵢ^β`, `S₁ = Σ_all tᵢ^β·ln(tᵢ)` -- the same Newton-Raphson
/// iteration as [`weibull_mle`], which is this function with no suspensions.
///
/// # Reference
/// Lawless, J.F. (2003). *Statistical Models and Methods for Lifetime Data*,
/// 2nd ed., §4.2 (censored Weibull likelihood). Abernethy, R.B. (2006).
/// *The New Weibull Handbook*, 5th ed., ch. 5 (suspensions).
///
/// # Returns
/// `None` if there are fewer than 2 failures, any time is non-positive or
/// non-finite, or the iteration does not converge.
///
/// # Examples
///
/// ```
/// use u_analytics::weibull::{weibull_mle, weibull_mle_censored};
/// let failures = [120.0, 190.0, 260.0, 310.0, 400.0];
/// let survivors = [450.0, 450.0, 450.0]; // still running at 450 h
///
/// let fit = weibull_mle_censored(&failures, &survivors).unwrap();
/// let ignored = weibull_mle(&failures).unwrap();
/// // The survivors' running time lengthens the characteristic life.
/// assert!(fit.scale > ignored.scale);
/// ```
pub fn weibull_mle_censored(
    failure_times: &[f64],
    suspension_times: &[f64],
) -> Option<WeibullMleResult> {
    let r = failure_times.len();
    if r < 2 {
        return None;
    }

    // Validate: all values must be positive and finite
    if !failure_times
        .iter()
        .chain(suspension_times)
        .all(|&t| t.is_finite() && t > 0.0)
    {
        return None;
    }

    // Every unit's time and its log -- the sums S0..S2 run over all of them;
    // only the failures' logs enter the density term.
    let all: Vec<f64> = failure_times
        .iter()
        .chain(suspension_times)
        .copied()
        .collect();
    let ln_t: Vec<f64> = all.iter().map(|t| t.ln()).collect();
    let sum_ln_t: f64 = ln_t[..r].iter().sum();
    let n_f = r as f64;

    // Newton-Raphson to solve the profile likelihood equation for beta.
    // f(beta) = n/beta + sum(ln(t_i)) - n * S1 / S0
    // where S0 = sum(t_i^beta), S1 = sum(t_i^beta * ln(t_i))
    //
    // f'(beta) = -n/beta^2 - n * [S2*S0 - S1^2] / S0^2
    // where S2 = sum(t_i^beta * (ln(t_i))^2)

    let mut beta = 1.2_f64; // Initial guess slightly above exponential
    let mut iterations = 0;

    for iter in 0..MAX_ITER {
        iterations = iter + 1;

        // Compute S0, S1, S2
        let mut s0 = 0.0_f64;
        let mut s1 = 0.0_f64;
        let mut s2 = 0.0_f64;

        for (i, &t) in all.iter().enumerate() {
            let t_beta = t.powf(beta);
            let lt = ln_t[i];
            s0 += t_beta;
            s1 += t_beta * lt;
            s2 += t_beta * lt * lt;
        }

        if s0 == 0.0 {
            return None;
        }

        let f_val = n_f / beta + sum_ln_t - n_f * s1 / s0;
        let f_prime = -n_f / (beta * beta) - n_f * (s2 * s0 - s1 * s1) / (s0 * s0);

        if f_prime.abs() < 1e-30 {
            return None; // Derivative too small, cannot proceed
        }

        let delta = f_val / f_prime;
        beta -= delta;

        // Ensure beta stays positive
        if beta <= 0.0 {
            beta = 0.01;
        }

        if delta.abs() < TOL {
            break;
        }

        if iter == MAX_ITER - 1 {
            return None; // Did not converge
        }
    }

    // Compute scale parameter: eta = (sum over all units of t_i^beta / r)^(1/beta)
    let s0: f64 = all.iter().map(|t| t.powf(beta)).sum();
    let eta = (s0 / n_f).powf(1.0 / beta);

    if !eta.is_finite() || eta <= 0.0 {
        return None;
    }

    // Compute log-likelihood at the fitted parameters
    let log_likelihood = n_f * beta.ln() - n_f * beta * eta.ln() + (beta - 1.0) * sum_ln_t
        - all.iter().map(|&t| (t / eta).powf(beta)).sum::<f64>();

    let covariance = observed_covariance(&all, r, beta, eta)?;

    Some(WeibullMleResult {
        shape: beta,
        scale: eta,
        log_likelihood,
        iterations,
        covariance,
    })
}

/// Inverse of the observed Fisher information of the censored Weibull
/// log-likelihood at `(beta, eta)`. With `z = (t/η)^β` and `y = ln(t/η)` over
/// every unit, and `r` failures:
///
/// ```text
/// −∂²ℓ/∂β²   = r/β² + Σ z·y²
/// −∂²ℓ/∂η²   = −rβ/η² + β(β+1)/η² · Σ z
/// −∂²ℓ/∂β∂η  = r/η − Σ z/η − (β/η) · Σ z·y
/// ```
///
/// `None` when the information is not positive definite (no curvature to
/// invert -- a fit on a likelihood ridge).
fn observed_covariance(all: &[f64], r: usize, beta: f64, eta: f64) -> Option<[[f64; 2]; 2]> {
    let r = r as f64;
    let (mut sz, mut szy, mut szy2) = (0.0, 0.0, 0.0);
    for &t in all {
        let y = (t / eta).ln();
        let z = (beta * y).exp();
        sz += z;
        szy += z * y;
        szy2 += z * y * y;
    }
    let i_bb = r / (beta * beta) + szy2;
    let i_ee = -r * beta / (eta * eta) + beta * (beta + 1.0) / (eta * eta) * sz;
    let i_be = r / eta - sz / eta - beta / eta * szy;
    let det = i_bb * i_ee - i_be * i_be;
    if !(det > 0.0 && i_bb > 0.0 && det.is_finite()) {
        return None;
    }
    Some([[i_ee / det, -i_be / det], [-i_be / det, i_bb / det]])
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Intervals -------------------------------------------------------

    /// A complete sample at the Weibull quantiles: the observed information
    /// approaches the expected one, whose inverse is known in closed form --
    /// n·Var(β̂)/β² → 6/π² = 0.6079 and n·Var(η̂)·β²/η² → 1 + 6(1 − γ)²/π²
    /// = 1.1087 (Cohen 1965; γ Euler's constant).
    #[test]
    fn covariance_approaches_the_asymptotic_weibull_variances() {
        let (beta, eta, n) = (2.0_f64, 100.0_f64, 2000usize);
        let data: Vec<f64> = (1..=n)
            .map(|i| {
                let p = (i as f64 - 0.5) / n as f64;
                eta * (-(1.0 - p).ln()).powf(1.0 / beta)
            })
            .collect();
        let fit = weibull_mle(&data).unwrap();
        let nf = n as f64;
        let vb = nf * fit.covariance[0][0] / (fit.shape * fit.shape);
        let ve = nf * fit.covariance[1][1] * fit.shape * fit.shape / (fit.scale * fit.scale);
        let gamma = 0.577_215_664_901_532_9_f64;
        let pi2 = std::f64::consts::PI.powi(2);
        assert!((vb - 6.0 / pi2).abs() < 0.02, "{vb}");
        assert!(
            (ve - (1.0 + 6.0 * (1.0 - gamma).powi(2) / pi2)).abs() < 0.03,
            "{ve}"
        );
    }

    /// The covariance is the inverse of the negated Hessian of the likelihood
    /// written out independently, by finite differences.
    #[test]
    fn covariance_is_the_inverse_observed_information() {
        let fit = weibull_mle_censored(&FAILURES, &SUSPENSIONS).unwrap();
        let (b, e) = (fit.shape, fit.scale);
        let l = |b: f64, e: f64| censored_loglik(&FAILURES, &SUSPENSIONS, b, e);
        let (hb, he) = (1e-4, 1e-2);
        let d_bb = (l(b + hb, e) - 2.0 * l(b, e) + l(b - hb, e)) / (hb * hb);
        let d_ee = (l(b, e + he) - 2.0 * l(b, e) + l(b, e - he)) / (he * he);
        let d_be = (l(b + hb, e + he) - l(b + hb, e - he) - l(b - hb, e + he) + l(b - hb, e - he))
            / (4.0 * hb * he);
        let (i_bb, i_ee, i_be) = (-d_bb, -d_ee, -d_be);
        let det = i_bb * i_ee - i_be * i_be;
        let want = [[i_ee / det, -i_be / det], [-i_be / det, i_bb / det]];
        for (row, wrow) in fit.covariance.iter().zip(&want) {
            for (got, w) in row.iter().zip(wrow) {
                assert!(
                    (got - w).abs() <= 1e-3 * w.abs().max(1e-6),
                    "{:?} vs {want:?}",
                    fit.covariance
                );
            }
        }
    }

    #[test]
    fn intervals_bracket_and_widen() {
        let fit = weibull_mle_censored(&FAILURES, &SUSPENSIONS).unwrap();
        let (s90, s99) = (
            fit.shape_interval(0.90).unwrap(),
            fit.shape_interval(0.99).unwrap(),
        );
        assert!(
            s99.lower < s90.lower
                && s90.lower < fit.shape
                && fit.shape < s90.upper
                && s90.upper < s99.upper
        );
        let c = fit.scale_interval(0.95).unwrap();
        assert!(c.lower < fit.scale && fit.scale < c.upper && c.lower > 0.0);
        let (b10, ci) = fit.b_life_interval(0.10, 0.95).unwrap();
        assert!(ci.lower < b10 && b10 < ci.upper);
        // B10 is the 10 % quantile of the fitted distribution.
        let want = fit.scale * (-(0.9_f64).ln()).powf(1.0 / fit.shape);
        assert!((b10 - want).abs() < 1e-9 * want);
        assert!(fit.shape_interval(1.0).is_none() && fit.b_life_interval(0.0, 0.9).is_none());
    }

    // --- Right censoring ------------------------------------------------

    /// The censored log-likelihood, written out independently of the fit.
    fn censored_loglik(f: &[f64], s: &[f64], beta: f64, eta: f64) -> f64 {
        let r = f.len() as f64;
        r * beta.ln() - r * beta * eta.ln() + (beta - 1.0) * f.iter().map(|t| t.ln()).sum::<f64>()
            - f.iter().chain(s).map(|t| (t / eta).powf(beta)).sum::<f64>()
    }

    const FAILURES: [f64; 6] = [55.0, 187.0, 216.0, 240.0, 244.0, 335.0];
    const SUSPENSIONS: [f64; 4] = [361.0, 373.0, 375.0, 386.0];

    /// The fit is the maximum of that likelihood: every step away from it in
    /// shape or scale lowers it, and the reported value is it.
    #[test]
    fn censored_fit_maximises_the_censored_likelihood() {
        let fit = weibull_mle_censored(&FAILURES, &SUSPENSIONS).expect("converges");
        let best = censored_loglik(&FAILURES, &SUSPENSIONS, fit.shape, fit.scale);
        assert!((fit.log_likelihood - best).abs() < 1e-9);
        for (db, de) in [
            (0.01, 0.0),
            (-0.01, 0.0),
            (0.0, 1.0),
            (0.0, -1.0),
            (0.01, 1.0),
        ] {
            let other = censored_loglik(&FAILURES, &SUSPENSIONS, fit.shape + db, fit.scale + de);
            assert!(other < best, "({db}, {de}): {other} >= {best}");
        }
    }

    /// No suspensions: exactly the uncensored fit (one path, not two).
    #[test]
    fn no_suspensions_is_the_uncensored_fit() {
        let a = weibull_mle_censored(&FAILURES, &[]).unwrap();
        let b = weibull_mle(&FAILURES).unwrap();
        assert_eq!(
            (a.shape, a.scale, a.log_likelihood),
            (b.shape, b.scale, b.log_likelihood)
        );
    }

    /// Survivors lengthen the characteristic life; later survivors more so.
    #[test]
    fn suspensions_lengthen_the_scale() {
        let without = weibull_mle(&FAILURES).unwrap().scale;
        let with = weibull_mle_censored(&FAILURES, &SUSPENSIONS).unwrap().scale;
        let later: Vec<f64> = SUSPENSIONS.iter().map(|t| t * 2.0).collect();
        let longer = weibull_mle_censored(&FAILURES, &later).unwrap().scale;
        assert!(without < with && with < longer, "{without} {with} {longer}");
    }

    /// Exponential check: with β fixed at 1 the censored MLE of the mean is
    /// total time on test over failures. The fitted shape is not exactly 1,
    /// so compare the score at β = 1 instead: zero exactly when η = TTT / r.
    #[test]
    fn exponential_special_case_is_time_on_test_over_failures() {
        let ttt: f64 = FAILURES.iter().chain(&SUSPENSIONS).sum();
        let eta = ttt / FAILURES.len() as f64;
        // d/dη of the log-likelihood at β = 1: −r/η + Σ t / η² = 0 at η = TTT / r.
        let score = -(FAILURES.len() as f64) / eta + ttt / (eta * eta);
        assert!(score.abs() < 1e-12);
        let h = 1e-6;
        let numeric = (censored_loglik(&FAILURES, &SUSPENSIONS, 1.0, eta + h)
            - censored_loglik(&FAILURES, &SUSPENSIONS, 1.0, eta - h))
            / (2.0 * h);
        assert!(numeric.abs() < 1e-6, "{numeric}");
    }

    #[test]
    fn censored_input_is_checked() {
        assert!(weibull_mle_censored(&[10.0], &[20.0, 30.0]).is_none()); // one failure
        assert!(weibull_mle_censored(&[10.0, 20.0], &[0.0]).is_none());
        assert!(weibull_mle_censored(&[10.0, 20.0], &[f64::NAN]).is_none());
    }

    #[test]
    fn test_mle_uniform_spacing() {
        // Uniformly spaced failure times
        let data = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0];
        let result = weibull_mle(&data).expect("MLE should converge");

        // Shape should be in a reasonable range for linearly spaced data
        assert!(
            result.shape > 1.5 && result.shape < 5.0,
            "shape = {}, expected in [1.5, 5.0]",
            result.shape
        );
        // Scale should be reasonable relative to the data range
        assert!(
            result.scale > 40.0 && result.scale < 100.0,
            "scale = {}, expected in [40, 100]",
            result.scale
        );
        assert!(result.log_likelihood.is_finite());
    }

    #[test]
    fn test_mle_near_exponential() {
        // For approximately exponential data, beta should be close to 1.0
        let data = [5.0, 10.0, 15.0, 25.0, 35.0, 50.0, 75.0, 100.0];
        let result = weibull_mle(&data).expect("MLE should converge");

        assert!(
            result.shape > 0.5 && result.shape < 2.0,
            "shape = {}, expected near 1.0",
            result.shape
        );
    }

    #[test]
    fn test_mle_converges_with_iterations() {
        let data = [10.0, 20.0, 30.0, 40.0, 50.0];
        let result = weibull_mle(&data).expect("MLE should converge");
        assert!(result.iterations > 0);
        assert!(result.iterations <= MAX_ITER);
    }

    #[test]
    fn test_mle_insufficient_data() {
        assert!(weibull_mle(&[]).is_none());
        assert!(weibull_mle(&[10.0]).is_none());
    }

    #[test]
    fn test_mle_invalid_data() {
        // Non-positive values
        assert!(weibull_mle(&[0.0, 10.0, 20.0]).is_none());
        assert!(weibull_mle(&[-5.0, 10.0, 20.0]).is_none());
        // NaN / Inf
        assert!(weibull_mle(&[f64::NAN, 10.0, 20.0]).is_none());
        assert!(weibull_mle(&[f64::INFINITY, 10.0, 20.0]).is_none());
    }

    #[test]
    fn test_mle_log_likelihood_is_negative() {
        let data = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0];
        let result = weibull_mle(&data).expect("MLE should converge");
        // Log-likelihood for continuous distributions is typically negative
        assert!(
            result.log_likelihood < 0.0,
            "log_likelihood = {}, expected < 0",
            result.log_likelihood
        );
    }

    #[test]
    fn test_mle_known_weibull_data() {
        // Data generated from Weibull(beta=2.0, eta=50.0) approximately
        // These are the quantiles: t_i = eta * (-ln(1 - F_i))^(1/beta)
        // with F_i = (i - 0.5) / n for i = 1..n, n=10
        let data: Vec<f64> = (1..=10)
            .map(|i| {
                let f = (i as f64 - 0.5) / 10.0;
                50.0 * (-(1.0 - f).ln()).powf(0.5) // beta=2.0, eta=50.0
            })
            .collect();

        let result = weibull_mle(&data).expect("MLE should converge");

        assert!(
            (result.shape - 2.0).abs() < 0.5,
            "shape = {}, expected near 2.0",
            result.shape
        );
        assert!(
            (result.scale - 50.0).abs() < 15.0,
            "scale = {}, expected near 50.0",
            result.scale
        );
    }

    #[test]
    fn test_mle_identical_values() {
        // All identical positive values — degenerate case
        let data = [10.0, 10.0, 10.0, 10.0, 10.0];
        // MLE may or may not converge; if it does, check validity
        if let Some(result) = weibull_mle(&data) {
            assert!(result.shape.is_finite() && result.shape > 0.0);
            assert!(result.scale.is_finite() && result.scale > 0.0);
        }
    }

    #[test]
    fn test_mle_lawless_ball_bearing() {
        // Lawless (2003), §4.2 — ball bearing failure times (million revolutions).
        // Reference MLE estimates: shape β ≈ 2.102, scale η ≈ 81.88.
        // Cross-checked with R: library(MASS); fitdistr(data, "weibull")
        let data = vec![
            17.88, 28.92, 33.0, 41.52, 42.12, 45.6, 48.48, 51.84, 51.96, 54.12, 55.56, 67.8, 68.64,
            68.64, 68.88, 84.12, 93.12, 98.64, 105.12, 105.84, 127.92, 128.04, 173.4_f64,
        ];
        let result = weibull_mle(&data).expect("MLE should converge on Lawless §4.2 data");
        assert!(
            (result.shape - 2.102).abs() < 0.01,
            "shape β = {:.6}, expected 2.102 ± 0.01",
            result.shape
        );
        assert!(
            (result.scale - 81.88).abs() < 0.5,
            "scale η = {:.6}, expected 81.88 ± 0.5",
            result.scale
        );
        assert!(result.log_likelihood.is_finite());
        assert!(
            result.iterations <= 20,
            "expected convergence in ≤ 20 iterations, got {}",
            result.iterations
        );
    }
}
