//! Maximum Likelihood Estimation (MLE) for Weibull parameters.
//!
//! Uses the Newton-Raphson method to find the shape parameter beta that
//! maximizes the Weibull log-likelihood, then derives the scale parameter
//! analytically.

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
}

/// Maximum Newton-Raphson iterations.
const MAX_ITER: usize = 100;

/// Convergence tolerance for Newton-Raphson.
const TOL: f64 = 1e-10;

/// Fit a Weibull distribution to failure time data using MLE.
///
/// Given failure times t_1, ..., t_n (all positive), this function finds
/// the shape (beta) and scale (eta) parameters that maximize the Weibull
/// log-likelihood:
///
/// ```text
/// L(beta, eta) = n*ln(beta) - n*beta*ln(eta)
///              + (beta-1)*sum(ln(t_i)) - sum((t_i/eta)^beta)
/// ```
///
/// The MLE for eta given beta is: `eta_hat = (sum(t_i^beta) / n)^(1/beta)`
///
/// Substituting into the profile log-likelihood, the equation for beta is:
///
/// ```text
/// f(beta) = n/beta + sum(ln(t_i))
///         - n * sum(t_i^beta * ln(t_i)) / sum(t_i^beta) = 0
/// ```
///
/// Newton-Raphson is used to solve f(beta) = 0 starting from beta_0 = 1.2.
///
/// # Arguments
/// * `failure_times` - Positive failure times (must have at least 2 values)
///
/// # Returns
/// `None` if data is insufficient (< 2 values), any value is non-positive,
/// or the algorithm does not converge.
///
/// # Reference
/// Lawless (2003), *Statistical Models and Methods for Lifetime Data*, 2nd ed.
pub fn weibull_mle(failure_times: &[f64]) -> Option<WeibullMleResult> {
    let n = failure_times.len();
    if n < 2 {
        return None;
    }

    // Validate: all values must be positive and finite
    if !failure_times
        .iter()
        .all(|&t| t.is_finite() && t > 0.0)
    {
        return None;
    }

    // Pre-compute ln(t_i)
    let ln_t: Vec<f64> = failure_times.iter().map(|t| t.ln()).collect();
    let sum_ln_t: f64 = ln_t.iter().sum();
    let n_f = n as f64;

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

        for (i, &t) in failure_times.iter().enumerate() {
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

    // Compute scale parameter: eta = (sum(t_i^beta) / n)^(1/beta)
    let s0: f64 = failure_times.iter().map(|t| t.powf(beta)).sum();
    let eta = (s0 / n_f).powf(1.0 / beta);

    if !eta.is_finite() || eta <= 0.0 {
        return None;
    }

    // Compute log-likelihood at the fitted parameters
    let log_likelihood = n_f * beta.ln() - n_f * beta * eta.ln()
        + (beta - 1.0) * sum_ln_t
        - failure_times
            .iter()
            .map(|&t| (t / eta).powf(beta))
            .sum::<f64>();

    Some(WeibullMleResult {
        shape: beta,
        scale: eta,
        log_likelihood,
        iterations,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
