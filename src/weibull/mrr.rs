//! Median Rank Regression (MRR) for Weibull parameter estimation.
//!
//! Fits Weibull parameters using least-squares regression on the
//! linearized Weibull probability plot.

/// Result of Weibull MRR fitting.
#[derive(Debug, Clone)]
pub struct WeibullMrrResult {
    /// Shape parameter (beta).
    pub shape: f64,
    /// Scale parameter (eta).
    pub scale: f64,
    /// Coefficient of determination (R-squared) measuring goodness of fit.
    pub r_squared: f64,
}

/// Fit Weibull distribution using Median Rank Regression.
///
/// The linearized Weibull CDF is:
///
/// ```text
/// ln(-ln(1 - F(t))) = beta * ln(t) - beta * ln(eta)
/// ```
///
/// Plotting y = ln(-ln(1 - F_i)) vs x = ln(t_i) gives a line with
/// slope beta and intercept -beta * ln(eta).
///
/// Median ranks are computed using Bernard's approximation:
///
/// ```text
/// F_i = (i - 0.3) / (n + 0.4)
/// ```
///
/// # Algorithm
/// 1. Sort failure times ascending
/// 2. Compute median ranks F_i for each rank
/// 3. Transform: x_i = ln(t_i), y_i = ln(-ln(1 - F_i))
/// 4. Fit line y = a + b*x using ordinary least squares
/// 5. beta = b, eta = exp(-a/b)
///
/// # Arguments
/// * `failure_times` - Positive failure times (must have at least 2 values)
///
/// # Returns
/// `None` if data is insufficient (< 2 values), any value is non-positive,
/// or the regression is degenerate.
///
/// # Examples
///
/// ```
/// use u_analytics::weibull::weibull_mrr;
/// let data = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
/// let result = weibull_mrr(&data).unwrap();
/// assert!(result.shape > 0.0);
/// assert!(result.scale > 0.0);
/// assert!(result.r_squared > 0.9);
/// ```
///
/// # Reference
/// Abernethy (2006), *The New Weibull Handbook*, 5th ed.
pub fn weibull_mrr(failure_times: &[f64]) -> Option<WeibullMrrResult> {
    let n = failure_times.len();
    if n < 2 {
        return None;
    }

    // Validate: all values must be positive and finite
    if !failure_times.iter().all(|&t| t.is_finite() && t > 0.0) {
        return None;
    }

    // Sort failure times ascending
    let mut sorted = failure_times.to_vec();
    sorted.sort_unstable_by(|a, b| a.partial_cmp(b).expect("NaN values already filtered above"));

    let n_f = n as f64;

    // Compute transformed coordinates
    let mut x_vals = Vec::with_capacity(n);
    let mut y_vals = Vec::with_capacity(n);

    for (i, &t) in sorted.iter().enumerate() {
        let rank = (i + 1) as f64;
        // Bernard's approximation for median ranks
        let f_i = (rank - 0.3) / (n_f + 0.4);

        let x = t.ln();
        let y = (-(1.0 - f_i).ln()).ln(); // ln(-ln(1 - F_i))

        if !x.is_finite() || !y.is_finite() {
            return None;
        }

        x_vals.push(x);
        y_vals.push(y);
    }

    // Ordinary least squares: y = a + b*x
    // b = (n*sum(x*y) - sum(x)*sum(y)) / (n*sum(x^2) - (sum(x))^2)
    // a = mean(y) - b*mean(x)
    let sum_x: f64 = x_vals.iter().sum();
    let sum_y: f64 = y_vals.iter().sum();
    let sum_xy: f64 = x_vals.iter().zip(y_vals.iter()).map(|(x, y)| x * y).sum();
    let sum_x2: f64 = x_vals.iter().map(|x| x * x).sum();
    let sum_y2: f64 = y_vals.iter().map(|y| y * y).sum();

    let denom = n_f * sum_x2 - sum_x * sum_x;
    if denom.abs() < 1e-30 {
        return None; // All x values are identical (degenerate)
    }

    let b = (n_f * sum_xy - sum_x * sum_y) / denom;
    let a = (sum_y - b * sum_x) / n_f;

    // beta = slope = b
    let beta = b;
    if beta <= 0.0 || !beta.is_finite() {
        return None;
    }

    // eta = exp(-a/b) = exp(-intercept/slope)
    let eta = (-a / b).exp();
    if !eta.is_finite() || eta <= 0.0 {
        return None;
    }

    // R-squared = 1 - SS_res / SS_tot
    let mean_y = sum_y / n_f;
    let ss_tot = sum_y2 - n_f * mean_y * mean_y;
    let ss_res = sum_y2 - 2.0 * a * sum_y - 2.0 * b * sum_xy
        + n_f * a * a
        + 2.0 * a * b * sum_x
        + b * b * sum_x2;

    let r_squared = if ss_tot.abs() < 1e-30 {
        1.0 // Perfect fit (all y values identical)
    } else {
        1.0 - ss_res / ss_tot
    };

    Some(WeibullMrrResult {
        shape: beta,
        scale: eta,
        r_squared,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mrr_uniform_spacing() {
        let data = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0];
        let result = weibull_mrr(&data).expect("MRR should succeed");

        assert!(
            result.shape > 1.0 && result.shape < 5.0,
            "shape = {}, expected in [1.0, 5.0]",
            result.shape
        );
        assert!(
            result.scale > 40.0 && result.scale < 100.0,
            "scale = {}, expected in [40, 100]",
            result.scale
        );
        assert!(
            result.r_squared > 0.9,
            "R^2 = {}, expected > 0.9",
            result.r_squared
        );
    }

    #[test]
    fn test_mrr_near_exponential() {
        let data = [5.0, 10.0, 15.0, 25.0, 35.0, 50.0, 75.0, 100.0];
        let result = weibull_mrr(&data).expect("MRR should succeed");

        assert!(
            result.shape > 0.5 && result.shape < 2.0,
            "shape = {}, expected near 1.0",
            result.shape
        );
    }

    #[test]
    fn test_mrr_insufficient_data() {
        assert!(weibull_mrr(&[]).is_none());
        assert!(weibull_mrr(&[10.0]).is_none());
    }

    #[test]
    fn test_mrr_invalid_data() {
        assert!(weibull_mrr(&[0.0, 10.0, 20.0]).is_none());
        assert!(weibull_mrr(&[-5.0, 10.0, 20.0]).is_none());
        assert!(weibull_mrr(&[f64::NAN, 10.0, 20.0]).is_none());
        assert!(weibull_mrr(&[f64::INFINITY, 10.0, 20.0]).is_none());
    }

    #[test]
    fn test_mrr_r_squared_range() {
        let data = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0];
        let result = weibull_mrr(&data).expect("MRR should succeed");
        assert!(
            result.r_squared >= 0.0 && result.r_squared <= 1.0,
            "R^2 = {}, expected in [0, 1]",
            result.r_squared
        );
    }

    #[test]
    fn test_mrr_known_weibull_data() {
        // Data generated from Weibull(beta=2.0, eta=50.0)
        let data: Vec<f64> = (1..=10)
            .map(|i| {
                let f = (i as f64 - 0.5) / 10.0;
                50.0 * (-(1.0 - f).ln()).powf(0.5)
            })
            .collect();

        let result = weibull_mrr(&data).expect("MRR should succeed");

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
        assert!(
            result.r_squared > 0.95,
            "R^2 = {}, expected > 0.95 for exact Weibull data",
            result.r_squared
        );
    }

    #[test]
    fn test_mrr_unsorted_input() {
        // MRR should sort internally; order should not matter
        let data1 = [10.0, 20.0, 30.0, 40.0, 50.0];
        let data2 = [50.0, 10.0, 40.0, 20.0, 30.0];

        let r1 = weibull_mrr(&data1).expect("MRR should succeed");
        let r2 = weibull_mrr(&data2).expect("MRR should succeed");

        assert!(
            (r1.shape - r2.shape).abs() < 1e-10,
            "shape should be order-independent"
        );
        assert!(
            (r1.scale - r2.scale).abs() < 1e-10,
            "scale should be order-independent"
        );
    }

    #[test]
    fn test_mrr_mle_agreement() {
        // MRR and MLE should give roughly similar results
        let data = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0];
        let mrr_result = weibull_mrr(&data).expect("MRR should succeed");
        let mle_result = super::super::mle::weibull_mle(&data).expect("MLE should converge");

        // They won't match exactly, but should be in the same ballpark
        assert!(
            (mrr_result.shape - mle_result.shape).abs() < 1.5,
            "MRR shape = {}, MLE shape = {}",
            mrr_result.shape,
            mle_result.shape
        );
        assert!(
            (mrr_result.scale - mle_result.scale).abs() / mle_result.scale < 0.3,
            "MRR scale = {}, MLE scale = {}",
            mrr_result.scale,
            mle_result.scale
        );
    }

    #[test]
    fn test_mrr_identical_values() {
        // All identical — x values are all equal, degenerate
        let data = [10.0, 10.0, 10.0, 10.0, 10.0];
        assert!(
            weibull_mrr(&data).is_none(),
            "MRR should return None for identical values"
        );
    }
}
