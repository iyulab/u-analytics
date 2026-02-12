//! Regression analysis.
//!
//! Simple and multiple linear regression with OLS, R², coefficient testing,
//! ANOVA, VIF, and residual diagnostics.
//!
//! # Examples
//!
//! ```
//! use u_analytics::regression::simple_linear_regression;
//!
//! let x = [1.0, 2.0, 3.0, 4.0, 5.0];
//! let y = [2.1, 3.9, 6.1, 7.9, 10.1];
//! let result = simple_linear_regression(&x, &y).unwrap();
//! assert!((result.slope - 2.0).abs() < 0.1);
//! assert!((result.intercept - 0.1).abs() < 0.2);
//! assert!(result.r_squared > 0.99);
//! ```

use u_numflow::matrix::Matrix;
use u_numflow::special;
use u_numflow::stats;

/// Result of a simple linear regression: y = intercept + slope · x.
#[derive(Debug, Clone)]
pub struct SimpleRegressionResult {
    /// Slope coefficient (β₁).
    pub slope: f64,
    /// Intercept (β₀).
    pub intercept: f64,
    /// Coefficient of determination (R²).
    pub r_squared: f64,
    /// Adjusted R² = 1 - (1-R²)(n-1)/(n-2).
    pub adjusted_r_squared: f64,
    /// Standard error of the slope.
    pub slope_se: f64,
    /// Standard error of the intercept.
    pub intercept_se: f64,
    /// t-statistic for slope (H₀: β₁ = 0).
    pub slope_t: f64,
    /// t-statistic for intercept (H₀: β₀ = 0).
    pub intercept_t: f64,
    /// p-value for slope.
    pub slope_p: f64,
    /// p-value for intercept.
    pub intercept_p: f64,
    /// Residual standard error (√(SSE/(n-2))).
    pub residual_se: f64,
    /// F-statistic (= t² for simple regression).
    pub f_statistic: f64,
    /// p-value for F-statistic.
    pub f_p_value: f64,
    /// Residuals (yᵢ - ŷᵢ).
    pub residuals: Vec<f64>,
    /// Fitted values (ŷᵢ).
    pub fitted: Vec<f64>,
    /// Sample size.
    pub n: usize,
}

/// Result of a multiple linear regression: y = Xβ + ε.
#[derive(Debug, Clone)]
pub struct MultipleRegressionResult {
    /// Coefficient vector [β₀, β₁, ..., βₚ] (intercept first).
    pub coefficients: Vec<f64>,
    /// Standard errors of coefficients.
    pub std_errors: Vec<f64>,
    /// t-statistics for each coefficient.
    pub t_statistics: Vec<f64>,
    /// p-values for each coefficient.
    pub p_values: Vec<f64>,
    /// Coefficient of determination (R²).
    pub r_squared: f64,
    /// Adjusted R² = 1 - (1-R²)(n-1)/(n-p-1).
    pub adjusted_r_squared: f64,
    /// F-statistic for overall significance.
    pub f_statistic: f64,
    /// p-value for F-statistic.
    pub f_p_value: f64,
    /// Residual standard error.
    pub residual_se: f64,
    /// Residuals.
    pub residuals: Vec<f64>,
    /// Fitted values.
    pub fitted: Vec<f64>,
    /// VIF (Variance Inflation Factor) for each predictor (excludes intercept).
    pub vif: Vec<f64>,
    /// Sample size.
    pub n: usize,
    /// Number of predictors (excluding intercept).
    pub p: usize,
}

// ---------------------------------------------------------------------------
// Simple Linear Regression
// ---------------------------------------------------------------------------

/// Computes simple linear regression (OLS closed-form).
///
/// # Algorithm
///
/// β₁ = cov(x,y) / var(x)
/// β₀ = ȳ - β₁·x̄
///
/// # Returns
///
/// `None` if fewer than 3 observations, slices differ in length, x has zero
/// variance, or inputs contain non-finite values.
///
/// # References
///
/// Draper & Smith (1998). "Applied Regression Analysis", 3rd edition.
///
/// # Examples
///
/// ```
/// use u_analytics::regression::simple_linear_regression;
///
/// let x = [1.0, 2.0, 3.0, 4.0, 5.0];
/// let y = [2.0, 4.0, 6.0, 8.0, 10.0];
/// let r = simple_linear_regression(&x, &y).unwrap();
/// assert!((r.slope - 2.0).abs() < 1e-10);
/// assert!((r.intercept).abs() < 1e-10);
/// assert!((r.r_squared - 1.0).abs() < 1e-10);
/// ```
pub fn simple_linear_regression(x: &[f64], y: &[f64]) -> Option<SimpleRegressionResult> {
    let n = x.len();
    if n < 3 || n != y.len() {
        return None;
    }
    if x.iter().any(|v| !v.is_finite()) || y.iter().any(|v| !v.is_finite()) {
        return None;
    }

    let x_mean = stats::mean(x)?;
    let y_mean = stats::mean(y)?;
    let x_var = stats::variance(x)?;
    let cov = stats::covariance(x, y)?;

    if x_var < 1e-300 {
        return None; // zero variance in x
    }

    let slope = cov / x_var;
    let intercept = y_mean - slope * x_mean;

    // Fitted values and residuals
    let fitted: Vec<f64> = x.iter().map(|&xi| intercept + slope * xi).collect();
    let residuals: Vec<f64> = y
        .iter()
        .zip(fitted.iter())
        .map(|(&yi, &fi)| yi - fi)
        .collect();

    // Sum of squares
    let ss_res: f64 = residuals.iter().map(|r| r * r).sum();
    let ss_tot: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum();

    let nf = n as f64;
    let df_res = nf - 2.0;

    let r_squared = if ss_tot > 1e-300 {
        1.0 - ss_res / ss_tot
    } else {
        1.0
    };
    let adjusted_r_squared = 1.0 - (1.0 - r_squared) * (nf - 1.0) / df_res;

    // Residual standard error
    let mse = ss_res / df_res;
    let residual_se = mse.sqrt();

    // Standard errors of coefficients
    let ss_x: f64 = x.iter().map(|&xi| (xi - x_mean).powi(2)).sum();
    let slope_se = (mse / ss_x).sqrt();
    let intercept_se = (mse * (1.0 / nf + x_mean * x_mean / ss_x)).sqrt();

    // t-statistics
    let slope_t = if slope_se > 1e-300 {
        slope / slope_se
    } else {
        f64::INFINITY
    };
    let intercept_t = if intercept_se > 1e-300 {
        intercept / intercept_se
    } else {
        f64::INFINITY
    };

    // p-values via t-distribution
    let slope_p = 2.0 * (1.0 - special::t_distribution_cdf(slope_t.abs(), df_res));
    let intercept_p = 2.0 * (1.0 - special::t_distribution_cdf(intercept_t.abs(), df_res));

    // F-statistic (= t² for simple regression)
    let f_statistic = slope_t * slope_t;
    let f_p_value = if f_statistic.is_infinite() {
        0.0
    } else {
        1.0 - special::f_distribution_cdf(f_statistic, 1.0, df_res)
    };

    Some(SimpleRegressionResult {
        slope,
        intercept,
        r_squared,
        adjusted_r_squared,
        slope_se,
        intercept_se,
        slope_t,
        intercept_t,
        slope_p,
        intercept_p,
        residual_se,
        f_statistic,
        f_p_value,
        residuals,
        fitted,
        n,
    })
}

// ---------------------------------------------------------------------------
// Multiple Linear Regression
// ---------------------------------------------------------------------------

/// Computes multiple linear regression via OLS (Cholesky solve).
///
/// # Arguments
///
/// * `predictors` — Slice of predictor variable slices. Each inner slice is
///   one predictor's observations. All must have the same length.
/// * `y` — Response variable observations.
///
/// # Algorithm
///
/// Constructs the design matrix X = [1 | x₁ | x₂ | ... | xₚ] (intercept column prepended),
/// then solves the normal equations X'Xβ = X'y via Cholesky decomposition.
///
/// # Returns
///
/// `None` if n < p+2, predictor lengths differ, inputs contain non-finite
/// values, or the system is singular.
///
/// # References
///
/// Draper & Smith (1998). "Applied Regression Analysis", 3rd edition.
/// Montgomery, Peck & Vining (2012). "Introduction to Linear Regression Analysis", 5th edition.
///
/// # Examples
///
/// ```
/// use u_analytics::regression::multiple_linear_regression;
///
/// let x1 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
/// let x2 = [2.0, 1.0, 3.0, 2.0, 4.0, 3.0, 5.0, 4.0];
/// let y  = [5.1, 5.0, 9.2, 8.9, 13.1, 12.0, 17.2, 15.9];
/// let result = multiple_linear_regression(&[&x1, &x2], &y).unwrap();
/// assert!(result.r_squared > 0.95);
/// assert_eq!(result.coefficients.len(), 3); // intercept + 2 predictors
/// ```
pub fn multiple_linear_regression(
    predictors: &[&[f64]],
    y: &[f64],
) -> Option<MultipleRegressionResult> {
    let p = predictors.len(); // number of predictors
    let n = y.len();

    if p == 0 || n < p + 2 {
        return None;
    }

    // Validate lengths and finite values
    for pred in predictors {
        if pred.len() != n {
            return None;
        }
        if pred.iter().any(|v| !v.is_finite()) {
            return None;
        }
    }
    if y.iter().any(|v| !v.is_finite()) {
        return None;
    }

    let ncols = p + 1; // intercept + predictors

    // Build design matrix X (n × ncols, row-major)
    let mut x_data = Vec::with_capacity(n * ncols);
    for i in 0..n {
        x_data.push(1.0); // intercept
        for pred in predictors {
            x_data.push(pred[i]);
        }
    }
    let x_mat = Matrix::new(n, ncols, x_data).ok()?;

    // X'X (ncols × ncols)
    let xt = x_mat.transpose();
    let xtx = xt.mul_mat(&x_mat).ok()?;

    // X'y (ncols × 1)
    let xty = xt.mul_vec(y).ok()?;

    // Solve via Cholesky: β = (X'X)⁻¹ X'y
    let coefficients = xtx.cholesky_solve(&xty).ok()?;

    // Fitted values and residuals
    let fitted = x_mat.mul_vec(&coefficients).ok()?;
    let residuals: Vec<f64> = y
        .iter()
        .zip(fitted.iter())
        .map(|(&yi, &fi)| yi - fi)
        .collect();

    // R² and adjusted R²
    let y_mean = stats::mean(y)?;
    let ss_res: f64 = residuals.iter().map(|r| r * r).sum();
    let ss_tot: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum();

    let nf = n as f64;
    let pf = p as f64;
    let df_res = nf - pf - 1.0;

    let r_squared = if ss_tot > 1e-300 {
        1.0 - ss_res / ss_tot
    } else {
        1.0
    };
    let adjusted_r_squared = 1.0 - (1.0 - r_squared) * (nf - 1.0) / df_res;

    // Residual standard error
    let mse = ss_res / df_res;
    let residual_se = mse.sqrt();

    // Standard errors: SE(β) = sqrt(diag((X'X)⁻¹) · MSE)
    let xtx_inv = xtx.inverse().ok()?;
    let mut std_errors = Vec::with_capacity(ncols);
    let mut t_statistics = Vec::with_capacity(ncols);
    let mut p_values = Vec::with_capacity(ncols);

    for (j, &coeff_j) in coefficients.iter().enumerate() {
        let se = (xtx_inv.get(j, j) * mse).sqrt();
        std_errors.push(se);
        let t = if se > 1e-300 {
            coeff_j / se
        } else {
            f64::INFINITY
        };
        t_statistics.push(t);
        let pv = 2.0 * (1.0 - special::t_distribution_cdf(t.abs(), df_res));
        p_values.push(pv);
    }

    // F-statistic: (SS_reg / p) / (SS_res / (n-p-1))
    let ss_reg = ss_tot - ss_res;
    let f_statistic = if pf > 0.0 && mse > 1e-300 {
        (ss_reg / pf) / mse
    } else {
        0.0
    };
    let f_p_value = if f_statistic.is_infinite() || f_statistic.is_nan() {
        0.0
    } else {
        1.0 - special::f_distribution_cdf(f_statistic, pf, df_res)
    };

    // VIF for each predictor: VIF_j = 1/(1 - R²_j) where R²_j is from
    // regressing x_j on all other predictors
    let vif = compute_vif(predictors);

    Some(MultipleRegressionResult {
        coefficients,
        std_errors,
        t_statistics,
        p_values,
        r_squared,
        adjusted_r_squared,
        f_statistic,
        f_p_value,
        residual_se,
        residuals,
        fitted,
        vif,
        n,
        p,
    })
}

/// Computes VIF for each predictor by regressing each on all others.
fn compute_vif(predictors: &[&[f64]]) -> Vec<f64> {
    let p = predictors.len();
    if p < 2 {
        return vec![1.0; p];
    }

    let mut vif = Vec::with_capacity(p);
    for j in 0..p {
        // Regress x_j on all other predictors
        let y_j = predictors[j];
        let others: Vec<&[f64]> = predictors
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != j)
            .map(|(_, v)| *v)
            .collect();

        if let Some(result) = multiple_linear_regression(&others, y_j) {
            let r2 = result.r_squared;
            if r2 < 1.0 - 1e-15 {
                vif.push(1.0 / (1.0 - r2));
            } else {
                vif.push(f64::INFINITY); // perfect multicollinearity
            }
        } else {
            vif.push(f64::NAN);
        }
    }
    vif
}

/// Predicts y values given new x data and a simple regression result.
///
/// # Examples
///
/// ```
/// use u_analytics::regression::{simple_linear_regression, predict_simple};
///
/// let x = [1.0, 2.0, 3.0, 4.0, 5.0];
/// let y = [2.0, 4.0, 6.0, 8.0, 10.0];
/// let model = simple_linear_regression(&x, &y).unwrap();
/// let pred = predict_simple(&model, &[6.0, 7.0]);
/// assert!((pred[0] - 12.0).abs() < 1e-10);
/// assert!((pred[1] - 14.0).abs() < 1e-10);
/// ```
pub fn predict_simple(model: &SimpleRegressionResult, x_new: &[f64]) -> Vec<f64> {
    x_new
        .iter()
        .map(|&xi| model.intercept + model.slope * xi)
        .collect()
}

/// Predicts y values given new predictor data and a multiple regression result.
///
/// # Arguments
///
/// * `model` — Multiple regression result.
/// * `predictors_new` — Slice of predictor slices (same order as training).
///
/// Returns `None` if predictor count doesn't match or lengths differ.
pub fn predict_multiple(
    model: &MultipleRegressionResult,
    predictors_new: &[&[f64]],
) -> Option<Vec<f64>> {
    if predictors_new.len() != model.p {
        return None;
    }
    let n = predictors_new.first()?.len();
    for pred in predictors_new {
        if pred.len() != n {
            return None;
        }
    }

    let mut result = Vec::with_capacity(n);
    for i in 0..n {
        let mut yi = model.coefficients[0]; // intercept
        for (j, pred) in predictors_new.iter().enumerate() {
            yi += model.coefficients[j + 1] * pred[i];
        }
        result.push(yi);
    }
    Some(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Simple regression tests
    // -----------------------------------------------------------------------

    #[test]
    fn simple_perfect_fit() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [3.0, 5.0, 7.0, 9.0, 11.0]; // y = 1 + 2x
        let r = simple_linear_regression(&x, &y).expect("should compute");
        assert!((r.slope - 2.0).abs() < 1e-10);
        assert!((r.intercept - 1.0).abs() < 1e-10);
        assert!((r.r_squared - 1.0).abs() < 1e-10);
    }

    #[test]
    fn simple_with_noise() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [2.1, 3.9, 6.1, 7.9, 10.1]; // y ≈ 2x + 0.1
        let r = simple_linear_regression(&x, &y).expect("should compute");
        assert!((r.slope - 2.0).abs() < 0.1);
        assert!(r.r_squared > 0.99);
        assert_eq!(r.residuals.len(), 5);
        assert_eq!(r.fitted.len(), 5);
    }

    #[test]
    fn simple_residuals_sum_to_zero() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [2.1, 3.9, 6.1, 7.9, 10.1];
        let r = simple_linear_regression(&x, &y).expect("should compute");
        let sum: f64 = r.residuals.iter().sum();
        assert!(sum.abs() < 1e-10, "residuals sum = {sum}");
    }

    #[test]
    fn simple_f_equals_t_squared() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [2.1, 3.9, 6.1, 7.9, 10.1];
        let r = simple_linear_regression(&x, &y).expect("should compute");
        assert!(
            (r.f_statistic - r.slope_t * r.slope_t).abs() < 1e-8,
            "F = {}, t² = {}",
            r.f_statistic,
            r.slope_t * r.slope_t
        );
    }

    #[test]
    fn simple_significant_slope() {
        let x: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&xi| 3.0 + 2.0 * xi).collect();
        let r = simple_linear_regression(&x, &y).expect("should compute");
        assert!(r.slope_p < 1e-10, "slope p = {}", r.slope_p);
        assert!(r.f_p_value < 1e-10, "F p = {}", r.f_p_value);
    }

    #[test]
    fn simple_negative_slope() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [10.0, 8.0, 6.0, 4.0, 2.0]; // y = 12 - 2x
        let r = simple_linear_regression(&x, &y).expect("should compute");
        assert!((r.slope + 2.0).abs() < 1e-10);
        assert!((r.intercept - 12.0).abs() < 1e-10);
    }

    #[test]
    fn simple_edge_cases() {
        assert!(simple_linear_regression(&[1.0, 2.0], &[3.0, 4.0]).is_none()); // n < 3
        assert!(simple_linear_regression(&[1.0, 2.0, 3.0], &[4.0, 5.0]).is_none()); // mismatch
        assert!(simple_linear_regression(&[5.0, 5.0, 5.0], &[1.0, 2.0, 3.0]).is_none()); // zero var
        assert!(simple_linear_regression(&[1.0, f64::NAN, 3.0], &[4.0, 5.0, 6.0]).is_none());
    }

    #[test]
    fn simple_adjusted_r_squared() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [2.1, 3.9, 6.1, 7.9, 10.1];
        let r = simple_linear_regression(&x, &y).expect("should compute");
        // For perfect fit, adjusted R² ≈ R²
        assert!(r.adjusted_r_squared <= r.r_squared);
        assert!(r.adjusted_r_squared > 0.98);
    }

    // -----------------------------------------------------------------------
    // Multiple regression tests
    // -----------------------------------------------------------------------

    #[test]
    fn multiple_perfect_fit() {
        // y = 1 + 2*x1 + 3*x2
        let x1 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let x2 = [2.0, 1.0, 3.0, 2.0, 4.0, 3.0, 5.0, 4.0, 6.0, 5.0];
        let y: Vec<f64> = x1
            .iter()
            .zip(x2.iter())
            .map(|(&a, &b)| 1.0 + 2.0 * a + 3.0 * b)
            .collect();
        let r = multiple_linear_regression(&[&x1, &x2], &y).expect("should compute");

        assert!(
            (r.coefficients[0] - 1.0).abs() < 1e-8,
            "β₀ = {}",
            r.coefficients[0]
        );
        assert!(
            (r.coefficients[1] - 2.0).abs() < 1e-8,
            "β₁ = {}",
            r.coefficients[1]
        );
        assert!(
            (r.coefficients[2] - 3.0).abs() < 1e-8,
            "β₂ = {}",
            r.coefficients[2]
        );
        assert!((r.r_squared - 1.0).abs() < 1e-8);
    }

    #[test]
    fn multiple_with_noise() {
        let x1 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let x2 = [2.0, 1.0, 3.0, 2.0, 4.0, 3.0, 5.0, 4.0];
        let y = [5.1, 5.0, 9.2, 8.9, 13.1, 12.0, 17.2, 15.9];
        let r = multiple_linear_regression(&[&x1, &x2], &y).expect("should compute");
        assert!(r.r_squared > 0.95);
        assert_eq!(r.coefficients.len(), 3);
        assert_eq!(r.std_errors.len(), 3);
        assert_eq!(r.residuals.len(), 8);
        assert_eq!(r.vif.len(), 2);
    }

    #[test]
    fn multiple_residuals_sum_to_zero() {
        let x1 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let x2 = [2.0, 1.0, 3.0, 2.0, 4.0, 3.0, 5.0, 4.0];
        let y = [5.1, 5.0, 9.2, 8.9, 13.1, 12.0, 17.2, 15.9];
        let r = multiple_linear_regression(&[&x1, &x2], &y).expect("should compute");
        let sum: f64 = r.residuals.iter().sum();
        assert!(sum.abs() < 1e-8, "residuals sum = {sum}");
    }

    #[test]
    fn multiple_single_predictor_matches_simple() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let y: Vec<f64> = x
            .iter()
            .map(|&xi| 3.0 + 2.5 * xi + 0.1 * (xi - 5.0))
            .collect();

        let simple = simple_linear_regression(&x, &y).expect("simple");
        let multi = multiple_linear_regression(&[&x], &y).expect("multiple");

        assert!(
            (simple.slope - multi.coefficients[1]).abs() < 1e-8,
            "slope: {} vs {}",
            simple.slope,
            multi.coefficients[1]
        );
        assert!(
            (simple.intercept - multi.coefficients[0]).abs() < 1e-8,
            "intercept: {} vs {}",
            simple.intercept,
            multi.coefficients[0]
        );
        assert!((simple.r_squared - multi.r_squared).abs() < 1e-8);
    }

    #[test]
    fn multiple_edge_cases() {
        let x1 = [1.0, 2.0];
        let y = [3.0, 4.0];
        assert!(multiple_linear_regression(&[&x1], &y).is_none()); // n < p+2

        let x2 = [1.0, 2.0, 3.0];
        let y2 = [4.0, 5.0];
        assert!(multiple_linear_regression(&[&x2], &y2).is_none()); // length mismatch
    }

    #[test]
    fn multiple_vif_independent_predictors() {
        // Independent predictors → VIF ≈ 1
        let x1 = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0];
        let x2 = [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let y: Vec<f64> = x1
            .iter()
            .zip(x2.iter())
            .map(|(&a, &b)| 1.0 + 2.0 * a + 3.0 * b)
            .collect();
        let r = multiple_linear_regression(&[&x1, &x2], &y).expect("should compute");
        for (i, &v) in r.vif.iter().enumerate() {
            assert!(
                v < 2.0,
                "VIF[{i}] = {v}, expected near 1.0 for independent predictors"
            );
        }
    }

    #[test]
    fn multiple_vif_correlated_predictors() {
        // Highly but not perfectly correlated predictors → higher VIF
        let x1: Vec<f64> = (0..20).map(|i| i as f64).collect();
        // Add small perturbation to break perfect collinearity
        let noise = [
            0.1, -0.2, 0.3, -0.1, 0.2, -0.3, 0.1, -0.1, 0.2, -0.2, 0.3, -0.1, 0.1, -0.2, 0.3, -0.3,
            0.1, -0.1, 0.2, -0.2,
        ];
        let x2: Vec<f64> = x1
            .iter()
            .zip(noise.iter())
            .map(|(&v, &n)| v * 0.9 + 1.0 + n)
            .collect();
        let y: Vec<f64> = x1
            .iter()
            .zip(x2.iter())
            .map(|(&a, &b)| 1.0 + a + b)
            .collect();
        let r = multiple_linear_regression(&[&x1, &x2], &y).expect("should compute");
        // VIF > 5 for highly correlated predictors
        assert!(
            r.vif[0] > 5.0,
            "VIF[0] = {}, expected > 5.0 for correlated predictors",
            r.vif[0]
        );
    }

    // -----------------------------------------------------------------------
    // Prediction tests
    // -----------------------------------------------------------------------

    #[test]
    fn predict_simple_basic() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [3.0, 5.0, 7.0, 9.0, 11.0]; // y = 1 + 2x
        let model = simple_linear_regression(&x, &y).expect("should compute");
        let pred = predict_simple(&model, &[6.0, 7.0]);
        assert!((pred[0] - 13.0).abs() < 1e-10);
        assert!((pred[1] - 15.0).abs() < 1e-10);
    }

    #[test]
    fn predict_multiple_basic() {
        let x1 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let x2 = [2.0, 1.0, 3.0, 2.0, 4.0, 3.0, 5.0, 4.0, 6.0, 5.0];
        let y: Vec<f64> = x1
            .iter()
            .zip(x2.iter())
            .map(|(&a, &b)| 1.0 + 2.0 * a + 3.0 * b)
            .collect();
        let model = multiple_linear_regression(&[&x1, &x2], &y).expect("should compute");

        let new_x1 = [11.0];
        let new_x2 = [6.0];
        let pred = predict_multiple(&model, &[&new_x1, &new_x2]).expect("should predict");
        let expected = 1.0 + 2.0 * 11.0 + 3.0 * 6.0;
        assert!(
            (pred[0] - expected).abs() < 1e-6,
            "pred = {}, expected = {}",
            pred[0],
            expected
        );
    }

    #[test]
    fn predict_multiple_wrong_predictors() {
        let x1 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let x2 = [2.0, 1.0, 3.0, 2.0, 4.0, 3.0, 5.0, 4.0, 6.0, 5.0];
        let y: Vec<f64> = x1.iter().zip(x2.iter()).map(|(&a, &b)| a + b).collect();
        let model = multiple_linear_regression(&[&x1, &x2], &y).expect("should compute");

        // Wrong number of predictors
        assert!(predict_multiple(&model, &[&[1.0]]).is_none());
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn simple_r_squared_bounded(
            data in proptest::collection::vec(-1e3_f64..1e3, 5..=30)
                .prop_flat_map(|x| {
                    let n = x.len();
                    (Just(x), proptest::collection::vec(-1e3_f64..1e3, n..=n))
                })
        ) {
            let (x, y) = data;
            if let Some(r) = simple_linear_regression(&x, &y) {
                prop_assert!(r.r_squared >= -0.01 && r.r_squared <= 1.01,
                    "R² = {}", r.r_squared);
            }
        }

        #[test]
        fn simple_residuals_orthogonal_to_x(
            data in proptest::collection::vec(-1e3_f64..1e3, 5..=30)
                .prop_flat_map(|x| {
                    let n = x.len();
                    (Just(x), proptest::collection::vec(-1e3_f64..1e3, n..=n))
                })
        ) {
            let (x, y) = data;
            if let Some(r) = simple_linear_regression(&x, &y) {
                // Σ(xᵢ · eᵢ) should be near zero (OLS normal equation)
                let dot: f64 = x.iter().zip(r.residuals.iter()).map(|(&xi, &ei)| xi * ei).sum();
                let norm = r.residuals.iter().map(|e| e * e).sum::<f64>().sqrt();
                let x_norm = x.iter().map(|xi| xi * xi).sum::<f64>().sqrt();
                if norm > 1e-10 && x_norm > 1e-10 {
                    prop_assert!((dot / (norm * x_norm)).abs() < 1e-6,
                        "residuals not orthogonal to x: dot={dot}");
                }
            }
        }

        #[test]
        fn multiple_r_squared_bounded(
            x1 in proptest::collection::vec(-1e3_f64..1e3, 8..=20),
            x2_seed in proptest::collection::vec(-1e3_f64..1e3, 8..=20),
            y_seed in proptest::collection::vec(-1e3_f64..1e3, 8..=20),
        ) {
            let n = x1.len().min(x2_seed.len()).min(y_seed.len());
            let x2 = &x2_seed[..n];
            let y = &y_seed[..n];
            let x1 = &x1[..n];
            if let Some(r) = multiple_linear_regression(&[x1, x2], y) {
                prop_assert!(r.r_squared >= -0.01 && r.r_squared <= 1.01,
                    "R² = {}", r.r_squared);
            }
        }
    }
}
