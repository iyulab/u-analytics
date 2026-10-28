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
    /// Covariance of the intercept and slope estimates, −x̄ · MSE / Sxx.
    /// With `intercept_se` and `slope_se` it gives the variance of the fitted
    /// mean at any x (see [`predict_simple_intervals`]).
    pub intercept_slope_covariance: f64,
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
    /// Covariance matrix of the coefficients, MSE · (X'X)⁻¹, row-major
    /// `(p+1) × (p+1)` in the order of `coefficients` (intercept first) --
    /// R's `vcov`. Its diagonal is the square of `std_errors`.
    pub covariance: Vec<f64>,
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
    let intercept_slope_covariance = -x_mean * mse / ss_x;

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
    let slope_p = 2.0 * special::t_distribution_sf(slope_t.abs(), df_res);
    let intercept_p = 2.0 * special::t_distribution_sf(intercept_t.abs(), df_res);

    // F-statistic (= t² for simple regression)
    let f_statistic = slope_t * slope_t;
    let f_p_value = if f_statistic.is_infinite() {
        0.0
    } else {
        special::f_distribution_sf(f_statistic, 1.0, df_res)
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
        intercept_slope_covariance,
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

    // Normal equations on the design with each column scaled to unit length.
    // The intercept column is all ones, so a predictor in a small unit (1e-9)
    // makes X'X span eighteen orders of magnitude and the Cholesky factor
    // refuses it; the scaled X'X is the correlation-like matrix of the same
    // problem, whose conditioning does not depend on the units. With
    // S = diag(1/‖column‖): β = S·β̃ and (X'X)⁻¹ = S·(X̃'X̃)⁻¹·S.
    let mut col_scale = vec![0.0_f64; ncols];
    for (j, c) in col_scale.iter_mut().enumerate() {
        let norm = (0..n).map(|i| x_mat.get(i, j).powi(2)).sum::<f64>().sqrt();
        if !norm.is_finite() || norm == 0.0 {
            return None;
        }
        *c = 1.0 / norm;
    }
    let mut scaled_data = Vec::with_capacity(n * ncols);
    for i in 0..n {
        for (j, c) in col_scale.iter().enumerate() {
            scaled_data.push(x_mat.get(i, j) * c);
        }
    }
    let x_scaled = Matrix::new(n, ncols, scaled_data).ok()?;

    // X̃'X̃ (ncols × ncols)
    let xt = x_scaled.transpose();
    let xtx = xt.mul_mat(&x_scaled).ok()?;

    // X̃'y (ncols × 1)
    let xty = xt.mul_vec(y).ok()?;

    // Solve via Cholesky: β̃ = (X̃'X̃)⁻¹ X̃'y, then back to the original units
    let coefficients: Vec<f64> = xtx
        .cholesky_solve(&xty)
        .ok()?
        .iter()
        .zip(&col_scale)
        .map(|(b, c)| b * c)
        .collect();

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
    let mut covariance = Vec::with_capacity(ncols * ncols);
    for i in 0..ncols {
        for j in 0..ncols {
            covariance.push(xtx_inv.get(i, j) * col_scale[i] * col_scale[j] * mse);
        }
    }
    let mut std_errors = Vec::with_capacity(ncols);
    let mut t_statistics = Vec::with_capacity(ncols);
    let mut p_values = Vec::with_capacity(ncols);

    for (j, &coeff_j) in coefficients.iter().enumerate() {
        let se = covariance[j * ncols + j].sqrt();
        std_errors.push(se);
        let t = if se > 1e-300 {
            coeff_j / se
        } else {
            f64::INFINITY
        };
        t_statistics.push(t);
        let pv = 2.0 * special::t_distribution_sf(t.abs(), df_res);
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
        special::f_distribution_sf(f_statistic, pf, df_res)
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
        covariance,
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

/// A fitted regression's prediction at one new point, with its uncertainty.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RegressionPrediction {
    /// The point prediction ŷ₀.
    pub fit: f64,
    /// Standard error of the fitted mean at the point, √(x₀' V x₀) with V the
    /// coefficients' covariance.
    pub se_fit: f64,
    /// Confidence interval of the mean response at the point:
    /// ŷ₀ ± t · `se_fit`.
    pub confidence_lower: f64,
    /// Upper end of the confidence interval.
    pub confidence_upper: f64,
    /// Prediction interval of a new observation at the point:
    /// ŷ₀ ± t · √(MSE + `se_fit`²).
    pub prediction_lower: f64,
    /// Upper end of the prediction interval.
    pub prediction_upper: f64,
}

/// The intervals of a prediction `fit` whose mean has variance `var_fit`,
/// with residual variance `mse` on `df` degrees of freedom, at two-sided
/// `level`.
fn prediction_at(fit: f64, var_fit: f64, mse: f64, df: f64, level: f64) -> RegressionPrediction {
    let t = special::t_distribution_quantile(0.5 + level / 2.0, df);
    // Rounding can leave x₀'Vx₀ a hair below zero at a point the fit passes
    // through exactly.
    let se_fit = var_fit.max(0.0).sqrt();
    let se_new = (mse + var_fit.max(0.0)).sqrt();
    RegressionPrediction {
        fit,
        se_fit,
        confidence_lower: fit - t * se_fit,
        confidence_upper: fit + t * se_fit,
        prediction_lower: fit - t * se_new,
        prediction_upper: fit + t * se_new,
    }
}

/// Predictions of a simple regression at `x_new`, each with the confidence
/// interval of the mean response and the prediction interval of a new
/// observation at two-sided `level` (Student's t on n − 2 degrees of freedom).
///
/// Var(ŷ₀) = SE(β₀)² + x₀² SE(β₁)² + 2 x₀ Cov(β₀, β₁) = MSE (1/n + (x₀ − x̄)²/Sxx).
///
/// # Returns
///
/// `None` when `level` is not strictly between 0 and 1, or an `x_new` value is
/// not finite.
///
/// # References
///
/// Montgomery, Peck & Vining (2012). "Introduction to Linear Regression
/// Analysis", 5th edition, §2.4.2 and §2.5.
///
/// # Examples
///
/// ```
/// use u_analytics::regression::{predict_simple_intervals, simple_linear_regression};
///
/// let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
/// let y = [2.2, 3.9, 6.1, 8.2, 9.8, 12.1];
/// let model = simple_linear_regression(&x, &y).unwrap();
/// let p = predict_simple_intervals(&model, &[3.5, 10.0], 0.95).unwrap();
/// // Wider for a new observation than for the mean, and wider away from x̄.
/// assert!(p[0].prediction_upper - p[0].prediction_lower > p[0].confidence_upper - p[0].confidence_lower);
/// assert!(p[1].se_fit > p[0].se_fit);
/// ```
pub fn predict_simple_intervals(
    model: &SimpleRegressionResult,
    x_new: &[f64],
    level: f64,
) -> Option<Vec<RegressionPrediction>> {
    if !(level > 0.0 && level < 1.0) || x_new.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let df = model.n as f64 - 2.0;
    let mse = model.residual_se * model.residual_se;
    Some(
        x_new
            .iter()
            .map(|&x0| {
                let var_fit = model.intercept_se * model.intercept_se
                    + x0 * x0 * model.slope_se * model.slope_se
                    + 2.0 * x0 * model.intercept_slope_covariance;
                prediction_at(model.intercept + model.slope * x0, var_fit, mse, df, level)
            })
            .collect(),
    )
}

/// Predictions of a multiple regression at the new points `predictors_new`
/// (one slice per predictor, in the training order), each with the
/// confidence interval of the mean response and the prediction interval of a
/// new observation at two-sided `level` (Student's t on n − p − 1 degrees of
/// freedom).
///
/// Var(ŷ₀) = x₀' V x₀, x₀ = (1, x₀₁, …, x₀ₚ) and V the coefficients'
/// covariance ([`MultipleRegressionResult::covariance`]).
///
/// # Returns
///
/// `None` when the predictor count differs from the model's, the slices
/// differ in length, a value is not finite, or `level` is not strictly
/// between 0 and 1.
///
/// # References
///
/// Montgomery, Peck & Vining (2012). "Introduction to Linear Regression
/// Analysis", 5th edition, §3.4.
///
/// # Examples
///
/// ```
/// use u_analytics::regression::{multiple_linear_regression, predict_multiple_intervals};
///
/// let x1 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
/// let x2 = [2.0, 1.0, 3.0, 2.0, 4.0, 3.0, 5.0, 4.0];
/// let y = [5.1, 5.0, 9.2, 8.9, 13.1, 12.0, 17.2, 15.9];
/// let model = multiple_linear_regression(&[&x1, &x2], &y).unwrap();
/// let p = predict_multiple_intervals(&model, &[&[4.5], &[3.0]], 0.95).unwrap();
/// assert!(p[0].prediction_lower < p[0].fit && p[0].fit < p[0].prediction_upper);
/// ```
pub fn predict_multiple_intervals(
    model: &MultipleRegressionResult,
    predictors_new: &[&[f64]],
    level: f64,
) -> Option<Vec<RegressionPrediction>> {
    if !(level > 0.0 && level < 1.0) {
        return None;
    }
    let fits = predict_multiple(model, predictors_new)?;
    if predictors_new
        .iter()
        .any(|p| p.iter().any(|v| !v.is_finite()))
    {
        return None;
    }
    let k = model.p + 1;
    let df = model.n as f64 - k as f64;
    let mse = model.residual_se * model.residual_se;
    Some(
        fits.iter()
            .enumerate()
            .map(|(i, &fit)| {
                let x0: Vec<f64> = std::iter::once(1.0)
                    .chain(predictors_new.iter().map(|p| p[i]))
                    .collect();
                let var_fit: f64 = (0..k)
                    .map(|a| {
                        (0..k)
                            .map(|b| x0[a] * model.covariance[a * k + b] * x0[b])
                            .sum::<f64>()
                    })
                    .sum();
                prediction_at(fit, var_fit, mse, df, level)
            })
            .collect(),
    )
}

// ---------------------------------------------------------------------------
// Multicollinearity diagnostics (public)
// ---------------------------------------------------------------------------

/// Computes the Variance Inflation Factor for each predictor.
///
/// `VIF_j = 1 / (1 - R²_j)`, where `R²_j` is the coefficient of determination
/// from regressing predictor `j` on all other predictors. Standard thresholds
/// are `VIF > 5` (moderate multicollinearity) and `VIF > 10` (severe).
///
/// # Returns
///
/// `None` if `predictors` is empty, lengths differ, fewer than `p+1`
/// observations, or any predictor contains a non-finite value. With a single
/// predictor, returns `Some(vec![1.0])` since VIF is undefined for one
/// regressor.
///
/// # References
///
/// Marquardt (1970). "Generalized inverses, ridge regression."
/// Belsley, Kuh, Welsch (1980). *Regression Diagnostics*.
///
/// # Examples
///
/// ```
/// use u_analytics::regression::vif;
///
/// let x1 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
/// let x2 = [8.0, 1.0, 6.0, 3.0, 5.0, 7.0, 2.0, 4.0];
/// let v = vif(&[&x1, &x2]).unwrap();
/// assert!(v[0] < 2.0); // independent predictors
/// ```
pub fn vif(predictors: &[&[f64]]) -> Option<Vec<f64>> {
    let p = predictors.len();
    if p == 0 {
        return None;
    }
    let n = predictors[0].len();
    if !predictors.iter().all(|c| c.len() == n) {
        return None;
    }
    if n <= p {
        return None;
    }
    if predictors.iter().any(|c| c.iter().any(|v| !v.is_finite())) {
        return None;
    }
    Some(compute_vif(predictors))
}

/// Computes the 2-norm condition number of the sample covariance matrix.
///
/// `cond(Σ) = λ_max / λ_min`, where `λ` are the eigenvalues of the unbiased
/// sample covariance matrix of the (column-major) predictors. Standard
/// threshold: `cond > 30` indicates multicollinearity worth investigating;
/// `cond > 100` is severe (Belsley 1991).
///
/// # Returns
///
/// `None` for empty input, length mismatch, fewer than `p+1` observations,
/// non-finite values, or eigenvalue decomposition failure. Returns
/// `f64::INFINITY` when the smallest eigenvalue is at most
/// `p * f64::EPSILON` times the largest (numerically singular).
///
/// # References
///
/// Belsley (1991). *Conditioning Diagnostics*.
///
/// # Examples
///
/// ```
/// use u_analytics::regression::condition_number;
///
/// let x1 = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
/// let x2 = [1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0];
/// let c = condition_number(&[&x1, &x2]).unwrap();
/// assert!(c < 5.0); // near-orthogonal columns: cond should be small
/// ```
pub fn condition_number(predictors: &[&[f64]]) -> Option<f64> {
    use u_numflow::matrix::Matrix;
    let p = predictors.len();
    if p == 0 {
        return None;
    }
    let n = predictors[0].len();
    if !predictors.iter().all(|c| c.len() == n) {
        return None;
    }
    if n <= p {
        return None;
    }
    if predictors.iter().any(|c| c.iter().any(|v| !v.is_finite())) {
        return None;
    }

    // Sample covariance matrix (unbiased, denominator n-1)
    let mut means = vec![0.0_f64; p];
    for (j, col) in predictors.iter().enumerate() {
        means[j] = col.iter().sum::<f64>() / n as f64;
    }
    let mut cov = vec![0.0_f64; p * p];
    let denom = (n - 1) as f64;
    for r in 0..p {
        let col_r = predictors[r];
        let mean_r = means[r];
        for c in r..p {
            let col_c = predictors[c];
            let mean_c = means[c];
            let sum: f64 = col_r
                .iter()
                .zip(col_c.iter())
                .map(|(&a, &b)| (a - mean_r) * (b - mean_c))
                .sum();
            let v = sum / denom;
            cov[r * p + c] = v;
            cov[c * p + r] = v;
        }
    }

    let mat = Matrix::new(p, p, cov).ok()?;
    let (eigenvalues, _eigenvectors) = mat.eigen_symmetric().ok()?;
    let lambda_max = eigenvalues
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let lambda_min = eigenvalues.iter().copied().fold(f64::INFINITY, f64::min);

    if !lambda_max.is_finite() || lambda_max <= 0.0 {
        return None;
    }
    // Numerically singular is judged against the matrix's own scale (the
    // tolerance NumPy's `matrix_rank` uses), not against a fixed number: the
    // same data in a smaller unit has a covariance 1e-18 as large and is no
    // less well conditioned.
    if lambda_min <= lambda_max * p as f64 * f64::EPSILON {
        return Some(f64::INFINITY);
    }
    Some(lambda_max / lambda_min)
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

    /// Verifies exact OLS numeric reference from the spec.
    ///
    /// x = [1,2,3,4,5], y = [2,4,5,4,5]
    /// x̄=3, ȳ=4
    /// Σ(xᵢ-x̄)(yᵢ-ȳ) = 6, Σ(xᵢ-x̄)² = 10
    /// β̂₁ = 6/10 = 0.6, β̂₀ = 4 - 0.6·3 = 2.2
    /// SS_tot = 6, SS_res = 2.4, R² = 1 - 2.4/6 = 0.6
    ///
    /// Reference: Draper & Smith (1998), Applied Regression Analysis, 3rd ed.
    #[test]
    fn simple_numeric_reference_ols() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [2.0, 4.0, 5.0, 4.0, 5.0];
        let r = simple_linear_regression(&x, &y).expect("should compute");

        assert!(
            (r.slope - 0.6).abs() < 1e-10,
            "β̂₁ expected 0.6, got {}",
            r.slope
        );
        assert!(
            (r.intercept - 2.2).abs() < 1e-10,
            "β̂₀ expected 2.2, got {}",
            r.intercept
        );
        assert!(
            (r.r_squared - 0.6).abs() < 1e-3,
            "R² expected 0.6, got {}",
            r.r_squared
        );

        // Verify fitted values
        // ŷ₁=2.8, ŷ₂=3.4, ŷ₃=4.0, ŷ₄=4.6, ŷ₅=5.2
        let expected_fitted = [2.8, 3.4, 4.0, 4.6, 5.2];
        for (i, (&fi, &ef)) in r.fitted.iter().zip(expected_fitted.iter()).enumerate() {
            assert!(
                (fi - ef).abs() < 1e-10,
                "ŷ_{} expected {}, got {}",
                i + 1,
                ef,
                fi
            );
        }
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

    // -----------------------------------------------------------------------
    // Multicollinearity diagnostics — public vif() / condition_number()
    // -----------------------------------------------------------------------

    #[test]
    fn vif_pub_independent_predictors() {
        let x1 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let x2 = [8.0, 1.0, 6.0, 3.0, 5.0, 7.0, 2.0, 4.0];
        let v = vif(&[&x1, &x2]).expect("should compute");
        for r in &v {
            assert!(*r < 2.0, "VIF should be near 1.0 for independent, got {r}");
        }
    }

    #[test]
    fn vif_pub_collinear_predictors() {
        let x1: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let x2: Vec<f64> = x1.iter().map(|&v| v * 2.0 + 0.001).collect();
        let v = vif(&[&x1[..], &x2[..]]).expect("should compute");
        assert!(
            v[0] > 50.0,
            "VIF should explode under near-collinearity, got {}",
            v[0]
        );
    }

    #[test]
    fn vif_pub_rejects_short_input() {
        let x = [1.0, 2.0];
        assert!(vif(&[&x[..], &x[..]]).is_none());
    }

    #[test]
    fn vif_pub_rejects_non_finite() {
        let x1 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x2 = [1.0, 2.0, f64::NAN, 4.0, 5.0, 6.0];
        assert!(vif(&[&x1[..], &x2[..]]).is_none());
    }

    #[test]
    fn condition_number_orthogonal() {
        let x1 = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
        let x2 = [1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0];
        let c = condition_number(&[&x1[..], &x2[..]]).expect("should compute");
        assert!(c < 5.0, "near-orthogonal: cond should be small, got {c}");
    }

    #[test]
    fn condition_number_collinear_explodes() {
        let x1: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let x2: Vec<f64> = x1.iter().map(|&v| v * 2.0 + 1e-6).collect();
        let c = condition_number(&[&x1[..], &x2[..]]).expect("should compute");
        assert!(
            c > 1e5 || c.is_infinite(),
            "collinear: cond should be huge, got {c}"
        );
    }

    #[test]
    fn condition_number_rejects_short_input() {
        let x = [1.0, 2.0];
        assert!(condition_number(&[&x[..], &x[..]]).is_none());
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

/// Slope, intercept and F p-values far in the tail against SciPy 1.17.1
/// (`scipy.stats.linregress`; the intercept's and the F test's from `t.sf` and
/// `f.sf`) -- the smallest is 8e-33.
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
    fn simple_regression_p_values_keep_their_tail() {
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
        let r = simple_linear_regression(&x, &y).expect("valid");
        close(r.slope_p, 8.203356920391111e-33, "slope p");
        close(r.intercept_p, 1.5689270135401997e-16, "intercept p");
        close(r.f_p_value, 8.203356920397853e-33, "F p");
    }
    /// A fit and a condition number are properties of the data, not of the unit
    /// it is written in: the predictors in a unit a billion times smaller give
    /// the same R², p-values and conditioning, and coefficients that scale
    /// with the unit.
    #[test]
    fn multiple_regression_and_conditioning_do_not_depend_on_the_unit() {
        let n = 24;
        let x1: Vec<f64> = (0..n)
            .map(|i| (i as f64 * 0.7).sin() * 3.0 + i as f64 * 0.1)
            .collect();
        let x2: Vec<f64> = (0..n)
            .map(|i| (i as f64 * 1.3).cos() + 0.5 * x1[i])
            .collect();
        let y: Vec<f64> = (0..n)
            .map(|i| 1.0 + 2.0 * x1[i] - 0.5 * x2[i] + (i as f64 * 2.1).sin() * 0.3)
            .collect();
        let base = multiple_linear_regression(&[&x1, &x2], &y).expect("valid");
        let cond = condition_number(&[&x1, &x2]).expect("valid");
        for factor in [1e-9, 1e6] {
            let s1: Vec<f64> = x1.iter().map(|v| v * factor).collect();
            let s2: Vec<f64> = x2.iter().map(|v| v * factor).collect();
            let r = multiple_linear_regression(&[&s1, &s2], &y)
                .unwrap_or_else(|| panic!("a fit in units of {factor:e} was refused"));
            close(r.r_squared, base.r_squared, "R2");
            close(r.f_p_value, base.f_p_value, "F p");
            close(r.coefficients[0], base.coefficients[0], "intercept");
            for j in 0..2 {
                close(
                    r.coefficients[j + 1] * factor,
                    base.coefficients[j + 1],
                    "slope",
                );
                close(r.p_values[j + 1], base.p_values[j + 1], "slope p");
                close(r.vif[j], base.vif[j], "vif");
            }
            close(
                condition_number(&[&s1, &s2]).expect("valid"),
                cond,
                "condition number",
            );
        }
    }
}

/// Prediction intervals and the coefficient covariance against statsmodels
/// 0.15.0 (`OLS(...).fit().get_prediction(...).summary_frame(alpha)`,
/// `cov_params()`) on seeded data rounded to 3 places: a simple regression at
/// 90 % with points left of, inside and far right of the data, and three
/// predictors on scales from 1 to 5000 at 95 %.
#[cfg(test)]
#[allow(clippy::approx_constant)] // seeded data, not mathematical constants
mod statsmodels_prediction_reference {
    use super::*;

    // Generated by pi_ref.py - statsmodels 0.15.0, seed 20261028.
    const SX: &[f64] = &[
        0.321, 3.382, 13.09, 0.433, 11.207, 1.243, 13.577, 12.607, 3.653, 3.965, 11.199, 16.295,
    ];
    const SY: &[f64] = &[
        1.864, 10.159, 26.906, 4.994, 22.783, 9.397, 26.273, 24.293, 9.363, 9.846, 23.056, 29.43,
    ];
    const SXN: &[f64] = &[-5.0, 0.0, 10.0, 31.5];
    const S_MEAN: &[f64] = &[
        -4.004473932920135,
        4.156565438454745,
        20.478644181204505,
        55.57111347811649,
    ];
    const S_MEAN_SE: &[f64] = &[
        1.0916157424918431,
        0.7485088578492345,
        0.4868022389382572,
        1.9451041579807988,
    ];
    const S_MEAN_CI_LOWER: &[f64] = &[
        -5.982985027235801,
        2.799922233522837,
        19.596334048631235,
        52.04568781195695,
    ];
    const S_MEAN_CI_UPPER: &[f64] = &[
        -2.0259628386044684,
        5.513208643386653,
        21.360954313777775,
        59.09653914427602,
    ];
    const S_OBS_CI_LOWER: &[f64] = &[
        -7.441261322990979,
        1.0360715352868972,
        17.533228661407467,
        51.06271824900933,
    ];
    const S_OBS_CI_UPPER: &[f64] = &[
        -0.5676865428492905,
        7.277059341622593,
        23.424059701001543,
        60.079508707223646,
    ];
    const MX1: &[f64] = &[
        8.195, 0.642, 0.336, 6.144, 0.722, 9.91, 3.078, 2.241, 6.569, 9.172, 5.051, 8.676, 4.217,
        2.188, 6.705,
    ];
    const MX2: &[f64] = &[
        0.949, -2.284, 1.833, -0.66, -4.189, -3.649, -0.08, 1.587, 3.075, 4.395, 2.358, -4.012,
        -0.279, 4.502, 3.984,
    ];
    const MX3: &[f64] = &[
        3826.5, 3834.4, 3708.9, 1362.4, 2662.0, 2973.7, 3001.6, 4864.0, 2948.0, 1882.4, 2032.4,
        4144.4, 2666.3, 2747.4, 2715.4,
    ];
    const MY: &[f64] = &[
        28.737, 14.267, 13.838, 17.156, 10.139, 31.8, 15.884, 19.999, 21.378, 20.225, 16.187,
        33.594, 18.621, 10.903, 19.907,
    ];
    const MN1: &[f64] = &[5.0, 12.0];
    const MN2: &[f64] = &[0.0, -6.0];
    const MN3: &[f64] = &[3000.0, 800.0];
    const M_MEAN: &[f64] = &[19.808760915895164, 28.616648153275186];
    const M_MEAN_SE: &[f64] = &[0.3007541072279165, 1.2540843118704124];
    const M_MEAN_CI_LOWER: &[f64] = &[19.14680558904991, 25.856427193344672];
    const M_MEAN_CI_UPPER: &[f64] = &[20.470716242740416, 31.3768691132057];
    const M_OBS_CI_LOWER: &[f64] = &[17.202883874579435, 24.878837290302403];
    const M_OBS_CI_UPPER: &[f64] = &[22.414637957210893, 32.354459016247965];
    const M_COV: &[f64] = &[
        1.7052113874211179,
        -0.07001041367117572,
        -0.03185830538231558,
        -0.0004156285010374516,
        -0.07001041367117572,
        0.009234030588502234,
        0.000334157657603741,
        8.061392712052953e-06,
        -0.03185830538231558,
        0.000334157657603741,
        0.0109414882400463,
        8.173035100524671e-06,
        -0.0004156285010374516,
        8.061392712052953e-06,
        8.173035100524671e-06,
        1.229360269395184e-07,
    ];

    fn close(got: f64, want: f64, what: &str) {
        let tol = 1e-9 * want.abs().max(1.0);
        assert!((got - want).abs() <= tol, "{what}: {got} vs {want}");
    }

    #[test]
    fn simple_intervals_match_statsmodels() {
        let model = simple_linear_regression(SX, SY).expect("fits");
        let p = predict_simple_intervals(&model, SXN, 0.90).expect("intervals");
        for (i, r) in p.iter().enumerate() {
            close(r.fit, S_MEAN[i], "fit");
            close(r.se_fit, S_MEAN_SE[i], "se_fit");
            close(r.confidence_lower, S_MEAN_CI_LOWER[i], "mean lower");
            close(r.confidence_upper, S_MEAN_CI_UPPER[i], "mean upper");
            close(r.prediction_lower, S_OBS_CI_LOWER[i], "obs lower");
            close(r.prediction_upper, S_OBS_CI_UPPER[i], "obs upper");
        }
    }

    #[test]
    fn multiple_intervals_and_covariance_match_statsmodels() {
        let model = multiple_linear_regression(&[MX1, MX2, MX3], MY).expect("fits");
        for (got, want) in model.covariance.iter().zip(M_COV) {
            close(*got, *want, "covariance");
        }
        let p = predict_multiple_intervals(&model, &[MN1, MN2, MN3], 0.95).expect("intervals");
        for (i, r) in p.iter().enumerate() {
            close(r.fit, M_MEAN[i], "fit");
            close(r.se_fit, M_MEAN_SE[i], "se_fit");
            close(r.confidence_lower, M_MEAN_CI_LOWER[i], "mean lower");
            close(r.confidence_upper, M_MEAN_CI_UPPER[i], "mean upper");
            close(r.prediction_lower, M_OBS_CI_LOWER[i], "obs lower");
            close(r.prediction_upper, M_OBS_CI_UPPER[i], "obs upper");
        }
    }
}
