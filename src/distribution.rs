//! Distribution analysis.
//!
//! Empirical distribution functions, histogram binning, QQ-plot data,
//! distribution fitting/testing, and kernel density estimation.
//!
//! # Examples
//!
//! ```
//! use u_analytics::distribution::{ecdf, histogram_bins, BinMethod};
//!
//! let data = [1.0, 2.0, 3.0, 4.0, 5.0];
//! let (values, probs) = ecdf(&data).unwrap();
//! assert_eq!(values.len(), 5);
//! assert!((probs[4] - 1.0).abs() < 1e-10);
//!
//! let bins = histogram_bins(&data, BinMethod::Sturges).unwrap();
//! assert!(bins.n_bins >= 2);
//! ```

use u_numflow::special;
use u_numflow::stats;

/// A point on the empirical CDF.
///
/// Computes the ECDF: F_n(x) = (number of observations ≤ x) / n.
///
/// # Returns
///
/// Tuple of (sorted unique values, cumulative probabilities). Returns
/// `None` if the data is empty or contains non-finite values.
///
/// # Examples
///
/// ```
/// use u_analytics::distribution::ecdf;
///
/// let data = [3.0, 1.0, 2.0, 1.0, 4.0];
/// let (vals, probs) = ecdf(&data).unwrap();
/// assert_eq!(vals, vec![1.0, 2.0, 3.0, 4.0]);
/// assert!((probs[0] - 0.4).abs() < 1e-10); // 2 values ≤ 1.0
/// assert!((probs[3] - 1.0).abs() < 1e-10);
/// ```
pub fn ecdf(data: &[f64]) -> Option<(Vec<f64>, Vec<f64>)> {
    if data.is_empty() || data.iter().any(|v| !v.is_finite()) {
        return None;
    }

    let n = data.len() as f64;
    let mut sorted: Vec<f64> = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("finite values"));

    let mut values = Vec::new();
    let mut probs = Vec::new();

    let mut i = 0;
    while i < sorted.len() {
        let val = sorted[i];
        // Count all occurrences of this value
        let mut j = i;
        while j < sorted.len() && (sorted[j] - val).abs() < 1e-300 {
            j += 1;
        }
        values.push(val);
        probs.push(j as f64 / n);
        i = j;
    }

    Some((values, probs))
}

// ---------------------------------------------------------------------------
// Histogram binning
// ---------------------------------------------------------------------------

/// Method for computing optimal number of histogram bins.
#[derive(Debug, Clone, Copy)]
pub enum BinMethod {
    /// Sturges' rule: k = ⌈log₂(n)⌉ + 1. Best for near-normal data.
    Sturges,
    /// Scott's rule: h = 3.49·σ·n^(-1/3). Width-based.
    Scott,
    /// Freedman-Diaconis rule: h = 2·IQR·n^(-1/3). Robust to outliers.
    FreedmanDiaconis,
}

/// Result of histogram bin computation.
#[derive(Debug, Clone)]
pub struct HistogramBins {
    /// Number of bins.
    pub n_bins: usize,
    /// Bin width.
    pub bin_width: f64,
    /// Bin edges (length = n_bins + 1).
    pub edges: Vec<f64>,
    /// Bin counts.
    pub counts: Vec<usize>,
}

/// Computes optimal histogram bins using the specified method.
///
/// # Returns
///
/// `None` if fewer than 2 data points, non-finite values, or zero range.
///
/// # Examples
///
/// ```
/// use u_analytics::distribution::{histogram_bins, BinMethod};
///
/// let data = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0];
/// let result = histogram_bins(&data, BinMethod::Sturges).unwrap();
/// assert!(result.n_bins >= 3);
/// assert_eq!(result.edges.len(), result.n_bins + 1);
/// assert_eq!(result.counts.len(), result.n_bins);
/// ```
pub fn histogram_bins(data: &[f64], method: BinMethod) -> Option<HistogramBins> {
    let n = data.len();
    if n < 2 || data.iter().any(|v| !v.is_finite()) {
        return None;
    }

    let min_val = data.iter().cloned().reduce(f64::min)?;
    let max_val = data.iter().cloned().reduce(f64::max)?;
    let range = max_val - min_val;

    if range < 1e-300 {
        return None; // all same value
    }

    let nf = n as f64;

    let n_bins = match method {
        BinMethod::Sturges => {
            let k = (nf.log2()).ceil() as usize + 1;
            k.max(2)
        }
        BinMethod::Scott => {
            let sd = stats::std_dev(data)?;
            if sd < 1e-300 {
                return None;
            }
            let h = 3.49 * sd * nf.powf(-1.0 / 3.0);
            (range / h).ceil() as usize
        }
        BinMethod::FreedmanDiaconis => {
            let q1 = stats::quantile(data, 0.25)?;
            let q3 = stats::quantile(data, 0.75)?;
            let iqr = q3 - q1;
            if iqr < 1e-300 {
                // Fall back to Sturges if IQR is zero
                let k = (nf.log2()).ceil() as usize + 1;
                k.max(2)
            } else {
                let h = 2.0 * iqr * nf.powf(-1.0 / 3.0);
                (range / h).ceil() as usize
            }
        }
    }
    .max(2);

    let bin_width = range / n_bins as f64;

    // Compute edges
    let mut edges = Vec::with_capacity(n_bins + 1);
    for i in 0..=n_bins {
        edges.push(min_val + i as f64 * bin_width);
    }

    // Count data in each bin
    let mut counts = vec![0_usize; n_bins];
    for &x in data {
        let bin = ((x - min_val) / bin_width).floor() as usize;
        let bin = bin.min(n_bins - 1); // last point goes in last bin
        counts[bin] += 1;
    }

    Some(HistogramBins {
        n_bins,
        bin_width,
        edges,
        counts,
    })
}

// ---------------------------------------------------------------------------
// QQ-plot
// ---------------------------------------------------------------------------

/// Generates QQ-plot data (theoretical quantiles vs sample quantiles).
///
/// Uses the standard normal distribution as the theoretical distribution.
/// Sample quantiles are the sorted data. Theoretical quantiles are computed
/// using the plotting position (i - 0.5) / n.
///
/// # Returns
///
/// Tuple of (theoretical_quantiles, sample_quantiles). Returns `None` if
/// fewer than 3 data points.
///
/// # Examples
///
/// ```
/// use u_analytics::distribution::qq_plot_normal;
///
/// let data = [-1.5, -0.5, 0.0, 0.5, 1.5];
/// let (theoretical, sample) = qq_plot_normal(&data).unwrap();
/// assert_eq!(theoretical.len(), 5);
/// assert_eq!(sample.len(), 5);
/// // Sample and theoretical should be roughly aligned for normal data
/// ```
pub fn qq_plot_normal(data: &[f64]) -> Option<(Vec<f64>, Vec<f64>)> {
    let n = data.len();
    if n < 3 || data.iter().any(|v| !v.is_finite()) {
        return None;
    }

    let mut sample: Vec<f64> = data.to_vec();
    sample.sort_by(|a, b| a.partial_cmp(b).expect("finite values"));

    let nf = n as f64;
    let theoretical: Vec<f64> = (0..n)
        .map(|i| {
            let p = (i as f64 + 0.5) / nf;
            special::inverse_normal_cdf(p)
        })
        .collect();

    Some((theoretical, sample))
}

// ---------------------------------------------------------------------------
// Kolmogorov-Smirnov test
// ---------------------------------------------------------------------------

/// Kolmogorov-Smirnov one-sample test against the standard normal distribution.
///
/// # Algorithm
///
/// D = max|F_n(x) - Φ(x)| where F_n is the ECDF and Φ is the CDF of N(0,1).
/// P-value approximation: Kolmogorov distribution (Marsaglia et al., 2003).
///
/// # Returns
///
/// `None` if fewer than 5 observations or non-finite values.
///
/// # Examples
///
/// ```
/// use u_analytics::distribution::ks_test_normal;
///
/// // Data from approximately N(0,1)
/// let data = [-1.2, -0.8, -0.3, 0.1, 0.5, 0.7, 1.1, 1.4];
/// let (d_stat, p_value) = ks_test_normal(&data).unwrap();
/// assert!(p_value > 0.05); // cannot reject normality
/// ```
pub fn ks_test_normal(data: &[f64]) -> Option<(f64, f64)> {
    let n = data.len();
    if n < 5 || data.iter().any(|v| !v.is_finite()) {
        return None;
    }

    let mean = stats::mean(data)?;
    let sd = stats::std_dev(data)?;
    if sd < 1e-300 {
        return None;
    }

    let mut sorted: Vec<f64> = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("finite values"));

    let nf = n as f64;
    let mut d_stat = 0.0_f64;

    for (i, &x) in sorted.iter().enumerate() {
        let z = (x - mean) / sd;
        let cdf = special::standard_normal_cdf(z);
        let ecdf_above = (i + 1) as f64 / nf;
        let ecdf_below = i as f64 / nf;
        d_stat = d_stat.max((ecdf_above - cdf).abs());
        d_stat = d_stat.max((ecdf_below - cdf).abs());
    }

    // P-value approximation (Kolmogorov distribution, simplified)
    // For large n: P(D > x) ≈ 2·Σ (-1)^(k-1) exp(-2k²n·x²)
    let lambda = (nf.sqrt() + 0.12 + 0.11 / nf.sqrt()) * d_stat;
    let mut p_value = 0.0;
    for k in 1..=100 {
        let kf = k as f64;
        let sign = if k % 2 == 1 { 1.0 } else { -1.0 };
        let term = sign * (-2.0 * kf * kf * lambda * lambda).exp();
        p_value += term;
        if term.abs() < 1e-15 {
            break;
        }
    }
    p_value = (2.0 * p_value).clamp(0.0, 1.0);

    Some((d_stat, p_value))
}

// ---------------------------------------------------------------------------
// Kernel Density Estimation
// ---------------------------------------------------------------------------

/// Bandwidth selection method for kernel density estimation.
#[derive(Debug, Clone, Copy)]
pub enum BandwidthMethod {
    /// Silverman's rule of thumb: h = 0.9 * min(σ, IQR/1.34) * n^(-1/5).
    /// Robust to outliers and multimodal distributions.
    ///
    /// Reference: Silverman (1986), "Density Estimation for Statistics and
    /// Data Analysis"
    Silverman,
    /// Scott's rule: h = 1.06 * σ * n^(-1/5).
    /// Slightly smoother, assumes approximately normal data.
    ///
    /// Reference: Scott (1992), "Multivariate Density Estimation"
    Scott,
    /// Manual bandwidth specification.
    Manual(f64),
}

/// Result of kernel density estimation.
#[derive(Debug, Clone)]
pub struct KdeResult {
    /// Evaluation points (x-axis).
    pub x: Vec<f64>,
    /// Density estimates at each evaluation point (y-axis).
    pub density: Vec<f64>,
    /// Bandwidth used.
    pub bandwidth: f64,
}

/// Gaussian kernel density estimation.
///
/// # Algorithm
///
/// f̂(x) = (1/nh) Σᵢ K((x - xᵢ)/h)
///
/// where K(u) = (1/√(2π)) exp(-u²/2) is the Gaussian kernel and
/// h is the bandwidth (smoothing parameter).
///
/// The evaluation grid extends 3h beyond the data range to capture tail
/// contributions from boundary points (3σ covers 99.7% of a Gaussian).
///
/// Reference: Silverman (1986), "Density Estimation for Statistics and
/// Data Analysis"
///
/// # Parameters
///
/// - `data`: Sample observations (must have ≥ 2 finite values)
/// - `method`: Bandwidth selection method
/// - `n_points`: Number of evaluation grid points (typical: 256–1024)
///
/// # Returns
///
/// `None` if fewer than 2 data points, fewer than 2 grid points,
/// non-finite values, or zero variance (for automatic bandwidth methods).
///
/// # Examples
///
/// ```
/// use u_analytics::distribution::{kde, BandwidthMethod};
///
/// let data = [1.0, 1.1, 1.2, 2.0, 2.1, 2.2, 5.0];
/// let result = kde(&data, BandwidthMethod::Silverman, 512).unwrap();
/// assert_eq!(result.x.len(), 512);
/// assert_eq!(result.density.len(), 512);
/// assert!(result.bandwidth > 0.0);
/// // Density should integrate approximately to 1
/// let dx = result.x[1] - result.x[0];
/// let integral: f64 = result.density.iter().sum::<f64>() * dx;
/// assert!((integral - 1.0).abs() < 0.05);
/// ```
pub fn kde(data: &[f64], method: BandwidthMethod, n_points: usize) -> Option<KdeResult> {
    let n = data.len();
    if n < 2 || n_points < 2 || data.iter().any(|v| !v.is_finite()) {
        return None;
    }

    let bandwidth = kde_bandwidth(data, method)?;

    // Evaluation grid: extend 3*bandwidth beyond data range
    let min_val = data.iter().cloned().reduce(f64::min)?;
    let max_val = data.iter().cloned().reduce(f64::max)?;
    let x_min = min_val - 3.0 * bandwidth;
    let x_max = max_val + 3.0 * bandwidth;
    let step = (x_max - x_min) / (n_points - 1) as f64;

    let x: Vec<f64> = (0..n_points).map(|i| x_min + i as f64 * step).collect();

    // Evaluate density at each grid point
    let inv_h = 1.0 / bandwidth;
    let inv_nh = inv_h / n as f64;
    let inv_sqrt_2pi = 1.0 / (2.0 * std::f64::consts::PI).sqrt();

    let density: Vec<f64> = x
        .iter()
        .map(|&xi| {
            let sum: f64 = data
                .iter()
                .map(|&xj| {
                    let u = (xi - xj) * inv_h;
                    inv_sqrt_2pi * (-0.5 * u * u).exp()
                })
                .sum();
            sum * inv_nh
        })
        .collect();

    Some(KdeResult {
        x,
        density,
        bandwidth,
    })
}

/// Evaluates the kernel density estimate at a single point.
///
/// Useful for point-wise evaluation without generating a full grid.
///
/// # Returns
///
/// `None` if data is empty, contains non-finite values, or bandwidth
/// is invalid.
pub fn kde_evaluate(data: &[f64], bandwidth: f64, x: f64) -> Option<f64> {
    let n = data.len();
    if n == 0 || bandwidth <= 0.0 || !bandwidth.is_finite() || !x.is_finite() {
        return None;
    }
    if data.iter().any(|v| !v.is_finite()) {
        return None;
    }

    let inv_h = 1.0 / bandwidth;
    let inv_nh = inv_h / n as f64;
    let inv_sqrt_2pi = 1.0 / (2.0 * std::f64::consts::PI).sqrt();

    let sum: f64 = data
        .iter()
        .map(|&xj| {
            let u = (x - xj) * inv_h;
            inv_sqrt_2pi * (-0.5 * u * u).exp()
        })
        .sum();

    Some(sum * inv_nh)
}

/// Computes the bandwidth for KDE using the specified method.
///
/// Useful when you want to inspect the bandwidth before running full KDE.
///
/// # Returns
///
/// `None` if fewer than 2 data points, non-finite values, or zero
/// variance (for automatic methods).
pub fn kde_bandwidth(data: &[f64], method: BandwidthMethod) -> Option<f64> {
    let n = data.len();
    if n < 2 || data.iter().any(|v| !v.is_finite()) {
        return None;
    }

    match method {
        BandwidthMethod::Silverman => {
            let sd = stats::std_dev(data)?;
            if sd < 1e-300 {
                return None;
            }
            let q1 = stats::quantile(data, 0.25)?;
            let q3 = stats::quantile(data, 0.75)?;
            let iqr = q3 - q1;
            let spread = if iqr > 1e-300 {
                sd.min(iqr / 1.34)
            } else {
                sd
            };
            Some(0.9 * spread * (n as f64).powf(-0.2))
        }
        BandwidthMethod::Scott => {
            let sd = stats::std_dev(data)?;
            if sd < 1e-300 {
                return None;
            }
            Some(1.06 * sd * (n as f64).powf(-0.2))
        }
        BandwidthMethod::Manual(h) => {
            if h <= 0.0 || !h.is_finite() {
                None
            } else {
                Some(h)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// MLE Distribution Fitting
// ---------------------------------------------------------------------------

/// Result of a distribution fit via maximum likelihood estimation.
#[derive(Debug, Clone)]
pub struct FitResult {
    /// Name of the fitted distribution (e.g., "Normal", "Exponential").
    pub distribution: String,
    /// Estimated parameters (name, value) pairs.
    pub parameters: Vec<(String, f64)>,
    /// Log-likelihood at MLE parameters.
    pub log_likelihood: f64,
    /// Akaike Information Criterion: -2·ℓ + 2k.
    pub aic: f64,
    /// Bayesian Information Criterion: -2·ℓ + k·ln(n).
    pub bic: f64,
    /// Number of estimated parameters (k).
    pub n_params: usize,
}

/// Fits a Normal distribution N(μ, σ²) via MLE.
///
/// # Estimators
///
/// - μ̂ = x̄ (sample mean)
/// - σ̂ = √((1/n) Σ(xᵢ - x̄)²) (biased MLE, not sample std dev)
///
/// # Returns
///
/// `None` if fewer than 2 data points, non-finite values, or zero variance.
///
/// # Examples
///
/// ```
/// use u_analytics::distribution::fit_normal;
///
/// let data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
/// let fit = fit_normal(&data).unwrap();
/// assert_eq!(fit.distribution, "Normal");
/// assert!((fit.parameters[0].1 - 5.0).abs() < 1e-10); // μ̂ = 5.0
/// assert!(fit.parameters[1].1 > 0.0); // σ̂ > 0
/// ```
pub fn fit_normal(data: &[f64]) -> Option<FitResult> {
    let n = data.len();
    if n < 2 || data.iter().any(|v| !v.is_finite()) {
        return None;
    }

    let nf = n as f64;
    let mu = stats::mean(data)?;

    // Biased MLE variance (denominator n, not n-1)
    let sum_sq: f64 = data.iter().map(|&x| (x - mu).powi(2)).sum();
    let sigma_sq = sum_sq / nf;
    if sigma_sq < 1e-300 {
        return None;
    }
    let sigma = sigma_sq.sqrt();

    // Log-likelihood: -n/2·ln(2π) - n·ln(σ) - n/2
    let log_lik = -nf / 2.0 * (2.0 * std::f64::consts::PI).ln() - nf * sigma.ln() - nf / 2.0;

    let k = 2;
    let aic = -2.0 * log_lik + 2.0 * k as f64;
    let bic = -2.0 * log_lik + k as f64 * nf.ln();

    Some(FitResult {
        distribution: "Normal".to_string(),
        parameters: vec![
            ("mu".to_string(), mu),
            ("sigma".to_string(), sigma),
        ],
        log_likelihood: log_lik,
        aic,
        bic,
        n_params: k,
    })
}

/// Fits an Exponential distribution Exp(λ) via MLE.
///
/// # Estimators
///
/// - λ̂ = 1/x̄ (rate parameter)
///
/// # Returns
///
/// `None` if fewer than 2 data points, non-positive values, or non-finite values.
///
/// # Examples
///
/// ```
/// use u_analytics::distribution::fit_exponential;
///
/// let data = [0.5, 1.2, 0.8, 2.1, 0.3, 1.5, 0.9, 1.8];
/// let fit = fit_exponential(&data).unwrap();
/// assert_eq!(fit.distribution, "Exponential");
/// assert!(fit.parameters[0].1 > 0.0); // λ̂ > 0
/// ```
pub fn fit_exponential(data: &[f64]) -> Option<FitResult> {
    let n = data.len();
    if n < 2 || data.iter().any(|v| !v.is_finite()) {
        return None;
    }
    // Exponential requires x > 0
    if data.iter().any(|&v| v <= 0.0) {
        return None;
    }

    let nf = n as f64;
    let mean = stats::mean(data)?;
    if mean < 1e-300 {
        return None;
    }

    let lambda = 1.0 / mean;

    // Log-likelihood: n·ln(λ) - λ·n·x̄ = n·ln(λ) - n
    let log_lik = nf * lambda.ln() - nf;

    let k = 1;
    let aic = -2.0 * log_lik + 2.0 * k as f64;
    let bic = -2.0 * log_lik + k as f64 * nf.ln();

    Some(FitResult {
        distribution: "Exponential".to_string(),
        parameters: vec![("lambda".to_string(), lambda)],
        log_likelihood: log_lik,
        aic,
        bic,
        n_params: k,
    })
}

/// Digamma function ψ(x) = d/dx ln(Γ(x)).
///
/// Uses the asymptotic expansion for x ≥ 8, with recurrence for smaller x.
/// Reference: Abramowitz & Stegun (1972), formula 6.3.18.
fn digamma(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::NAN;
    }

    // Recurrence: ψ(x) = ψ(x+1) - 1/x
    let mut val = 0.0;
    let mut x = x;
    while x < 8.0 {
        val -= 1.0 / x;
        x += 1.0;
    }

    // Asymptotic expansion for x ≥ 8
    // ψ(x) ≈ ln(x) - 1/(2x) - 1/(12x²) + 1/(120x⁴) - 1/(252x⁶) + ...
    let inv_x = 1.0 / x;
    let inv_x2 = inv_x * inv_x;
    val += x.ln() - 0.5 * inv_x
        - inv_x2
            * (1.0 / 12.0 - inv_x2 * (1.0 / 120.0 - inv_x2 * (1.0 / 252.0)));

    val
}

/// Trigamma function ψ'(x) = d²/dx² ln(Γ(x)).
///
/// Asymptotic expansion for x ≥ 8, with recurrence for smaller x.
fn trigamma(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::NAN;
    }

    // Recurrence: ψ'(x) = ψ'(x+1) + 1/x²
    let mut val = 0.0;
    let mut x = x;
    while x < 8.0 {
        val += 1.0 / (x * x);
        x += 1.0;
    }

    // Asymptotic expansion for x ≥ 8
    // ψ'(x) ≈ 1/x + 1/(2x²) + 1/(6x³) - 1/(30x⁵) + 1/(42x⁷) - ...
    let inv_x = 1.0 / x;
    let inv_x2 = inv_x * inv_x;
    val += inv_x
        + 0.5 * inv_x2
        + inv_x2
            * inv_x
            * (1.0 / 6.0 - inv_x2 * (1.0 / 30.0 - inv_x2 * (1.0 / 42.0)));

    val
}

/// Fits a Gamma distribution Gamma(α, β) via MLE.
///
/// # Algorithm
///
/// Rate parameter β̂ = α̂/x̄ (closed-form once α is known).
/// Shape parameter α̂ solved via Newton-Raphson on:
/// ln(α) - ψ(α) = ln(x̄) - (1/n)Σln(xᵢ)
///
/// Initial guess from method of moments: α₀ = x̄²/s².
///
/// Reference: Minka (2002), "Estimating a Gamma distribution"
///
/// # Returns
///
/// `None` if fewer than 2 data points, non-positive values, non-finite values,
/// or Newton-Raphson fails to converge.
///
/// # Examples
///
/// ```
/// use u_analytics::distribution::fit_gamma;
///
/// let data = [2.1, 3.5, 1.8, 4.2, 2.9, 3.1, 2.5, 3.8, 1.5, 2.7];
/// let fit = fit_gamma(&data).unwrap();
/// assert_eq!(fit.distribution, "Gamma");
/// assert!(fit.parameters[0].1 > 0.0); // α̂ > 0
/// assert!(fit.parameters[1].1 > 0.0); // β̂ > 0
/// ```
pub fn fit_gamma(data: &[f64]) -> Option<FitResult> {
    let n = data.len();
    if n < 2 || data.iter().any(|v| !v.is_finite()) {
        return None;
    }
    // Gamma requires x > 0
    if data.iter().any(|&v| v <= 0.0) {
        return None;
    }

    let nf = n as f64;
    let mean = stats::mean(data)?;
    if mean < 1e-300 {
        return None;
    }

    // Biased variance (denominator n)
    let var: f64 = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / nf;
    if var < 1e-300 {
        return None;
    }

    let sum_log: f64 = data.iter().map(|&x| x.ln()).sum();
    let mean_log = sum_log / nf;
    let target = mean.ln() - mean_log; // ln(x̄) - (1/n)Σln(xᵢ)

    // Method of moments initial guess
    let mut alpha = mean * mean / var;
    if alpha < 1e-10 {
        alpha = 0.5;
    }

    // Newton-Raphson: solve ln(α) - ψ(α) = target
    for _ in 0..200 {
        let f = alpha.ln() - digamma(alpha) - target;
        let f_prime = 1.0 / alpha - trigamma(alpha);
        if f_prime.abs() < 1e-30 {
            break;
        }

        let delta = f / f_prime;
        alpha -= delta;

        // Keep alpha positive
        if alpha <= 0.0 {
            alpha = 1e-10;
        }

        if delta.abs() < 1e-10 {
            break;
        }
    }

    if !alpha.is_finite() || alpha <= 0.0 {
        return None;
    }

    let beta = alpha / mean; // rate parameter

    // Log-likelihood: n·α·ln(β) - n·ln(Γ(α)) + (α-1)·Σln(xᵢ) - β·Σxᵢ
    let sum_x: f64 = data.iter().sum();
    let log_lik = nf * alpha * beta.ln() - nf * special::ln_gamma(alpha)
        + (alpha - 1.0) * sum_log
        - beta * sum_x;

    let k = 2;
    let aic = -2.0 * log_lik + 2.0 * k as f64;
    let bic = -2.0 * log_lik + k as f64 * nf.ln();

    Some(FitResult {
        distribution: "Gamma".to_string(),
        parameters: vec![
            ("alpha".to_string(), alpha),
            ("beta".to_string(), beta),
        ],
        log_likelihood: log_lik,
        aic,
        bic,
        n_params: k,
    })
}

/// Fits a LogNormal distribution via MLE.
///
/// # Estimators
///
/// If Y = ln(X), then Y ~ N(μ, σ²):
/// - μ̂ = (1/n) Σ ln(xᵢ)
/// - σ̂ = √((1/n) Σ (ln(xᵢ) - μ̂)²)
///
/// # Returns
///
/// `None` if fewer than 2 data points, non-positive values, or non-finite values.
///
/// # Examples
///
/// ```
/// use u_analytics::distribution::fit_lognormal;
///
/// let data = [1.2, 2.5, 1.8, 3.1, 0.9, 2.0, 1.5, 4.0];
/// let fit = fit_lognormal(&data).unwrap();
/// assert_eq!(fit.distribution, "LogNormal");
/// assert!(fit.parameters[0].1.is_finite()); // μ̂
/// assert!(fit.parameters[1].1 > 0.0); // σ̂ > 0
/// ```
pub fn fit_lognormal(data: &[f64]) -> Option<FitResult> {
    let n = data.len();
    if n < 2 || data.iter().any(|v| !v.is_finite()) {
        return None;
    }
    if data.iter().any(|&v| v <= 0.0) {
        return None;
    }

    let nf = n as f64;
    let log_data: Vec<f64> = data.iter().map(|&x| x.ln()).collect();

    let mu = log_data.iter().sum::<f64>() / nf;
    let sigma_sq = log_data.iter().map(|&y| (y - mu).powi(2)).sum::<f64>() / nf;
    if sigma_sq < 1e-300 {
        return None;
    }
    let sigma = sigma_sq.sqrt();

    // Log-likelihood: -n/2·ln(2π) - n·ln(σ) - (1/2σ²)Σ(ln(xᵢ)-μ)² - Σln(xᵢ)
    let sum_log: f64 = log_data.iter().sum();
    let log_lik =
        -nf / 2.0 * (2.0 * std::f64::consts::PI).ln() - nf * sigma.ln() - nf / 2.0 - sum_log;

    let k = 2;
    let aic = -2.0 * log_lik + 2.0 * k as f64;
    let bic = -2.0 * log_lik + k as f64 * nf.ln();

    Some(FitResult {
        distribution: "LogNormal".to_string(),
        parameters: vec![("mu".to_string(), mu), ("sigma".to_string(), sigma)],
        log_likelihood: log_lik,
        aic,
        bic,
        n_params: k,
    })
}

/// Fits a Poisson distribution via MLE.
///
/// # Estimators
///
/// λ̂ = x̄ (sample mean). Data should be non-negative integers
/// (represented as f64 for API consistency).
///
/// # Returns
///
/// `None` if fewer than 2 data points, negative values, or non-finite values.
///
/// # Examples
///
/// ```
/// use u_analytics::distribution::fit_poisson;
///
/// let data = [2.0, 3.0, 1.0, 4.0, 2.0, 3.0, 1.0, 2.0];
/// let fit = fit_poisson(&data).unwrap();
/// assert_eq!(fit.distribution, "Poisson");
/// assert!((fit.parameters[0].1 - 2.25).abs() < 1e-10); // λ̂ = mean
/// ```
pub fn fit_poisson(data: &[f64]) -> Option<FitResult> {
    let n = data.len();
    if n < 2 || data.iter().any(|v| !v.is_finite()) {
        return None;
    }
    if data.iter().any(|&v| v < 0.0) {
        return None;
    }

    let nf = n as f64;
    let lambda = stats::mean(data)?;
    if lambda < 1e-300 {
        return None; // All zeros → degenerate
    }

    // Log-likelihood: Σ(xᵢ·ln(λ) - λ - ln(xᵢ!))
    // = n·x̄·ln(λ) - n·λ - Σln(xᵢ!)
    let sum_log_fact: f64 = data
        .iter()
        .map(|&x| special::ln_gamma(x + 1.0))
        .sum();
    let log_lik = nf * lambda * lambda.ln() - nf * lambda - sum_log_fact;

    let k = 1;
    let aic = -2.0 * log_lik + 2.0 * k as f64;
    let bic = -2.0 * log_lik + k as f64 * nf.ln();

    Some(FitResult {
        distribution: "Poisson".to_string(),
        parameters: vec![("lambda".to_string(), lambda)],
        log_likelihood: log_lik,
        aic,
        bic,
        n_params: k,
    })
}

/// Fits a Beta distribution Beta(α, β) via MLE.
///
/// # Algorithm
///
/// Newton-Raphson on the system:
/// - ψ(α) - ψ(α + β) = (1/n) Σ ln(xᵢ)
/// - ψ(β) - ψ(α + β) = (1/n) Σ ln(1 - xᵢ)
///
/// Initial guess from method of moments:
/// - α₀ = x̄ · ((x̄(1-x̄)/s²) - 1)
/// - β₀ = (1-x̄) · ((x̄(1-x̄)/s²) - 1)
///
/// Data must be in (0, 1).
///
/// # Returns
///
/// `None` if fewer than 2 data points, values outside (0,1), or convergence failure.
///
/// # Examples
///
/// ```
/// use u_analytics::distribution::fit_beta;
///
/// let data = [0.2, 0.35, 0.5, 0.15, 0.45, 0.3, 0.6, 0.25, 0.4, 0.55];
/// let fit = fit_beta(&data).unwrap();
/// assert_eq!(fit.distribution, "Beta");
/// assert!(fit.parameters[0].1 > 0.0); // α̂ > 0
/// assert!(fit.parameters[1].1 > 0.0); // β̂ > 0
/// ```
pub fn fit_beta(data: &[f64]) -> Option<FitResult> {
    let n = data.len();
    if n < 2 || data.iter().any(|v| !v.is_finite()) {
        return None;
    }
    // Beta requires 0 < x < 1
    if data.iter().any(|&v| v <= 0.0 || v >= 1.0) {
        return None;
    }

    let nf = n as f64;
    let mean = stats::mean(data)?;
    let var = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / nf;
    if var < 1e-300 {
        return None;
    }

    let sum_log_x: f64 = data.iter().map(|&x| x.ln()).sum();
    let sum_log_1mx: f64 = data.iter().map(|&x| (1.0 - x).ln()).sum();
    let mean_log_x = sum_log_x / nf;
    let mean_log_1mx = sum_log_1mx / nf;

    // Method of moments initial guess
    let factor = mean * (1.0 - mean) / var - 1.0;
    if factor <= 0.0 {
        return None;
    }
    let mut alpha = mean * factor;
    let mut beta = (1.0 - mean) * factor;

    if alpha < 0.01 {
        alpha = 0.5;
    }
    if beta < 0.01 {
        beta = 0.5;
    }

    // Newton-Raphson (joint update)
    for _ in 0..200 {
        let psi_ab = digamma(alpha + beta);
        let tri_ab = trigamma(alpha + beta);

        let f1 = digamma(alpha) - psi_ab - mean_log_x;
        let f2 = digamma(beta) - psi_ab - mean_log_1mx;

        let j11 = trigamma(alpha) - tri_ab;
        let j12 = -tri_ab;
        let j21 = -tri_ab;
        let j22 = trigamma(beta) - tri_ab;

        let det = j11 * j22 - j12 * j21;
        if det.abs() < 1e-30 {
            break;
        }

        let da = (j22 * f1 - j12 * f2) / det;
        let db = (j11 * f2 - j21 * f1) / det;

        alpha -= da;
        beta -= db;

        if alpha <= 0.0 {
            alpha = 1e-10;
        }
        if beta <= 0.0 {
            beta = 1e-10;
        }

        if da.abs() < 1e-10 && db.abs() < 1e-10 {
            break;
        }
    }

    if !alpha.is_finite() || !beta.is_finite() || alpha <= 0.0 || beta <= 0.0 {
        return None;
    }

    // Log-likelihood: n·[ln(Γ(α+β)) - ln(Γ(α)) - ln(Γ(β))]
    //                + (α-1)·Σln(xᵢ) + (β-1)·Σln(1-xᵢ)
    let log_lik = nf
        * (special::ln_gamma(alpha + beta) - special::ln_gamma(alpha) - special::ln_gamma(beta))
        + (alpha - 1.0) * sum_log_x
        + (beta - 1.0) * sum_log_1mx;

    let k = 2;
    let aic = -2.0 * log_lik + 2.0 * k as f64;
    let bic = -2.0 * log_lik + k as f64 * nf.ln();

    Some(FitResult {
        distribution: "Beta".to_string(),
        parameters: vec![
            ("alpha".to_string(), alpha),
            ("beta".to_string(), beta),
        ],
        log_likelihood: log_lik,
        aic,
        bic,
        n_params: k,
    })
}

/// Fits multiple distributions and returns results sorted by AIC (best first).
///
/// Tries Normal, Exponential, Gamma, LogNormal, and Poisson fits.
/// Beta is excluded since it requires data in (0, 1).
/// Only distributions that successfully fit are included in the output.
///
/// # Examples
///
/// ```
/// use u_analytics::distribution::fit_best;
///
/// let data = [2.1, 3.5, 1.8, 4.2, 2.9, 3.1, 2.5, 3.8, 1.5, 2.7];
/// let fits = fit_best(&data);
/// assert!(!fits.is_empty());
/// // Results sorted by AIC (lowest = best fit)
/// if fits.len() >= 2 {
///     assert!(fits[0].aic <= fits[1].aic);
/// }
/// ```
pub fn fit_best(data: &[f64]) -> Vec<FitResult> {
    let mut results = Vec::new();

    if let Some(r) = fit_normal(data) {
        results.push(r);
    }
    if let Some(r) = fit_exponential(data) {
        results.push(r);
    }
    if let Some(r) = fit_gamma(data) {
        results.push(r);
    }
    if let Some(r) = fit_lognormal(data) {
        results.push(r);
    }
    if let Some(r) = fit_poisson(data) {
        results.push(r);
    }

    results.sort_by(|a, b| {
        a.aic
            .partial_cmp(&b.aic)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // ECDF
    // -----------------------------------------------------------------------

    #[test]
    fn ecdf_basic() {
        let (vals, probs) = ecdf(&[3.0, 1.0, 2.0]).expect("should compute");
        assert_eq!(vals, vec![1.0, 2.0, 3.0]);
        assert!((probs[0] - 1.0 / 3.0).abs() < 1e-10);
        assert!((probs[1] - 2.0 / 3.0).abs() < 1e-10);
        assert!((probs[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn ecdf_with_ties() {
        let (vals, probs) = ecdf(&[1.0, 2.0, 2.0, 3.0]).expect("should compute");
        assert_eq!(vals, vec![1.0, 2.0, 3.0]);
        assert!((probs[0] - 0.25).abs() < 1e-10);
        assert!((probs[1] - 0.75).abs() < 1e-10); // 3 out of 4 ≤ 2.0
        assert!((probs[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn ecdf_single() {
        let (vals, probs) = ecdf(&[5.0]).expect("should compute");
        assert_eq!(vals, vec![5.0]);
        assert!((probs[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn ecdf_empty() {
        assert!(ecdf(&[]).is_none());
    }

    #[test]
    fn ecdf_nan() {
        assert!(ecdf(&[1.0, f64::NAN, 3.0]).is_none());
    }

    // -----------------------------------------------------------------------
    // Histogram bins
    // -----------------------------------------------------------------------

    #[test]
    fn hist_sturges_basic() {
        let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let r = histogram_bins(&data, BinMethod::Sturges).expect("should compute");
        assert!(r.n_bins >= 5);
        assert_eq!(r.edges.len(), r.n_bins + 1);
        assert_eq!(r.counts.len(), r.n_bins);
        let total: usize = r.counts.iter().sum();
        assert_eq!(total, 100);
    }

    #[test]
    fn hist_scott() {
        let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let r = histogram_bins(&data, BinMethod::Scott).expect("should compute");
        assert!(r.n_bins >= 3);
        let total: usize = r.counts.iter().sum();
        assert_eq!(total, 100);
    }

    #[test]
    fn hist_fd() {
        let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let r = histogram_bins(&data, BinMethod::FreedmanDiaconis).expect("should compute");
        assert!(r.n_bins >= 3);
        let total: usize = r.counts.iter().sum();
        assert_eq!(total, 100);
    }

    #[test]
    fn hist_edge_cases() {
        assert!(histogram_bins(&[1.0], BinMethod::Sturges).is_none()); // < 2
        assert!(histogram_bins(&[5.0, 5.0, 5.0], BinMethod::Sturges).is_none()); // zero range
    }

    // -----------------------------------------------------------------------
    // QQ-plot
    // -----------------------------------------------------------------------

    #[test]
    fn qq_basic() {
        let data = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5];
        let (theo, samp) = qq_plot_normal(&data).expect("should compute");
        assert_eq!(theo.len(), 7);
        assert_eq!(samp.len(), 7);
        // Sorted sample
        assert!((samp[0] + 1.5).abs() < 1e-10);
        assert!((samp[6] - 1.5).abs() < 1e-10);
        // Theoretical: median should be near 0
        assert!(theo[3].abs() < 0.1);
    }

    #[test]
    fn qq_edge_cases() {
        assert!(qq_plot_normal(&[1.0, 2.0]).is_none()); // < 3
    }

    // -----------------------------------------------------------------------
    // KS test
    // -----------------------------------------------------------------------

    #[test]
    fn ks_normal_data() {
        // Approximately normal data
        let data = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5];
        let (d, p) = ks_test_normal(&data).expect("should compute");
        assert!(d > 0.0 && d < 1.0, "D = {d}");
        assert!(p > 0.05, "p = {p}");
    }

    #[test]
    fn ks_non_normal_data() {
        // Clearly non-normal (uniform 0-100)
        let data: Vec<f64> = (0..50).map(|i| i as f64 * 2.0).collect();
        let (d, _p) = ks_test_normal(&data).expect("should compute");
        assert!(d > 0.0);
    }

    #[test]
    fn ks_edge_cases() {
        assert!(ks_test_normal(&[1.0, 2.0, 3.0, 4.0]).is_none()); // < 5
        assert!(ks_test_normal(&[5.0, 5.0, 5.0, 5.0, 5.0]).is_none()); // zero var
    }

    // -----------------------------------------------------------------------
    // KDE
    // -----------------------------------------------------------------------

    #[test]
    fn kde_silverman_basic() {
        let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let r = kde(&data, BandwidthMethod::Silverman, 256).expect("should compute");
        assert_eq!(r.x.len(), 256);
        assert_eq!(r.density.len(), 256);
        assert!(r.bandwidth > 0.0);
        // All densities non-negative
        assert!(r.density.iter().all(|&d| d >= 0.0));
        // Integral ≈ 1
        let dx = r.x[1] - r.x[0];
        let integral: f64 = r.density.iter().sum::<f64>() * dx;
        assert!(
            (integral - 1.0).abs() < 0.05,
            "integral = {integral}, expected ≈ 1.0"
        );
    }

    #[test]
    fn kde_scott_basic() {
        let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let r = kde(&data, BandwidthMethod::Scott, 256).expect("should compute");
        assert!(r.bandwidth > 0.0);
        let dx = r.x[1] - r.x[0];
        let integral: f64 = r.density.iter().sum::<f64>() * dx;
        assert!(
            (integral - 1.0).abs() < 0.05,
            "integral = {integral}, expected ≈ 1.0"
        );
    }

    #[test]
    fn kde_manual_bandwidth() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let r = kde(&data, BandwidthMethod::Manual(0.5), 128).expect("should compute");
        assert!((r.bandwidth - 0.5).abs() < 1e-10);
        assert_eq!(r.x.len(), 128);
    }

    #[test]
    fn kde_edge_cases() {
        // Too few points
        assert!(kde(&[1.0], BandwidthMethod::Silverman, 256).is_none());
        // Empty
        assert!(kde(&[], BandwidthMethod::Silverman, 256).is_none());
        // NaN
        assert!(kde(&[1.0, f64::NAN], BandwidthMethod::Silverman, 256).is_none());
        // Constant data (zero variance)
        assert!(kde(&[5.0, 5.0, 5.0], BandwidthMethod::Silverman, 256).is_none());
        // Too few grid points
        assert!(kde(&[1.0, 2.0, 3.0], BandwidthMethod::Manual(1.0), 1).is_none());
        // Invalid manual bandwidth
        assert!(kde(&[1.0, 2.0, 3.0], BandwidthMethod::Manual(0.0), 256).is_none());
        assert!(kde(&[1.0, 2.0, 3.0], BandwidthMethod::Manual(-1.0), 256).is_none());
    }

    #[test]
    fn kde_bimodal() {
        // Two clusters: around 0 and around 10
        let mut data = Vec::new();
        for i in 0..50 {
            data.push(i as f64 * 0.1); // 0.0 to 4.9
        }
        for i in 0..50 {
            data.push(10.0 + i as f64 * 0.1); // 10.0 to 14.9
        }
        let r = kde(&data, BandwidthMethod::Silverman, 512).expect("should compute");
        // Should have higher density near the two clusters
        // Find density at x ≈ 2.5 (in cluster 1) and x ≈ 7.5 (between clusters)
        let idx_cluster = r.x.iter().position(|&x| x >= 2.5).expect("grid point");
        let idx_valley = r.x.iter().position(|&x| x >= 7.5).expect("grid point");
        assert!(
            r.density[idx_cluster] > r.density[idx_valley],
            "cluster density ({}) should exceed valley density ({})",
            r.density[idx_cluster],
            r.density[idx_valley]
        );
    }

    #[test]
    fn kde_evaluate_single() {
        let data = [0.0, 1.0, 2.0, 3.0, 4.0];
        let h = 1.0;
        let d = kde_evaluate(&data, h, 2.0).expect("should compute");
        assert!(d > 0.0);
        // At the center of the data, density should be highest
        let d_edge = kde_evaluate(&data, h, -5.0).expect("should compute");
        assert!(d > d_edge, "center density ({d}) > edge density ({d_edge})");
    }

    #[test]
    fn kde_evaluate_edge_cases() {
        assert!(kde_evaluate(&[], 1.0, 0.0).is_none());
        assert!(kde_evaluate(&[1.0], 0.0, 0.0).is_none());
        assert!(kde_evaluate(&[1.0], -1.0, 0.0).is_none());
        assert!(kde_evaluate(&[1.0], 1.0, f64::NAN).is_none());
        assert!(kde_evaluate(&[f64::NAN], 1.0, 0.0).is_none());
    }

    #[test]
    fn kde_bandwidth_methods() {
        let data: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let h_silv = kde_bandwidth(&data, BandwidthMethod::Silverman).expect("silverman");
        let h_scott = kde_bandwidth(&data, BandwidthMethod::Scott).expect("scott");
        let h_manual = kde_bandwidth(&data, BandwidthMethod::Manual(2.5)).expect("manual");
        assert!(h_silv > 0.0);
        assert!(h_scott > 0.0);
        assert!((h_manual - 2.5).abs() < 1e-10);
        // Scott produces slightly larger bandwidth than Silverman for uniform data
        // (1.06σ vs 0.9*min(σ, IQR/1.34))
        assert!(h_scott > h_silv);
    }

    #[test]
    fn kde_bandwidth_edge_cases() {
        assert!(kde_bandwidth(&[1.0], BandwidthMethod::Silverman).is_none());
        assert!(kde_bandwidth(&[5.0, 5.0], BandwidthMethod::Silverman).is_none());
        assert!(kde_bandwidth(&[1.0, 2.0], BandwidthMethod::Manual(0.0)).is_none());
    }

    #[test]
    fn kde_grid_extends_beyond_data() {
        let data = [10.0, 20.0, 30.0];
        let r = kde(&data, BandwidthMethod::Manual(2.0), 64).expect("should compute");
        // Grid should extend 3*h = 6 beyond data range [10, 30]
        assert!(r.x[0] < 10.0 - 2.0, "grid starts at {}", r.x[0]);
        assert!(
            *r.x.last().expect("non-empty") > 30.0 + 2.0,
            "grid ends at {}",
            r.x.last().expect("non-empty")
        );
    }

    // -----------------------------------------------------------------------
    // Distribution Fitting
    // -----------------------------------------------------------------------

    #[test]
    fn fit_normal_basic() {
        let data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let r = fit_normal(&data).expect("should compute");
        assert_eq!(r.distribution, "Normal");
        assert_eq!(r.n_params, 2);
        assert!((r.parameters[0].1 - 5.0).abs() < 1e-10, "μ̂ = 5.0");
        // Biased MLE σ̂ = √(Σ(xᵢ-x̄)²/n) = √(32/8) = 2.0
        // (xᵢ-5)² = 9,1,1,1,0,0,4,16 → sum = 32
        let expected_sigma = (32.0_f64 / 8.0).sqrt();
        assert!(
            (r.parameters[1].1 - expected_sigma).abs() < 1e-10,
            "σ̂ = {}, expected {}",
            r.parameters[1].1,
            expected_sigma
        );
        assert!(r.log_likelihood.is_finite());
        assert!(r.aic.is_finite());
        assert!(r.bic.is_finite());
    }

    #[test]
    fn fit_normal_edge_cases() {
        assert!(fit_normal(&[1.0]).is_none()); // < 2
        assert!(fit_normal(&[]).is_none());
        assert!(fit_normal(&[1.0, f64::NAN]).is_none());
        assert!(fit_normal(&[5.0, 5.0, 5.0]).is_none()); // zero variance
    }

    #[test]
    fn fit_exponential_basic() {
        let data = [0.5, 1.2, 0.8, 2.1, 0.3, 1.5, 0.9, 1.8];
        let r = fit_exponential(&data).expect("should compute");
        assert_eq!(r.distribution, "Exponential");
        assert_eq!(r.n_params, 1);
        let mean: f64 = data.iter().sum::<f64>() / data.len() as f64;
        let expected_lambda = 1.0 / mean;
        assert!(
            (r.parameters[0].1 - expected_lambda).abs() < 1e-10,
            "λ̂ = {}, expected {}",
            r.parameters[0].1,
            expected_lambda
        );
    }

    #[test]
    fn fit_exponential_edge_cases() {
        assert!(fit_exponential(&[1.0]).is_none()); // < 2
        assert!(fit_exponential(&[1.0, -1.0]).is_none()); // negative
        assert!(fit_exponential(&[0.0, 1.0]).is_none()); // zero value
        assert!(fit_exponential(&[1.0, f64::NAN]).is_none());
    }

    #[test]
    fn fit_gamma_basic() {
        // Generate Gamma-like data (known α ≈ 4, β ≈ 2, mean = α/β = 2)
        let data = [
            1.5, 2.1, 1.8, 2.5, 1.9, 2.3, 2.0, 1.7, 2.4, 2.2, 1.6, 2.6,
            1.4, 2.8, 2.1, 1.9, 2.3, 2.0, 1.8, 2.5,
        ];
        let r = fit_gamma(&data).expect("should compute");
        assert_eq!(r.distribution, "Gamma");
        assert_eq!(r.n_params, 2);
        assert!(r.parameters[0].1 > 0.0, "α̂ > 0");
        assert!(r.parameters[1].1 > 0.0, "β̂ > 0");
        assert!(r.log_likelihood.is_finite());
    }

    #[test]
    fn fit_gamma_edge_cases() {
        assert!(fit_gamma(&[1.0]).is_none()); // < 2
        assert!(fit_gamma(&[1.0, -1.0]).is_none()); // negative
        assert!(fit_gamma(&[0.0, 1.0]).is_none()); // zero
        assert!(fit_gamma(&[1.0, f64::NAN]).is_none());
        assert!(fit_gamma(&[5.0, 5.0, 5.0]).is_none()); // zero variance
    }

    #[test]
    fn fit_gamma_mean_rate_relationship() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let r = fit_gamma(&data).expect("should compute");
        let alpha = r.parameters[0].1;
        let beta = r.parameters[1].1;
        let mean: f64 = data.iter().sum::<f64>() / data.len() as f64;
        // β̂ = α̂/x̄, so α̂/β̂ ≈ x̄
        assert!(
            (alpha / beta - mean).abs() < 1e-6,
            "α/β = {}, mean = {}",
            alpha / beta,
            mean
        );
    }

    #[test]
    fn fit_best_sorted_by_aic() {
        let data = [2.1, 3.5, 1.8, 4.2, 2.9, 3.1, 2.5, 3.8, 1.5, 2.7];
        let fits = fit_best(&data);
        assert!(!fits.is_empty());
        // Should include Normal and Gamma (and Exponential since all > 0)
        assert!(fits.len() >= 2, "got {} fits", fits.len());
        // Verify AIC ordering
        for w in fits.windows(2) {
            assert!(
                w[0].aic <= w[1].aic,
                "AIC not sorted: {} > {}",
                w[0].aic,
                w[1].aic
            );
        }
    }

    #[test]
    fn fit_best_mixed_sign_data() {
        // Negative values → only Normal should succeed
        let data = [-2.0, -1.0, 0.5, 1.0, 2.0, 3.0, 1.5, -0.5];
        let fits = fit_best(&data);
        assert_eq!(fits.len(), 1, "only Normal should fit");
        assert_eq!(fits[0].distribution, "Normal");
    }

    #[test]
    fn digamma_known_values() {
        // ψ(1) = -γ ≈ -0.5772156649
        assert!((digamma(1.0) + 0.5772156649).abs() < 1e-8);
        // ψ(2) = 1 - γ ≈ 0.4227843351
        assert!((digamma(2.0) - 0.4227843351).abs() < 1e-8);
        // ψ(0.5) = -γ - 2·ln(2) ≈ -1.9635100260
        assert!((digamma(0.5) + 1.9635100260).abs() < 1e-7);
    }

    #[test]
    fn trigamma_known_values() {
        // ψ'(1) = π²/6 ≈ 1.6449340668
        assert!(
            (trigamma(1.0) - std::f64::consts::PI.powi(2) / 6.0).abs() < 1e-7
        );
        // ψ'(2) = π²/6 - 1 ≈ 0.6449340668
        assert!(
            (trigamma(2.0) - (std::f64::consts::PI.powi(2) / 6.0 - 1.0)).abs() < 1e-7
        );
    }

    // -----------------------------------------------------------------------
    // LogNormal fitting
    // -----------------------------------------------------------------------

    #[test]
    fn fit_lognormal_basic() {
        let data = [1.2, 2.5, 1.8, 3.1, 0.9, 2.0, 1.5, 4.0];
        let r = fit_lognormal(&data).expect("should compute");
        assert_eq!(r.distribution, "LogNormal");
        assert_eq!(r.n_params, 2);
        assert!(r.parameters[1].1 > 0.0); // σ̂ > 0
        assert!(r.log_likelihood.is_finite());
    }

    #[test]
    fn fit_lognormal_params_match_log_transform() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let r = fit_lognormal(&data).expect("should compute");
        let mu = r.parameters[0].1;
        let sigma = r.parameters[1].1;
        // μ̂ should be mean of ln(data)
        let log_data: Vec<f64> = data.iter().map(|&x| x.ln()).collect();
        let expected_mu: f64 = log_data.iter().sum::<f64>() / 5.0;
        assert!(
            (mu - expected_mu).abs() < 1e-10,
            "μ = {}, expected {}",
            mu,
            expected_mu
        );
        assert!(sigma > 0.0);
    }

    #[test]
    fn fit_lognormal_edge_cases() {
        assert!(fit_lognormal(&[1.0]).is_none()); // < 2
        assert!(fit_lognormal(&[1.0, -1.0]).is_none()); // negative
        assert!(fit_lognormal(&[0.0, 1.0]).is_none()); // zero
    }

    // -----------------------------------------------------------------------
    // Poisson fitting
    // -----------------------------------------------------------------------

    #[test]
    fn fit_poisson_basic() {
        let data = [2.0, 3.0, 1.0, 4.0, 2.0, 3.0, 1.0, 2.0];
        let r = fit_poisson(&data).expect("should compute");
        assert_eq!(r.distribution, "Poisson");
        assert_eq!(r.n_params, 1);
        let mean: f64 = data.iter().sum::<f64>() / data.len() as f64;
        assert!(
            (r.parameters[0].1 - mean).abs() < 1e-10,
            "λ̂ = {}, expected {}",
            r.parameters[0].1,
            mean
        );
    }

    #[test]
    fn fit_poisson_edge_cases() {
        assert!(fit_poisson(&[1.0]).is_none()); // < 2
        assert!(fit_poisson(&[1.0, -1.0]).is_none()); // negative
        assert!(fit_poisson(&[0.0, 0.0]).is_none()); // all zeros
    }

    // -----------------------------------------------------------------------
    // Beta fitting
    // -----------------------------------------------------------------------

    #[test]
    fn fit_beta_basic() {
        let data = [0.2, 0.35, 0.5, 0.15, 0.45, 0.3, 0.6, 0.25, 0.4, 0.55];
        let r = fit_beta(&data).expect("should compute");
        assert_eq!(r.distribution, "Beta");
        assert_eq!(r.n_params, 2);
        assert!(r.parameters[0].1 > 0.0, "α̂ > 0");
        assert!(r.parameters[1].1 > 0.0, "β̂ > 0");
        assert!(r.log_likelihood.is_finite());
    }

    #[test]
    fn fit_beta_symmetric() {
        // Symmetric data around 0.5 → α ≈ β
        let data = [0.3, 0.4, 0.5, 0.6, 0.7, 0.35, 0.45, 0.55, 0.65, 0.5];
        let r = fit_beta(&data).expect("should compute");
        let alpha = r.parameters[0].1;
        let beta = r.parameters[1].1;
        // α and β should be roughly equal for symmetric data
        assert!(
            (alpha - beta).abs() / alpha < 0.3,
            "α = {alpha}, β = {beta}, should be roughly equal"
        );
    }

    #[test]
    fn fit_beta_edge_cases() {
        assert!(fit_beta(&[0.5]).is_none()); // < 2
        assert!(fit_beta(&[0.5, 1.0]).is_none()); // value = 1.0 (boundary)
        assert!(fit_beta(&[0.0, 0.5]).is_none()); // value = 0.0 (boundary)
        assert!(fit_beta(&[0.5, 1.5]).is_none()); // value > 1
        assert!(fit_beta(&[0.5, -0.5]).is_none()); // value < 0
    }

    // -----------------------------------------------------------------------
    // fit_best (updated)
    // -----------------------------------------------------------------------

    #[test]
    fn fit_best_includes_lognormal_and_poisson() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let fits = fit_best(&data);
        let names: Vec<&str> = fits.iter().map(|f| f.distribution.as_str()).collect();
        // All > 0, so should include Normal, Exponential, Gamma, LogNormal, Poisson
        assert!(names.contains(&"Normal"), "should include Normal");
        assert!(names.contains(&"Exponential"), "should include Exponential");
        assert!(names.contains(&"Gamma"), "should include Gamma");
        assert!(names.contains(&"LogNormal"), "should include LogNormal");
        assert!(names.contains(&"Poisson"), "should include Poisson");
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn ecdf_last_is_one(
            data in proptest::collection::vec(-1e3_f64..1e3, 1..=50)
        ) {
            if let Some((_, probs)) = ecdf(&data) {
                let last = *probs.last().expect("non-empty");
                prop_assert!((last - 1.0).abs() < 1e-10, "last prob = {last}");
            }
        }

        #[test]
        fn ecdf_monotonic(
            data in proptest::collection::vec(-1e3_f64..1e3, 2..=50)
        ) {
            if let Some((_, probs)) = ecdf(&data) {
                for w in probs.windows(2) {
                    prop_assert!(w[1] >= w[0] - 1e-15, "{} < {}", w[1], w[0]);
                }
            }
        }

        #[test]
        fn hist_counts_sum_to_n(
            data in proptest::collection::vec(-1e3_f64..1e3, 5..=100)
        ) {
            if let Some(r) = histogram_bins(&data, BinMethod::Sturges) {
                let total: usize = r.counts.iter().sum();
                prop_assert_eq!(total, data.len());
            }
        }

        #[test]
        fn kde_density_non_negative(
            data in proptest::collection::vec(-100.0_f64..100.0, 5..=50)
        ) {
            if let Some(r) = kde(&data, BandwidthMethod::Silverman, 128) {
                for &d in &r.density {
                    prop_assert!(d >= 0.0, "negative density: {d}");
                }
            }
        }

        #[test]
        fn kde_integral_approx_one(
            data in proptest::collection::vec(-100.0_f64..100.0, 10..=80)
        ) {
            if let Some(r) = kde(&data, BandwidthMethod::Silverman, 512) {
                let dx = r.x[1] - r.x[0];
                let integral: f64 = r.density.iter().sum::<f64>() * dx;
                prop_assert!(
                    (integral - 1.0).abs() < 0.1,
                    "integral = {integral}, expected ≈ 1.0"
                );
            }
        }

        #[test]
        fn fit_normal_aic_finite(
            data in proptest::collection::vec(-100.0_f64..100.0, 5..=50)
        ) {
            if let Some(r) = fit_normal(&data) {
                prop_assert!(r.aic.is_finite(), "AIC = {}", r.aic);
                prop_assert!(r.bic.is_finite(), "BIC = {}", r.bic);
                prop_assert!(r.log_likelihood.is_finite(), "LL = {}", r.log_likelihood);
            }
        }

        #[test]
        fn fit_gamma_alpha_positive(
            data in proptest::collection::vec(0.1_f64..100.0, 5..=50)
        ) {
            if let Some(r) = fit_gamma(&data) {
                let alpha = r.parameters[0].1;
                let beta = r.parameters[1].1;
                prop_assert!(alpha > 0.0, "α = {alpha}");
                prop_assert!(beta > 0.0, "β = {beta}");
                prop_assert!(r.aic.is_finite(), "AIC = {}", r.aic);
            }
        }
    }
}
