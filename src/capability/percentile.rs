//! Percentile-based process capability (ISO 22514-2).
//!
//! When process data is non-normal, capability indices based on mean and
//! standard deviation can be misleading. The percentile method replaces the
//! ±3σ spread with the distance between the 0.135 % and 99.865 % quantiles of
//! the process distribution, and the mean with its median.
//!
//! Those quantiles are taken from a **distribution fitted to the data**, not
//! from the data's own order statistics. With fewer than about 740 values the
//! sample's 0.135 % point is its minimum (and the 99.865 % point its maximum),
//! so an empirical spread is the sample range -- which is shorter than the
//! process spread, and the shorter the smaller the sample: on normal data with
//! a true Cp of 1.0 it reported about 1.37 at n = 50. The fitted model is the
//! one [`fit_best`](crate::distribution::fit_best) ranks first by AIC, or the
//! family the caller names.
//!
//! # Indices
//!
//! - **Cp\*** — (USL − LSL) / (X₉₉.₈₆₅ − X₀.₁₃₅)
//! - **Cpk\*** — min(Cpu\*, Cpl\*)
//! - **Cpu\*** — (USL − X₅₀) / (X₉₉.₈₆₅ − X₅₀)
//! - **Cpl\*** — (X₅₀ − LSL) / (X₅₀ − X₀.₁₃₅)
//!
//! # References
//!
//! - ISO 22514-2, "Statistical methods in process management — Capability and
//!   performance — Part 2: Process capability and performance of time-dependent
//!   process models" (quantile method on an identified distribution).
//! - Clements, J.A. (1989). "Process capability calculations for non-normal
//!   distributions." *Quality Progress*, 22(9), 95–100 (quantiles of a fitted
//!   distribution, there Pearson curves).

use std::fmt;

use u_numflow::distributions::{
    BetaDistribution, Exponential, GammaDistribution, LogNormal, Normal, Weibull,
};

use crate::distribution::{
    fit_best, fit_beta, fit_exponential, fit_gamma, fit_lognormal, fit_normal, fit_weibull,
    FitResult,
};

/// Lower tail probability: the normal 3σ point, Φ(−3).
const P_LOWER: f64 = 0.001_349_898_031_630_094_6;
/// Upper tail probability: Φ(3).
const P_UPPER: f64 = 1.0 - P_LOWER;

/// Fewest values the method accepts.
pub const MIN_PERCENTILE_DATA: usize = 20;

/// The distribution family a percentile capability is computed from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum CapabilityModel {
    /// The family [`fit_best`] ranks first by AIC.
    #[default]
    Auto,
    /// Normal.
    Normal,
    /// Log-normal (positive data).
    LogNormal,
    /// Two-parameter Weibull (positive data).
    Weibull,
    /// Gamma (positive data).
    Gamma,
    /// Exponential (positive data).
    Exponential,
    /// Beta (data within (0, 1)).
    Beta,
}

impl CapabilityModel {
    /// The fitted families, in the names [`FitResult::distribution`] uses.
    fn of_fit(name: &str) -> Option<Self> {
        Some(match name {
            "Normal" => Self::Normal,
            "LogNormal" => Self::LogNormal,
            "Weibull" => Self::Weibull,
            "Gamma" => Self::Gamma,
            "Exponential" => Self::Exponential,
            "Beta" => Self::Beta,
            _ => return None,
        })
    }

    fn fit(self, data: &[f64]) -> Option<FitResult> {
        match self {
            Self::Auto => fit_best(data)
                .into_iter()
                .find(|f| Self::of_fit(&f.distribution).is_some()),
            Self::Normal => fit_normal(data),
            Self::LogNormal => fit_lognormal(data),
            Self::Weibull => fit_weibull(data),
            Self::Gamma => fit_gamma(data),
            Self::Exponential => fit_exponential(data),
            Self::Beta => fit_beta(data),
        }
    }
}

/// Why a percentile capability was refused.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum PercentileCapabilityError {
    /// Neither USL nor LSL was given.
    NoSpecification,
    /// USL is not above LSL.
    InvalidSpecification,
    /// Fewer than [`MIN_PERCENTILE_DATA`] values.
    InsufficientData {
        /// How many were given.
        got: usize,
    },
    /// A value is NaN or infinite.
    NonFiniteData {
        /// Position of the first one.
        index: usize,
    },
    /// Every value is the same.
    NoVariation,
    /// The named family cannot describe the data (log-normal, Weibull, gamma
    /// and exponential need positive values, beta values within (0, 1)).
    ModelNotFitted(CapabilityModel),
}

impl fmt::Display for PercentileCapabilityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoSpecification => {
                write!(f, "at least one specification limit (USL or LSL) is required")
            }
            Self::InvalidSpecification => write!(f, "USL must be greater than LSL"),
            Self::InsufficientData { got } => write!(
                f,
                "at least {MIN_PERCENTILE_DATA} data points are required for percentile capability, got {got}"
            ),
            Self::NonFiniteData { index } => {
                write!(f, "data[{index}] is not a finite number")
            }
            Self::NoVariation => write!(f, "data has no variation"),
            Self::ModelNotFitted(model) => {
                write!(f, "the {model:?} distribution cannot be fitted to this data")
            }
        }
    }
}

impl std::error::Error for PercentileCapabilityError {}

/// Result of percentile-based process capability analysis.
#[derive(Debug, Clone)]
pub struct PercentileCapabilityResult {
    /// Cp* = (USL − LSL) / (X₉₉.₈₆₅ − X₀.₁₃₅). Requires both limits.
    pub cp_star: Option<f64>,
    /// Cpk* = min(Cpu*, Cpl*). Requires at least one limit.
    pub cpk_star: Option<f64>,
    /// Cpu* = (USL − X₅₀) / (X₉₉.₈₆₅ − X₅₀). Requires USL.
    pub cpu_star: Option<f64>,
    /// Cpl* = (X₅₀ − LSL) / (X₅₀ − X₀.₁₃₅). Requires LSL.
    pub cpl_star: Option<f64>,
    /// Median of the fitted distribution, X₅₀.
    pub median: f64,
    /// 0.135 % quantile of the fitted distribution (lower natural process limit).
    pub percentile_lower: f64,
    /// 99.865 % quantile of the fitted distribution (upper natural process limit).
    pub percentile_upper: f64,
    /// The family the quantiles come from — never [`CapabilityModel::Auto`].
    pub model: CapabilityModel,
    /// Its fitted parameters, as [`FitResult::parameters`] names them.
    pub parameters: Vec<(String, f64)>,
}

/// Computes percentile-based process capability indices.
///
/// Fits `model` to `data` (or, for [`CapabilityModel::Auto`], the family
/// [`fit_best`] ranks first) and takes the 0.135 %, 50 % and 99.865 %
/// quantiles from it.
///
/// # Errors
///
/// [`PercentileCapabilityError`] — no limit, USL ≤ LSL, fewer than
/// [`MIN_PERCENTILE_DATA`] values, a value that is not finite, no variation,
/// or a named family the data does not admit.
///
/// # Examples
///
/// ```
/// use u_analytics::capability::{percentile_capability, CapabilityModel};
///
/// let data: Vec<f64> = (0..100).map(|i| 10.0 + (i as f64) * 0.1).collect();
/// let result = percentile_capability(&data, Some(5.0), Some(15.0), CapabilityModel::Auto).unwrap();
/// assert!(result.cp_star.is_some());
/// assert_ne!(result.model, CapabilityModel::Auto);
/// ```
pub fn percentile_capability(
    data: &[f64],
    lsl: Option<f64>,
    usl: Option<f64>,
    model: CapabilityModel,
) -> Result<PercentileCapabilityResult, PercentileCapabilityError> {
    if usl.is_none() && lsl.is_none() {
        return Err(PercentileCapabilityError::NoSpecification);
    }
    if let (Some(u), Some(l)) = (usl, lsl) {
        if u <= l {
            return Err(PercentileCapabilityError::InvalidSpecification);
        }
    }
    if data.len() < MIN_PERCENTILE_DATA {
        return Err(PercentileCapabilityError::InsufficientData { got: data.len() });
    }
    if let Some(index) = data.iter().position(|v| !v.is_finite()) {
        return Err(PercentileCapabilityError::NonFiniteData { index });
    }
    if data.iter().all(|&v| v == data[0]) {
        return Err(PercentileCapabilityError::NoVariation);
    }

    let fit = model
        .fit(data)
        .ok_or(PercentileCapabilityError::ModelNotFitted(model))?;
    let fitted = CapabilityModel::of_fit(&fit.distribution)
        .ok_or(PercentileCapabilityError::ModelNotFitted(model))?;
    let (p_lower, median, p_upper) =
        quantiles(fitted, &fit).ok_or(PercentileCapabilityError::ModelNotFitted(model))?;

    let spread = p_upper - p_lower;
    if spread <= 0.0 {
        return Err(PercentileCapabilityError::NoVariation);
    }
    let upper_spread = p_upper - median;
    let lower_spread = median - p_lower;

    let cp_star = match (usl, lsl) {
        (Some(u), Some(l)) => Some((u - l) / spread),
        _ => None,
    };
    let cpu_star = usl.and_then(|u| (upper_spread > 0.0).then(|| (u - median) / upper_spread));
    let cpl_star = lsl.and_then(|l| (lower_spread > 0.0).then(|| (median - l) / lower_spread));
    let cpk_star = match (cpu_star, cpl_star) {
        (Some(u), Some(l)) => Some(u.min(l)),
        (one, other) => one.or(other),
    };

    Ok(PercentileCapabilityResult {
        cp_star,
        cpk_star,
        cpu_star,
        cpl_star,
        median,
        percentile_lower: p_lower,
        percentile_upper: p_upper,
        model: fitted,
        parameters: fit.parameters,
    })
}

/// The 0.135 %, 50 % and 99.865 % quantiles of a fitted family.
fn quantiles(model: CapabilityModel, fit: &FitResult) -> Option<(f64, f64, f64)> {
    let param = |name: &str| {
        fit.parameters
            .iter()
            .find(|(n, _)| n == name)
            .map(|&(_, v)| v)
    };
    let q: Box<dyn Fn(f64) -> Option<f64>> = match model {
        CapabilityModel::Normal => {
            let d = Normal::new(param("mu")?, param("sigma")?).ok()?;
            Box::new(move |p| d.quantile(p))
        }
        CapabilityModel::LogNormal => {
            let d = LogNormal::new(param("mu")?, param("sigma")?).ok()?;
            Box::new(move |p| d.quantile(p))
        }
        CapabilityModel::Weibull => {
            let d = Weibull::new(param("shape")?, param("scale")?).ok()?;
            Box::new(move |p| d.quantile(p))
        }
        CapabilityModel::Gamma => {
            // `fit_gamma` reports the rate as `beta`.
            let d = GammaDistribution::new(param("alpha")?, param("beta")?).ok()?;
            Box::new(move |p| d.quantile(p))
        }
        CapabilityModel::Exponential => {
            let d = Exponential::new(param("lambda")?).ok()?;
            Box::new(move |p| d.quantile(p))
        }
        CapabilityModel::Beta => {
            let d = BetaDistribution::new(param("alpha")?, param("beta")?).ok()?;
            Box::new(move |p| d.quantile(p))
        }
        CapabilityModel::Auto => return None,
    };
    let (lo, mid, hi) = (q(P_LOWER)?, q(0.5)?, q(P_UPPER)?);
    (lo.is_finite() && mid.is_finite() && hi.is_finite()).then_some((lo, mid, hi))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use u_numflow::distributions::Sample;

    /// Generate uniform-like data for testing.
    fn uniform_data(n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| 10.0 + (i as f64) / (n as f64 - 1.0) * 10.0)
            .collect()
    }

    fn pc(
        data: &[f64],
        lsl: Option<f64>,
        usl: Option<f64>,
    ) -> Result<PercentileCapabilityResult, PercentileCapabilityError> {
        percentile_capability(data, lsl, usl, CapabilityModel::Auto)
    }

    #[test]
    fn basic_two_sided() {
        let data = uniform_data(100);
        let result = pc(&data, Some(5.0), Some(25.0)).unwrap();

        assert!(result.cp_star.is_some());
        assert!(result.cpk_star.is_some());
        assert!(result.cpu_star.is_some());
        assert!(result.cpl_star.is_some());
        assert!(result.cp_star.unwrap() > 0.0);
        assert!(
            (result.median - 15.0).abs() < 0.5,
            "median should be ~15, got {}",
            result.median
        );
    }

    #[test]
    fn usl_only() {
        let data = uniform_data(50);
        let result = pc(&data, None, Some(25.0)).unwrap();

        assert!(result.cp_star.is_none(), "Cp* requires both limits");
        assert!(result.cpu_star.is_some());
        assert!(result.cpl_star.is_none());
        assert_eq!(result.cpk_star, result.cpu_star);
    }

    #[test]
    fn lsl_only() {
        let data = uniform_data(50);
        let result = pc(&data, Some(5.0), None).unwrap();

        assert!(result.cp_star.is_none());
        assert!(result.cpu_star.is_none());
        assert!(result.cpl_star.is_some());
        assert_eq!(result.cpk_star, result.cpl_star);
    }

    #[test]
    fn refusals_name_their_reason() {
        let few: Vec<f64> = (0..19).map(|i| i as f64).collect();
        assert_eq!(
            pc(&few, Some(0.0), Some(20.0)).unwrap_err(),
            PercentileCapabilityError::InsufficientData { got: 19 }
        );
        let data = uniform_data(30);
        assert_eq!(
            pc(&data, None, None).unwrap_err(),
            PercentileCapabilityError::NoSpecification
        );
        assert_eq!(
            pc(&data, Some(10.0), Some(5.0)).unwrap_err(),
            PercentileCapabilityError::InvalidSpecification
        );
        assert_eq!(
            pc(&data, Some(10.0), Some(10.0)).unwrap_err(),
            PercentileCapabilityError::InvalidSpecification
        );
        let mut nan = uniform_data(30);
        nan[10] = f64::NAN;
        assert_eq!(
            pc(&nan, Some(0.0), Some(30.0)).unwrap_err(),
            PercentileCapabilityError::NonFiniteData { index: 10 }
        );
        assert_eq!(
            pc(&[3.0; 25], Some(0.0), Some(30.0)).unwrap_err(),
            PercentileCapabilityError::NoVariation
        );
        let mut negative = uniform_data(30);
        negative[0] = -1.0;
        assert_eq!(
            percentile_capability(
                &negative,
                Some(-5.0),
                Some(30.0),
                CapabilityModel::LogNormal
            )
            .unwrap_err(),
            PercentileCapabilityError::ModelNotFitted(CapabilityModel::LogNormal)
        );
    }

    #[test]
    fn cpk_star_equals_min_of_cpu_cpl() {
        let data = uniform_data(100);
        let result = pc(&data, Some(5.0), Some(25.0)).unwrap();
        let (cpu, cpl, cpk) = (
            result.cpu_star.unwrap(),
            result.cpl_star.unwrap(),
            result.cpk_star.unwrap(),
        );
        assert!((cpk - cpu.min(cpl)).abs() < 1e-10);
    }

    #[test]
    fn symmetric_data_gives_balanced_cpu_cpl() {
        let data: Vec<f64> = (0..200).map(|i| 40.0 + (i as f64) * 0.1).collect();
        let result = pc(&data, Some(30.0), Some(70.0)).unwrap();
        let (cpu, cpl) = (result.cpu_star.unwrap(), result.cpl_star.unwrap());
        assert!(
            (cpu - cpl).abs() < cpu * 0.2,
            "symmetric data should give similar Cpu* ({cpu}) and Cpl* ({cpl})"
        );
    }

    /// The defect this module was rewritten for: on normal data with a true Cp
    /// of 1.0 (limits at ±3σ), the empirical 0.135 % / 99.865 % points are the
    /// sample minimum and maximum, so Cp* averaged 1.37 at n = 50 and 1.68 at
    /// n = 20. Quantiles of the fitted normal stay near 1.0 at every size.
    #[test]
    fn small_samples_do_not_inflate_cp_star() {
        let process = Normal::new(10.0, 1.0).unwrap();
        for (n, empirical_at_least) in [(20_usize, 1.5), (50, 1.25), (100, 1.15)] {
            let reps = 400;
            let (mut sum, mut empirical) = (0.0, 0.0);
            for rep in 0..reps {
                let data = process.sample_n(n, 1_000 + rep as u64 * 7 + n as u64);
                let r =
                    percentile_capability(&data, Some(7.0), Some(13.0), CapabilityModel::Normal)
                        .unwrap();
                sum += r.cp_star.unwrap();
                // What the empirical points gave on the same samples.
                let lo = u_numflow::stats::quantile(&data, P_LOWER).unwrap();
                let hi = u_numflow::stats::quantile(&data, P_UPPER).unwrap();
                empirical += 6.0 / (hi - lo);
            }
            let mean = sum / reps as f64;
            let empirical = empirical / reps as f64;
            assert!(
                empirical > empirical_at_least,
                "n = {n}: the empirical Cp* {empirical} no longer shows the bias this guards"
            );
            // The MLE σ is low by about (n−1)/n, and 1/σ̂ is biased up; both
            // are a few percent at these sizes. The empirical method was 37 %
            // high at n = 50.
            assert!(
                (mean - 1.0).abs() < 0.08,
                "n = {n}: mean Cp* {mean} (true 1.0)"
            );
        }
    }

    #[test]
    fn quantiles_are_the_fitted_model_s() {
        let process = Normal::new(50.0, 2.0).unwrap();
        let data = process.sample_n(200, 42);
        let r =
            percentile_capability(&data, Some(40.0), Some(60.0), CapabilityModel::Normal).unwrap();
        let mu = r.parameters.iter().find(|(n, _)| n == "mu").unwrap().1;
        let sigma = r.parameters.iter().find(|(n, _)| n == "sigma").unwrap().1;
        assert_eq!(r.model, CapabilityModel::Normal);
        assert!((r.median - mu).abs() < 1e-9);
        assert!((r.percentile_lower - (mu - 3.0 * sigma)).abs() < 1e-6);
        assert!((r.percentile_upper - (mu + 3.0 * sigma)).abs() < 1e-6);
    }

    #[test]
    fn auto_reports_the_family_it_used() {
        let process = LogNormal::new(1.0, 0.6).unwrap();
        let data = process.sample_n(300, 7);
        let r = pc(&data, None, Some(30.0)).unwrap();
        assert_ne!(r.model, CapabilityModel::Auto);
        assert_eq!(
            CapabilityModel::of_fit(&fit_best(&data)[0].distribution),
            Some(r.model)
        );
    }
}

/// Against SciPy 1.17.1: the family fitted by maximum likelihood
/// (`lognorm.fit(x, floc=0)`; the Weibull's from its likelihood equation), its
/// `ppf` at `norm.cdf(-3)`, 0.5 and `norm.cdf(3)` -- the equivalents of
/// ±3 sigma that ISO 22514-2 rounds to 0.135 % and 99.865 %, which this
/// function takes exactly -- and the ISO 22514-2 ratios taken from them.
#[cfg(test)]
mod scipy_reference {
    use super::*;

    const DATA: [f64; 40] = [
        3.195, 5.365, 1.217, 5.806, 2.661, 1.896, 1.894, 1.67, 2.458, 3.956, 3.535, 3.623, 1.268,
        1.341, 5.47, 4.204, 7.26, 4.685, 1.714, 4.847, 3.607, 2.994, 1.88, 2.722, 2.544, 4.062,
        2.875, 5.546, 1.918, 3.36, 3.402, 1.842, 1.597, 4.289, 3.431, 2.527, 3.246, 2.006, 3.026,
        2.07,
    ];

    fn close(got: f64, want: f64, rel: f64, what: &str) {
        assert!(
            (got - want).abs() <= rel * want.abs(),
            "{what}: got {got}, SciPy {want}"
        );
    }

    fn check(model: CapabilityModel, rel: f64, want: [f64; 7]) {
        let r = percentile_capability(&DATA, Some(0.9), Some(9.0), model).expect("valid");
        assert_eq!(r.model, model);
        let [lo, med, hi, cp, cpk, cpu, cpl] = want;
        close(r.percentile_lower, lo, rel, "X0.135");
        close(r.median, med, rel, "X50");
        close(r.percentile_upper, hi, rel, "X99.865");
        close(r.cp_star.expect("both limits"), cp, rel, "Cp*");
        close(r.cpk_star.expect("limits"), cpk, rel, "Cpk*");
        close(r.cpu_star.expect("USL"), cpu, rel, "Cpu*");
        close(r.cpl_star.expect("LSL"), cpl, rel, "Cpl*");
    }

    #[test]
    fn log_normal_percentile_capability_reproduces_scipy() {
        // The log-normal MLE is closed form: tight.
        check(
            CapabilityModel::LogNormal,
            1e-9,
            [
                0.7542968892328896,
                2.8798090991738787,
                10.994743006455773,
                0.7909811650077455,
                0.754188631817962,
                0.754188631817962,
                0.9314503534321473,
            ],
        );
    }

    #[test]
    fn weibull_percentile_capability_reproduces_scipy() {
        // The exact MLE: the shape's likelihood equation solved to 1e-15
        // (`scipy.optimize.brentq`), the scale from it. `weibull_min.fit` stops
        // short of it -- its shape is 2.413418 against 2.413394 -- so its
        // quantiles are not the reference.
        check(
            CapabilityModel::Weibull,
            1e-9,
            [
                0.23258289429884546,
                3.0872615285886305,
                7.858145308508201,
                1.0622167336676158,
                0.7662023676906101,
                1.2393381906090049,
                0.7662023676906101,
            ],
        );
    }
}
