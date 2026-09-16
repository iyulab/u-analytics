//! Non-normal process capability using Box-Cox transformation.
//!
//! When process data is non-normal (e.g., right-skewed exponential-like data),
//! standard capability indices (Cp, Cpk) computed on raw data can be misleading.
//! This module applies the Box-Cox power transformation to map the data to
//! approximate normality, then computes standard capability indices on the
//! transformed scale.
//!
//! # Algorithm
//!
//! 1. Estimate the optimal λ via maximum likelihood (`estimate_lambda`) over a
//!    search range — by default `[-5, 5]`, the range Minitab searches.
//! 2. Transform the data: `y(λ)`.
//! 3. Transform the specification limits using the same λ.
//! 4. Compute capability indices on the transformed scale.
//!
//! # References
//!
//! - Box, G. E. P. & Cox, D. R. (1964). "An analysis of transformations."
//!   *Journal of the Royal Statistical Society, Series B*, 26(2), 211–252.
//! - Clements, J. A. (1989). "Process capability calculations for non-normal
//!   distributions." *Quality Progress*, 22(9), 95–100.

use std::fmt;

use u_numflow::transforms::{box_cox, estimate_lambda, TransformError};

use crate::capability::{CapabilityIndices, ProcessCapability};

/// The default λ search range, `[-5, 5]`.
///
/// This is the range Minitab searches for the optimal Box-Cox λ. Practice
/// commonly prefers a λ within `[-2, 2]`; a narrower range can be passed, and
/// [`NonNormalCapabilityResult::lambda_at_bound`] reports when it cut the
/// search short.
pub const DEFAULT_LAMBDA_RANGE: (f64, f64) = (-5.0, 5.0);

// ── Error type ────────────────────────────────────────────────────────────────

/// Errors that can arise from non-normal process capability analysis.
#[derive(Debug, Clone, PartialEq)]
pub enum NonNormalCapabilityError {
    /// All data values must be strictly positive for Box-Cox transformation.
    NonPositiveData,
    /// Data contains NaN or an infinity.
    NonFiniteData,
    /// At least 4 data points are required for reliable capability analysis.
    InsufficientData,
    /// Failed to transform a specification limit (e.g., limit is not positive).
    SpecTransformError,
    /// Capability computation failed (e.g., all transformed data are identical).
    CapabilityError,
    /// The λ search range is not finite with `min < max`.
    InvalidLambdaRange,
}

impl fmt::Display for NonNormalCapabilityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NonNormalCapabilityError::NonPositiveData => {
                write!(
                    f,
                    "Box-Cox requires all data values to be strictly positive"
                )
            }
            NonNormalCapabilityError::NonFiniteData => {
                write!(f, "data must not contain NaN or infinite values")
            }
            NonNormalCapabilityError::InsufficientData => {
                write!(
                    f,
                    "at least 4 data points are required for capability analysis"
                )
            }
            NonNormalCapabilityError::SpecTransformError => {
                write!(
                    f,
                    "failed to transform specification limit — limit must be positive"
                )
            }
            NonNormalCapabilityError::CapabilityError => {
                write!(
                    f,
                    "capability computation failed — check that data has non-zero variance"
                )
            }
            NonNormalCapabilityError::InvalidLambdaRange => {
                write!(f, "lambda range must be finite with min < max")
            }
        }
    }
}

impl std::error::Error for NonNormalCapabilityError {}

impl From<TransformError> for NonNormalCapabilityError {
    fn from(e: TransformError) -> Self {
        match e {
            TransformError::NonPositiveData => NonNormalCapabilityError::NonPositiveData,
            TransformError::NonFiniteData => NonNormalCapabilityError::NonFiniteData,
            TransformError::InsufficientData => NonNormalCapabilityError::InsufficientData,
            TransformError::InvalidInverse => NonNormalCapabilityError::SpecTransformError,
            // The data at the estimated λ does not fit in f64: nothing to compute on.
            TransformError::InvalidTransform => NonNormalCapabilityError::CapabilityError,
            TransformError::InvalidLambdaRange => NonNormalCapabilityError::InvalidLambdaRange,
        }
    }
}

// ── Result type ───────────────────────────────────────────────────────────────

/// Result of a Box-Cox-based non-normal capability analysis.
#[derive(Debug, Clone)]
pub struct NonNormalCapabilityResult {
    /// The estimated optimal Box-Cox transformation parameter λ.
    ///
    /// λ ≈ 0 corresponds to a log transform; λ = 1 is the identity (no transform);
    /// λ = 0.5 is approximately a square-root transform.
    pub lambda: f64,
    /// `true` when the likelihood maximum lies on an end of the λ search range:
    /// the likelihood was still rising there, so `lambda` is that range limit
    /// (exactly) rather than an interior estimate, and the transformed-scale
    /// indices are computed at a λ the data did not choose. Widen the range to
    /// find the unconstrained optimum.
    pub lambda_at_bound: bool,
    /// Capability indices computed on the Box-Cox-transformed scale, or `None`
    /// when no specification limit was given (the transform alone was asked
    /// for).
    ///
    /// Only the **long-term** indices (`pp`, `ppk`, `ppu`, `ppl`) are reported.
    /// `cp`, `cpk`, `cpu` and `cpl` are always `None`: they are defined against
    /// a within-subgroup sigma, and a flat observation vector carries no
    /// subgroup structure to estimate one from. `cpm` follows its usual rule
    /// (both limits and a target, on the transformed scale).
    pub indices: Option<CapabilityIndices>,
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Compute process capability indices for non-normal data via Box-Cox transformation.
///
/// The data are first transformed to approximate normality using the optimal
/// Box-Cox parameter λ, estimated via maximum likelihood over `lambda_range`
/// (usually [`DEFAULT_LAMBDA_RANGE`]). Specification limits are transformed
/// using the same λ. The **long-term** capability indices (Pp, Ppk, Ppu, Ppl)
/// are then computed on the transformed scale. The short-term indices
/// (Cp, Cpk, Cpu, Cpl) are **not** reported — see
/// [`NonNormalCapabilityResult::indices`] for why.
///
/// Specification limits are optional: without either, λ and
/// [`NonNormalCapabilityResult::lambda_at_bound`] are still estimated and
/// `indices` is `None` — the "does a transform achieve normality" half of a
/// non-normal capability workflow comes before the limits are known.
///
/// # Arguments
///
/// * `data` — Process observations. All values must be strictly positive.
/// * `usl` — Upper specification limit (optional). Must be positive if provided.
/// * `lsl` — Lower specification limit (optional). Must be positive if provided.
/// * `lambda_range` — `(min, max)` λ search range, finite with `min < max`.
///
/// # Errors
///
/// Returns [`NonNormalCapabilityError`] if:
/// - Fewer than 4 data points are provided.
/// - Any data value is ≤ 0 (Box-Cox requires strictly positive data).
/// - `lambda_range` is not finite with `min < max`.
/// - A specification limit is ≤ 0 (cannot be Box-Cox transformed).
/// - Capability computation fails (e.g., zero variance in transformed data).
///
/// # Examples
///
/// ```
/// use u_analytics::capability::{boxcox_capability, DEFAULT_LAMBDA_RANGE};
///
/// // Right-skewed data
/// let data: Vec<f64> = (1..=20).map(|i| (i as f64 * 0.3_f64).exp()).collect();
/// let result = boxcox_capability(&data, Some(100.0), Some(1.0), DEFAULT_LAMBDA_RANGE).unwrap();
/// assert!(!result.lambda_at_bound);
/// assert!(result.indices.expect("limits given").ppk.is_some());
///
/// // No limits yet: the transform alone.
/// let lambda_only = boxcox_capability(&data, None, None, DEFAULT_LAMBDA_RANGE).unwrap();
/// assert_eq!(lambda_only.lambda, result.lambda);
/// assert!(lambda_only.indices.is_none());
/// ```
pub fn boxcox_capability(
    data: &[f64],
    usl: Option<f64>,
    lsl: Option<f64>,
    lambda_range: (f64, f64),
) -> Result<NonNormalCapabilityResult, NonNormalCapabilityError> {
    // Validate: sufficient data
    if data.len() < 4 {
        return Err(NonNormalCapabilityError::InsufficientData);
    }

    // Validate: finite, then strictly positive
    if data.iter().any(|v| !v.is_finite()) {
        return Err(NonNormalCapabilityError::NonFiniteData);
    }
    if data.iter().any(|&v| v <= 0.0) {
        return Err(NonNormalCapabilityError::NonPositiveData);
    }

    // Estimate optimal λ
    let estimate = estimate_lambda(data, lambda_range.0, lambda_range.1)?;
    let lambda = estimate.lambda;

    if usl.is_none() && lsl.is_none() {
        return Ok(NonNormalCapabilityResult {
            lambda,
            lambda_at_bound: estimate.at_bound,
            indices: None,
        });
    }

    // Transform data
    let y_t = box_cox(data, lambda)?;

    let transform_limit = |limit: f64| {
        if limit <= 0.0 {
            return Err(NonNormalCapabilityError::SpecTransformError);
        }
        // box_cox needs at least 2 values; pair the limit with data[0] (positive)
        let pair = [limit, data[0]];
        box_cox(&pair, lambda)
            .map(|v| v[0])
            .map_err(|_| NonNormalCapabilityError::SpecTransformError)
    };
    let usl_t = usl.map(transform_limit).transpose()?;
    let lsl_t = lsl.map(transform_limit).transpose()?;

    // Build ProcessCapability on transformed scale
    // ProcessCapability::new validates usl > lsl when both present
    let spec = ProcessCapability::new(usl_t, lsl_t)
        .map_err(|_| NonNormalCapabilityError::CapabilityError)?;

    // A Box-Cox analysis starts from a flat vector, so there is no rational
    // subgrouping and therefore no short-term sigma to estimate: only the
    // long-term indices are reported, which is what `compute_overall` does.
    let indices = spec
        .compute_overall(&y_t)
        .ok_or(NonNormalCapabilityError::CapabilityError)?;

    Ok(NonNormalCapabilityResult {
        lambda,
        lambda_at_bound: estimate.at_bound,
        indices: Some(indices),
    })
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const R: (f64, f64) = DEFAULT_LAMBDA_RANGE;

    fn indices(r: &NonNormalCapabilityResult) -> &CapabilityIndices {
        r.indices.as_ref().expect("limits were given")
    }

    /// A narrow sample's λ sits on the end of the range, and the result says so.
    ///
    /// The estimator lives in the foundation crate, so what this pins is the
    /// path: a caller reaching Box-Cox capability through this function is told
    /// the same thing a caller reaching the estimator directly is told. The
    /// sample is one a consumer reported -- 250 readings inside a 0.13 % band,
    /// where the likelihood is monotone and the estimate used to stop just
    /// short of the bound and call itself interior.
    #[test]
    fn a_narrow_sample_reports_a_lambda_that_sits_on_the_range_limit() {
        let data = vec![
            24.994, 25.007, 24.998, 25.006, 24.993, 25.002, 24.996, 25.004, 24.995, 25.005, 25.007,
            24.995, 25.005, 24.993, 25.006, 24.996, 25.004, 24.998, 25.002, 24.994, 24.996, 25.004,
            24.994, 25.006, 24.998, 25.002, 24.993, 25.007, 24.995, 25.005, 25.005, 24.995, 25.003,
            24.993, 25.007, 24.996, 25.006, 24.994, 25.004, 24.997, 24.993, 25.006, 24.997, 25.005,
            24.994, 25.007, 24.996, 25.004, 24.998, 25.0, 25.009, 24.999, 25.01, 24.998, 25.008,
            25.002, 24.997, 25.006, 25.011, 25.0, 25.011, 25.0, 24.998, 25.006, 24.999, 25.009,
            25.002, 24.997, 25.008, 25.01, 24.997, 25.01, 25.002, 24.999, 25.008, 25.011, 24.998,
            25.009, 25.0, 25.006, 25.006, 24.999, 25.011, 25.002, 24.998, 25.01, 25.009, 24.997,
            25.0, 25.008, 25.008, 25.01, 24.997, 25.006, 25.011, 24.999, 25.009, 25.002, 24.998,
            25.0, 25.0, 24.997, 25.009, 25.011, 25.006, 24.998, 25.01, 24.999, 25.008, 25.002,
            25.002, 25.008, 25.0, 24.999, 25.01, 25.006, 24.997, 25.011, 24.998, 25.009, 25.01,
            25.002, 25.009, 24.997, 25.001, 25.008, 24.999, 25.006, 25.011, 24.997, 24.999, 25.01,
            25.006, 24.998, 25.009, 25.001, 25.011, 25.008, 25.002, 24.996, 25.005, 24.994, 25.007,
            24.996, 24.993, 25.006, 24.998, 25.004, 24.995, 25.002, 24.996, 25.003, 24.993, 25.007,
            25.004, 24.995, 25.006, 24.994, 25.002, 25.0, 25.007, 24.998, 25.004, 24.993, 25.005,
            24.996, 25.002, 24.994, 25.006, 24.995, 24.993, 25.006, 24.995, 25.004, 24.997, 25.007,
            24.994, 25.002, 24.996, 25.006, 25.004, 24.997, 25.006, 24.994, 25.002, 24.993, 25.005,
            24.996, 25.007, 24.996, 24.995, 25.005, 24.998, 25.003, 24.993, 25.007, 24.996, 25.004,
            24.994, 25.005, 24.996, 25.003, 24.998, 25.002, 24.997, 25.004, 25.0, 25.023, 25.025,
            25.021, 25.006, 24.994, 25.003, 24.997, 25.005, 24.998, 25.007, 24.995, 25.002, 24.993,
            24.994, 25.007, 24.997, 25.006, 24.993, 25.005, 24.996, 25.003, 24.999, 25.0, 25.003,
            24.996, 25.007, 24.994, 25.005, 24.997, 25.002, 24.993, 25.006, 24.997, 24.997, 25.004,
            24.993, 25.006, 24.998, 25.005, 24.995, 25.007, 24.994, 25.001,
        ];
        assert_eq!(data.len(), 250);

        let wide = boxcox_capability(&data, None, None, R).expect("positive data, no spec");
        assert!(wide.lambda_at_bound, "{wide:?}");
        assert_eq!(wide.lambda, R.0, "{wide:?}");
        assert!(wide.indices.is_none(), "no spec was given");

        // A spec does not change where λ came from, and the indices are then
        // computed at that same λ rather than at a point beside it.
        let with_spec =
            boxcox_capability(&data, Some(25.05), Some(24.95), R).expect("positive data");
        assert!(with_spec.lambda_at_bound, "{with_spec:?}");
        assert_eq!(with_spec.lambda, wide.lambda);
        assert!(
            indices(&with_spec).pp.is_some_and(f64::is_finite),
            "{with_spec:?}"
        );
    }

    /// Data exactly normal after a Box-Cox transform with `lambda0`: normal
    /// quantiles pushed through the inverse transform.
    fn normal_after_boxcox(lambda0: f64, n: usize) -> Vec<f64> {
        (1..=n)
            .map(|i| {
                let p = (i as f64 - 0.5) / n as f64;
                let z = 10.0 + 2.0 * u_numflow::special::inverse_normal_cdf(p);
                (lambda0 * z + 1.0).powf(1.0 / lambda0)
            })
            .collect()
    }

    #[test]
    fn boxcox_capability_skewed_data() {
        // Right-skewed exponential-like data
        let data: Vec<f64> = (1..=25).map(|i| (i as f64 * 0.2).exp()).collect();
        let result = boxcox_capability(&data, Some(150.0), Some(1.0), R).unwrap();
        assert!(result.lambda.abs() < 0.6, "lambda={}", result.lambda);
        assert!(!result.lambda_at_bound);
        assert!(indices(&result).pp.is_some() || indices(&result).ppk.is_some());
    }

    #[test]
    fn boxcox_capability_reports_no_short_term_indices() {
        // Cp/Cpk need a within-subgroup sigma; a flat vector has none. Filling
        // them from the overall sigma would make Cp == Pp for every input.
        let data: Vec<f64> = (1..=25).map(|i| (i as f64 * 0.2).exp()).collect();
        let r = boxcox_capability(&data, Some(150.0), Some(1.0), R).unwrap();
        let i = indices(&r);
        assert!(i.cp.is_none());
        assert!(i.cpk.is_none());
        assert!(i.cpu.is_none());
        assert!(i.cpl.is_none());
        assert!(i.pp.is_some(), "the long-term indices are reported");
    }

    #[test]
    fn boxcox_capability_says_when_the_range_cut_lambda_short() {
        // Likelihood peaks near lambda = 4: the default range finds it inside,
        // a [-2, 2] range stops at 2 and must say so.
        let data = normal_after_boxcox(4.0, 100);
        let wide = boxcox_capability(&data, Some(40.0), None, R).unwrap();
        assert!(!wide.lambda_at_bound, "lambda={}", wide.lambda);
        assert!((wide.lambda - 4.0).abs() < 0.5, "lambda={}", wide.lambda);

        let narrow = boxcox_capability(&data, Some(40.0), None, (-2.0, 2.0)).unwrap();
        assert!(narrow.lambda_at_bound);
        assert_eq!(narrow.lambda, 2.0);
        // The indices are still computed — at the bound — and they differ from
        // the ones at the data's own lambda.
        assert_ne!(indices(&narrow).ppk, indices(&wide).ppk);
    }

    #[test]
    fn boxcox_capability_without_limits_estimates_lambda_only() {
        let data: Vec<f64> = (1..=25).map(|i| (i as f64 * 0.2).exp()).collect();
        let bare = boxcox_capability(&data, None, None, R).unwrap();
        let with = boxcox_capability(&data, Some(150.0), Some(1.0), R).unwrap();
        assert!(bare.indices.is_none());
        assert_eq!(bare.lambda, with.lambda);
        assert_eq!(bare.lambda_at_bound, with.lambda_at_bound);
    }

    #[test]
    fn boxcox_capability_rejects_an_invalid_lambda_range() {
        let data: Vec<f64> = (1..=10).map(|i| i as f64).collect();
        for range in [(2.0, -2.0), (1.0, 1.0), (f64::NAN, 2.0)] {
            assert_eq!(
                boxcox_capability(&data, Some(20.0), None, range).unwrap_err(),
                NonNormalCapabilityError::InvalidLambdaRange,
                "{range:?}"
            );
        }
    }

    #[test]
    fn boxcox_capability_non_finite_error() {
        let data = vec![1.0, f64::NAN, 2.0, 3.0, 4.0];
        assert_eq!(
            boxcox_capability(&data, Some(10.0), None, R).unwrap_err(),
            NonNormalCapabilityError::NonFiniteData
        );
    }

    #[test]
    fn boxcox_capability_non_positive_error() {
        let data = vec![1.0, -1.0, 2.0, 3.0, 4.0, 5.0];
        assert!(boxcox_capability(&data, Some(10.0), None, R).is_err());
    }

    #[test]
    fn boxcox_capability_insufficient_data() {
        let data = vec![1.0, 2.0, 3.0]; // < 4 points
        assert!(boxcox_capability(&data, Some(10.0), None, R).is_err());
    }

    #[test]
    fn boxcox_capability_lambda_in_range() {
        let data: Vec<f64> = (1..=20).map(|i| i as f64).collect();
        let result = boxcox_capability(&data, Some(25.0), Some(0.5), R).unwrap();
        assert!(result.lambda >= R.0 && result.lambda <= R.1);
    }

    #[test]
    fn boxcox_capability_usl_only() {
        let data: Vec<f64> = (1..=20).map(|i| i as f64 * 0.5).collect();
        let result = boxcox_capability(&data, Some(20.0), None, R).unwrap();
        // With USL only: pp is None (needs both limits), ppk should be Some
        assert!(indices(&result).ppk.is_some());
        assert!(indices(&result).pp.is_none());
    }

    #[test]
    fn boxcox_capability_lsl_only() {
        let data: Vec<f64> = (1..=20).map(|i| i as f64).collect();
        let result = boxcox_capability(&data, None, Some(0.5), R).unwrap();
        assert!(indices(&result).ppk.is_some());
        assert!(indices(&result).pp.is_none());
    }

    #[test]
    fn boxcox_capability_two_sided() {
        // Both limits present → the long-term pair is reported. The short-term
        // pair is not: this assertion used to require `cp`/`cpk` to be `Some`,
        // which pinned the very behaviour that made Cp equal Pp for every
        // input. See `boxcox_capability_reports_no_short_term_indices`.
        let data: Vec<f64> = (1..=30).map(|i| (i as f64 * 0.1).exp()).collect();
        let result = boxcox_capability(&data, Some(20.0), Some(1.0), R).unwrap();
        let i = indices(&result);
        assert!(i.pp.is_some());
        assert!(i.ppk.is_some());
        assert!(i.cp.is_none());
        assert!(i.cpk.is_none());
    }

    #[test]
    fn boxcox_capability_non_positive_spec_error() {
        let data: Vec<f64> = (1..=10).map(|i| i as f64).collect();
        // LSL = -1 is non-positive → SpecTransformError
        assert!(boxcox_capability(&data, Some(20.0), Some(-1.0), R).is_err());
    }

    #[test]
    fn boxcox_capability_result_has_valid_lambda() {
        let data: Vec<f64> = (1..=15).map(|i| (i as f64).powi(2)).collect();
        let result = boxcox_capability(&data, Some(250.0), Some(0.5), R).unwrap();
        assert!(result.lambda.is_finite());
        assert!(result.lambda >= R.0 && result.lambda <= R.1);
    }
}
