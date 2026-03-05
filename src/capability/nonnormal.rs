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
//! 1. Estimate the optimal λ via maximum likelihood (`estimate_lambda`).
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

// ── Error type ────────────────────────────────────────────────────────────────

/// Errors that can arise from non-normal process capability analysis.
#[derive(Debug, Clone, PartialEq)]
pub enum NonNormalCapabilityError {
    /// All data values must be strictly positive for Box-Cox transformation.
    NonPositiveData,
    /// At least 4 data points are required for reliable capability analysis.
    InsufficientData,
    /// Failed to transform a specification limit (e.g., limit is not positive).
    SpecTransformError,
    /// Capability computation failed (e.g., all transformed data are identical).
    CapabilityError,
    /// At least one specification limit (USL or LSL) must be provided.
    NoSpecLimits,
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
            NonNormalCapabilityError::NoSpecLimits => {
                write!(
                    f,
                    "at least one specification limit (USL or LSL) must be provided"
                )
            }
        }
    }
}

impl std::error::Error for NonNormalCapabilityError {}

impl From<TransformError> for NonNormalCapabilityError {
    fn from(e: TransformError) -> Self {
        match e {
            TransformError::NonPositiveData => NonNormalCapabilityError::NonPositiveData,
            TransformError::InsufficientData => NonNormalCapabilityError::InsufficientData,
            TransformError::InvalidInverse => NonNormalCapabilityError::SpecTransformError,
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
    /// Capability indices computed on the Box-Cox-transformed scale.
    pub indices: CapabilityIndices,
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Compute process capability indices for non-normal data via Box-Cox transformation.
///
/// The data are first transformed to approximate normality using the optimal
/// Box-Cox parameter λ (estimated via maximum likelihood over `[-2, 2]`).
/// Specification limits are transformed using the same λ. Standard capability
/// indices (Cp, Cpk, Pp, Ppk) are then computed on the transformed scale.
///
/// # Arguments
///
/// * `data` — Process observations. All values must be strictly positive.
/// * `usl` — Upper specification limit (optional). Must be positive if provided.
/// * `lsl` — Lower specification limit (optional). Must be positive if provided.
///
/// # Errors
///
/// Returns [`NonNormalCapabilityError`] if:
/// - Fewer than 4 data points are provided.
/// - Any data value is ≤ 0 (Box-Cox requires strictly positive data).
/// - Neither `usl` nor `lsl` is provided.
/// - A specification limit is ≤ 0 (cannot be Box-Cox transformed).
/// - Capability computation fails (e.g., zero variance in transformed data).
///
/// # Examples
///
/// ```
/// use u_analytics::capability::boxcox_capability;
///
/// // Right-skewed data
/// let data: Vec<f64> = (1..=20).map(|i| (i as f64 * 0.3_f64).exp()).collect();
/// let result = boxcox_capability(&data, Some(100.0), Some(1.0)).unwrap();
/// assert!(result.lambda >= -2.0 && result.lambda <= 2.0);
/// assert!(result.indices.ppk.is_some());
/// ```
pub fn boxcox_capability(
    data: &[f64],
    usl: Option<f64>,
    lsl: Option<f64>,
) -> Result<NonNormalCapabilityResult, NonNormalCapabilityError> {
    // Validate: at least one spec limit
    if usl.is_none() && lsl.is_none() {
        return Err(NonNormalCapabilityError::NoSpecLimits);
    }

    // Validate: sufficient data
    if data.len() < 4 {
        return Err(NonNormalCapabilityError::InsufficientData);
    }

    // Validate: all data must be strictly positive
    if data.iter().any(|&v| v <= 0.0) {
        return Err(NonNormalCapabilityError::NonPositiveData);
    }

    // Estimate optimal λ
    let lambda = estimate_lambda(data, -2.0, 2.0)?;

    // Transform data
    let y_t = box_cox(data, lambda)?;

    // Transform spec limits (box_cox requires len >= 2, use a dummy second element)
    let usl_t = usl
        .map(|u| {
            if u <= 0.0 {
                return Err(NonNormalCapabilityError::SpecTransformError);
            }
            // pair with data[0] (positive) so box_cox gets len=2
            let pair = [u, data[0]];
            box_cox(&pair, lambda)
                .map(|v| v[0])
                .map_err(|_| NonNormalCapabilityError::SpecTransformError)
        })
        .transpose()?;

    let lsl_t = lsl
        .map(|l| {
            if l <= 0.0 {
                return Err(NonNormalCapabilityError::SpecTransformError);
            }
            let pair = [l, data[0]];
            box_cox(&pair, lambda)
                .map(|v| v[0])
                .map_err(|_| NonNormalCapabilityError::SpecTransformError)
        })
        .transpose()?;

    // Compute overall std of transformed data (for Pp/Ppk)
    let n = y_t.len();
    let mean_t = y_t.iter().sum::<f64>() / n as f64;
    let overall_std_t =
        (y_t.iter().map(|&v| (v - mean_t).powi(2)).sum::<f64>() / (n - 1) as f64).sqrt();

    // Build ProcessCapability on transformed scale
    // ProcessCapability::new validates usl > lsl when both present
    let spec = ProcessCapability::new(usl_t, lsl_t)
        .map_err(|_| NonNormalCapabilityError::CapabilityError)?;

    // compute() uses overall_std_t as sigma_within (no rational subgrouping)
    let indices = spec
        .compute(&y_t, overall_std_t)
        .ok_or(NonNormalCapabilityError::CapabilityError)?;

    Ok(NonNormalCapabilityResult { lambda, indices })
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boxcox_capability_skewed_data() {
        // Right-skewed exponential-like data
        let data: Vec<f64> = (1..=25).map(|i| (i as f64 * 0.2).exp()).collect();
        let result = boxcox_capability(&data, Some(150.0), Some(1.0)).unwrap();
        assert!(result.lambda.abs() < 0.6, "lambda={}", result.lambda);
        assert!(result.indices.pp.is_some() || result.indices.ppk.is_some());
    }

    #[test]
    fn boxcox_capability_non_positive_error() {
        let data = vec![1.0, -1.0, 2.0, 3.0, 4.0, 5.0];
        assert!(boxcox_capability(&data, Some(10.0), None).is_err());
    }

    #[test]
    fn boxcox_capability_insufficient_data() {
        let data = vec![1.0, 2.0, 3.0]; // < 4 points
        assert!(boxcox_capability(&data, Some(10.0), None).is_err());
    }

    #[test]
    fn boxcox_capability_lambda_in_range() {
        let data: Vec<f64> = (1..=20).map(|i| i as f64).collect();
        let result = boxcox_capability(&data, Some(25.0), Some(0.5)).unwrap();
        assert!(result.lambda >= -2.0 && result.lambda <= 2.0);
    }

    #[test]
    fn boxcox_capability_no_spec_error() {
        let data: Vec<f64> = (1..=10).map(|i| i as f64).collect();
        assert!(boxcox_capability(&data, None, None).is_err());
    }

    #[test]
    fn boxcox_capability_usl_only() {
        let data: Vec<f64> = (1..=20).map(|i| i as f64 * 0.5).collect();
        let result = boxcox_capability(&data, Some(20.0), None).unwrap();
        // With USL only: pp is None (needs both limits), ppk should be Some
        assert!(result.indices.ppk.is_some());
        assert!(result.indices.pp.is_none());
    }

    #[test]
    fn boxcox_capability_lsl_only() {
        let data: Vec<f64> = (1..=20).map(|i| i as f64).collect();
        let result = boxcox_capability(&data, None, Some(0.5)).unwrap();
        assert!(result.indices.ppk.is_some());
        assert!(result.indices.pp.is_none());
    }

    #[test]
    fn boxcox_capability_two_sided() {
        // Both USL and LSL provided → pp, ppk, cp, cpk all Some
        let data: Vec<f64> = (1..=30).map(|i| (i as f64 * 0.1).exp()).collect();
        let result = boxcox_capability(&data, Some(20.0), Some(1.0)).unwrap();
        assert!(result.indices.pp.is_some());
        assert!(result.indices.ppk.is_some());
        assert!(result.indices.cp.is_some());
        assert!(result.indices.cpk.is_some());
    }

    #[test]
    fn boxcox_capability_non_positive_spec_error() {
        let data: Vec<f64> = (1..=10).map(|i| i as f64).collect();
        // LSL = -1 is non-positive → SpecTransformError
        assert!(boxcox_capability(&data, Some(20.0), Some(-1.0)).is_err());
    }

    #[test]
    fn boxcox_capability_result_has_valid_lambda() {
        let data: Vec<f64> = (1..=15).map(|i| (i as f64).powi(2)).collect();
        let result = boxcox_capability(&data, Some(250.0), Some(0.5)).unwrap();
        assert!(result.lambda.is_finite());
        assert!(result.lambda >= -2.0 && result.lambda <= 2.0);
    }
}
