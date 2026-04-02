//! Percentile-based process capability (ISO 22514-2).
//!
//! When process data is non-normal, traditional capability indices based on
//! mean and standard deviation can be misleading. The percentile method uses
//! empirical quantiles (0.135th and 99.865th percentiles) as the natural
//! process spread — equivalent to the ±3σ spread for normal data.
//!
//! # Indices
//!
//! - **Cp\*** — (USL − LSL) / (X₉₉.₈₆₅ − X₀.₁₃₅)
//! - **Cpk\*** — min(Cpu\*, Cpl\*)
//! - **Cpu\*** — (USL − median) / (X₉₉.₈₆₅ − median)
//! - **Cpl\*** — (median − LSL) / (median − X₀.₁₃₅)
//!
//! # References
//!
//! - ISO 22514-2:2017. "Statistical methods in process management — Capability and performance",
//!   §4.3, percentile method.
//! - Clements, J.A. (1989). "Process capability calculations for non-normal distributions."

use u_numflow::stats;

/// Result of percentile-based process capability analysis.
#[derive(Debug, Clone)]
pub struct PercentileCapabilityResult {
    /// Cp* = (USL − LSL) / (X₉₉.₈₆₅ − X₀.₁₃₅). Requires both limits.
    pub cp_star: Option<f64>,
    /// Cpk* = min(Cpu*, Cpl*). Requires at least one limit.
    pub cpk_star: Option<f64>,
    /// Cpu* = (USL − median) / (X₉₉.₈₆₅ − median). Requires USL.
    pub cpu_star: Option<f64>,
    /// Cpl* = (median − LSL) / (median − X₀.₁₃₅). Requires LSL.
    pub cpl_star: Option<f64>,
    /// Sample median.
    pub median: f64,
    /// 0.135th percentile value (lower natural process limit).
    pub percentile_lower: f64,
    /// 99.865th percentile value (upper natural process limit).
    pub percentile_upper: f64,
}

/// Compute percentile-based capability indices (ISO 22514-2).
///
/// Uses empirical quantiles at 0.00135 and 0.99865 (corresponding to ±3σ
/// for normal distributions) to define the natural process spread.
///
/// # Arguments
///
/// * `data` — Process observations (at least 20 data points required).
/// * `lsl` — Lower specification limit (optional).
/// * `usl` — Upper specification limit (optional).
///
/// # Returns
///
/// `Err` if:
/// - Fewer than 20 data points (insufficient for extreme percentile estimation).
/// - Neither `usl` nor `lsl` is provided.
/// - Data contains non-finite values.
/// - The percentile spread is zero or negative (degenerate data).
///
/// # Examples
///
/// ```
/// use u_analytics::capability::percentile_capability;
///
/// let data: Vec<f64> = (0..100).map(|i| 10.0 + (i as f64) * 0.1).collect();
/// let result = percentile_capability(&data, Some(5.0), Some(15.0)).unwrap();
/// assert!(result.cp_star.is_some());
/// assert!(result.median > 0.0);
/// ```
pub fn percentile_capability(
    data: &[f64],
    lsl: Option<f64>,
    usl: Option<f64>,
) -> Result<PercentileCapabilityResult, &'static str> {
    // Validate inputs
    if usl.is_none() && lsl.is_none() {
        return Err("at least one specification limit (USL or LSL) is required");
    }
    if data.len() < 20 {
        return Err("at least 20 data points are required for percentile capability");
    }
    if data.iter().any(|v| !v.is_finite()) {
        return Err("all data values must be finite");
    }
    if let (Some(u), Some(l)) = (usl, lsl) {
        if u <= l {
            return Err("USL must be greater than LSL");
        }
    }

    // Compute percentiles using u-numflow (R-7 linear interpolation)
    let p_lower = stats::quantile(data, 0.00135).ok_or("failed to compute 0.135th percentile")?;
    let p_upper = stats::quantile(data, 0.99865).ok_or("failed to compute 99.865th percentile")?;
    let median = stats::quantile(data, 0.5).ok_or("failed to compute median")?;

    // Validate spread
    let spread = p_upper - p_lower;
    if spread < 1e-300 {
        return Err("percentile spread is zero — data has no variation");
    }

    let upper_spread = p_upper - median;
    let lower_spread = median - p_lower;

    // Cp* = (USL - LSL) / (p_upper - p_lower)
    let cp_star = match (usl, lsl) {
        (Some(u), Some(l)) => Some((u - l) / spread),
        _ => None,
    };

    // Cpu* = (USL - median) / (p_upper - median)
    let cpu_star = usl.and_then(|u| {
        if upper_spread > 1e-300 {
            Some((u - median) / upper_spread)
        } else {
            None
        }
    });

    // Cpl* = (median - LSL) / (median - p_lower)
    let cpl_star = lsl.and_then(|l| {
        if lower_spread > 1e-300 {
            Some((median - l) / lower_spread)
        } else {
            None
        }
    });

    // Cpk* = min of available Cpu*/Cpl*
    let cpk_star = match (cpu_star, cpl_star) {
        (Some(u), Some(l)) => Some(u.min(l)),
        (Some(u), None) => Some(u),
        (None, Some(l)) => Some(l),
        (None, None) => None,
    };

    Ok(PercentileCapabilityResult {
        cp_star,
        cpk_star,
        cpu_star,
        cpl_star,
        median,
        percentile_lower: p_lower,
        percentile_upper: p_upper,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Generate uniform-like data for testing.
    fn uniform_data(n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| 10.0 + (i as f64) / (n as f64 - 1.0) * 10.0)
            .collect()
    }

    #[test]
    fn basic_two_sided() {
        let data = uniform_data(100);
        let result = percentile_capability(&data, Some(5.0), Some(25.0)).unwrap();

        assert!(result.cp_star.is_some());
        assert!(result.cpk_star.is_some());
        assert!(result.cpu_star.is_some());
        assert!(result.cpl_star.is_some());
        assert!(result.cp_star.unwrap() > 0.0);

        // Median should be approximately the midpoint of the data
        assert!(
            (result.median - 15.0).abs() < 0.5,
            "median should be ~15, got {}",
            result.median
        );
    }

    #[test]
    fn usl_only() {
        let data = uniform_data(50);
        let result = percentile_capability(&data, None, Some(25.0)).unwrap();

        assert!(result.cp_star.is_none(), "Cp* requires both limits");
        assert!(result.cpu_star.is_some());
        assert!(result.cpl_star.is_none());
        assert!(result.cpk_star.is_some());
        assert_eq!(result.cpk_star.unwrap(), result.cpu_star.unwrap());
    }

    #[test]
    fn lsl_only() {
        let data = uniform_data(50);
        let result = percentile_capability(&data, Some(5.0), None).unwrap();

        assert!(result.cp_star.is_none());
        assert!(result.cpu_star.is_none());
        assert!(result.cpl_star.is_some());
        assert!(result.cpk_star.is_some());
        assert_eq!(result.cpk_star.unwrap(), result.cpl_star.unwrap());
    }

    #[test]
    fn rejects_insufficient_data() {
        let data: Vec<f64> = (0..19).map(|i| i as f64).collect();
        assert!(percentile_capability(&data, Some(0.0), Some(20.0)).is_err());
    }

    #[test]
    fn rejects_no_spec_limits() {
        let data = uniform_data(30);
        assert!(percentile_capability(&data, None, None).is_err());
    }

    #[test]
    fn rejects_non_finite_data() {
        let mut data = uniform_data(30);
        data[10] = f64::NAN;
        assert!(percentile_capability(&data, Some(0.0), Some(30.0)).is_err());
    }

    #[test]
    fn rejects_usl_leq_lsl() {
        let data = uniform_data(30);
        assert!(percentile_capability(&data, Some(10.0), Some(5.0)).is_err());
        assert!(percentile_capability(&data, Some(10.0), Some(10.0)).is_err());
    }

    #[test]
    fn cpk_star_equals_min_of_cpu_cpl() {
        let data = uniform_data(100);
        let result = percentile_capability(&data, Some(5.0), Some(25.0)).unwrap();

        let cpu = result.cpu_star.unwrap();
        let cpl = result.cpl_star.unwrap();
        let cpk = result.cpk_star.unwrap();
        assert!(
            (cpk - cpu.min(cpl)).abs() < 1e-10,
            "Cpk* should equal min(Cpu*, Cpl*)"
        );
    }

    #[test]
    fn percentiles_bracket_data() {
        let data = uniform_data(200);
        let result = percentile_capability(&data, Some(5.0), Some(25.0)).unwrap();

        // Lower percentile should be near but >= data minimum
        assert!(
            result.percentile_lower >= data.iter().copied().fold(f64::INFINITY, f64::min) - 1e-10,
            "lower percentile below data min"
        );
        // Upper percentile should be near but <= data maximum
        assert!(
            result.percentile_upper
                <= data.iter().copied().fold(f64::NEG_INFINITY, f64::max) + 1e-10,
            "upper percentile above data max"
        );
        // p_lower < median < p_upper
        assert!(result.percentile_lower < result.median);
        assert!(result.median < result.percentile_upper);
    }

    #[test]
    fn symmetric_data_gives_balanced_cpu_cpl() {
        // Symmetric data centered at 50
        let data: Vec<f64> = (0..200).map(|i| 40.0 + (i as f64) * 0.1).collect();
        let lsl = 30.0;
        let usl = 70.0;
        let result = percentile_capability(&data, Some(lsl), Some(usl)).unwrap();

        let cpu = result.cpu_star.unwrap();
        let cpl = result.cpl_star.unwrap();

        // For roughly symmetric data centered between limits, Cpu ≈ Cpl
        assert!(
            (cpu - cpl).abs() < cpu * 0.2,
            "symmetric data should give similar Cpu* ({}) and Cpl* ({})",
            cpu,
            cpl
        );
    }
}
