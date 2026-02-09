//! Sigma quality level conversions.
//!
//! Converts between parts-per-million (PPM) defect rates and sigma quality
//! levels using the standard 1.5-sigma shift convention.
//!
//! # Convention
//!
//! The Motorola Six Sigma methodology assumes a 1.5-sigma long-term shift
//! from the process mean. Under this convention:
//!
//! | Sigma | PPM (defects per million) |
//! |-------|--------------------------|
//! | 6.0   | 3.4                      |
//! | 5.0   | 233                      |
//! | 4.0   | 6,210                    |
//! | 3.0   | 66,807                   |
//! | 2.0   | 308,538                  |
//!
//! # References
//!
//! - Harry & Schroeder (2000), *Six Sigma: The Breakthrough Management
//!   Strategy Revolutionizing the World's Top Corporations*.
//! - Motorola University, "The Six Sigma Process" (defining the 1.5-sigma
//!   shift convention).

use u_optim::special::{inverse_normal_cdf, standard_normal_cdf};

/// Converts a sigma quality level to parts-per-million (PPM) defect rate.
///
/// Uses the standard 1.5-sigma shift convention:
///
/// ```text
/// PPM = 1,000,000 * (1 - Phi(sigma - 1.5))
/// ```
///
/// where Phi is the standard normal CDF.
///
/// # Arguments
///
/// * `sigma` - The sigma quality level (e.g., 6.0 for Six Sigma)
///
/// # Examples
///
/// ```
/// use u_analytics::capability::sigma_to_ppm;
///
/// // Six Sigma => ~3.4 PPM
/// let ppm = sigma_to_ppm(6.0);
/// assert!((ppm - 3.4).abs() < 1.0);
///
/// // Three Sigma => ~66,807 PPM
/// let ppm = sigma_to_ppm(3.0);
/// assert!((ppm - 66_807.0).abs() < 500.0);
/// ```
///
/// # Reference
///
/// Motorola Six Sigma convention (Harry & Schroeder, 2000).
pub fn sigma_to_ppm(sigma: f64) -> f64 {
    1_000_000.0 * (1.0 - standard_normal_cdf(sigma - 1.5))
}

/// Converts a parts-per-million (PPM) defect rate to a sigma quality level.
///
/// Inverse of [`sigma_to_ppm`]:
///
/// ```text
/// sigma = Phi_inv(1 - PPM / 1,000,000) + 1.5
/// ```
///
/// where Phi_inv is the inverse standard normal CDF.
///
/// # Arguments
///
/// * `ppm` - Defects per million opportunities. Must be in `(0, 1_000_000)`.
///
/// # Returns
///
/// `None` if `ppm` is outside the valid range `(0, 1_000_000)` (exclusive),
/// or if `ppm` is NaN.
///
/// # Examples
///
/// ```
/// use u_analytics::capability::ppm_to_sigma;
///
/// // ~3.4 PPM => Six Sigma
/// let sigma = ppm_to_sigma(3.4).unwrap();
/// assert!((sigma - 6.0).abs() < 0.1);
///
/// // ~66,807 PPM => Three Sigma
/// let sigma = ppm_to_sigma(66_807.0).unwrap();
/// assert!((sigma - 3.0).abs() < 0.1);
/// ```
///
/// # Reference
///
/// Motorola Six Sigma convention (Harry & Schroeder, 2000).
pub fn ppm_to_sigma(ppm: f64) -> Option<f64> {
    if ppm.is_nan() || ppm <= 0.0 || ppm >= 1_000_000.0 {
        return None;
    }
    let p = 1.0 - ppm / 1_000_000.0;
    let z = inverse_normal_cdf(p);
    if z.is_finite() {
        Some(z + 1.5)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // sigma_to_ppm -- known values
    // -----------------------------------------------------------------------

    /// Six Sigma: 3.4 PPM
    #[test]
    fn six_sigma_ppm() {
        let ppm = sigma_to_ppm(6.0);
        assert!(
            (ppm - 3.4).abs() < 1.0,
            "6-sigma should be ~3.4 PPM, got {ppm}"
        );
    }

    /// Five Sigma: 233 PPM
    #[test]
    fn five_sigma_ppm() {
        let ppm = sigma_to_ppm(5.0);
        assert!(
            (ppm - 233.0).abs() < 20.0,
            "5-sigma should be ~233 PPM, got {ppm}"
        );
    }

    /// Four Sigma: 6,210 PPM
    #[test]
    fn four_sigma_ppm() {
        let ppm = sigma_to_ppm(4.0);
        assert!(
            (ppm - 6_210.0).abs() < 200.0,
            "4-sigma should be ~6,210 PPM, got {ppm}"
        );
    }

    /// Three Sigma: 66,807 PPM
    #[test]
    fn three_sigma_ppm() {
        let ppm = sigma_to_ppm(3.0);
        assert!(
            (ppm - 66_807.0).abs() < 500.0,
            "3-sigma should be ~66,807 PPM, got {ppm}"
        );
    }

    /// Two Sigma: 308,538 PPM
    #[test]
    fn two_sigma_ppm() {
        let ppm = sigma_to_ppm(2.0);
        assert!(
            (ppm - 308_538.0).abs() < 3_000.0,
            "2-sigma should be ~308,538 PPM, got {ppm}"
        );
    }

    // -----------------------------------------------------------------------
    // sigma_to_ppm -- properties
    // -----------------------------------------------------------------------

    /// Higher sigma = fewer defects (monotonically decreasing).
    #[test]
    fn sigma_to_ppm_is_monotonically_decreasing() {
        let sigmas = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let ppms: Vec<f64> = sigmas.iter().map(|&s| sigma_to_ppm(s)).collect();
        for window in ppms.windows(2) {
            assert!(
                window[0] > window[1],
                "PPM should decrease with higher sigma: {} > {}",
                window[0],
                window[1]
            );
        }
    }

    /// PPM is always non-negative.
    #[test]
    fn sigma_to_ppm_non_negative() {
        for sigma in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0] {
            let ppm = sigma_to_ppm(sigma);
            assert!(ppm >= 0.0, "PPM must be >= 0 for sigma={sigma}, got {ppm}");
        }
    }

    /// PPM at sigma = 0 should be close to 933,193.
    #[test]
    fn sigma_zero_ppm() {
        let ppm = sigma_to_ppm(0.0);
        // Phi(-1.5) ~ 0.0668, so PPM ~ 933,193
        assert!(
            ppm > 900_000.0 && ppm < 950_000.0,
            "0-sigma PPM should be ~933,193, got {ppm}"
        );
    }

    // -----------------------------------------------------------------------
    // ppm_to_sigma -- roundtrip
    // -----------------------------------------------------------------------

    /// Roundtrip: sigma_to_ppm -> ppm_to_sigma should recover original sigma.
    #[test]
    fn roundtrip_sigma_ppm_sigma() {
        for &sigma in &[2.0, 3.0, 4.0, 5.0, 6.0] {
            let ppm = sigma_to_ppm(sigma);
            let recovered = ppm_to_sigma(ppm).expect("roundtrip should succeed");
            assert!(
                (recovered - sigma).abs() < 0.1,
                "roundtrip failed: sigma={sigma}, ppm={ppm}, recovered={recovered}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // ppm_to_sigma -- edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn ppm_to_sigma_rejects_zero() {
        assert!(ppm_to_sigma(0.0).is_none());
    }

    #[test]
    fn ppm_to_sigma_rejects_million() {
        assert!(ppm_to_sigma(1_000_000.0).is_none());
    }

    #[test]
    fn ppm_to_sigma_rejects_negative() {
        assert!(ppm_to_sigma(-1.0).is_none());
    }

    #[test]
    fn ppm_to_sigma_rejects_nan() {
        assert!(ppm_to_sigma(f64::NAN).is_none());
    }

    #[test]
    fn ppm_to_sigma_rejects_above_million() {
        assert!(ppm_to_sigma(1_500_000.0).is_none());
    }

    // -----------------------------------------------------------------------
    // ppm_to_sigma -- known values
    // -----------------------------------------------------------------------

    #[test]
    fn ppm_to_sigma_known_values() {
        let cases: &[(f64, f64)] = &[
            (3.4, 6.0),
            (233.0, 5.0),
            (6_210.0, 4.0),
            (66_807.0, 3.0),
            (308_538.0, 2.0),
        ];
        for &(ppm, expected_sigma) in cases {
            let sigma = ppm_to_sigma(ppm).expect("valid PPM should return Some");
            assert!(
                (sigma - expected_sigma).abs() < 0.15,
                "PPM={ppm}: expected sigma~{expected_sigma}, got {sigma}"
            );
        }
    }
}
