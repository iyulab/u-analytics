//! Type 1 gage study: the capability of a measurement device from repeated
//! measurements of one reference part.
//!
//! One operator measures a single part, whose true value `x_ref` is known,
//! `n` times (50 is the usual recommendation, 25 the least). The spread of the
//! readings is the device's repeatability, their mean against `x_ref` its bias,
//! and both are set against the tolerance `T = USL − LSL`.
//!
//! With `x̄` the mean and `s` the sample standard deviation (divisor `n − 1`),
//! a fraction `K` of the tolerance and `L` standard deviations as the width of
//! the measurement spread (the *study variation* `SV = L·s`):
//!
//! ```text
//! bias  = x̄ − x_ref
//! Cg    = (K·T) / SV                     = (K·T) / (L·s)
//! Cgk   = (K·T/2 − |bias|) / (SV / 2)    = (K·T/2 − |bias|) / (L·s/2)
//!       = Cg − |bias| / (L·s/2)
//! t     = bias / (s / √n),  df = n − 1,  p = P(|T| ≥ |t|)      (H₀: bias = 0)
//! %Var(Repeatability)          = 100·SV / T        = 100·K / Cg
//! %Var(Repeatability and Bias) = 100·K / Cgk
//! ```
//!
//! With the common defaults `K = 20 %` and `L = 6` these are the familiar
//! `Cg = 0.2·T / 6s` and `Cgk = (0.1·T − |x̄ − x_ref|) / 3s`. A value of 1.33 is
//! the customary benchmark for both. `Cgk ≤ Cg`, with equality only for zero
//! bias; `Cgk` is zero or negative when the bias alone uses up the band
//! `K·T/2`.
//!
//! # Editions differ — so the multipliers are parameters
//!
//! The 20 % and the 6 are conventions, not constants of nature: Minitab lets
//! `K` be anything from 0 to 100 % ("10 to 20 are common") and `L` be changed
//! (its example is 5.15, for 99 % coverage); other schemes quote 15 % or a
//! different spread. [`Type1Options`] holds both, with the Minitab defaults,
//! so a different edition is a configuration. Which values a given standard
//! (for instance an edition of VDA 5) prescribes is not asserted here; the
//! formulas are those of the sources below, and VDA 5's measurement-uncertainty
//! statistics (`u_MS`, `%QMS`) are a different analysis that is not implemented.
//!
//! `L` is one number for both indices: the full study variation is `L·s` and
//! `Cgk` uses half of it, as Minitab's `SV / 2` does (Minitab's help phrases
//! the same thing as "3 standard deviations for half the spread").
//!
//! The t statistic is signed (`bias / (s/√n)`); Minitab prints it as
//! `√n·|x̄ − x_m|/s`. The p-value is two-sided and the same either way.
//!
//! # Examples
//!
//! ```
//! use u_analytics::msa::type1::{type1_gage_study, Type1Options};
//!
//! // Ten readings of a 12.306 reference with limits 12.285 and 12.335.
//! let readings = [
//!     12.3075, 12.3018, 12.3116, 12.3074, 12.3077, 12.3127, 12.3035, 12.2985, 12.3059, 12.3127,
//! ];
//! let r = type1_gage_study(&readings, 12.306, 12.285, 12.335, &Type1Options::default()).unwrap();
//! assert!((r.bias - 0.00093).abs() < 1e-12);
//! assert!((r.cg - 0.35378).abs() < 1e-5);
//! assert!((r.cgk - 0.28798).abs() < 1e-5);
//! assert!(r.cgk < r.cg);
//! assert!(r.cg < 1.33); // not capable
//! ```
//!
//! # References
//!
//! - Minitab, LLC. "Methods and formulas for Type 1 Gage Study" (Minitab
//!   Statistical Software support): `Cg`, `CgK`, bias, the t statistic with
//!   `n − 1` degrees of freedom, `%Var (Repeatability)` and
//!   `%Var (Repeatability and Bias)`, with `K` (default 20) and the study
//!   variation multiple `k₁` (default 6). <https://support.minitab.com/en-us/minitab/help-and-how-to/quality-and-process-improvement/measurement-system-analysis/how-to/gage-study/type-1-gage-study/methods-and-formulas/methods-and-formulas/>
//! - BPI Consulting, LLC (2024). "Type 1 Gage Study" (SPC for Excel): the same
//!   formulas in text form, `Cgk = ((P/200)·Tolerance − |Bias|) / (SV/2)`,
//!   `%Var = P/Cg` and `P/Cgk`, and the worked example reproduced by this
//!   crate's reference tests (reference 12.306, 50 readings, `Cg` 0.343, `Cgk`
//!   0.240, `t` 2.185, `p` 0.0337). <https://www.spcforexcel.com/knowledge/measurement-systems-analysis-gage-rr/type-1-gage-study/>
//! - SYMESTIC GmbH. "Messsystemanalyse (MSA): Verfahren und Beispiel": the
//!   default-parameter forms `Cg = 0.2·T / (6·s)` and
//!   `Cgk = (0.1·T − |x̄ − x_m|) / (3·s)` and the 1.33 requirement of Bosch
//!   booklet 10. <https://www.symestic.com/de-de/was-ist/messsystemanalyse>
//! - JMP Statistical Discovery. "The Type 1 Gauge Study Report":
//!   `%Var = p/Cg` and `p/Cgk`. <https://www.jmp.com/support/help/en/19.0/jmp/the-type-1-gauge-study-report.shtml>

use u_numflow::special;
use u_numflow::stats;

/// The two conventions of a Type 1 study, both of which differ between
/// editions and tools. The default is Minitab's: 20 % of the tolerance and a
/// spread of 6 standard deviations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Type1Options {
    /// `K` as a **fraction** of the tolerance (`0.2` for 20 %): the band the
    /// measurement spread is allowed to occupy. In `(0, 1]` — Minitab accepts
    /// 0 to 100 %.
    pub tolerance_fraction: f64,
    /// `L`: the number of standard deviations that make the full measurement
    /// spread (`6` covers 99.73 %, `5.15` 99 %, `4` about 95 %). Positive.
    /// `Cgk` uses half of it.
    pub sigma_multiple: f64,
}

impl Default for Type1Options {
    fn default() -> Self {
        Self {
            tolerance_fraction: 0.2,
            sigma_multiple: 6.0,
        }
    }
}

/// The outcome of a Type 1 gage study.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Type1GageResult {
    /// Number of measurements `n`.
    pub n: usize,
    /// Mean of the measurements `x̄`.
    pub mean: f64,
    /// Sample standard deviation `s` (divisor `n − 1`).
    pub std_dev: f64,
    /// `bias = x̄ − x_ref`.
    pub bias: f64,
    /// The study variation `SV = L·s`.
    pub study_variation: f64,
    /// Repeatability against the tolerance: `K·T / SV`.
    pub cg: f64,
    /// Repeatability and bias against the tolerance:
    /// `(K·T/2 − |bias|) / (SV/2)`. Zero or negative when `|bias| ≥ K·T/2`.
    pub cgk: f64,
    /// `%Var (Repeatability) = 100·SV / T`, which is `100·K / Cg`.
    pub percent_var_repeatability: f64,
    /// `%Var (Repeatability and Bias) = 100·K / Cgk`. `None` when `Cgk ≤ 0`
    /// (the bias alone fills the band, so no percentage of it is defined).
    pub percent_var_repeatability_and_bias: Option<f64>,
    /// The statistic of the one-sample t test of `bias = 0`:
    /// `bias / (s/√n)`. Signed.
    pub t_statistic: f64,
    /// Degrees of freedom of that test, `n − 1`.
    pub df: f64,
    /// Two-sided p-value of that test, `P(|T| ≥ |t|)`.
    pub p_value: f64,
}

/// Type 1 gage study of `measurements` of a reference part of known value
/// `reference`, against the specification limits `lsl` and `usl`.
///
/// # Algorithm
///
/// See the [module documentation](self). The p-value is formed from the tail
/// of the t distribution directly (`2·P(T ≥ |t|)`), so one far below `1e-16`
/// keeps its digits.
///
/// # Returns
///
/// `None` if there are fewer than 2 measurements, a measurement, the reference
/// or a limit is not finite, `usl ≤ lsl`, the measurements are all equal (the
/// spread is zero, so neither index is defined), `tolerance_fraction` is not
/// in `(0, 1]`, `sigma_multiple` is not positive and finite, or a result is
/// not a finite number.
pub fn type1_gage_study(
    measurements: &[f64],
    reference: f64,
    lsl: f64,
    usl: f64,
    options: &Type1Options,
) -> Option<Type1GageResult> {
    let Type1Options {
        tolerance_fraction,
        sigma_multiple,
    } = *options;
    if measurements.len() < 2
        || measurements.iter().any(|v| !v.is_finite())
        || !(reference.is_finite() && lsl.is_finite() && usl.is_finite())
        || usl <= lsl
        || !(tolerance_fraction > 0.0 && tolerance_fraction <= 1.0)
        || !(sigma_multiple > 0.0 && sigma_multiple.is_finite())
    {
        return None;
    }
    // Exactly equal readings have no spread; the standard deviation of such a
    // sample can come out as rounding noise rather than 0, so it is not asked.
    if measurements.iter().all(|&v| v == measurements[0]) {
        return None;
    }
    let n = measurements.len();
    let nf = n as f64;
    let mean = stats::mean(measurements)?;
    let std_dev = stats::std_dev(measurements)?;
    if std_dev.is_nan() || std_dev <= 0.0 {
        return None;
    }

    let tolerance = usl - lsl;
    let study_variation = sigma_multiple * std_dev;
    let bias = mean - reference;
    let cg = tolerance_fraction * tolerance / study_variation;
    let cgk = (tolerance_fraction * tolerance / 2.0 - bias.abs()) / (study_variation / 2.0);
    let percent_var_repeatability = 100.0 * study_variation / tolerance;
    let percent_var_repeatability_and_bias = (cgk > 0.0).then(|| 100.0 * tolerance_fraction / cgk);

    let df = nf - 1.0;
    let t_statistic = bias / (std_dev / nf.sqrt());
    // `t_distribution_sf` of the magnitude is the upper tail; doubled it is the
    // two-sided p, capped at 1 against rounding at t = 0.
    let p_value = (2.0 * special::t_distribution_sf(t_statistic.abs(), df)).min(1.0);

    let result = Type1GageResult {
        n,
        mean,
        std_dev,
        bias,
        study_variation,
        cg,
        cgk,
        percent_var_repeatability,
        percent_var_repeatability_and_bias,
        t_statistic,
        df,
        p_value,
    };
    let finite = [
        mean,
        std_dev,
        bias,
        study_variation,
        cg,
        cgk,
        percent_var_repeatability,
        t_statistic,
        p_value,
    ]
    .iter()
    .all(|v| v.is_finite())
        && percent_var_repeatability_and_bias.is_none_or(|v| v.is_finite());
    finite.then_some(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    const READINGS: [f64; 8] = [10.02, 9.98, 10.05, 10.01, 9.97, 10.03, 10.0, 10.04];

    #[test]
    fn the_definitions_hold_for_a_small_study() {
        let r = type1_gage_study(&READINGS, 10.0, 9.0, 11.0, &Type1Options::default()).unwrap();
        let mean = READINGS.iter().sum::<f64>() / 8.0;
        let s = (READINGS.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / 7.0).sqrt();
        assert_eq!(r.n, 8);
        assert!((r.mean - mean).abs() < 1e-12 && (r.std_dev - s).abs() < 1e-12);
        assert!((r.bias - (mean - 10.0)).abs() < 1e-12);
        assert!((r.cg - 0.4 / (6.0 * s)).abs() < 1e-9);
        assert!((r.cgk - (0.2 - (mean - 10.0).abs()) / (3.0 * s)).abs() < 1e-9);
        assert!((r.percent_var_repeatability - 100.0 * 6.0 * s / 2.0).abs() < 1e-9);
        assert!((r.percent_var_repeatability - 20.0 / r.cg).abs() < 1e-9);
        let both = r.percent_var_repeatability_and_bias.unwrap();
        assert!((both - 20.0 / r.cgk).abs() < 1e-9);
        assert_eq!(r.df, 7.0);
    }

    #[test]
    fn cgk_is_cg_less_the_bias_term_and_never_above_it() {
        let o = Type1Options {
            tolerance_fraction: 0.15,
            sigma_multiple: 4.0,
        };
        let r = type1_gage_study(&READINGS, 10.03, 9.0, 11.0, &o).unwrap();
        let expected = r.cg - r.bias.abs() / (4.0 * r.std_dev / 2.0);
        assert!((r.cgk - expected).abs() < 1e-12);
        assert!(r.cgk < r.cg);
    }

    #[test]
    fn a_bias_that_fills_the_band_leaves_no_repeatability_and_bias_percentage() {
        // |bias| well above K*T/2 = 0.2.
        let r = type1_gage_study(&READINGS, 9.5, 9.0, 11.0, &Type1Options::default()).unwrap();
        assert!(r.cgk < 0.0);
        assert!(r.percent_var_repeatability_and_bias.is_none());
        assert!(r.cg > 0.0);
    }

    #[test]
    fn the_bias_test_is_the_one_sample_t_test_against_the_reference() {
        let r = type1_gage_study(&READINGS, 10.1, 9.0, 11.0, &Type1Options::default()).unwrap();
        let t = crate::testing::one_sample_t_test(&READINGS, 10.1).unwrap();
        assert!((r.t_statistic - t.statistic).abs() < 1e-12 * t.statistic.abs());
        assert!((r.p_value - t.p_value).abs() < 1e-12);
        assert_eq!(r.df, t.df);
    }

    #[test]
    fn refusals() {
        let o = Type1Options::default();
        let go = |m: &[f64], r: f64, lo: f64, hi: f64, o: &Type1Options| {
            type1_gage_study(m, r, lo, hi, o)
        };
        assert!(go(&READINGS, 10.0, 9.0, 11.0, &o).is_some());
        assert!(go(&[10.0], 10.0, 9.0, 11.0, &o).is_none());
        assert!(go(&[], 10.0, 9.0, 11.0, &o).is_none());
        assert!(go(&[10.0, f64::NAN, 10.1], 10.0, 9.0, 11.0, &o).is_none());
        assert!(go(&[10.0, f64::INFINITY], 10.0, 9.0, 11.0, &o).is_none());
        assert!(go(&READINGS, f64::NAN, 9.0, 11.0, &o).is_none());
        assert!(go(&READINGS, 10.0, f64::NEG_INFINITY, 11.0, &o).is_none());
        assert!(go(&READINGS, 10.0, 9.0, f64::NAN, &o).is_none());
        // Limits: equal or reversed.
        assert!(go(&READINGS, 10.0, 11.0, 11.0, &o).is_none());
        assert!(go(&READINGS, 10.0, 11.0, 9.0, &o).is_none());
        // No variation, including a constant that does not sum exactly.
        assert!(go(&[10.0, 10.0, 10.0], 10.0, 9.0, 11.0, &o).is_none());
        assert!(go(&[0.1, 0.1, 0.1], 0.1, 0.0, 1.0, &o).is_none());
        // Options.
        for fraction in [0.0, -0.2, 1.0000001, f64::NAN, f64::INFINITY] {
            let bad = Type1Options {
                tolerance_fraction: fraction,
                ..o
            };
            assert!(go(&READINGS, 10.0, 9.0, 11.0, &bad).is_none(), "{fraction}");
        }
        for multiple in [0.0, -6.0, f64::NAN, f64::INFINITY] {
            let bad = Type1Options {
                sigma_multiple: multiple,
                ..o
            };
            assert!(go(&READINGS, 10.0, 9.0, 11.0, &bad).is_none(), "{multiple}");
        }
        // A whole tolerance is allowed.
        let whole = Type1Options {
            tolerance_fraction: 1.0,
            ..o
        };
        assert!(go(&READINGS, 10.0, 9.0, 11.0, &whole).is_some());
    }

    #[test]
    fn zero_bias_is_t_zero_p_one() {
        let r = type1_gage_study(&[9.0, 11.0], 10.0, 0.0, 20.0, &Type1Options::default()).unwrap();
        assert_eq!((r.bias, r.t_statistic, r.p_value), (0.0, 0.0, 1.0));
        assert!((r.cg - r.cgk).abs() < 1e-12);
    }
}
