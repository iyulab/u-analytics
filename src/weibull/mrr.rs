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
/// Median ranks are computed using Benard's approximation:
///
/// ```text
/// F_i = (i - 0.3) / (n + 0.4)
/// ```
///
/// Reference: Benard & Bos-Levenbach (1953), "Het uitzetten van waarnemingen
/// op waarschijnlijkheidspaier", *Statistica Neerlandica*, 7(3), 163–173.
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
/// # References
/// - Benard & Bos-Levenbach (1953), *Statistica Neerlandica*, 7(3), 163–173.
/// - Abernethy, R.B. (2006). *The New Weibull Handbook*, 5th ed., §2.4.
///
/// # Complexity
/// O(n log n) for sorting; O(n) for regression.
pub fn weibull_mrr(failure_times: &[f64]) -> Option<WeibullMrrResult> {
    weibull_mrr_censored(failure_times, &[])
}

/// Median Rank Regression with right-censored units (suspensions) -- units
/// still running, or removed unfailed, at their time.
///
/// A suspension has no point on the probability plot, but it moves the ranks
/// of the failures after it: Johnson's adjusted rank
///
/// ```text
/// A_i = A_{i-1} + (N + 1 − A_{i-1}) / (1 + reverse rank)
/// ```
///
/// over all `N` units sorted by time (a failure before a suspension at the
/// same time), then Benard's median rank `(A_i − 0.3) / (N + 0.4)` and the
/// same regression as [`weibull_mrr`]. Without suspensions `A_i = i` and the
/// fit is [`weibull_mrr`]'s exactly. Dropping the suspensions instead treats
/// survivors as never having been at risk and pulls the scale (and every
/// B-life) low.
///
/// `None` with fewer than 2 failures, a non-positive or non-finite time, or a
/// degenerate regression.
///
/// # Examples
///
/// ```
/// use u_analytics::weibull::{weibull_mrr, weibull_mrr_censored};
/// let failures = [55.0, 187.0, 216.0, 240.0, 244.0, 335.0];
/// let censored = weibull_mrr_censored(&failures, &[361.0, 373.0, 375.0, 386.0]).unwrap();
/// let dropped = weibull_mrr(&failures).unwrap();
/// assert!(censored.scale > dropped.scale);
/// ```
///
/// # References
/// - Johnson, L.G. (1964). *Theory and Technique of Variation Research*, Elsevier.
/// - Abernethy, R.B. (2006). *The New Weibull Handbook*, 5th ed., §2.9 ("Suspensions").
pub fn weibull_mrr_censored(
    failure_times: &[f64],
    suspension_times: &[f64],
) -> Option<WeibullMrrResult> {
    if failure_times.len() < 2 {
        return None;
    }
    let valid = |t: &f64| t.is_finite() && *t > 0.0;
    if !failure_times.iter().all(valid) || !suspension_times.iter().all(valid) {
        return None;
    }

    // Every unit by time; a failure sorts before a suspension at the same time.
    let mut units: Vec<(f64, bool)> = failure_times
        .iter()
        .map(|&t| (t, true))
        .chain(suspension_times.iter().map(|&t| (t, false)))
        .collect();
    units.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .expect("NaN values already filtered above")
            .then(b.1.cmp(&a.1))
    });

    let total = units.len();
    let total_f = total as f64;
    let mut x_vals = Vec::with_capacity(failure_times.len());
    let mut y_vals = Vec::with_capacity(failure_times.len());
    let mut adjusted = 0.0_f64;
    for (i, &(t, failed)) in units.iter().enumerate() {
        if !failed {
            continue;
        }
        let reverse_rank = (total - i) as f64;
        adjusted += (total_f + 1.0 - adjusted) / (1.0 + reverse_rank);
        // Benard's approximation for median ranks
        let f_i = (adjusted - 0.3) / (total_f + 0.4);

        let x = t.ln();
        let y = (-(1.0 - f_i).ln()).ln(); // ln(-ln(1 - F_i))

        if !x.is_finite() || !y.is_finite() {
            return None;
        }

        x_vals.push(x);
        y_vals.push(y);
    }
    let n_f = x_vals.len() as f64;

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
    fn without_suspensions_it_is_the_plain_fit_bit_for_bit() {
        let data = [
            72.0, 82.0, 97.0, 103.0, 113.0, 117.0, 126.0, 127.0, 127.0, 139.0,
        ];
        let plain = weibull_mrr(&data).unwrap();
        let censored = weibull_mrr_censored(&data, &[]).unwrap();
        assert_eq!(
            (
                plain.shape.to_bits(),
                plain.scale.to_bits(),
                plain.r_squared.to_bits()
            ),
            (
                censored.shape.to_bits(),
                censored.scale.to_bits(),
                censored.r_squared.to_bits()
            )
        );
    }

    /// Johnson's ranks by hand for F10 S20 F30 F40 S50 (N = 5): 1, then
    /// 1 + (6 − 1)/(1 + 3) = 2.25, then 2.25 + (6 − 2.25)/(1 + 2) = 3.5; the
    /// fit is the plain regression on those three points.
    #[test]
    fn adjusted_ranks_are_johnson_s() {
        let fit = weibull_mrr_censored(&[10.0, 30.0, 40.0], &[20.0, 50.0]).unwrap();
        let points: Vec<(f64, f64)> = [(10.0, 1.0), (30.0, 2.25), (40.0, 3.5)]
            .iter()
            .map(|&(t, a): &(f64, f64)| {
                let f = (a - 0.3) / 5.4;
                (t.ln(), (-(1.0 - f).ln()).ln())
            })
            .collect();
        let n = 3.0;
        let (sx, sy) = (
            points.iter().map(|p| p.0).sum::<f64>(),
            points.iter().map(|p| p.1).sum::<f64>(),
        );
        let sxy: f64 = points.iter().map(|p| p.0 * p.1).sum();
        let sxx: f64 = points.iter().map(|p| p.0 * p.0).sum();
        let b = (n * sxy - sx * sy) / (n * sxx - sx * sx);
        let a = (sy - b * sx) / n;
        assert!((fit.shape - b).abs() < 1e-12, "{} vs {b}", fit.shape);
        assert!((fit.scale - (-a / b).exp()).abs() < 1e-9);
    }

    /// A failure and a suspension at the same time: the failure is ranked
    /// first (the unit was at risk until then), so the suspension does not
    /// lift that failure's rank.
    #[test]
    fn a_failure_is_ranked_before_a_suspension_at_its_time() {
        let tied = weibull_mrr_censored(&[10.0, 30.0, 40.0], &[10.0]).unwrap();
        let after = weibull_mrr_censored(&[10.0, 30.0, 40.0], &[10.0 + 1e-9]).unwrap();
        assert!((tied.shape - after.shape).abs() < 1e-12);
    }

    /// Survivors beyond every failure lengthen the characteristic life; the
    /// censored MLE moves the same way (both see the survivors at risk).
    #[test]
    fn suspensions_lengthen_the_scale_like_the_mle() {
        let failures = [55.0, 187.0, 216.0, 240.0, 244.0, 335.0];
        let survivors = [361.0, 373.0, 375.0, 386.0];
        let mrr = weibull_mrr_censored(&failures, &survivors).unwrap();
        let dropped = weibull_mrr(&failures).unwrap();
        let mle = crate::weibull::weibull_mle_censored(&failures, &survivors).unwrap();
        assert!(mrr.scale > dropped.scale);
        assert!(
            (mrr.scale / mle.scale - 1.0).abs() < 0.25,
            "{} vs {}",
            mrr.scale,
            mle.scale
        );
    }

    #[test]
    fn censored_refuses_what_the_plain_fit_refuses() {
        assert!(weibull_mrr_censored(&[10.0], &[20.0, 30.0]).is_none());
        assert!(weibull_mrr_censored(&[10.0, 20.0], &[-1.0]).is_none());
        assert!(weibull_mrr_censored(&[10.0, 20.0], &[f64::NAN]).is_none());
    }

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

    /// Helper mirroring the Benard formula used inside `weibull_mrr`.
    fn benard_rank(i: usize, n: usize) -> f64 {
        (i as f64 - 0.3) / (n as f64 + 0.4)
    }

    /// Benard & Bos-Levenbach (1953): F_i = (i - 0.3) / (n + 0.4)
    ///
    /// Spot-check the three boundary-adjacent values for n = 10.
    #[test]
    fn test_benard_median_rank_values() {
        let n = 10_usize;
        assert!(
            (benard_rank(1, n) - 0.067_308).abs() < 1e-5,
            "rank 1: got {}, expected 0.067308",
            benard_rank(1, n)
        );
        assert!(
            (benard_rank(5, n) - 0.451_923).abs() < 1e-5,
            "rank 5: got {}, expected 0.451923",
            benard_rank(5, n)
        );
        assert!(
            (benard_rank(10, n) - 0.932_692).abs() < 1e-5,
            "rank 10: got {}, expected 0.932692",
            benard_rank(10, n)
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

/// Against Johnson's adjusted ranks and Benard's median ranks recomputed in
/// Python 3 with SciPy 1.17.1 (`scipy.stats.linregress` of
/// `ln(-ln(1 - F))` on `ln t`; `shape` = slope, `scale` = `exp(-intercept / slope)`,
/// `r_squared` = `rvalue ** 2`).
#[cfg(test)]
mod scipy_reference {
    use super::*;

    fn check(fit: Option<WeibullMrrResult>, want: (f64, f64, f64), what: &str) {
        let r = fit.expect("fits");
        for (name, got, want) in [
            ("shape", r.shape, want.0),
            ("scale", r.scale, want.1),
            ("r_squared", r.r_squared, want.2),
        ] {
            assert!(
                (got - want).abs() <= 1e-9 * want.abs().max(1.0),
                "{what} {name}: got {got}, SciPy {want}"
            );
        }
    }

    #[test]
    fn uncensored_fits_reproduce_scipy() {
        check(
            weibull_mrr(&[55.0, 187.0, 216.0, 240.0, 244.0, 335.0]),
            (1.5377497056805067, 259.77140279952795, 0.8380959661182655),
            "six failures",
        );
        check(
            weibull_mrr(&[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]),
            (1.541393435175759, 52.54491760626878, 0.9849252880906245),
            "evenly spaced",
        );
        check(
            weibull_mrr(&[37.0, 51.0, 52.0, 78.0, 113.0, 140.0, 190.0, 262.0, 301.0]),
            (1.4126088617100814, 154.27762605393247, 0.9349554705608731),
            "nine failures",
        );
    }

    #[test]
    fn suspensions_move_the_ranks_as_johnson_adjusts_them() {
        check(
            weibull_mrr_censored(
                &[55.0, 187.0, 216.0, 240.0, 244.0, 335.0],
                &[361.0, 373.0, 375.0, 386.0],
            ),
            (1.3230645186146017, 461.70788427887305, 0.8763600658552623),
            "suspensions after every failure",
        );
        // A suspension at a failure's time sorts after it; others fall between.
        check(
            weibull_mrr_censored(
                &[37.0, 51.0, 52.0, 78.0, 113.0, 140.0, 190.0, 262.0, 301.0],
                &[78.0, 200.0, 400.0],
            ),
            (1.3403400442383264, 205.54499329275637, 0.9300385638418595),
            "tied and interleaved suspensions",
        );
    }
}
