//! Profile-likelihood (likelihood-ratio) confidence intervals for the
//! Weibull MLE, right-censored units included.
//!
//! The Wald intervals of [`WeibullMleResult`](super::WeibullMleResult) rest on
//! the curvature at the fit alone; with few failures the log-likelihood is
//! not quadratic and their coverage falls short of nominal. The likelihood-ratio
//! interval keeps every value `θ` whose profile log-likelihood is within
//! `χ²₁(confidence) / 2` of the maximum:
//!
//! ```text
//! { θ : 2 (ℓ̂ − ℓ_p(θ)) ≤ χ²₁,c },   ℓ_p(θ) = max ℓ(β, η) subject to the quantity = θ
//! ```
//!
//! It covers markedly nearer its level than the Wald interval with few
//! failures (eight failures, 90 % B10: about 0.86 against 0.81 by simulation),
//! though still a little short at that size.
//!
//! # Reference
//! Meeker, W.Q. & Escobar, L.A. (1998), *Statistical Methods for Reliability
//! Data*, Wiley, §8.3 (likelihood-based intervals) and §8.4 (functions of the
//! parameters). Lawless (2003), *Statistical Models and Methods for Lifetime
//! Data*, 2nd ed., §4.2.

use super::mle::{two_sided_z, weibull_mle_censored, WeibullInterval};

/// The quantity an interval is for.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WeibullQuantity {
    /// The shape β.
    Shape,
    /// The scale η (characteristic life).
    Scale,
    /// The B-life at a fraction failed in (0, 1): 0.1 is B10.
    BLife(f64),
}

/// The censored log-likelihood and its data.
struct Likelihood {
    /// Logs of every unit's time; the first `r` are the failures.
    ln_t: Vec<f64>,
    r: f64,
    sum_ln_fail: f64,
}

impl Likelihood {
    fn new(failures: &[f64], suspensions: &[f64]) -> Self {
        let ln_t: Vec<f64> = failures.iter().chain(suspensions).map(|t| t.ln()).collect();
        let sum_ln_fail = ln_t[..failures.len()].iter().sum();
        Self {
            ln_t,
            r: failures.len() as f64,
            sum_ln_fail,
        }
    }

    /// `ℓ(β, η)` with `ln η` given.
    fn at(&self, beta: f64, ln_eta: f64) -> f64 {
        let sum_z: f64 = self
            .ln_t
            .iter()
            .map(|lt| (beta * (lt - ln_eta)).exp())
            .sum();
        self.r * beta.ln() - self.r * beta * ln_eta + (beta - 1.0) * self.sum_ln_fail - sum_z
    }

    /// `ℓ` maximized over `η` at this `β`: `η̂(β) = (Σ_all t^β / r)^(1/β)`.
    fn profile_shape(&self, beta: f64) -> f64 {
        let s0: f64 = self.ln_t.iter().map(|lt| (beta * lt).exp()).sum();
        self.at(beta, (s0 / self.r).ln() / beta)
    }

    /// `ℓ` maximized over `β`, `ln η` being a function of `β`. Every use here
    /// is concave in `β` (`−r/β² − Σ z·y² < 0`, also with `η = t_p e^(−w/β)`),
    /// so a golden-section search over `ln β` finds the one maximum.
    fn profile_over_shape(&self, beta_hat: f64, ln_eta: impl Fn(f64) -> f64) -> f64 {
        let f = |s: f64| {
            let beta = s.exp();
            self.at(beta, ln_eta(beta))
        };
        let (mut a, mut b) = (beta_hat.ln() - 8.0, beta_hat.ln() + 8.0);
        let g = (5f64.sqrt() - 1.0) / 2.0;
        let (mut c, mut d) = (b - g * (b - a), a + g * (b - a));
        let (mut fc, mut fd) = (f(c), f(d));
        while b - a > 1e-12 {
            if fc > fd {
                b = d;
                d = c;
                fd = fc;
                c = b - g * (b - a);
                fc = f(c);
            } else {
                a = c;
                c = d;
                fc = fd;
                d = a + g * (b - a);
                fd = f(d);
            }
        }
        f((a + b) / 2.0)
    }
}

/// The profile-likelihood interval for `quantity` at `confidence` from
/// failure and suspension times (the data [`weibull_mle_censored`] takes).
///
/// Returns `(estimate, interval)` -- the estimate is the MLE's. `None` when
/// the fit is (fewer than 2 failures, a non-positive time, no convergence),
/// when `confidence` or a B-life fraction is outside (0, 1), or when a limit
/// cannot be bracketed (a likelihood that never falls `χ²₁/2` below its
/// maximum on that side).
///
/// # Examples
///
/// ```
/// use u_analytics::weibull::{weibull_likelihood_interval, weibull_mle_censored, WeibullQuantity};
/// let failures = [55.0, 187.0, 216.0, 240.0, 244.0, 335.0];
/// let survivors = [361.0, 373.0, 375.0, 386.0];
/// let (b10, lr) =
///     weibull_likelihood_interval(&failures, &survivors, WeibullQuantity::BLife(0.1), 0.9).unwrap();
/// let (_, wald) = weibull_mle_censored(&failures, &survivors)
///     .unwrap()
///     .b_life_interval(0.1, 0.9)
///     .unwrap();
/// assert!(lr.lower < b10 && b10 < lr.upper);
/// // Few failures: the likelihood is skewed, and so is its interval.
/// assert!((lr.lower - wald.lower).abs() > 1.0);
/// ```
pub fn weibull_likelihood_interval(
    failure_times: &[f64],
    suspension_times: &[f64],
    quantity: WeibullQuantity,
    confidence: f64,
) -> Option<(f64, WeibullInterval)> {
    let z = two_sided_z(confidence)?;
    let fit = weibull_mle_censored(failure_times, suspension_times)?;
    let data = Likelihood::new(failure_times, suspension_times);
    let (beta_hat, eta_hat) = (fit.shape, fit.scale);
    let max = data.at(beta_hat, eta_hat.ln());
    let cutoff = z * z; // χ²₁ at `confidence`

    // The estimate, its log-scale Wald standard error (the first step out),
    // and the profile log-likelihood as a function of the log of the quantity.
    let ra = super::ReliabilityAnalysis::from_mle(&fit);
    let (estimate, se_log, profile): (f64, f64, Box<dyn Fn(f64) -> f64 + '_>) = match quantity {
        WeibullQuantity::Shape => (
            beta_hat,
            fit.covariance[0][0].sqrt() / beta_hat,
            Box::new(|s: f64| data.profile_shape(s.exp())),
        ),
        WeibullQuantity::Scale => (
            eta_hat,
            fit.covariance[1][1].sqrt() / eta_hat,
            Box::new(|s: f64| data.profile_over_shape(beta_hat, |_| s)),
        ),
        WeibullQuantity::BLife(p) => {
            let (t_p, wald) = ra.b_life_interval(p, confidence)?;
            let w = (-(1.0 - p).ln()).ln();
            (
                t_p,
                (wald.upper / t_p).ln() / z,
                // ln t_p = ln η + w/β  →  ln η = s − w/β
                Box::new(move |s: f64| data.profile_over_shape(beta_hat, |beta| s - w / beta)),
            )
        }
    };
    let deviance = |s: f64| 2.0 * (max - profile(s)) - cutoff;

    let s0 = estimate.ln();
    let step = if se_log.is_finite() && se_log > 0.0 {
        se_log
    } else {
        0.1
    };
    let limit = |direction: f64| -> Option<f64> {
        // Walk out until the deviance crosses the cutoff, then bisect.
        let mut inner = s0;
        let mut h = step;
        let mut outer = s0 + direction * h;
        let mut tries = 0;
        while deviance(outer) < 0.0 {
            inner = outer;
            h *= 2.0;
            outer = s0 + direction * h;
            tries += 1;
            if tries > 40 || !outer.is_finite() {
                return None;
            }
        }
        for _ in 0..200 {
            let mid = (inner + outer) / 2.0;
            if (outer - inner).abs() < 1e-12 {
                break;
            }
            if deviance(mid) < 0.0 {
                inner = mid;
            } else {
                outer = mid;
            }
        }
        Some(((inner + outer) / 2.0).exp())
    };
    let lower = limit(-1.0)?;
    let upper = limit(1.0)?;
    Some((estimate, WeibullInterval { lower, upper }))
}

#[cfg(test)]
mod tests {
    use super::*;

    const FAILURES: [f64; 6] = [55.0, 187.0, 216.0, 240.0, 244.0, 335.0];
    const SURVIVORS: [f64; 4] = [361.0, 373.0, 375.0, 386.0];

    fn chi2_1(confidence: f64) -> f64 {
        let z = two_sided_z(confidence).unwrap();
        z * z
    }

    /// The limits are where the profile deviance meets χ²₁ -- with the
    /// profile taken by a brute-force grid over β, not the search above.
    #[test]
    fn limits_sit_on_the_chi_square_cutoff() {
        let data = Likelihood::new(&FAILURES, &SURVIVORS);
        let fit = weibull_mle_censored(&FAILURES, &SURVIVORS).unwrap();
        let max = data.at(fit.shape, fit.scale.ln());
        let grid_max = |ln_eta: &dyn Fn(f64) -> f64| {
            (1..=200_000)
                .map(|i| {
                    let beta = i as f64 * 1e-4;
                    data.at(beta, ln_eta(beta))
                })
                .fold(f64::NEG_INFINITY, f64::max)
        };
        let q = chi2_1(0.9);
        let (_, scale) =
            weibull_likelihood_interval(&FAILURES, &SURVIVORS, WeibullQuantity::Scale, 0.9)
                .unwrap();
        for eta in [scale.lower, scale.upper] {
            let dev = 2.0 * (max - grid_max(&|_| eta.ln()));
            assert!(
                (dev - q).abs() < 1e-4,
                "scale limit {eta}: deviance {dev} vs {q}"
            );
        }
        let (_, b10) =
            weibull_likelihood_interval(&FAILURES, &SURVIVORS, WeibullQuantity::BLife(0.1), 0.9)
                .unwrap();
        let w = (-(0.9f64).ln()).ln();
        for t in [b10.lower, b10.upper] {
            let dev = 2.0 * (max - grid_max(&|beta| t.ln() - w / beta));
            assert!(
                (dev - q).abs() < 1e-4,
                "B10 limit {t}: deviance {dev} vs {q}"
            );
        }
        let (_, shape) =
            weibull_likelihood_interval(&FAILURES, &SURVIVORS, WeibullQuantity::Shape, 0.9)
                .unwrap();
        for beta in [shape.lower, shape.upper] {
            let dev = 2.0 * (max - data.profile_shape(beta));
            assert!(
                (dev - q).abs() < 1e-8,
                "shape limit {beta}: deviance {dev} vs {q}"
            );
        }
    }

    /// With many failures the likelihood is nearly quadratic and the two
    /// intervals agree.
    #[test]
    fn with_much_data_it_meets_the_wald_interval() {
        use u_numflow::distributions::{Sample, Weibull};
        let law = Weibull::new(2.0, 100.0).unwrap();
        let mut rng = u_numflow::random::create_rng(7);
        let data: Vec<f64> = (0..2000).map(|_| law.sample(&mut rng)).collect();
        let fit = weibull_mle_censored(&data, &[]).unwrap();
        let (_, lr) =
            weibull_likelihood_interval(&data, &[], WeibullQuantity::Shape, 0.95).unwrap();
        let wald = fit.shape_interval(0.95).unwrap();
        assert!(
            (lr.lower / wald.lower - 1.0).abs() < 0.005,
            "{lr:?} vs {wald:?}"
        );
        assert!(
            (lr.upper / wald.upper - 1.0).abs() < 0.005,
            "{lr:?} vs {wald:?}"
        );
    }

    /// Monte Carlo, n = 8 failures of Weibull(1.5, 100), 90 % for B10 over
    /// 800 samples (binomial s.e. ≈ 1.1 %). Measured: likelihood ratio 0.86,
    /// Wald 0.81 -- the likelihood interval is markedly nearer nominal, and
    /// still a little short at eight failures, as small-sample studies of it
    /// show (a calibrated interval is a further method).
    #[test]
    fn with_few_failures_it_covers_near_its_level() {
        use u_numflow::distributions::{Sample, Weibull};
        let law = Weibull::new(1.5, 100.0).unwrap();
        let truth = 100.0 * (-(0.9f64).ln()).powf(1.0 / 1.5);
        let mut rng = u_numflow::random::create_rng(31);
        let runs = 800;
        let (mut lr_in, mut wald_in, mut used) = (0, 0, 0);
        for _ in 0..runs {
            let data: Vec<f64> = (0..8).map(|_| law.sample(&mut rng)).collect();
            let Some((_, lr)) =
                weibull_likelihood_interval(&data, &[], WeibullQuantity::BLife(0.1), 0.9)
            else {
                continue;
            };
            let (_, wald) = weibull_mle_censored(&data, &[])
                .unwrap()
                .b_life_interval(0.1, 0.9)
                .unwrap();
            used += 1;
            lr_in += usize::from(lr.lower <= truth && truth <= lr.upper);
            wald_in += usize::from(wald.lower <= truth && truth <= wald.upper);
        }
        let (lr_c, wald_c) = (lr_in as f64 / used as f64, wald_in as f64 / used as f64);
        assert!(used >= runs * 99 / 100, "{used} of {runs} fitted");
        assert!(
            (0.84..=0.94).contains(&lr_c),
            "likelihood-ratio coverage {lr_c} (Wald {wald_c})"
        );
        assert!(
            lr_c > wald_c + 0.025,
            "likelihood-ratio {lr_c} vs Wald {wald_c}"
        );
    }

    #[test]
    fn refuses_what_it_cannot_answer() {
        assert!(
            weibull_likelihood_interval(&FAILURES, &SURVIVORS, WeibullQuantity::Shape, 1.0)
                .is_none()
        );
        assert!(weibull_likelihood_interval(
            &FAILURES,
            &SURVIVORS,
            WeibullQuantity::BLife(1.0),
            0.9
        )
        .is_none());
        assert!(weibull_likelihood_interval(&[10.0], &[], WeibullQuantity::Scale, 0.9).is_none());
    }
}
