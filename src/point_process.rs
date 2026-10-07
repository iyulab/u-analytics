//! Trend tests and the power-law fit for the event times of a point process.
//!
//! The data is one sequence of event times `0 < t₁ ≤ t₂ ≤ … ≤ tₙ` observed on
//! a single unit — the failures of one repairable system, the incidents of
//! one service — and the question is whether events arrive at a constant
//! rate (a homogeneous Poisson process, HPP) or a changing one. That is a
//! different question from [`mann_kendall_test`](crate::testing::mann_kendall_test)
//! (a trend in an ordered series of values) and from
//! [`weibull`](crate::weibull) fitting (independent lifetimes of many units,
//! which assumes exactly the absence of trend these tests check for).
//!
//! # Observation
//!
//! [`Truncation::Time`]`(T)`: the unit was watched until `T ≥ tₙ`, whatever
//! happened. [`Truncation::Failure`]: watching stopped at the last event, so
//! `T = tₙ` and that event carries no information about the trend — the
//! statistics use the `n − 1` events before it.
//!
//! # Methods
//!
//! With `m` the number of events used (`n` time-truncated, `n − 1`
//! failure-truncated) and `T` the end of observation:
//!
//! - **Laplace test** — `U = (Σtᵢ/m − T/2) / (T·√(1/(12m)))`, approximately
//!   standard normal under an HPP. `U > 0`: events crowd towards the end
//!   (rate increasing).
//! - **MIL-HDBK-189 test** — `χ² = 2·Σ ln(T/tᵢ)`, exactly χ² with `2m`
//!   degrees of freedom under an HPP. A *small* value means events crowd
//!   towards the end (rate increasing).
//! - **Power-law process** (Crow-AMSAA) — intensity `λβt^(β−1)`, expected
//!   events `λt^β`. MLE `β̂ = n / Σ ln(T/tᵢ)` (the same sum as the MIL-HDBK-189
//!   statistic, over the same events), `λ̂ = n / T^β̂`. `β̂` is biased upward;
//!   conditionally on `n`, `2nβ/β̂ ~ χ²(2m)`, so `E[β̂] = nβ/(m − 1)` and
//!   `β̄ = (m − 1)/n · β̂` is unbiased — `(n − 1)/n` time-truncated,
//!   `(n − 2)/n` failure-truncated. `β < 1`: rate decreasing; `β > 1`:
//!   increasing.
//!
//! # References
//!
//! - Ascher & Feingold (1984), *Repairable Systems Reliability*, ch. 5.
//! - Crow (1974), "Reliability analysis for complex, repairable systems",
//!   in *Reliability and Biometry*, SIAM, 379–410.
//! - MIL-HDBK-189C (2011), *Reliability Growth Management*, §5.
//! - IEC 61710 (2013), *Power law model — Goodness-of-fit tests and
//!   estimation methods*.
//! - Rausand & Høyland (2004), *System Reliability Theory*, 2nd ed., §7.4.

use u_numflow::special::{chi_squared_cdf, standard_normal_sf};

/// How observation of the unit ended.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Truncation {
    /// Watched until this time, which is at or after the last event.
    Time(f64),
    /// Watching stopped at the last event.
    Failure,
}

/// Which way the observed rate moves — the sign of the deviation from a
/// constant rate, not a significance verdict (see `p_value`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrendDirection {
    /// Events crowd towards the end of observation.
    Increasing,
    /// Events crowd towards the start.
    Decreasing,
    /// Exactly as a constant rate would place them on average.
    Flat,
}

/// Why event times were refused.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PointProcessError {
    /// Fewer events than the method needs.
    TooFewEvents {
        /// The least number of events the method accepts.
        min: usize,
        /// The number given.
        got: usize,
    },
    /// An event time or the end of observation is NaN or infinite.
    NotFinite {
        /// The event's position; `None` for the end of observation.
        index: Option<usize>,
    },
    /// An event time or the end of observation is not greater than 0.
    NotPositive {
        /// The event's position; `None` for the end of observation.
        index: Option<usize>,
        /// The value given.
        got: f64,
    },
    /// An event time is earlier than the one before it.
    Unordered {
        /// The event's position.
        index: usize,
    },
    /// An event time is after the end of observation.
    AfterEnd {
        /// The event's position.
        index: usize,
    },
}

impl std::fmt::Display for PointProcessError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Self::TooFewEvents { min, got } => {
                write!(f, "at least {min} events are needed, got {got}")
            }
            Self::NotFinite { index: Some(i) } => write!(f, "event {i} is not a finite number"),
            Self::NotFinite { index: None } => write!(f, "the end of observation is not finite"),
            Self::NotPositive {
                index: Some(i),
                got,
            } => {
                write!(f, "event {i} is at {got}; event times must be > 0")
            }
            Self::NotPositive { index: None, got } => {
                write!(f, "the end of observation is {got}; it must be > 0")
            }
            Self::Unordered { index } => {
                write!(f, "event {index} is earlier than event {}", index - 1)
            }
            Self::AfterEnd { index } => write!(f, "event {index} is after the end of observation"),
        }
    }
}

impl std::error::Error for PointProcessError {}

/// A trend test's outcome.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrendTestResult {
    /// The test statistic (`U` for Laplace, `χ²` for MIL-HDBK-189).
    pub statistic: f64,
    /// Two-sided p-value against a constant rate. The one-sided p-value in
    /// the observed `direction` is half of it.
    pub p_value: f64,
    /// Which way the observed rate moves.
    pub direction: TrendDirection,
    /// Events the statistic used: `n`, or `n − 1` failure-truncated.
    pub events_used: usize,
    /// χ² degrees of freedom (`2 · events_used`); `None` for the Laplace test.
    pub degrees_of_freedom: Option<f64>,
}

/// A power-law process fit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PowerLawFit {
    /// Shape β̂ (MLE). `< 1`: rate decreasing; `> 1`: increasing.
    pub beta: f64,
    /// The unbiased shape `β̄ = (m − 1)/n · β̂`.
    pub beta_unbiased: f64,
    /// Scale λ̂ (MLE): expected events by time `t` are `λ̂·t^β̂`.
    pub lambda: f64,
    /// Intensity at the end of observation, `λ̂β̂T^(β̂−1)` (= `nβ̂/T`).
    pub intensity_at_end: f64,
    /// The end of observation `T` the fit used.
    pub end: f64,
    /// Number of events `n`.
    pub events: usize,
}

/// Checks the event times and returns `(T, m)`: the end of observation and
/// the number of events the statistics use.
fn observe(
    times: &[f64],
    truncation: Truncation,
    min: usize,
) -> Result<(f64, usize), PointProcessError> {
    for (i, &t) in times.iter().enumerate() {
        if !t.is_finite() {
            return Err(PointProcessError::NotFinite { index: Some(i) });
        }
        if t <= 0.0 {
            return Err(PointProcessError::NotPositive {
                index: Some(i),
                got: t,
            });
        }
        if i > 0 && t < times[i - 1] {
            return Err(PointProcessError::Unordered { index: i });
        }
    }
    if times.len() < min {
        return Err(PointProcessError::TooFewEvents {
            min,
            got: times.len(),
        });
    }
    let n = times.len();
    match truncation {
        Truncation::Time(end) => {
            if !end.is_finite() {
                return Err(PointProcessError::NotFinite { index: None });
            }
            if end <= 0.0 {
                return Err(PointProcessError::NotPositive {
                    index: None,
                    got: end,
                });
            }
            if let Some(i) = times.iter().position(|&t| t > end) {
                return Err(PointProcessError::AfterEnd { index: i });
            }
            Ok((end, n))
        }
        Truncation::Failure => Ok((times[n - 1], n - 1)),
    }
}

/// Minimum events: one used event, plus the dropped last one when failure-truncated.
fn min_events(truncation: Truncation) -> usize {
    match truncation {
        Truncation::Time(_) => 1,
        Truncation::Failure => 2,
    }
}

fn direction(increasing: bool, decreasing: bool) -> TrendDirection {
    if increasing {
        TrendDirection::Increasing
    } else if decreasing {
        TrendDirection::Decreasing
    } else {
        TrendDirection::Flat
    }
}

/// Laplace trend test against a constant event rate.
///
/// `U = (Σtᵢ/m − T/2) / (T·√(1/(12m)))` over the `m` events used, compared
/// with the standard normal (two-sided). The normal approximation is already
/// close for `m ≥ 4`.
///
/// # Errors
/// [`PointProcessError`] for a time that is not finite or `≤ 0`, out of order,
/// after the end of observation, or too few events (1 time-truncated, 2
/// failure-truncated).
///
/// # Examples
/// ```
/// use u_analytics::point_process::{laplace_trend_test, Truncation};
///
/// // Minitab's example: six events, watched until 60.
/// let r = laplace_trend_test(&[12.0, 15.0, 27.0, 34.0, 44.0, 53.0], Truncation::Time(60.0)).unwrap();
/// assert!((r.statistic - 0.118).abs() < 1e-3);
/// assert!((r.p_value - 0.906).abs() < 1e-3);
/// ```
pub fn laplace_trend_test(
    times: &[f64],
    truncation: Truncation,
) -> Result<TrendTestResult, PointProcessError> {
    let (end, m) = observe(times, truncation, min_events(truncation))?;
    let mf = m as f64;
    let mean = times[..m].iter().sum::<f64>() / mf;
    let u = (mean - end / 2.0) / (end * (1.0 / (12.0 * mf)).sqrt());
    Ok(TrendTestResult {
        statistic: u,
        p_value: (2.0 * standard_normal_sf(u.abs())).min(1.0),
        direction: direction(u > 0.0, u < 0.0),
        events_used: m,
        degrees_of_freedom: None,
    })
}

/// MIL-HDBK-189 test against a constant event rate.
///
/// `χ² = 2·Σ ln(T/tᵢ)` over the `m` events used, exactly χ²(2m) under a
/// constant rate (two-sided). Small values mean the rate is increasing.
///
/// # Errors
/// As [`laplace_trend_test`].
///
/// # Examples
/// ```
/// use u_analytics::point_process::{mil_hdbk_189_test, Truncation};
///
/// let r = mil_hdbk_189_test(&[12.0, 15.0, 27.0, 34.0, 44.0, 53.0], Truncation::Time(60.0)).unwrap();
/// assert!((r.statistic - 9.593).abs() < 1e-3);
/// assert_eq!(r.degrees_of_freedom, Some(12.0));
/// assert!((r.p_value - 0.697).abs() < 1e-3);
/// ```
pub fn mil_hdbk_189_test(
    times: &[f64],
    truncation: Truncation,
) -> Result<TrendTestResult, PointProcessError> {
    let (end, m) = observe(times, truncation, min_events(truncation))?;
    let chi2 = 2.0 * times[..m].iter().map(|&t| (end / t).ln()).sum::<f64>();
    let df = 2.0 * m as f64;
    let lower = chi_squared_cdf(chi2, df);
    Ok(TrendTestResult {
        statistic: chi2,
        p_value: (2.0 * lower.min(1.0 - lower)).clamp(0.0, 1.0),
        // The χ² mean under a constant rate is df; below it, events crowd late.
        direction: direction(chi2 < df, chi2 > df),
        events_used: m,
        degrees_of_freedom: Some(df),
    })
}

/// Maximum-likelihood fit of a power-law process (Crow-AMSAA).
///
/// # Errors
/// As [`laplace_trend_test`], with at least 2 events time-truncated and 3
/// failure-truncated (fewer leave `β̄ ≤ 0`). Events that all sit at the end
/// of observation leave `β̂` unbounded and are refused as too few.
///
/// # Examples
/// ```
/// use u_analytics::point_process::{power_law_process_fit, Truncation};
///
/// let fit = power_law_process_fit(&[12.0, 15.0, 27.0, 34.0, 44.0, 53.0], Truncation::Time(60.0)).unwrap();
/// assert!((fit.beta - 1.2509).abs() < 1e-4);
/// assert!((fit.beta_unbiased - fit.beta * 5.0 / 6.0).abs() < 1e-12);
/// // λ̂·T^β̂ reproduces the observed count.
/// assert!((fit.lambda * 60f64.powf(fit.beta) - 6.0).abs() < 1e-9);
/// ```
pub fn power_law_process_fit(
    times: &[f64],
    truncation: Truncation,
) -> Result<PowerLawFit, PointProcessError> {
    let min = min_events(truncation) + 1;
    let (end, m) = observe(times, truncation, min)?;
    let n = times.len();
    let sum: f64 = times[..m].iter().map(|&t| (end / t).ln()).sum();
    if sum <= 0.0 {
        // Every used event sits at the end: no spread to estimate β from.
        return Err(PointProcessError::TooFewEvents { min, got: 0 });
    }
    let nf = n as f64;
    let beta = nf / sum;
    let lambda = nf / end.powf(beta);
    Ok(PowerLawFit {
        beta,
        beta_unbiased: (m as f64 - 1.0) / nf * beta,
        lambda,
        intensity_at_end: nf * beta / end,
        end,
        events: n,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const T: [f64; 6] = [12.0, 15.0, 27.0, 34.0, 44.0, 53.0];

    /// Minitab 16 (Rausand, NTNU TMA4275 slides): MIL-Hdbk-189 9.59 (DF 12,
    /// p 0.697), Laplace 0.12 (p 0.906), shape 1.25093, scale 0.238749 on
    /// time scaled by T = 60.
    #[test]
    fn matches_the_published_example() {
        let l = laplace_trend_test(&T, Truncation::Time(60.0)).unwrap();
        assert!((l.statistic - 5.0 / 1800f64.sqrt()).abs() < 1e-12);
        assert!((l.p_value - 0.906).abs() < 5e-4);
        assert_eq!(l.direction, TrendDirection::Increasing);
        let m = mil_hdbk_189_test(&T, Truncation::Time(60.0)).unwrap();
        assert!((m.statistic - 9.59).abs() < 5e-3);
        assert!((m.p_value - 0.697).abs() < 5e-4);
        assert_eq!(m.direction, TrendDirection::Increasing);
        let f = power_law_process_fit(&T, Truncation::Time(60.0)).unwrap();
        assert!((f.beta - 1.25093).abs() < 5e-6);
        // Minitab's scale θ with W(t) = (t/θ)^β, here on t/60: θ = (1/λ)^(1/β) / 60.
        let theta = (1.0 / f.lambda).powf(1.0 / f.beta) / 60.0;
        assert!((theta - 0.238749).abs() < 5e-6);
        assert!((f.intensity_at_end - 6.0 * f.beta / 60.0).abs() < 1e-12);
    }

    #[test]
    fn failure_truncation_drops_the_last_event() {
        let t = [5.0, 9.0, 20.0, 31.0, 40.0];
        let ft = mil_hdbk_189_test(&t, Truncation::Failure).unwrap();
        let tt = mil_hdbk_189_test(&t[..4], Truncation::Time(40.0)).unwrap();
        assert_eq!(ft.statistic, tt.statistic);
        assert_eq!((ft.events_used, ft.degrees_of_freedom), (4, Some(8.0)));
        let f = power_law_process_fit(&t, Truncation::Failure).unwrap();
        // n = 5 in β̂, m = 4 in the sum; unbiased factor (n − 2)/n.
        assert!((f.beta_unbiased - 3.0 / 5.0 * f.beta).abs() < 1e-12);
        assert!((f.lambda * 40f64.powf(f.beta) - 5.0).abs() < 1e-9);
    }

    #[test]
    fn early_crowding_is_decreasing() {
        let t = [1.0, 2.0, 3.0, 5.0, 8.0, 50.0];
        let l = laplace_trend_test(&t, Truncation::Time(100.0)).unwrap();
        let m = mil_hdbk_189_test(&t, Truncation::Time(100.0)).unwrap();
        assert_eq!(
            (l.direction, m.direction),
            (TrendDirection::Decreasing, TrendDirection::Decreasing)
        );
        assert!(
            power_law_process_fit(&t, Truncation::Time(100.0))
                .unwrap()
                .beta
                < 1.0
        );
    }

    #[test]
    fn refusals_name_the_event() {
        use PointProcessError as E;
        let tt = Truncation::Time(10.0);
        assert_eq!(
            laplace_trend_test(&[], tt),
            Err(E::TooFewEvents { min: 1, got: 0 })
        );
        assert_eq!(
            laplace_trend_test(&[3.0], Truncation::Failure),
            Err(E::TooFewEvents { min: 2, got: 1 })
        );
        assert_eq!(
            power_law_process_fit(&[1.0, 2.0], Truncation::Failure),
            Err(E::TooFewEvents { min: 3, got: 2 })
        );
        assert_eq!(
            laplace_trend_test(&[1.0, f64::NAN], tt),
            Err(E::NotFinite { index: Some(1) })
        );
        assert_eq!(
            laplace_trend_test(&[0.0, 1.0], tt),
            Err(E::NotPositive {
                index: Some(0),
                got: 0.0
            })
        );
        assert_eq!(
            laplace_trend_test(&[2.0, 1.0], tt),
            Err(E::Unordered { index: 1 })
        );
        assert_eq!(
            laplace_trend_test(&[2.0, 11.0], tt),
            Err(E::AfterEnd { index: 1 })
        );
        assert_eq!(
            laplace_trend_test(&[2.0], Truncation::Time(-1.0)),
            Err(E::NotPositive {
                index: None,
                got: -1.0
            })
        );
        assert_eq!(
            laplace_trend_test(&[2.0], Truncation::Time(f64::INFINITY)),
            Err(E::NotFinite { index: None })
        );
        assert_eq!(
            power_law_process_fit(&[10.0, 10.0], tt),
            Err(E::TooFewEvents { min: 2, got: 0 })
        );
    }

    /// Under a constant rate the MIL-HDBK-189 p-value is uniform: at α = 0.05
    /// about 5 % of simulated HPP paths reject (Monte Carlo, 4 000 paths).
    #[test]
    fn mil_hdbk_189_holds_its_size_under_a_constant_rate() {
        use u_numflow::distributions::{Exponential, Sample};
        let gaps = Exponential::new(1.0).unwrap();
        let mut rng = u_numflow::random::create_rng(11);
        let paths = 4000;
        let mut rejected = 0;
        for _ in 0..paths {
            let mut t = 0.0;
            let mut times = Vec::new();
            loop {
                t += gaps.sample(&mut rng);
                if t > 20.0 {
                    break;
                }
                times.push(t);
            }
            if times.is_empty() {
                continue;
            }
            if mil_hdbk_189_test(&times, Truncation::Time(20.0))
                .unwrap()
                .p_value
                < 0.05
            {
                rejected += 1;
            }
        }
        let rate = rejected as f64 / paths as f64;
        assert!((rate - 0.05).abs() < 0.015, "rejection rate {rate}");
    }
}
