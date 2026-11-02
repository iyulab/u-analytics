//! Forecasts of the smoothers with their prediction intervals.
//!
//! The additive smoothers are the exponential smoothing state-space models
//! with additive errors -- SES is ETS(A,N,N), Holt ETS(A,A,N), additive
//! Holt-Winters ETS(A,A,A) -- whose h-step forecast variance has a closed form
//! (Hyndman, Koehler, Ord & Snyder 2008, Table 6.1, class 1):
//!
//! ```text
//! Var(h) = σ² (1 + Σ_{j=1}^{h-1} c_j²)
//! SES:          c_j = α
//! Holt:         c_j = α (1 + j β)
//! Holt-Winters: c_j = α (1 + j β) + γ (1 − α) [j mod m = 0]
//! ```
//!
//! with the smoothing constants of the component form used here (the
//! state-space form's trend and seasonal constants are αβ and γ(1 − α)), and σ²
//! the mean squared one-step error over the observations the smoother updates
//! on (not those its initialisation reproduces). Multiplicative seasonality has
//! no such closed form (class 3), so its forecasts carry no interval.
//!
//! The forecast of the total over steps 1..=H -- what a cumulative quantity
//! (stock consumed, wear, a budget spent) reaches -- has the variance of the
//! sum of the h-step errors. Each error is `ε_{n+h} + Σ_{j<h} c_j ε_{n+h−j}`,
//! so the innovation of step t enters every later step and
//!
//! ```text
//! Var(Σ_{h≤H}) = σ² Σ_{k=0}^{H−1} (1 + C_k)²,   C_k = Σ_{j=1}^{k} c_j
//! ```
//!
//! (the variance of an aggregated forecast, Hyndman et al. 2008, chapter 6) -- wider than
//! the sum of the marginal variances, narrower than the sum of the marginal
//! interval widths. [`first_passage`] simulates the same innovations to give
//! the distribution of the time the cumulative path first reaches a level.
//!
//! # Reference
//!
//! Hyndman, R.J., Koehler, A.B., Ord, J.K. & Snyder, R.D. (2008).
//! *Forecasting with Exponential Smoothing: The State Space Approach*.
//! Springer, chapter 6.

/// One forecast of a smoother: `step` periods after the last observation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SmoothingForecast {
    /// Steps ahead, from 1.
    pub step: usize,
    /// The point forecast.
    pub mean: f64,
    /// Lower end of the prediction interval; `None` when the smoother has no
    /// closed-form interval (multiplicative seasonality) or no one-step errors
    /// to estimate σ from.
    pub lower: Option<f64>,
    /// Upper end of the prediction interval.
    pub upper: Option<f64>,
    /// The point forecast of the total over steps 1..=`step`.
    pub cumulative_mean: f64,
    /// Lower end of the total's prediction interval (`None` as for `lower`).
    pub cumulative_lower: Option<f64>,
    /// Upper end of the total's prediction interval.
    pub cumulative_upper: Option<f64>,
}

/// Mean squared one-step error `data[t] − fitted[t]` for `t >= first`, or
/// `None` when there is none.
pub(crate) fn one_step_mse(data: &[f64], fitted: &[f64], first: usize) -> Option<f64> {
    let errors: Vec<f64> = data
        .iter()
        .zip(fitted)
        .skip(first)
        .map(|(x, f)| x - f)
        .collect();
    (!errors.is_empty()).then(|| errors.iter().map(|e| e * e).sum::<f64>() / errors.len() as f64)
}

/// The one-step errors from `first` on, `None` before it (where the fitted
/// value is the initialisation).
pub(crate) fn residuals(data: &[f64], fitted: &[f64], first: usize) -> Vec<Option<f64>> {
    data.iter()
        .zip(fitted)
        .enumerate()
        .map(|(t, (x, f))| (t >= first).then(|| x - f))
        .collect()
}

/// Forecasts `mean(h)` for h = 1..=horizon with the class-1 interval at
/// two-sided `level` when `mse` and `c` are given.
pub(crate) fn with_intervals(
    horizon: usize,
    level: f64,
    mse: Option<f64>,
    c: Option<&dyn Fn(usize) -> f64>,
    mean: impl Fn(usize) -> f64,
) -> Vec<SmoothingForecast> {
    let z = u_numflow::special::inverse_normal_cdf(0.5 + level / 2.0);
    let mut sum_c2 = 0.0;
    // C_{h-1} = Σ_{j<h} c_j, and Σ_{k<h} (1 + C_k)² for the total.
    let mut sum_c = 0.0;
    let mut total_weight = 0.0;
    let mut total = 0.0;
    (1..=horizon)
        .map(|h| {
            let m = mean(h);
            total += m;
            let (half, total_half) = match (mse, c) {
                (Some(mse), Some(c)) => {
                    if h > 1 {
                        let cj = c(h - 1);
                        sum_c2 += cj.powi(2);
                        sum_c += cj;
                    }
                    total_weight += (1.0 + sum_c).powi(2);
                    (
                        Some(z * (mse * (1.0 + sum_c2)).sqrt()),
                        Some(z * (mse * total_weight).sqrt()),
                    )
                }
                _ => (None, None),
            };
            SmoothingForecast {
                step: h,
                mean: m,
                lower: half.map(|w| m - w),
                upper: half.map(|w| m + w),
                cumulative_mean: total,
                cumulative_lower: total_half.map(|w| total - w),
                cumulative_upper: total_half.map(|w| total + w),
            }
        })
        .collect()
}

/// `level` strictly inside (0, 1).
pub(crate) fn valid_level(level: f64) -> bool {
    level > 0.0 && level < 1.0
}

/// The arguments [`first_passage`] takes: some steps, a non-zero finite
/// level, quantiles in (0, 1], at least one future.
pub(crate) fn passage_arguments(
    horizon: usize,
    threshold: f64,
    quantiles: &[f64],
    paths: usize,
) -> bool {
    horizon > 0
        && paths > 0
        && threshold.is_finite()
        && threshold != 0.0
        && quantiles.iter().all(|&q| q > 0.0 && q <= 1.0)
}

/// When the forecast total first reaches a level: over simulated futures of
/// the fitted model ([`SesResult::first_passage`](super::SesResult::first_passage) and the
/// same method of the other additive smoothers).
#[derive(Debug, Clone, PartialEq)]
pub struct FirstPassage {
    /// The level the total over steps 1, 2, … reaches: from below when
    /// positive, from above when negative.
    pub threshold: f64,
    /// Share of the simulated futures whose total reaches it within the
    /// horizon.
    pub probability: f64,
    /// `(q, time)` for each quantile asked: the time, in steps after the last
    /// observation, by which a share `q` of the futures has reached the level
    /// -- interpolated within the step it is reached in (the total is taken to
    /// grow evenly through a step). `None` when fewer than that share reach it
    /// within the horizon.
    pub quantiles: Vec<(f64, Option<f64>)>,
    /// The number of futures simulated.
    pub paths: usize,
}

/// The weights `c_j` an innovation carries `j` steps later in the additive
/// smoothers: `level + slope · j`, plus `season` every `period` steps --
/// SES `(α, 0, 0)`, Holt `(α, αβ, 0)`, Holt-Winters `(α, αβ, γ(1 − α))` with its
/// period (the c_j of the module documentation).
#[derive(Debug, Clone, Copy)]
pub(crate) struct InnovationWeights {
    pub(crate) level: f64,
    pub(crate) slope: f64,
    pub(crate) season: f64,
    /// `0`: no season.
    pub(crate) period: usize,
}

impl InnovationWeights {
    pub(crate) fn c(&self, j: usize) -> f64 {
        self.level
            + self.slope * j as f64
            + if self.period > 0 && j % self.period == 0 {
                self.season
            } else {
                0.0
            }
    }
}

/// Simulates `paths` futures of an additive smoother over `means.len()` steps
/// -- step h is `means[h−1] + ε_h + Σ_{j<h} c_j ε_{h−j}` with independent
/// `ε ~ N(0, mse)` drawn from a generator seeded with `seed` -- and reports
/// when each one's running total first reaches `threshold` (non-zero).
/// Quantiles are the inverse empirical distribution of those times, a future
/// that never reaches it counting as later than every horizon.
///
/// `O(paths · H)` for `H` steps: with `c_j = a + b·j + g·[j mod m = 0]` the
/// weighted sum of the earlier innovations is `a·S₀ + b·S₁ + g·R`, where
/// `S₀ = Σ_{t<h} ε_t`, `S₁ = Σ_{t<h} (h − t) ε_t` and `R` sums the innovations
/// a whole number of periods back -- each updated in constant time a step.
pub(crate) fn first_passage(
    means: &[f64],
    mse: f64,
    weights: InnovationWeights,
    threshold: f64,
    quantiles: &[f64],
    paths: usize,
    seed: u64,
) -> FirstPassage {
    use u_numflow::distributions::{Normal, Sample};
    let horizon = means.len();
    let InnovationWeights {
        level: a,
        slope: b,
        season: g,
        period: m,
    } = weights;
    // σ = 0 (a perfect fit): every future is the point forecast.
    let noise = (mse > 0.0).then(|| Normal::new(0.0, mse.sqrt()).expect("a finite, positive σ"));
    let mut rng = u_numflow::random::create_rng(seed);
    let reached = |total: f64| {
        if threshold > 0.0 {
            total >= threshold
        } else {
            total <= threshold
        }
    };
    // Innovations a whole number of periods apart, by residue.
    let mut by_residue = vec![0.0_f64; m.max(1)];
    let mut times: Vec<f64> = Vec::with_capacity(paths);
    for _ in 0..paths {
        // All of a future's innovations are drawn, in order, whether or not it
        // reaches the level early -- the draws of the next future do not
        // depend on where this one stopped.
        let draws: Vec<f64> = (0..horizon)
            .map(|_| noise.as_ref().map_or(0.0, |n| n.sample(&mut rng)))
            .collect();
        let (mut s0, mut s1) = (0.0_f64, 0.0_f64);
        by_residue.iter_mut().for_each(|r| *r = 0.0);
        let mut total = 0.0_f64;
        let mut time = f64::INFINITY;
        for h in 0..horizon {
            // Step h+1: s0 = Σ_{t≤h} ε_t, s1 = Σ_{t≤h} (h+1−t) ε_t over the
            // earlier steps t = 1..=h (zero-based t < h here).
            let earlier_season = if m > 0 { by_residue[h % m] } else { 0.0 };
            let error = draws[h] + a * s0 + b * s1 + g * earlier_season;
            s0 += draws[h];
            s1 += s0;
            if m > 0 {
                by_residue[h % m] += draws[h];
            }
            let next = total + means[h] + error;
            if reached(next) {
                // The share of step h+1 the total needs to reach the level.
                let fraction = ((threshold - total) / (next - total)).clamp(0.0, 1.0);
                time = h as f64 + fraction;
                break;
            }
            total = next;
        }
        times.push(time);
    }
    times.sort_by(f64::total_cmp);
    let crossed = times.iter().filter(|t| t.is_finite()).count();
    let quantiles = quantiles
        .iter()
        .map(|&q| {
            // The smallest time t with a share >= q of the futures by t.
            let rank = ((q * paths as f64).ceil() as usize).clamp(1, paths);
            let t = times[rank - 1];
            (q, t.is_finite().then_some(t))
        })
        .collect();
    FirstPassage {
        threshold,
        probability: crossed as f64 / paths as f64,
        quantiles,
        paths,
    }
}

/// The fitted values, forecasts and prediction intervals of the additive
/// smoothers against statsmodels 0.15.0 `ETSModel(..., error="add",
/// initialization_method="known").smooth(params)` with the same initial states
/// (`get_prediction(...).summary_frame(alpha=0.1)`), on seeded series rounded to
/// 3 places: a random walk (SES), a noisy line (Holt) and a period-4 seasonal
/// line (Holt-Winters), six steps ahead.
#[cfg(test)]
#[allow(clippy::approx_constant)] // seeded data, not mathematical constants
mod statsmodels_reference {
    use super::*;
    use crate::smoothing::{
        HoltLinear, HoltResult, HoltWinters, Seasonality, SesResult, SimpleExponentialSmoothing,
    };

    // Generated by ets_ref.py - statsmodels 0.15.0, seed 20261028.
    const SES_DATA: &[f64] = &[
        49.695, 48.755, 48.385, 48.678, 47.146, 48.813, 49.806, 49.983, 50.957, 49.731, 48.969,
        49.761, 48.5, 49.557, 50.796, 51.74, 52.288, 55.501, 55.645, 55.541, 55.655, 55.734,
        56.497, 55.544, 53.348, 52.352, 53.479, 52.5, 53.481, 55.59,
    ];
    const SES_FITTED: &[f64] = &[
        49.695,
        49.366,
        49.02265,
        48.9020225,
        48.287414625000004,
        48.47136950625,
        48.9384901790625,
        49.30406861639062,
        49.8825946006539,
        49.82953649042504,
        49.52834871877628,
        49.609776667204585,
        49.22135483368298,
        49.338830641893935,
        49.848839917231054,
        50.51074594620019,
        51.13278486503012,
        52.66166016226957,
        53.70582910547522,
        54.34813891855889,
        54.80554029706328,
        55.130501193091135,
        55.60877577550924,
        55.586104254081,
        54.80276776515265,
        53.94499904734923,
        53.781899380777,
        53.33323459750505,
        53.38495248837828,
    ];
    const SES_MEAN: &[f64] = &[
        54.15671911744588,
        54.15671911744588,
        54.15671911744588,
        54.15671911744588,
        54.15671911744588,
        54.15671911744588,
    ];
    const SES_LOWER: &[f64] = &[
        51.50271152380141,
        51.344848484798455,
        51.195388909183464,
        51.053118506866284,
        50.917089987805795,
        50.786547453385566,
    ];
    const SES_UPPER: &[f64] = &[
        56.81072671109035,
        56.96858975009331,
        57.1180493257083,
        57.26031972802548,
        57.39634824708597,
        57.5268907815062,
    ];
    const HOLT_DATA: &[f64] = &[
        18.624, 22.028, 22.022, 22.77, 23.168, 25.438, 23.856, 26.135, 25.839, 25.885, 27.652,
        29.208, 30.853, 32.14, 31.311, 31.097, 32.108, 31.913, 36.539, 34.939, 34.931, 36.448,
        37.208, 36.854, 38.229, 38.696, 41.158, 43.842, 42.405, 41.971,
    ];
    const HOLT_FITTED: &[f64] = &[
        25.432,
        27.199199999999998,
        28.204383999999997,
        28.563783679999997,
        29.4373607936,
        28.882198198272,
        29.24092478521344,
        29.065606754561227,
        28.724367395805,
        29.140634388886866,
        30.018183833625034,
        31.269498793777935,
        32.60472786636744,
        32.97076708061175,
        32.954889242709385,
        33.28201140055122,
        33.29076378321223,
        35.406274110151855,
        35.99819837750347,
        36.26477706771417,
        37.04618211642346,
        37.83197057633515,
        38.08360600617535,
        38.796218783585445,
        39.402568947344676,
        40.89161352981263,
        43.094671196908386,
        43.78653210141317,
    ];
    const HOLT_MEAN: &[f64] = &[
        43.88280607600299,
        44.70529289115808,
        45.52777970631317,
        46.35026652146826,
        47.172753336623344,
        47.99524015177843,
    ];
    const HOLT_LOWER: &[f64] = &[
        39.881805953904774,
        40.26724764211682,
        40.55622511518835,
        40.758018971303315,
        40.88214280288195,
        40.93713118743697,
    ];
    const HOLT_UPPER: &[f64] = &[
        47.88380619810121,
        49.14333814019934,
        50.49933429743799,
        51.9425140716332,
        53.46336387036474,
        55.0533491161199,
    ];
    const HW_DATA: &[f64] = &[
        107.195, 99.797, 99.111, 104.953, 111.536, 103.86, 105.869, 111.842, 117.337, 112.456,
        110.451, 115.115, 124.234, 116.183, 117.719, 123.434, 127.431, 122.666, 123.802, 130.548,
        135.536, 127.843, 129.797, 133.84, 142.645, 133.886, 134.436, 141.34, 148.881, 140.176,
        141.191, 145.134,
    ];
    const HW_FITTED: &[f64] = &[
        108.57318749999999,
        103.53110312499999,
        104.42071096874999,
        112.21758513031247,
        116.37452167699682,
        110.342324187491,
        112.08772508093094,
        118.67183234829378,
        122.04503373510259,
        116.55551472827399,
        116.797425126349,
        123.78843943015733,
        129.19938105911905,
        122.02718587096355,
        122.78208766197088,
        129.56537788125988,
        135.15618721562316,
        129.1403441795491,
        129.4136407564024,
        136.01289275800133,
        140.4703440485413,
        134.67208484162995,
        135.38085344677242,
        141.08242228813216,
        147.0508746050361,
        140.6418295369387,
        141.43920083128222,
        147.6014010922094,
    ];
    const HW_MEAN: &[f64] = &[
        152.98701643046797,
        145.53169868698552,
        146.42543580819583,
        152.19971424007915,
        158.7573450971372,
        151.30202735365475,
    ];
    const HW_LOWER: &[f64] = &[
        150.4614216897758,
        142.8721382966387,
        143.61475408277198,
        149.22143937008624,
        155.42143213962436,
        147.77782707686427,
    ];
    const HW_UPPER: &[f64] = &[
        155.51261117116016,
        148.19125907733235,
        149.23611753361968,
        155.17798911007205,
        162.09325805465005,
        154.82622763044523,
    ];

    fn close(got: f64, want: f64, what: &str) {
        let tol = 1e-9 * want.abs().max(1.0);
        assert!((got - want).abs() <= tol, "{what}: {got} vs {want}");
    }

    fn compare(
        fitted: &[f64],
        first: usize,
        forecasts: &[SmoothingForecast],
        want_fitted: &[f64],
        want: [&[f64]; 3],
    ) {
        assert_eq!(fitted.len() - first, want_fitted.len());
        for (got, want) in fitted[first..].iter().zip(want_fitted) {
            close(*got, *want, "fitted");
        }
        for (i, f) in forecasts.iter().enumerate() {
            close(f.mean, want[0][i], "mean");
            close(f.lower.expect("interval"), want[1][i], "lower");
            close(f.upper.expect("interval"), want[2][i], "upper");
        }
    }

    #[test]
    fn ses_matches_ets_ann() {
        let r = SimpleExponentialSmoothing::new(0.35)
            .unwrap()
            .smooth(SES_DATA)
            .unwrap();
        let f = r.forecasts(SES_DATA, 6, 0.9).unwrap();
        compare(
            &r.fitted,
            SesResult::FIRST_ERROR,
            &f,
            SES_FITTED,
            [SES_MEAN, SES_LOWER, SES_UPPER],
        );
    }

    #[test]
    fn holt_matches_ets_aan() {
        let r = HoltLinear::new(0.4, 0.2)
            .unwrap()
            .smooth(HOLT_DATA)
            .unwrap();
        let f = r.forecasts(HOLT_DATA, 6, 0.9).unwrap();
        compare(
            &r.fitted,
            HoltResult::FIRST_ERROR,
            &f,
            HOLT_FITTED,
            [HOLT_MEAN, HOLT_LOWER, HOLT_UPPER],
        );
    }

    #[test]
    fn additive_holt_winters_matches_ets_aaa() {
        let r = HoltWinters::new(0.3, 0.1, 0.25, 4, Seasonality::Additive)
            .unwrap()
            .smooth(HW_DATA)
            .unwrap();
        let f = r.forecasts(HW_DATA, 6, 0.9).unwrap();
        compare(
            &r.fitted,
            r.first_error(),
            &f,
            HW_FITTED,
            [HW_MEAN, HW_LOWER, HW_UPPER],
        );
    }
}

#[cfg(test)]
mod cumulative {
    use super::*;

    const SES_03: InnovationWeights = InnovationWeights {
        level: 0.3,
        slope: 0.0,
        season: 0.0,
        period: 0,
    };

    /// Var(Σ_{h≤H} e_h) from the covariances of the h-step errors themselves:
    /// e_h = Σ_{t≤h} w_h(t) ε_t with w_h(h) = 1, w_h(t) = c_{h−t}.
    fn total_variance_by_covariance(mse: f64, c: &dyn Fn(usize) -> f64, horizon: usize) -> f64 {
        let w = |h: usize, t: usize| if t == h { 1.0 } else { c(h - t) };
        let mut v = 0.0;
        for h in 1..=horizon {
            for g in 1..=horizon {
                for t in 1..=h.min(g) {
                    v += w(h, t) * w(g, t);
                }
            }
        }
        mse * v
    }

    #[test]
    fn the_totals_interval_is_the_sum_of_the_error_covariances() {
        let z = u_numflow::special::inverse_normal_cdf(0.975);
        let holt_winters =
            |j: usize| 0.3 * (1.0 + j as f64 * 0.1) + if j % 4 == 0 { 0.2 * 0.7 } else { 0.0 };
        let cases: [(&str, &dyn Fn(usize) -> f64); 3] = [
            ("ses", &|_| 0.4),
            ("holt", &|j| 0.3 * (1.0 + j as f64 * 0.1)),
            ("holt_winters", &holt_winters),
        ];
        for (name, c) in cases {
            let fs = with_intervals(12, 0.95, Some(2.5), Some(c), |h| 10.0 + h as f64);
            for f in &fs {
                let half = z * total_variance_by_covariance(2.5, c, f.step).sqrt();
                let total: f64 = (1..=f.step).map(|h| 10.0 + h as f64).sum();
                assert!(
                    (f.cumulative_mean - total).abs() < 1e-9,
                    "{name} h {}",
                    f.step
                );
                let lower = f.cumulative_lower.expect("an interval");
                assert!(
                    (lower - (total - half)).abs() < 1e-9 * total.abs().max(1.0),
                    "{name} h {}",
                    f.step
                );
            }
            // One step: the total is that step.
            assert_eq!(fs[0].cumulative_lower, fs[0].lower);
        }
        // No error model, no interval for the total either.
        let fs = with_intervals(3, 0.95, None, None, |_| 1.0);
        assert!(fs
            .iter()
            .all(|f| f.cumulative_lower.is_none() && f.cumulative_upper.is_none()));
        assert_eq!(fs[2].cumulative_mean, 3.0);
    }

    #[test]
    fn without_noise_the_level_is_reached_when_the_forecast_total_reaches_it() {
        // 760 a step, a level of 4 820: 6 whole steps (4 560) and 260/760 of the 7th.
        let fp = first_passage(&[760.0; 10], 0.0, SES_03, 4820.0, &[0.1, 0.5, 0.9], 50, 1);
        assert_eq!(fp.probability, 1.0);
        for (_, t) in &fp.quantiles {
            assert!((t.expect("reached") - (6.0 + 260.0 / 760.0)).abs() < 1e-12);
        }
        // Never reached within the horizon: no time.
        let fp = first_passage(&[760.0; 5], 0.0, SES_03, 4820.0, &[0.5], 10, 1);
        assert_eq!(fp.probability, 0.0);
        assert_eq!(fp.quantiles, vec![(0.5, None)]);
        // A negative level is reached from above.
        let fp = first_passage(&[-2.0; 5], 0.0, SES_03, -5.0, &[0.5], 10, 1);
        assert!((fp.quantiles[0].1.expect("reached") - 2.5).abs() < 1e-12);
    }

    #[test]
    fn the_simulated_totals_spread_as_the_closed_form_says() {
        // The futures' totals at the horizon have the variance of the total's
        // interval: check the simulation against the closed form.
        let c = |j: usize| 0.3 * (1.0 + j as f64 * 0.1);
        let (mse, horizon, paths) = (4.0, 8, 40_000);
        let want = total_variance_by_covariance(mse, &c, horizon);
        let coefficients: Vec<f64> = (1..horizon).map(c).collect();
        use u_numflow::distributions::{Normal, Sample};
        let noise = Normal::new(0.0, mse.sqrt()).expect("valid");
        let mut rng = u_numflow::random::create_rng(9);
        let totals: Vec<f64> = (0..paths)
            .map(|_| {
                let eps: Vec<f64> = (0..horizon).map(|_| noise.sample(&mut rng)).collect();
                (0..horizon)
                    .map(|h| {
                        eps[h]
                            + (1..=h)
                                .map(|j| coefficients[j - 1] * eps[h - j])
                                .sum::<f64>()
                    })
                    .sum::<f64>()
            })
            .collect();
        let mean = totals.iter().sum::<f64>() / paths as f64;
        let var = totals.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / (paths - 1) as f64;
        assert!(
            (var / want - 1.0).abs() < 0.03,
            "simulated {var}, closed form {want}"
        );
    }

    #[test]
    fn the_same_seed_gives_the_same_times_and_the_median_sits_near_the_mean_path() {
        let a = first_passage(
            &[760.0; 12],
            400.0,
            SES_03,
            4820.0,
            &[0.1, 0.5, 0.9],
            4000,
            42,
        );
        let b = first_passage(
            &[760.0; 12],
            400.0,
            SES_03,
            4820.0,
            &[0.1, 0.5, 0.9],
            4000,
            42,
        );
        assert_eq!(a, b);
        let t = |q: usize| a.quantiles[q].1.expect("reached");
        assert!(t(0) < t(1) && t(1) < t(2));
        assert!(
            (t(1) - (6.0 + 260.0 / 760.0)).abs() < 0.15,
            "median {}",
            t(1)
        );
        assert_eq!(a.paths, 4000);
    }

    /// The times by direct convolution of the innovations, with the same draws.
    fn times_by_convolution(
        means: &[f64],
        mse: f64,
        w: InnovationWeights,
        threshold: f64,
        paths: usize,
        seed: u64,
    ) -> Vec<f64> {
        use u_numflow::distributions::{Normal, Sample};
        let noise = Normal::new(0.0, mse.sqrt()).expect("valid");
        let mut rng = u_numflow::random::create_rng(seed);
        (0..paths)
            .map(|_| {
                let eps: Vec<f64> = (0..means.len()).map(|_| noise.sample(&mut rng)).collect();
                let mut total = 0.0;
                for h in 0..means.len() {
                    let next = total
                        + means[h]
                        + eps[h]
                        + (1..=h).map(|j| w.c(j) * eps[h - j]).sum::<f64>();
                    if next >= threshold {
                        return h as f64 + (threshold - total) / (next - total);
                    }
                    total = next;
                }
                f64::INFINITY
            })
            .collect()
    }

    #[test]
    fn the_running_sums_give_the_times_the_convolution_does() {
        let hw = InnovationWeights {
            level: 0.3,
            slope: 0.03,
            season: 0.14,
            period: 4,
        };
        let means: Vec<f64> = (1..=40).map(|h| 10.0 + (h % 4) as f64).collect();
        let mut want = times_by_convolution(&means, 9.0, hw, 250.0, 300, 5);
        want.sort_by(f64::total_cmp);
        // (r − ½)/300 ranks to r exactly (r/300 · 300 may round up to r + 1).
        let qs: Vec<f64> = (1..=300).map(|r| (r as f64 - 0.5) / 300.0).collect();
        let got = first_passage(&means, 9.0, hw, 250.0, &qs, 300, 5);
        for ((_, t), w) in got.quantiles.iter().zip(&want) {
            match t {
                Some(t) => assert!((t - w).abs() < 1e-9, "{t} vs {w}"),
                None => assert!(w.is_infinite()),
            }
        }
    }
}
