//! Seasonality: the periodogram of a univariate series and the estimation of
//! its dominant period.
//!
//! Smoothing with [`crate::smoothing::HoltWinters`] and any seasonal
//! decomposition need the period as an input. This module estimates it from
//! the data by the two-stage procedure of AutoPeriod (Vlachos, Yu & Castelli
//! 2005):
//!
//! 1. **Periodogram** — the series is linearly detrended, zero-padded to a
//!    power of two at least four times its length, and transformed with the
//!    FFT; the power at each frequency bin is `|X_k|² / n`. Padding
//!    interpolates the spectrum so that a period which does not divide the
//!    length still shows one peak at its true frequency rather than two
//!    half-height neighbours. A local maximum is a candidate when its power
//!    exceeds what the same
//!    values in random order produce: the series is permuted 100 times and
//!    the 99th percentile of the largest periodogram ordinate is the
//!    threshold (AutoPeriod's permutation test, deterministic here). A
//!    periodogram bin names a *frequency band*, not an integer period: bin
//!    `k` of a padded length `m` covers periods between `m/(k+1)` and
//!    `m/(k−1)`.
//! 2. **Autocorrelation refinement** — within each candidate's band the
//!    autocorrelation function is searched for its hill top: the lag with
//!    the largest ACF that is a local maximum and, after correcting the
//!    biased estimator's `(n − lag)/n` shrinkage, exceeds the 95% white-noise
//!    bound `1.96/√n`. That lag is the integer period. A candidate whose band
//!    has no such hill is discarded — it was spectral leakage or a harmonic
//!    without its own structure.
//!
//! The dominant period is the validated candidate with the largest ACF; every
//! validated candidate is returned so a caller can see harmonics and
//! competing cycles. The ACF is the biased estimator (see
//! [`crate::correlation::acf`]), which slightly favours shorter lags — the
//! fundamental over its multiples.
//!
//! # References
//! - Vlachos, M., Yu, P. & Castelli, V. (2005). "On periodicity detection and
//!   structural periodic similarity." *SIAM International Conference on Data
//!   Mining*, 449–460.
//! - Box, G. E. P. & Jenkins, G. M. (1976). *Time Series Analysis: Forecasting
//!   and Control*, ch. 2 (autocorrelation).

use u_numflow::fourier::rfft;
use u_numflow::random::{create_rng, shuffle};

use crate::correlation::acf;

/// Fewest observations for which a period is estimated: two cycles of the
/// longest admissible period (`n/2`) with a few bins to test against.
pub const MIN_OBSERVATIONS: usize = 8;

/// Permutations used to set the periodogram threshold.
const PERMUTATIONS: usize = 100;

/// The threshold is this quantile of the largest ordinate over permutations.
const PERMUTATION_QUANTILE: f64 = 0.99;

/// Seed of the permutation stream: the estimate is a deterministic function
/// of the series.
const PERMUTATION_SEED: u64 = 0x5EA5_0A11;

/// How many periodogram peaks are carried into the autocorrelation stage.
const MAX_CANDIDATES: usize = 3;

/// Periodogram of a linearly detrended, zero-padded series.
#[derive(Debug, Clone, PartialEq)]
pub struct Periodogram {
    /// Number of observations.
    pub n: usize,
    /// Padded transform length: the power of two at least `4 n`.
    pub padded_len: usize,
    /// Power `|X_k|² / n` at bins `k = 1..=padded_len/2`; `power[i]` is bin
    /// `i + 1`.
    pub power: Vec<f64>,
    /// Frequency of each bin in cycles per observation, `k / padded_len`.
    pub frequency: Vec<f64>,
}

impl Periodogram {
    /// Period, in observations, of bin `k` (1-based): `padded_len / k`.
    pub fn period_of_bin(&self, k: usize) -> f64 {
        self.padded_len as f64 / k as f64
    }

    /// Bin (1-based) of the largest ordinate, or `None` for an all-zero
    /// periodogram.
    pub fn peak_bin(&self) -> Option<usize> {
        let (i, &p) = self
            .power
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).expect("finite power"))?;
        (p > 0.0).then_some(i + 1)
    }
}

/// One periodicity that passed both stages.
#[derive(Debug, Clone, PartialEq)]
pub struct PeriodCandidate {
    /// Integer period, in observations.
    pub period: usize,
    /// Autocorrelation at that lag, with the `(n − lag)/n` bias of the
    /// estimator undone — the same value this candidate was selected and
    /// accepted on, so it is comparable against
    /// [`SeasonalityResult::acf_threshold`] and against other candidates.
    pub acf: f64,
    /// Periodogram bin (1-based) whose band produced this candidate.
    pub bin: usize,
    /// Periodogram power of that bin.
    pub power: f64,
    /// That bin's share of the total periodogram power.
    pub power_share: f64,
}

/// Result of period estimation.
#[derive(Debug, Clone, PartialEq)]
pub struct SeasonalityResult {
    /// The dominant period, or `None` when no periodicity passed both
    /// stages — a constant, a pure trend, white noise.
    pub period: Option<usize>,
    /// Every validated periodicity, strongest first. Empty when `period` is
    /// `None`.
    pub candidates: Vec<PeriodCandidate>,
    /// Number of observations.
    pub n: usize,
    /// The 95% white-noise bound on the ACF, `1.96 / √n`.
    pub acf_threshold: f64,
    /// Periodogram power a bin had to exceed to become a candidate: the 99th
    /// percentile of the largest ordinate over 100 permutations of the series.
    pub power_threshold: f64,
}

/// Periodogram of the series after linear detrending.
///
/// Returns `None` when the series has fewer than [`MIN_OBSERVATIONS`] points
/// or a non-finite value. A constant series returns a periodogram of zeros.
///
/// # Examples
///
/// ```
/// use u_analytics::seasonality::periodogram;
///
/// let series: Vec<f64> = (0..48).map(|t| (2.0 * std::f64::consts::PI * t as f64 / 12.0).sin()).collect();
/// let p = periodogram(&series).unwrap();
/// // Padded to 256: bin 256 / 12 ≈ 21.3, so the peak sits at bin 21.
/// let peak = p.peak_bin().unwrap();
/// assert_eq!(peak, 21);
/// assert!((p.period_of_bin(peak) - 12.0).abs() < 0.3);
/// ```
pub fn periodogram(series: &[f64]) -> Option<Periodogram> {
    let n = series.len();
    if n < MIN_OBSERVATIONS || series.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let detrended = detrend(series);
    let padded_len = padded_len(n);
    let power = padded_power(&detrended, padded_len);
    let frequency: Vec<f64> = (1..=padded_len / 2)
        .map(|k| k as f64 / padded_len as f64)
        .collect();
    Some(Periodogram {
        n,
        padded_len,
        power,
        frequency,
    })
}

/// Estimate the dominant period of the series.
///
/// Returns `None` when the input cannot be analysed (fewer than
/// [`MIN_OBSERVATIONS`] points, or a non-finite value). Otherwise the result
/// says explicitly whether a period was found: [`SeasonalityResult::period`]
/// is `None` for a series with no periodicity that survives both the
/// periodogram and the autocorrelation stage.
///
/// Only periods from 2 to `n / 2` are admissible — anything longer has not
/// been observed twice.
///
/// # Examples
///
/// ```
/// use u_analytics::seasonality::estimate_period;
///
/// // A sawtooth of period 7 over 40 observations.
/// let series: Vec<f64> = (0..40).map(|i| (i % 7) as f64).collect();
/// let result = estimate_period(&series).unwrap();
/// assert_eq!(result.period, Some(7));
///
/// // A straight line has no period.
/// let line: Vec<f64> = (0..40).map(|i| 2.0 * i as f64).collect();
/// assert_eq!(estimate_period(&line).unwrap().period, None);
/// ```
pub fn estimate_period(series: &[f64]) -> Option<SeasonalityResult> {
    let pg = periodogram(series)?;
    let n = pg.n;
    let acf_threshold = 1.96 / (n as f64).sqrt();
    let detrended = detrend(series);
    let total: f64 = pg.power.iter().sum();
    let power_threshold = if total > 0.0 && total.is_finite() {
        permutation_threshold(&detrended)
    } else {
        f64::INFINITY
    };
    let no_period = || SeasonalityResult {
        period: None,
        candidates: Vec::new(),
        n,
        acf_threshold,
        power_threshold,
    };
    if !power_threshold.is_finite() {
        return Some(no_period());
    }

    // Stage 1: local maxima of the periodogram above the permutation
    // threshold, strongest first, at most MAX_CANDIDATES.
    let m = pg.padded_len;
    let mut peaks: Vec<(usize, f64)> = pg
        .power
        .iter()
        .enumerate()
        .filter(|&(i, &p)| {
            p > power_threshold
                && (i == 0 || p >= pg.power[i - 1])
                && pg.power.get(i + 1).is_none_or(|&next| p >= next)
        })
        .map(|(i, &p)| (i + 1, p))
        .collect();
    peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).expect("finite power"));
    peaks.truncate(MAX_CANDIDATES);
    if peaks.is_empty() {
        return Some(no_period());
    }

    // Stage 2: the ACF hill inside each candidate's frequency band.
    let Some(acf_result) = acf(&detrended, n / 2) else {
        return Some(no_period());
    };
    let r = &acf_result.acf;

    let mut candidates: Vec<PeriodCandidate> = Vec::new();
    for (bin, power) in peaks {
        let lo = (m / (bin + 1)).max(2);
        let hi = if bin == 1 {
            n / 2
        } else {
            (m as f64 / (bin - 1) as f64).ceil() as usize
        }
        .min(n / 2);
        if lo > hi {
            continue;
        }
        // The biased ACF shrinks by (n − lag)/n, and that shrinkage is
        // monotone in the lag -- so on a hill spanning several lags it tilts
        // every comparison toward the shorter one, and the estimate lands
        // below the true period whenever the series is not a whole number of
        // cycles long. Undoing the bias is exactly what removes that tilt, so
        // the corrected value decides all three questions here: which lag in
        // the band, whether it is a hill top, and whether it clears the bound.
        let corrected_at = |lag: usize| r[lag] * n as f64 / (n - lag) as f64;
        let Some(lag) = (lo..=hi).filter(|&lag| lag < r.len()).max_by(|&a, &b| {
            corrected_at(a)
                .partial_cmp(&corrected_at(b))
                .expect("finite acf")
        }) else {
            continue;
        };
        let corrected = corrected_at(lag);
        let is_hill_top = corrected > acf_threshold
            && corrected >= corrected_at(lag - 1)
            && (lag + 1 >= r.len() || corrected >= corrected_at(lag + 1));
        if !is_hill_top {
            continue;
        }
        if let Some(existing) = candidates.iter_mut().find(|c| c.period == lag) {
            // Two bins straddling one period: keep the stronger bin's numbers.
            if power > existing.power {
                existing.bin = bin;
                existing.power = power;
                existing.power_share = power / total;
            }
            continue;
        }
        candidates.push(PeriodCandidate {
            period: lag,
            acf: corrected,
            bin,
            power,
            power_share: power / total,
        });
    }

    candidates.sort_by(|a, b| b.acf.partial_cmp(&a.acf).expect("finite acf"));
    let period = candidates.first().map(|c| c.period);
    Some(SeasonalityResult {
        period,
        candidates,
        n,
        acf_threshold,
        power_threshold,
    })
}

/// Residuals of the least-squares line through the series.
fn detrend(series: &[f64]) -> Vec<f64> {
    let n = series.len() as f64;
    let t_mean = (n - 1.0) / 2.0;
    let y_mean = series.iter().sum::<f64>() / n;
    let (mut sxy, mut sxx) = (0.0, 0.0);
    for (i, &y) in series.iter().enumerate() {
        let dt = i as f64 - t_mean;
        sxy += dt * (y - y_mean);
        sxx += dt * dt;
    }
    let slope = if sxx > 0.0 { sxy / sxx } else { 0.0 };
    series
        .iter()
        .enumerate()
        .map(|(i, &y)| y - y_mean - slope * (i as f64 - t_mean))
        .collect()
}

/// Transform length: the power of two at least four times `n`.
fn padded_len(n: usize) -> usize {
    (4 * n).next_power_of_two()
}

/// `|X_k|² / n` for bins `1..=m/2` of the series zero-padded to length `m`.
fn padded_power(detrended: &[f64], m: usize) -> Vec<f64> {
    let n = detrended.len();
    let mut padded = vec![0.0; m];
    padded[..n].copy_from_slice(detrended);
    rfft(&padded)[1..=m / 2]
        .iter()
        .map(|z| z.norm_sqr() / n as f64)
        .collect()
}

/// Largest periodogram ordinate the values produce in random order, at the
/// 99th percentile over 100 seeded permutations. Power above this is
/// structure the ordering carries, not the marginal distribution.
fn permutation_threshold(detrended: &[f64]) -> f64 {
    let m = padded_len(detrended.len());
    let mut rng = create_rng(PERMUTATION_SEED);
    let mut shuffled = detrended.to_vec();
    let mut maxima: Vec<f64> = (0..PERMUTATIONS)
        .map(|_| {
            shuffle(&mut shuffled, &mut rng);
            padded_power(&shuffled, m)
                .into_iter()
                .fold(0.0f64, f64::max)
        })
        .collect();
    maxima.sort_by(|a, b| a.partial_cmp(b).expect("finite power"));
    let idx =
        ((PERMUTATIONS as f64 * PERMUTATION_QUANTILE).ceil() as usize).clamp(1, PERMUTATIONS) - 1;
    maxima[idx]
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// Deterministic uniform noise in [-1, 1) from a linear congruential
    /// generator, so the tests do not depend on a random crate.
    fn noise(seed: u64, n: usize) -> Vec<f64> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
            })
            .collect()
    }

    #[test]
    fn sawtooth_of_period_seven_is_found() {
        // The case a well-known library answers with "none": 40 rows of i % 7.
        let series: Vec<f64> = (0..40).map(|i| (i % 7) as f64).collect();
        let r = estimate_period(&series).unwrap();
        assert_eq!(r.period, Some(7), "{r:?}");
        assert!(r.candidates[0].acf > 0.5);
        assert!(r.candidates[0].power > r.power_threshold);
    }

    #[test]
    fn sine_with_noise_is_found_and_the_period_is_exact() {
        let n = 120;
        let series: Vec<f64> = noise(7, n)
            .iter()
            .enumerate()
            .map(|(t, e)| (2.0 * PI * t as f64 / 12.0).sin() + 0.3 * e)
            .collect();
        let r = estimate_period(&series).unwrap();
        assert_eq!(r.period, Some(12), "{r:?}");
    }

    #[test]
    fn square_wave_reports_the_fundamental_first() {
        let series: Vec<f64> = (0..100)
            .map(|t| if (t / 5) % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let r = estimate_period(&series).unwrap();
        assert_eq!(r.period, Some(10), "{r:?}");
    }

    #[test]
    fn period_survives_a_linear_trend() {
        let series: Vec<f64> = (0..60)
            .map(|t| 0.5 * t as f64 + 3.0 * (2.0 * PI * t as f64 / 6.0).cos())
            .collect();
        let r = estimate_period(&series).unwrap();
        assert_eq!(r.period, Some(6), "{r:?}");
    }

    #[test]
    fn line_constant_and_white_noise_have_no_period() {
        let line: Vec<f64> = (0..50).map(|t| 3.0 - 0.25 * t as f64).collect();
        assert_eq!(estimate_period(&line).unwrap().period, None);

        let constant = vec![4.2; 30];
        let r = estimate_period(&constant).unwrap();
        assert_eq!(r.period, None);
        assert!(r.candidates.is_empty());

        // The permutation threshold has a 1% false-positive rate per series;
        // with fixed seeds the answer is a fixed fact.
        let r = estimate_period(&noise(11, 200)).unwrap();
        assert_eq!(r.period, None, "{r:?}");
    }

    #[test]
    fn every_period_from_2_to_20_is_recovered_from_sines_and_sawtooths() {
        for period in 2..=20usize {
            for cycles in [4usize, 7, 13] {
                let n = period * cycles;
                if n < 16 {
                    // Eight or twelve points rarely beat the permutation
                    // threshold: too few orderings differ from the observed.
                    continue;
                }
                let sine: Vec<f64> = noise(period as u64 * 31 + cycles as u64, n)
                    .iter()
                    .enumerate()
                    .map(|(t, e)| (2.0 * PI * t as f64 / period as f64 + 0.7).sin() + 0.2 * e)
                    .collect();
                let r = estimate_period(&sine).unwrap();
                assert_eq!(r.period, Some(period), "sine period {period}, n {n}: {r:?}");

                let saw: Vec<f64> = (0..n + 3).map(|t| (t % period) as f64).collect();
                let r = estimate_period(&saw).unwrap();
                assert_eq!(
                    r.period,
                    Some(period),
                    "sawtooth period {period}, n {}: {r:?}",
                    n + 3
                );
            }
        }
    }

    /// Reported by a consumer: a 63-point period over 300 observations --
    /// 4.76 cycles -- came back as 60.
    ///
    /// The estimate has to hold when the series is not a whole number of
    /// cycles long, which is the ordinary case for real data. The sweep below
    /// is the general statement; this is the series as reported.
    #[test]
    fn a_period_is_exact_when_the_series_is_not_a_whole_number_of_cycles() {
        fn jitter(i: usize) -> f64 {
            (((i as f64) * 12.9898).sin() * 43758.5453) % 1.0
        }
        let spikes = [50usize, 120, 200, 250];
        let reported: Vec<f64> = (0..300)
            .map(|i| {
                25.0 + 2.2 * (2.0 * PI * i as f64 / 63.0).sin()
                    + 0.4 * jitter(i)
                    + if spikes.contains(&i) { 12.0 } else { 0.0 }
            })
            .collect();
        let r = estimate_period(&reported).unwrap();
        assert_eq!(r.period, Some(63), "{r:?}");

        // The same series at whole cycles was never in doubt; the two together
        // say the fractional tail is what used to move the answer.
        let whole: Vec<f64> = (0..315)
            .map(|i| 25.0 + 2.2 * (2.0 * PI * i as f64 / 63.0).sin() + 0.4 * jitter(i))
            .collect();
        assert_eq!(estimate_period(&whole).unwrap().period, Some(63));

        for period in [11usize, 17, 23, 31, 47, 63] {
            let n = ((period as f64) * 4.76).round() as usize;
            let sine: Vec<f64> = noise(period as u64 * 17 + 3, n)
                .iter()
                .enumerate()
                .map(|(t, e)| (2.0 * PI * t as f64 / period as f64 + 0.7).sin() + 0.2 * e)
                .collect();
            let r = estimate_period(&sine).unwrap();
            assert_eq!(r.period, Some(period), "period {period}, n {n}: {r:?}");
        }
    }

    /// A candidate's `acf` is the value it was judged on, so a consumer
    /// comparing it against `acf_threshold` reaches the crate's own verdict.
    #[test]
    fn a_candidate_reports_the_autocorrelation_it_was_accepted_on() {
        let series: Vec<f64> = (0..137)
            .map(|t| (2.0 * PI * t as f64 / 19.0).sin())
            .collect();
        let r = estimate_period(&series).unwrap();
        assert_eq!(r.period, Some(19));
        for c in &r.candidates {
            assert!(
                c.acf > r.acf_threshold,
                "candidate {} reports {} against a bound of {}",
                c.period,
                c.acf,
                r.acf_threshold
            );
        }
    }

    #[test]
    fn short_or_non_finite_input_cannot_be_analysed() {
        assert!(estimate_period(&[1.0; 7]).is_none());
        assert!(periodogram(&[1.0; 7]).is_none());
        let mut series: Vec<f64> = (0..20).map(|i| (i % 4) as f64).collect();
        series[3] = f64::NAN;
        assert!(estimate_period(&series).is_none());
    }

    #[test]
    fn periodogram_bins_and_frequencies_line_up() {
        let series: Vec<f64> = (0..48)
            .map(|t| (2.0 * PI * t as f64 / 12.0).sin())
            .collect();
        let p = periodogram(&series).unwrap();
        assert_eq!(p.padded_len, 256);
        assert_eq!(p.power.len(), 128);
        assert_eq!(p.frequency.len(), 128);
        assert!((p.frequency[20] - 21.0 / 256.0).abs() < 1e-15);
        let peak = p.peak_bin().unwrap();
        assert_eq!(peak, 21);
        assert!((p.period_of_bin(peak) - 12.0).abs() < 0.3);
        // The mainlobe peak carries the sine's energy: n/4 · amplitude² = 12
        // before detrending, a little less after it.
        assert!(
            p.power[peak - 1] > 10.5 && p.power[peak - 1] <= 12.0,
            "{}",
            p.power[peak - 1]
        );
    }
}

/// The periodogram against NumPy (linear detrend by least squares, zero padding
/// to the power of two at least 4 n, `|rfft|^2 / n` at bins 1..=m/2), and the
/// dominant period against series generated with a known one -- a sine of
/// period 12, one of period 7 that does not divide the length of 100, one of
/// period 24 on a trend -- and against pure noise, which has none. Seeded
/// NumPy generator.
#[cfg(test)]
// The generated samples are data; one of them happens to be near 1/pi.
#[allow(clippy::approx_constant)]
mod numpy_reference {
    use super::*;

    const PG_SERIES: [f64; 30] = [
        4.872, 5.961, 5.825, 5.144, 4.146, 4.708, 5.075, 5.872, 6.02, 5.134, 4.962, 4.284, 4.897,
        6.327, 6.276, 5.223, 4.726, 4.67, 5.472, 6.109, 6.148, 5.732, 4.675, 4.688, 5.575, 6.406,
        6.708, 5.811, 4.756, 4.49,
    ];
    const PG_POWER: [f64; 64] = [
        0.000284033512117391,
        0.00553405987724846,
        0.0302722144153745,
        0.0831085833321147,
        0.133085087378873,
        0.123289480834556,
        0.0533851897200275,
        0.01637782366091,
        0.087488531746546,
        0.190533285812008,
        0.175364989155614,
        0.0614983006464574,
        0.0775763298722834,
        0.330671796059938,
        0.537579817078249,
        0.350336825311503,
        0.0246598611773083,
        0.474810160665647,
        2.31103325540996,
        4.83383291955508,
        6.33862799215007,
        5.67039790499692,
        3.37456537314838,
        1.12427168848378,
        0.0891876510214037,
        0.0814269240692162,
        0.235139118197625,
        0.131491953209051,
        0.0138091767541502,
        0.140722132759862,
        0.354827745604986,
        0.371281570269601,
        0.200282895493422,
        0.0741310942371521,
        0.086678695001399,
        0.116160506813996,
        0.0716411804794686,
        0.0245416422268119,
        0.0531807286271523,
        0.102252619738399,
        0.0838655381832552,
        0.0281521341166196,
        0.0335787253260554,
        0.110927863849889,
        0.169919048372977,
        0.149793848613065,
        0.0851619721280352,
        0.0362266166967703,
        0.0194195967158166,
        0.0241675506804475,
        0.0424368069304849,
        0.0560057396007925,
        0.0401962394364771,
        0.00919485591496622,
        0.0165631435231124,
        0.0766705537531582,
        0.12478599870531,
        0.100432590523266,
        0.0383177907494266,
        0.0192509305437197,
        0.0504432405513664,
        0.0630009356163572,
        0.0282428274414305,
        0.00221180307769972,
    ];
    const P12: [f64; 120] = [
        -0.646, 0.989, 0.558, 1.047, 0.831, 0.359, 0.402, -0.871, -1.05, -1.197, -0.95, -0.373,
        -0.499, 0.521, 0.755, 1.054, 0.744, 1.032, -0.269, -0.377, -0.816, -0.701, -1.722, -0.523,
        -0.196, 1.257, 0.572, 0.928, 0.32, 0.553, -0.575, -0.699, -0.782, -1.035, -0.956, -0.929,
        0.084, 0.964, 1.454, 0.834, 0.347, 0.275, 0.001, -0.06, -1.261, -0.974, -1.11, -0.107,
        -0.485, 0.75, 1.173, 0.926, 0.917, 0.431, -0.318, -0.357, -0.72, -1.194, -0.913, -0.324,
        0.038, 0.502, 0.616, 0.972, 1.212, 0.708, -0.438, -0.696, -1.184, -0.46, -0.18, -0.604,
        0.112, 0.374, 1.036, 1.533, 0.869, 0.928, -0.06, -0.609, -0.397, -1.135, -1.117, -0.784,
        -0.179, 0.484, 1.204, 1.122, 1.094, 0.359, -0.081, -0.472, -0.45, -0.628, -0.729, -0.764,
        0.158, 0.663, 1.035, 1.097, 0.704, 0.632, 0.296, -0.587, -0.489, -1.33, -0.983, -0.562,
        -0.068, 0.927, 1.332, 1.139, 0.99, 0.623, -0.464, -0.444, -0.937, -1.257, -0.309, -0.191,
    ];
    const P7: [f64; 100] = [
        -0.405, 1.024, 2.014, 1.316, -0.877, -2.408, -1.315, 0.387, 2.192, 1.706, -0.547, -0.904,
        -2.012, -1.411, 0.729, 1.961, 2.296, 0.007, -1.402, -2.969, -0.806, 0.026, 2.031, 1.991,
        0.952, -0.252, -2.209, -2.173, -0.284, 1.642, 2.091, 1.239, -1.528, -2.441, -2.426, 1.019,
        2.494, 1.57, 1.408, -0.864, -1.606, -0.917, -0.686, 1.56, 2.092, 1.082, -0.255, -1.492,
        -1.347, -0.327, 1.32, 2.703, 0.523, -2.251, -1.55, -1.012, -0.555, 1.509, 0.575, 1.387,
        -1.439, -1.585, -1.999, 0.493, 1.583, 3.155, 0.222, -0.852, -0.946, -1.859, 0.365, 0.714,
        1.485, 2.541, -0.938, -2.451, -0.513, -0.993, 1.238, 1.649, 1.857, -0.15, -2.286, -1.927,
        0.366, 1.54, 2.751, 0.67, -1.793, -1.811, -2.868, -0.376, 1.697, 2.691, 1.455, -0.128,
        -1.157, -1.959, -0.437, 1.787,
    ];
    const P24: [f64; 240] = [
        -0.028, 0.609, 1.787, 3.199, 2.769, 2.915, 2.48, 3.235, 4.652, 2.793, 2.392, 0.66, 0.923,
        0.008, -0.341, -0.667, -1.32, -2.107, -2.182, -2.668, -1.531, -1.029, -0.357, 0.924, 1.311,
        1.784, 2.404, 3.154, 3.956, 4.717, 4.096, 5.754, 4.309, 3.522, 3.076, 2.995, 1.66, 0.922,
        0.645, -0.043, -0.087, -0.652, -0.581, -0.576, 0.028, -0.035, 1.385, 1.435, 2.499, 3.093,
        3.761, 4.778, 5.178, 5.009, 5.745, 5.81, 6.222, 4.029, 3.829, 3.966, 2.815, 1.903, 2.21,
        1.727, 0.01, -0.152, -0.285, 0.326, -0.05, 1.523, 1.86, 2.699, 4.633, 3.971, 4.718, 5.651,
        6.12, 7.712, 7.003, 7.008, 7.167, 6.163, 5.666, 4.697, 3.391, 3.98, 2.201, 2.345, 1.461,
        2.003, 1.194, 1.89, 2.753, 2.393, 3.207, 3.877, 4.394, 6.182, 7.156, 7.105, 7.659, 7.796,
        6.801, 7.941, 7.674, 7.449, 6.244, 5.446, 6.2, 3.994, 4.43, 3.946, 3.43, 2.693, 2.847,
        1.732, 3.037, 3.464, 4.177, 3.99, 5.991, 6.142, 7.402, 8.317, 8.725, 8.623, 9.919, 9.348,
        8.86, 8.834, 8.686, 8.241, 6.759, 5.253, 5.143, 4.807, 3.718, 3.204, 4.262, 3.956, 4.669,
        4.538, 5.857, 6.221, 6.643, 7.757, 9.384, 10.244, 9.055, 10.805, 10.256, 10.614, 10.293,
        10.565, 10.015, 9.31, 7.56, 6.553, 6.546, 5.058, 5.211, 5.108, 5.024, 5.025, 5.484, 6.953,
        7.197, 7.658, 8.36, 10.328, 10.93, 11.583, 11.354, 12.083, 12.908, 12.226, 11.578, 11.467,
        9.58, 9.983, 8.632, 7.656, 7.346, 7.162, 6.621, 5.586, 6.699, 5.726, 6.188, 6.726, 7.194,
        9.107, 10.245, 9.727, 11.195, 11.672, 12.642, 13.356, 13.334, 12.319, 11.958, 11.449,
        11.538, 11.463, 10.018, 9.416, 9.472, 8.601, 7.43, 7.247, 7.852, 7.077, 7.99, 9.097, 8.928,
        9.882, 11.225, 11.207, 12.784, 13.327, 13.394, 13.874, 14.363, 13.898, 13.256, 12.659,
        13.573, 12.496, 12.22, 10.514, 9.669, 9.548, 9.926, 8.647, 8.947, 9.164, 8.904, 10.123,
        10.426, 11.634,
    ];
    const NOISE: [f64; 120] = [
        0.511, 1.186, -0.731, -0.094, 0.062, -0.891, 1.251, -1.361, 0.273, -0.62, -0.567, 0.373,
        1.054, 1.262, 1.618, -1.328, -0.023, 0.04, 0.273, 0.835, -1.165, -0.103, 1.077, 0.782,
        0.185, -0.316, -0.429, -1.648, 0.497, 0.477, 1.667, -0.206, -0.936, 0.578, 0.02, 1.225,
        0.542, -0.442, -0.157, 0.397, 0.58, 0.897, 1.913, 0.751, 0.478, -0.977, -0.226, 0.75,
        -1.707, 0.961, -0.863, 0.807, 0.594, 0.15, -0.712, -0.87, 1.278, -1.761, -0.247, 0.622,
        -1.098, -0.95, -2.189, 0.41, -0.227, 0.848, 0.332, -1.668, -1.059, 0.356, 0.257, -0.343,
        1.902, 0.674, 2.441, 0.564, 1.348, -0.64, 0.909, -0.767, -1.069, 1.354, 0.339, -0.123,
        0.663, 1.073, -1.47, 0.433, -0.865, -1.236, 0.53, -0.353, -0.699, -0.434, 1.476, 0.381,
        0.302, 1.587, -0.238, 3.151, -0.864, 1.442, 2.826, -0.295, -0.208, 1.768, -1.3, 0.474,
        0.368, 0.265, 0.209, 0.803, 0.38, -0.744, 0.217, 0.417, 1.981, 0.881, 0.286, -1.222,
    ];

    #[test]
    fn the_periodogram_reproduces_numpy() {
        let pg = periodogram(&PG_SERIES).expect("a valid series");
        assert_eq!(pg.padded_len, 128);
        assert_eq!(pg.power.len(), PG_POWER.len());
        let scale = PG_POWER.iter().cloned().fold(0.0, f64::max);
        for (k, (got, want)) in pg.power.iter().zip(PG_POWER).enumerate() {
            assert!(
                (got - want).abs() <= 1e-9 * scale,
                "bin {}: got {got}, NumPy {want}",
                k + 1
            );
        }
    }

    #[test]
    fn the_dominant_period_is_the_one_the_series_was_made_with() {
        for (series, want, what) in [
            (&P12[..], Some(12), "period 12"),
            (&P7[..], Some(7), "period 7 in 100 points"),
            (&P24[..], Some(24), "period 24 on a trend"),
            (&NOISE[..], None, "white noise"),
        ] {
            let found = estimate_period(series).expect("a valid series");
            assert_eq!(found.period, want, "{what}");
        }
    }
}
