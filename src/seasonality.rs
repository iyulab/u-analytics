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
    /// Autocorrelation at that lag — the strength of the periodicity.
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
        let Some(lag) = (lo..=hi)
            .filter(|&lag| lag < r.len())
            .max_by(|&a, &b| r[a].partial_cmp(&r[b]).expect("finite acf"))
        else {
            continue;
        };
        // The biased ACF shrinks by (n − lag)/n, which at short lengths keeps
        // even a perfect periodicity under the bound; the bound is compared
        // against the bias-corrected value, the hill is judged on the biased.
        let corrected = r[lag] * n as f64 / (n - lag) as f64;
        let is_hill_top = corrected > acf_threshold
            && r[lag] >= r[lag - 1]
            && r.get(lag + 1).is_none_or(|&next| r[lag] >= next);
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
            acf: r[lag],
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
