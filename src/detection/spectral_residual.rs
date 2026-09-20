//! Spectral residual saliency for one-shot anomaly scoring of a series.
//!
//! The spectral residual transform (Hou & Zhang 2007, brought to time series
//! by Ren et al. 2019) takes the log amplitude spectrum of the series,
//! subtracts its local average, and transforms back with the original phase.
//! What is left — the saliency map — is large exactly where the series has a
//! feature its spectrum cannot explain as regular structure: a spike, a
//! step, a dropout. No model is trained and no period is assumed.
//!
//! Each point is scored relative to the saliency of the points before it,
//! `score = (S_i − mean(S over the judgement window)) / mean(...)`; a point
//! is an anomaly when its score exceeds the threshold (Ren et al. use τ = 3)
//! and, so that a bump of a few thousandths on a flat series is not one, its
//! value stands at least `min_zscore` standard deviations from the level of
//! the window before it.
//!
//! The result also carries an **expected value** — the series with the
//! anomalies replaced by their neighbours, reconstructed from its
//! low-frequency components — and a band around it whose half-width is the
//! normal quantile for `sensitivity` percent coverage times the robust
//! spread of the residuals. The band is information for a chart, not the
//! anomaly decision, which is the saliency score.
//!
//! # References
//! - Ren, H., Xu, B., Wang, Y., Yi, C., Huang, C., Kou, X., Xing, T., Yang,
//!   M., Tong, J. & Zhang, Q. (2019). "Time-Series Anomaly Detection Service
//!   at Microsoft." *KDD 2019*, 3009–3017.
//! - Hou, X. & Zhang, L. (2007). "Saliency Detection: A Spectral Residual
//!   Approach." *CVPR 2007*.

use u_numflow::fourier::{fft, ifft, Complex};
use u_numflow::special::inverse_normal_cdf;
use u_numflow::stats::median;

/// Fewest observations the transform is applied to at once.
pub const MIN_OBSERVATIONS: usize = 12;

/// Points appended before the transform to soften its boundary effect, and
/// the number of preceding points their slope is estimated from (κ = m = 5
/// in Ren et al.).
const EXTENSION: usize = 5;

/// Why a [`SpectralResidual`] run was refused.
///
/// One condition, not the rulebook. A consumer has to tell its user which
/// setting to change, and a refusal that lists every rule leaves it
/// re-validating the options the crate already checks -- which is how the
/// same rules end up written twice, in two places that can drift.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SpectralResidualError {
    /// An option is outside its domain.
    OptionOutOfRange {
        /// The option's name, spelled as the builder and the wire formats
        /// spell it (`threshold`, `sensitivity`, ...).
        option: &'static str,
        /// What that option has to satisfy, for a message a consumer can show.
        requirement: &'static str,
    },
    /// Fewer observations than the transform is applied to at once.
    TooFewObservations {
        /// [`MIN_OBSERVATIONS`].
        needed: usize,
        /// How many were given.
        got: usize,
    },
    /// The series holds a NaN or an infinity.
    ValueNotFinite {
        /// Position of the first such value.
        index: usize,
    },
}

impl core::fmt::Display for SpectralResidualError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            SpectralResidualError::OptionOutOfRange {
                option,
                requirement,
            } => write!(f, "{option} must be {requirement}"),
            SpectralResidualError::TooFewObservations { needed, got } => {
                write!(f, "needs at least {needed} observations, got {got}")
            }
            SpectralResidualError::ValueNotFinite { index } => {
                write!(f, "data[{index}] is not a finite number")
            }
        }
    }
}

impl std::error::Error for SpectralResidualError {}

/// Spectral residual anomaly scorer.
///
/// Built with [`SpectralResidual::new`] (the defaults of Ren et al. 2019) and
/// adjusted with the `with_*` methods; run with [`SpectralResidual::analyze`].
#[derive(Debug, Clone, PartialEq)]
pub struct SpectralResidual {
    /// Width of the moving average applied to the log amplitude spectrum
    /// (`q` in Ren et al.; default 3).
    pub averaging_window: usize,
    /// How many preceding saliency values a point is compared against
    /// (`z`; default 40).
    pub judgement_window: usize,
    /// Score above which a point is an anomaly (`τ`; default 3).
    pub threshold: f64,
    /// Minimum `|value − mean| / std` of the point against the judgement
    /// window before it, for it to be an anomaly (default 1.5), so that
    /// saliency alone does not flag noise on a flat series. A window with no
    /// spread passes any departure from its level.
    pub min_zscore: f64,
    /// Coverage of the band around the expected value, in percent (default
    /// 70): its half-width is the normal quantile for that two-sided coverage
    /// times the robust spread of the non-anomalous residuals.
    pub sensitivity: f64,
    /// Process the series in consecutive batches of this many points instead
    /// of at once (default `None`). Long series with slow drift are scored
    /// against their local context this way; the last batch absorbs a short
    /// remainder.
    pub batch_size: Option<usize>,
}

impl Default for SpectralResidual {
    fn default() -> Self {
        Self::new()
    }
}

impl SpectralResidual {
    /// The defaults of Ren et al. (2019): q = 3, z = 40, τ = 3, z-score gate
    /// 1.5, 70% band, no batching.
    pub fn new() -> Self {
        Self {
            averaging_window: 3,
            judgement_window: 40,
            threshold: 3.0,
            min_zscore: 1.5,
            sensitivity: 70.0,
            batch_size: None,
        }
    }

    /// Sets the spectral averaging window (≥ 1).
    pub fn with_averaging_window(mut self, q: usize) -> Self {
        self.averaging_window = q;
        self
    }

    /// Sets the judgement window (≥ 1).
    pub fn with_judgement_window(mut self, z: usize) -> Self {
        self.judgement_window = z;
        self
    }

    /// Sets the anomaly threshold on the score (> 0).
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.threshold = threshold;
        self
    }

    /// Sets the z-score gate (≥ 0; 0 disables it).
    pub fn with_min_zscore(mut self, min_zscore: f64) -> Self {
        self.min_zscore = min_zscore;
        self
    }

    /// Sets the band coverage in percent (0 < s < 100).
    pub fn with_sensitivity(mut self, sensitivity: f64) -> Self {
        self.sensitivity = sensitivity;
        self
    }

    /// Sets the batch size (≥ [`MIN_OBSERVATIONS`]), or `None` for one batch.
    pub fn with_batch_size(mut self, batch_size: Option<usize>) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// The first option that is outside its domain, in a fixed order.
    fn check_options(&self) -> Result<(), SpectralResidualError> {
        let out_of_range = |option, requirement| {
            Err(SpectralResidualError::OptionOutOfRange {
                option,
                requirement,
            })
        };
        if self.averaging_window < 1 {
            return out_of_range("averaging_window", ">= 1");
        }
        if self.judgement_window < 1 {
            return out_of_range("judgement_window", ">= 1");
        }
        if !self.threshold.is_finite() || self.threshold <= 0.0 {
            return out_of_range("threshold", "a finite number > 0");
        }
        if !self.min_zscore.is_finite() || self.min_zscore < 0.0 {
            return out_of_range("min_zscore", "a finite number >= 0");
        }
        if !self.sensitivity.is_finite() || self.sensitivity <= 0.0 || self.sensitivity >= 100.0 {
            return out_of_range("sensitivity", "a finite number strictly between 0 and 100");
        }
        if self.batch_size.is_some_and(|b| b < MIN_OBSERVATIONS) {
            return out_of_range("batch_size", "unset, or at least MIN_OBSERVATIONS");
        }
        Ok(())
    }

    /// Scores every point of the series.
    ///
    /// The result has one [`SrPoint`] per input point, in order.
    ///
    /// # Errors
    ///
    /// Returns the **one** condition that was not met — the option that is out
    /// of its domain, the shortfall in observations, or the position of the
    /// first non-finite value — rather than a refusal a caller has to match
    /// against the whole rulebook to interpret.
    ///
    /// # Examples
    ///
    /// ```
    /// use u_analytics::detection::{SpectralResidual, SpectralResidualError};
    ///
    /// let mut series: Vec<f64> = (0..60).map(|t| (t as f64 * 0.3).sin()).collect();
    /// series[40] += 4.0; // a spike
    /// let points = SpectralResidual::new().analyze(&series).unwrap();
    /// assert!(points[40].is_anomaly);
    /// assert!(points[40].score > points[20].score);
    ///
    /// // A refusal names the option it is about.
    /// let refused = SpectralResidual::new().with_threshold(0.0).analyze(&series);
    /// assert_eq!(
    ///     refused,
    ///     Err(SpectralResidualError::OptionOutOfRange {
    ///         option: "threshold",
    ///         requirement: "a finite number > 0",
    ///     })
    /// );
    /// ```
    pub fn analyze(&self, series: &[f64]) -> Result<Vec<SrPoint>, SpectralResidualError> {
        self.check_options()?;
        if series.len() < MIN_OBSERVATIONS {
            return Err(SpectralResidualError::TooFewObservations {
                needed: MIN_OBSERVATIONS,
                got: series.len(),
            });
        }
        if let Some(index) = series.iter().position(|v| !v.is_finite()) {
            return Err(SpectralResidualError::ValueNotFinite { index });
        }
        let n = series.len();
        let batch = self.batch_size.unwrap_or(n).min(n);
        let mut points = Vec::with_capacity(n);
        let mut start = 0;
        while start < n {
            // The last batch absorbs a remainder shorter than a batch.
            let end = if n - start < 2 * batch {
                n
            } else {
                start + batch
            };
            points.extend(self.analyze_batch(&series[start..end], start));
            start = end;
        }
        Ok(points)
    }

    fn analyze_batch(&self, data: &[f64], offset: usize) -> Vec<SrPoint> {
        let n = data.len();
        let saliency = saliency_map(data, self.averaging_window);

        // Score against the preceding judgement window (the point included,
        // as in Ren et al.'s implementation), relative to that average.
        let scores: Vec<f64> = (0..n)
            .map(|i| {
                let from = i.saturating_sub(self.judgement_window - 1);
                let avg = saliency[from..=i].iter().sum::<f64>() / (i + 1 - from) as f64;
                let avg = avg.max(1e-8);
                ((saliency[i] - avg) / avg).max(0.0)
            })
            .collect();

        // The z-score gate is local: the point against the mean and spread of
        // the judgement window before it, so a step is judged against the
        // level it departs from, not against a mean that already straddles it.
        let flagged: Vec<bool> = (0..n)
            .map(|i| {
                if scores[i] <= self.threshold {
                    return false;
                }
                if self.min_zscore == 0.0 || i == 0 {
                    return true;
                }
                let from = i.saturating_sub(self.judgement_window);
                let window = &data[from..i];
                let mean = window.iter().sum::<f64>() / window.len() as f64;
                let std = (window.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
                    / window.len() as f64)
                    .sqrt();
                if std > 0.0 {
                    (data[i] - mean).abs() / std >= self.min_zscore
                } else {
                    data[i] != mean
                }
            })
            .collect();

        let expected = expected_values(data, &flagged);
        let residuals: Vec<f64> = (0..n)
            .filter(|&i| !flagged[i])
            .map(|i| (data[i] - expected[i]).abs())
            .collect();
        // Robust spread: 1.4826 · median absolute residual (the MAD of a
        // centred residual), then the two-sided normal quantile for the
        // requested coverage.
        let spread = median(&residuals).unwrap_or(0.0) * 1.4826;
        let z = inverse_normal_cdf(0.5 + self.sensitivity / 200.0);
        let margin = z * spread;

        (0..n)
            .map(|i| SrPoint {
                index: offset + i,
                value: data[i],
                saliency: saliency[i],
                score: scores[i],
                expected: expected[i],
                lower: expected[i] - margin,
                upper: expected[i] + margin,
                is_anomaly: flagged[i],
            })
            .collect()
    }
}

/// One scored point.
#[derive(Debug, Clone, PartialEq)]
pub struct SrPoint {
    /// Position in the input series.
    pub index: usize,
    /// The observed value.
    pub value: f64,
    /// Spectral residual saliency of the point (≥ 0).
    pub saliency: f64,
    /// Saliency relative to the preceding judgement window (≥ 0).
    pub score: f64,
    /// Low-frequency reconstruction of the series with anomalies removed.
    pub expected: f64,
    /// `expected − margin`.
    pub lower: f64,
    /// `expected + margin`.
    pub upper: f64,
    /// `score > threshold` and the value clears the z-score gate.
    pub is_anomaly: bool,
}

/// The saliency map of Ren et al. (2019): extend the series by κ = 5 points
/// along the slope of its last m = 5, take the FFT, average the log
/// amplitude over `q` bins, keep the residual with the original phase,
/// transform back, and drop the extension.
fn saliency_map(data: &[f64], q: usize) -> Vec<f64> {
    let n = data.len();
    let mut extended = data.to_vec();
    let m = EXTENSION.min(n - 1);
    let last = data[n - 1];
    let slope = (1..=m)
        .map(|i| (last - data[n - 1 - i]) / i as f64)
        .sum::<f64>()
        / m as f64;
    let predicted = data[n - 1 - m] + slope * m as f64;
    extended.extend(std::iter::repeat_n(predicted, EXTENSION));

    let spectrum = fft(&extended
        .iter()
        .map(|&x| Complex::real(x))
        .collect::<Vec<_>>());
    let len = spectrum.len();
    let log_amp: Vec<f64> = spectrum.iter().map(|z| (z.norm() + 1e-8).ln()).collect();
    // Moving average of width q over the preceding bins (cumulative form).
    let mut avg = vec![0.0; len];
    let mut cum = 0.0;
    let mut cums = Vec::with_capacity(len + 1);
    cums.push(0.0);
    for &v in &log_amp {
        cum += v;
        cums.push(cum);
    }
    for i in 0..len {
        let from = (i + 1).saturating_sub(q);
        avg[i] = (cums[i + 1] - cums[from]) / (i + 1 - from) as f64;
    }
    let residual_spectrum: Vec<Complex> = spectrum
        .iter()
        .enumerate()
        .map(|(i, z)| {
            let r = (log_amp[i] - avg[i]).exp();
            let phase = z.arg();
            Complex::new(r * phase.cos(), r * phase.sin())
        })
        .collect();
    ifft(&residual_spectrum)
        .into_iter()
        .take(n)
        .map(|z| z.norm())
        .collect()
}

/// The series with each flagged point replaced by the mean of its nearest
/// unflagged neighbours, reconstructed from its low-frequency components:
/// the middle band of the spectrum (bins from 3/8 to 5/8 of the length) is
/// zeroed, as in the Microsoft service's expected value.
fn expected_values(data: &[f64], flagged: &[bool]) -> Vec<f64> {
    let n = data.len();
    let mut cleaned = data.to_vec();
    for i in 0..n {
        if !flagged[i] {
            continue;
        }
        let left = (0..i).rev().find(|&j| !flagged[j]).map(|j| data[j]);
        let right = (i + 1..n).find(|&j| !flagged[j]).map(|j| data[j]);
        cleaned[i] = match (left, right) {
            (Some(a), Some(b)) => (a + b) / 2.0,
            (Some(a), None) | (None, Some(a)) => a,
            (None, None) => data[i],
        };
    }
    let mut spectrum = fft(&cleaned
        .iter()
        .map(|&x| Complex::real(x))
        .collect::<Vec<_>>());
    let lo = n * 3 / 8;
    let hi = n * 5 / 8;
    for (i, z) in spectrum.iter_mut().enumerate() {
        if i > lo && i < hi {
            *z = Complex::default();
        }
    }
    ifft(&spectrum).into_iter().map(|z| z.re).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

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

    fn check_shape(points: &[SrPoint], n: usize) {
        assert_eq!(points.len(), n);
        for (i, p) in points.iter().enumerate() {
            assert_eq!(p.index, i);
            assert!(p.saliency >= 0.0 && p.saliency.is_finite(), "{p:?}");
            assert!(p.score >= 0.0 && p.score.is_finite(), "{p:?}");
            assert!(
                p.expected.is_finite() && p.lower <= p.expected && p.expected <= p.upper,
                "{p:?}"
            );
        }
    }

    #[test]
    fn a_spike_on_a_sine_is_the_anomaly() {
        let mut series: Vec<f64> = noise(3, 120)
            .iter()
            .enumerate()
            .map(|(t, e)| (t as f64 * 0.25).sin() + 0.1 * e)
            .collect();
        series[80] += 5.0;
        let points = SpectralResidual::new().analyze(&series).unwrap();
        check_shape(&points, 120);
        assert!(points[80].is_anomaly, "{:?}", points[80]);
        let flagged: Vec<usize> = points
            .iter()
            .filter(|p| p.is_anomaly)
            .map(|p| p.index)
            .collect();
        assert!(flagged.len() <= 3, "flagged {flagged:?}");
        assert!(points[80].value > points[80].upper);
        // The expected value is not dragged up by the spike.
        assert!(
            (points[80].expected - (80.0f64 * 0.25).sin()).abs() < 0.6,
            "{:?}",
            points[80]
        );
    }

    #[test]
    fn a_step_is_flagged_where_it_happens() {
        let series: Vec<f64> = (0..100).map(|t| if t < 60 { 1.0 } else { 6.0 }).collect();
        let points = SpectralResidual::new().analyze(&series).unwrap();
        check_shape(&points, 100);
        let first = points.iter().position(|p| p.is_anomaly);
        assert_eq!(
            first,
            Some(60),
            "{:?}",
            points
                .iter()
                .filter(|p| p.is_anomaly)
                .map(|p| p.index)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn a_constant_and_plain_noise_raise_no_anomaly() {
        let constant = vec![3.0; 50];
        let points = SpectralResidual::new().analyze(&constant).unwrap();
        check_shape(&points, 50);
        assert!(points.iter().all(|p| !p.is_anomaly));
        assert!(points.iter().all(|p| (p.expected - 3.0).abs() < 1e-9));

        let series = noise(9, 200);
        let points = SpectralResidual::new().analyze(&series).unwrap();
        check_shape(&points, 200);
        let flagged = points.iter().filter(|p| p.is_anomaly).count();
        assert!(flagged <= 4, "{flagged} of 200 noise points flagged");
    }

    #[test]
    fn scoring_is_deterministic_and_batching_keeps_the_shape() {
        let mut series: Vec<f64> = noise(5, 300).iter().map(|e| 10.0 + e).collect();
        series[150] = 20.0;
        series[260] = -5.0;
        let sr = SpectralResidual::new();
        let a = sr.analyze(&series).unwrap();
        let b = sr.analyze(&series).unwrap();
        assert_eq!(a, b);
        let batched = sr.with_batch_size(Some(100)).analyze(&series).unwrap();
        check_shape(&batched, 300);
        assert!(batched[150].is_anomaly && batched[260].is_anomaly);
        // Each batch is scored on its own: the first point of a batch has no
        // history but the shape is unchanged and indices are global.
        assert_eq!(batched[200].index, 200);
    }

    #[test]
    fn the_band_widens_with_sensitivity() {
        let series: Vec<f64> = noise(1, 80).iter().map(|e| 5.0 + e).collect();
        let narrow = SpectralResidual::new()
            .with_sensitivity(50.0)
            .analyze(&series)
            .unwrap();
        let wide = SpectralResidual::new()
            .with_sensitivity(99.0)
            .analyze(&series)
            .unwrap();
        for (a, b) in narrow.iter().zip(&wide) {
            assert!(b.upper - b.lower > a.upper - a.lower);
            assert_eq!(a.expected, b.expected);
        }
    }

    /// A refusal names the one condition that failed, so a consumer can point
    /// at the setting to change instead of restating every rule.
    #[test]
    fn a_refusal_names_the_condition_that_failed() {
        let ok = vec![1.0; 20];
        let out_of_range = |option, requirement| {
            Err(SpectralResidualError::OptionOutOfRange {
                option,
                requirement,
            })
        };

        assert_eq!(
            SpectralResidual::new().analyze(&[1.0; 11]),
            Err(SpectralResidualError::TooFewObservations {
                needed: MIN_OBSERVATIONS,
                got: 11
            })
        );

        let mut series = vec![1.0; 20];
        series[7] = f64::INFINITY;
        assert_eq!(
            SpectralResidual::new().analyze(&series),
            Err(SpectralResidualError::ValueNotFinite { index: 7 })
        );

        // Every option, each naming itself — the four the consumer report
        // reached through one indistinguishable message, and the two beside
        // them.
        assert_eq!(
            SpectralResidual::new().with_threshold(0.0).analyze(&ok),
            out_of_range("threshold", "a finite number > 0")
        );
        assert_eq!(
            SpectralResidual::new().with_sensitivity(100.0).analyze(&ok),
            out_of_range("sensitivity", "a finite number strictly between 0 and 100")
        );
        assert_eq!(
            SpectralResidual::new().with_sensitivity(0.0).analyze(&ok),
            out_of_range("sensitivity", "a finite number strictly between 0 and 100")
        );
        assert_eq!(
            SpectralResidual::new()
                .with_batch_size(Some(5))
                .analyze(&ok),
            out_of_range("batch_size", "unset, or at least MIN_OBSERVATIONS")
        );
        assert_eq!(
            SpectralResidual::new()
                .with_averaging_window(0)
                .analyze(&ok),
            out_of_range("averaging_window", ">= 1")
        );
        assert_eq!(
            SpectralResidual::new()
                .with_judgement_window(0)
                .analyze(&ok),
            out_of_range("judgement_window", ">= 1")
        );
        assert_eq!(
            SpectralResidual::new().with_min_zscore(-1.0).analyze(&ok),
            out_of_range("min_zscore", "a finite number >= 0")
        );
        assert_eq!(
            SpectralResidual::new()
                .with_threshold(f64::NAN)
                .analyze(&ok),
            out_of_range("threshold", "a finite number > 0")
        );

        // The message a transport shows repeats the option, not the rulebook.
        let message = SpectralResidual::new()
            .with_threshold(0.0)
            .analyze(&ok)
            .expect_err("refused")
            .to_string();
        assert_eq!(message, "threshold must be a finite number > 0");
        assert!(!message.contains("sensitivity"));

        // `min_zscore = 0` disables the gate; it is not out of range.
        assert!(SpectralResidual::new()
            .with_min_zscore(0.0)
            .analyze(&ok)
            .is_ok());
    }
}
