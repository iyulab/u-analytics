//! Seasonal-trend decomposition of a univariate series by Loess (STL).
//!
//! A series `y` with a known period is split additively into a trend, a
//! seasonal component and a remainder, `y = trend + seasonal + residual`, by
//! the procedure of Cleveland, Cleveland, McRae & Terpenning (1990). The
//! period is an input — [`crate::seasonality::estimate_period`] estimates it
//! when it is not known.
//!
//! # Algorithm
//!
//! An **inner loop** refines the two components, starting from a zero trend:
//!
//! 1. *Detrend*: `y − trend`.
//! 2. *Cycle-subseries smoothing*: the values at each position of the cycle
//!    (every `period`-th value) are smoothed by Loess with span `seasonal`,
//!    and each subseries is extended one cycle before and after by Loess
//!    extrapolation.
//! 3. *Low-pass filter* of that extended series: moving averages of length
//!    `period`, `period` and 3, then Loess with span `low_pass`.
//! 4. *Seasonal* = step 2 − step 3 (the low-pass removes what of the trend
//!    leaked into the subseries).
//! 5. *Trend* = Loess with span `trend` of `y − seasonal`.
//!
//! The **outer loop** (only when `robust`) gives each point a robustness
//! weight from its remainder — the bisquare `(1 − (r/h)²)²` with
//! `h = 6 · median|r|` — and runs the inner loop again with every Loess fit
//! weighted by it, so that outliers go to the remainder instead of bending
//! the trend and seasonal components.
//!
//! Each Loess fit is a local polynomial of degree 0 or 1 with tricube
//! weights over the `span` nearest points. This is the original Fortran
//! implementation (netlib `stl`), which statsmodels' `STL` also follows, with
//! its defaults: degree 1 everywhere, `seasonal = 7`, `trend` the smallest
//! odd integer at least `1.5 · period / (1 − 1.5 / seasonal)`, `low_pass` the
//! smallest odd integer above `period`, and 5 inner / 0 outer iterations
//! (2 / 15 when robust).
//!
//! # Strength of trend and seasonality
//!
//! The result also reports `F_T = max(0, 1 − Var(R)/Var(T + R))` and
//! `F_S = max(0, 1 − Var(R)/Var(S + R))` (Wang, Smith & Hyndman 2006): near 1
//! when the component explains most of what the remainder does not, near 0
//! when it is indistinguishable from noise.
//!
//! # Complexity
//!
//! O(iterations · n · span) for spans `seasonal`, `trend` and `low_pass`.
//!
//! # References
//! - Cleveland, R. B., Cleveland, W. S., McRae, J. E. & Terpenning, I. (1990).
//!   "STL: A seasonal-trend decomposition procedure based on Loess."
//!   *Journal of Official Statistics*, 6(1), 3–73.
//! - Wang, X., Smith, K. & Hyndman, R. J. (2006). "Characteristic-based
//!   clustering for time series data." *Data Mining and Knowledge Discovery*,
//!   13(3), 335–364.

/// Default span of the seasonal smoother (statsmodels' `seasonal=7`).
pub const DEFAULT_SEASONAL: usize = 7;

/// Why a decomposition was refused.
#[derive(Debug, Clone, PartialEq)]
pub enum StlError {
    /// An option is outside its domain. `min` / `max` are inclusive bounds;
    /// `exclusive_min` an exclusive one (a span longer than the period).
    OptionOutOfRange {
        /// The option's name as the builder and the wire formats spell it.
        option: &'static str,
        /// What that option has to satisfy.
        requirement: &'static str,
        /// The value given.
        got: f64,
        /// Inclusive lower bound.
        min: Option<f64>,
        /// Exclusive lower bound.
        exclusive_min: Option<f64>,
        /// Inclusive upper bound.
        max: Option<f64>,
    },
    /// A span is even: a Loess span is centred on its point, so it is odd.
    EvenSpan {
        /// `seasonal`, `trend` or `low_pass`.
        option: &'static str,
        /// The value given.
        got: usize,
    },
    /// Fewer than two full cycles.
    TooFewObservations {
        /// `2 · period`.
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

impl core::fmt::Display for StlError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            StlError::OptionOutOfRange {
                option,
                requirement,
                got,
                ..
            } => write!(f, "stl: `{option}` must be {requirement} (got {got})"),
            StlError::EvenSpan { option, got } => write!(
                f,
                "stl: `{option}` must be odd — a Loess span is centred on its point (got {got})"
            ),
            StlError::TooFewObservations { needed, got } => write!(
                f,
                "stl: needs at least two full cycles, {needed} values (got {got})"
            ),
            StlError::ValueNotFinite { index } => {
                write!(f, "stl: the value at index {index} is not finite")
            }
        }
    }
}

impl std::error::Error for StlError {}

/// An STL decomposition: `data[i] = trend[i] + seasonal[i] + residual[i]`.
#[derive(Debug, Clone, PartialEq)]
pub struct StlResult {
    /// Trend component.
    pub trend: Vec<f64>,
    /// Seasonal component.
    pub seasonal: Vec<f64>,
    /// Remainder `data − trend − seasonal`.
    pub residual: Vec<f64>,
    /// Robustness weight of each point in the last pass, in `[0, 1]` (all 1
    /// without `robust`). A weight near 0 marks a point treated as an outlier.
    pub weights: Vec<f64>,
    /// Strength of the trend, `max(0, 1 − Var(R)/Var(T + R))`.
    pub trend_strength: f64,
    /// Strength of the seasonality, `max(0, 1 − Var(R)/Var(S + R))`.
    pub seasonal_strength: f64,
    /// The period decomposed.
    pub period: usize,
    /// Span of the seasonal smoother used.
    pub seasonal_span: usize,
    /// Span of the trend smoother used (the default when none was given).
    pub trend_span: usize,
    /// Span of the low-pass smoother used (the default when none was given).
    pub low_pass_span: usize,
}

/// STL decomposition with a builder for its options.
///
/// # Examples
///
/// ```
/// use u_analytics::decomposition::Stl;
///
/// // A rising line plus a cycle of four.
/// let pattern = [2.0, -1.0, 0.5, -1.5];
/// let y: Vec<f64> = (0..48).map(|i| 0.3 * i as f64 + pattern[i % 4]).collect();
/// let r = Stl::new(4).decompose(&y).unwrap();
/// assert!((r.seasonal[20] - 2.0).abs() < 1e-6);
/// assert!(r.seasonal_strength > 0.99);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Stl {
    period: usize,
    seasonal: usize,
    trend: Option<usize>,
    low_pass: Option<usize>,
    seasonal_degree: usize,
    trend_degree: usize,
    low_pass_degree: usize,
    robust: bool,
    inner_iterations: Option<usize>,
    outer_iterations: Option<usize>,
}

impl Stl {
    /// STL for a series with this period, with statsmodels' defaults.
    pub fn new(period: usize) -> Self {
        Self {
            period,
            seasonal: DEFAULT_SEASONAL,
            trend: None,
            low_pass: None,
            seasonal_degree: 1,
            trend_degree: 1,
            low_pass_degree: 1,
            robust: false,
            inner_iterations: None,
            outer_iterations: None,
        }
    }

    /// Span of the seasonal smoother: odd, at least 3. Larger = a seasonal
    /// pattern that changes more slowly from cycle to cycle.
    pub fn seasonal(mut self, span: usize) -> Self {
        self.seasonal = span;
        self
    }

    /// Span of the trend smoother: odd, at least 3, above the period.
    pub fn trend(mut self, span: usize) -> Self {
        self.trend = Some(span);
        self
    }

    /// Span of the low-pass smoother: odd, at least 3, above the period.
    pub fn low_pass(mut self, span: usize) -> Self {
        self.low_pass = Some(span);
        self
    }

    /// Degree (0 or 1) of the seasonal Loess.
    pub fn seasonal_degree(mut self, degree: usize) -> Self {
        self.seasonal_degree = degree;
        self
    }

    /// Degree (0 or 1) of the trend Loess.
    pub fn trend_degree(mut self, degree: usize) -> Self {
        self.trend_degree = degree;
        self
    }

    /// Degree (0 or 1) of the low-pass Loess.
    pub fn low_pass_degree(mut self, degree: usize) -> Self {
        self.low_pass_degree = degree;
        self
    }

    /// Down-weight outliers with the outer loop.
    pub fn robust(mut self, robust: bool) -> Self {
        self.robust = robust;
        self
    }

    /// Passes of the inner loop (default 5, or 2 when robust). At least 1.
    pub fn inner_iterations(mut self, n: usize) -> Self {
        self.inner_iterations = Some(n);
        self
    }

    /// Passes of the outer loop (default 0, or 15 when robust).
    pub fn outer_iterations(mut self, n: usize) -> Self {
        self.outer_iterations = Some(n);
        self
    }

    /// The trend span used when none is given: the smallest odd integer at
    /// least `1.5 · period / (1 − 1.5 / seasonal)` (Cleveland et al. 1990,
    /// §3.5; statsmodels' default).
    pub fn default_trend_span(period: usize, seasonal: usize) -> usize {
        let t = (1.5 * period as f64 / (1.0 - 1.5 / seasonal as f64)).ceil() as usize;
        if t % 2 == 0 {
            t + 1
        } else {
            t
        }
    }

    /// The low-pass span used when none is given: the smallest odd integer
    /// above the period.
    pub fn default_low_pass_span(period: usize) -> usize {
        let l = period + 1;
        if l % 2 == 0 {
            l + 1
        } else {
            l
        }
    }

    /// Decompose `data`.
    ///
    /// # Errors
    ///
    /// [`StlError`] when an option is outside its domain, a span is even,
    /// there are fewer than two full cycles, or a value is not finite.
    pub fn decompose(&self, data: &[f64]) -> Result<StlResult, StlError> {
        let period = self.period;
        if period < 2 {
            return Err(StlError::OptionOutOfRange {
                option: "period",
                requirement: "at least 2",
                got: period as f64,
                min: Some(2.0),
                exclusive_min: None,
                max: None,
            });
        }
        let span = |option: &'static str, got: usize, above_period: bool| {
            if got < 3 || (above_period && got <= period) {
                let (min, exclusive_min, requirement) = if above_period && period >= 3 {
                    (None, Some(period as f64), "odd and above the period")
                } else {
                    (Some(3.0), None, "odd and at least 3")
                };
                return Err(StlError::OptionOutOfRange {
                    option,
                    requirement,
                    got: got as f64,
                    min,
                    exclusive_min,
                    max: None,
                });
            }
            if got % 2 == 0 {
                return Err(StlError::EvenSpan { option, got });
            }
            Ok(got)
        };
        let seasonal = span("seasonal", self.seasonal, false)?;
        let trend = span(
            "trend",
            self.trend
                .unwrap_or_else(|| Self::default_trend_span(period, seasonal)),
            true,
        )?;
        let low_pass = span(
            "low_pass",
            self.low_pass
                .unwrap_or_else(|| Self::default_low_pass_span(period)),
            true,
        )?;
        for (option, degree) in [
            ("seasonal_degree", self.seasonal_degree),
            ("trend_degree", self.trend_degree),
            ("low_pass_degree", self.low_pass_degree),
        ] {
            if degree > 1 {
                return Err(StlError::OptionOutOfRange {
                    option,
                    requirement: "0 or 1",
                    got: degree as f64,
                    min: Some(0.0),
                    exclusive_min: None,
                    max: Some(1.0),
                });
            }
        }
        let inner = self
            .inner_iterations
            .unwrap_or(if self.robust { 2 } else { 5 });
        if inner == 0 {
            return Err(StlError::OptionOutOfRange {
                option: "inner_iterations",
                requirement: "at least 1",
                got: 0.0,
                min: Some(1.0),
                exclusive_min: None,
                max: None,
            });
        }
        let outer = self
            .outer_iterations
            .unwrap_or(if self.robust { 15 } else { 0 });

        let n = data.len();
        if n < 2 * period {
            return Err(StlError::TooFewObservations {
                needed: 2 * period,
                got: n,
            });
        }
        if let Some(index) = data.iter().position(|v| !v.is_finite()) {
            return Err(StlError::ValueNotFinite { index });
        }

        let spans = Spans {
            period,
            seasonal,
            trend,
            low_pass,
            seasonal_degree: self.seasonal_degree,
            trend_degree: self.trend_degree,
            low_pass_degree: self.low_pass_degree,
        };
        let mut trend_c = vec![0.0; n];
        let mut seasonal_c = vec![0.0; n];
        let mut weights = vec![1.0; n];
        let mut use_weights = false;
        let mut pass = 0;
        loop {
            inner_loop(
                data,
                &spans,
                inner,
                use_weights.then_some(weights.as_slice()),
                &mut seasonal_c,
                &mut trend_c,
            );
            pass += 1;
            if pass > outer {
                break;
            }
            let fit: Vec<f64> = trend_c
                .iter()
                .zip(&seasonal_c)
                .map(|(t, s)| t + s)
                .collect();
            robustness_weights(data, &fit, &mut weights);
            use_weights = true;
        }
        if outer == 0 {
            weights.iter_mut().for_each(|w| *w = 1.0);
        }

        let residual: Vec<f64> = (0..n)
            .map(|i| data[i] - trend_c[i] - seasonal_c[i])
            .collect();
        let var_r = variance(&residual);
        let strength = |component: &[f64]| {
            let sum: Vec<f64> = component
                .iter()
                .zip(&residual)
                .map(|(c, r)| c + r)
                .collect();
            let v = variance(&sum);
            if v > 0.0 {
                (1.0 - var_r / v).max(0.0)
            } else {
                0.0
            }
        };
        Ok(StlResult {
            trend_strength: strength(&trend_c),
            seasonal_strength: strength(&seasonal_c),
            trend: trend_c,
            seasonal: seasonal_c,
            residual,
            weights,
            period,
            seasonal_span: seasonal,
            trend_span: trend,
            low_pass_span: low_pass,
        })
    }
}

struct Spans {
    period: usize,
    seasonal: usize,
    trend: usize,
    low_pass: usize,
    seasonal_degree: usize,
    trend_degree: usize,
    low_pass_degree: usize,
}

fn variance(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    let mean = x.iter().sum::<f64>() / n;
    x.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n
}

/// One run of the inner loop (`stlstp`): `passes` refinements of the
/// seasonal and trend components, in place.
fn inner_loop(
    y: &[f64],
    s: &Spans,
    passes: usize,
    rw: Option<&[f64]>,
    seasonal: &mut [f64],
    trend: &mut [f64],
) {
    let n = y.len();
    let np = s.period;
    let mut detrended = vec![0.0; n];
    let mut cycle = vec![0.0; n + 2 * np];
    let mut low = vec![0.0; n];
    for _ in 0..passes {
        for i in 0..n {
            detrended[i] = y[i] - trend[i];
        }
        cycle_subseries(
            &detrended,
            np,
            s.seasonal,
            s.seasonal_degree,
            rw,
            &mut cycle,
        );
        let filtered = low_pass_filter(&cycle, np);
        loess(&filtered, s.low_pass, s.low_pass_degree, None, &mut low);
        for i in 0..n {
            seasonal[i] = cycle[np + i] - low[i];
            detrended[i] = y[i] - seasonal[i];
        }
        loess(&detrended, s.trend, s.trend_degree, rw, trend);
    }
}

/// Cycle-subseries smoothing (`stlss`): each subseries smoothed and extended
/// one value at each end; the result has `n + 2 · period` values.
fn cycle_subseries(
    y: &[f64],
    np: usize,
    span: usize,
    degree: usize,
    rw: Option<&[f64]>,
    out: &mut [f64],
) {
    let n = y.len();
    let mut sub = Vec::with_capacity(n / np + 1);
    let mut sub_w = Vec::with_capacity(n / np + 1);
    let mut smooth = Vec::with_capacity(n / np + 3);
    let mut work = vec![0.0; n / np + 1];
    for j in 0..np {
        sub.clear();
        sub_w.clear();
        sub.extend((j..n).step_by(np).map(|i| y[i]));
        if let Some(rw) = rw {
            sub_w.extend((j..n).step_by(np).map(|i| rw[i]));
        }
        let w = rw.map(|_| sub_w.as_slice());
        let k = sub.len();
        smooth.clear();
        smooth.resize(k + 2, 0.0);
        loess(&sub, span, degree, w, &mut smooth[1..=k]);
        let right = span.min(k);
        smooth[0] = local_fit(&sub, span, degree, 0.0, 1, right, w, &mut work).unwrap_or(smooth[1]);
        let left = if k >= span { k - span + 1 } else { 1 };
        smooth[k + 1] = local_fit(&sub, span, degree, (k + 1) as f64, left, k, w, &mut work)
            .unwrap_or(smooth[k]);
        for (m, v) in smooth.iter().enumerate() {
            out[m * np + j] = *v;
        }
    }
}

/// Moving averages of length `period`, `period`, 3 (`stlfts`): `n + 2 · period`
/// values in, `n` out.
fn low_pass_filter(x: &[f64], np: usize) -> Vec<f64> {
    moving_average(&moving_average(&moving_average(x, np), np), 3)
}

/// Running-sum moving average of length `len` (`stlma`).
fn moving_average(x: &[f64], len: usize) -> Vec<f64> {
    let out_len = x.len() + 1 - len;
    let flen = len as f64;
    let mut out = Vec::with_capacity(out_len);
    let mut v: f64 = x[..len].iter().sum();
    out.push(v / flen);
    for k in 1..out_len {
        v = v - x[k - 1] + x[k + len - 1];
        out.push(v / flen);
    }
    out
}

/// Loess smoothing of `y` at every position (`stless`, jump 1).
fn loess(y: &[f64], span: usize, degree: usize, rw: Option<&[f64]>, out: &mut [f64]) {
    let n = y.len();
    if n < 2 {
        out[0] = y[0];
        return;
    }
    let mut work = vec![0.0; n];
    if span >= n {
        for i in 1..=n {
            out[i - 1] =
                local_fit(y, span, degree, i as f64, 1, n, rw, &mut work).unwrap_or(y[i - 1]);
        }
        return;
    }
    let half = span.div_ceil(2);
    let (mut left, mut right) = (1, span);
    for i in 1..=n {
        if i > half && right != n {
            left += 1;
            right += 1;
        }
        out[i - 1] =
            local_fit(y, span, degree, i as f64, left, right, rw, &mut work).unwrap_or(y[i - 1]);
    }
}

/// Local fit at `xs` from the points `left..=right` (1-based) with tricube
/// weights (`stlest`). `None` when every weight is zero.
#[allow(clippy::too_many_arguments)]
fn local_fit(
    y: &[f64],
    span: usize,
    degree: usize,
    xs: f64,
    left: usize,
    right: usize,
    rw: Option<&[f64]>,
    w: &mut [f64],
) -> Option<f64> {
    let n = y.len();
    let range = (n - 1) as f64;
    let mut h = (xs - left as f64).max(right as f64 - xs);
    if span > n {
        h += ((span - n) / 2) as f64;
    }
    let h9 = 0.999 * h;
    let h1 = 0.001 * h;
    let mut a = 0.0;
    for j in left..=right {
        let r = (j as f64 - xs).abs();
        let mut wj = 0.0;
        if r <= h9 {
            wj = if r <= h1 {
                1.0
            } else {
                (1.0 - (r / h).powi(3)).powi(3)
            };
            if let Some(rw) = rw {
                wj *= rw[j - 1];
            }
            a += wj;
        }
        w[j - 1] = wj;
    }
    if a <= 0.0 {
        return None;
    }
    for wj in &mut w[left - 1..right] {
        *wj /= a;
    }
    if h > 0.0 && degree > 0 {
        let a: f64 = (left..=right).map(|j| w[j - 1] * j as f64).sum();
        let b = xs - a;
        let c: f64 = (left..=right)
            .map(|j| w[j - 1] * (j as f64 - a) * (j as f64 - a))
            .sum();
        if c.sqrt() > 0.001 * range {
            let b = b / c;
            for j in left..=right {
                w[j - 1] *= b * (j as f64 - a) + 1.0;
            }
        }
    }
    Some((left..=right).map(|j| w[j - 1] * y[j - 1]).sum())
}

/// Bisquare robustness weights from the remainder (`stlrwt`).
fn robustness_weights(y: &[f64], fit: &[f64], rw: &mut [f64]) {
    let n = y.len();
    let r: Vec<f64> = y.iter().zip(fit).map(|(a, b)| (a - b).abs()).collect();
    let mut sorted = r.clone();
    let m1 = n / 2; // 0-based of Fortran's n/2 + 1
    let m2 = n - m1 - 1; // 0-based of n − mid1 + 1
    sorted.sort_by(|a, b| a.total_cmp(b));
    let cmad = 3.0 * (sorted[m1] + sorted[m2]);
    let scale = y.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    if cmad <= 1e-12 * scale {
        // More than half the points fit to the rounding of the data: the
        // remainder is arithmetic noise, not a scale to judge outliers by.
        rw.iter_mut().for_each(|w| *w = 1.0);
        return;
    }
    let c9 = 0.999 * cmad;
    let c1 = 0.001 * cmad;
    for (w, &ri) in rw.iter_mut().zip(&r) {
        *w = if ri <= c1 {
            1.0
        } else if ri <= c9 {
            let u = ri / cmad;
            (1.0 - u * u).powi(2)
        } else {
            0.0
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seasonal_line(n: usize) -> Vec<f64> {
        let pattern = [2.0, -1.0, 0.5, -1.5];
        (0..n)
            .map(|i| 0.3 * i as f64 + pattern[i % 4] + 0.05 * (((i * 7919) % 13) as f64 - 6.0))
            .collect()
    }

    #[test]
    fn components_add_up_to_the_series() {
        let y = seasonal_line(40);
        let r = Stl::new(4).decompose(&y).unwrap();
        for (i, v) in y.iter().enumerate() {
            assert!((r.trend[i] + r.seasonal[i] + r.residual[i] - v).abs() < 1e-12);
        }
        assert_eq!((r.seasonal_span, r.trend_span, r.low_pass_span), (7, 9, 5));
        assert!(r.weights.iter().all(|&w| w == 1.0));
    }

    #[test]
    fn default_spans_follow_statsmodels() {
        assert_eq!(Stl::default_trend_span(12, 7), 23);
        assert_eq!(Stl::default_trend_span(7, 7), 15);
        assert_eq!(Stl::default_low_pass_span(12), 13);
        assert_eq!(Stl::default_low_pass_span(7), 9);
    }

    #[test]
    fn refusals_name_the_option() {
        let y = seasonal_line(40);
        assert!(matches!(
            Stl::new(1).decompose(&y),
            Err(StlError::OptionOutOfRange {
                option: "period",
                ..
            })
        ));
        assert!(matches!(
            Stl::new(4).seasonal(8).decompose(&y),
            Err(StlError::EvenSpan {
                option: "seasonal",
                got: 8
            })
        ));
        assert!(matches!(
            Stl::new(4).trend(3).decompose(&y),
            Err(StlError::OptionOutOfRange { option: "trend", exclusive_min: Some(e), .. }) if e == 4.0
        ));
        assert!(matches!(
            Stl::new(4).trend_degree(2).decompose(&y),
            Err(StlError::OptionOutOfRange {
                option: "trend_degree",
                ..
            })
        ));
        assert_eq!(
            Stl::new(4).decompose(&y[..7]),
            Err(StlError::TooFewObservations { needed: 8, got: 7 })
        );
        let mut bad = y.clone();
        bad[5] = f64::NAN;
        assert_eq!(
            Stl::new(4).decompose(&bad),
            Err(StlError::ValueNotFinite { index: 5 })
        );
    }

    #[test]
    fn robust_pass_isolates_an_outlier() {
        let mut y = seasonal_line(48);
        y[21] += 50.0;
        let plain = Stl::new(4).decompose(&y).unwrap();
        let robust = Stl::new(4).robust(true).decompose(&y).unwrap();
        assert!(robust.weights[21] < 0.01);
        assert!(robust.residual[21] > 45.0);
        assert!(robust.residual[21] > plain.residual[21]);
    }

    #[test]
    fn two_cycles_fit_exactly_and_rounding_marks_no_outlier() {
        // Two values per subseries: the seasonal fit interpolates them, the
        // remainder is rounding — robust weights must not read it as outliers.
        let y = seasonal_line(24);
        let r = Stl::new(12).robust(true).decompose(&y).unwrap();
        assert!(r.residual.iter().all(|v| v.abs() < 1e-9));
        assert!(r.weights.iter().all(|&w| w == 1.0));
    }

    #[test]
    fn unit_and_offset_do_not_change_the_shape() {
        let y = seasonal_line(48);
        let base = Stl::new(4).robust(true).decompose(&y).unwrap();
        for scale in [1e-9, 1e9] {
            let ys: Vec<f64> = y.iter().map(|v| v * scale + 7.0 * scale).collect();
            let r = Stl::new(4).robust(true).decompose(&ys).unwrap();
            for i in 0..y.len() {
                assert!((r.seasonal[i] / scale - base.seasonal[i]).abs() < 1e-6);
                assert!((r.weights[i] - base.weights[i]).abs() < 1e-9);
            }
            assert!((r.seasonal_strength - base.seasonal_strength).abs() < 1e-9);
        }
    }
}

/// STL against statsmodels 0.15.0 `STL(y, period, ...).fit()` on seeded
/// series (rounded to 3 places): plain, robust with outliers, explicit spans,
/// degree 0, a robust walk, a seasonal span longer than the subseries.
#[cfg(test)]
#[allow(clippy::approx_constant)] // seeded data, not mathematical constants
mod statsmodels_reference {
    use super::*;

    // Generated by stl_ref.py - statsmodels 0.15.0 STL, seed 20261027.
    // case a: n=60 period=12 plain {}
    const Y_A: [f64; 60] = [
        -0.837, 1.092, 2.924, 2.522, 2.875, 1.517, -0.253, -2.621, -1.958, -1.908, -2.977, -2.431,
        0.283, 2.564, 4.129, 3.407, 3.442, 2.796, 1.797, -1.15, -1.333, -3.02, -1.687, -0.955,
        -0.182, 2.155, 3.519, 4.249, 3.793, 3.02, 2.088, -0.256, -2.699, -1.328, -0.227, -0.623,
        0.974, 4.007, 3.766, 5.139, 4.737, 3.505, 1.892, 0.397, -1.058, -0.66, -1.369, 0.869,
        1.651, 3.504, 5.64, 5.463, 6.604, 3.893, 1.852, 1.122, 0.75, -1.866, 0.37, 0.427,
    ];
    const T_A: [f64; 60] = [
        -0.7146485637078752,
        -0.6164832210335263,
        -0.5181516830760083,
        -0.4196844754873599,
        -0.32110131859995555,
        -0.2225207692561986,
        -0.12423555675326292,
        -0.02655075377094378,
        0.07045746328083816,
        0.16696728502205307,
        0.26338291889548854,
        0.36043534415399964,
        0.44834507397027185,
        0.5280747524929036,
        0.6002229363186898,
        0.6611364297563962,
        0.7062536526796435,
        0.7365967978054677,
        0.7584153148362388,
        0.7761606513272636,
        0.7923364502503742,
        0.8093936697027934,
        0.8296387158227432,
        0.8518631642150369,
        0.8773150661668782,
        0.9087084512521527,
        0.9464431041493069,
        0.9938060179447082,
        1.054557457023869,
        1.1249976100247374,
        1.199865521544602,
        1.2737985030102073,
        1.3436229324321316,
        1.4106131175610293,
        1.47365203587191,
        1.530351119602024,
        1.5832977086243996,
        1.6333441280837124,
        1.6832582510569378,
        1.7331639399795236,
        1.7828201268627029,
        1.8322863129013094,
        1.8800526846196015,
        1.9286762638050863,
        1.9832437306954993,
        2.0422080162708522,
        2.1022504858694147,
        2.163628155127377,
        2.2235596908381847,
        2.268879067727309,
        2.3095696049217467,
        2.3490127808421275,
        2.387720572510347,
        2.4257236700994116,
        2.462893961270545,
        2.4991492682583005,
        2.534496788787009,
        2.568912499320822,
        2.6022119100174876,
        2.634401344550984,
    ];
    const S_A: [f64; 60] = [
        -0.17020102326156605,
        1.726556436052315,
        3.5192952736717285,
        2.801246475765134,
        2.916301498311998,
        1.8679901171016524,
        0.385417573060896,
        -2.4616280221274254,
        -2.1589307740556993,
        -2.624717947693237,
        -2.9033791431503775,
        -2.569979272275229,
        -0.3949369581430479,
        1.747341176718214,
        3.1565219997842857,
        2.96737239548915,
        2.9255075319079604,
        1.8498863004764663,
        0.4439005079073706,
        -2.0819707336913336,
        -2.5733476274600338,
        -2.8396345182460423,
        -2.716653100846297,
        -2.232532099802241,
        -0.5813597218343396,
        1.7742640717146578,
        2.846471756328533,
        3.1091422937578663,
        2.977866441638085,
        1.8125842519758864,
        0.44065198659717053,
        -1.7333062411783062,
        -2.877358310844143,
        -3.1228573040388334,
        -2.5814859038818416,
        -1.923662331769904,
        -0.6589242066436268,
        1.654448134865381,
        2.7902213581215958,
        3.196646615959359,
        3.378743148206909,
        1.6664508812510805,
        0.03202021680374002,
        -1.5119953659601402,
        -2.6801711348329444,
        -3.4754184948832547,
        -2.53811618842274,
        -1.8778570095140745,
        -0.6882272489291716,
        1.5713092637842123,
        2.769808959128488,
        3.291131276516374,
        3.8327348075558714,
        1.5116377387724362,
        -0.4415786081405177,
        -1.3117640740058498,
        -2.3837945654928134,
        -3.8267872984984574,
        -2.570245955232843,
        -1.8449407343136188,
    ];
    const W_A: [f64; 60] = [
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    ];

    // case b: n=61 period=7 outlier {'robust': True}
    const Y_B: [f64; 61] = [
        -0.025, 1.069, 3.35, 3.942, -1.766, -2.181, -2.153, 0.516, 2.39, 3.566, 2.573, -0.231,
        -2.106, -0.692, 1.693, 3.686, 4.933, 2.177, -0.175, -1.992, 22.633, 0.344, 3.065, 3.292,
        1.623, 1.014, -1.099, -1.114, 0.943, 3.337, 4.413, 2.255, 0.203, 0.065, -1.017, 2.144,
        5.494, 3.122, 2.931, 1.845, -18.96, 0.318, 1.661, 5.137, 6.434, 3.989, 0.746, -0.527,
        -0.577, 2.835, 4.683, 6.083, 3.598, 2.246, -0.686, 0.157, 2.89, 4.609, 6.722, 4.243, 0.62,
    ];
    const T_B: [f64; 61] = [
        -0.3442510947669004,
        -0.23062424435746,
        -0.11685262949421589,
        -0.002469718500161184,
        0.11274093104960016,
        0.2289590590691011,
        0.34669621930681893,
        0.4641410268250529,
        0.5918511164309425,
        0.7336052455122047,
        0.884323396730363,
        1.0360398044200754,
        1.1636505625530387,
        1.245837797738338,
        1.286580011423615,
        1.290432042200402,
        1.2442565697083043,
        1.164855851233538,
        1.063971152139175,
        0.9472392292523064,
        0.8329130621863098,
        0.7639822496461729,
        0.7514682659961501,
        0.771524888489271,
        0.8125694910960958,
        0.8849167375990827,
        0.9743782676420534,
        1.0741244472560494,
        1.1874184134193109,
        1.3058037914844787,
        1.424472836211481,
        1.5601850416849212,
        1.7151903867377953,
        1.8770424109280648,
        2.044406635246177,
        2.2091025127247335,
        2.3597781235451203,
        2.4870176685979795,
        2.580461292190404,
        2.6573004568075813,
        2.7290947382698336,
        2.768976483362874,
        2.775035332744085,
        2.7573688301410217,
        2.7140131730950867,
        2.645656622357796,
        2.58230075551713,
        2.524365699956627,
        2.4667235014472038,
        2.4239646079783688,
        2.4138328106340237,
        2.428357847754513,
        2.453585556625788,
        2.4814970344726954,
        2.513165475314854,
        2.5480943246238934,
        2.5842123463425395,
        2.620406502455848,
        2.6563542843093986,
        2.6922064934988774,
        2.728112204519468,
    ];
    const S_B: [f64; 61] = [
        0.2806980187795315,
        1.4424643683777076,
        3.309859801264952,
        1.8893345562968902,
        -1.66195680288314,
        -2.9614845621636157,
        -2.2475741103014277,
        0.15409820365643145,
        1.7409003514457582,
        3.1842055432617626,
        1.5373450003572735,
        -1.5168859483966926,
        -2.7663839335106224,
        -2.28021361493963,
        0.02868900722678074,
        2.0190783320351633,
        3.055816690497411,
        1.209301757094185,
        -1.392748079058499,
        -2.557150028631557,
        -2.3283217690321734,
        -0.08277082758805875,
        2.26725469072666,
        2.957666606262043,
        0.8714716745433282,
        -1.3087917264876636,
        -2.257299363938695,
        -2.3825384207403197,
        -0.20064860020793335,
        2.3923190059595694,
        3.049359773786675,
        0.7423574645215033,
        -1.3291972803458882,
        -2.27500508802811,
        -2.5646136176361476,
        -0.11024987307195383,
        2.408543693561008,
        3.283056252051252,
        0.8446694985351011,
        -1.4320403635697274,
        -2.602957908650004,
        -2.63457485800498,
        0.06978123588918689,
        2.338058755805587,
        3.537084619549874,
        1.0453567720871335,
        -1.6473847818099971,
        -2.8964683776519093,
        -2.637053599801231,
        0.22227977439498894,
        2.2364162812135167,
        3.7987107994003124,
        1.2751337860004317,
        -1.877933876664946,
        -3.218492433488227,
        -2.620328624516707,
        0.3863930871033018,
        2.1108508097586216,
        4.051933593412693,
        1.5187065616313047,
        -2.114415902711208,
    ];
    const W_B: [f64; 61] = [
        0.9997292269068923,
        0.9695652397933096,
        0.9809542316008585,
        0.0,
        0.9423651229627944,
        0.734405988773694,
        0.9232852137137939,
        0.9917886807301683,
        0.9940206965090626,
        0.8889394114902743,
        0.9971333240611042,
        0.9303323027332928,
        0.760428447475915,
        0.86637004183294,
        0.8339724328930439,
        0.8424500018058209,
        0.6052806385563598,
        0.9432511306815423,
        0.9808039178471991,
        0.8567328788380366,
        0.0,
        0.8849392676251042,
        0.9985147579627649,
        0.7986949677342619,
        0.9943515347120641,
        0.0,
        0.9664450636558382,
        0.9614963970696023,
        0.9989195146322842,
        0.852399171466217,
        0.9951604693611221,
        0.9977196196085271,
        0.9408384687736601,
        0.7908358409889987,
        0.7534909697893347,
        0.9901271716333155,
        0.5527536287838176,
        0.0,
        0.7648624635843374,
        0.6529578612993352,
        0.0,
        0.95046361562721,
        0.12304637167915942,
        0.9961449380002448,
        0.9534801123689413,
        0.8917773805691143,
        0.9607518121979802,
        0.9769105086278805,
        0.8343505718889977,
        0.945216049382716,
        0.999470548492894,
        0.9760149956852816,
        0.9807071000485312,
        0.0,
        0.9997986989258324,
        0.9469564672618057,
        0.9967764725460029,
        0.9830052938078254,
        0.9999124350801569,
        0.9991755348009493,
        0.9999553771693069,
    ];

    // case c: n=50 period=5 walk {'seasonal': 11, 'trend': 17, 'low_pass': 7}
    const Y_C: [f64; 50] = [
        2.899, 1.404, -1.09, -1.191, 1.22, 1.92, 1.528, -1.227, -1.219, 0.985, 4.068, 3.003, 0.249,
        0.273, 2.171, 1.173, 0.425, -0.886, -2.126, -0.761, 1.133, 0.238, -0.288, 1.395, 3.725,
        4.073, 1.126, -2.636, -4.191, -1.998, -1.086, -2.35, -4.801, -6.16, -4.845, -3.834, -4.53,
        -4.438, -4.914, -4.304, -2.863, -3.678, -5.134, -4.719, -2.169, -1.517, -2.931, -4.141,
        -4.018, -1.645,
    ];
    const T_C: [f64; 50] = [
        0.3595203697419831,
        0.4133930642182823,
        0.4747768840530127,
        0.5435546680555853,
        0.619198114753108,
        0.7006157218307316,
        0.7857221424193703,
        0.8726106252710931,
        0.9722872255626283,
        1.0672531388379005,
        1.1095123655999835,
        1.1087515830857375,
        1.062241411605581,
        0.9564646591506943,
        0.7940533028867663,
        0.5869080747761846,
        0.37508509647620825,
        0.23031218932473474,
        0.21030891092760245,
        0.3074701544826551,
        0.46153339094123064,
        0.5941559397386557,
        0.6383212264487284,
        0.5719836910961416,
        0.42021275781920436,
        0.19694515652981875,
        -0.1280104380675565,
        -0.5930481811158689,
        -1.2029650206895672,
        -1.9105559538321204,
        -2.6131289498929537,
        -3.1972277710202404,
        -3.6137910639406168,
        -3.8913542357283624,
        -4.084695546244734,
        -4.232297509050723,
        -4.336838455231511,
        -4.3695194142421405,
        -4.307424797070813,
        -4.163872109579644,
        -3.9851002160601596,
        -3.815136335577719,
        -3.665168545308158,
        -3.5115010353483163,
        -3.352547491530393,
        -3.189082529735992,
        -3.022825774562644,
        -2.8554492683085404,
        -2.687940406880207,
        -2.520467223905384,
    ];
    const S_C: [f64; 50] = [
        1.9811770199303513,
        0.9964325672477123,
        -1.6639496911020202,
        -1.7074363882518788,
        0.48439109227970223,
        1.8971925211052052,
        0.8767641970404536,
        -1.5650572383620416,
        -1.6403984718625118,
        0.5075879485704992,
        1.8265999571068101,
        0.7716356997912741,
        -1.4679414387841518,
        -1.5869990529831526,
        0.5123046197969641,
        1.7642672260377779,
        0.6856967005677134,
        -1.361321906507085,
        -1.5368582227494896,
        0.4993615211265455,
        1.6909560314383185,
        0.5994208438145355,
        -1.2267892658829993,
        -1.468209067376339,
        0.4848383868349154,
        1.5872641684260307,
        0.4792404577037621,
        -1.1167226637194017,
        -1.3821272181795852,
        0.4995775453353193,
        1.540497098999799,
        0.3958723226446683,
        -1.0968850533070467,
        -1.33581145863842,
        0.5583095598817691,
        1.50199003911766,
        0.32930446900491595,
        -1.0679208315418778,
        -1.300179407712935,
        0.5912549540997699,
        1.4462761765144987,
        0.2639949024928305,
        -1.0226430045502393,
        -1.2463805712810467,
        0.624199874857744,
        1.3748658333196466,
        0.18963002569731405,
        -0.9811266241591107,
        -1.1788791887612886,
        0.6766480651519423,
    ];
    const W_C: [f64; 50] = [
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    ];

    // case d: n=37 period=4 plain {'seasonal_deg': 0, 'trend_deg': 0, 'low_pass_deg': 0}
    const Y_D: [f64; 37] = [
        0.39, 1.758, 0.311, -3.652, -0.216, 1.701, 0.579, -2.206, 0.144, 3.245, 1.514, -3.3, 0.043,
        3.213, -0.027, -0.615, 0.923, 2.396, 0.684, -2.364, 1.723, 4.237, 1.198, -1.383, 0.675,
        4.491, 1.556, -1.63, -0.445, 5.039, 1.742, -1.983, 1.801, 3.564, 2.798, -1.145, 2.008,
    ];
    const T_D: [f64; 37] = [
        -0.29061594267302715,
        -0.2973927741153516,
        -0.3073351786431117,
        -0.33505220397377183,
        -0.3528489557335598,
        -0.19649320522216404,
        0.0015922709096244866,
        0.2563307771874918,
        0.46800059701173313,
        0.4614461082021752,
        0.39318706163134826,
        0.3623442483985528,
        0.31390915172430783,
        0.43178562855612257,
        0.6346582301858984,
        0.6933269615676654,
        0.6858712436708,
        0.6810799527108491,
        0.6961762130679717,
        0.8353162435221444,
        1.0504968974462212,
        1.2171926624324578,
        1.3034891280694905,
        1.3002424793726404,
        1.284409768450904,
        1.232090942519993,
        1.1317427109376321,
        1.144457822901661,
        1.1614598274190113,
        1.182347001119035,
        1.3048667475329754,
        1.4304187646720594,
        1.5053123019505208,
        1.6567582985022575,
        1.727099183257099,
        1.7613622559962399,
        1.7873551899593902,
    ];
    const S_D: [f64; 37] = [
        0.11612195647986145,
        2.286570671419279,
        0.43281480402315803,
        -2.823946636684542,
        0.10299028357493027,
        2.3068525711905274,
        0.3950957416846182,
        -2.7864712784457293,
        0.07385800252794973,
        2.342457241902704,
        0.33444143049657177,
        -2.71854857507006,
        -0.003007898777023888,
        2.456249404673185,
        0.1469183033101349,
        -2.6506653723197506,
        0.0741383971341427,
        2.581125321312145,
        -0.04872975464060914,
        -2.590364519446663,
        -0.09283995102014814,
        2.811044525732896,
        0.0684953240208349,
        -2.750899327511905,
        -0.33664736335837303,
        2.896990376996522,
        0.30837041696305617,
        -2.9085954324631014,
        -0.28606373645371885,
        2.858851132982773,
        0.3834687089169155,
        -2.9563709321899485,
        -0.2944567709915975,
        2.860919506781294,
        0.4185738278504357,
        -2.985719585141617,
        -0.2970207951936875,
    ];
    const W_D: [f64; 37] = [
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0,
    ];

    // case e: n=72 period=6 walk {'robust': True, 'seasonal': 9}
    const Y_E: [f64; 72] = [
        4.042, 5.551, 4.218, 3.099, 4.548, 7.424, 9.549, 8.295, 7.601, 3.885, 4.435, 6.254, 6.491,
        5.891, 2.247, 0.91, 2.648, 6.619, 7.529, 4.301, 1.437, 0.36, 3.008, 5.856, 5.425, 4.295,
        2.234, 2.376, 4.021, 5.98, 8.462, 6.309, 4.615, 4.104, 7.024, 9.063, 9.708, 10.862, 10.036,
        9.95, 11.243, 12.526, 13.671, 11.613, 6.923, 5.188, 8.301, 11.248, 12.66, 12.045, 9.566,
        7.377, 7.505, 9.268, 10.486, 10.731, 7.876, 8.43, 9.713, 10.987, 12.515, 11.422, 8.938,
        9.57, 10.861, 13.347, 14.964, 12.097, 10.042, 8.738, 10.763, 13.8,
    ];
    const T_E: [f64; 72] = [
        3.339306089958577,
        3.936040801996553,
        4.513634527373273,
        5.075725970051677,
        5.614832651267695,
        6.0683623780139815,
        6.181311572026499,
        6.153886286287091,
        6.070148485580003,
        5.898473948102176,
        5.541241355166631,
        5.111083944943771,
        4.66813222273837,
        4.288545143975496,
        4.034269325238091,
        3.890909374187796,
        3.73906765585241,
        3.560939709361493,
        3.459624331809137,
        3.4403711504012495,
        3.3946846794457546,
        3.34982867709257,
        3.3939078100554814,
        3.5084558197784657,
        3.700409001638404,
        3.8968127769838627,
        4.072908914767636,
        4.341796781310583,
        4.702158733733236,
        5.058445212925836,
        5.401853515859843,
        5.806971010092806,
        6.268468654449328,
        6.756668732663874,
        7.360134551973213,
        8.13906742863687,
        9.002951248471529,
        9.818516724457417,
        10.506075819602279,
        11.063411373712176,
        11.39344614357849,
        11.465363428807176,
        11.220689137195905,
        10.853423936605795,
        10.644705192861233,
        10.486584943448353,
        10.331462836588795,
        10.227850324730847,
        10.28987472663667,
        10.14422469058004,
        9.847429251258013,
        9.51545630322874,
        9.240702219063252,
        9.059832640212925,
        9.074539040286753,
        9.279917912435625,
        9.571662000982107,
        9.84376783217375,
        10.059888991860799,
        10.236556699656727,
        10.434106025265493,
        10.656183133857752,
        10.937951965978689,
        11.261390761658166,
        11.520065205611745,
        11.684189475376927,
        11.737357833534594,
        11.708505384958219,
        11.704033954115422,
        11.693595803020095,
        11.677894141351029,
        11.664305406011493,
    ];
    const S_E: [f64; 72] = [
        1.1126902137899433,
        1.8379682033882991,
        -0.927071577446163,
        -2.3238633529327313,
        -1.2284197104748877,
        1.5359778414382725,
        1.4047229533612233,
        1.595614520920293,
        -1.1454993343666757,
        -2.3518690349787548,
        -1.0445088219928764,
        1.5379476124877398,
        1.6769094513632663,
        1.362520392555553,
        -1.3345685339187126,
        -2.3565301285315505,
        -0.8721897999797636,
        1.5089071688733984,
        1.9441842262045688,
        1.1480801878829157,
        -1.5183913666279765,
        -2.3466488367937175,
        -0.7123780044790777,
        1.4804547629483278,
        2.136443552326483,
        0.894099525883592,
        -1.5957807491918263,
        -2.3225593071949175,
        -0.5925688623681842,
        1.447935936177117,
        2.117685328503887,
        0.7537995877402939,
        -1.3190812555563058,
        -2.128220934647446,
        -0.7219658188317613,
        1.1528277157911888,
        2.1215467318319248,
        0.8568272917558482,
        -1.1341572582869293,
        -2.010951974927005,
        -0.8899197129330984,
        0.8950277113474407,
        2.077046539284205,
        1.0388518177054709,
        -1.1293932849701909,
        -1.8717932049623573,
        -0.9320068913964269,
        0.8334259895662277,
        2.1350820675942908,
        1.007931965545156,
        -1.3013537516608538,
        -1.9021995095323365,
        -0.9084178171162189,
        1.0310998679169345,
        2.2578451026966886,
        0.943348384895613,
        -1.4520157504566429,
        -2.018746437089826,
        -0.916577749317464,
        1.2427566354732489,
        2.3936208032606547,
        0.8754973995565433,
        -1.6296243983051377,
        -2.1453100006550945,
        -0.8896514111482849,
        1.4675882585751068,
        2.546493075368808,
        0.7809012506411024,
        -1.835016183294741,
        -2.2928259604857235,
        -0.835198338818193,
        1.7229189005107237,
    ];
    const W_E: [f64; 72] = [
        0.9401165139649331,
        0.9876068156092807,
        0.876950773282701,
        0.9612801236722237,
        0.9919584851011588,
        0.9881418586247281,
        0.20743174403561665,
        0.9170418564428926,
        0.0,
        0.9644944849112693,
        0.9992069563405819,
        0.953852508099292,
        0.9968581477424536,
        0.9812457709849693,
        0.9376624898765162,
        0.8797447877694339,
        0.9828637757719135,
        0.4144038623780093,
        0.12027826338340111,
        0.9708344475119918,
        0.9361147893912269,
        0.8740040807275263,
        0.9667270170094414,
        0.7833753726323832,
        0.9423509675665717,
        0.9273683194305313,
        0.9826285464756584,
        0.9604072839051773,
        0.997720180476704,
        0.9148685738764122,
        0.7529768396354195,
        0.9808597866668539,
        0.9660631941661675,
        0.9172071367009891,
        0.9538031560745991,
        0.9840571222159182,
        0.4771059260415243,
        0.9890107263787425,
        0.8683971207372099,
        0.7673469315935755,
        0.838306888028481,
        0.9916911211086208,
        0.9570748598884854,
        0.9759546419486816,
        0.0,
        0.0,
        0.665610573243546,
        0.9890288666540932,
        0.982736338927162,
        0.7703251192512738,
        0.7070474163811469,
        0.9826181433631739,
        0.8012852544333962,
        0.8027683208231016,
        0.7922314772664304,
        0.9223648278373678,
        0.9815822781413498,
        0.890377388772185,
        0.9020001884367043,
        0.9268013694133308,
        0.9700890121951367,
        0.9962159725017129,
        0.957911345947805,
        0.9374306237042292,
        0.9835909068631806,
        0.9882631495511387,
        0.8626900372898038,
        0.9529699167079625,
        0.9908436728118781,
        0.8695666651620589,
        0.9980522765504592,
        0.9480112199625286,
    ];

    // case f: n=45 period=6 outlier {'seasonal': 35}
    const Y_F: [f64; 45] = [
        1.056, 3.197, 3.911, -0.245, -4.601, -2.297, 1.244, 1.754, 2.532, 0.779, -3.543, -1.519,
        1.139, 3.565, 3.235, 25.844, -2.516, -2.279, 1.154, 4.023, 4.291, 1.902, -2.419, -3.06,
        1.374, 4.344, 4.052, 2.651, -1.69, -1.157, -15.845, 3.859, 4.079, 0.858, -1.596, -0.349,
        1.416, 4.74, 4.014, 1.605, 0.355, -0.696, 3.752, 4.817, 4.537,
    ];
    const T_F: [f64; 45] = [
        0.5612109998433703,
        0.40904308280921975,
        0.3063556110383589,
        0.22385919710585547,
        0.09288356206653685,
        -0.06107135622732654,
        -0.10624822414054202,
        0.014656325116133434,
        0.08007451660115499,
        0.17480881836228845,
        0.34546558881282635,
        1.0492469568824512,
        2.732084319389148,
        4.266955192867556,
        4.889599645363789,
        4.967180414397434,
        4.925277894417086,
        4.40743780439494,
        3.0939602933044923,
        1.6068942617149302,
        1.0262740696598924,
        0.9947585154941324,
        1.0073511748106747,
        1.0672624282725802,
        1.1447399240254628,
        1.2797580965614856,
        1.0521677462295975,
        0.08610739958879997,
        -0.9072612628388329,
        -1.3994989371162496,
        -1.5683716226276025,
        -1.589653319134234,
        -1.164231654289145,
        -0.0933757185688997,
        1.0720090219362588,
        1.534751267723855,
        1.6603674495583198,
        1.8085935594848528,
        1.910476451507037,
        2.0833589830238215,
        2.3070527257477664,
        2.4249224569681234,
        2.586710395846874,
        2.7919327883617777,
        2.9968909089345734,
    ];
    const S_F: [f64; 45] = [
        -0.08345334568748934,
        1.6366687579701773,
        2.2622933452398972,
        5.431115109096656,
        -5.3922930507010385,
        -3.505289669825159,
        -0.6191111992026053,
        1.8810402778161728,
        2.3220436132227444,
        4.834300782291442,
        -4.774952916421866,
        -3.3041102488029876,
        -1.1313723504797544,
        2.122313273146454,
        2.378702967592607,
        4.219940286112908,
        -4.157014623089618,
        -3.102247585149322,
        -1.623120106864433,
        2.3587764651119216,
        2.4301355561852147,
        3.5983008476292553,
        -3.5404359142344077,
        -2.902168758924804,
        -2.0928761240257416,
        2.5935060023538785,
        2.480730507566989,
        2.9521352217578634,
        -2.9210860698571364,
        -2.699118331152398,
        -2.5475910193111697,
        2.830743247941963,
        2.534882383818799,
        2.2798838568181043,
        -2.2968557478495795,
        -2.4888933119395698,
        -2.997565358026707,
        3.073266962087194,
        2.5949563104032625,
        1.5759042735441489,
        -1.667681434069707,
        -2.270656579310069,
        -3.434954163903007,
        3.3192723414156355,
        2.6589654682730384,
    ];
    const W_F: [f64; 45] = [
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    ];

    fn check(y: &[f64], t: &[f64], s: &[f64], w: &[f64], stl: Stl) {
        let r = stl.decompose(y).unwrap();
        let scale = y.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        let d = |a: &[f64], b: &[f64]| {
            a.iter()
                .zip(b)
                .fold(0.0_f64, |m, (x, y)| m.max((x - y).abs()))
        };
        let (dt, ds, dw) = (d(&r.trend, t), d(&r.seasonal, s), d(&r.weights, w));
        assert!(dt <= 1e-10 * scale && ds <= 1e-10 * scale && dw <= 1e-10);
    }

    #[test]
    fn matches_statsmodels() {
        check(&Y_A, &T_A, &S_A, &W_A, Stl::new(12));
    }
    #[test]
    fn matches_statsmodels_b() {
        check(&Y_B, &T_B, &S_B, &W_B, Stl::new(7).robust(true));
        check(
            &Y_C,
            &T_C,
            &S_C,
            &W_C,
            Stl::new(5).seasonal(11).trend(17).low_pass(7),
        );
    }
    #[test]
    fn matches_statsmodels_d() {
        check(
            &Y_D,
            &T_D,
            &S_D,
            &W_D,
            Stl::new(4)
                .seasonal_degree(0)
                .trend_degree(0)
                .low_pass_degree(0),
        );
    }
    #[test]
    fn matches_statsmodels_e() {
        check(&Y_E, &T_E, &S_E, &W_E, Stl::new(6).robust(true).seasonal(9));
    }
    #[test]
    fn matches_statsmodels_f() {
        check(&Y_F, &T_F, &S_F, &W_F, Stl::new(6).seasonal(35));
    }
}
