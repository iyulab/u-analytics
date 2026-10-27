//! Refusals shared by the functions of more than one module.

/// Why a time-series statistic was refused: [`crate::correlation::acf`],
/// [`crate::correlation::pacf`], [`crate::testing::adf_test`] and
/// [`crate::testing::kpss_test`].
///
/// The data is checked before the options, and the options before any
/// computation: a refusal of an option never hides a refusal of the data,
/// and [`TimeSeriesError::NoVariation`] / [`TimeSeriesError::DegenerateRegression`]
/// only follow options that were in range.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum TimeSeriesError {
    /// The series holds a NaN or an infinity.
    ValueNotFinite {
        /// Position of the first such value.
        index: usize,
    },
    /// Fewer observations than the statistic is defined for.
    TooFewObservations {
        /// The fewest it takes.
        needed: usize,
        /// How many were given.
        got: usize,
    },
    /// An option outside its domain, which depends on the number of
    /// observations: `min` and `max` are both inclusive.
    OptionOutOfRange {
        /// The option's name, spelled as the wire formats spell it
        /// (`max_lag`, `max_lags`, `lags`).
        option: &'static str,
        /// The value given (a count as a float).
        got: f64,
        /// Inclusive lower bound.
        min: f64,
        /// Inclusive upper bound.
        max: f64,
    },
    /// The series does not vary (about its level, or its trend, for the KPSS
    /// test) beyond the rounding of its own size, so the statistic divides
    /// by zero.
    NoVariation,
    /// The ADF test regression is singular, or fits the differences exactly
    /// (a deterministic series), so the t-ratio is undefined.
    DegenerateRegression,
}

impl core::fmt::Display for TimeSeriesError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            TimeSeriesError::ValueNotFinite { index } => {
                write!(f, "the value at index {index} is not finite")
            }
            TimeSeriesError::TooFewObservations { needed, got } => {
                write!(f, "at least {needed} values are needed, got {got}")
            }
            TimeSeriesError::OptionOutOfRange {
                option,
                got,
                min,
                max,
            } => write!(f, "`{option}` must be in [{min}, {max}] (got {got})"),
            TimeSeriesError::NoVariation => {
                write!(f, "the series does not vary, so the statistic is undefined")
            }
            TimeSeriesError::DegenerateRegression => write!(
                f,
                "the test regression is singular, or an exact fit of the differences \
                 (a deterministic series), so the t-ratio is undefined"
            ),
        }
    }
}

impl std::error::Error for TimeSeriesError {}

/// The series' own checks, in the order every time-series statistic applies
/// them: a non-finite value, then too few values.
pub(crate) fn check_series(data: &[f64], needed: usize) -> Result<(), TimeSeriesError> {
    if let Some(index) = data.iter().position(|v| !v.is_finite()) {
        return Err(TimeSeriesError::ValueNotFinite { index });
    }
    if data.len() < needed {
        return Err(TimeSeriesError::TooFewObservations {
            needed,
            got: data.len(),
        });
    }
    Ok(())
}

/// Refuses a count option outside `[min, max]`.
pub(crate) fn check_option(
    option: &'static str,
    value: usize,
    min: usize,
    max: usize,
) -> Result<usize, TimeSeriesError> {
    if (min..=max).contains(&value) {
        Ok(value)
    } else {
        Err(TimeSeriesError::OptionOutOfRange {
            option,
            got: value as f64,
            min: min as f64,
            max: max as f64,
        })
    }
}
