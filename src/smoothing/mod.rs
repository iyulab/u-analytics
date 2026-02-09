//! Time series smoothing and forecasting.
//!
//! Exponential smoothing methods for level, trend, and seasonal
//! decomposition of time series data.
//!
//! # Methods
//!
//! - [`SimpleExponentialSmoothing`] — Level-only smoothing (Brown, 1956)
//! - [`HoltLinear`] — Double exponential smoothing with trend (Holt, 1957)
//! - [`HoltWinters`] — Triple exponential smoothing with trend and seasonality
//!   (Winters, 1960)
//!
//! # References
//!
//! - Brown, R.G. (1956). *Exponential Smoothing for Predicting Demand*.
//! - Holt, C.C. (1957). "Forecasting Seasonals and Trends by
//!   Exponentially Weighted Moving Averages", ONR Memo 52.
//! - Winters, P.R. (1960). "Forecasting Sales by Exponentially Weighted
//!   Moving Averages", *Management Science* 6(3), pp. 324-342.

mod holt;
mod holt_winters;
mod ses;

pub use holt::{HoltLinear, HoltResult};
pub use holt_winters::{HoltWinters, HoltWintersResult, Seasonality};
pub use ses::{SesResult, SimpleExponentialSmoothing};
