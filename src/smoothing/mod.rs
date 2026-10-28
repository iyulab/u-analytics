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
//! # Prediction intervals
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
//! with the constants of the component form used here (the state-space
//! form's trend and seasonal constants are αβ and γ(1 − α)), and σ² the mean
//! squared one-step error over the observations the smoother updates on (not
//! those its initialisation reproduces). Multiplicative seasonality has no
//! such closed form, so its forecasts carry no interval. The seasonal update is
//! Winters' (on `x_t − L_t`), which is the state-space model's.
//!
//! # Estimating the constants
//!
//! `fit` sets the constants left unset to those minimising the sum of squared
//! one-step errors over the same observations (the least-squares estimate,
//! Hyndman et al. 2008 §5.1). Each free constant is searched on the logistic
//! scale, so it stays strictly inside (0, 1): a 5-point grid per constant
//! gives the start, then the Nelder–Mead simplex method refines it. The search
//! is deterministic and local.
//!
//! # References
//!
//! - Brown, R.G. (1956). *Exponential Smoothing for Predicting Demand*.
//! - Holt, C.C. (1957). "Forecasting Seasonals and Trends by
//!   Exponentially Weighted Moving Averages", ONR Memo 52.
//! - Hyndman, R.J., Koehler, A.B., Ord, J.K. & Snyder, R.D. (2008).
//!   *Forecasting with Exponential Smoothing: The State Space Approach*.
//!   Springer.
//! - Winters, P.R. (1960). "Forecasting Sales by Exponentially Weighted
//!   Moving Averages", *Management Science* 6(3), pp. 324-342.

mod estimate;
mod forecast;
mod holt;
mod holt_winters;
mod ses;

pub use forecast::SmoothingForecast;

pub use holt::{HoltLinear, HoltResult};
pub use holt_winters::{HoltWinters, HoltWintersResult, Seasonality};
pub use ses::{SesResult, SimpleExponentialSmoothing};
