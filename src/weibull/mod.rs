//! Weibull parameter estimation and reliability analysis.
//!
//! Provides maximum likelihood estimation (MLE) and median rank regression
//! (MRR) for fitting Weibull distributions to failure data.
//!
//! # Modules
//!
//! - [`weibull_mle`] — Newton-Raphson MLE for shape and scale parameters
//! - [`weibull_mrr`] — Median Rank Regression via Bernard's approximation
//! - [`ReliabilityAnalysis`] — R(t), hazard rate, MTBF, B-life from fitted parameters
//!
//! # References
//!
//! - Abernethy, R.B. (2006). *The New Weibull Handbook*, 5th ed.
//! - Dodson, B. (2006). *The Weibull Analysis Handbook*, 2nd ed.

mod mle;
mod mrr;
mod reliability;

pub use mle::{weibull_mle, WeibullMleResult};
pub use mrr::{weibull_mrr, WeibullMrrResult};
pub use reliability::ReliabilityAnalysis;
