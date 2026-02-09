//! Change-point and shift detection.
//!
//! Algorithms for detecting process mean shifts and trend changes.
//!
//! # Charts
//!
//! - [`Cusum`] — Cumulative Sum chart (Page, 1954) for detecting small persistent shifts
//! - [`Ewma`] — Exponentially Weighted Moving Average chart (Roberts, 1959)
//!
//! # References
//!
//! - Page, E.S. (1954). "Continuous Inspection Schemes",
//!   *Biometrika* 41(1/2), pp. 100-115.
//! - Roberts, S.W. (1959). "Control Chart Tests Based on Geometric Moving Averages",
//!   *Technometrics* 1(3), pp. 239-250.

mod cusum;
mod ewma;

pub use cusum::{Cusum, CusumResult};
pub use ewma::{Ewma, EwmaResult};
