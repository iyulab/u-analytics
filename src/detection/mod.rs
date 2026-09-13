//! Change-point and shift detection.
//!
//! Algorithms for detecting process mean shifts and trend changes, covering
//! both **online** (sequential surveillance) and **offline** (retrospective)
//! approaches.
//!
//! # Online Detection (Sequential Surveillance)
//!
//! - [`Cusum`] — Cumulative Sum chart (Page, 1954) for detecting small persistent shifts
//! - [`Ewma`] — Exponentially Weighted Moving Average chart (Roberts, 1959)
//!
//! # Offline Detection (Retrospective Changepoint Analysis)
//!
//! - [`Pelt`] — Pruned Exact Linear Time algorithm (Killick et al., 2012) for
//!   detecting multiple changepoints with O(n) expected complexity
//! - [`SpectralResidual`] — Saliency-based one-shot anomaly scoring
//!   (Ren et al., 2019): spikes, steps and dropouts without a trained model
//!
//! # References
//!
//! - Page, E.S. (1954). "Continuous Inspection Schemes",
//!   *Biometrika* 41(1/2), pp. 100-115.
//! - Roberts, S.W. (1959). "Control Chart Tests Based on Geometric Moving Averages",
//!   *Technometrics* 1(3), pp. 239-250.
//! - Killick, R., Fearnhead, P., & Eckley, I.A. (2012). "Optimal Detection of
//!   Changepoints with a Linear Computational Cost", *JASA* 107(500), pp. 1590-1598.
//! - Ren, H. et al. (2019). "Time-Series Anomaly Detection Service at Microsoft",
//!   *KDD 2019*, pp. 3009-3017.

mod cusum;
mod ewma;
mod pelt;
mod spectral_residual;

pub use cusum::{Cusum, CusumResult};
pub use ewma::{Ewma, EwmaResult};
pub use pelt::{CostFunction, MultiPeltResult, Pelt, PeltResult, Penalty};
pub use spectral_residual::{SpectralResidual, SrPoint, MIN_OBSERVATIONS as SR_MIN_OBSERVATIONS};
