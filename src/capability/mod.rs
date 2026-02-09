//! Process capability analysis.
//!
//! Computes standard capability indices for assessing how well a process
//! meets specification limits.
//!
//! # Indices
//!
//! - **Cp** — Potential capability (spread vs tolerance)
//! - **Cpk** — Actual capability (centering considered)
//! - **Pp**, **Ppk** — Long-term performance indices
//! - **Cpm** — Taguchi capability (target deviation)
//!
//! # Sigma Level
//!
//! - [`sigma_to_ppm`] — Convert sigma level to PPM defect rate
//! - [`ppm_to_sigma`] — Convert PPM defect rate to sigma level
//!
//! # References
//!
//! - Montgomery (2019), *Introduction to Statistical Quality Control*, 8th ed.

mod indices;
mod sigma_level;

pub use indices::{CapabilityIndices, ProcessCapability};
pub use sigma_level::{ppm_to_sigma, sigma_to_ppm};
