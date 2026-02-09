//! # u-analytics
//!
//! Statistical process control (SPC), process capability analysis, Weibull
//! reliability, and change-point detection.
//!
//! This crate provides industrial quality analysis tools that are
//! domain-agnostic — they operate on raw `f64` data without knowledge
//! of manufacturing, scheduling, or any specific consumer domain.
//!
//! ## Modules
//!
//! - [`spc`] — Control charts (X̄-R, X̄-S, I-MR, P, NP, C, U) with run rules
//! - [`capability`] — Process capability indices (Cp, Cpk, Pp, Ppk, Cpm)
//! - [`weibull`] — Weibull parameter estimation (MLE, MRR) and reliability analysis
//! - [`detection`] — Change-point detection (CUSUM, EWMA)
//!
//! ## Design Philosophy
//!
//! - **Domain-agnostic**: No manufacturing or process-specific types
//! - **Numerical stability**: Leverages `u-optim` for stable statistics
//! - **Research-backed**: All algorithms reference academic literature

pub mod capability;
pub mod detection;
pub mod spc;
pub mod weibull;
