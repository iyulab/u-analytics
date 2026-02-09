//! Statistical Process Control (SPC) charts.
//!
//! Provides control chart implementations for both variables and attributes data.
//!
//! # Variables Charts
//!
//! - [`XBarRChart`] — X-bar and Range chart for subgroup data (n=2..10)
//! - [`XBarSChart`] — X-bar and Standard Deviation chart for subgroup data (n=2..10)
//! - [`IndividualMRChart`] — Individual and Moving Range chart for individual observations
//!
//! # Attributes Charts
//!
//! - [`PChart`] — Proportion nonconforming (variable sample size)
//! - [`NPChart`] — Count of nonconforming items (constant sample size)
//! - [`CChart`] — Count of defects per unit (constant area of opportunity)
//! - [`UChart`] — Defects per unit (variable area of opportunity)
//!
//! # Run Rules
//!
//! - [`WesternElectricRules`] — 4 classic run rules
//! - [`NelsonRules`] — 8 rules (superset of Western Electric)
//!
//! # References
//!
//! - Montgomery, D.C. (2019). *Introduction to Statistical Quality Control*, 8th ed.
//! - ASTM E2587 — Standard Practice for Use of Control Charts
//! - Nelson, L.S. (1984). "The Shewhart Control Chart — Tests for Special Causes",
//!   *Journal of Quality Technology* 16(4), pp. 237-239.

mod attributes;
mod chart;
mod rules;
mod variables;

pub use attributes::{CChart, NPChart, PChart, UChart};
pub use chart::{ChartPoint, ControlChart, ControlLimits, Violation, ViolationType};
pub use rules::{NelsonRules, RunRule, WesternElectricRules};
pub use variables::{IndividualMRChart, XBarRChart, XBarSChart};
