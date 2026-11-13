//! Statistical Process Control (SPC) charts.
//!
//! Provides control chart implementations for both variables and attributes data.
//!
//! # Variables Charts
//!
//! - [`XBarRChart`] — X-bar and Range chart for subgroup data (n=2..=25)
//! - [`XBarSChart`] — X-bar and Standard Deviation chart for subgroup data (n=2..=25)
//! - [`IndividualMRChart`] — Individual and Moving Range chart for individual observations
//!
//! # Attributes Charts
//!
//! - [`PChart`] — Proportion nonconforming (variable sample size)
//! - [`NPChart`] — Count of nonconforming items (constant sample size)
//! - [`CChart`] — Count of defects per unit (constant area of opportunity)
//! - [`UChart`] — Defects per unit (variable area of opportunity)
//!
//! # Short-Run Charts
//!
//! - [`deviation_chart`], [`standardized_chart`] — I-MR charts of `x − target`
//!   and `(x − target) / spread` for a process that runs in groups
//! - [`standardized_p_chart`], [`standardized_u_chart`] — P and U points against
//!   the centre of each point's own group
//!
//! # Multivariate Charts
//!
//! - [`hotelling`] — Hotelling's T² for individual observations of correlated variables,
//!   Phase I (exact Beta limit) and Phase II (exact F limit)
//!
//! # Run Rules
//!
//! - [`RuleSet`] — any subset of the tests; [`RuleSet::shewhart`] (the 3-sigma
//!   test alone) is the default, [`RuleSet::named`] looks up a named set
//! - [`WesternElectricRules`] — the 4 classic run rules (eight points on one side)
//! - [`NelsonRules`] — Nelson's 8 rules (nine points on one side)
//!
//! # References
//!
//! - Montgomery, D.C. (2019). *Introduction to Statistical Quality Control*, 8th ed.
//! - ASTM E2587 — Standard Practice for Use of Control Charts
//! - Nelson, L.S. (1984). "The Shewhart Control Chart — Tests for Special Causes",
//!   *Journal of Quality Technology* 16(4), pp. 237-239.

mod attributes;
mod chart;
pub mod hotelling;
mod rules;
mod short_run;
mod variables;

pub use attributes::{
    g_chart, laney_p_chart, laney_u_chart, t_chart, AttributeChartPoint, CChart, GChart,
    GChartPoint, LaneyAttributePoint, LaneyPChart, LaneyStandard, LaneyUChart, NPChart, PChart,
    TChart, TChartPoint, UChart,
};
pub use chart::{
    ChartInputError, ChartPoint, ControlChart, ControlChartError, ControlLimits, Violation,
    ViolationType, MAX_SUBGROUP_SIZE, MIN_SUBGROUP_SIZE,
};
pub use rules::{NelsonRules, RuleSet, RunRule, WesternElectricRules};
pub use short_run::{
    deviation_chart, standardized_chart, standardized_p_chart, standardized_u_chart, GroupStandard,
    ShortRunAttributeChart, ShortRunChart, ShortRunError, ShortRunScale, MIN_GROUP_OBSERVATIONS,
    MIN_GROUP_SAMPLES,
};
pub use variables::{range_chart_factors, IndividualMRChart, XBarRChart, XBarSChart};
