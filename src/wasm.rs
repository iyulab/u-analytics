//! WASM bindings for u-analytics.
//!
//! Exposes SPC (Statistical Process Control) and capability analysis functions
//! to JavaScript/TypeScript via `wasm-bindgen`.
//!
//! # Feature
//!
//! Only compiled when the `wasm` feature is enabled:
//! ```toml
//! [dependencies]
//! u-analytics = { version = "...", features = ["wasm"] }
//! ```

use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

// ---------------------------------------------------------------------------
// Serializable DTO types
// ---------------------------------------------------------------------------

#[derive(Serialize, Debug)]
struct XbarRChartDto {
    xbar_cl: f64,
    xbar_ucl: f64,
    xbar_lcl: f64,
    r_cl: f64,
    r_ucl: f64,
    r_lcl: f64,
    /// Short-term sigma implied by this chart (`R-bar / d2`). Feed it to
    /// `process_capability` as `sigma_within` -- it is the quantity a
    /// capability study needs and the one a flat measurement vector cannot
    /// carry.
    sigma_hat: Option<f64>,
    xbar_points: Vec<ChartPointDto>,
    r_points: Vec<ChartPointDto>,
    in_control: bool,
}

#[derive(Serialize, Debug)]
struct ChartPointDto {
    index: usize,
    value: f64,
    violations: Vec<String>,
}

#[derive(Serialize, Debug)]
struct XbarSChartDto {
    xbar_cl: f64,
    xbar_ucl: f64,
    xbar_lcl: f64,
    s_cl: f64,
    s_ucl: f64,
    s_lcl: f64,
    /// Short-term sigma implied by this chart (`S-bar / c4`), for
    /// `process_capability`'s `sigma_within`.
    sigma_hat: Option<f64>,
    xbar_points: Vec<ChartPointDto>,
    s_points: Vec<ChartPointDto>,
    in_control: bool,
}

#[derive(Serialize, Debug)]
struct ImrChartDto {
    i_cl: f64,
    i_ucl: f64,
    i_lcl: f64,
    mr_cl: f64,
    mr_ucl: f64,
    mr_lcl: f64,
    /// Short-term sigma implied by this chart (`MR-bar / d2(2)`).
    sigma_hat: Option<f64>,
    i_points: Vec<ChartPointDto>,
    /// Starts at index 1: the first value has no moving range.
    mr_points: Vec<ChartPointDto>,
    in_control: bool,
}

/// Control limits a caller supplies to `run_rules`.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LimitsInputDto {
    ucl: f64,
    cl: f64,
    lcl: f64,
}

#[derive(Serialize, Debug)]
struct PChartDto {
    p_bar: f64,
    points: Vec<AttributeChartPointDto>,
    in_control: bool,
}

#[derive(Serialize, Debug)]
struct AttributeChartPointDto {
    index: usize,
    value: f64,
    ucl: f64,
    cl: f64,
    lcl: f64,
    out_of_control: bool,
}

/// Input for `process_capability`.
///
/// Every field except `data` is optional, but at least one of `usl`/`lsl` must
/// be present -- a capability index without a specification limit is undefined.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CapabilityInputDto {
    data: Vec<f64>,
    #[serde(default)]
    usl: Option<f64>,
    #[serde(default)]
    lsl: Option<f64>,
    /// Short-term (within-subgroup) sigma, normally estimated from a control
    /// chart as R-bar/d2 or S-bar/c4. It cannot be recovered from `data`: the
    /// subgroup structure is not in a flat measurement vector.
    #[serde(default)]
    sigma_within: Option<f64>,
    /// Process target for Cpm. Without it `cpm` is `null`.
    #[serde(default)]
    target: Option<f64>,
}

#[derive(Serialize, Debug)]
struct CapabilityDto {
    mean: f64,
    /// `"within"` when `sigma_within` was supplied, `"overall"` otherwise.
    /// Without it the short-term indices are not computed at all rather than
    /// being filled with the long-term sigma -- a number under the wrong name
    /// is harder to notice than a null.
    sigma_source: &'static str,
    std_dev_within: Option<f64>,
    std_dev_overall: f64,
    cp: Option<f64>,
    cpk: Option<f64>,
    cpu: Option<f64>,
    cpl: Option<f64>,
    pp: Option<f64>,
    ppk: Option<f64>,
    ppu: Option<f64>,
    ppl: Option<f64>,
    cpm: Option<f64>,
}

#[derive(Serialize)]
struct AdNormalityDto {
    statistic: f64,
    statistic_modified: f64,
    p_value: f64,
}

#[derive(Serialize, Debug)]
struct LaneyPChartDto {
    p_bar: f64,
    phi: f64,
    points: Vec<AttributeChartPointDto>,
}

/// NP and C charts: one set of limits for every point.
#[derive(Serialize, Debug)]
struct FixedLimitChartDto {
    cl: f64,
    ucl: f64,
    lcl: f64,
    points: Vec<AttributeChartPointDto>,
    in_control: bool,
}

#[derive(Serialize, Debug)]
struct UChartDto {
    u_bar: f64,
    points: Vec<AttributeChartPointDto>,
    in_control: bool,
}

#[derive(Serialize, Debug)]
struct LaneyUChartDto {
    u_bar: f64,
    phi: f64,
    points: Vec<AttributeChartPointDto>,
}
#[derive(Serialize)]
struct GChartDto {
    g_bar: f64,
    points: Vec<GChartPointDto>,
}

#[derive(Serialize)]
struct GChartPointDto {
    index: usize,
    value: f64,
    ucl: f64,
    cl: f64,
    lcl: f64,
    out_of_control: bool,
}

#[derive(Serialize)]
struct TChartDto {
    t_bar: f64,
    points: Vec<TChartPointDto>,
}

#[derive(Serialize)]
struct TChartPointDto {
    index: usize,
    value: f64,
    ucl: f64,
    cl: f64,
    lcl: f64,
    out_of_control: bool,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn js_err(msg: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&msg.to_string())
}

fn to_js<T: Serialize>(val: &T) -> Result<JsValue, JsValue> {
    serde_wasm_bindgen::to_value(val).map_err(js_err)
}

/// Deserialize a native JS value, rejecting JSON strings with an actionable
/// message and prefixing the offending parameter name to any serde error.
fn from_js<T: serde::de::DeserializeOwned>(value: JsValue, param: &str) -> Result<T, JsValue> {
    if value.as_string().is_some() {
        return Err(js_err(format!(
            "{param}: expected a native JS object/array, got a string — \
             pass the value directly, not JSON.stringify(...)"
        )));
    }
    // serde-wasm-bindgen reads only a struct's declared fields from a JS
    // object, so `deny_unknown_fields` never sees extra keys. Round-trip
    // through serde_json::Value so the strict wire schema is enforced.
    let json: serde_json::Value =
        serde_wasm_bindgen::from_value(value).map_err(|e| js_err(format!("{param}: {e}")))?;
    from_json(json, param).map_err(js_err)
}

/// The half of [`from_js`] that enforces the wire schema, split out so it can
/// be exercised without a `JsValue` -- which cannot be constructed off
/// `wasm32`. Tests calling this walk the same deserialization path a JS caller
/// does, rather than a parallel one that could drift from it.
fn from_json<T: serde::de::DeserializeOwned>(
    json: serde_json::Value,
    param: &str,
) -> Result<T, String> {
    serde_json::from_value(json).map_err(|e| format!("{param}: {e}"))
}

fn violation_name(v: crate::spc::ViolationType) -> &'static str {
    use crate::spc::ViolationType;
    match v {
        ViolationType::BeyondLimits => "BeyondLimits",
        ViolationType::NineOneSide => "NineOneSide",
        ViolationType::SixTrend => "SixTrend",
        ViolationType::FourteenAlternating => "FourteenAlternating",
        ViolationType::TwoOfThreeBeyond2Sigma => "TwoOfThreeBeyond2Sigma",
        ViolationType::FourOfFiveBeyond1Sigma => "FourOfFiveBeyond1Sigma",
        ViolationType::FifteenWithin1Sigma => "FifteenWithin1Sigma",
        ViolationType::EightBeyond1Sigma => "EightBeyond1Sigma",
    }
}

// ---------------------------------------------------------------------------
// WASM exports
// ---------------------------------------------------------------------------

/// Parse an optional `{ rules: [...] }` options object into a rule set.
///
/// Absent, `undefined`, `null` or an object without `rules` all mean "the
/// default set", so a caller that never passes options keeps the behaviour it
/// had. Split out from the binding so it can be exercised without a `JsValue`.
fn rules_from_json(options: Option<serde_json::Value>) -> Result<crate::spc::RuleSet, String> {
    use crate::spc::{RuleSet, ViolationType};

    let Some(value) = options else {
        return Ok(RuleSet::default());
    };
    if value.is_null() {
        return Ok(RuleSet::default());
    }
    let Some(names) = value.get("rules") else {
        return Ok(RuleSet::default());
    };
    if names.is_null() {
        return Ok(RuleSet::default());
    }
    let names = names
        .as_array()
        .ok_or_else(|| "rules: expected an array of rule names".to_string())?;

    let mut set = RuleSet::none();
    for name in names {
        let name = name
            .as_str()
            .ok_or_else(|| "rules: expected an array of rule names".to_string())?;
        let rule = match name {
            "BeyondLimits" => ViolationType::BeyondLimits,
            "NineOneSide" => ViolationType::NineOneSide,
            "SixTrend" => ViolationType::SixTrend,
            "FourteenAlternating" => ViolationType::FourteenAlternating,
            "TwoOfThreeBeyond2Sigma" => ViolationType::TwoOfThreeBeyond2Sigma,
            "FourOfFiveBeyond1Sigma" => ViolationType::FourOfFiveBeyond1Sigma,
            "FifteenWithin1Sigma" => ViolationType::FifteenWithin1Sigma,
            "EightBeyond1Sigma" => ViolationType::EightBeyond1Sigma,
            other => {
                return Err(format!(
                    "rules: unknown rule {other:?} -- the names are the values                      `violations` reports"
                ))
            }
        };
        set = set.with(rule);
    }
    Ok(set)
}

/// Compute an X-bar R chart from subgroups.
///
/// # Input JSON
///
/// Array of arrays: `[[x1, x2, ...], [x1, x2, ...], ...]`
/// All subgroups must have the same length, within the range the crate's
/// factor tables cover (see `MIN_SUBGROUP_SIZE`/`MAX_SUBGROUP_SIZE`).
///
/// # Output JSON
///
/// Object with fields: `xbar_cl`, `xbar_ucl`, `xbar_lcl`, `r_cl`, `r_ucl`,
/// `r_lcl`, `sigma_hat`, `xbar_points`, `r_points`, `in_control`.
///
/// `sigma_hat` is `R-bar / d2` -- the short-term sigma a capability study
/// needs. Pass it to [`process_capability`] as `sigma_within`; it is `null`
/// when there is not enough data for control limits.
///
/// # Options (optional second argument)
///
/// `{ rules?: string[] }` -- which run tests to apply. Names are the same
/// values that appear in each point's `violations`, so the set is written in
/// the vocabulary the output already uses:
/// `BeyondLimits`, `NineOneSide`, `SixTrend`, `FourteenAlternating`,
/// `TwoOfThreeBeyond2Sigma`, `FourOfFiveBeyond1Sigma`, `FifteenWithin1Sigma`,
/// `EightBeyond1Sigma`.
///
/// Omitted, `undefined`, `null`, or an object without `rules` all mean all
/// eight (Nelson), which is what this binding did before the option existed.
/// `{ rules: [] }` applies none, leaving control limits only.
#[wasm_bindgen]
pub fn xbar_r_chart(data: JsValue, options: Option<JsValue>) -> Result<JsValue, JsValue> {
    let subgroups: Vec<Vec<f64>> = from_js(data, "data")?;
    let rules = rules_option(options)?;
    to_js(&xbar_r_dto(subgroups, rules).map_err(js_err)?)
}

/// Compute an X-bar S chart from subgroups.
///
/// # Input JSON
///
/// Array of arrays, as for [`xbar_r_chart`]. The standard deviation uses
/// every value in a subgroup rather than only the extremes, which is why this
/// chart is the usual choice once subgroups exceed about ten.
///
/// # Output JSON
///
/// Object with fields: `xbar_cl`, `xbar_ucl`, `xbar_lcl`, `s_cl`, `s_ucl`,
/// `s_lcl`, `sigma_hat` (`S-bar / c4`), `xbar_points`, `s_points`,
/// `in_control`.
///
/// # Options (optional second argument)
///
/// `{ rules?: string[] }`, exactly as for [`xbar_r_chart`].
#[wasm_bindgen]
pub fn xbar_s_chart(data: JsValue, options: Option<JsValue>) -> Result<JsValue, JsValue> {
    let subgroups: Vec<Vec<f64>> = from_js(data, "data")?;
    let rules = rules_option(options)?;
    to_js(&xbar_s_dto(subgroups, rules).map_err(js_err)?)
}

/// Compute an Individual and Moving Range (I-MR) chart.
///
/// # Input JSON
///
/// Array of individual observations in time order: `[x1, x2, ...]` (need >= 2).
///
/// # Output JSON
///
/// Object with fields: `i_cl`, `i_ucl`, `i_lcl`, `mr_cl`, `mr_ucl`, `mr_lcl`,
/// `sigma_hat` (`MR-bar / d2(2)`), `i_points`, `mr_points`, `in_control`.
/// `mr_points` starts at index 1: the first observation has no moving range.
///
/// # Options (optional second argument)
///
/// `{ rules?: string[] }`, exactly as for [`xbar_r_chart`].
#[wasm_bindgen]
pub fn imr_chart(values: JsValue, options: Option<JsValue>) -> Result<JsValue, JsValue> {
    let values: Vec<f64> = from_js(values, "values")?;
    let rules = rules_option(options)?;
    to_js(&imr_dto(values, rules).map_err(js_err)?)
}

/// Apply run tests to a series against control limits the caller supplies.
///
/// The same engine the charts use, reachable on its own: for a statistic the
/// crate does not chart, or limits fixed from an earlier phase-I study.
///
/// # Input JSON
///
/// - `values`: `[x1, x2, ...]`
/// - `limits`: `{ ucl, cl, lcl }` with `lcl <= cl <= ucl`
/// - `options` (optional): `{ rules?: string[] }`, as for [`xbar_r_chart`]
///
/// # Output JSON
///
/// One `{ index, value, violations }` per input value, in input order.
#[wasm_bindgen]
pub fn run_rules(
    values: JsValue,
    limits: JsValue,
    options: Option<JsValue>,
) -> Result<JsValue, JsValue> {
    let values: Vec<f64> = from_js(values, "values")?;
    let limits: LimitsInputDto = from_js(limits, "limits")?;
    let rules = rules_option(options)?;
    to_js(&run_rules_dto(values, limits, rules).map_err(js_err)?)
}

// ---------------------------------------------------------------------------
// Pure cores of the variables-chart bindings
// ---------------------------------------------------------------------------
//
// Each binding above is a thin adapter over one of these, so the contract a
// JavaScript caller observes can be tested on a host with no WebAssembly
// runner.

/// The optional `{ rules?: [...] }` argument the variables charts and
/// `run_rules` share.
fn rules_option(options: Option<JsValue>) -> Result<crate::spc::RuleSet, JsValue> {
    match options {
        Some(o) if !o.is_undefined() && !o.is_null() => {
            rules_from_json(Some(from_js(o, "options")?)).map_err(js_err)
        }
        _ => Ok(crate::spc::RuleSet::default()),
    }
}

fn point_dtos(points: &[crate::spc::ChartPoint]) -> Vec<ChartPointDto> {
    points
        .iter()
        .map(|p| ChartPointDto {
            index: p.index,
            value: p.value,
            violations: p
                .violations
                .iter()
                .map(|&v| violation_name(v).to_owned())
                .collect(),
        })
        .collect()
}

/// Checks the matrix a subgroup chart takes and returns its subgroup size.
///
/// A ragged subgroup is refused by its row: the chart would skip it, and every
/// later point would then carry an index one short of the row it came from.
///
/// The supported subgroup range is not restated here. It used to be, as a
/// literal `2..=10` alongside the same literal in the constructor, so widening
/// the crate's factor tables left the binding rejecting sizes the crate had
/// just learned to handle -- a disagreement no test in either crate could see,
/// because each one was right about its own copy.
fn subgroup_size(subgroups: &[Vec<f64>]) -> Result<usize, String> {
    let n = subgroups
        .first()
        .map(Vec::len)
        .ok_or("at least one subgroup required")?;
    if let Some(i) = subgroups.iter().position(|g| g.len() != n) {
        return Err(format!(
            "subgroup {i} has {} values; subgroup 0 has {n} -- all subgroups must have the same size",
            subgroups[i].len()
        ));
    }
    Ok(n)
}

fn xbar_r_dto(
    subgroups: Vec<Vec<f64>>,
    rules: crate::spc::RuleSet,
) -> Result<XbarRChartDto, String> {
    use crate::spc::{ControlChart, XBarRChart};

    let n = subgroup_size(&subgroups)?;
    let mut chart = XBarRChart::new(n)
        .map_err(|e| e.to_string())?
        .with_rules(rules);
    for subgroup in &subgroups {
        chart.add_sample(subgroup);
    }
    let x = chart
        .control_limits()
        .ok_or("insufficient data for control limits")?;
    let r = chart
        .r_limits()
        .ok_or("insufficient data for R chart limits")?;
    Ok(XbarRChartDto {
        xbar_cl: x.cl,
        xbar_ucl: x.ucl,
        xbar_lcl: x.lcl,
        r_cl: r.cl,
        r_ucl: r.ucl,
        r_lcl: r.lcl,
        sigma_hat: chart.sigma_hat(),
        xbar_points: point_dtos(chart.points()),
        r_points: point_dtos(chart.r_points()),
        in_control: chart.is_in_control(),
    })
}

fn xbar_s_dto(
    subgroups: Vec<Vec<f64>>,
    rules: crate::spc::RuleSet,
) -> Result<XbarSChartDto, String> {
    use crate::spc::{ControlChart, XBarSChart};

    let n = subgroup_size(&subgroups)?;
    let mut chart = XBarSChart::new(n)
        .map_err(|e| e.to_string())?
        .with_rules(rules);
    for subgroup in &subgroups {
        chart.add_sample(subgroup);
    }
    let x = chart
        .control_limits()
        .ok_or("insufficient data for control limits")?;
    let s = chart
        .s_limits()
        .ok_or("insufficient data for S chart limits")?;
    Ok(XbarSChartDto {
        xbar_cl: x.cl,
        xbar_ucl: x.ucl,
        xbar_lcl: x.lcl,
        s_cl: s.cl,
        s_ucl: s.ucl,
        s_lcl: s.lcl,
        sigma_hat: chart.sigma_hat(),
        xbar_points: point_dtos(chart.points()),
        s_points: point_dtos(chart.s_points()),
        in_control: chart.is_in_control(),
    })
}

fn imr_dto(values: Vec<f64>, rules: crate::spc::RuleSet) -> Result<ImrChartDto, String> {
    use crate::spc::{ControlChart, IndividualMRChart};

    // Unreachable over the wire -- JSON has no NaN or infinity -- but the
    // chart would skip such a value and renumber everything after it.
    if let Some(i) = values.iter().position(|x| !x.is_finite()) {
        return Err(format!("values[{i}] is not a finite number"));
    }
    let mut chart = IndividualMRChart::new().with_rules(rules);
    for &x in &values {
        chart.add_sample(&[x]);
    }
    let i = chart
        .control_limits()
        .ok_or("at least two values are needed for control limits")?;
    let mr = chart
        .mr_limits()
        .ok_or("at least two values are needed for control limits")?;
    Ok(ImrChartDto {
        i_cl: i.cl,
        i_ucl: i.ucl,
        i_lcl: i.lcl,
        mr_cl: mr.cl,
        mr_ucl: mr.ucl,
        mr_lcl: mr.lcl,
        sigma_hat: chart.sigma_hat(),
        i_points: point_dtos(chart.points()),
        mr_points: point_dtos(chart.mr_points()),
        in_control: chart.is_in_control(),
    })
}

fn run_rules_dto(
    values: Vec<f64>,
    limits: LimitsInputDto,
    rules: crate::spc::RuleSet,
) -> Result<Vec<ChartPointDto>, String> {
    use crate::spc::{ChartPoint, ControlLimits, RunRule};

    let LimitsInputDto { ucl, cl, lcl } = limits;
    // Written so that a NaN fails it too.
    if !(lcl <= cl && cl <= ucl) {
        return Err(format!(
            "limits: need lcl <= cl <= ucl, got lcl={lcl} cl={cl} ucl={ucl}"
        ));
    }
    if let Some(i) = values.iter().position(|x| !x.is_finite()) {
        return Err(format!("values[{i}] is not a finite number"));
    }

    let mut points: Vec<ChartPoint> = values
        .iter()
        .enumerate()
        .map(|(index, &value)| ChartPoint {
            value,
            index,
            violations: Vec::new(),
        })
        .collect();
    for (index, rule) in rules.check(&points, &ControlLimits { ucl, cl, lcl }) {
        if let Some(point) = points.get_mut(index) {
            point.violations.push(rule);
        }
    }
    Ok(point_dtos(&points))
}
/// Compute a P chart from (defectives, sample_size) pairs.
///
/// # Input JSON
///
/// Array of `[defectives, sample_size]` pairs (as integers):
/// `[[3, 100], [5, 100], ...]`. A pair with `sample_size == 0`, or with more
/// defectives than `sample_size`, is rejected with its index.
///
/// # Output JSON
///
/// Object with fields: `p_bar`, `points` (array), `in_control`.
#[wasm_bindgen]
pub fn p_chart(samples: JsValue) -> Result<JsValue, JsValue> {
    let raw: Vec<[u64; 2]> = from_js(samples, "samples (expected [[defectives, size], ...])")?;
    to_js(&p_chart_dto(&raw).map_err(js_err)?)
}
/// Compute process capability indices (Cp, Cpk, Pp, Ppk, Cpm).
///
/// # Input JSON
///
/// ```text
/// { data: number[], usl?: number, lsl?: number,
///   sigma_within?: number, target?: number }
/// ```
///
/// At least one of `usl`/`lsl` is required; supplying one gives a one-sided
/// specification, which is routine for characteristics such as flatness,
/// contamination or runout.
///
/// `sigma_within` is the short-term (within-subgroup) standard deviation,
/// normally estimated from a control chart as R-bar/d2 or S-bar/c4. It is not
/// derivable from `data`: a flat measurement vector no longer carries the
/// subgroup structure. Omit it and the short-term indices are reported as
/// `null` rather than being computed from the long-term sigma.
///
/// `target` sets the Cpm target. Omit it and `cpm` is `null`: Cpm measures
/// clustering about a declared target, and one substituted on the caller's
/// behalf could not be told apart from it in the result.
///
/// # Output JSON
///
/// Object with fields: `mean`, `sigma_source`, `std_dev_within`,
/// `std_dev_overall`, `cp`, `cpk`, `cpu`, `cpl`, `pp`, `ppk`, `ppu`, `ppl`,
/// `cpm`. Indices that the specification does not support are `null`
/// (a one-sided specification has no `cp`, `pp` or `cpm`).
///
/// `sigma_source` is `"within"` when `sigma_within` was supplied and
/// `"overall"` otherwise. In the `"overall"` case `std_dev_within`, `cp`,
/// `cpk`, `cpu` and `cpl` are all `null`: the short-term indices are
/// undefined without a short-term sigma, and reporting the long-term one in
/// their place would make `cp` equal `pp` for every input. `cpm` uses neither
/// sigma -- it is the spread of `data` about `target` -- so it is reported in
/// both cases.
#[wasm_bindgen]
pub fn process_capability(input: JsValue) -> Result<JsValue, JsValue> {
    let input: CapabilityInputDto = from_js(input, "input")?;
    to_js(&capability_dto(input).map_err(js_err)?)
}

/// Pure core of [`process_capability`]: the specification, sigma-source and
/// null-handling contract, with no `JsValue` in sight.
///
/// The binding above is a two-line adapter over this. The split is what makes
/// the contract testable at all on a host without a WebAssembly runner --
/// `JsValue` cannot be constructed off `wasm32`, so a monolithic binding is
/// only reachable from a browser or Node. Everything this function decides
/// (which index family is computed, which fields come back `null`, which
/// specification shapes are legal) is the part a consumer actually observes.
fn capability_dto(input: CapabilityInputDto) -> Result<CapabilityDto, String> {
    use crate::capability::ProcessCapability;

    let mut spec = ProcessCapability::new(input.usl, input.lsl)
        .map_err(|e| format!("invalid specification limits: {e}"))?;
    if let Some(target) = input.target {
        if !target.is_finite() {
            return Err("target must be finite".to_string());
        }
        spec = spec.with_target(target);
    }

    let dto = match input.sigma_within {
        Some(sigma_within) => {
            if !sigma_within.is_finite() || sigma_within <= 0.0 {
                return Err("sigma_within must be a positive, finite number \
                     (R-bar/d2 or S-bar/c4 from the control chart)"
                    .to_string());
            }
            let indices = spec
                .compute(&input.data, sigma_within)
                .ok_or("insufficient or invalid data (need >= 2 finite values)")?;
            CapabilityDto {
                mean: indices.mean,
                sigma_source: "within",
                std_dev_within: Some(indices.std_dev_within),
                std_dev_overall: indices.std_dev_overall,
                cp: indices.cp,
                cpk: indices.cpk,
                cpu: indices.cpu,
                cpl: indices.cpl,
                pp: indices.pp,
                ppk: indices.ppk,
                ppu: indices.ppu,
                ppl: indices.ppl,
                cpm: indices.cpm,
            }
        }
        None => {
            // No short-term sigma: report the long-term indices only. The
            // crate computes both from the same sigma in this mode, so
            // carrying the short-term names through would publish Pp under the
            // name Cp for every input.
            let indices = spec
                .compute_overall(&input.data)
                .ok_or("insufficient or invalid data (need >= 2 finite values)")?;
            CapabilityDto {
                mean: indices.mean,
                sigma_source: "overall",
                std_dev_within: None,
                std_dev_overall: indices.std_dev_overall,
                cp: None,
                cpk: None,
                cpu: None,
                cpl: None,
                pp: indices.pp,
                ppk: indices.ppk,
                ppu: indices.ppu,
                ppl: indices.ppl,
                // Not a short-term index: Cpm is the spread about the target,
                // the same whichever sigma the caller could supply.
                cpm: indices.cpm,
            }
        }
    };
    Ok(dto)
}

/// Anderson-Darling normality test (Stephens 1974).
///
/// H₀: data is normally distributed.
///
/// # Parameters
///
/// - `data`: slice of observations (need >= 3)
///
/// # Output JSON
///
/// Object with fields: `statistic` (A²), `statistic_modified` (A²*), `p_value`.
#[wasm_bindgen]
pub fn anderson_darling_normality(data: &[f64]) -> Result<JsValue, JsValue> {
    let result = crate::testing::anderson_darling_normality(data).ok_or_else(|| {
        js_err("insufficient or invalid data (need >= 3 finite non-constant values)")
    })?;

    let dto = AdNormalityDto {
        statistic: result.statistic,
        statistic_modified: result.statistic_modified,
        p_value: result.p_value,
    };
    to_js(&dto)
}

/// Compute the Laney P' chart from (defectives, sample_size) pairs.
///
/// Adjusts control limits for overdispersion via a φ correction factor.
///
/// # Input JSON
///
/// Array of `[defectives, sample_size]` pairs:
/// `[[3, 100], [5, 100], ...]` (need >= 3 subgroups). A pair with
/// `sample_size == 0`, or with more defectives than `sample_size`, is rejected
/// with its index.
///
/// # Output JSON
///
/// Object with fields: `p_bar`, `phi`, `points` (array).
#[wasm_bindgen]
pub fn laney_p_chart(samples: JsValue) -> Result<JsValue, JsValue> {
    let raw: Vec<[u64; 2]> = from_js(samples, "samples (expected [[defectives, size], ...])")?;
    to_js(&laney_p_dto(&raw).map_err(js_err)?)
}

/// Compute an NP chart: the count of defectives in samples of one size.
///
/// # Input
///
/// - `defectives`: `[d1, d2, ...]`, each at most `sample_size`
/// - `sample_size`: the constant sample size, above 0
///
/// # Output JSON
///
/// Object with fields: `cl`, `ucl`, `lcl`, `points` (array), `in_control`.
#[wasm_bindgen]
pub fn np_chart(defectives: JsValue, sample_size: JsValue) -> Result<JsValue, JsValue> {
    let defectives: Vec<u64> = from_js(defectives, "defectives")?;
    let sample_size: u64 = from_js(sample_size, "sample_size")?;
    to_js(&np_chart_dto(&defectives, sample_size).map_err(js_err)?)
}

/// Compute a C chart: the count of defects per inspection unit of one size.
///
/// # Input JSON
///
/// `[c1, c2, ...]` (need >= 1).
///
/// # Output JSON
///
/// Object with fields: `cl`, `ucl`, `lcl`, `points` (array), `in_control`.
#[wasm_bindgen]
pub fn c_chart(defects: JsValue) -> Result<JsValue, JsValue> {
    let defects: Vec<u64> = from_js(defects, "defects")?;
    to_js(&c_chart_dto(&defects).map_err(js_err)?)
}

/// Compute a U chart: defects per unit when the quantity inspected varies.
///
/// # Input JSON
///
/// Array of `[defects, units]` pairs. `units` may be fractional (an area, a
/// length) and must be positive; a pair whose `units` are not is rejected with
/// its index.
///
/// # Output JSON
///
/// Object with fields: `u_bar`, `points` (array), `in_control`.
#[wasm_bindgen]
pub fn u_chart(samples: JsValue) -> Result<JsValue, JsValue> {
    let raw: Vec<(u64, f64)> = from_js(samples, "samples (expected [[defects, units], ...])")?;
    to_js(&u_chart_dto(&raw).map_err(js_err)?)
}

/// Compute the Laney U' chart from `[defects, units]` pairs (need >= 3).
///
/// Adjusts the U chart's limits for overdispersion via a φ correction factor.
///
/// # Output JSON
///
/// Object with fields: `u_bar`, `phi`, `points` (array).
#[wasm_bindgen]
pub fn laney_u_chart(samples: JsValue) -> Result<JsValue, JsValue> {
    let raw: Vec<(u64, f64)> = from_js(samples, "samples (expected [[defects, units], ...])")?;
    to_js(&laney_u_dto(&raw).map_err(js_err)?)
}

// ---------------------------------------------------------------------------
// Pure cores of the attributes-chart bindings
// ---------------------------------------------------------------------------
//
// Every attributes chart drops a row it cannot use, and every later point then
// carries the index of the wrong row. The cores refuse such a row by its index
// before the chart sees it.

fn attribute_point_dtos(points: &[crate::spc::AttributeChartPoint]) -> Vec<AttributeChartPointDto> {
    points
        .iter()
        .map(|p| AttributeChartPointDto {
            index: p.index,
            value: p.value,
            ucl: p.ucl,
            cl: p.cl,
            lcl: p.lcl,
            out_of_control: p.out_of_control,
        })
        .collect()
}

fn laney_point_dtos(points: &[crate::spc::LaneyAttributePoint]) -> Vec<AttributeChartPointDto> {
    points
        .iter()
        .map(|p| AttributeChartPointDto {
            index: p.index,
            value: p.value,
            ucl: p.ucl,
            cl: p.cl,
            lcl: p.lcl,
            out_of_control: p.out_of_control,
        })
        .collect()
}

/// `[defectives, sample_size]` pairs as `(defectives, sample_size)`, refusing
/// a pair that has no proportion.
fn proportion_samples(raw: &[[u64; 2]]) -> Result<Vec<(u64, u64)>, String> {
    raw.iter()
        .enumerate()
        .map(|(i, &[d, n])| {
            if n == 0 || d > n {
                Err(format!(
                    "samples[{i}]: {d} defectives out of {n} -- each sample needs a size \
                     above 0 and at most that many defectives"
                ))
            } else {
                Ok((d, n))
            }
        })
        .collect()
}

/// Refuses a `[defects, units]` pair whose units are not a positive number.
fn check_rate_samples(raw: &[(u64, f64)]) -> Result<(), String> {
    match raw.iter().position(|&(_, u)| !(u.is_finite() && u > 0.0)) {
        Some(i) => Err(format!(
            "samples[{i}]: units must be a positive number, got {}",
            raw[i].1
        )),
        None => Ok(()),
    }
}

fn p_chart_dto(raw: &[[u64; 2]]) -> Result<PChartDto, String> {
    use crate::spc::PChart;

    let samples = proportion_samples(raw)?;
    let mut chart = PChart::new();
    for &(d, n) in &samples {
        chart.add_sample(d, n);
    }
    let p_bar = chart.p_bar().ok_or("no samples provided")?;
    Ok(PChartDto {
        p_bar,
        points: attribute_point_dtos(chart.points()),
        in_control: chart.is_in_control(),
    })
}

fn laney_p_dto(raw: &[[u64; 2]]) -> Result<LaneyPChartDto, String> {
    let samples = proportion_samples(raw)?;
    let chart = crate::spc::laney_p_chart(&samples).ok_or("at least 3 samples are needed")?;
    Ok(LaneyPChartDto {
        p_bar: chart.p_bar,
        phi: chart.phi,
        points: laney_point_dtos(&chart.points),
    })
}

fn np_chart_dto(defectives: &[u64], sample_size: u64) -> Result<FixedLimitChartDto, String> {
    use crate::spc::NPChart;

    let mut chart = NPChart::new(sample_size).map_err(|e| e.to_string())?;
    if let Some(i) = defectives.iter().position(|&d| d > sample_size) {
        return Err(format!(
            "defectives[{i}] is {}, more than sample_size {sample_size}",
            defectives[i]
        ));
    }
    for &d in defectives {
        chart.add_sample(d);
    }
    let (ucl, cl, lcl) = chart
        .control_limits()
        .ok_or("defectives must not be empty")?;
    Ok(FixedLimitChartDto {
        cl,
        ucl,
        lcl,
        points: attribute_point_dtos(chart.points()),
        in_control: chart.is_in_control(),
    })
}

fn c_chart_dto(defects: &[u64]) -> Result<FixedLimitChartDto, String> {
    use crate::spc::CChart;

    let mut chart = CChart::new();
    for &c in defects {
        chart.add_sample(c);
    }
    let (ucl, cl, lcl) = chart.control_limits().ok_or("defects must not be empty")?;
    Ok(FixedLimitChartDto {
        cl,
        ucl,
        lcl,
        points: attribute_point_dtos(chart.points()),
        in_control: chart.is_in_control(),
    })
}

fn u_chart_dto(raw: &[(u64, f64)]) -> Result<UChartDto, String> {
    use crate::spc::UChart;

    check_rate_samples(raw)?;
    let mut chart = UChart::new();
    for &(d, u) in raw {
        chart.add_sample(d, u);
    }
    let u_bar = chart.u_bar().ok_or("samples must not be empty")?;
    Ok(UChartDto {
        u_bar,
        points: attribute_point_dtos(chart.points()),
        in_control: chart.is_in_control(),
    })
}

fn laney_u_dto(raw: &[(u64, f64)]) -> Result<LaneyUChartDto, String> {
    check_rate_samples(raw)?;
    let chart = crate::spc::laney_u_chart(raw).ok_or("at least 3 samples are needed")?;
    Ok(LaneyUChartDto {
        u_bar: chart.u_bar,
        phi: chart.phi,
        points: laney_point_dtos(&chart.points),
    })
}
/// Compute the G chart for rare-event monitoring (inter-event conforming counts).
///
/// Suitable when defect rates are very low (< 1%).
///
/// # Parameters
///
/// - `gaps`: slice of inter-event conforming counts (need >= 2 finite positive values)
///
/// # Output JSON
///
/// Object with fields: `g_bar`, `points` (array with ucl/cl/lcl/out_of_control).
#[wasm_bindgen]
pub fn g_chart(gaps: &[f64]) -> Result<JsValue, JsValue> {
    let chart = crate::spc::g_chart(gaps)
        .ok_or_else(|| js_err("insufficient data (need >= 2 finite positive values)"))?;

    let points = chart
        .points
        .iter()
        .map(|p| GChartPointDto {
            index: p.index,
            value: p.value,
            ucl: p.ucl,
            cl: p.cl,
            lcl: p.lcl,
            out_of_control: p.out_of_control,
        })
        .collect();

    let dto = GChartDto {
        g_bar: chart.g_bar,
        points,
    };
    to_js(&dto)
}

/// Compute the T chart for rare-event monitoring (inter-event times).
///
/// Control limits are derived from exponential distribution percentiles.
///
/// # Parameters
///
/// - `times`: slice of inter-event times (need >= 2 finite positive values)
///
/// # Output JSON
///
/// Object with fields: `t_bar`, `points` (array with ucl/cl/lcl/out_of_control).
#[wasm_bindgen]
pub fn t_chart(times: &[f64]) -> Result<JsValue, JsValue> {
    let chart = crate::spc::t_chart(times)
        .ok_or_else(|| js_err("insufficient data (need >= 2 finite positive values)"))?;

    let points = chart
        .points
        .iter()
        .map(|p| TChartPointDto {
            index: p.index,
            value: p.value,
            ucl: p.ucl,
            cl: p.cl,
            lcl: p.lcl,
            out_of_control: p.out_of_control,
        })
        .collect();

    let dto = TChartDto {
        t_bar: chart.t_bar,
        points,
    };
    to_js(&dto)
}

// ---------------------------------------------------------------------------
// PELT changepoint detection
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PeltInputDto {
    data: Vec<f64>,
    #[serde(default = "default_cost")]
    cost: String,
    #[serde(default = "default_penalty")]
    penalty: PeltPenaltyDto,
    #[serde(default = "default_min_seg")]
    min_segment_len: usize,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum PeltPenaltyDto {
    Named(String),
    Value(f64),
}

fn default_cost() -> String {
    "l2".to_owned()
}

fn default_penalty() -> PeltPenaltyDto {
    PeltPenaltyDto::Named("bic".to_owned())
}

fn default_min_seg() -> usize {
    2
}

#[derive(Serialize)]
struct PeltResultDto {
    changepoints: Vec<usize>,
    n_segments: usize,
}

/// Detect changepoints using the PELT algorithm (Killick et al., 2012).
///
/// # Input JSON
///
/// ```json
/// {
///   "data": [1.0, 1.1, 0.9, 5.0, 5.1, 4.9],
///   "cost": "l2",
///   "penalty": "bic",
///   "min_segment_len": 2
/// }
/// ```
///
/// - `data` (required): Array of f64 values.
/// - `cost` (optional): `"l2"` (mean change, default) or `"normal"` (mean+variance).
/// - `penalty` (optional): `"bic"` (default) or a positive number.
/// - `min_segment_len` (optional): Minimum segment length (default 2, must be >= 2).
///
/// # Output JSON
///
/// ```json
/// { "changepoints": [3], "n_segments": 2 }
/// ```
#[wasm_bindgen]
pub fn detect_changepoints(input: JsValue) -> Result<JsValue, JsValue> {
    let input: PeltInputDto = from_js(input, "input")?;

    if input.data.is_empty() {
        return Err(js_err("data must not be empty"));
    }

    let cost = match input.cost.as_str() {
        "l2" => crate::detection::CostFunction::L2,
        "normal" => crate::detection::CostFunction::Normal,
        other => return Err(js_err(format!("unknown cost function: {other}"))),
    };

    let penalty = match input.penalty {
        PeltPenaltyDto::Named(ref s) if s == "bic" => crate::detection::Penalty::Bic,
        PeltPenaltyDto::Named(ref s) => return Err(js_err(format!("unknown penalty: {s}"))),
        PeltPenaltyDto::Value(v) => crate::detection::Penalty::Custom(v),
    };

    let pelt = crate::detection::Pelt::with_min_segment_len(cost, penalty, input.min_segment_len)
        .ok_or_else(|| {
        js_err("invalid parameters (penalty must be positive, min_segment_len >= 2)")
    })?;

    let result = pelt.detect(&input.data);

    let dto = PeltResultDto {
        n_segments: result.changepoints.len() + 1,
        changepoints: result.changepoints,
    };
    to_js(&dto)
}

// ---------------------------------------------------------------------------
// Multi-signal PELT
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct MultiPeltInputDto {
    signals: Vec<Vec<f64>>,
    #[serde(default = "default_cost")]
    cost: String,
    #[serde(default = "default_penalty")]
    penalty: PeltPenaltyDto,
    #[serde(default = "default_min_seg")]
    min_segment_len: usize,
}

/// Detect changepoints in multi-signal data using PELT.
///
/// # Input JSON
///
/// ```json
/// {
///   "signals": [[0.0, 0.0, 5.0, 5.0], [0.0, 0.0, 3.0, 3.0]],
///   "cost": "l2",
///   "penalty": "bic",
///   "min_segment_len": 2
/// }
/// ```
///
/// - `signals` (required): Array of signal channels (each same length).
/// - `cost`, `penalty`, `min_segment_len`: Same as `detect_changepoints`.
///
/// # Output JSON
///
/// ```json
/// { "changepoints": [2], "n_segments": 2 }
/// ```
#[wasm_bindgen]
pub fn detect_changepoints_multi(input: JsValue) -> Result<JsValue, JsValue> {
    let input: MultiPeltInputDto = from_js(input, "input")?;

    if input.signals.is_empty() {
        return Err(js_err("signals must not be empty"));
    }

    let cost = match input.cost.as_str() {
        "l2" => crate::detection::CostFunction::L2,
        "normal" => crate::detection::CostFunction::Normal,
        other => return Err(js_err(format!("unknown cost function: {other}"))),
    };

    let penalty = match input.penalty {
        PeltPenaltyDto::Named(ref s) if s == "bic" => crate::detection::Penalty::Bic,
        PeltPenaltyDto::Named(ref s) => return Err(js_err(format!("unknown penalty: {s}"))),
        PeltPenaltyDto::Value(v) => crate::detection::Penalty::Custom(v),
    };

    let pelt = crate::detection::Pelt::with_min_segment_len(cost, penalty, input.min_segment_len)
        .ok_or_else(|| js_err("invalid parameters"))?;

    let refs: Vec<&[f64]> = input.signals.iter().map(|s| s.as_slice()).collect();
    let result = pelt
        .detect_multi(&refs)
        .ok_or_else(|| js_err("all signals must have the same length"))?;

    let dto = PeltResultDto {
        n_segments: result.changepoints.len() + 1,
        changepoints: result.changepoints,
    };
    to_js(&dto)
}

// ---------------------------------------------------------------------------
// Gage R&R (MSA)
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct GageRRInputDto {
    measurements: Vec<Vec<Vec<f64>>>,
    tolerance: Option<f64>,
}

/// Input for `percentile_capability`.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PercentileCapabilityInputDto {
    data: Vec<f64>,
    lsl: Option<f64>,
    usl: Option<f64>,
}

#[derive(Serialize)]
struct GageRRResultDto {
    ev: f64,
    av: f64,
    grr: f64,
    pv: f64,
    tv: f64,
    percent_ev: f64,
    percent_av: f64,
    percent_grr: f64,
    percent_pv: f64,
    percent_tolerance: Option<f64>,
    ndc: u32,
    status: String,
}

#[derive(Serialize)]
struct GageRRAnovaResultDto {
    anova_table: Vec<AnovaRowDto>,
    variance_components: VarianceComponentsDto,
    ev: f64,
    av: f64,
    grr: f64,
    pv: f64,
    tv: f64,
    percent_grr: f64,
    percent_tolerance: Option<f64>,
    ndc: u32,
    status: String,
    interaction_significant: bool,
    interaction_pooled: bool,
}

#[derive(Serialize)]
struct AnovaRowDto {
    source: String,
    df: f64,
    ss: f64,
    ms: f64,
    f_value: Option<f64>,
    p_value: Option<f64>,
}

#[derive(Serialize)]
struct VarianceComponentsDto {
    part: f64,
    operator: f64,
    interaction: f64,
    repeatability: f64,
    reproducibility: f64,
    total: f64,
}

#[derive(Serialize)]
struct PercentileCapabilityDto {
    cp_star: Option<f64>,
    cpk_star: Option<f64>,
    cpu_star: Option<f64>,
    cpl_star: Option<f64>,
    median: f64,
    percentile_lower: f64,
    percentile_upper: f64,
}

fn grr_status_str(status: crate::msa::GrrStatus) -> &'static str {
    match status {
        crate::msa::GrrStatus::Acceptable => "Acceptable",
        crate::msa::GrrStatus::Marginal => "Marginal",
        crate::msa::GrrStatus::Unacceptable => "Unacceptable",
    }
}

/// Compute Gage R&R using the X̄-R (Average & Range) method.
///
/// # Input JSON
///
/// ```json
/// {
///   "measurements": [[[0.29, 0.41], [0.08, 0.25]], [[1.34, 1.17], [1.19, 0.94]]],
///   "tolerance": 4.0
/// }
/// ```
///
/// `measurements[part][operator][trial]`. Tolerance is optional.
///
/// # Output JSON
///
/// Object with fields: `ev`, `av`, `grr`, `pv`, `tv`, `percent_ev`, `percent_av`,
/// `percent_grr`, `percent_pv`, `percent_tolerance`, `ndc`, `status`.
#[wasm_bindgen]
pub fn gage_rr_xbar_r(input: JsValue) -> Result<JsValue, JsValue> {
    let dto: GageRRInputDto = from_js(input, "input")?;

    let input = crate::msa::GageRRInput {
        measurements: dto.measurements,
        tolerance: dto.tolerance,
    };

    let result = crate::msa::gage_rr_xbar_r(&input).map_err(js_err)?;

    let out = GageRRResultDto {
        ev: result.ev,
        av: result.av,
        grr: result.grr,
        pv: result.pv,
        tv: result.tv,
        percent_ev: result.percent_ev,
        percent_av: result.percent_av,
        percent_grr: result.percent_grr,
        percent_pv: result.percent_pv,
        percent_tolerance: result.percent_tolerance,
        ndc: result.ndc,
        status: grr_status_str(result.status).to_owned(),
    };
    to_js(&out)
}

/// Compute Gage R&R using the two-factor crossed ANOVA method.
///
/// # Input JSON
///
/// Same format as `gage_rr_xbar_r`.
///
/// # Output JSON
///
/// Object with fields: `anova_table`, `variance_components`, `ev`, `av`, `grr`,
/// `pv`, `tv`, `percent_grr`, `percent_tolerance`, `ndc`, `status`,
/// `interaction_significant`, `interaction_pooled`.
#[wasm_bindgen]
pub fn gage_rr_anova(input: JsValue) -> Result<JsValue, JsValue> {
    let dto: GageRRInputDto = from_js(input, "input")?;

    let input = crate::msa::GageRRInput {
        measurements: dto.measurements,
        tolerance: dto.tolerance,
    };

    let result = crate::msa::gage_rr_anova(&input).map_err(js_err)?;

    let anova_rows: Vec<AnovaRowDto> = result
        .anova_table
        .rows
        .iter()
        .map(|r| AnovaRowDto {
            source: r.source.clone(),
            df: r.df,
            ss: r.ss,
            ms: r.ms,
            f_value: r.f_value,
            p_value: r.p_value,
        })
        .collect();

    let vc = &result.variance_components;
    let out = GageRRAnovaResultDto {
        anova_table: anova_rows,
        variance_components: VarianceComponentsDto {
            part: vc.part,
            operator: vc.operator,
            interaction: vc.interaction,
            repeatability: vc.repeatability,
            reproducibility: vc.reproducibility,
            total: vc.total,
        },
        ev: result.ev,
        av: result.av,
        grr: result.grr,
        pv: result.pv,
        tv: result.tv,
        percent_grr: result.percent_grr,
        percent_tolerance: result.percent_tolerance,
        ndc: result.ndc,
        status: grr_status_str(result.status).to_owned(),
        interaction_significant: result.interaction_significant,
        interaction_pooled: result.interaction_pooled,
    };
    to_js(&out)
}

/// Compute percentile-based capability indices (ISO 22514-2).
///
/// # Input JSON
///
/// ```json
/// {
///   "data": [1.0, 2.0, ...],
///   "lsl": 0.0,
///   "usl": 10.0
/// }
/// ```
///
/// At least one of `lsl` or `usl` must be provided. Requires >= 20 data points.
///
/// # Output JSON
///
/// Object with fields: `cp_star`, `cpk_star`, `cpu_star`, `cpl_star`,
/// `median`, `percentile_lower`, `percentile_upper`.
#[wasm_bindgen]
pub fn percentile_capability(input: JsValue) -> Result<JsValue, JsValue> {
    let input: PercentileCapabilityInputDto = from_js(input, "input")?;

    let result = crate::capability::percentile_capability(&input.data, input.lsl, input.usl)
        .map_err(js_err)?;

    let dto = PercentileCapabilityDto {
        cp_star: result.cp_star,
        cpk_star: result.cpk_star,
        cpu_star: result.cpu_star,
        cpl_star: result.cpl_star,
        median: result.median,
        percentile_lower: result.percentile_lower,
        percentile_upper: result.percentile_upper,
    };
    to_js(&dto)
}

// ── Wire-schema strictness tests ─────────────────────────────────────

#[cfg(test)]
mod dto_strictness_tests {
    use serde_json::json;

    fn assert_rejects_unknown<T: serde::de::DeserializeOwned>(v: serde_json::Value) {
        match super::from_json::<T>(v, "input") {
            Ok(_) => panic!("unknown key must be rejected"),
            Err(e) => assert!(e.contains("unknown field"), "{e}"),
        }
    }

    #[test]
    fn capability_input_rejects_unknown_keys() {
        assert_rejects_unknown::<super::CapabilityInputDto>(
            json!({ "data": [1.0, 2.0], "usl": 3.0, "sigmaWithin": 0.5 }),
        );
    }

    #[test]
    fn pelt_input_rejects_unknown_keys() {
        assert_rejects_unknown::<super::PeltInputDto>(
            json!({ "data": [1.0, 2.0], "minSegmentLen": 3 }),
        );
    }

    #[test]
    fn multi_pelt_input_rejects_unknown_keys() {
        assert_rejects_unknown::<super::MultiPeltInputDto>(
            json!({ "signals": [[1.0, 2.0]], "penalty_value": 1.0 }),
        );
    }

    #[test]
    fn gage_rr_input_rejects_unknown_keys() {
        assert_rejects_unknown::<super::GageRRInputDto>(
            json!({ "measurements": [[[1.0]]], "tol": 0.5 }),
        );
    }

    #[test]
    fn percentile_capability_input_rejects_unknown_keys() {
        assert_rejects_unknown::<super::PercentileCapabilityInputDto>(
            json!({ "data": [1.0, 2.0], "target": 1.5 }),
        );
    }
}

// ── Binding-contract tests ───────────────────────────────────────────
//
// These pin what a JavaScript caller observes, not what the underlying
// statistics compute -- the crate's own modules already cover the latter. The
// distinction matters: the defect these exist to prevent was never in
// `ProcessCapability`, which was correct throughout. It was in the binding,
// which called the long-term entry point and then labelled the result with
// short-term field names, so `cp == pp` held for every input while every unit
// test in the crate stayed green.
//
// They run under `cargo test --features wasm` on any host. `JsValue` is absent
// from every assertion below by design; the adapter that wraps it is two lines
// and has nothing left to get wrong.

#[cfg(test)]
mod binding_contract_tests {
    use super::*;

    // --- rules option on xbar_r_chart ---

    /// Every way of not asking for a rule set must mean the set the binding
    /// applied before the option existed. A caller that never passes options is
    /// the common case, and it must not have changed.
    #[test]
    fn rules_option_absent_means_the_default_set() {
        use crate::spc::RuleSet;
        for absent in [
            None,
            Some(serde_json::Value::Null),
            Some(serde_json::json!({})),
            Some(serde_json::json!({ "rules": null })),
        ] {
            assert_eq!(rules_from_json(absent).unwrap(), RuleSet::default());
        }
    }

    #[test]
    fn rules_option_selects_exactly_what_it_names() {
        use crate::spc::{RuleSet, ViolationType};
        let set = rules_from_json(Some(serde_json::json!({
            "rules": ["BeyondLimits", "SixTrend"]
        })))
        .unwrap();
        assert_eq!(
            set,
            RuleSet::from_iter([ViolationType::BeyondLimits, ViolationType::SixTrend])
        );
        assert!(!set.contains(ViolationType::NineOneSide));
    }

    #[test]
    fn rules_option_empty_array_applies_no_test() {
        use crate::spc::RuleSet;
        assert_eq!(
            rules_from_json(Some(serde_json::json!({ "rules": [] }))).unwrap(),
            RuleSet::none()
        );
    }

    #[test]
    fn rules_option_rejects_a_name_that_is_not_a_rule() {
        let err = rules_from_json(Some(serde_json::json!({ "rules": ["Rule1"] }))).unwrap_err();
        assert!(err.contains("Rule1"), "{err}");
        let err =
            rules_from_json(Some(serde_json::json!({ "rules": "BeyondLimits" }))).unwrap_err();
        assert!(err.contains("array"), "{err}");
    }

    use super::{capability_dto, from_json, CapabilityInputDto};
    use serde_json::json;

    /// The reporter's fixture (docket #219): six subgroups of five, with real
    /// between-subgroup drift, so the short-term and long-term families are
    /// genuinely different numbers.
    const SUBGROUPS: [[f64; 5]; 6] = [
        [9.9, 10.1, 10.0, 9.8, 10.2],
        [10.3, 9.7, 10.0, 10.1, 9.9],
        [9.8, 10.2, 10.1, 9.9, 10.0],
        [10.5, 9.5, 10.0, 10.2, 9.8],
        [9.6, 10.4, 10.0, 9.9, 10.1],
        [10.1, 9.9, 10.0, 10.3, 9.7],
    ];

    fn flat() -> Vec<f64> {
        SUBGROUPS.iter().flatten().copied().collect()
    }

    fn dto(v: serde_json::Value) -> Result<super::CapabilityDto, String> {
        capability_dto(from_json::<CapabilityInputDto>(v, "input")?)
    }

    /// `sigma_hat` from the chart the measurements actually came from -- the
    /// quantity the flat vector cannot carry.
    fn sigma_within_from_chart() -> f64 {
        use crate::spc::{ControlChart, XBarRChart};
        let mut chart = XBarRChart::new(5).expect("5 is within range");
        for g in SUBGROUPS {
            chart.add_sample(&g);
        }
        chart.sigma_hat().expect("limits available for 6 subgroups")
    }

    #[test]
    fn omitting_sigma_within_yields_nulls_not_long_term_numbers_under_short_term_names() {
        let d = dto(json!({ "data": flat(), "usl": 11.0, "lsl": 9.0 })).expect("valid input");

        assert_eq!(d.sigma_source, "overall");
        // The whole short-term family is absent rather than borrowed. A reader
        // can ignore a label; it cannot ignore a null.
        assert!(d.std_dev_within.is_none());
        assert!(d.cp.is_none());
        assert!(d.cpk.is_none());
        assert!(d.cpu.is_none());
        assert!(d.cpl.is_none());
        // The long-term family is what this input can support, so it is present.
        assert!(d.pp.is_some());
        assert!(d.ppk.is_some());
    }

    #[test]
    fn supplying_sigma_within_separates_the_two_index_families() {
        let sigma_within = sigma_within_from_chart();
        let d = dto(json!({
            "data": flat(), "usl": 11.0, "lsl": 9.0, "sigma_within": sigma_within,
        }))
        .expect("valid input");

        assert_eq!(d.sigma_source, "within");
        assert_eq!(d.std_dev_within, Some(sigma_within));

        let (cp, pp) = (d.cp.expect("cp"), d.pp.expect("pp"));
        let (cpk, ppk) = (d.cpk.expect("cpk"), d.ppk.expect("ppk"));

        // This is the shipped regression, stated as an assertion: the binding
        // used to make these pairs equal for every input by construction.
        assert!(
            (cp - pp).abs() > 1e-9,
            "cp and pp must come from different sigmas: cp={cp} pp={pp}"
        );
        assert!(
            (cpk - ppk).abs() > 1e-9,
            "cpk and ppk must come from different sigmas: cpk={cpk} ppk={ppk}"
        );
        // Stronger than "they differ": each family must be tied to *its own*
        // sigma. Cp = (USL-LSL)/(6*sigma), so the ratio of the two indices is
        // the inverse ratio of the two sigmas -- an identity that only holds
        // if the binding fed the short-term sigma to the short-term family and
        // the long-term sigma to the long-term one. The old binding satisfied
        // "they differ" trivially by never differing; it could not satisfy
        // this.
        let sigma_overall = d.std_dev_overall;
        assert!(
            (cp / pp - sigma_overall / sigma_within).abs() < 1e-12,
            "cp/pp must equal sigma_overall/sigma_within: \
             cp={cp} pp={pp} sigma_within={sigma_within} sigma_overall={sigma_overall}"
        );

        // Which of the two is larger is a property of the data, not of the
        // binding: this fixture's subgroup means sit close together, so the
        // range-based within estimate exceeds the pooled overall one.
        assert!(sigma_within > sigma_overall);
    }

    #[test]
    fn one_sided_specification_drops_only_the_indices_it_cannot_support() {
        let upper = dto(json!({ "data": flat(), "usl": 11.0 })).expect("usl-only is legal");
        assert!(upper.ppu.is_some(), "an upper limit supports Ppu");
        assert!(upper.ppl.is_none(), "no lower limit, no Ppl");
        assert!(upper.pp.is_none(), "Pp needs both limits");
        assert!(upper.cpm.is_none(), "Cpm needs both limits");

        let lower = dto(json!({ "data": flat(), "lsl": 9.0 })).expect("lsl-only is legal");
        assert!(lower.ppl.is_some());
        assert!(lower.ppu.is_none());
        assert!(lower.pp.is_none());
    }

    #[test]
    fn a_specification_with_no_limit_at_all_is_rejected() {
        let e = dto(json!({ "data": flat() })).expect_err("no limit is not a specification");
        assert!(e.contains("specification"), "{e}");
    }

    #[test]
    fn cpm_needs_a_declared_target_and_no_sigma() {
        // This test used to assert that omitting `target` reproduced the
        // midpoint exactly -- it pinned the substitution as the contract. Cpm
        // is defined only against a declared target, so there is nothing to
        // reproduce.
        let within = json!({ "data": flat(), "usl": 11.0, "lsl": 9.0, "sigma_within": 0.2 });
        let overall = json!({ "data": flat(), "usl": 11.0, "lsl": 9.0 });
        assert!(dto(within.clone()).expect("valid").cpm.is_none());
        assert!(dto(overall.clone()).expect("valid").cpm.is_none());

        // With a target, Cpm is the spread about it: no sigma enters, so the
        // two sigma sources give one value.
        let (mut within, mut overall) = (within, overall);
        within["target"] = json!(10.4);
        overall["target"] = json!(10.4);
        let a = dto(within).expect("valid").cpm.expect("cpm with a target");
        let b = dto(overall).expect("valid").cpm.expect("cpm without sigma_within");
        assert_eq!(a, b);
    }

    #[test]
    fn sigma_within_must_be_a_usable_standard_deviation() {
        // Representable on the wire: JSON carries these, so a JS caller can
        // actually send them.
        for bad in [0.0, -1.0] {
            let v = json!({ "data": flat(), "usl": 11.0, "lsl": 9.0, "sigma_within": bad });
            let e = dto(v).expect_err("a non-positive sigma is not a standard deviation");
            assert!(e.contains("sigma_within"), "{bad} -> {e}");
        }

        // NaN and infinity are *not* representable in JSON, so the wire cannot
        // deliver them however hard a caller tries -- `json!(f64::NAN)` is
        // `null`, which reads back as "omitted". The guard in the core is
        // therefore about direct Rust callers, and this is the only way to
        // reach it. Asserting it here documents which layer stops what.
        for bad in [f64::NAN, f64::INFINITY] {
            let input = CapabilityInputDto {
                data: flat(),
                usl: Some(11.0),
                lsl: Some(9.0),
                sigma_within: Some(bad),
                target: None,
            };
            let e = capability_dto(input).expect_err("non-finite sigma must be refused");
            assert!(e.contains("sigma_within"), "{bad} -> {e}");
        }
    }

    #[test]
    fn too_little_data_is_an_error_not_a_nan() {
        let e = dto(json!({ "data": [1.0], "usl": 11.0, "lsl": 9.0 }))
            .expect_err("one point has no dispersion");
        assert!(e.contains("insufficient"), "{e}");
    }

    // --- #220: X-bar-S, I-MR and the standalone rule engine ---

    /// Twenty subgroups whose mean steps up halfway, so the run tests have
    /// something to find. Every subgroup holds the same five offsets, so its
    /// mean is exactly 10.0 or 10.6.
    fn shifted_subgroups() -> Vec<Vec<f64>> {
        (0..20)
            .map(|g| {
                let shift = if g >= 10 { 0.6 } else { 0.0 };
                (0..5)
                    .map(|i| 10.0 + shift + 0.1 * (((g * 3 + i * 7) % 5) as f64 - 2.0))
                    .collect()
            })
            .collect()
    }

    #[test]
    fn xbar_s_chart_reports_the_crate_chart() {
        use crate::spc::{ControlChart, RuleSet, XBarSChart};
        let subgroups = shifted_subgroups();
        let d = xbar_s_dto(subgroups.clone(), RuleSet::default()).expect("valid input");

        let mut chart = XBarSChart::new(5).expect("5 is in range");
        for g in &subgroups {
            chart.add_sample(g);
        }
        let x = chart.control_limits().expect("limits");
        let s = chart.s_limits().expect("limits");
        assert_eq!((d.xbar_ucl, d.xbar_cl, d.xbar_lcl), (x.ucl, x.cl, x.lcl));
        assert_eq!((d.s_ucl, d.s_cl, d.s_lcl), (s.ucl, s.cl, s.lcl));
        assert_eq!(d.sigma_hat, chart.sigma_hat());
        assert_eq!(d.xbar_points.len(), 20);
        assert_eq!(d.s_points.len(), 20);
        assert_eq!(d.in_control, chart.is_in_control());
    }

    #[test]
    fn a_ragged_subgroup_is_rejected_by_its_row_not_skipped() {
        // The chart would skip it and number every later point one short.
        use crate::spc::RuleSet;
        let mut subgroups = shifted_subgroups();
        subgroups[7].pop();
        for result in [
            xbar_s_dto(subgroups.clone(), RuleSet::default()).map(|_| ()),
            xbar_r_dto(subgroups, RuleSet::default()).map(|_| ()),
        ] {
            let e = result.expect_err("ragged subgroup");
            assert!(e.contains("subgroup 7"), "{e}");
        }
    }

    #[test]
    fn imr_chart_reports_the_crate_chart() {
        use crate::spc::{ControlChart, IndividualMRChart, RuleSet};
        let values: Vec<f64> = shifted_subgroups().iter().map(|g| g[0]).collect();
        let d = imr_dto(values.clone(), RuleSet::default()).expect("valid input");

        let mut chart = IndividualMRChart::new();
        for &x in &values {
            chart.add_sample(&[x]);
        }
        let i = chart.control_limits().expect("limits");
        let mr = chart.mr_limits().expect("limits");
        assert_eq!((d.i_ucl, d.i_cl, d.i_lcl), (i.ucl, i.cl, i.lcl));
        assert_eq!((d.mr_ucl, d.mr_cl, d.mr_lcl), (mr.ucl, mr.cl, mr.lcl));
        assert_eq!(d.sigma_hat, chart.sigma_hat());
        assert_eq!(d.i_points.len(), values.len());
        // MR_0 is undefined: the moving-range series starts at the second value.
        assert_eq!(d.mr_points.len(), values.len() - 1);
        assert_eq!(d.mr_points[0].index, 1);
    }

    #[test]
    fn imr_chart_needs_two_values() {
        use crate::spc::RuleSet;
        let e = imr_dto(vec![1.0], RuleSet::default()).expect_err("one value has no range");
        assert!(e.contains("two"), "{e}");
    }

    /// The standalone engine must be the engine the charts use: fed a chart's
    /// own points and limits, it has to find exactly what the chart found.
    #[test]
    fn run_rules_finds_what_the_chart_found() {
        use crate::spc::RuleSet;
        let chart = xbar_r_dto(shifted_subgroups(), RuleSet::default()).expect("valid");
        let flagged = chart
            .xbar_points
            .iter()
            .filter(|p| !p.violations.is_empty())
            .count();
        assert!(
            flagged > 0,
            "the fixture must give the rules something to find"
        );

        let values = chart.xbar_points.iter().map(|p| p.value).collect();
        let limits = LimitsInputDto {
            ucl: chart.xbar_ucl,
            cl: chart.xbar_cl,
            lcl: chart.xbar_lcl,
        };
        let standalone = run_rules_dto(values, limits, RuleSet::default()).expect("valid");

        assert_eq!(standalone.len(), chart.xbar_points.len());
        for (a, b) in standalone.iter().zip(&chart.xbar_points) {
            assert_eq!(a.index, b.index);
            assert_eq!(a.violations, b.violations, "point {}", a.index);
        }
    }

    #[test]
    fn run_rules_applies_only_the_rules_it_is_given() {
        use crate::spc::{RuleSet, ViolationType};
        // One point far outside, the rest on the centre line.
        let mut values = vec![0.0; 10];
        values[4] = 5.0;
        let limits = || LimitsInputDto {
            ucl: 3.0,
            cl: 0.0,
            lcl: -3.0,
        };

        let all = run_rules_dto(values.clone(), limits(), RuleSet::default()).expect("valid");
        assert!(all[4].violations.contains(&"BeyondLimits".to_owned()));

        let none = run_rules_dto(values.clone(), limits(), RuleSet::none()).expect("valid");
        assert!(none.iter().all(|p| p.violations.is_empty()));

        let only_trend = RuleSet::none().with(ViolationType::SixTrend);
        let trend = run_rules_dto(values, limits(), only_trend).expect("valid");
        assert!(trend[4].violations.is_empty());
    }

    #[test]
    fn run_rules_reports_every_value_in_order() {
        use crate::spc::RuleSet;
        let limits = LimitsInputDto {
            ucl: 5.0,
            cl: 2.0,
            lcl: -1.0,
        };
        let out = run_rules_dto(vec![1.0, 2.0, 3.0], limits, RuleSet::default()).expect("valid");
        let indices: Vec<usize> = out.iter().map(|p| p.index).collect();
        assert_eq!(indices, vec![0, 1, 2]);
        assert_eq!(out[2].value, 3.0);
    }

    #[test]
    fn run_rules_rejects_limits_out_of_order() {
        use crate::spc::RuleSet;
        let limits = LimitsInputDto {
            ucl: 1.0,
            cl: 2.0,
            lcl: 0.0,
        };
        let e = run_rules_dto(vec![1.0], limits, RuleSet::default()).expect_err("cl above ucl");
        assert!(e.contains("lcl <= cl <= ucl"), "{e}");
    }

    #[test]
    fn limits_input_rejects_unknown_keys() {
        let e = from_json::<LimitsInputDto>(
            json!({ "ucl": 1.0, "cl": 0.0, "lcl": -1.0, "usl": 2.0 }),
            "limits",
        )
        .err()
        .expect("unknown key");
        assert!(e.contains("unknown field"), "{e}");
    }

    // --- #220: attributes charts ---

    #[test]
    fn np_chart_reports_the_crate_chart() {
        use crate::spc::NPChart;
        let defectives = [5, 8, 3, 6, 4, 7, 2, 9, 5, 6];
        let d = np_chart_dto(&defectives, 100).expect("valid");
        let mut chart = NPChart::new(100).expect("valid size");
        for &x in &defectives {
            chart.add_sample(x);
        }
        let (ucl, cl, lcl) = chart.control_limits().expect("limits");
        assert_eq!((d.ucl, d.cl, d.lcl), (ucl, cl, lcl));
        assert_eq!(d.points.len(), defectives.len());
        assert_eq!(d.in_control, chart.is_in_control());
    }

    #[test]
    fn np_chart_refuses_rows_it_cannot_chart() {
        let e = np_chart_dto(&[5, 101, 3], 100).expect_err("more defectives than inspected");
        assert!(e.contains("defectives[1]"), "{e}");
        // A zero sample size is a value, not a panic that would trap the module.
        let e = np_chart_dto(&[0, 0], 0).expect_err("nothing inspected");
        assert!(e.contains("sample size"), "{e}");
    }

    #[test]
    fn c_chart_reports_the_crate_chart() {
        use crate::spc::CChart;
        let defects = [3, 5, 2, 4, 6, 3, 1, 4];
        let d = c_chart_dto(&defects).expect("valid");
        let mut chart = CChart::new();
        for &x in &defects {
            chart.add_sample(x);
        }
        let (ucl, cl, lcl) = chart.control_limits().expect("limits");
        assert_eq!((d.ucl, d.cl, d.lcl), (ucl, cl, lcl));
        assert!(c_chart_dto(&[]).is_err());
    }

    #[test]
    fn u_chart_reports_the_crate_chart_and_refuses_empty_units() {
        use crate::spc::UChart;
        let samples = [(3, 1.0), (5, 1.5), (2, 0.8), (4, 1.2)];
        let d = u_chart_dto(&samples).expect("valid");
        let mut chart = UChart::new();
        for &(x, u) in &samples {
            chart.add_sample(x, u);
        }
        assert_eq!(Some(d.u_bar), chart.u_bar());
        assert_eq!(d.points.len(), samples.len());

        let e = u_chart_dto(&[(3, 1.0), (5, 0.0), (2, 0.8)]).expect_err("zero units");
        assert!(e.contains("samples[1]"), "{e}");
    }

    #[test]
    fn laney_u_chart_reports_the_crate_chart() {
        let samples = [(3, 1.0), (5, 1.5), (2, 0.8), (4, 1.2), (6, 1.1)];
        let d = laney_u_dto(&samples).expect("valid");
        let expected = crate::spc::laney_u_chart(&samples).expect("valid");
        assert_eq!((d.u_bar, d.phi), (expected.u_bar, expected.phi));
        assert!(laney_u_dto(&samples[..2]).is_err(), "fewer than 3 samples");
        let e = laney_u_dto(&[(3, 1.0), (5, -1.0), (2, 0.8)]).expect_err("negative units");
        assert!(e.contains("samples[1]"), "{e}");
    }

    /// `p_chart` used to hand every pair to the chart, which dropped the ones
    /// it could not use -- shifting the index of every later point.
    #[test]
    fn p_chart_refuses_a_sample_with_no_proportion() {
        let e = p_chart_dto(&[[3, 100], [12, 10], [4, 100]]).expect_err("12 of 10");
        assert!(e.contains("samples[1]"), "{e}");
        let e = p_chart_dto(&[[3, 100], [0, 0]]).expect_err("0 of 0");
        assert!(e.contains("samples[1]"), "{e}");
        let ok = p_chart_dto(&[[3, 100], [5, 120], [2, 80]]).expect("valid");
        assert_eq!(ok.points.len(), 3);
    }

    #[test]
    fn laney_p_chart_refuses_a_sample_with_no_proportion() {
        let e = laney_p_dto(&[[3, 100], [0, 0], [4, 100], [2, 100]]).expect_err("0 of 0");
        assert!(e.contains("samples[1]"), "{e}");
        assert!(
            laney_p_dto(&[[3, 100], [5, 120]]).is_err(),
            "fewer than 3 samples"
        );
    }
}
