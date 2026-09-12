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

// The shapes below are the wire contract, shared with the C FFI so the two
// transports cannot drift apart again. See `crate::wire`.
use crate::wire::{
    add_rows, attribute_point_dtos, capability_dto, default_cost, default_min_seg, default_penalty,
    gage_rr_anova_dto, gage_rr_xbar_r_dto, laney_p_dto, laney_point_dtos, p_chart_dto, pelt_dto,
    percentile_capability_dto, point_dtos, rules_from_json, subgroup_size, xbar_r_dto,
    AttributeChartPointDto, CapabilityInputDto, ChartPointDto, GageRRInputDto, PeltInputDto,
    PeltPenaltyDto, PeltResultDto, PercentileCapabilityInputDto,
};

// ---------------------------------------------------------------------------
// Serializable DTO types
// ---------------------------------------------------------------------------

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

#[derive(Serialize)]
struct AdNormalityDto {
    statistic: f64,
    statistic_modified: f64,
    p_value: f64,
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

// ---------------------------------------------------------------------------
// WASM exports
// ---------------------------------------------------------------------------

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

fn xbar_s_dto(
    subgroups: Vec<Vec<f64>>,
    rules: crate::spc::RuleSet,
) -> Result<XbarSChartDto, String> {
    use crate::spc::{ControlChart, XBarSChart};

    let n = subgroup_size(&subgroups)?;
    let mut chart = XBarSChart::new(n)
        .map_err(|e| e.to_string())?
        .with_rules(rules);
    add_rows(&subgroups, "subgroups", |g| chart.add_sample(g))?;
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

    let mut chart = IndividualMRChart::new().with_rules(rules);
    add_rows(&values, "values", |x| {
        chart.add_sample(std::slice::from_ref(x))
    })?;
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

fn np_chart_dto(defectives: &[u64], sample_size: u64) -> Result<FixedLimitChartDto, String> {
    use crate::spc::NPChart;

    let mut chart = NPChart::new(sample_size).map_err(|e| e.to_string())?;
    add_rows(defectives, "defectives", |&d| chart.add_sample(d))?;
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

    let mut chart = UChart::new();
    add_rows(raw, "samples", |&(d, u)| chart.add_sample(d, u))?;
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
    to_js(&pelt_dto(input).map_err(js_err)?)
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
    let input: GageRRInputDto = from_js(input, "input")?;
    to_js(&gage_rr_xbar_r_dto(input).map_err(js_err)?)
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
    let input: GageRRInputDto = from_js(input, "input")?;
    to_js(&gage_rr_anova_dto(input).map_err(js_err)?)
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
    to_js(&percentile_capability_dto(input).map_err(js_err)?)
}

// ---------------------------------------------------------------------------
// CUSUM / EWMA -- online (sequential) shift detection
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CusumInputDto {
    data: Vec<f64>,
    target: f64,
    sigma: f64,
    #[serde(default = "default_cusum_k")]
    k: f64,
    #[serde(default = "default_cusum_h")]
    h: f64,
}

fn default_cusum_k() -> f64 {
    0.5
}

fn default_cusum_h() -> f64 {
    5.0
}

#[derive(Serialize, Debug)]
struct CusumPointDto {
    index: usize,
    s_upper: f64,
    s_lower: f64,
    signal: bool,
}

#[derive(Serialize, Debug)]
struct CusumDto {
    /// Decision interval actually used, echoed so a caller can draw the
    /// boundary without restating its own input. Unlike EWMA's widening
    /// limits this one is constant, so it belongs to the chart, not the point.
    h: f64,
    points: Vec<CusumPointDto>,
    signal_indices: Vec<usize>,
    in_control: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct EwmaInputDto {
    data: Vec<f64>,
    target: f64,
    sigma: f64,
    #[serde(default = "default_ewma_lambda")]
    lambda: f64,
    #[serde(default = "default_ewma_l_factor")]
    l_factor: f64,
}

fn default_ewma_lambda() -> f64 {
    0.2
}

fn default_ewma_l_factor() -> f64 {
    3.0
}

#[derive(Serialize, Debug)]
struct EwmaPointDto {
    index: usize,
    ewma: f64,
    ucl: f64,
    lcl: f64,
    signal: bool,
}

#[derive(Serialize, Debug)]
struct EwmaDto {
    points: Vec<EwmaPointDto>,
    signal_indices: Vec<usize>,
    in_control: bool,
}

/// CUSUM chart (Page, 1954) -- detects small persistent shifts in the mean.
///
/// # Input JSON
///
/// ```json
/// { "data": [10.1, 9.8, 12.0], "target": 10.0, "sigma": 1.0, "k": 0.5, "h": 5.0 }
/// ```
///
/// - `data` (required): observations, in order. Must be non-empty.
/// - `target` (required): process target mean (mu_0).
/// - `sigma` (required): known process standard deviation. Must be positive.
/// - `k` (optional): reference value / allowance, default `0.5` (optimal for a
///   1-sigma shift). Must be non-negative.
/// - `h` (optional): decision interval, default `5.0` (ARL_0 ~ 465). Must be positive.
///
/// # Output JSON
///
/// ```json
/// {
///   "h": 5.0,
///   "points": [{ "index": 0, "s_upper": 0.0, "s_lower": 0.0, "signal": false }],
///   "signal_indices": [],
///   "in_control": true
/// }
/// ```
///
/// Both cumulative sums are on the **standardized** scale (`z = (x - target) / sigma`)
/// and start at zero, so they compare against `h` directly.
#[wasm_bindgen]
pub fn cusum(input: JsValue) -> Result<JsValue, JsValue> {
    let input: CusumInputDto = from_js(input, "input")?;
    let dto = cusum_dto(input).map_err(js_err)?;
    to_js(&dto)
}

/// The half of [`cusum`] below the `JsValue` boundary, so the contract is
/// testable off `wasm32`.
fn cusum_dto(input: CusumInputDto) -> Result<CusumDto, String> {
    if input.data.is_empty() {
        return Err("data must not be empty".to_owned());
    }

    let chart = crate::detection::Cusum::with_params(input.target, input.sigma, input.k, input.h)
        .ok_or_else(|| {
        format!(
            "invalid parameters (target must be finite, sigma > 0, k >= 0, h > 0); \
                 got target={}, sigma={}, k={}, h={}",
            input.target, input.sigma, input.k, input.h
        )
    })?;

    let results = chart.analyze(&input.data);
    let signal_indices: Vec<usize> = results
        .iter()
        .filter(|r| r.signal)
        .map(|r| r.index)
        .collect();

    Ok(CusumDto {
        h: input.h,
        in_control: signal_indices.is_empty(),
        signal_indices,
        points: results
            .into_iter()
            .map(|r| CusumPointDto {
                index: r.index,
                s_upper: r.s_upper,
                s_lower: r.s_lower,
                signal: r.signal,
            })
            .collect(),
    })
}

/// EWMA chart (Roberts, 1959) -- exponentially weighted moving average of the mean.
///
/// # Input JSON
///
/// ```json
/// { "data": [10.1, 9.8, 12.0], "target": 10.0, "sigma": 1.0, "lambda": 0.2, "l_factor": 3.0 }
/// ```
///
/// - `data` (required): observations, in order. Must be non-empty.
/// - `target` (required): process target mean (mu_0).
/// - `sigma` (required): known process standard deviation. Must be positive.
/// - `lambda` (optional): smoothing constant, default `0.2`. Must be in `(0, 1]`.
/// - `l_factor` (optional): control-limit width factor, default `3.0`. Must be positive.
///
/// # Output JSON
///
/// ```json
/// {
///   "points": [{ "index": 0, "ewma": 10.02, "ucl": 10.6, "lcl": 9.4, "signal": false }],
///   "signal_indices": [],
///   "in_control": true
/// }
/// ```
///
/// The limits **widen with the observation index** (they are exact, not asymptotic),
/// so they are returned per point rather than once for the chart.
#[wasm_bindgen]
pub fn ewma(input: JsValue) -> Result<JsValue, JsValue> {
    let input: EwmaInputDto = from_js(input, "input")?;
    let dto = ewma_dto(input).map_err(js_err)?;
    to_js(&dto)
}

/// The half of [`ewma`] below the `JsValue` boundary, so the contract is
/// testable off `wasm32`.
fn ewma_dto(input: EwmaInputDto) -> Result<EwmaDto, String> {
    if input.data.is_empty() {
        return Err("data must not be empty".to_owned());
    }

    let chart = crate::detection::Ewma::with_params(
        input.target,
        input.sigma,
        input.lambda,
        input.l_factor,
    )
    .ok_or_else(|| {
        format!(
            "invalid parameters (target must be finite, sigma > 0, 0 < lambda <= 1, \
             l_factor > 0); got target={}, sigma={}, lambda={}, l_factor={}",
            input.target, input.sigma, input.lambda, input.l_factor
        )
    })?;

    let results = chart.analyze(&input.data);
    let signal_indices: Vec<usize> = results
        .iter()
        .filter(|r| r.signal)
        .map(|r| r.index)
        .collect();

    Ok(EwmaDto {
        in_control: signal_indices.is_empty(),
        signal_indices,
        points: results
            .into_iter()
            .map(|r| EwmaPointDto {
                index: r.index,
                ewma: r.ewma,
                ucl: r.ucl,
                lcl: r.lcl,
                signal: r.signal,
            })
            .collect(),
    })
}

// ---------------------------------------------------------------------------
// Non-normal capability (Box-Cox) and sigma level <-> PPM
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct BoxcoxCapabilityInputDto {
    data: Vec<f64>,
    #[serde(default)]
    usl: Option<f64>,
    #[serde(default)]
    lsl: Option<f64>,
}

#[derive(Serialize, Debug)]
struct BoxcoxCapabilityDto {
    /// Estimated optimal Box-Cox parameter. `0` is a log transform, `1` the
    /// identity, `0.5` approximately a square root.
    lambda: f64,
    /// Every index below is on the **transformed** scale, which is where the
    /// normal-theory formulas are valid -- they are not comparable to indices
    /// computed on the raw non-normal data.
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

/// Process capability for non-normal data, via a Box-Cox transformation.
///
/// # Input JSON
///
/// ```json
/// { "data": [1.2, 3.4, 9.1, 22.0], "usl": 100.0, "lsl": 1.0 }
/// ```
///
/// - `data` (required): at least 4 observations, **all strictly positive**.
/// - `usl` / `lsl` (optional): at least one is required; each must be positive.
///
/// # Output JSON
///
/// ```json
/// { "lambda": 0.13, "cp": null, "cpk": null, "cpu": null, "cpl": null,
///   "pp": 1.42, "ppk": 1.19, "ppu": 1.19, "ppl": 1.65, "cpm": null }
/// ```
///
/// The optimal lambda is estimated by maximum likelihood over `[-2, 2]`, the
/// specification limits are transformed with that same lambda, and the indices
/// are computed on the transformed scale. `cp`/`cpk`/`cpu`/`cpl` need a
/// short-term sigma, which a flat vector cannot carry, so they come back
/// `null` here -- exactly as they do from `process_capability` without
/// `sigma_within`.
#[wasm_bindgen]
pub fn boxcox_capability(input: JsValue) -> Result<JsValue, JsValue> {
    let input: BoxcoxCapabilityInputDto = from_js(input, "input")?;
    let dto = boxcox_capability_dto(input).map_err(js_err)?;
    to_js(&dto)
}

/// The half of [`boxcox_capability`] below the `JsValue` boundary, so the
/// contract is testable off `wasm32`.
fn boxcox_capability_dto(input: BoxcoxCapabilityInputDto) -> Result<BoxcoxCapabilityDto, String> {
    let result = crate::capability::boxcox_capability(&input.data, input.usl, input.lsl)
        .map_err(|e| e.to_string())?;
    let i = result.indices;
    Ok(BoxcoxCapabilityDto {
        lambda: result.lambda,
        cp: i.cp,
        cpk: i.cpk,
        cpu: i.cpu,
        cpl: i.cpl,
        pp: i.pp,
        ppk: i.ppk,
        ppu: i.ppu,
        ppl: i.ppl,
        cpm: i.cpm,
    })
}

/// Converts a sigma quality level to a defect rate in parts per million.
///
/// ```js
/// sigma_to_ppm(6.0);  // -> ~3.4
/// sigma_to_ppm(3.0);  // -> ~66807
/// ```
///
/// **This is the Motorola convention, which includes the 1.5-sigma shift**
/// (`PPM = 10^6 * (1 - Phi(sigma - 1.5))`). A "six sigma" process therefore
/// reports ~3.4 PPM rather than the ~0.002 PPM an unshifted normal tail gives.
/// A consumer comparing against an unshifted table will see a different number
/// for the same input, so the convention is named here rather than inferred.
///
/// Rejects a non-finite `sigma` instead of returning `NaN`.
#[wasm_bindgen]
pub fn sigma_to_ppm(sigma: f64) -> Result<f64, JsValue> {
    if !sigma.is_finite() {
        return Err(js_err(format!("sigma must be finite, got {sigma}")));
    }
    Ok(crate::capability::sigma_to_ppm(sigma))
}

/// Converts a defect rate in parts per million to a sigma quality level.
///
/// The inverse of [`sigma_to_ppm`], on the same (1.5-shifted) convention.
///
/// ```js
/// ppm_to_sigma(3.4);      // -> ~6.0
/// ppm_to_sigma(66807.0);  // -> ~3.0
/// ```
///
/// `ppm` must lie strictly inside `(0, 1_000_000)`: both ends are limits the
/// sigma scale does not reach, so they are rejected rather than mapped to an
/// infinity.
#[wasm_bindgen]
pub fn ppm_to_sigma(ppm: f64) -> Result<f64, JsValue> {
    crate::capability::ppm_to_sigma(ppm).ok_or_else(|| {
        js_err(format!(
            "ppm must be finite and strictly inside (0, 1000000), got {ppm}"
        ))
    })
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
    fn cusum_input_rejects_unknown_keys() {
        assert_rejects_unknown::<super::CusumInputDto>(
            json!({ "data": [1.0, 2.0], "target": 1.0, "sigma": 1.0, "H": 5.0 }),
        );
    }

    #[test]
    fn ewma_input_rejects_unknown_keys() {
        assert_rejects_unknown::<super::EwmaInputDto>(
            json!({ "data": [1.0, 2.0], "target": 1.0, "sigma": 1.0, "lFactor": 3.0 }),
        );
    }

    #[test]
    fn boxcox_capability_input_rejects_unknown_keys() {
        assert_rejects_unknown::<super::BoxcoxCapabilityInputDto>(
            json!({ "data": [1.0, 2.0, 3.0, 4.0], "USL": 9.0 }),
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

    fn dto(v: serde_json::Value) -> Result<crate::wire::CapabilityDto, String> {
        capability_dto(from_json::<CapabilityInputDto>(v, "input")?)
    }

    /// `sigma_hat` from the chart the measurements actually came from -- the
    /// quantity the flat vector cannot carry.
    fn sigma_within_from_chart() -> f64 {
        use crate::spc::{ControlChart, XBarRChart};
        let mut chart = XBarRChart::new(5).expect("5 is within range");
        for g in SUBGROUPS {
            chart.add_sample(&g).unwrap();
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
        let b = dto(overall)
            .expect("valid")
            .cpm
            .expect("cpm without sigma_within");
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
            chart.add_sample(g).unwrap();
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
            assert!(e.contains("subgroups[7]"), "{e}");
        }
    }

    #[test]
    fn imr_chart_reports_the_crate_chart() {
        use crate::spc::{ControlChart, IndividualMRChart, RuleSet};
        let values: Vec<f64> = shifted_subgroups().iter().map(|g| g[0]).collect();
        let d = imr_dto(values.clone(), RuleSet::default()).expect("valid input");

        let mut chart = IndividualMRChart::new();
        for &x in &values {
            chart.add_sample(&[x]).unwrap();
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
            chart.add_sample(x).unwrap();
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
            chart.add_sample(x, u).unwrap();
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

    // Box-Cox capability and sigma level <-> PPM

    #[test]
    fn boxcox_reports_lambda_and_long_term_indices_matching_the_crate() {
        let data: Vec<f64> = (1..=20).map(|i| (i as f64 * 0.3_f64).exp()).collect();
        let input: super::BoxcoxCapabilityInputDto =
            super::from_json(json!({ "data": data, "usl": 100.0, "lsl": 1.0 }), "input")
                .expect("valid input");
        let dto = super::boxcox_capability_dto(input).expect("skewed data is analysable");

        let native =
            crate::capability::boxcox_capability(&data, Some(100.0), Some(1.0)).expect("same call");
        assert!((dto.lambda - native.lambda).abs() < 1e-12);
        assert_eq!(dto.pp, native.indices.pp);
        assert_eq!(dto.ppk, native.indices.ppk);

        // A flat vector carries no subgroup structure, so the short-term
        // indices must stay absent rather than borrow the long-term sigma.
        assert!(dto.cp.is_none() && dto.cpk.is_none());
        assert!(
            dto.ppk.is_some(),
            "long-term indices are the available ones"
        );
    }

    #[test]
    fn boxcox_refuses_the_inputs_the_transform_cannot_take() {
        let cases = [
            json!({ "data": [1.0, 2.0, 0.0, 4.0], "usl": 9.0 }), // non-positive value
            json!({ "data": [1.0, 2.0, 3.0, 4.0] }),             // no specification limit
            json!({ "data": [1.0, 2.0, 3.0], "usl": 9.0 }),      // fewer than four points
        ];
        for case in cases {
            let parsed: super::BoxcoxCapabilityInputDto =
                super::from_json(case.clone(), "input").expect("parses");
            assert!(
                super::boxcox_capability_dto(parsed).is_err(),
                "should be refused: {case}"
            );
        }
    }

    #[test]
    fn sigma_and_ppm_round_trip_on_the_shifted_convention() {
        // The 1.5-sigma shift is the whole contract here: an unshifted table
        // would put six sigma near 0.002 PPM, not 3.4.
        for sigma in [3.0_f64, 4.5, 6.0] {
            let ppm = crate::capability::sigma_to_ppm(sigma);
            let back = crate::capability::ppm_to_sigma(ppm).expect("in range");
            // The inverse normal CDF here is a rational approximation, so the
            // round trip closes to ~3e-4, not to machine precision. Pinning the
            // measured accuracy rather than an aspirational one.
            assert!((back - sigma).abs() < 1e-3, "{sigma} -> {ppm} -> {back}");
        }
        assert!((crate::capability::sigma_to_ppm(6.0) - 3.4).abs() < 1.0);
        assert!((crate::capability::sigma_to_ppm(3.0) - 66_807.0).abs() < 500.0);
    }

    #[test]
    fn ppm_to_sigma_rejects_both_ends_of_the_range() {
        assert!(crate::capability::ppm_to_sigma(0.0).is_none());
        assert!(crate::capability::ppm_to_sigma(1_000_000.0).is_none());
        assert!(crate::capability::ppm_to_sigma(-1.0).is_none());
    }

    // ── CUSUM / EWMA ────────────────────────────────────────────────

    fn cusum_in(data: &[f64], target: f64, sigma: f64) -> super::CusumInputDto {
        super::from_json(
            json!({ "data": data, "target": target, "sigma": sigma }),
            "input",
        )
        .expect("valid cusum input")
    }

    fn ewma_in(data: &[f64], target: f64, sigma: f64) -> super::EwmaInputDto {
        super::from_json(
            json!({ "data": data, "target": target, "sigma": sigma }),
            "input",
        )
        .expect("valid ewma input")
    }

    #[test]
    fn cusum_defaults_match_the_crate_and_stay_in_control_on_stable_data() {
        let data = [10.1, 9.8, 10.2, 9.9, 10.0, 10.1, 9.7, 10.3];
        let dto = super::cusum_dto(cusum_in(&data, 10.0, 1.0)).expect("in-control run");

        // The binding must not invent its own defaults: k=0.5, h=5.0 (Page 1954).
        assert_eq!(dto.h, 5.0);
        assert_eq!(dto.points.len(), data.len());
        assert!(dto.in_control);
        assert!(dto.signal_indices.is_empty());

        // Same numbers the crate produces, point for point.
        let native = crate::detection::Cusum::new(10.0, 1.0)
            .expect("valid chart")
            .analyze(&data);
        for (p, n) in dto.points.iter().zip(native.iter()) {
            assert_eq!(p.index, n.index);
            assert!((p.s_upper - n.s_upper).abs() < 1e-12);
            assert!((p.s_lower - n.s_lower).abs() < 1e-12);
            assert_eq!(p.signal, n.signal);
        }
    }

    #[test]
    fn cusum_signals_a_sustained_upward_shift_and_reports_where() {
        let mut data = vec![10.0; 10];
        data.extend(vec![12.0; 10]);
        let dto = super::cusum_dto(cusum_in(&data, 10.0, 1.0)).expect("shifted run");

        assert!(!dto.in_control);
        assert!(!dto.signal_indices.is_empty());
        // Every flagged index is inside the shifted half, and signal_indices
        // agrees with the per-point flags (the redundancy must not drift).
        assert!(
            dto.signal_indices.iter().all(|&i| i >= 10),
            "{:?}",
            dto.signal_indices
        );
        let from_points: Vec<usize> = dto
            .points
            .iter()
            .filter(|p| p.signal)
            .map(|p| p.index)
            .collect();
        assert_eq!(from_points, dto.signal_indices);
    }

    #[test]
    fn cusum_refuses_empty_data_and_invalid_parameters() {
        let empty: super::CusumInputDto =
            super::from_json(json!({ "data": [], "target": 1.0, "sigma": 1.0 }), "input")
                .expect("parses");
        assert!(super::cusum_dto(empty)
            .expect_err("empty")
            .contains("empty"));

        let bad: super::CusumInputDto = super::from_json(
            json!({ "data": [1.0], "target": 1.0, "sigma": 0.0 }),
            "input",
        )
        .expect("parses");
        let e = super::cusum_dto(bad).expect_err("sigma must be > 0");
        assert!(e.contains("sigma"), "{e}");
    }

    #[test]
    fn ewma_limits_widen_with_the_index_and_match_the_crate() {
        let data = [10.1, 9.8, 10.2, 9.9, 10.0, 10.1];
        let dto = super::ewma_dto(ewma_in(&data, 10.0, 1.0)).expect("in-control run");

        assert_eq!(dto.points.len(), data.len());
        assert!(dto.in_control);

        // Exact (not asymptotic) limits: the half-width grows monotonically.
        let widths: Vec<f64> = dto.points.iter().map(|p| p.ucl - p.lcl).collect();
        for w in widths.windows(2) {
            assert!(w[1] > w[0], "limits must widen: {widths:?}");
        }

        let native = crate::detection::Ewma::new(10.0, 1.0)
            .expect("valid chart")
            .analyze(&data);
        for (p, n) in dto.points.iter().zip(native.iter()) {
            assert_eq!(p.index, n.index);
            assert!((p.ewma - n.ewma).abs() < 1e-12);
            assert!((p.ucl - n.ucl).abs() < 1e-12);
            assert!((p.lcl - n.lcl).abs() < 1e-12);
            assert_eq!(p.signal, n.signal);
        }
    }

    #[test]
    fn ewma_refuses_empty_data_and_a_lambda_outside_zero_to_one() {
        let empty: super::EwmaInputDto =
            super::from_json(json!({ "data": [], "target": 1.0, "sigma": 1.0 }), "input")
                .expect("parses");
        assert!(super::ewma_dto(empty).expect_err("empty").contains("empty"));

        for lambda in [0.0, 1.5] {
            let bad: super::EwmaInputDto = super::from_json(
                json!({ "data": [1.0, 2.0], "target": 1.0, "sigma": 1.0, "lambda": lambda }),
                "input",
            )
            .expect("parses");
            let e = super::ewma_dto(bad).expect_err("lambda out of range");
            assert!(e.contains("lambda"), "{e}");
        }
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
