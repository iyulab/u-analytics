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

#[derive(Serialize)]
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

#[derive(Serialize)]
struct ChartPointDto {
    index: usize,
    value: f64,
    violations: Vec<String>,
}

#[derive(Serialize)]
struct PChartDto {
    p_bar: f64,
    points: Vec<AttributeChartPointDto>,
    in_control: bool,
}

#[derive(Serialize)]
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
    /// Process target for Cpm. Defaults to the specification midpoint.
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

#[derive(Serialize)]
struct LaneyPChartDto {
    p_bar: f64,
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

/// Compute an X-bar R chart from subgroups.
///
/// # Input JSON
///
/// Array of arrays: `[[x1, x2, ...], [x1, x2, ...], ...]`
/// All subgroups must have the same length (2..=10).
///
/// # Output JSON
///
/// Object with fields: `xbar_cl`, `xbar_ucl`, `xbar_lcl`, `r_cl`, `r_ucl`,
/// `r_lcl`, `sigma_hat`, `xbar_points`, `r_points`, `in_control`.
///
/// `sigma_hat` is `R-bar / d2` -- the short-term sigma a capability study
/// needs. Pass it to [`process_capability`] as `sigma_within`; it is `null`
/// when there is not enough data for control limits.
#[wasm_bindgen]
pub fn xbar_r_chart(data: JsValue) -> Result<JsValue, JsValue> {
    use crate::spc::{ControlChart, XBarRChart};

    let subgroups: Vec<Vec<f64>> = from_js(data, "data")?;

    if subgroups.is_empty() {
        return Err(js_err("at least one subgroup required"));
    }
    let n = subgroups[0].len();
    if !(2..=10).contains(&n) {
        return Err(js_err(format!("subgroup size must be 2..=10, got {n}")));
    }
    if subgroups.iter().any(|g| g.len() != n) {
        return Err(js_err("all subgroups must have the same size"));
    }

    let mut chart = XBarRChart::new(n);
    for subgroup in &subgroups {
        chart.add_sample(subgroup);
    }

    let xbar_limits = chart
        .control_limits()
        .ok_or_else(|| js_err("insufficient data for control limits"))?;
    let r_limits = chart
        .r_limits()
        .ok_or_else(|| js_err("insufficient data for R chart limits"))?;

    let xbar_points = chart
        .points()
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
        .collect();
    let r_points = chart
        .r_points()
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
        .collect();

    let dto = XbarRChartDto {
        xbar_cl: xbar_limits.cl,
        xbar_ucl: xbar_limits.ucl,
        xbar_lcl: xbar_limits.lcl,
        r_cl: r_limits.cl,
        r_ucl: r_limits.ucl,
        r_lcl: r_limits.lcl,
        sigma_hat: chart.sigma_hat(),
        xbar_points,
        r_points,
        in_control: chart.is_in_control(),
    };
    to_js(&dto)
}

/// Compute a P chart from (defectives, sample_size) pairs.
///
/// # Input JSON
///
/// Array of `[defectives, sample_size]` pairs (as integers):
/// `[[3, 100], [5, 100], ...]`
///
/// # Output JSON
///
/// Object with fields: `p_bar`, `points` (array), `in_control`.
#[wasm_bindgen]
pub fn p_chart(samples: JsValue) -> Result<JsValue, JsValue> {
    use crate::spc::PChart;

    let raw: Vec<[u64; 2]> = from_js(samples, "samples (expected [[defectives, size], ...])")?;

    let mut chart = PChart::new();
    for pair in &raw {
        chart.add_sample(pair[0], pair[1]);
    }

    let p_bar = chart
        .p_bar()
        .ok_or_else(|| js_err("no valid samples provided"))?;

    let points = chart
        .points()
        .iter()
        .map(|p| AttributeChartPointDto {
            index: p.index,
            value: p.value,
            ucl: p.ucl,
            cl: p.cl,
            lcl: p.lcl,
            out_of_control: p.out_of_control,
        })
        .collect();

    let dto = PChartDto {
        p_bar,
        points,
        in_control: chart.is_in_control(),
    };
    to_js(&dto)
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
/// `target` sets the Cpm target. Omit it and the specification midpoint is
/// used, which is what Cpm falls back to when no target is declared.
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
/// `cpk`, `cpu`, `cpl` and `cpm` are all `null`: the short-term indices are
/// undefined without a short-term sigma, and reporting the long-term one in
/// their place would make `cp` equal `pp` for every input.
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
                cpm: None,
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
/// `[[3, 100], [5, 100], ...]` (need >= 3 subgroups)
///
/// # Output JSON
///
/// Object with fields: `p_bar`, `phi`, `points` (array).
#[wasm_bindgen]
pub fn laney_p_chart(samples: JsValue) -> Result<JsValue, JsValue> {
    let raw: Vec<[u64; 2]> = from_js(samples, "samples (expected [[defectives, size], ...])")?;

    let samples: Vec<(u64, u64)> = raw.into_iter().map(|p| (p[0], p[1])).collect();

    let chart = crate::spc::laney_p_chart(&samples)
        .ok_or_else(|| js_err("insufficient data (need >= 3 subgroups) or degenerate p_bar"))?;

    let points = chart
        .points
        .iter()
        .map(|p| AttributeChartPointDto {
            index: p.index,
            value: p.value,
            ucl: p.ucl,
            cl: p.cl,
            lcl: p.lcl,
            out_of_control: p.out_of_control,
        })
        .collect();

    let dto = LaneyPChartDto {
        p_bar: chart.p_bar,
        phi: chart.phi,
        points,
    };
    to_js(&dto)
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
        let mut chart = XBarRChart::new(5);
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
        assert!(d.cpm.is_none());
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
    fn target_reaches_cpm_instead_of_the_midpoint() {
        let base = json!({ "data": flat(), "usl": 11.0, "lsl": 9.0, "sigma_within": 0.2 });
        let midpoint = dto(base.clone()).expect("valid").cpm.expect("cpm");

        let mut shifted = base.clone();
        shifted["target"] = json!(10.4);
        let against_target = dto(shifted).expect("valid").cpm.expect("cpm");

        assert!(
            (midpoint - against_target).abs() > 1e-9,
            "Cpm against a declared target is a different quantity, not a rounder one"
        );

        // Stating the midpoint explicitly must reproduce the default exactly:
        // the fallback is the midpoint, not something near it.
        let mut explicit = base;
        explicit["target"] = json!(10.0);
        assert_eq!(dto(explicit).expect("valid").cpm, Some(midpoint));
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
}
