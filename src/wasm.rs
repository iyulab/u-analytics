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

#[derive(Serialize)]
struct CapabilityDto {
    mean: f64,
    std_dev_within: f64,
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
/// `r_lcl`, `xbar_points`, `r_points`, `in_control`.
#[wasm_bindgen]
pub fn xbar_r_chart(data_json: JsValue) -> Result<JsValue, JsValue> {
    use crate::spc::{ControlChart, XBarRChart};

    let subgroups: Vec<Vec<f64>> = serde_wasm_bindgen::from_value(data_json)
        .map_err(|e| js_err(format!("invalid input: {e}")))?;

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
pub fn p_chart(samples_json: JsValue) -> Result<JsValue, JsValue> {
    use crate::spc::PChart;

    let raw: Vec<[u64; 2]> = serde_wasm_bindgen::from_value(samples_json).map_err(|e| {
        js_err(format!(
            "invalid input — expected [[defectives, size], ...]: {e}"
        ))
    })?;

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
/// Uses overall sigma for both short-term and long-term estimates
/// (`ProcessCapability::compute_overall`).
///
/// # Parameters
///
/// - `data`: slice of measurements
/// - `usl`: upper specification limit
/// - `lsl`: lower specification limit
///
/// # Output JSON
///
/// Object with fields: `mean`, `std_dev_within`, `std_dev_overall`,
/// `cp`, `cpk`, `cpu`, `cpl`, `pp`, `ppk`, `ppu`, `ppl`, `cpm`.
/// One-sided indices will have `null` for inapplicable fields.
#[wasm_bindgen]
pub fn process_capability(data: &[f64], usl: f64, lsl: f64) -> Result<JsValue, JsValue> {
    use crate::capability::ProcessCapability;

    let spec = ProcessCapability::new(Some(usl), Some(lsl))
        .map_err(|e| js_err(format!("invalid specification limits: {e}")))?;

    let indices = spec
        .compute_overall(data)
        .ok_or_else(|| js_err("insufficient or invalid data (need >= 2 finite values)"))?;

    let dto = CapabilityDto {
        mean: indices.mean,
        std_dev_within: indices.std_dev_within,
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
    };
    to_js(&dto)
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
pub fn laney_p_chart(samples_json: JsValue) -> Result<JsValue, JsValue> {
    let raw: Vec<[u64; 2]> = serde_wasm_bindgen::from_value(samples_json).map_err(|e| {
        js_err(format!(
            "invalid input — expected [[defectives, size], ...]: {e}"
        ))
    })?;

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
pub fn detect_changepoints(input_json: JsValue) -> Result<JsValue, JsValue> {
    let input: PeltInputDto = serde_wasm_bindgen::from_value(input_json)
        .map_err(|e| js_err(format!("invalid input: {e}")))?;

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
pub fn detect_changepoints_multi(input_json: JsValue) -> Result<JsValue, JsValue> {
    let input: MultiPeltInputDto = serde_wasm_bindgen::from_value(input_json)
        .map_err(|e| js_err(format!("invalid input: {e}")))?;

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
struct GageRRInputDto {
    measurements: Vec<Vec<Vec<f64>>>,
    tolerance: Option<f64>,
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
pub fn gage_rr_xbar_r(input_json: JsValue) -> Result<JsValue, JsValue> {
    let dto: GageRRInputDto = serde_wasm_bindgen::from_value(input_json)
        .map_err(|e| js_err(format!("invalid input: {e}")))?;

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
pub fn gage_rr_anova(input_json: JsValue) -> Result<JsValue, JsValue> {
    let dto: GageRRInputDto = serde_wasm_bindgen::from_value(input_json)
        .map_err(|e| js_err(format!("invalid input: {e}")))?;

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
pub fn percentile_capability(input_json: JsValue) -> Result<JsValue, JsValue> {
    #[derive(Deserialize)]
    struct Input {
        data: Vec<f64>,
        lsl: Option<f64>,
        usl: Option<f64>,
    }

    let input: Input = serde_wasm_bindgen::from_value(input_json)
        .map_err(|e| js_err(format!("invalid input: {e}")))?;

    let result =
        crate::capability::percentile_capability(&input.data, input.lsl, input.usl)
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
