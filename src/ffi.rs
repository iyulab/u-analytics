//! FFI module for u-analytics — JSON-in/JSON-out pattern
//!
//! Error codes:
//!   0 = OK
//!  -1 = null pointer input
//!  -2 = JSON parse error
//!  -3 = computation error

#[cfg(feature = "ffi")]
use std::ffi::{CStr, CString};

#[cfg(feature = "ffi")]
use serde::{Deserialize, Serialize};

// ── Helpers ──────────────────────────────────────────────────

#[cfg(feature = "ffi")]
unsafe fn read_json(ptr: *const libc::c_char) -> Result<String, i32> {
    if ptr.is_null() {
        return Err(-1);
    }
    let cstr = unsafe { CStr::from_ptr(ptr) };
    cstr.to_str().map(|s| s.to_string()).map_err(|_| -2)
}

#[cfg(feature = "ffi")]
fn write_json<T: Serialize>(result_ptr: *mut *mut libc::c_char, value: &T) -> i32 {
    match serde_json::to_string(value) {
        Ok(json) => match CString::new(json) {
            Ok(cstr) => {
                unsafe { *result_ptr = cstr.into_raw() };
                0
            }
            Err(_) => -3,
        },
        Err(_) => -3,
    }
}

#[cfg(feature = "ffi")]
fn write_error(result_ptr: *mut *mut libc::c_char, msg: &str) -> i32 {
    let err = serde_json::json!({ "error": msg });
    write_json(result_ptr, &err);
    -3
}

// ── SPC: X-bar/R Chart ──────────────────────────────────────

#[cfg(feature = "ffi")]
#[derive(Deserialize)]
struct SpcChartRequest {
    subgroups: Vec<Vec<f64>>,
}

#[cfg(feature = "ffi")]
#[derive(Serialize)]
struct SpcChartResponse {
    x_bar_cl: f64,
    x_bar_ucl: f64,
    x_bar_lcl: f64,
    r_cl: f64,
    r_ucl: f64,
    r_lcl: f64,
    x_bars: Vec<f64>,
    ranges: Vec<f64>,
}

/// SPC X-bar/R control chart
#[cfg(feature = "ffi")]
#[no_mangle]
pub unsafe extern "C" fn uanalytics_xbar_r_chart(
    request_json: *const libc::c_char,
    result_ptr: *mut *mut libc::c_char,
) -> i32 {
    let json = match unsafe { read_json(request_json) } {
        Ok(j) => j,
        Err(e) => return e,
    };

    let req: SpcChartRequest = match serde_json::from_str(&json) {
        Ok(r) => r,
        Err(_) => return write_error(result_ptr, "Invalid JSON: expected {subgroups: [[f64]]}"),
    };

    if req.subgroups.is_empty() || req.subgroups[0].is_empty() {
        return write_error(result_ptr, "subgroups must not be empty");
    }

    let n = req.subgroups[0].len();

    // Compute X-bars and Ranges
    let mut x_bars = Vec::with_capacity(req.subgroups.len());
    let mut ranges = Vec::with_capacity(req.subgroups.len());

    for sg in &req.subgroups {
        let mean = sg.iter().sum::<f64>() / sg.len() as f64;
        let range = sg.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
            - sg.iter().cloned().fold(f64::INFINITY, f64::min);
        x_bars.push(mean);
        ranges.push(range);
    }

    let x_bar_bar = x_bars.iter().sum::<f64>() / x_bars.len() as f64;
    let r_bar = ranges.iter().sum::<f64>() / ranges.len() as f64;

    // A2, D3, D4 constants for subgroup size
    let (a2, d3, d4) = get_xbar_r_constants(n);

    let resp = SpcChartResponse {
        x_bar_cl: x_bar_bar,
        x_bar_ucl: x_bar_bar + a2 * r_bar,
        x_bar_lcl: x_bar_bar - a2 * r_bar,
        r_cl: r_bar,
        r_ucl: d4 * r_bar,
        r_lcl: d3 * r_bar,
        x_bars,
        ranges,
    };

    write_json(result_ptr, &resp)
}

#[cfg(feature = "ffi")]
fn get_xbar_r_constants(n: usize) -> (f64, f64, f64) {
    match n {
        2 => (1.880, 0.0, 3.267),
        3 => (1.023, 0.0, 2.574),
        4 => (0.729, 0.0, 2.282),
        5 => (0.577, 0.0, 2.114),
        6 => (0.483, 0.0, 2.004),
        7 => (0.419, 0.076, 1.924),
        8 => (0.373, 0.136, 1.864),
        9 => (0.337, 0.184, 1.816),
        10 => (0.308, 0.223, 1.777),
        _ => (0.308, 0.223, 1.777), // default to n=10
    }
}

// ── SPC: P Chart ────────────────────────────────────────────

#[cfg(feature = "ffi")]
#[derive(Deserialize)]
struct PChartRequest {
    samples: Vec<(u64, u64)>, // (inspected, defective)
}

#[cfg(feature = "ffi")]
#[derive(Serialize)]
struct PChartResponse {
    p_bar: f64,
    proportions: Vec<f64>,
    ucls: Vec<f64>,
    lcls: Vec<f64>,
}

/// SPC P chart
#[cfg(feature = "ffi")]
#[no_mangle]
pub unsafe extern "C" fn uanalytics_p_chart(
    request_json: *const libc::c_char,
    result_ptr: *mut *mut libc::c_char,
) -> i32 {
    let json = match unsafe { read_json(request_json) } {
        Ok(j) => j,
        Err(e) => return e,
    };

    let req: PChartRequest = match serde_json::from_str(&json) {
        Ok(r) => r,
        Err(_) => return write_error(result_ptr, "Invalid JSON"),
    };

    let total_inspected: u64 = req.samples.iter().map(|s| s.0).sum();
    let total_defective: u64 = req.samples.iter().map(|s| s.1).sum();
    let p_bar = total_defective as f64 / total_inspected as f64;

    let mut proportions = Vec::with_capacity(req.samples.len());
    let mut ucls = Vec::with_capacity(req.samples.len());
    let mut lcls = Vec::with_capacity(req.samples.len());

    for (n, d) in &req.samples {
        let p = *d as f64 / *n as f64;
        let sigma = (p_bar * (1.0 - p_bar) / *n as f64).sqrt();
        proportions.push(p);
        ucls.push(p_bar + 3.0 * sigma);
        lcls.push((p_bar - 3.0 * sigma).max(0.0));
    }

    write_json(result_ptr, &PChartResponse { p_bar, proportions, ucls, lcls })
}

// ── SPC: Laney P' Chart ─────────────────────────────────────

/// SPC Laney P' chart (delegates to crate's laney_p_chart)
#[cfg(feature = "ffi")]
#[no_mangle]
pub unsafe extern "C" fn uanalytics_laney_p_chart(
    request_json: *const libc::c_char,
    result_ptr: *mut *mut libc::c_char,
) -> i32 {
    let json = match unsafe { read_json(request_json) } {
        Ok(j) => j,
        Err(e) => return e,
    };

    let req: PChartRequest = match serde_json::from_str(&json) {
        Ok(r) => r,
        Err(_) => return write_error(result_ptr, "Invalid JSON"),
    };

    match crate::spc::laney_p_chart(&req.samples) {
        Some(result) => {
            let proportions: Vec<f64> = result.points.iter().map(|p| p.value).collect();
            let ucls: Vec<f64> = result.points.iter().map(|p| p.ucl).collect();
            let lcls: Vec<f64> = result.points.iter().map(|p| p.lcl).collect();
            let resp = serde_json::json!({
                "p_bar": result.p_bar,
                "phi": result.phi,
                "proportions": proportions,
                "ucls": ucls,
                "lcls": lcls,
            });
            write_json(result_ptr, &resp)
        }
        None => write_error(result_ptr, "Laney P' chart computation failed"),
    }
}

// ── Process Capability ──────────────────────────────────────

#[cfg(feature = "ffi")]
#[derive(Deserialize)]
struct CapabilityRequest {
    data: Vec<f64>,
    usl: Option<f64>,
    lsl: Option<f64>,
    target: Option<f64>,
}

/// Process capability analysis (Cp, Cpk, Pp, Ppk, Cpm)
#[cfg(feature = "ffi")]
#[no_mangle]
pub unsafe extern "C" fn uanalytics_process_capability(
    request_json: *const libc::c_char,
    result_ptr: *mut *mut libc::c_char,
) -> i32 {
    let json = match unsafe { read_json(request_json) } {
        Ok(j) => j,
        Err(e) => return e,
    };

    let req: CapabilityRequest = match serde_json::from_str(&json) {
        Ok(r) => r,
        Err(_) => return write_error(result_ptr, "Invalid JSON"),
    };

    let pc = match crate::capability::ProcessCapability::new(req.usl, req.lsl) {
        Ok(pc) => match req.target {
            Some(t) => pc.with_target(t),
            None => pc,
        },
        Err(e) => return write_error(result_ptr, e),
    };

    // Estimate sigma_within from data (using moving range)
    let sigma_within = estimate_sigma_within(&req.data);

    match pc.compute(&req.data, sigma_within) {
        Some(indices) => {
            let resp = serde_json::json!({
                "cp": indices.cp,
                "cpk": indices.cpk,
                "cpu": indices.cpu,
                "cpl": indices.cpl,
                "pp": indices.pp,
                "ppk": indices.ppk,
                "ppu": indices.ppu,
                "ppl": indices.ppl,
                "cpm": indices.cpm,
                "mean": indices.mean,
                "std_dev_within": indices.std_dev_within,
                "std_dev_overall": indices.std_dev_overall,
            });
            write_json(result_ptr, &resp)
        }
        None => write_error(result_ptr, "Capability computation failed"),
    }
}

#[cfg(feature = "ffi")]
fn estimate_sigma_within(data: &[f64]) -> f64 {
    if data.len() < 2 {
        return 0.0;
    }
    let mr_sum: f64 = data.windows(2).map(|w| (w[1] - w[0]).abs()).sum();
    let mr_bar = mr_sum / (data.len() - 1) as f64;
    mr_bar / 1.128 // d2 for n=2
}

// ── Percentile Capability ───────────────────────────────────

#[cfg(feature = "ffi")]
#[derive(Deserialize)]
struct PercentileCapabilityRequest {
    data: Vec<f64>,
    usl: Option<f64>,
    lsl: Option<f64>,
}

/// Percentile-based process capability
#[cfg(feature = "ffi")]
#[no_mangle]
pub unsafe extern "C" fn uanalytics_percentile_capability(
    request_json: *const libc::c_char,
    result_ptr: *mut *mut libc::c_char,
) -> i32 {
    let json = match unsafe { read_json(request_json) } {
        Ok(j) => j,
        Err(e) => return e,
    };

    let req: PercentileCapabilityRequest = match serde_json::from_str(&json) {
        Ok(r) => r,
        Err(_) => return write_error(result_ptr, "Invalid JSON"),
    };

    match crate::capability::percentile_capability(&req.data, req.lsl, req.usl) {
        Ok(result) => {
            let resp = serde_json::json!({
                "cp_star": result.cp_star,
                "cpk_star": result.cpk_star,
                "cpu_star": result.cpu_star,
                "cpl_star": result.cpl_star,
                "median": result.median,
            });
            write_json(result_ptr, &resp)
        }
        Err(e) => write_error(result_ptr, e),
    }
}

// ── MSA: Gage R&R X-bar/R ───────────────────────────────────

#[cfg(feature = "ffi")]
#[derive(Deserialize)]
struct GageRRRequest {
    measurements: Vec<Vec<Vec<f64>>>, // [part][operator][trial]
    tolerance: Option<f64>,
}

/// Gage R&R (X-bar/R method)
#[cfg(feature = "ffi")]
#[no_mangle]
pub unsafe extern "C" fn uanalytics_gage_rr_xbar_r(
    request_json: *const libc::c_char,
    result_ptr: *mut *mut libc::c_char,
) -> i32 {
    let json = match unsafe { read_json(request_json) } {
        Ok(j) => j,
        Err(e) => return e,
    };

    let req: GageRRRequest = match serde_json::from_str(&json) {
        Ok(r) => r,
        Err(_) => return write_error(result_ptr, "Invalid JSON"),
    };

    let input = crate::msa::GageRRInput {
        measurements: req.measurements,
        tolerance: req.tolerance,
    };

    match crate::msa::gage_rr_xbar_r(&input) {
        Ok(result) => {
            let resp = serde_json::json!({
                "ev": result.ev,
                "av": result.av,
                "grr": result.grr,
                "pv": result.pv,
                "tv": result.tv,
                "percent_ev": result.percent_ev,
                "percent_av": result.percent_av,
                "percent_grr": result.percent_grr,
                "percent_pv": result.percent_pv,
                "percent_tolerance": result.percent_tolerance,
                "ndc": result.ndc,
                "status": format!("{:?}", result.status),
            });
            write_json(result_ptr, &resp)
        }
        Err(e) => write_error(result_ptr, e),
    }
}

/// Gage R&R (ANOVA method)
#[cfg(feature = "ffi")]
#[no_mangle]
pub unsafe extern "C" fn uanalytics_gage_rr_anova(
    request_json: *const libc::c_char,
    result_ptr: *mut *mut libc::c_char,
) -> i32 {
    let json = match unsafe { read_json(request_json) } {
        Ok(j) => j,
        Err(e) => return e,
    };

    let req: GageRRRequest = match serde_json::from_str(&json) {
        Ok(r) => r,
        Err(_) => return write_error(result_ptr, "Invalid JSON"),
    };

    let input = crate::msa::GageRRInput {
        measurements: req.measurements,
        tolerance: req.tolerance,
    };

    match crate::msa::gage_rr_anova(&input) {
        Ok(result) => {
            let resp = serde_json::json!({
                "ev": result.ev,
                "av": result.av,
                "grr": result.grr,
                "pv": result.pv,
                "tv": result.tv,
                "percent_grr": result.percent_grr,
                "ndc": result.ndc,
                "status": format!("{:?}", result.status),
                "interaction_significant": result.interaction_significant,
                "interaction_pooled": result.interaction_pooled,
            });
            write_json(result_ptr, &resp)
        }
        Err(e) => write_error(result_ptr, e),
    }
}

// ── Weibull MLE ─────────────────────────────────────────────

#[cfg(feature = "ffi")]
#[derive(Deserialize)]
struct WeibullRequest {
    failure_times: Vec<f64>,
}

/// Weibull MLE parameter estimation
#[cfg(feature = "ffi")]
#[no_mangle]
pub unsafe extern "C" fn uanalytics_weibull_mle(
    request_json: *const libc::c_char,
    result_ptr: *mut *mut libc::c_char,
) -> i32 {
    let json = match unsafe { read_json(request_json) } {
        Ok(j) => j,
        Err(e) => return e,
    };

    let req: WeibullRequest = match serde_json::from_str(&json) {
        Ok(r) => r,
        Err(_) => return write_error(result_ptr, "Invalid JSON"),
    };

    match crate::weibull::weibull_mle(&req.failure_times) {
        Some(result) => {
            let resp = serde_json::json!({
                "shape": result.shape,
                "scale": result.scale,
            });
            write_json(result_ptr, &resp)
        }
        None => write_error(result_ptr, "Weibull MLE estimation failed"),
    }
}

// ── Change-Point Detection (PELT) ───────────────────────────

#[cfg(feature = "ffi")]
#[derive(Deserialize)]
struct PeltRequest {
    data: Vec<f64>,
    penalty: Option<String>, // "BIC", "AIC", "MBIC", or numeric
    min_segment_len: Option<usize>,
}

/// PELT change-point detection
#[cfg(feature = "ffi")]
#[no_mangle]
pub unsafe extern "C" fn uanalytics_detect_changepoints(
    request_json: *const libc::c_char,
    result_ptr: *mut *mut libc::c_char,
) -> i32 {
    let json = match unsafe { read_json(request_json) } {
        Ok(j) => j,
        Err(e) => return e,
    };

    let req: PeltRequest = match serde_json::from_str(&json) {
        Ok(r) => r,
        Err(_) => return write_error(result_ptr, "Invalid JSON"),
    };

    let penalty = match req.penalty.as_deref() {
        Some(s) => match s.parse::<f64>() {
            Ok(v) => crate::detection::Penalty::Custom(v),
            Err(_) => crate::detection::Penalty::Bic,
        },
        None => crate::detection::Penalty::Bic,
    };

    let min_seg = req.min_segment_len.unwrap_or(2);

    let pelt = match crate::detection::Pelt::with_min_segment_len(
        crate::detection::CostFunction::Normal,
        penalty,
        min_seg,
    ) {
        Some(p) => p,
        None => return write_error(result_ptr, "Failed to create PELT detector"),
    };

    let result = pelt.detect(&req.data);

    let resp = serde_json::json!({
        "changepoints": result.changepoints,
    });
    write_json(result_ptr, &resp)
}

// ── Correlation Matrix ──────────────────────────────────────

#[cfg(feature = "ffi")]
#[derive(Deserialize)]
struct CorrelationRequest {
    variables: Vec<Vec<f64>>,
}

/// Correlation matrix (Pearson)
#[cfg(feature = "ffi")]
#[no_mangle]
pub unsafe extern "C" fn uanalytics_correlation_matrix(
    request_json: *const libc::c_char,
    result_ptr: *mut *mut libc::c_char,
) -> i32 {
    let json = match unsafe { read_json(request_json) } {
        Ok(j) => j,
        Err(e) => return e,
    };

    let req: CorrelationRequest = match serde_json::from_str(&json) {
        Ok(r) => r,
        Err(_) => return write_error(result_ptr, "Invalid JSON"),
    };

    let refs: Vec<&[f64]> = req.variables.iter().map(|v| v.as_slice()).collect();

    match crate::correlation::correlation_matrix(&refs) {
        Some(matrix) => {
            let resp = serde_json::json!({
                "rows": matrix.rows(),
                "cols": matrix.cols(),
                "data": matrix.data(),
            });
            write_json(result_ptr, &resp)
        }
        None => write_error(result_ptr, "Correlation matrix computation failed"),
    }
}

// ── Simple Regression ───────────────────────────────────────

#[cfg(feature = "ffi")]
#[derive(Deserialize)]
struct RegressionRequest {
    x: Vec<f64>,
    y: Vec<f64>,
}

/// Simple linear regression
#[cfg(feature = "ffi")]
#[no_mangle]
pub unsafe extern "C" fn uanalytics_simple_regression(
    request_json: *const libc::c_char,
    result_ptr: *mut *mut libc::c_char,
) -> i32 {
    let json = match unsafe { read_json(request_json) } {
        Ok(j) => j,
        Err(e) => return e,
    };

    let req: RegressionRequest = match serde_json::from_str(&json) {
        Ok(r) => r,
        Err(_) => return write_error(result_ptr, "Invalid JSON"),
    };

    match crate::regression::simple_linear_regression(&req.x, &req.y) {
        Some(result) => {
            let resp = serde_json::json!({
                "slope": result.slope,
                "intercept": result.intercept,
                "r_squared": result.r_squared,
                "adjusted_r_squared": result.adjusted_r_squared,
                "slope_se": result.slope_se,
                "intercept_se": result.intercept_se,
            });
            write_json(result_ptr, &resp)
        }
        None => write_error(result_ptr, "Regression computation failed"),
    }
}

// ── Distribution Fit ────────────────────────────────────────

#[cfg(feature = "ffi")]
#[derive(Deserialize)]
struct FitBestRequest {
    data: Vec<f64>,
}

/// Fit best distribution
#[cfg(feature = "ffi")]
#[no_mangle]
pub unsafe extern "C" fn uanalytics_fit_best(
    request_json: *const libc::c_char,
    result_ptr: *mut *mut libc::c_char,
) -> i32 {
    let json = match unsafe { read_json(request_json) } {
        Ok(j) => j,
        Err(e) => return e,
    };

    let req: FitBestRequest = match serde_json::from_str(&json) {
        Ok(r) => r,
        Err(_) => return write_error(result_ptr, "Invalid JSON"),
    };

    let results = crate::distribution::fit_best(&req.data);

    let resp: Vec<serde_json::Value> = results
        .iter()
        .map(|r| {
            serde_json::json!({
                "distribution": r.distribution,
                "parameters": r.parameters,
                "log_likelihood": r.log_likelihood,
                "aic": r.aic,
                "bic": r.bic,
            })
        })
        .collect();

    write_json(result_ptr, &resp)
}

// ── Memory Management ───────────────────────────────────────

/// Free a string allocated by u-analytics FFI functions
#[cfg(feature = "ffi")]
#[no_mangle]
pub unsafe extern "C" fn uanalytics_free_string(ptr: *mut libc::c_char) {
    if !ptr.is_null() {
        unsafe { drop(CString::from_raw(ptr)) };
    }
}

/// Get u-analytics version string
#[cfg(feature = "ffi")]
#[no_mangle]
pub extern "C" fn uanalytics_version() -> *mut libc::c_char {
    let version = env!("CARGO_PKG_VERSION");
    CString::new(version).expect("version string has no interior NUL").into_raw()
}
