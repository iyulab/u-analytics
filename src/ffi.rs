//! FFI module for u-analytics — JSON-in/JSON-out pattern
//!
//! Error codes:
//!   0 = OK
//!  -1 = null pointer input
//!  -2 = JSON parse error
//!  -3 = computation error
//!  -4 = internal panic
//!
//! All FFI entry points are wrapped in `catch_unwind` to prevent panic propagation.

#[cfg(feature = "ffi")]
use std::ffi::{CStr, CString};
#[cfg(feature = "ffi")]
use std::panic;

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
    if result_ptr.is_null() {
        return -1;
    }
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

/// Status for a request whose JSON could not be read into the expected shape.
#[cfg(feature = "ffi")]
const ERR_PARSE: i32 = -2;

/// Status for a well-formed request the computation rejected.
#[cfg(feature = "ffi")]
const ERR_COMPUTE: i32 = -3;

/// Writes `{"error": msg}` and returns `status`.
///
/// The status is the caller's to choose and is returned as given: the error
/// body is a diagnostic, not a result, so writing it successfully must not
/// turn the call into a success.
#[cfg(feature = "ffi")]
fn write_error(result_ptr: *mut *mut libc::c_char, status: i32, msg: &str) -> i32 {
    let err = serde_json::json!({ "error": msg });
    match write_json(result_ptr, &err) {
        0 => status,
        write_failure => write_failure,
    }
}

/// Parses a request body, reporting a malformed one under [`ERR_PARSE`].
#[cfg(feature = "ffi")]
fn parse_request<T: serde::de::DeserializeOwned>(
    json: &str,
    result_ptr: *mut *mut libc::c_char,
) -> Result<T, i32> {
    serde_json::from_str(json)
        .map_err(|e| write_error(result_ptr, ERR_PARSE, &format!("Invalid JSON: {e}")))
}

/// Wraps an FFI body in `catch_unwind`, initializing `result_ptr` to null.
#[cfg(feature = "ffi")]
fn ffi_catch(
    result_ptr: *mut *mut libc::c_char,
    f: impl FnOnce() -> i32 + panic::UnwindSafe,
) -> i32 {
    if !result_ptr.is_null() {
        unsafe { *result_ptr = std::ptr::null_mut() };
    }
    match panic::catch_unwind(f) {
        Ok(code) => code,
        Err(_) => write_error(result_ptr, -4, "internal panic"),
    }
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
    /// Short-term sigma implied by the chart (`R-bar / d2`) -- the
    /// `sigma_within` a capability study of the same data needs.
    sigma_hat: Option<f64>,
    x_bars: Vec<f64>,
    ranges: Vec<f64>,
}

/// SPC X-bar/R control chart.
///
/// A thin adapter over [`crate::spc::XBarRChart`]: the supported subgroup
/// sizes, the factor tables and the arithmetic are all the crate's. This entry
/// point used to carry its own copy of each, which stayed at n <= 10 with
/// three-decimal constants after the crate had moved on.
///
/// # Safety
///
/// `request_json` must be null or point to a NUL-terminated string, and
/// `result_ptr` must be null or valid for writing one pointer. A string written
/// there is owned by the caller and must be released with
/// [`uanalytics_free_string`].
#[cfg(feature = "ffi")]
#[no_mangle]
pub unsafe extern "C" fn uanalytics_xbar_r_chart(
    request_json: *const libc::c_char,
    result_ptr: *mut *mut libc::c_char,
) -> i32 {
    use crate::spc::{ControlChart, XBarRChart};

    ffi_catch(result_ptr, || {
        let json = match unsafe { read_json(request_json) } {
            Ok(j) => j,
            Err(e) => return e,
        };
        let req: SpcChartRequest = match parse_request(&json, result_ptr) {
            Ok(r) => r,
            Err(status) => return status,
        };

        let Some(n) = req.subgroups.first().map(Vec::len) else {
            return write_error(result_ptr, ERR_COMPUTE, "subgroups must not be empty");
        };
        let mut chart = match XBarRChart::new(n) {
            Ok(chart) => chart,
            Err(e) => return write_error(result_ptr, ERR_COMPUTE, &e.to_string()),
        };
        // `x_bars` and `ranges` carry no index, so a ragged or non-finite
        // subgroup is refused by its row rather than left out.
        for (i, subgroup) in req.subgroups.iter().enumerate() {
            if let Err(e) = chart.add_sample(subgroup) {
                return write_error(result_ptr, ERR_COMPUTE, &format!("subgroup {i}: {e}"));
            }
        }

        let (Some(x), Some(r)) = (chart.control_limits(), chart.r_limits()) else {
            return write_error(
                result_ptr,
                ERR_COMPUTE,
                "insufficient data for control limits",
            );
        };
        let resp = SpcChartResponse {
            x_bar_cl: x.cl,
            x_bar_ucl: x.ucl,
            x_bar_lcl: x.lcl,
            r_cl: r.cl,
            r_ucl: r.ucl,
            r_lcl: r.lcl,
            sigma_hat: chart.sigma_hat(),
            x_bars: chart.points().iter().map(|p| p.value).collect(),
            ranges: chart.r_points().iter().map(|p| p.value).collect(),
        };
        write_json(result_ptr, &resp)
    })
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

/// Finds the first sample with no defined proportion: nothing inspected, or
/// more defectives than inspected items.
#[cfg(feature = "ffi")]
fn invalid_proportion_sample(samples: &[(u64, u64)]) -> Option<String> {
    samples
        .iter()
        .position(|&(inspected, defective)| inspected == 0 || defective > inspected)
        .map(|i| {
            let (inspected, defective) = samples[i];
            format!(
                "sample {i} has {defective} defectives out of {inspected} inspected; \
                 each sample needs 0 <= defective <= inspected and inspected > 0"
            )
        })
}

/// SPC P chart.
///
/// A thin adapter over [`crate::spc::PChart`]. Requests carry
/// `[inspected, defective]` pairs.
///
/// # Safety
///
/// `request_json` must be null or point to a NUL-terminated string, and
/// `result_ptr` must be null or valid for writing one pointer. A string written
/// there is owned by the caller and must be released with
/// [`uanalytics_free_string`].
#[cfg(feature = "ffi")]
#[no_mangle]
pub unsafe extern "C" fn uanalytics_p_chart(
    request_json: *const libc::c_char,
    result_ptr: *mut *mut libc::c_char,
) -> i32 {
    use crate::spc::PChart;

    ffi_catch(result_ptr, || {
        let json = match unsafe { read_json(request_json) } {
            Ok(j) => j,
            Err(e) => return e,
        };
        let req: PChartRequest = match parse_request(&json, result_ptr) {
            Ok(r) => r,
            Err(status) => return status,
        };
        // The chart drops a sample it cannot use, and the response arrays
        // carry no index -- a dropped sample would misalign every later row.
        if let Some(msg) = invalid_proportion_sample(&req.samples) {
            return write_error(result_ptr, ERR_COMPUTE, &msg);
        }

        let mut chart = PChart::new();
        for (i, &(inspected, defective)) in req.samples.iter().enumerate() {
            if let Err(e) = chart.add_sample(defective, inspected) {
                return write_error(result_ptr, ERR_COMPUTE, &format!("sample {i}: {e}"));
            }
        }
        let Some(p_bar) = chart.p_bar() else {
            return write_error(result_ptr, ERR_COMPUTE, "samples must not be empty");
        };

        let points = chart.points();
        write_json(
            result_ptr,
            &PChartResponse {
                p_bar,
                proportions: points.iter().map(|p| p.value).collect(),
                ucls: points.iter().map(|p| p.ucl).collect(),
                lcls: points.iter().map(|p| p.lcl).collect(),
            },
        )
    })
}

// ── SPC: Laney P' Chart ─────────────────────────────────────

/// SPC Laney P' chart (delegates to crate's laney_p_chart)
///
/// # Safety
///
/// `request_json` must be null or point to a NUL-terminated string, and
/// `result_ptr` must be null or valid for writing one pointer. A string written
/// there is owned by the caller and must be released with
/// [`uanalytics_free_string`].
#[cfg(feature = "ffi")]
#[no_mangle]
pub unsafe extern "C" fn uanalytics_laney_p_chart(
    request_json: *const libc::c_char,
    result_ptr: *mut *mut libc::c_char,
) -> i32 {
    ffi_catch(result_ptr, || {
        let json = match unsafe { read_json(request_json) } {
            Ok(j) => j,
            Err(e) => return e,
        };

        let req: PChartRequest = match parse_request(&json, result_ptr) {
            Ok(r) => r,
            Err(status) => return status,
        };

        if let Some(msg) = invalid_proportion_sample(&req.samples) {
            return write_error(result_ptr, ERR_COMPUTE, &msg);
        }
        // The request carries `[inspected, defective]`, like the P chart's;
        // the crate function takes `(defective, sample_size)`.
        let samples: Vec<(u64, u64)> = req
            .samples
            .iter()
            .map(|&(inspected, defective)| (defective, inspected))
            .collect();

        match crate::spc::laney_p_chart(&samples) {
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
            None => write_error(result_ptr, ERR_COMPUTE, "Laney P' chart computation failed"),
        }
    })
}

// ── Process Capability ──────────────────────────────────────

#[cfg(feature = "ffi")]
#[derive(Deserialize)]
struct CapabilityRequest {
    data: Vec<f64>,
    usl: Option<f64>,
    lsl: Option<f64>,
    target: Option<f64>,
    /// Short-term sigma from a control chart (`R-bar/d2`, `S-bar/c4`; the
    /// X-bar/R entry point returns it as `sigma_hat`).
    sigma_within: Option<f64>,
}

/// Process capability analysis (Cp, Cpk, Pp, Ppk, Cpm).
///
/// The short-term indices need a within sigma. `sigma_within` supplies it;
/// without one it is estimated from the moving range of `data` taken in order
/// (`MR-bar / d2(2)`, as an Individual-MR chart does), which is the standard
/// estimate for individual observations and wrong for data that was
/// flattened out of subgroups. `sigma_source` in the response says which was
/// used -- `"within"` or `"moving_range"` -- so the assumption is visible
/// rather than folded silently into `cp`.
///
/// `cpm` is `null` unless both limits and a `target` are given; it uses
/// neither sigma, but the spread of `data` about the target.
///
/// # Safety
///
/// `request_json` must be null or point to a NUL-terminated string, and
/// `result_ptr` must be null or valid for writing one pointer. A string written
/// there is owned by the caller and must be released with
/// [`uanalytics_free_string`].
#[cfg(feature = "ffi")]
#[no_mangle]
pub unsafe extern "C" fn uanalytics_process_capability(
    request_json: *const libc::c_char,
    result_ptr: *mut *mut libc::c_char,
) -> i32 {
    use crate::spc::{ControlChart, IndividualMRChart};

    ffi_catch(result_ptr, || {
        let json = match unsafe { read_json(request_json) } {
            Ok(j) => j,
            Err(e) => return e,
        };
        let req: CapabilityRequest = match parse_request(&json, result_ptr) {
            Ok(r) => r,
            Err(status) => return status,
        };

        let pc = match crate::capability::ProcessCapability::new(req.usl, req.lsl) {
            Ok(pc) => match req.target {
                Some(t) => pc.with_target(t),
                None => pc,
            },
            Err(e) => return write_error(result_ptr, ERR_COMPUTE, e),
        };

        let (sigma_within, sigma_source) = match req.sigma_within {
            Some(s) if s.is_finite() && s > 0.0 => (s, "within"),
            Some(_) => {
                return write_error(
                    result_ptr,
                    ERR_COMPUTE,
                    "sigma_within must be a positive, finite number",
                )
            }
            None => {
                let mut imr = IndividualMRChart::new();
                for (i, &x) in req.data.iter().enumerate() {
                    if let Err(e) = imr.add_sample(&[x]) {
                        return write_error(result_ptr, ERR_COMPUTE, &format!("data[{i}]: {e}"));
                    }
                }
                match imr.sigma_hat() {
                    Some(s) if s > 0.0 => (s, "moving_range"),
                    _ => {
                        return write_error(
                            result_ptr,
                            ERR_COMPUTE,
                            "cannot estimate a within sigma from data: need at least two \
                             finite values that are not all equal, or supply sigma_within",
                        )
                    }
                }
            }
        };

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
                    "sigma_source": sigma_source,
                    "std_dev_within": indices.std_dev_within,
                    "std_dev_overall": indices.std_dev_overall,
                });
                write_json(result_ptr, &resp)
            }
            None => write_error(
                result_ptr,
                ERR_COMPUTE,
                "insufficient or invalid data (need >= 2 finite values)",
            ),
        }
    })
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
///
/// # Safety
///
/// `request_json` must be null or point to a NUL-terminated string, and
/// `result_ptr` must be null or valid for writing one pointer. A string written
/// there is owned by the caller and must be released with
/// [`uanalytics_free_string`].
#[cfg(feature = "ffi")]
#[no_mangle]
pub unsafe extern "C" fn uanalytics_percentile_capability(
    request_json: *const libc::c_char,
    result_ptr: *mut *mut libc::c_char,
) -> i32 {
    ffi_catch(result_ptr, || {
        let json = match unsafe { read_json(request_json) } {
            Ok(j) => j,
            Err(e) => return e,
        };

        let req: PercentileCapabilityRequest = match parse_request(&json, result_ptr) {
            Ok(r) => r,
            Err(status) => return status,
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
            Err(e) => write_error(result_ptr, ERR_COMPUTE, e),
        }
    })
}

// ── MSA: Gage R&R X-bar/R ───────────────────────────────────

#[cfg(feature = "ffi")]
#[derive(Deserialize)]
struct GageRRRequest {
    measurements: Vec<Vec<Vec<f64>>>, // [part][operator][trial]
    tolerance: Option<f64>,
}

/// Gage R&R (X-bar/R method)
///
/// # Safety
///
/// `request_json` must be null or point to a NUL-terminated string, and
/// `result_ptr` must be null or valid for writing one pointer. A string written
/// there is owned by the caller and must be released with
/// [`uanalytics_free_string`].
#[cfg(feature = "ffi")]
#[no_mangle]
pub unsafe extern "C" fn uanalytics_gage_rr_xbar_r(
    request_json: *const libc::c_char,
    result_ptr: *mut *mut libc::c_char,
) -> i32 {
    ffi_catch(result_ptr, || {
        let json = match unsafe { read_json(request_json) } {
            Ok(j) => j,
            Err(e) => return e,
        };

        let req: GageRRRequest = match parse_request(&json, result_ptr) {
            Ok(r) => r,
            Err(status) => return status,
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
            Err(e) => write_error(result_ptr, ERR_COMPUTE, e),
        }
    })
}

/// Gage R&R (ANOVA method)
///
/// # Safety
///
/// `request_json` must be null or point to a NUL-terminated string, and
/// `result_ptr` must be null or valid for writing one pointer. A string written
/// there is owned by the caller and must be released with
/// [`uanalytics_free_string`].
#[cfg(feature = "ffi")]
#[no_mangle]
pub unsafe extern "C" fn uanalytics_gage_rr_anova(
    request_json: *const libc::c_char,
    result_ptr: *mut *mut libc::c_char,
) -> i32 {
    ffi_catch(result_ptr, || {
        let json = match unsafe { read_json(request_json) } {
            Ok(j) => j,
            Err(e) => return e,
        };

        let req: GageRRRequest = match parse_request(&json, result_ptr) {
            Ok(r) => r,
            Err(status) => return status,
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
            Err(e) => write_error(result_ptr, ERR_COMPUTE, e),
        }
    })
}

// ── Weibull MLE ─────────────────────────────────────────────

#[cfg(feature = "ffi")]
#[derive(Deserialize)]
struct WeibullRequest {
    failure_times: Vec<f64>,
}

/// Weibull MLE parameter estimation
///
/// # Safety
///
/// `request_json` must be null or point to a NUL-terminated string, and
/// `result_ptr` must be null or valid for writing one pointer. A string written
/// there is owned by the caller and must be released with
/// [`uanalytics_free_string`].
#[cfg(feature = "ffi")]
#[no_mangle]
pub unsafe extern "C" fn uanalytics_weibull_mle(
    request_json: *const libc::c_char,
    result_ptr: *mut *mut libc::c_char,
) -> i32 {
    ffi_catch(result_ptr, || {
        let json = match unsafe { read_json(request_json) } {
            Ok(j) => j,
            Err(e) => return e,
        };

        let req: WeibullRequest = match parse_request(&json, result_ptr) {
            Ok(r) => r,
            Err(status) => return status,
        };

        match crate::weibull::weibull_mle(&req.failure_times) {
            Some(result) => {
                let resp = serde_json::json!({
                    "shape": result.shape,
                    "scale": result.scale,
                });
                write_json(result_ptr, &resp)
            }
            None => write_error(result_ptr, ERR_COMPUTE, "Weibull MLE estimation failed"),
        }
    })
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
///
/// # Safety
///
/// `request_json` must be null or point to a NUL-terminated string, and
/// `result_ptr` must be null or valid for writing one pointer. A string written
/// there is owned by the caller and must be released with
/// [`uanalytics_free_string`].
#[cfg(feature = "ffi")]
#[no_mangle]
pub unsafe extern "C" fn uanalytics_detect_changepoints(
    request_json: *const libc::c_char,
    result_ptr: *mut *mut libc::c_char,
) -> i32 {
    ffi_catch(result_ptr, || {
        let json = match unsafe { read_json(request_json) } {
            Ok(j) => j,
            Err(e) => return e,
        };

        let req: PeltRequest = match parse_request(&json, result_ptr) {
            Ok(r) => r,
            Err(status) => return status,
        };

        // The crate implements BIC and a caller-chosen value. Any other name
        // used to fall through to BIC, including the "AIC" and "MBIC" this
        // entry point once documented -- a different penalty than the one
        // asked for, reported as success.
        let penalty = match req.penalty.as_deref() {
            None => crate::detection::Penalty::Bic,
            Some(s) if s.eq_ignore_ascii_case("bic") => crate::detection::Penalty::Bic,
            Some(s) => match s.parse::<f64>() {
                Ok(v) => crate::detection::Penalty::Custom(v),
                Err(_) => {
                    return write_error(
                        result_ptr,
                        ERR_COMPUTE,
                        &format!("unsupported penalty {s:?}: use \"BIC\" or a positive number"),
                    )
                }
            },
        };

        let min_seg = req.min_segment_len.unwrap_or(2);

        let pelt = match crate::detection::Pelt::with_min_segment_len(
            crate::detection::CostFunction::Normal,
            penalty,
            min_seg,
        ) {
            Some(p) => p,
            None => return write_error(result_ptr, ERR_COMPUTE, "Failed to create PELT detector"),
        };

        let result = pelt.detect(&req.data);

        let resp = serde_json::json!({
            "changepoints": result.changepoints,
        });
        write_json(result_ptr, &resp)
    })
}

// ── Correlation Matrix ──────────────────────────────────────

#[cfg(feature = "ffi")]
#[derive(Deserialize)]
struct CorrelationRequest {
    variables: Vec<Vec<f64>>,
}

/// Correlation matrix (Pearson)
///
/// # Safety
///
/// `request_json` must be null or point to a NUL-terminated string, and
/// `result_ptr` must be null or valid for writing one pointer. A string written
/// there is owned by the caller and must be released with
/// [`uanalytics_free_string`].
#[cfg(feature = "ffi")]
#[no_mangle]
pub unsafe extern "C" fn uanalytics_correlation_matrix(
    request_json: *const libc::c_char,
    result_ptr: *mut *mut libc::c_char,
) -> i32 {
    ffi_catch(result_ptr, || {
        let json = match unsafe { read_json(request_json) } {
            Ok(j) => j,
            Err(e) => return e,
        };

        let req: CorrelationRequest = match parse_request(&json, result_ptr) {
            Ok(r) => r,
            Err(status) => return status,
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
            None => write_error(
                result_ptr,
                ERR_COMPUTE,
                "Correlation matrix computation failed",
            ),
        }
    })
}

// ── Simple Regression ───────────────────────────────────────

#[cfg(feature = "ffi")]
#[derive(Deserialize)]
struct RegressionRequest {
    x: Vec<f64>,
    y: Vec<f64>,
}

/// Simple linear regression
///
/// # Safety
///
/// `request_json` must be null or point to a NUL-terminated string, and
/// `result_ptr` must be null or valid for writing one pointer. A string written
/// there is owned by the caller and must be released with
/// [`uanalytics_free_string`].
#[cfg(feature = "ffi")]
#[no_mangle]
pub unsafe extern "C" fn uanalytics_simple_regression(
    request_json: *const libc::c_char,
    result_ptr: *mut *mut libc::c_char,
) -> i32 {
    ffi_catch(result_ptr, || {
        let json = match unsafe { read_json(request_json) } {
            Ok(j) => j,
            Err(e) => return e,
        };

        let req: RegressionRequest = match parse_request(&json, result_ptr) {
            Ok(r) => r,
            Err(status) => return status,
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
            None => write_error(result_ptr, ERR_COMPUTE, "Regression computation failed"),
        }
    })
}

// ── Distribution Fit ────────────────────────────────────────

#[cfg(feature = "ffi")]
#[derive(Deserialize)]
struct FitBestRequest {
    data: Vec<f64>,
}

/// Fit best distribution
///
/// # Safety
///
/// `request_json` must be null or point to a NUL-terminated string, and
/// `result_ptr` must be null or valid for writing one pointer. A string written
/// there is owned by the caller and must be released with
/// [`uanalytics_free_string`].
#[cfg(feature = "ffi")]
#[no_mangle]
pub unsafe extern "C" fn uanalytics_fit_best(
    request_json: *const libc::c_char,
    result_ptr: *mut *mut libc::c_char,
) -> i32 {
    ffi_catch(result_ptr, || {
        let json = match unsafe { read_json(request_json) } {
            Ok(j) => j,
            Err(e) => return e,
        };

        let req: FitBestRequest = match parse_request(&json, result_ptr) {
            Ok(r) => r,
            Err(status) => return status,
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
    })
}

// ── Memory Management ───────────────────────────────────────

/// Free a string allocated by u-analytics FFI functions
///
/// # Safety
///
/// `ptr` must be null or a string returned by this library that has not
/// already been freed.
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
    CString::new(version)
        .expect("version string has no interior NUL")
        .into_raw()
}

// ── Tests ───────────────────────────────────────────────────
//
// These drive the exported symbols exactly as a C caller does -- a C string
// in, a C string out, a status code -- so they pin the wire contract rather
// than the Rust functions behind it.

#[cfg(all(test, feature = "ffi"))]
mod tests {
    use super::*;

    type Entry = unsafe extern "C" fn(*const libc::c_char, *mut *mut libc::c_char) -> i32;

    fn call(entry: Entry, request: &str) -> (i32, serde_json::Value) {
        let request = CString::new(request).expect("test request has no interior NUL");
        let mut out: *mut libc::c_char = std::ptr::null_mut();
        let code = unsafe { entry(request.as_ptr(), &mut out) };
        assert!(!out.is_null(), "every status must come with a JSON body");
        let body = unsafe { CStr::from_ptr(out) }
            .to_str()
            .expect("body is UTF-8")
            .to_owned();
        unsafe { uanalytics_free_string(out) };
        let value = serde_json::from_str(&body).expect("body is JSON");
        (code, value)
    }

    fn subgroups(n: usize, count: usize) -> String {
        let groups: Vec<Vec<f64>> = (0..count)
            .map(|g| {
                (0..n)
                    .map(|i| 10.0 + 0.1 * ((g * 7 + i * 3) % 5) as f64)
                    .collect()
            })
            .collect();
        serde_json::json!({ "subgroups": groups }).to_string()
    }

    // -- status codes ---------------------------------------------------

    #[test]
    fn a_rejected_request_reports_a_failure_status() {
        // An error body under status 0 reaches a caller as a successful
        // result whose fields are all missing. The C# client, for one, only
        // raises on a non-zero status.
        let (code, body) = call(uanalytics_xbar_r_chart, r#"{"subgroups": []}"#);
        assert_eq!(code, -3);
        assert!(body["error"].is_string());
    }

    #[test]
    fn malformed_json_reports_the_parse_status() {
        let (code, body) = call(uanalytics_xbar_r_chart, "{not json");
        assert_eq!(code, -2);
        assert!(body["error"].is_string());
    }

    // -- X-bar-R ----------------------------------------------------------

    #[test]
    fn xbar_r_accepts_every_subgroup_size_the_crate_supports() {
        use crate::spc::{MAX_SUBGROUP_SIZE, MIN_SUBGROUP_SIZE};
        for n in MIN_SUBGROUP_SIZE..=MAX_SUBGROUP_SIZE {
            let (code, body) = call(uanalytics_xbar_r_chart, &subgroups(n, 5));
            assert_eq!(code, 0, "n={n}: {body}");
        }
        let (code, _) = call(
            uanalytics_xbar_r_chart,
            &subgroups(MAX_SUBGROUP_SIZE + 1, 5),
        );
        assert_eq!(code, -3);
    }

    #[test]
    fn xbar_r_limits_are_the_crate_chart_limits() {
        use crate::spc::{ControlChart, XBarRChart};
        let request = subgroups(12, 6);
        let (code, body) = call(uanalytics_xbar_r_chart, &request);
        assert_eq!(code, 0, "{body}");

        let parsed: SpcChartRequest = serde_json::from_str(&request).expect("request parses");
        let mut chart = XBarRChart::new(12).expect("12 is in range");
        for g in &parsed.subgroups {
            chart.add_sample(g).unwrap();
        }
        let x = chart.control_limits().expect("limits");
        let r = chart.r_limits().expect("limits");
        assert_eq!(body["x_bar_ucl"].as_f64(), Some(x.ucl));
        assert_eq!(body["x_bar_lcl"].as_f64(), Some(x.lcl));
        assert_eq!(body["r_ucl"].as_f64(), Some(r.ucl));
        assert_eq!(body["r_lcl"].as_f64(), Some(r.lcl));
        assert_eq!(body["sigma_hat"].as_f64(), chart.sigma_hat());
    }

    #[test]
    fn xbar_r_rejects_ragged_or_non_finite_subgroups() {
        // The crate chart skips such subgroups; the response arrays carry no
        // index, so a skipped subgroup would shift every later value onto the
        // wrong input row. The adapter refuses them instead.
        let (code, _) = call(
            uanalytics_xbar_r_chart,
            r#"{"subgroups": [[1.0, 2.0, 3.0], [1.0, 2.0]]}"#,
        );
        assert_eq!(code, -3);
        let (code, _) = call(
            uanalytics_xbar_r_chart,
            r#"{"subgroups": [[1.0, 2.0, 3.0], [1.0, 1e400, 2.0]]}"#,
        );
        assert_ne!(code, 0);
    }

    // -- P chart ----------------------------------------------------------

    #[test]
    fn p_chart_matches_the_crate_chart() {
        use crate::spc::PChart;
        let (code, body) = call(
            uanalytics_p_chart,
            r#"{"samples": [[100, 3], [120, 5], [80, 2], [100, 4]]}"#,
        );
        assert_eq!(code, 0, "{body}");
        let mut chart = PChart::new();
        for (n, d) in [(100, 3), (120, 5), (80, 2), (100, 4)] {
            chart.add_sample(d, n).unwrap();
        }
        assert_eq!(body["p_bar"].as_f64(), chart.p_bar());
        let ucls: Vec<f64> = chart.points().iter().map(|p| p.ucl).collect();
        assert_eq!(body["ucls"], serde_json::json!(ucls));
    }

    #[test]
    fn p_chart_rejects_an_impossible_sample() {
        // More defectives than inspected, or nothing inspected, has no
        // proportion. Computing one anyway produced p > 1 or NaN.
        let (code, body) = call(uanalytics_p_chart, r#"{"samples": [[100, 3], [10, 12]]}"#);
        assert_eq!(code, -3, "{body}");
        let (code, body) = call(uanalytics_p_chart, r#"{"samples": [[100, 3], [0, 0]]}"#);
        assert_eq!(code, -3, "{body}");
    }

    #[test]
    fn laney_p_chart_reads_pairs_in_the_order_the_p_chart_does() {
        // Both entry points take `[inspected, defective]`; the crate function
        // takes `(defective, sample_size)`. Passing the pairs through
        // unchanged fed every Laney request in reverse.
        use crate::spc::laney_p_chart;
        let pairs = [(100, 3), (120, 9), (80, 2), (100, 7), (110, 4)];
        let request = serde_json::json!({ "samples": pairs });
        let (code, body) = call(uanalytics_laney_p_chart, &request.to_string());
        assert_eq!(code, 0, "{body}");
        let swapped: Vec<(u64, u64)> = pairs.iter().map(|&(n, d)| (d, n)).collect();
        let expected = laney_p_chart(&swapped).expect("valid samples");
        assert_eq!(body["p_bar"].as_f64(), Some(expected.p_bar));
        assert_eq!(body["phi"].as_f64(), Some(expected.phi));

        let (code, body) = call(
            uanalytics_laney_p_chart,
            r#"{"samples": [[100, 3], [10, 12], [90, 2]]}"#,
        );
        assert_eq!(code, -3, "{body}");
    }

    // -- capability -------------------------------------------------------

    #[test]
    fn capability_names_where_its_short_term_sigma_came_from() {
        use crate::spc::{ControlChart, IndividualMRChart};
        let data = [10.1, 9.8, 10.3, 10.0, 9.7, 10.2, 10.1, 9.9];

        let request = serde_json::json!({ "data": data, "usl": 11.0, "lsl": 9.0 });
        let (code, body) = call(uanalytics_process_capability, &request.to_string());
        assert_eq!(code, 0, "{body}");
        assert_eq!(body["sigma_source"], "moving_range");
        let mut imr = IndividualMRChart::new();
        for x in data {
            imr.add_sample(&[x]).unwrap();
        }
        assert_eq!(body["std_dev_within"].as_f64(), imr.sigma_hat());

        let request =
            serde_json::json!({ "data": data, "usl": 11.0, "lsl": 9.0, "sigma_within": 0.25 });
        let (code, body) = call(uanalytics_process_capability, &request.to_string());
        assert_eq!(code, 0, "{body}");
        assert_eq!(body["sigma_source"], "within");
        assert_eq!(body["std_dev_within"].as_f64(), Some(0.25));
    }

    #[test]
    fn capability_rejects_a_non_positive_sigma_within() {
        let request = r#"{"data": [1.0, 2.0, 3.0], "usl": 5.0, "sigma_within": 0.0}"#;
        let (code, _) = call(uanalytics_process_capability, request);
        assert_eq!(code, -3);
    }

    // -- change points ----------------------------------------------------

    #[test]
    fn changepoints_rejects_a_penalty_it_does_not_implement() {
        // "AIC" and "MBIC" used to be documented and then computed as BIC.
        for penalty in ["AIC", "MBIC", "bogus"] {
            let request =
                format!(r#"{{"data": [1.0, 1.0, 1.0, 5.0, 5.0, 5.0], "penalty": "{penalty}"}}"#);
            let (code, body) = call(uanalytics_detect_changepoints, &request);
            assert_eq!(code, -3, "{penalty}: {body}");
        }
        for penalty in ["BIC", "bic", "3.5"] {
            let request =
                format!(r#"{{"data": [1.0, 1.0, 1.0, 5.0, 5.0, 5.0], "penalty": "{penalty}"}}"#);
            let (code, body) = call(uanalytics_detect_changepoints, &request);
            assert_eq!(code, 0, "{penalty}: {body}");
        }
    }
}
