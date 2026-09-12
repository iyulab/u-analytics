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
    /// Which run tests to apply, as the WASM binding's optional second
    /// argument takes them: `{ "rules": ["BeyondLimits", ...] }`. Absent means
    /// all eight Nelson tests, which is what this entry point did before the
    /// option existed anywhere.
    #[serde(default)]
    rules: Option<serde_json::Value>,
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
    ffi_catch(result_ptr, || {
        let json = match unsafe { read_json(request_json) } {
            Ok(j) => j,
            Err(e) => return e,
        };
        let req: SpcChartRequest = match parse_request(&json, result_ptr) {
            Ok(r) => r,
            Err(status) => return status,
        };
        // `rules_from_json` reads the WASM binding's options object; this
        // request carries the array at top level, so hand it over wrapped.
        let options = req.rules.map(|r| serde_json::json!({ "rules": r }));
        let rules = match crate::wire::rules_from_json(options) {
            Ok(r) => r,
            Err(e) => return write_error(result_ptr, ERR_COMPUTE, &e),
        };
        match crate::wire::xbar_r_dto(req.subgroups, rules) {
            Ok(dto) => write_json(result_ptr, &dto),
            Err(e) => write_error(result_ptr, ERR_COMPUTE, &e),
        }
    })
}

// ── SPC: P Chart ────────────────────────────────────────────

/// A P-chart-family request. `samples` carries `[defectives, sample_size]`
/// pairs -- the order `PChart::add_sample` takes, and the order the WASM
/// binding has always used. This entry point used to accept the pair reversed,
/// which produced a plausible-looking chart from a caller that read the field
/// names the other way round.
#[cfg(feature = "ffi")]
#[derive(Deserialize)]
struct ProportionSamplesRequest {
    samples: Vec<[u64; 2]>,
}

/// SPC P chart.
///
/// A thin adapter over [`crate::spc::PChart`]. Requests carry
/// `[defectives, sample_size]` pairs -- the same shape and order as the WASM
/// binding, and the order the crate's `add_sample` takes.
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
    ffi_catch(result_ptr, || {
        let json = match unsafe { read_json(request_json) } {
            Ok(j) => j,
            Err(e) => return e,
        };
        let req: ProportionSamplesRequest = match parse_request(&json, result_ptr) {
            Ok(r) => r,
            Err(status) => return status,
        };
        match crate::wire::p_chart_dto(&req.samples) {
            Ok(dto) => write_json(result_ptr, &dto),
            Err(e) => write_error(result_ptr, ERR_COMPUTE, &e),
        }
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

        let req: ProportionSamplesRequest = match parse_request(&json, result_ptr) {
            Ok(r) => r,
            Err(status) => return status,
        };
        match crate::wire::laney_p_dto(&req.samples) {
            Ok(dto) => write_json(result_ptr, &dto),
            Err(e) => write_error(result_ptr, ERR_COMPUTE, &e),
        }
    })
}

// ── Process Capability ──────────────────────────────────────

/// Process capability analysis (Cp, Cpk, Pp, Ppk, Cpm).
///
/// The short-term indices need a within sigma, and a flat `data` vector cannot
/// supply one -- the subgroup structure is gone. `sigma_within` carries it
/// (the `sigma_hat` an X-bar/R, X-bar/S or I-MR chart returns). Without it the
/// short-term indices are `null` and `sigma_source` is `"overall"`, exactly as
/// over WASM. This entry point used to estimate a within sigma from the moving
/// range instead, which is right for individual observations and wrong for
/// data flattened out of subgroups -- and it could not tell the two apart. A
/// caller with individual observations gets the same number by running
/// `imr_chart` first and passing its `sigma_hat`.
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
    ffi_catch(result_ptr, || {
        let json = match unsafe { read_json(request_json) } {
            Ok(j) => j,
            Err(e) => return e,
        };
        let req: crate::wire::CapabilityInputDto = match parse_request(&json, result_ptr) {
            Ok(r) => r,
            Err(status) => return status,
        };
        match crate::wire::capability_dto(req) {
            Ok(dto) => write_json(result_ptr, &dto),
            Err(e) => write_error(result_ptr, ERR_COMPUTE, &e),
        }
    })
}

// ── Percentile Capability ───────────────────────────────────

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
        let req: crate::wire::PercentileCapabilityInputDto = match parse_request(&json, result_ptr)
        {
            Ok(r) => r,
            Err(status) => return status,
        };
        match crate::wire::percentile_capability_dto(req) {
            Ok(dto) => write_json(result_ptr, &dto),
            Err(e) => write_error(result_ptr, ERR_COMPUTE, &e),
        }
    })
}

// ── MSA: Gage R&R X-bar/R ───────────────────────────────────

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
        let req: crate::wire::GageRRInputDto = match parse_request(&json, result_ptr) {
            Ok(r) => r,
            Err(status) => return status,
        };
        match crate::wire::gage_rr_xbar_r_dto(req) {
            Ok(dto) => write_json(result_ptr, &dto),
            Err(e) => write_error(result_ptr, ERR_COMPUTE, &e),
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
        let req: crate::wire::GageRRInputDto = match parse_request(&json, result_ptr) {
            Ok(r) => r,
            Err(status) => return status,
        };
        match crate::wire::gage_rr_anova_dto(req) {
            Ok(dto) => write_json(result_ptr, &dto),
            Err(e) => write_error(result_ptr, ERR_COMPUTE, &e),
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
        let req: crate::wire::PeltInputDto = match parse_request(&json, result_ptr) {
            Ok(r) => r,
            Err(status) => return status,
        };
        match crate::wire::pelt_dto(req) {
            Ok(dto) => write_json(result_ptr, &dto),
            Err(e) => write_error(result_ptr, ERR_COMPUTE, &e),
        }
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
        // Field names are the wire contract's -- the ones the WASM binding
        // has always used -- not this entry point's former `x_bar_*`.
        assert_eq!(body["xbar_ucl"].as_f64(), Some(x.ucl));
        assert_eq!(body["xbar_lcl"].as_f64(), Some(x.lcl));
        assert_eq!(body["r_ucl"].as_f64(), Some(r.ucl));
        assert_eq!(body["r_lcl"].as_f64(), Some(r.lcl));
        assert_eq!(body["sigma_hat"].as_f64(), chart.sigma_hat());
        assert!(
            body.get("x_bar_ucl").is_none(),
            "the old spelling must be gone"
        );
        // The per-point payload that used to exist only over WASM.
        assert_eq!(body["xbar_points"].as_array().map(Vec::len), Some(6));
        assert!(body["xbar_points"][0]["violations"].is_array());
        assert!(body["in_control"].is_boolean());
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
        // `[defectives, sample_size]` -- the crate's order.
        let (code, body) = call(
            uanalytics_p_chart,
            r#"{"samples": [[3, 100], [5, 120], [2, 80], [4, 100]]}"#,
        );
        assert_eq!(code, 0, "{body}");
        let mut chart = PChart::new();
        for (d, n) in [(3, 100), (5, 120), (2, 80), (4, 100)] {
            chart.add_sample(d, n).unwrap();
        }
        assert_eq!(body["p_bar"].as_f64(), chart.p_bar());
        let ucls: Vec<f64> = chart.points().iter().map(|p| p.ucl).collect();
        let got: Vec<f64> = body["points"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| p["ucl"].as_f64().unwrap())
            .collect();
        assert_eq!(got, ucls);
    }

    #[test]
    fn p_chart_pair_order_is_the_crate_order_and_the_reverse_is_refused() {
        // This entry point used to take `[inspected, defective]`. A caller
        // sending that order now gets an error naming the row, not a chart
        // built from proportions above one.
        let (code, body) = call(uanalytics_p_chart, r#"{"samples": [[100, 3], [120, 5]]}"#);
        assert_eq!(code, -3, "{body}");
        assert!(
            body["error"].as_str().unwrap().contains("samples[0]"),
            "{body}"
        );

        let (code, body) = call(uanalytics_p_chart, r#"{"samples": [[3, 100], [5, 120]]}"#);
        assert_eq!(code, 0, "{body}");
        let p0 = body["points"][0]["value"].as_f64().unwrap();
        assert!((p0 - 0.03).abs() < 1e-12, "3 of 100 is 0.03, got {p0}");
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
        // Both entry points now take `[defectives, sample_size]`, which is
        // also what the crate function takes -- no swap anywhere.
        use crate::spc::laney_p_chart;
        let pairs = [(3, 100), (9, 120), (2, 80), (7, 100), (4, 110)];
        let request = serde_json::json!({ "samples": pairs });
        let (code, body) = call(uanalytics_laney_p_chart, &request.to_string());
        assert_eq!(code, 0, "{body}");
        let expected = laney_p_chart(&pairs).expect("valid samples");
        assert_eq!(body["p_bar"].as_f64(), Some(expected.p_bar));
        assert_eq!(body["phi"].as_f64(), Some(expected.phi));

        let (code, body) = call(
            uanalytics_laney_p_chart,
            r#"{"samples": [[3, 100], [12, 10], [2, 90]]}"#,
        );
        assert_eq!(code, -3, "{body}");
    }

    // -- one contract, two transports ---------------------------------------

    /// What this module returns over the C boundary must be byte-for-byte the
    /// JSON the WASM binding returns for the same logical request. Both are
    /// `serde_json` renderings of the same `crate::wire` value, so this pins
    /// that neither side has grown a private shape again.
    #[test]
    fn ffi_bodies_are_the_wire_contract_the_wasm_binding_emits() {
        use crate::spc::RuleSet;

        let groups: Vec<Vec<f64>> = (0..6)
            .map(|g| {
                (0..5)
                    .map(|i| 10.0 + 0.1 * ((g * 7 + i * 3) % 5) as f64)
                    .collect()
            })
            .collect();
        let (code, body) = call(
            uanalytics_xbar_r_chart,
            &serde_json::json!({ "subgroups": groups }).to_string(),
        );
        assert_eq!(code, 0, "{body}");
        let wire = crate::wire::xbar_r_dto(groups, RuleSet::nelson()).expect("chart");
        assert_eq!(body, serde_json::to_value(&wire).unwrap());

        let samples = [[3_u64, 100], [5, 120], [2, 80], [4, 100]];
        let (code, body) = call(
            uanalytics_p_chart,
            &serde_json::json!({ "samples": samples }).to_string(),
        );
        assert_eq!(code, 0, "{body}");
        let wire = crate::wire::p_chart_dto(&samples).expect("chart");
        assert_eq!(body, serde_json::to_value(&wire).unwrap());

        let (code, body) = call(
            uanalytics_laney_p_chart,
            &serde_json::json!({ "samples": samples }).to_string(),
        );
        assert_eq!(code, 0, "{body}");
        let wire = crate::wire::laney_p_dto(&samples).expect("chart");
        assert_eq!(body, serde_json::to_value(&wire).unwrap());

        // -- capability, percentile, gage R&R (both methods), changepoints --
        let cap = serde_json::json!({
            "data": [10.1, 9.8, 10.3, 10.0, 9.7, 10.2, 10.1, 9.9],
            "usl": 11.0, "lsl": 9.0, "sigma_within": 0.2, "target": 10.0
        });
        let (code, body) = call(uanalytics_process_capability, &cap.to_string());
        assert_eq!(code, 0, "{body}");
        let wire = crate::wire::capability_dto(serde_json::from_value(cap).unwrap()).unwrap();
        assert_eq!(body, serde_json::to_value(&wire).unwrap());

        let pct = serde_json::json!({
            "data": (0..40).map(|i| 10.0 + (i as f64 * 0.37).sin()).collect::<Vec<_>>(),
            "usl": 11.5, "lsl": 8.5
        });
        let (code, body) = call(uanalytics_percentile_capability, &pct.to_string());
        assert_eq!(code, 0, "{body}");
        let wire =
            crate::wire::percentile_capability_dto(serde_json::from_value(pct).unwrap()).unwrap();
        assert_eq!(body, serde_json::to_value(&wire).unwrap());

        let grr = serde_json::json!({
            "measurements": (0..5).map(|p| (0..3).map(|o| (0..3).map(|t|
                10.0 + p as f64 + 0.1 * o as f64 + 0.03 * ((p * 7 + o * 3 + t * 5) % 4) as f64
            ).collect::<Vec<_>>()).collect::<Vec<_>>()).collect::<Vec<_>>(),
            "tolerance": 6.0
        });
        let (code, body) = call(uanalytics_gage_rr_xbar_r, &grr.to_string());
        assert_eq!(code, 0, "{body}");
        let wire =
            crate::wire::gage_rr_xbar_r_dto(serde_json::from_value(grr.clone()).unwrap()).unwrap();
        assert_eq!(body, serde_json::to_value(&wire).unwrap());
        let (code, body) = call(uanalytics_gage_rr_anova, &grr.to_string());
        assert_eq!(code, 0, "{body}");
        let wire = crate::wire::gage_rr_anova_dto(serde_json::from_value(grr).unwrap()).unwrap();
        assert_eq!(body, serde_json::to_value(&wire).unwrap());

        let pelt = serde_json::json!({ "data": [1.0, 1.1, 0.9, 1.0, 5.0, 5.1, 4.9, 5.0] });
        let (code, body) = call(uanalytics_detect_changepoints, &pelt.to_string());
        assert_eq!(code, 0, "{body}");
        let wire = crate::wire::pelt_dto(serde_json::from_value(pelt).unwrap()).unwrap();
        assert_eq!(body, serde_json::to_value(&wire).unwrap());
    }

    #[test]
    fn xbar_r_honours_the_rules_option_like_the_wasm_binding() {
        // A steady upward drift trips Nelson's trend test under the default
        // rule set. `rules: []` keeps only the limit check -- as the WASM
        // binding documents it -- so no pattern test may fire.
        let groups: Vec<Vec<f64>> = (0..12)
            .map(|g| {
                let base = 10.0 + 0.05 * g as f64;
                vec![base - 0.3, base + 0.2, base - 0.1, base + 0.4]
            })
            .collect();
        let with_nelson = serde_json::json!({ "subgroups": groups });
        let limits_only = serde_json::json!({ "subgroups": groups, "rules": [] });

        let (code, body) = call(uanalytics_xbar_r_chart, &with_nelson.to_string());
        assert_eq!(code, 0, "{body}");
        let fired: Vec<String> = body["xbar_points"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|p| p["violations"].as_array().unwrap().clone())
            .map(|v| v.as_str().unwrap().to_owned())
            .collect();
        assert!(
            fired.iter().any(|v| v == "SixTrend"),
            "drift should trip the trend test: {fired:?}"
        );

        let (code, body) = call(uanalytics_xbar_r_chart, &limits_only.to_string());
        assert_eq!(code, 0, "{body}");
        for p in body["xbar_points"].as_array().unwrap() {
            for v in p["violations"].as_array().unwrap() {
                assert_eq!(v, "BeyondLimits", "only the limit check may fire: {p}");
            }
        }

        let (code, body) = call(
            uanalytics_xbar_r_chart,
            &serde_json::json!({ "subgroups": [[1.0, 2.0]], "rules": ["NoSuchRule"] }).to_string(),
        );
        assert_eq!(code, -3, "{body}");
    }

    // -- capability -------------------------------------------------------

    #[test]
    fn capability_reports_no_short_term_indices_without_a_within_sigma() {
        use crate::spc::{ControlChart, IndividualMRChart};
        let data = [10.1, 9.8, 10.3, 10.0, 9.7, 10.2, 10.1, 9.9];

        // No `sigma_within`: the long-term indices only, and the source says
        // so. This entry point used to estimate one from the moving range and
        // report it as `"moving_range"`; the WASM binding never did.
        let request = serde_json::json!({ "data": data, "usl": 11.0, "lsl": 9.0 });
        let (code, body) = call(uanalytics_process_capability, &request.to_string());
        assert_eq!(code, 0, "{body}");
        assert_eq!(body["sigma_source"], "overall");
        assert!(body["cp"].is_null() && body["cpk"].is_null(), "{body}");
        assert!(body["std_dev_within"].is_null(), "{body}");
        assert!(body["pp"].is_f64() && body["ppk"].is_f64(), "{body}");

        // The moving-range estimate is still one call away -- through the chart
        // that owns it, so the assumption is the caller's and visible.
        let mut imr = IndividualMRChart::new();
        for x in data {
            imr.add_sample(&[x]).unwrap();
        }
        let sigma_hat = imr.sigma_hat().expect("eight values");
        let request = serde_json::json!({
            "data": data, "usl": 11.0, "lsl": 9.0, "sigma_within": sigma_hat
        });
        let (code, body) = call(uanalytics_process_capability, &request.to_string());
        assert_eq!(code, 0, "{body}");
        assert_eq!(body["sigma_source"], "within");
        assert_eq!(body["std_dev_within"].as_f64(), Some(sigma_hat));
        assert!(body["cp"].is_f64(), "{body}");
    }

    #[test]
    fn capability_rejects_a_non_positive_sigma_within() {
        let request = r#"{"data": [1.0, 2.0, 3.0], "usl": 5.0, "sigma_within": 0.0}"#;
        let (code, _) = call(uanalytics_process_capability, request);
        assert_eq!(code, -3);
    }

    // -- change points ----------------------------------------------------

    #[test]
    fn changepoints_takes_the_wire_penalty_and_cost_like_the_wasm_binding() {
        // "AIC" and "MBIC" used to be documented and then computed as BIC.
        // The contract is now the WASM one: `"bic"` or a JSON number, and an
        // optional `cost` of `"l2"` (default) or `"normal"`.
        let data = [1.0, 1.0, 1.0, 5.0, 5.0, 5.0];
        for penalty in [
            serde_json::json!("AIC"),
            serde_json::json!("MBIC"),
            serde_json::json!("3.5"),
        ] {
            let request = serde_json::json!({ "data": data, "penalty": penalty });
            let (code, body) = call(uanalytics_detect_changepoints, &request.to_string());
            assert_eq!(code, -3, "{penalty}: {body}");
        }
        for penalty in [serde_json::json!("bic"), serde_json::json!(3.5)] {
            let request = serde_json::json!({ "data": data, "penalty": penalty });
            let (code, body) = call(uanalytics_detect_changepoints, &request.to_string());
            assert_eq!(code, 0, "{penalty}: {body}");
            assert!(body["n_segments"].is_u64(), "{body}");
        }
        // This entry point used to default to the normal (mean+variance) cost
        // while WASM defaulted to L2 -- same input, different changepoints.
        // Both now default to L2, and `normal` is opt-in on both.
        let (code, body) = call(
            uanalytics_detect_changepoints,
            &serde_json::json!({ "data": data, "cost": "normal" }).to_string(),
        );
        assert_eq!(code, 0, "{body}");
        let (code, body) = call(
            uanalytics_detect_changepoints,
            &serde_json::json!({ "data": data, "cost": "l1" }).to_string(),
        );
        assert_eq!(code, -3, "{body}");
    }

    #[test]
    fn ffi_rejects_unknown_request_fields_like_the_wasm_binding_does() {
        // Over WASM a misspelt option has always been an error. Over the FFI it
        // was silently ignored -- `sigmaWithin` produced a long-term-only
        // answer with no hint that the caller's sigma never arrived.
        let (code, body) = call(
            uanalytics_process_capability,
            r#"{"data": [1.0, 2.0, 3.0], "usl": 5.0, "sigmaWithin": 0.5}"#,
        );
        assert_eq!(code, -2, "{body}");
        assert!(
            body["error"].as_str().unwrap().contains("sigmaWithin"),
            "{body}"
        );
    }
}
