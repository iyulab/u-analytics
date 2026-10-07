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

/// Writes `{"error": message, "code": code, "index": index}` and returns
/// `status`.
///
/// `error` keeps the human-readable text the body has always carried; `code`
/// (stable) and `index` (the offending array position, or `null`) are the
/// shape every transport reports -- see [`crate::wire::WireError`].
///
/// The status is the caller's to choose and is returned as given: the error
/// body is a diagnostic, not a result, so writing it successfully must not
/// turn the call into a success.
#[cfg(feature = "ffi")]
fn write_error(
    result_ptr: *mut *mut libc::c_char,
    status: i32,
    error: impl Into<crate::wire::WireError>,
) -> i32 {
    let e = error.into();
    let err = serde_json::json!({
        "error": e.message, "code": e.code, "index": e.index, "parameter": e.parameter,
    });
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
    serde_json::from_str(json).map_err(|e| {
        write_error(
            result_ptr,
            ERR_PARSE,
            crate::wire::WireError::new(
                crate::wire::code::MALFORMED_INPUT,
                None,
                format!("Invalid JSON: {e}"),
            ),
        )
    })
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
#[serde(deny_unknown_fields)]
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
        let rules = match rules_of(req.rules) {
            Ok(r) => r,
            Err(e) => return write_error(result_ptr, ERR_COMPUTE, &e),
        };
        match crate::wire::xbar_r_dto(req.subgroups, rules) {
            Ok(dto) => write_json(result_ptr, &dto),
            Err(e) => write_error(result_ptr, ERR_COMPUTE, &e),
        }
    })
}

// ── SPC: X-bar/S · I-MR · run rules ─────────────────────────

/// Request for the individual-observations charts and the run-rule engine.
/// `values` is the series in order; `rules` is the optional run-test list,
/// exactly as `uanalytics_xbar_r_chart` takes it.
#[cfg(feature = "ffi")]
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ValuesRequest {
    values: Vec<f64>,
    #[serde(default)]
    rules: Option<serde_json::Value>,
}

/// Request for `uanalytics_run_rules`: a series, one set of limits, and the
/// optional run-test list.
#[cfg(feature = "ffi")]
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RunRulesRequest {
    values: Vec<f64>,
    limits: crate::wire::LimitsInputDto,
    #[serde(default)]
    rules: Option<serde_json::Value>,
}

#[cfg(feature = "ffi")]
fn rules_of(rules: Option<serde_json::Value>) -> Result<crate::spc::RuleSet, String> {
    crate::wire::rules_from_json(rules.map(|r| serde_json::json!({ "rules": r })))
}

/// SPC X-bar/S control chart -- the same request as `uanalytics_xbar_r_chart`
/// and the same response shape with `s_*` limits in place of `r_*`. Returns
/// `sigma_hat` (`S-bar / c4`) for `uanalytics_process_capability`.
///
/// # Safety
///
/// `request_json` must be null or point to a NUL-terminated string, and
/// `result_ptr` must be null or valid for writing one pointer. A string written
/// there is owned by the caller and must be released with
/// [`uanalytics_free_string`].
#[cfg(feature = "ffi")]
#[no_mangle]
pub unsafe extern "C" fn uanalytics_xbar_s_chart(
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
        let rules = match rules_of(req.rules) {
            Ok(r) => r,
            Err(e) => return write_error(result_ptr, ERR_COMPUTE, &e),
        };
        match crate::wire::xbar_s_dto(req.subgroups, rules) {
            Ok(dto) => write_json(result_ptr, &dto),
            Err(e) => write_error(result_ptr, ERR_COMPUTE, &e),
        }
    })
}

/// SPC Individual / Moving-Range chart for a series of single observations.
/// Returns `sigma_hat` (`MR-bar / d2(2)`) -- the short-term sigma that
/// `uanalytics_process_capability` needs for individual data, now that it no
/// longer estimates one itself.
///
/// # Safety
///
/// `request_json` must be null or point to a NUL-terminated string, and
/// `result_ptr` must be null or valid for writing one pointer. A string written
/// there is owned by the caller and must be released with
/// [`uanalytics_free_string`].
#[cfg(feature = "ffi")]
#[no_mangle]
pub unsafe extern "C" fn uanalytics_imr_chart(
    request_json: *const libc::c_char,
    result_ptr: *mut *mut libc::c_char,
) -> i32 {
    ffi_catch(result_ptr, || {
        let json = match unsafe { read_json(request_json) } {
            Ok(j) => j,
            Err(e) => return e,
        };
        let req: ValuesRequest = match parse_request(&json, result_ptr) {
            Ok(r) => r,
            Err(status) => return status,
        };
        let rules = match rules_of(req.rules) {
            Ok(r) => r,
            Err(e) => return write_error(result_ptr, ERR_COMPUTE, &e),
        };
        match crate::wire::imr_dto(req.values, rules) {
            Ok(dto) => write_json(result_ptr, &dto),
            Err(e) => write_error(result_ptr, ERR_COMPUTE, &e),
        }
    })
}

/// Applies the run tests to a series against one set of limits -- the engine
/// the charts use, callable on its own. One point per value, in order.
///
/// # Safety
///
/// `request_json` must be null or point to a NUL-terminated string, and
/// `result_ptr` must be null or valid for writing one pointer. A string written
/// there is owned by the caller and must be released with
/// [`uanalytics_free_string`].
#[cfg(feature = "ffi")]
#[no_mangle]
pub unsafe extern "C" fn uanalytics_run_rules(
    request_json: *const libc::c_char,
    result_ptr: *mut *mut libc::c_char,
) -> i32 {
    ffi_catch(result_ptr, || {
        let json = match unsafe { read_json(request_json) } {
            Ok(j) => j,
            Err(e) => return e,
        };
        let req: RunRulesRequest = match parse_request(&json, result_ptr) {
            Ok(r) => r,
            Err(status) => return status,
        };
        let rules = match rules_of(req.rules) {
            Ok(r) => r,
            Err(e) => return write_error(result_ptr, ERR_COMPUTE, &e),
        };
        match crate::wire::run_rules_dto(req.values, req.limits, rules) {
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
#[serde(deny_unknown_fields)]
struct ProportionSamplesRequest {
    /// `[[defectives, sample_size], ...]`, read as JSON numbers so a count that
    /// is not a whole number is refused with its row rather than by the parser.
    samples: serde_json::Value,
    /// Known centre line (Phase II); see the WASM `p_chart` options.
    #[serde(default)]
    p_bar: Option<f64>,
    /// Known sigma-inflation factor, Laney P' only, together with `p_bar`.
    #[serde(default)]
    phi: Option<f64>,
}

#[cfg(feature = "ffi")]
impl ProportionSamplesRequest {
    fn standard(&self) -> crate::wire::AttributeStandardDto {
        crate::wire::AttributeStandardDto {
            p_bar: self.p_bar,
            u_bar: None,
            phi: self.phi,
        }
    }
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
        match crate::wire::count_pairs(&req.samples, "samples")
            .and_then(|s| crate::wire::p_chart_dto(&s, &req.standard()))
        {
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
        match crate::wire::count_pairs(&req.samples, "samples")
            .and_then(|s| crate::wire::laney_p_dto(&s, &req.standard()))
        {
            Ok(dto) => write_json(result_ptr, &dto),
            Err(e) => write_error(result_ptr, ERR_COMPUTE, &e),
        }
    })
}

// ── SPC: NP, C, U, Laney U', G and T charts ─────────────────

/// `{ "defectives": [...], "sample_size": n }` for the NP chart.
#[cfg(feature = "ffi")]
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct NpChartRequest {
    defectives: serde_json::Value,
    sample_size: serde_json::Value,
}

/// `{ "defects": [...] }` for the C chart.
#[cfg(feature = "ffi")]
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DefectsRequest {
    defects: serde_json::Value,
}

/// `{ "samples": [[defects, units], ...], "u_bar"?, "phi"? }` for the U and
/// Laney U' charts.
#[cfg(feature = "ffi")]
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RateSamplesRequest {
    samples: serde_json::Value,
    #[serde(default)]
    u_bar: Option<f64>,
    /// Laney U' only, together with `u_bar`.
    #[serde(default)]
    phi: Option<f64>,
}

#[cfg(feature = "ffi")]
impl RateSamplesRequest {
    fn standard(&self) -> crate::wire::AttributeStandardDto {
        crate::wire::AttributeStandardDto {
            p_bar: None,
            u_bar: self.u_bar,
            phi: self.phi,
        }
    }
}

/// `{ "gaps": [...] }` (G chart) or `{ "times": [...] }` (T chart).
#[cfg(feature = "ffi")]
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct GapsRequest {
    gaps: Vec<f64>,
}

#[cfg(feature = "ffi")]
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TimesRequest {
    times: Vec<f64>,
}

/// Parses `request_json` as `R` and writes what `compute` returns -- the
/// shape every chart entry point below shares.
#[cfg(feature = "ffi")]
unsafe fn chart_entry<R, T, F>(
    request_json: *const libc::c_char,
    result_ptr: *mut *mut libc::c_char,
    compute: F,
) -> i32
where
    R: serde::de::DeserializeOwned,
    T: Serialize,
    F: FnOnce(R) -> Result<T, crate::wire::WireError>,
{
    let json = match unsafe { read_json(request_json) } {
        Ok(j) => j,
        Err(e) => return e,
    };
    let req: R = match parse_request(&json, result_ptr) {
        Ok(r) => r,
        Err(status) => return status,
    };
    match compute(req) {
        Ok(dto) => write_json(result_ptr, &dto),
        Err(e) => write_error(result_ptr, ERR_COMPUTE, &e),
    }
}

/// SPC NP chart: defectives per subgroup of one fixed `sample_size`.
///
/// # Safety
///
/// As [`uanalytics_p_chart`].
#[cfg(feature = "ffi")]
#[no_mangle]
pub unsafe extern "C" fn uanalytics_np_chart(
    request_json: *const libc::c_char,
    result_ptr: *mut *mut libc::c_char,
) -> i32 {
    ffi_catch(result_ptr, || unsafe {
        chart_entry(request_json, result_ptr, |req: NpChartRequest| {
            let defectives = crate::wire::count_rows(&req.defectives, "defectives")?;
            let size = crate::wire::sample_size_value(&req.sample_size, "sample_size")?;
            crate::wire::np_chart_dto(&defectives, size)
        })
    })
}

/// SPC C chart: defects per inspection unit of one size.
///
/// # Safety
///
/// As [`uanalytics_p_chart`].
#[cfg(feature = "ffi")]
#[no_mangle]
pub unsafe extern "C" fn uanalytics_c_chart(
    request_json: *const libc::c_char,
    result_ptr: *mut *mut libc::c_char,
) -> i32 {
    ffi_catch(result_ptr, || unsafe {
        chart_entry(request_json, result_ptr, |req: DefectsRequest| {
            let defects = crate::wire::count_rows(&req.defects, "defects")?;
            crate::wire::c_chart_dto(&defects)
        })
    })
}

/// SPC U chart: defects per unit when the quantity inspected varies; a known
/// `u_bar` fixes the centre line (Phase II).
///
/// # Safety
///
/// As [`uanalytics_p_chart`].
#[cfg(feature = "ffi")]
#[no_mangle]
pub unsafe extern "C" fn uanalytics_u_chart(
    request_json: *const libc::c_char,
    result_ptr: *mut *mut libc::c_char,
) -> i32 {
    ffi_catch(result_ptr, || unsafe {
        chart_entry(request_json, result_ptr, |req: RateSamplesRequest| {
            let samples = crate::wire::rate_pairs(&req.samples, "samples")?;
            crate::wire::u_chart_dto(&samples, &req.standard())
        })
    })
}

/// SPC Laney U' chart; `u_bar` and `phi` together fix the limits (Phase II).
///
/// # Safety
///
/// As [`uanalytics_p_chart`].
#[cfg(feature = "ffi")]
#[no_mangle]
pub unsafe extern "C" fn uanalytics_laney_u_chart(
    request_json: *const libc::c_char,
    result_ptr: *mut *mut libc::c_char,
) -> i32 {
    ffi_catch(result_ptr, || unsafe {
        chart_entry(request_json, result_ptr, |req: RateSamplesRequest| {
            let samples = crate::wire::rate_pairs(&req.samples, "samples")?;
            crate::wire::laney_u_dto(&samples, &req.standard())
        })
    })
}

/// SPC G chart: inter-event conforming counts (rare events).
///
/// # Safety
///
/// As [`uanalytics_p_chart`].
#[cfg(feature = "ffi")]
#[no_mangle]
pub unsafe extern "C" fn uanalytics_g_chart(
    request_json: *const libc::c_char,
    result_ptr: *mut *mut libc::c_char,
) -> i32 {
    ffi_catch(result_ptr, || unsafe {
        chart_entry(request_json, result_ptr, |req: GapsRequest| {
            crate::wire::g_chart_dto(&req.gaps)
        })
    })
}

/// SPC T chart: inter-event times (rare events).
///
/// # Safety
///
/// As [`uanalytics_p_chart`].
#[cfg(feature = "ffi")]
#[no_mangle]
pub unsafe extern "C" fn uanalytics_t_chart(
    request_json: *const libc::c_char,
    result_ptr: *mut *mut libc::c_char,
) -> i32 {
    ffi_catch(result_ptr, || unsafe {
        chart_entry(request_json, result_ptr, |req: TimesRequest| {
            crate::wire::t_chart_dto(&req.times)
        })
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
#[serde(deny_unknown_fields)]
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

// ── Point processes (event-time trend) ─────────────────────

#[cfg(feature = "ffi")]
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PointProcessRequest {
    times: Vec<f64>,
    observation: serde_json::Value,
}

/// Reads `{ times, observation }`, checking `observation.truncation` by name
/// first so an unknown one is `unknown_option` rather than a parse error.
#[cfg(feature = "ffi")]
fn point_process_request(
    json: &str,
    result_ptr: *mut *mut libc::c_char,
) -> Result<(Vec<f64>, crate::wire::point_process::ObservationDto), i32> {
    use crate::wire::point_process::check_truncation;
    let req: PointProcessRequest = parse_request(json, result_ptr)?;
    let name = req
        .observation
        .get("truncation")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default()
        .to_string();
    if let Err(e) = check_truncation(&name) {
        return Err(write_error(result_ptr, ERR_COMPUTE, e));
    }
    let observation = serde_json::from_value(req.observation).map_err(|e| {
        write_error(
            result_ptr,
            ERR_PARSE,
            crate::wire::WireError::new(
                crate::wire::code::MALFORMED_INPUT,
                None,
                format!("observation: {e}"),
            )
            .about("observation"),
        )
    })?;
    Ok((req.times, observation))
}

macro_rules! point_process_export {
    ($(#[$doc:meta])* $name:ident => $dto:path) => {
        $(#[$doc])*
        ///
        /// Request: `{"times": [t1, ...], "observation": {"truncation": "time", "end": T}}`
        /// or `{"truncation": "failure"}` (observation stopped at the last event).
        ///
        /// # Safety
        ///
        /// `request_json` must be null or point to a NUL-terminated string, and
        /// `result_ptr` must be null or valid for writing one pointer. A string written
        /// there is owned by the caller and must be released with
        /// [`uanalytics_free_string`].
        #[cfg(feature = "ffi")]
        #[no_mangle]
        pub unsafe extern "C" fn $name(
            request_json: *const libc::c_char,
            result_ptr: *mut *mut libc::c_char,
        ) -> i32 {
            ffi_catch(result_ptr, || {
                let json = match unsafe { read_json(request_json) } {
                    Ok(j) => j,
                    Err(e) => return e,
                };
                let (times, observation) = match point_process_request(&json, result_ptr) {
                    Ok(r) => r,
                    Err(status) => return status,
                };
                match $dto(&times, observation) {
                    Ok(dto) => write_json(result_ptr, &dto),
                    Err(e) => write_error(result_ptr, ERR_COMPUTE, e),
                }
            })
        }
    };
}

point_process_export!(
    /// Laplace trend test of event times against a constant rate:
    /// `{statistic, p_value, direction, events_used, df: null}`.
    uanalytics_laplace_trend_test => crate::wire::point_process::laplace_dto
);

point_process_export!(
    /// MIL-HDBK-189 trend test of event times against a constant rate:
    /// `{statistic, p_value, direction, events_used, df}`.
    uanalytics_mil_hdbk_189_test => crate::wire::point_process::mil_hdbk_189_dto
);

point_process_export!(
    /// Power-law process (Crow-AMSAA) fit of event times:
    /// `{beta, beta_unbiased, lambda, intensity_at_end, end, events}`.
    uanalytics_power_law_process_fit => crate::wire::point_process::power_law_dto
);

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

// ── Seasonality ─────────────────────────────────────────────

/// Dominant-period estimation (AutoPeriod)
///
/// Request `{ "data": [...] }` (at least 8 finite values); response
/// `{ "period": 7 | null, "candidates": [...], "n", "acf_threshold",
/// "power_threshold" }` — the same shape as the WASM `estimate_period`.
///
/// # Safety
///
/// `request_json` must be null or point to a NUL-terminated string, and
/// `result_ptr` must be null or valid for writing one pointer. A string written
/// there is owned by the caller and must be released with
/// [`uanalytics_free_string`].
#[cfg(feature = "ffi")]
#[no_mangle]
pub unsafe extern "C" fn uanalytics_estimate_period(
    request_json: *const libc::c_char,
    result_ptr: *mut *mut libc::c_char,
) -> i32 {
    ffi_catch(result_ptr, || {
        let json = match unsafe { read_json(request_json) } {
            Ok(j) => j,
            Err(e) => return e,
        };
        let req: crate::wire::SeasonalityInputDto = match parse_request(&json, result_ptr) {
            Ok(r) => r,
            Err(status) => return status,
        };
        match crate::wire::seasonality_dto(req) {
            Ok(dto) => write_json(result_ptr, &dto),
            Err(e) => write_error(result_ptr, ERR_COMPUTE, &e),
        }
    })
}

/// Spectral residual anomaly scoring (Ren et al. 2019)
///
/// Request `{ "data": [...], "averaging_window"?, "judgement_window"?,
/// "threshold"?, "min_zscore"?, "sensitivity"?, "batch_size"? }`; response
/// `{ "points": [...], "anomalies": [...] }` — the same shape as the WASM
/// `spectral_residual`.
///
/// # Safety
///
/// `request_json` must be null or point to a NUL-terminated string, and
/// `result_ptr` must be null or valid for writing one pointer. A string written
/// there is owned by the caller and must be released with
/// [`uanalytics_free_string`].
#[cfg(feature = "ffi")]
#[no_mangle]
pub unsafe extern "C" fn uanalytics_spectral_residual(
    request_json: *const libc::c_char,
    result_ptr: *mut *mut libc::c_char,
) -> i32 {
    ffi_catch(result_ptr, || {
        let json = match unsafe { read_json(request_json) } {
            Ok(j) => j,
            Err(e) => return e,
        };
        let req: crate::wire::SpectralResidualInputDto = match parse_request(&json, result_ptr) {
            Ok(r) => r,
            Err(status) => return status,
        };
        match crate::wire::spectral_residual_dto(req) {
            Ok(dto) => write_json(result_ptr, &dto),
            Err(e) => write_error(result_ptr, ERR_COMPUTE, &e),
        }
    })
}

// ── Correlation Matrix ──────────────────────────────────────

#[cfg(feature = "ffi")]
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
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
#[serde(deny_unknown_fields)]
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
#[serde(deny_unknown_fields)]
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

    #[test]
    fn point_process_tests_round_trip_and_refuse_with_codes() {
        let req = r#"{"times": [12, 15, 27, 34, 44, 53], "observation": {"truncation": "time", "end": 60}}"#;
        let (code, b) = call(uanalytics_mil_hdbk_189_test, req);
        assert_eq!(code, 0, "{b}");
        assert!((b["statistic"].as_f64().unwrap() - 9.593).abs() < 1e-3);
        assert_eq!(
            (b["df"].as_f64(), b["direction"].as_str()),
            (Some(12.0), Some("increasing"))
        );
        let (_, b) = call(uanalytics_laplace_trend_test, req);
        assert!((b["p_value"].as_f64().unwrap() - 0.906).abs() < 5e-4);
        assert!(b["df"].is_null());
        let (_, b) = call(uanalytics_power_law_process_fit, req);
        assert!((b["beta"].as_f64().unwrap() - 1.25093).abs() < 5e-6);

        let (code, b) = call(
            uanalytics_laplace_trend_test,
            r#"{"times": [1, 2], "observation": {"truncation": "timed", "end": 3}}"#,
        );
        assert_eq!(
            (code, b["code"].as_str(), b["parameter"].as_str()),
            (-3, Some("unknown_option"), Some("observation.truncation"))
        );
        let (code, b) = call(
            uanalytics_laplace_trend_test,
            r#"{"times": [2, 1], "observation": {"truncation": "failure"}}"#,
        );
        assert_eq!(
            (code, b["code"].as_str(), b["index"].as_u64()),
            (-3, Some("events_unordered"), Some(1))
        );
        let (code, b) = call(
            uanalytics_laplace_trend_test,
            r#"{"times": [1], "observation": {"truncation": "time"}}"#,
        );
        assert_eq!(
            (code, b["code"].as_str(), b["parameter"].as_str()),
            (-2, Some("malformed_input"), Some("observation"))
        );
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

    // -- NP, C, U, Laney U', G, T ----------------------------------------

    #[test]
    fn every_attributes_chart_the_wasm_binding_has_is_reachable_from_c() {
        let cases: [(Entry, &str, &str); 6] = [
            (
                uanalytics_np_chart,
                r#"{"defectives": [2, 3, 1, 4], "sample_size": 50}"#,
                "cl",
            ),
            (uanalytics_c_chart, r#"{"defects": [3, 5, 2, 4]}"#, "cl"),
            (
                uanalytics_u_chart,
                r#"{"samples": [[3, 1.5], [5, 2], [2, 1]]}"#,
                "u_bar",
            ),
            (
                uanalytics_laney_u_chart,
                r#"{"samples": [[3, 1.5], [5, 2], [2, 1], [6, 2.5]]}"#,
                "phi",
            ),
            (uanalytics_g_chart, r#"{"gaps": [10, 20, 30, 15]}"#, "g_bar"),
            (
                uanalytics_t_chart,
                r#"{"times": [1.5, 2.0, 0.5, 3.0]}"#,
                "t_bar",
            ),
        ];
        for (entry, request, field) in cases {
            let (code, body) = call(entry, request);
            assert_eq!(code, 0, "{request}: {body}");
            assert!(body[field].is_number(), "{request}: {body}");
            assert!(
                body["points"].as_array().is_some_and(|p| !p.is_empty()),
                "{body}"
            );
        }
    }

    /// The same refusal, code and row as over WebAssembly: both transports
    /// call one core in `wire`.
    #[test]
    fn attributes_refusals_carry_their_row() {
        let (code, body) = call(uanalytics_c_chart, r#"{"defects": [3, 2.5, 4]}"#);
        assert_eq!(code, -3, "{body}");
        assert_eq!(body["code"], "count_not_whole", "{body}");
        assert_eq!(body["index"], 1, "{body}");

        let (code, body) = call(
            uanalytics_u_chart,
            r#"{"samples": [[3, 1.5]], "u_bar": -1}"#,
        );
        assert_eq!(code, -3, "{body}");
        assert_eq!(body["code"], "standard_out_of_range", "{body}");

        let (code, body) = call(uanalytics_t_chart, r#"{"times": [1, 0, 2]}"#);
        assert_eq!(code, -3, "{body}");
        assert_eq!(body["code"], "parameter_out_of_range", "{body}");
        assert_eq!(body["index"], 1, "{body}");
    }

    #[test]
    fn a_known_u_bar_fixes_the_centre_line() {
        let (code, body) = call(
            uanalytics_u_chart,
            r#"{"samples": [[3, 1.5], [5, 2], [2, 1]], "u_bar": 2.0}"#,
        );
        assert_eq!(code, 0, "{body}");
        assert_eq!(body["u_bar"], 2.0, "{body}");
        assert!(body["points"][0]["z"].is_number(), "{body}");
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
        let (code, body) = call(uanalytics_p_chart, r#"{"samples": [[3, 100], [0, 0]]}"#);
        assert_eq!(code, -3, "{body}");
        assert_eq!(body["code"], "sample_size_not_whole", "{body}");
        assert_eq!(body["index"], 1, "{body}");
        // A size that is negative or fractional is the same mistake, and now
        // leaves by the same code instead of arriving as `count_not_whole`.
        for bad in ["-3", "10.5"] {
            let req = format!(r#"{{"samples": [[3, 100], [1, {bad}]]}}"#);
            let (code, body) = call(uanalytics_p_chart, &req);
            assert_eq!(code, -3, "{body}");
            assert_eq!(body["code"], "sample_size_not_whole", "{body}");
            assert_eq!(body["index"], 1, "{body}");
        }
    }

    #[test]
    fn error_bodies_carry_a_code_and_the_row() {
        // A fractional count used to fail in the request parser with no row.
        let (code, body) = call(uanalytics_p_chart, r#"{"samples": [[3, 100], [1.5, 100]]}"#);
        assert_eq!(code, -3, "{body}");
        assert_eq!(body["code"], "count_not_whole", "{body}");
        assert_eq!(body["index"], 1, "{body}");
        assert!(
            body["error"].as_str().unwrap().contains("samples[1]"),
            "{body}"
        );

        let (code, body) = call(uanalytics_laney_p_chart, r#"{"samples": [[3, 100]]}"#);
        assert_eq!(code, -3, "{body}");
        assert_eq!(body["code"], "insufficient_data", "{body}");
        assert!(body["index"].is_null(), "{body}");

        // A request that is not JSON at all keeps its parse status.
        let (code, body) = call(uanalytics_p_chart, "{");
        assert_eq!(code, -2, "{body}");
        assert_eq!(body["code"], "malformed_input", "{body}");
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
        let expected = laney_p_chart(&pairs, None).expect("valid samples");
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
        let pairs: Vec<(u64, u64)> = samples.iter().map(|&[d, n]| (d, n)).collect();
        let wire = crate::wire::p_chart_dto(&pairs, &Default::default()).expect("chart");
        assert_eq!(body, serde_json::to_value(&wire).unwrap());

        let (code, body) = call(
            uanalytics_laney_p_chart,
            &serde_json::json!({ "samples": samples }).to_string(),
        );
        assert_eq!(code, 0, "{body}");
        let wire = crate::wire::laney_p_dto(&pairs, &Default::default()).expect("chart");
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

        // -- seasonality: the sawtooth a well-known library answers with "none" --
        let saw =
            serde_json::json!({ "data": (0..40).map(|i| (i % 7) as f64).collect::<Vec<_>>() });
        let (code, body) = call(uanalytics_estimate_period, &saw.to_string());
        assert_eq!(code, 0, "{body}");
        assert_eq!(body["period"], 7, "{body}");
        let wire = crate::wire::seasonality_dto(serde_json::from_value(saw).unwrap()).unwrap();
        assert_eq!(body, serde_json::to_value(&wire).unwrap());
        let line = serde_json::json!({ "data": (0..40).map(|i| i as f64).collect::<Vec<_>>() });
        let (code, body) = call(uanalytics_estimate_period, &line.to_string());
        assert_eq!(code, 0, "{body}");
        assert!(body["period"].is_null(), "a line has no period: {body}");
        let short = serde_json::json!({ "data": [1.0, 2.0, 3.0] });
        let (code, body) = call(uanalytics_estimate_period, &short.to_string());
        assert_eq!(code, ERR_COMPUTE, "{body}");

        // -- spectral residual: a spike on a flat series --
        let mut spiked = vec![1.0; 40];
        spiked[25] = 9.0;
        let sr = serde_json::json!({ "data": spiked, "sensitivity": 90 });
        let (code, body) = call(uanalytics_spectral_residual, &sr.to_string());
        assert_eq!(code, 0, "{body}");
        assert_eq!(body["anomalies"], serde_json::json!([25]), "{body}");
        assert_eq!(body["points"].as_array().unwrap().len(), 40);
        let wire = crate::wire::spectral_residual_dto(serde_json::from_value(sr).unwrap()).unwrap();
        assert_eq!(body, serde_json::to_value(&wire).unwrap());
        let bad = serde_json::json!({ "data": vec![1.0; 40], "threshold": 0.0 });
        let (code, body) = call(uanalytics_spectral_residual, &bad.to_string());
        assert_eq!(code, ERR_COMPUTE, "{body}");

        // -- X-bar/S, I-MR, run rules --
        let groups: Vec<Vec<f64>> = (0..6)
            .map(|g| {
                (0..12)
                    .map(|i| 10.0 + 0.1 * ((g * 7 + i * 3) % 5) as f64)
                    .collect()
            })
            .collect();
        let (code, body) = call(
            uanalytics_xbar_s_chart,
            &serde_json::json!({ "subgroups": groups }).to_string(),
        );
        assert_eq!(code, 0, "{body}");
        let wire = crate::wire::xbar_s_dto(groups, RuleSet::nelson()).unwrap();
        assert_eq!(body, serde_json::to_value(&wire).unwrap());

        let values: Vec<f64> = (0..20).map(|i| 10.0 + 0.3 * ((i * 7) % 5) as f64).collect();
        let (code, body) = call(
            uanalytics_imr_chart,
            &serde_json::json!({ "values": values }).to_string(),
        );
        assert_eq!(code, 0, "{body}");
        let wire = crate::wire::imr_dto(values.clone(), RuleSet::nelson()).unwrap();
        assert_eq!(body, serde_json::to_value(&wire).unwrap());

        let limits = serde_json::json!({ "ucl": 11.0, "cl": 10.5, "lcl": 10.0 });
        let (code, body) = call(
            uanalytics_run_rules,
            &serde_json::json!({ "values": values, "limits": limits }).to_string(),
        );
        assert_eq!(code, 0, "{body}");
        let wire = crate::wire::run_rules_dto(
            values,
            serde_json::from_value(limits).unwrap(),
            RuleSet::nelson(),
        )
        .unwrap();
        assert_eq!(body, serde_json::to_value(&wire).unwrap());
    }

    #[test]
    fn imr_sigma_hat_feeds_capability_the_number_the_old_default_produced() {
        // The route the changelog points individual-data callers to: the I-MR
        // chart owns the moving-range estimate, capability takes it as
        // `sigma_within`. This is the number `process_capability` used to
        // compute silently on its own.
        use crate::spc::{ControlChart, IndividualMRChart};
        let data = [10.1, 9.8, 10.3, 10.0, 9.7, 10.2, 10.1, 9.9];
        let (code, body) = call(
            uanalytics_imr_chart,
            &serde_json::json!({ "values": data }).to_string(),
        );
        assert_eq!(code, 0, "{body}");
        let sigma_hat = body["sigma_hat"]
            .as_f64()
            .expect("eight values give a sigma");
        let mut imr = IndividualMRChart::new();
        for x in data {
            imr.add_sample(&[x]).unwrap();
        }
        assert_eq!(Some(sigma_hat), imr.sigma_hat());

        let (code, body) = call(
            uanalytics_process_capability,
            &serde_json::json!({ "data": data, "usl": 11.0, "lsl": 9.0, "sigma_within": sigma_hat })
                .to_string(),
        );
        assert_eq!(code, 0, "{body}");
        assert_eq!(body["sigma_source"], "within");
        assert!(body["cp"].is_f64(), "{body}");
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

        // The four entry points the WASM binding does not carry hold the same
        // line: a request type is a contract, not a suggestion.
        let (code, body) = call(
            uanalytics_simple_regression,
            r#"{"x": [1.0, 2.0, 3.0], "y": [2.0, 4.0, 6.1], "weights": [1.0, 1.0, 1.0]}"#,
        );
        assert_eq!(code, -2, "{body}");
        assert!(
            body["error"].as_str().unwrap().contains("weights"),
            "{body}"
        );
    }
}
