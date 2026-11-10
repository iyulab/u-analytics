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

use serde::Serialize;
use wasm_bindgen::prelude::*;

// The shapes below are the wire contract, shared with the C FFI so the two
// transports cannot drift apart again. See `crate::wire`.
use crate::wire::{
    acf_dto, c_chart_dto, capability_dto, correlation_matrix_dto, count_pairs, count_rows,
    cusum_dto, default_correlation_method, ewma_dto, failure_site, fit_best_dto, g_chart_dto,
    gage_rr_anova_dto, gage_rr_xbar_r_dto, goodness_of_fit_dto,
    hypothesis::{
        adf_dto, anderson_darling_dto, chi_squared_gof_dto, chi_squared_independence_dto,
        fisher_exact_dto, groups_test_dto, jarque_bera_dto, kpss_dto, mann_kendall_dto,
        mann_whitney_dto, one_sample_t_dto, one_way_anova_dto, p_adjust_dto, paired_t_dto,
        shapiro_wilk_dto, tost_one_sample_dto, tost_paired_dto, tost_two_sample_dto,
        two_sample_t_dto, wilcoxon_dto, AdfInputDto, KpssInputDto,
    },
    imr_dto, kde_dto, laney_p_dto, laney_u_dto, multi_pelt_dto, np_chart_dto, p_chart_dto,
    pacf_dto, pelt_dto, percentile_capability_dto,
    point_process::{
        check_truncation, laplace_dto, mil_hdbk_189_dto, power_law_dto, ObservationDto,
    },
    rate_pairs,
    reliability::{
        boxcox_capability_dto, ppm_to_sigma_value, sigma_to_ppm_value, weibull_mle_dto,
        weibull_mrr_dto, weibull_reliability_dto, BoxcoxCapabilityInputDto,
        WeibullReliabilityInputDto,
    },
    rules_from_json, run_rules_dto, sample_size_value, simple_regression_dto, t_chart_dto,
    u_chart_dto, xbar_r_dto, xbar_s_dto, AttributeStandardDto, AutocorrelationInputDto,
    CapabilityInputDto, CorrelationInputDto, CusumInputDto, EwmaInputDto, GageRRInputDto,
    GoodnessOfFitInputDto, KdeInputDto, LimitsInputDto, MultiPeltInputDto,
    MultipleRegressionInputDto, PeltInputDto, PercentileCapabilityInputDto,
    PredictionErrorsInputDto, RegressionInputDto, SeasonalityInputDto, SmoothingInputDto,
    SpectralResidualInputDto, StlInputDto, WireError,
};
use crate::wire::{
    tolerance::{tolerance_factor_value, tolerance_interval_dto},
    type1_gage::type1_gage_study_dto,
    Whole,
};

// ---------------------------------------------------------------------------
// Serializable DTO types
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Every refusal `code` the functions of this package throw, with the fields
/// that can come with it: `{ refusals: [{ code, fields }], violations: [] }`.
///
/// It is built from the table the refusals come from, so a caller that
/// translates codes can check its list against the package's.
#[wasm_bindgen(unchecked_return_type = "Codes")]
pub fn codes() -> Result<JsValue, JsValue> {
    serde_wasm_bindgen::to_value(&crate::wire::code_catalog()).map_err(|e| {
        js_err(WireError::new(
            crate::wire::code::INTERNAL,
            None,
            e.to_string(),
        ))
    })
}

/// Every refusal crosses into JavaScript as an `Error` carrying `code` (a
/// stable reason), `index` (the offending array position, or `null`) and
/// `parameter` (the option the refusal is about, or `null`), with
/// the human-readable text as its `message` -- the shape Node gives its own
/// errors, so `err.message` still reads as before while a program can branch on
/// `err.code` and point at `err.index`.
fn js_err(error: impl Into<WireError>) -> JsValue {
    let error = error.into();
    let js = js_sys::Error::new(&error.message);
    let index = error
        .index
        .map_or(JsValue::NULL, |i| JsValue::from_f64(i as f64));
    // `Reflect::set` on a freshly created ordinary object cannot fail.
    let _ = js_sys::Reflect::set(
        &js,
        &JsValue::from_str("code"),
        &JsValue::from_str(error.code),
    );
    let _ = js_sys::Reflect::set(&js, &JsValue::from_str("index"), &index);
    let parameter = error
        .parameter
        .as_deref()
        .map_or(JsValue::NULL, JsValue::from_str);
    let _ = js_sys::Reflect::set(&js, &JsValue::from_str("parameter"), &parameter);
    for (key, value) in &error.details {
        let value = match value {
            crate::wire::Detail::Num(n) => JsValue::from_f64(*n),
            crate::wire::Detail::Text(s) => JsValue::from_str(s),
            crate::wire::Detail::List(v) => v
                .iter()
                .map(|s| JsValue::from_str(s))
                .collect::<js_sys::Array>()
                .into(),
            crate::wire::Detail::Null => JsValue::NULL,
        };
        let _ = js_sys::Reflect::set(&js, &JsValue::from_str(key), &value);
    }
    js.into()
}

impl From<serde_wasm_bindgen::Error> for WireError {
    /// Only results are serialized here, so a failure is the package's.
    fn from(e: serde_wasm_bindgen::Error) -> Self {
        WireError::new(crate::wire::code::INTERNAL, None, e.to_string())
    }
}

/// Serializes a response. An absent value (`None`) crosses as `null`, not as a
/// missing key: the FFI renders the same wire value with `serde_json`, which
/// writes `null`, and this crate documents `null` -- serde-wasm-bindgen's
/// default of omitting the key made the two transports disagree on every
/// optional field.
fn to_js<T: Serialize>(val: &T) -> Result<JsValue, JsValue> {
    #[cfg(test)]
    crate::finite_out::assert_finite(val);
    val.serialize(&serde_wasm_bindgen::Serializer::new().serialize_missing_as_null(true))
        .map_err(js_err)
}

/// A NaN or ±Infinity found in a JS argument, and where it sits.
///
/// JSON has no non-finite numbers, so on the way to the wire schema
/// `serde_json` turns one into `null` and the caller would be told a value has
/// the wrong type. [`find_non_finite`] looks before that happens, so the
/// refusal names the real reason and the place.
struct NonFinite {
    /// The argument's name, then `.key` and `[i]` steps down to the array or
    /// field that holds the number.
    parameter: String,
    /// The number's position, when it is an array element.
    index: Option<usize>,
    value: f64,
}

impl NonFinite {
    fn message(&self) -> String {
        let at = match self.index {
            Some(i) => format!("{}[{i}]", self.parameter),
            None => self.parameter.clone(),
        };
        let got = if self.value.is_nan() {
            "NaN"
        } else if self.value > 0.0 {
            "Infinity"
        } else {
            "-Infinity"
        };
        format!("{at}: expected a finite number, got {got}")
    }
}

/// The first NaN or ±Infinity in `value`, searching arrays, iterables and
/// plain objects. `allow_nan` lets NaN through for an input that reads it as a
/// missing value; it then arrives as `null`.
fn find_non_finite(value: &JsValue, parameter: &str, allow_nan: bool) -> Option<NonFinite> {
    let refused = |n: f64| !n.is_finite() && !(allow_nan && n.is_nan());
    let found = |index: Option<usize>, value: f64| NonFinite {
        parameter: parameter.to_string(),
        index,
        value,
    };
    if let Some(n) = value.as_f64() {
        return refused(n).then(|| found(None, n));
    }
    if !value.is_object() {
        return None;
    }
    if let Ok(Some(items)) = js_sys::try_iter(value) {
        for (i, item) in items.enumerate() {
            // An iterator that throws is left for serde to report.
            let item = item.ok()?;
            match item.as_f64() {
                Some(n) if refused(n) => return Some(found(Some(i), n)),
                Some(_) => {}
                None => {
                    let inner = find_non_finite(&item, &format!("{parameter}[{i}]"), allow_nan);
                    if inner.is_some() {
                        return inner;
                    }
                }
            }
        }
        return None;
    }
    let object: &js_sys::Object = wasm_bindgen::JsCast::unchecked_ref(value);
    for entry in js_sys::Object::entries(object).iter() {
        let pair: js_sys::Array = wasm_bindgen::JsCast::unchecked_into(entry);
        let key = pair.get(0).as_string().unwrap_or_default();
        let inner = find_non_finite(&pair.get(1), &format!("{parameter}.{key}"), allow_nan);
        if inner.is_some() {
            return inner;
        }
    }
    None
}

/// Deserialize a native JS value, rejecting JSON strings with an actionable
/// message and prefixing the offending parameter name to any serde error.
fn from_js<T: serde::de::DeserializeOwned>(value: JsValue, param: &str) -> Result<T, JsValue> {
    if value.as_string().is_some() {
        return Err(js_err(
            WireError::new(
                crate::wire::code::MALFORMED_INPUT,
                None,
                format!(
                    "{param}: expected a native JS object/array, got a string — \
                     pass the value directly, not JSON.stringify(...)"
                ),
            )
            .about(param.to_owned()),
        ));
    }
    if let Some(found) = find_non_finite(&value, param, false) {
        return Err(js_err(
            WireError::new(
                crate::wire::code::VALUE_NOT_FINITE,
                found.index,
                found.message(),
            )
            .about(found.parameter),
        ));
    }
    // serde-wasm-bindgen reads only a struct's declared fields from a JS
    // object, so `deny_unknown_fields` never sees extra keys. Round-trip
    // through serde_json::Value so the strict wire schema is enforced.
    let json: serde_json::Value = serde_wasm_bindgen::from_value(value).map_err(|e| {
        js_err(WireError::new(
            crate::wire::code::MALFORMED_INPUT,
            None,
            format!("{param}: {e}"),
        ))
    })?;
    from_json(json, param).map_err(js_err)
}

/// The half of [`from_js`] that enforces the wire schema, split out so it can
/// be exercised without a `JsValue` -- which cannot be constructed off
/// `wasm32`. Tests calling this walk the same deserialization path a JS caller
/// does, rather than a parallel one that could drift from it.
fn from_json<T: serde::de::DeserializeOwned>(
    json: serde_json::Value,
    param: &str,
) -> Result<T, WireError> {
    serde_path_to_error::deserialize(json).map_err(|e| {
        let (parameter, index) = failure_site(param, e.path());
        WireError::new(
            crate::wire::code::MALFORMED_INPUT,
            index,
            format!("{param}: {}: {}", e.path(), e.inner()),
        )
        .about(parameter)
    })
}

/// One element of a JS number array, as found.
#[derive(Debug, Clone, PartialEq)]
enum Element {
    Number(f64),
    /// Anything else, by its JS type name (`"null"`, `"string"`, …).
    Other(String),
}

/// The values of a number array, refusing the first element that is not a
/// finite number at its index. Kept apart from the `JsValue` walk so it runs
/// off `wasm32`.
fn numbers_from(
    param: &'static str,
    elements: impl IntoIterator<Item = Element>,
) -> Result<Vec<f64>, WireError> {
    let mut out = Vec::new();
    for (i, element) in elements.into_iter().enumerate() {
        match element {
            Element::Number(x) if x.is_finite() => out.push(x),
            Element::Number(x) => {
                return Err(WireError::new(
                    crate::wire::code::VALUE_NOT_FINITE,
                    Some(i),
                    format!("{param}[{i}]: expected a finite number, got {x}"),
                )
                .about(param))
            }
            Element::Other(kind) => {
                return Err(WireError::new(
                    crate::wire::code::MALFORMED_INPUT,
                    Some(i),
                    format!("{param}[{i}]: expected a number, got {kind}"),
                )
                .about(param))
            }
        }
    }
    Ok(out)
}

/// Reads a `number[]` or `Float64Array` argument as sent.
///
/// A `&[f64]` parameter would let the generated glue copy a plain array into a
/// typed array first, so `null` would arrive as 0 and a string as NaN.
fn read_numbers(value: &JsValue, param: &'static str) -> Result<Vec<f64>, WireError> {
    use wasm_bindgen::JsCast;
    if let Some(typed) = value.dyn_ref::<js_sys::Float64Array>() {
        return numbers_from(param, typed.to_vec().into_iter().map(Element::Number));
    }
    if !js_sys::Array::is_array(value) {
        return Err(WireError::new(
            crate::wire::code::MALFORMED_INPUT,
            None,
            format!("{param}: expected an array of numbers or a Float64Array"),
        )
        .about(param));
    }
    let array: &js_sys::Array = value.unchecked_ref();
    numbers_from(
        param,
        array.iter().map(|item| match item.as_f64() {
            Some(x) => Element::Number(x),
            None if item.is_null() => Element::Other("null".to_string()),
            None => Element::Other(item.js_typeof().as_string().unwrap_or_default()),
        }),
    )
}

/// Places a row reader's refusal in its row, as the crate's error contract
/// reads: `parameter` is the path to the array (`data[1]`) and `index` the
/// position in it -- the shape `find_non_finite` already gives a NaN there.
fn at_row(e: WireError, param: &'static str, row: usize) -> WireError {
    let path = format!("{param}[{row}]");
    let message = e.message.replacen(param, &path, 1);
    WireError { message, ..e }.about(path)
}

/// Reads a `number[][]` argument as sent -- each row as [`read_numbers`] does
/// -- so `null` in a row is refused at that row rather than by the parser.
fn read_number_rows(value: &JsValue, param: &'static str) -> Result<Vec<Vec<f64>>, WireError> {
    use wasm_bindgen::JsCast;
    if !js_sys::Array::is_array(value) {
        return Err(WireError::new(
            crate::wire::code::MALFORMED_INPUT,
            None,
            format!("{param}: expected an array of number arrays"),
        )
        .about(param));
    }
    let array: &js_sys::Array = value.unchecked_ref();
    array
        .iter()
        .enumerate()
        .map(|(row, item)| read_numbers(&item, param).map_err(|e| at_row(e, param, row)))
        .collect()
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
/// `{ rules?: RuleSetName | RunRule[] }` -- which run tests to apply: a named
/// set (`"shewhart"` -- the 3-sigma test alone; `"western_electric"`;
/// `"nelson"` -- all eight of Nelson's), or the tests spelt out by the names
/// that appear in each point's `violations`:
/// `BeyondLimits`, `NineOneSide`, `SixTrend`, `FourteenAlternating`,
/// `TwoOfThreeBeyond2Sigma`, `FourOfFiveBeyond1Sigma`, `FifteenWithin1Sigma`,
/// `EightBeyond1Sigma`, `EightOneSide` (Western Electric's eight-point run;
/// Nelson's is `NineOneSide`).
///
/// Omitted, `undefined`, `null`, or an object without `rules` all mean
/// `"shewhart"`: every added test raises the false alarm rate of a stable
/// process, so the supplementary tests are asked for, not assumed.
/// `{ rules: [] }` applies none, leaving control limits only.
///
/// `{ sigma?: number }` -- the within sigma fixed by an earlier (Phase I)
/// study, greater than 0 (ISO 7870-2 §7.5): the centre line of the X-bar chart
/// is still the grand mean of the subgroups given, its limits are
/// `CL ± 3 sigma / sqrt(n)`, and the R chart's centre line and limits are
/// those of `sigma` (`d2 sigma`, `D3`/`D4` times it) instead of the subgroups'
/// own spread. `sigma_hat` is then `sigma` and `sigma_hat_df` is `null`.
/// Pass the `sigma_hat` of the Phase I dispersion chart after its subgroups
/// have been homogenised, and the subgroups that remain here.
#[wasm_bindgen(unchecked_return_type = "XbarRChartDto")]
pub fn xbar_r_chart(
    #[wasm_bindgen(unchecked_param_type = "number[][]")] data: JsValue,
    #[wasm_bindgen(unchecked_optional_param_type = "VariablesChartOptions | null")] options: Option<
        JsValue,
    >,
) -> Result<JsValue, JsValue> {
    let subgroups = read_number_rows(&data, "data").map_err(js_err)?;
    let (rules, sigma) = chart_options(options)?;
    to_js(&xbar_r_dto(subgroups, rules, sigma).map_err(js_err)?)
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
/// `{ rules?: RuleSetName | RunRule[] }`, exactly as for [`xbar_r_chart`].
#[wasm_bindgen(unchecked_return_type = "XbarSChartDto")]
pub fn xbar_s_chart(
    #[wasm_bindgen(unchecked_param_type = "number[][]")] data: JsValue,
    #[wasm_bindgen(unchecked_optional_param_type = "VariablesChartOptions | null")] options: Option<
        JsValue,
    >,
) -> Result<JsValue, JsValue> {
    let subgroups = read_number_rows(&data, "data").map_err(js_err)?;
    let (rules, sigma) = chart_options(options)?;
    to_js(&xbar_s_dto(subgroups, rules, sigma).map_err(js_err)?)
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
/// `{ rules?: RuleSetName | RunRule[] }`, exactly as for [`xbar_r_chart`].
#[wasm_bindgen(unchecked_return_type = "ImrChartDto")]
pub fn imr_chart(
    #[wasm_bindgen(unchecked_param_type = "number[] | Float64Array")] values: JsValue,
    #[wasm_bindgen(unchecked_optional_param_type = "VariablesChartOptions | null")] options: Option<
        JsValue,
    >,
) -> Result<JsValue, JsValue> {
    let values = read_numbers(&values, "values").map_err(js_err)?;
    let (rules, sigma) = chart_options(options)?;
    to_js(&imr_dto(values, rules, sigma).map_err(js_err)?)
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
/// - `options` (optional): `{ rules?: RuleSetName | RunRule[] }`, as for [`xbar_r_chart`]
///
/// # Output JSON
///
/// One `{ index, value, violations }` per input value, in input order.
#[wasm_bindgen(unchecked_return_type = "ChartPointDto[]")]
pub fn run_rules(
    #[wasm_bindgen(unchecked_param_type = "number[] | Float64Array")] values: JsValue,
    #[wasm_bindgen(unchecked_param_type = "LimitsInputDto")] limits: JsValue,
    #[wasm_bindgen(unchecked_optional_param_type = "RuleOptions | null")] options: Option<JsValue>,
) -> Result<JsValue, JsValue> {
    let values = read_numbers(&values, "values").map_err(js_err)?;
    let limits: LimitsInputDto = from_js(limits, "limits")?;
    let rules = rules_option(options)?;
    to_js(&run_rules_dto(values, limits, rules).map_err(js_err)?)
}

// ---------------------------------------------------------------------------
// TypeScript declarations for the option arguments
// ---------------------------------------------------------------------------

/// The optional last argument of the charts and `run_rules`, declared for
/// TypeScript. The variables charts read a rule set; each attributes chart
/// reads only its own Phase I standard, so each is declared with only that.
#[wasm_bindgen(typescript_custom_section)]
const OPTIONS_TS: &'static str = r#"
/** A run test, by the name a point's `violations` reports it under. */
export type RunRule = "BeyondLimits" | "NineOneSide" | "SixTrend" | "FourteenAlternating" | "TwoOfThreeBeyond2Sigma" | "FourOfFiveBeyond1Sigma" | "FifteenWithin1Sigma" | "EightBeyond1Sigma" | "EightOneSide";
/** A named set of run tests: the 3-sigma test alone, Western Electric's four, or Nelson's eight. */
export type RuleSetName = "shewhart" | "western_electric" | "nelson";
/** Which run tests to apply: a named set, or the tests spelt out. Omitted or `null`: `"shewhart"`. `[]`: none. */
export interface RuleOptions {
    rules?: RuleSetName | RunRule[] | null;
}
/** The options of the variables charts (X-bar R, X-bar S, I-MR): the run tests, and the within sigma fixed by an earlier (Phase I) study — greater than 0. */
export interface VariablesChartOptions extends RuleOptions {
    sigma?: number | null;
}
/** A known centre line from a Phase I study. */
export interface PChartOptions {
    p_bar?: number | null;
}
/** A known centre line and dispersion factor from a Phase I study. */
export interface LaneyPChartOptions {
    p_bar?: number | null;
    phi?: number | null;
}
/** A known centre line from a Phase I study. */
export interface UChartOptions {
    u_bar?: number | null;
}
/** A known centre line and dispersion factor from a Phase I study. */
export interface LaneyUChartOptions {
    u_bar?: number | null;
    phi?: number | null;
}
"#;

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

/// The optional `{ rules?, sigma? }` argument of the variables charts: the run
/// tests and the within sigma fixed by an earlier study.
fn chart_options(options: Option<JsValue>) -> Result<(crate::spc::RuleSet, Option<f64>), JsValue> {
    match options {
        Some(o) if !o.is_undefined() && !o.is_null() => {
            let value: serde_json::Value = from_js(o, "options")?;
            let sigma = crate::wire::sigma_from_json(Some(&value)).map_err(js_err)?;
            let rules = rules_from_json(Some(value)).map_err(js_err)?;
            Ok((rules, sigma))
        }
        _ => Ok((crate::spc::RuleSet::default(), None)),
    }
}

/// Reads the optional `{ p_bar?, u_bar?, phi? }` standard of an attributes chart.
fn standard_option(options: Option<JsValue>) -> Result<AttributeStandardDto, JsValue> {
    match options {
        Some(o) if !o.is_undefined() && !o.is_null() => from_js(o, "options"),
        _ => Ok(AttributeStandardDto::default()),
    }
}

/// Compute a P chart from (defectives, sample_size) pairs.
///
/// # Input JSON
///
/// Array of `[defectives, sample_size]` pairs (as integers):
/// `[[3, 100], [5, 100], ...]`.
///
/// # Errors
///
/// Throws an `Error` with `code` and `index` (the row): `count_not_whole`
/// for a defective count that is not a whole number >= 0,
/// `sample_size_not_whole` for a sample size that is not a whole number >= 1
/// (zero, negative, fractional or not a number alike),
/// `defectives_exceed_sample`, `malformed_input` for a row that is not a pair,
/// `insufficient_data` for an empty array, `parameter_out_of_range` for a
/// `p_bar` outside (0, 1).
///
/// # Options
///
/// `{ p_bar?: number }` -- a known centre line from a Phase I study. Every
/// limit then uses it with the sample's own size (Phase II); without it p-bar
/// is estimated from `samples`.
///
/// # Output JSON
///
/// Object with fields: `p_bar` (the centre line used), `points` (array, each
/// with `z` -- the standardized value, on which every limit is +/-3),
/// `in_control`.
#[wasm_bindgen(unchecked_return_type = "PChartDto")]
pub fn p_chart(
    #[wasm_bindgen(unchecked_param_type = "[number, number][]")] samples: JsValue,
    #[wasm_bindgen(unchecked_optional_param_type = "PChartOptions | null")] options: Option<
        JsValue,
    >,
) -> Result<JsValue, JsValue> {
    let json: serde_json::Value = from_js(samples, "samples")?;
    let samples = count_pairs(&json, "samples").map_err(js_err)?;
    let standard = standard_option(options)?;
    to_js(&p_chart_dto(&samples, &standard).map_err(js_err)?)
}
/// Compute process capability indices (Cp, Cpk, Pp, Ppk, Cpm).
///
/// # Input JSON
///
/// ```text
/// { data: number[], usl?: number, lsl?: number,
///   sigma_within?: number, target?: number,
///   sigma_within_df?: number, confidence?: number }
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
/// `cpm`, `intervals`. Indices that the specification does not support are
/// `null` (a one-sided specification has no `cp`, `pp` or `cpm`).
///
/// `intervals` is `{ confidence, cp, cpk, pp, ppk }`, each a two-sided
/// `{ lower, upper }` at `confidence` (default 0.95) or `null` where the index
/// is: Cp/Pp from the chi-square distribution of the variance, Cpk/Ppk by
/// Bissell's (1990) approximation. Pp/Ppk use `n - 1` degrees of freedom; the
/// within sigma's depend on how it was estimated, so `cp` / `cpk` intervals
/// need `sigma_within_df` (`n - m` for a pooled standard deviation of `m`
/// subgroups). A one-sided lower bound at 95 % is the `lower` at
/// `confidence: 0.90`.
///
/// `sigma_source` is `"within"` when `sigma_within` was supplied and
/// `"overall"` otherwise. In the `"overall"` case `std_dev_within`, `cp`,
/// `cpk`, `cpu` and `cpl` are all `null`: the short-term indices are
/// undefined without a short-term sigma, and reporting the long-term one in
/// their place would make `cp` equal `pp` for every input. `cpm` uses neither
/// sigma -- it is the spread of `data` about `target` -- so it is reported in
/// both cases.
#[wasm_bindgen(unchecked_return_type = "CapabilityDto")]
pub fn process_capability(
    #[wasm_bindgen(unchecked_param_type = "CapabilityInputDto")] input: JsValue,
) -> Result<JsValue, JsValue> {
    let input: CapabilityInputDto = from_js(input, "input")?;
    to_js(&capability_dto(input).map_err(js_err)?)
}

/// Anderson-Darling normality test. H₀: data is normally distributed.
///
/// # Parameters
///
/// - `data`: `number[]` or `Float64Array` of observations (at least 8, not all
///   equal -- below 8 the sample-size correction is not tabulated)
///
/// # Output JSON
///
/// Object with fields: `statistic` (A²), `statistic_modified` (A²* = A²·(1 +
/// 0.75/n + 2.25/n²)), `p_value`.
#[wasm_bindgen(unchecked_return_type = "AndersonDarlingDto")]
pub fn anderson_darling_test(
    #[wasm_bindgen(unchecked_param_type = "number[] | Float64Array")] data: JsValue,
) -> Result<JsValue, JsValue> {
    let data = read_numbers(&data, "data").map_err(js_err)?;
    to_js(&anderson_darling_dto(&data).map_err(js_err)?)
}

/// Augmented Dickey-Fuller unit-root test. H₀: the series has a unit root
/// (is not stationary).
///
/// # Input JSON
///
/// `{ data: number[], model?: "none" | "constant" | "constant_trend", max_lags?: number | null,
/// lags?: number | null }` -- at least 10 values; `model` defaults to `"constant"`;
/// the lag count is chosen by AIC from 0 to `max_lags` (absent or `null`:
/// Schwert's rule), or fixed by `lags` (not both).
///
/// # Output JSON
///
/// `{ statistic, n_lags, n_obs, levels: [{ level, critical_value, rejected }] }` for
/// the 1 %, 5 % and 10 % levels.
#[wasm_bindgen(unchecked_return_type = "AdfDto")]
pub fn adf_test(
    #[wasm_bindgen(unchecked_param_type = "AdfInputDto")] input: JsValue,
) -> Result<JsValue, JsValue> {
    let input: AdfInputDto = from_js(input, "input")?;
    to_js(&adf_dto(input).map_err(js_err)?)
}

/// KPSS stationarity test (Kwiatkowski et al. 1992): H₀ the series is
/// stationary around a level (`model: "level"`, default) or a trend
/// (`"trend"`) — the complement of [`adf_test`], whose null is the unit root.
/// At least 10 values. The Newey-West truncation lag is chosen by `lag_rule`
/// (`"hobijn"`, default — Hobijn et al. 1998; `"schwert"`) or fixed by `lags`
/// (0 to n − 1; not both).
///
/// # Output JSON
///
/// `{ statistic, n_lags, p_value, p_value_bound, levels: [{ level, critical_value,
/// rejected }] }` for 1 %, 2.5 %, 5 % and 10 %; `p_value` is interpolated in that
/// table, and `p_value_bound` says `"at_most"` / `"at_least"` when the statistic
/// lies beyond it. As statsmodels' `kpss`.
#[wasm_bindgen(unchecked_return_type = "KpssDto")]
pub fn kpss_test(
    #[wasm_bindgen(unchecked_param_type = "KpssInputDto")] input: JsValue,
) -> Result<JsValue, JsValue> {
    let input: KpssInputDto = from_js(input, "input")?;
    to_js(&kpss_dto(input).map_err(js_err)?)
}

// ---------------------------------------------------------------------------
// Correlation, simple regression, distribution fitting
// ---------------------------------------------------------------------------

/// `{ method? }` for [`correlation_matrix`].
#[derive(serde::Deserialize, Default, tsify::Tsify)]
#[cfg_attr(test, derive(schemars::JsonSchema))]
#[tsify(missing_as_null)]
#[serde(deny_unknown_fields)]
struct CorrelationOptions {
    #[serde(default)]
    #[tsify(optional, type = "\"pearson\" | \"spearman\" | \"kendall\"")]
    #[cfg_attr(
        test,
        schemars(schema_with = "crate::wire::card_schema::correlation_method_or_null")
    )]
    method: Option<String>,
}

/// Correlation matrix of `variables` (each a `number[]`, all the same length,
/// at least 3 values, at least 2 variables). `method` is `"pearson"` (default),
/// `"spearman"` or `"kendall"` (tau-b). Returns `{ matrix: number[][] }`.
#[wasm_bindgen(unchecked_return_type = "CorrelationMatrixDto")]
pub fn correlation_matrix(
    #[wasm_bindgen(unchecked_param_type = "number[][]")] variables: JsValue,
    #[wasm_bindgen(unchecked_optional_param_type = "CorrelationOptions | null")] options: Option<
        JsValue,
    >,
) -> Result<JsValue, JsValue> {
    let variables = read_number_rows(&variables, "variables").map_err(js_err)?;
    let options: CorrelationOptions = match options {
        Some(v) if !v.is_null() && !v.is_undefined() => from_js(v, "options")?,
        _ => CorrelationOptions::default(),
    };
    let input = CorrelationInputDto {
        variables,
        method: options.method.unwrap_or_else(default_correlation_method),
    };
    to_js(&correlation_matrix_dto(input).map_err(js_err)?)
}

/// `{ predict?, level? }` for [`simple_regression`].
#[derive(serde::Deserialize, Default, tsify::Tsify)]
#[cfg_attr(test, derive(schemars::JsonSchema))]
#[tsify(missing_as_null)]
#[serde(deny_unknown_fields)]
struct RegressionOptions {
    /// New x values to predict at.
    #[serde(default)]
    #[tsify(optional, type = "number[] | null")]
    predict: Option<Vec<f64>>,
    /// Two-sided level of the intervals, in (0, 1). Omitted: 0.95.
    #[serde(default)]
    #[tsify(optional, type = "number | null")]
    level: Option<f64>,
}

/// Simple linear regression of `y` on `x` (same length, at least 3, `x` not
/// constant): slope, intercept, R², their standard errors, t and p values, the
/// F test, residuals, fitted values and the intercept-slope covariance. A t or
/// F with no finite value (an exact fit) is `null`. `options.predict` lists new
/// x values; `predictions` gives each its `fit`, `se_fit`, the `confidence`
/// interval of the mean and the `prediction` interval of a new observation at
/// `options.level` (default 0.95).
#[wasm_bindgen(unchecked_return_type = "RegressionDto")]
pub fn simple_regression(
    #[wasm_bindgen(unchecked_param_type = "number[] | Float64Array")] x: JsValue,
    #[wasm_bindgen(unchecked_param_type = "number[] | Float64Array")] y: JsValue,
    #[wasm_bindgen(unchecked_optional_param_type = "RegressionOptions | null")] options: Option<
        JsValue,
    >,
) -> Result<JsValue, JsValue> {
    let x = read_numbers(&x, "x").map_err(js_err)?;
    let y = read_numbers(&y, "y").map_err(js_err)?;
    let options: RegressionOptions = match options {
        Some(v) if !v.is_null() && !v.is_undefined() => from_js(v, "options")?,
        _ => RegressionOptions::default(),
    };
    to_js(
        &simple_regression_dto(RegressionInputDto {
            x,
            y,
            predict: options.predict,
            level: options.level,
        })
        .map_err(js_err)?,
    )
}

/// Multiple linear regression of `y` on `predictors` (one array per predictor,
/// each as long as `y`; at least p + 2 observations).
///
/// # Input JSON
///
/// `{ predictors, y, predict?, level? }`. `predict` holds new points laid out as
/// `predictors` (one array per predictor, all the same length); `level` is the
/// two-sided level of their intervals, in (0, 1), default 0.95.
///
/// # Output JSON
///
/// `{ coefficients, std_errors, t_statistics, p_values, r_squared,
/// adjusted_r_squared, f_statistic, f_p_value, residual_se, df_residual,
/// residuals, fitted, vif, covariance, level, predictions }`: coefficients with
/// the intercept first, `covariance[i][j]` the coefficients' covariance, and
/// for each `predict` point its `fit`, `se_fit`, the `confidence` interval of
/// the mean and the `prediction` interval of a new observation. A constant
/// predictor, or one the others determine exactly, is `degenerate_data`.
#[wasm_bindgen(unchecked_return_type = "MultipleRegressionDto")]
pub fn multiple_regression(
    #[wasm_bindgen(unchecked_param_type = "MultipleRegressionInputDto")] input: JsValue,
) -> Result<JsValue, JsValue> {
    let input: MultipleRegressionInputDto = from_js(input, "input")?;
    to_js(&crate::wire::multiple_regression_dto(input).map_err(js_err)?)
}

/// Exponential smoothing of `data` and its forecasts.
///
/// # Input JSON
///
/// `{ data, method, alpha?, beta?, gamma?, period?, seasonality?, horizon?, level? }`.
/// `method` is `"simple"` (level, `alpha`; at least 2 values), `"holt"` (level
/// and trend, `alpha` and `beta`; at least 3) or `"holt_winters"` (level, trend
/// and season, `alpha`, `beta`, `gamma` and `period` >= 2; at least two cycles,
/// `seasonality` `"additive"` (default) or `"multiplicative"`, which needs
/// positive data). A constant given is used, in (0, 1); one left out is
/// estimated by least squares of the one-step errors. A setting the method
/// does not read, or `holt_winters` without `period`, is `invalid_option`.
/// `horizon` steps are forecast (default 0); `level` is the intervals'
/// (default 0.95).
///
/// # Output JSON
///
/// `{ alpha, beta, gamma, estimated, components: { level, trend, seasonal },
/// fitted, residuals, sigma, level, forecasts: [{ step, mean, prediction:
/// { lower, upper } | null }] }`: the constants used (`null` where the method
/// has none) and the names of those estimated.
/// `residuals` are `null` where `fitted` is the initialisation; `sigma` is the
/// root mean squared one-step error. The prediction intervals are those of
/// the additive-error state-space model (ETS(A,N,N), (A,A,N), (A,A,A) --
/// Hyndman et al. 2008); multiplicative seasonality has no closed form, so its
/// forecasts carry `null`.
#[wasm_bindgen(unchecked_return_type = "SmoothingDto")]
pub fn exponential_smoothing(
    #[wasm_bindgen(unchecked_param_type = "SmoothingInputDto")] input: JsValue,
) -> Result<JsValue, JsValue> {
    let input: SmoothingInputDto = from_js(input, "input")?;
    to_js(&crate::wire::exponential_smoothing_dto(input).map_err(js_err)?)
}

/// Error measures of `predicted` against `observed` (errors are observed −
/// predicted).
///
/// # Input JSON
///
/// `{ observed, predicted, parameters? }`, the arrays of one length.
/// `parameters` (default 0, below n) is the number of parameters the model
/// estimated from these observations: the RMSE and NMBE divide by n − p.
///
/// # Output JSON
///
/// `{ n, mean_error, mean_absolute_error, rmse, cv_rmse, nmbe, parameters }`:
/// `cv_rmse` = rmse / mean(observed), `nmbe` = Σ errors / ((n − p) ·
/// mean(observed)) -- fractions, `null` when the mean of `observed` is 0.

#[wasm_bindgen(unchecked_return_type = "PredictionErrorsDto")]
pub fn prediction_errors(
    #[wasm_bindgen(unchecked_param_type = "PredictionErrorsInputDto")] input: JsValue,
) -> Result<JsValue, JsValue> {
    let input: PredictionErrorsInputDto = from_js(input, "input")?;
    to_js(&crate::wire::prediction_errors_dto(input).map_err(js_err)?)
}

/// Every continuous family that fits `data` (at least 2 values, not all equal),
/// best AIC first: `[{ distribution, parameters: { name: value }, log_likelihood,
/// aic, bic }]`. Normal always; Exponential, Gamma, LogNormal and Weibull for
/// positive data; Beta for data in (0, 1).
#[wasm_bindgen(unchecked_return_type = "DistributionFitDto[]")]
pub fn fit_best(
    #[wasm_bindgen(unchecked_param_type = "number[] | Float64Array")] data: JsValue,
) -> Result<JsValue, JsValue> {
    let data = read_numbers(&data, "data").map_err(js_err)?;
    to_js(&fit_best_dto(&data).map_err(js_err)?)
}

/// Whether a family describes the data: fits `distribution` to `data` (at
/// least 8 values) and tests it — `{ distribution, parameters, statistic,
/// p_value, replicates }`. `statistic` is Anderson–Darling A² against the fit;
/// `p_value` is the parametric bootstrap (refit on `replicates` samples drawn
/// from the fit, default 199, seeded by `seed`, default 0), so it accounts for
/// the parameters having been estimated from the same data. Small = the family
/// does not describe the data.
#[wasm_bindgen(unchecked_return_type = "GoodnessOfFitResultDto")]
pub fn goodness_of_fit(
    #[wasm_bindgen(unchecked_param_type = "GoodnessOfFitInputDto")] input: JsValue,
) -> Result<JsValue, JsValue> {
    let input: GoodnessOfFitInputDto = from_js(input, "input")?;
    to_js(&goodness_of_fit_dto(input).map_err(js_err)?)
}

/// Compute the Laney P' chart from (defectives, sample_size) pairs.
///
/// Adjusts control limits for overdispersion via a φ correction factor.
///
/// # Input JSON
///
/// Array of `[defectives, sample_size]` pairs:
/// `[[3, 100], [5, 100], ...]` (need >= 3 subgroups).
///
/// # Errors
///
/// As [`p_chart`], with `insufficient_data` below 3 samples (1 with a standard).
///
/// # Options
///
/// `{ p_bar?: number, phi?: number }` -- both from a Phase I study, or neither:
/// phi scales the standard error about that centre, so the two are one
/// standard.
///
/// # Output JSON
///
/// Object with fields: `p_bar`, `phi` (the values used), `points` (array, each
/// with `z`).
#[wasm_bindgen(unchecked_return_type = "LaneyPChartDto")]
pub fn laney_p_chart(
    #[wasm_bindgen(unchecked_param_type = "[number, number][]")] samples: JsValue,
    #[wasm_bindgen(unchecked_optional_param_type = "LaneyPChartOptions | null")] options: Option<
        JsValue,
    >,
) -> Result<JsValue, JsValue> {
    let json: serde_json::Value = from_js(samples, "samples")?;
    let samples = count_pairs(&json, "samples").map_err(js_err)?;
    let standard = standard_option(options)?;
    to_js(&laney_p_dto(&samples, &standard).map_err(js_err)?)
}

/// Compute an NP chart: the count of defectives in samples of one size.
///
/// # Input
///
/// - `defectives`: `[d1, d2, ...]`, each at most `sample_size`
/// - `sample_size`: the constant sample size, above 0
///
/// # Errors
///
/// Throws an `Error` with `code` and `index`: `count_not_whole` with the row
/// for a defective count, `sample_size_not_whole` with `index: null` for the
/// `sample_size` argument, `defectives_exceed_sample`, `insufficient_data` for
/// an empty array.
///
/// # Output JSON
///
/// Object with fields: `cl`, `ucl`, `lcl`, `points` (array), `in_control`.
#[wasm_bindgen(unchecked_return_type = "FixedLimitChartDto")]
pub fn np_chart(
    #[wasm_bindgen(unchecked_param_type = "number[]")] defectives: JsValue,
    #[wasm_bindgen(unchecked_param_type = "number")] sample_size: JsValue,
) -> Result<JsValue, JsValue> {
    let defectives: serde_json::Value = from_js(defectives, "defectives")?;
    let defectives = count_rows(&defectives, "defectives").map_err(js_err)?;
    let sample_size: serde_json::Value = from_js(sample_size, "sample_size")?;
    let sample_size = sample_size_value(&sample_size, "sample_size").map_err(js_err)?;
    to_js(&np_chart_dto(&defectives, sample_size).map_err(js_err)?)
}

/// Compute a C chart: the count of defects per inspection unit of one size.
///
/// # Input JSON
///
/// `[c1, c2, ...]` (need >= 1).
///
/// # Errors
///
/// Throws an `Error` with `code` and `index`: `count_not_whole` with the row,
/// `insufficient_data` for an empty array.
///
/// # Output JSON
///
/// Object with fields: `cl`, `ucl`, `lcl`, `points` (array), `in_control`.
#[wasm_bindgen(unchecked_return_type = "FixedLimitChartDto")]
pub fn c_chart(
    #[wasm_bindgen(unchecked_param_type = "number[]")] defects: JsValue,
) -> Result<JsValue, JsValue> {
    let defects: serde_json::Value = from_js(defects, "defects")?;
    let defects = count_rows(&defects, "defects").map_err(js_err)?;
    to_js(&c_chart_dto(&defects).map_err(js_err)?)
}

/// Compute a U chart: defects per unit when the quantity inspected varies.
///
/// # Input JSON
///
/// Array of `[defects, units]` pairs. `units` may be fractional (an area, a
/// length) and must be positive.
///
/// # Errors
///
/// Throws an `Error` with `code` and `index` (the row): `count_not_whole`,
/// `units_not_positive`, `malformed_input`, `insufficient_data`,
/// `parameter_out_of_range` (with the domain and `got`).
///
/// # Options
///
/// `{ u_bar?: number }` -- a known centre line (Phase II), as for [`p_chart`].
///
/// # Output JSON
///
/// Object with fields: `u_bar`, `points` (array, each with `z`), `in_control`.
#[wasm_bindgen(unchecked_return_type = "UChartDto")]
pub fn u_chart(
    #[wasm_bindgen(unchecked_param_type = "[number, number][]")] samples: JsValue,
    #[wasm_bindgen(unchecked_optional_param_type = "UChartOptions | null")] options: Option<
        JsValue,
    >,
) -> Result<JsValue, JsValue> {
    let json: serde_json::Value = from_js(samples, "samples")?;
    let samples = rate_pairs(&json, "samples").map_err(js_err)?;
    let standard = standard_option(options)?;
    to_js(&u_chart_dto(&samples, &standard).map_err(js_err)?)
}

/// Compute the Laney U' chart from `[defects, units]` pairs (need >= 3).
///
/// Adjusts the U chart's limits for overdispersion via a φ correction factor.
///
/// # Errors
///
/// As [`u_chart`], with `insufficient_data` below 3 samples (1 with a standard).
///
/// # Options
///
/// `{ u_bar?: number, phi?: number }` -- both or neither, as for
/// [`laney_p_chart`].
///
/// # Output JSON
///
/// Object with fields: `u_bar`, `phi`, `points` (array, each with `z`).
#[wasm_bindgen(unchecked_return_type = "LaneyUChartDto")]
pub fn laney_u_chart(
    #[wasm_bindgen(unchecked_param_type = "[number, number][]")] samples: JsValue,
    #[wasm_bindgen(unchecked_optional_param_type = "LaneyUChartOptions | null")] options: Option<
        JsValue,
    >,
) -> Result<JsValue, JsValue> {
    let json: serde_json::Value = from_js(samples, "samples")?;
    let samples = rate_pairs(&json, "samples").map_err(js_err)?;
    let standard = standard_option(options)?;
    to_js(&laney_u_dto(&samples, &standard).map_err(js_err)?)
}

// ---------------------------------------------------------------------------
// Pure cores of the attributes-chart bindings
// ---------------------------------------------------------------------------
//
// Every attributes chart drops a row it cannot use, and every later point then
// carries the index of the wrong row. The cores refuse such a row by its index
// before the chart sees it.

/// Compute the G chart for rare-event monitoring (inter-event conforming counts).
///
/// Suitable when defect rates are very low (< 1%).
///
/// # Parameters
///
/// - `gaps`: `number[]` or `Float64Array` of inter-event conforming counts
///   (need >= 3, each >= 0)
///
/// # Output JSON
///
/// Object with fields: `g_bar`, `points` (array with ucl/cl/lcl/out_of_control).
#[wasm_bindgen(unchecked_return_type = "GChartDto")]
pub fn g_chart(
    #[wasm_bindgen(unchecked_param_type = "number[] | Float64Array")] gaps: JsValue,
) -> Result<JsValue, JsValue> {
    let gaps = read_numbers(&gaps, "gaps").map_err(js_err)?;
    to_js(&g_chart_dto(&gaps).map_err(js_err)?)
}

/// Compute the T chart for rare-event monitoring (inter-event times).
///
/// Control limits are derived from exponential distribution percentiles.
///
/// # Parameters
///
/// - `times`: `number[]` or `Float64Array` of inter-event times (need >= 3, each > 0)
///
/// # Output JSON
///
/// Object with fields: `t_bar`, `points` (array with ucl/cl/lcl/out_of_control).
#[wasm_bindgen(unchecked_return_type = "TChartDto")]
pub fn t_chart(
    #[wasm_bindgen(unchecked_param_type = "number[] | Float64Array")] times: JsValue,
) -> Result<JsValue, JsValue> {
    let times = read_numbers(&times, "times").map_err(js_err)?;
    to_js(&t_chart_dto(&times).map_err(js_err)?)
}

// ---------------------------------------------------------------------------
// Hypothesis tests
// ---------------------------------------------------------------------------
//
// Every sample is a `number[]` or `Float64Array` read as sent (`read_numbers`):
// `null` or a string is `malformed_input` and a NaN `value_not_finite`, each at
// its index. Too few values is `insufficient_data`; data with no variation,
// for which the statistic is undefined, is `invalid_input`.

/// One-sample t test of `H0: mean = mu0` (need >= 2 values).
#[wasm_bindgen(unchecked_return_type = "TestResultDto")]
pub fn one_sample_t_test(
    #[wasm_bindgen(unchecked_param_type = "number[] | Float64Array")] data: JsValue,
    mu0: f64,
) -> Result<JsValue, JsValue> {
    let data = read_numbers(&data, "data").map_err(js_err)?;
    to_js(&one_sample_t_dto(&data, mu0).map_err(js_err)?)
}

/// Welch's two-sample t test (unequal variances; need >= 2 values each).
#[wasm_bindgen(unchecked_return_type = "TestResultDto")]
pub fn two_sample_t_test(
    #[wasm_bindgen(unchecked_param_type = "number[] | Float64Array")] a: JsValue,
    #[wasm_bindgen(unchecked_param_type = "number[] | Float64Array")] b: JsValue,
) -> Result<JsValue, JsValue> {
    let a = read_numbers(&a, "a").map_err(js_err)?;
    let b = read_numbers(&b, "b").map_err(js_err)?;
    to_js(&two_sample_t_dto(&a, &b).map_err(js_err)?)
}

/// Paired t test on `y - x` (same length, need >= 2 pairs).
#[wasm_bindgen(unchecked_return_type = "TestResultDto")]
pub fn paired_t_test(
    #[wasm_bindgen(unchecked_param_type = "number[] | Float64Array")] x: JsValue,
    #[wasm_bindgen(unchecked_param_type = "number[] | Float64Array")] y: JsValue,
) -> Result<JsValue, JsValue> {
    let x = read_numbers(&x, "x").map_err(js_err)?;
    let y = read_numbers(&y, "y").map_err(js_err)?;
    to_js(&paired_t_dto(&x, &y).map_err(js_err)?)
}

/// Mann-Whitney U test, normal approximation with tie correction (need >= 2 values each).
#[wasm_bindgen(unchecked_return_type = "TestResultDto")]
pub fn mann_whitney_u_test(
    #[wasm_bindgen(unchecked_param_type = "number[] | Float64Array")] a: JsValue,
    #[wasm_bindgen(unchecked_param_type = "number[] | Float64Array")] b: JsValue,
) -> Result<JsValue, JsValue> {
    let a = read_numbers(&a, "a").map_err(js_err)?;
    let b = read_numbers(&b, "b").map_err(js_err)?;
    to_js(&mann_whitney_dto(&a, &b).map_err(js_err)?)
}

/// Wilcoxon signed-rank test on paired samples (same length; pairs with `x = y` are dropped, >= 2 must remain).
#[wasm_bindgen(unchecked_return_type = "TestResultDto")]
pub fn wilcoxon_signed_rank_test(
    #[wasm_bindgen(unchecked_param_type = "number[] | Float64Array")] x: JsValue,
    #[wasm_bindgen(unchecked_param_type = "number[] | Float64Array")] y: JsValue,
) -> Result<JsValue, JsValue> {
    let x = read_numbers(&x, "x").map_err(js_err)?;
    let y = read_numbers(&y, "y").map_err(js_err)?;
    to_js(&wilcoxon_dto(&x, &y).map_err(js_err)?)
}

/// Equivalence of one mean to a target, by two one-sided tests (TOST, Schuirmann 1987):
/// is `mean(data) - target` inside `(low, high)`? `low` and `high` are margins on the
/// difference, so `target = 100, low = -0.5, high = 0.5` asks whether the mean is within
/// 0.5 of 100. Equivalence is declared when both one-sided tests reject, that is when
/// `p_value` (the larger of `p_lower` and `p_upper`) is below `alpha` -- the same as the
/// `1 - 2 * alpha` interval of the difference lying inside the margins. Need >= 2 values
/// that are not all equal, `low < high` and `alpha` in (0, 0.5), default 0.05.
#[wasm_bindgen(unchecked_return_type = "EquivalenceDto")]
pub fn tost_one_sample(
    #[wasm_bindgen(unchecked_param_type = "number[] | Float64Array")] data: JsValue,
    target: f64,
    low: f64,
    high: f64,
    #[wasm_bindgen(unchecked_optional_param_type = "number | null")] alpha: Option<f64>,
) -> Result<JsValue, JsValue> {
    let data = read_numbers(&data, "data").map_err(js_err)?;
    to_js(&tost_one_sample_dto(&data, target, low, high, alpha).map_err(js_err)?)
}

/// Equivalence of paired data by TOST: is the mean of `x - y` inside `(low, high)`? Note the
/// direction: `paired_t_test` works on `y - x`. Same length, need >= 2 pairs whose
/// differences are not all equal; `alpha` in (0, 0.5), default 0.05.
#[wasm_bindgen(unchecked_return_type = "EquivalenceDto")]
pub fn tost_paired(
    #[wasm_bindgen(unchecked_param_type = "number[] | Float64Array")] x: JsValue,
    #[wasm_bindgen(unchecked_param_type = "number[] | Float64Array")] y: JsValue,
    low: f64,
    high: f64,
    #[wasm_bindgen(unchecked_optional_param_type = "number | null")] alpha: Option<f64>,
) -> Result<JsValue, JsValue> {
    let x = read_numbers(&x, "x").map_err(js_err)?;
    let y = read_numbers(&y, "y").map_err(js_err)?;
    to_js(&tost_paired_dto(&x, &y, low, high, alpha).map_err(js_err)?)
}

/// Equivalence of two independent samples by TOST: is `mean(a) - mean(b)` inside
/// `(low, high)`? Welch's standard error and degrees of freedom, or with `equal_variance`
/// `true` the pooled ones (default `false`). Need >= 2 values each, not all equal across both
/// samples; `alpha` in (0, 0.5), default 0.05.
#[wasm_bindgen(unchecked_return_type = "EquivalenceDto")]
pub fn tost_two_sample(
    #[wasm_bindgen(unchecked_param_type = "number[] | Float64Array")] a: JsValue,
    #[wasm_bindgen(unchecked_param_type = "number[] | Float64Array")] b: JsValue,
    low: f64,
    high: f64,
    #[wasm_bindgen(unchecked_optional_param_type = "number | null")] alpha: Option<f64>,
    #[wasm_bindgen(unchecked_optional_param_type = "boolean | null")] equal_variance: Option<bool>,
) -> Result<JsValue, JsValue> {
    let a = read_numbers(&a, "a").map_err(js_err)?;
    let b = read_numbers(&b, "b").map_err(js_err)?;
    to_js(&tost_two_sample_dto(&a, &b, low, high, alpha, equal_variance).map_err(js_err)?)
}

/// A name argument as sent: a string, or `malformed_input` at `param`. Read from the
/// `JsValue` rather than taken as a `String` so that a number or `null` is refused
/// here, not coerced by the glue.
fn read_name(value: &JsValue, param: &'static str) -> Result<String, WireError> {
    value.as_string().ok_or_else(|| {
        WireError::new(
            crate::wire::code::MALFORMED_INPUT,
            None,
            format!("{param}: expected a name (a string)"),
        )
        .about(param)
    })
}

/// The factor `k` of a normal tolerance interval `mean ± k * s`: the number of sample
/// standard deviations either side of the mean that contains at least `coverage` of the
/// population with probability `confidence`, for a sample of `n` values.
///
/// It is the root of the exact confidence equation (Eisenhart, Hastay and Wallis 1947 for
/// the two-sided case; the noncentral `t` for the one-sided one), not an approximation, so
/// it differs in the third or fourth digit from the factors in tables that print the
/// Wald-Wolfowitz or Howe formula. `sided` is `"two_sided"` (the symmetric interval) or
/// `"one_sided"` (one bound: `mean - k * s` or `mean + k * s`).
///
/// `n` is a whole number >= 2 (a fraction throws `malformed_input`, anything below 2
/// `parameter_out_of_range` with `min` 2 and `got`); `coverage` and `confidence` are in
/// (0, 1), open at both ends (`parameter_out_of_range` with `exclusive_min` 0,
/// `exclusive_max` 1 and `got`); another `sided` is `unknown_option` with `got` and
/// `expected`.
#[wasm_bindgen]
pub fn tolerance_factor(
    #[wasm_bindgen(unchecked_param_type = "number")] n: JsValue,
    coverage: f64,
    confidence: f64,
    #[wasm_bindgen(unchecked_param_type = "\"two_sided\" | \"one_sided\"")] sided: JsValue,
) -> Result<f64, JsValue> {
    let n: Whole = from_js(n, "n")?;
    let sided = read_name(&sided, "sided").map_err(js_err)?;
    tolerance_factor_value(n, coverage, confidence, &sided).map_err(js_err)
}

/// The normal tolerance interval of a sample: `mean ± k * std_dev` with the factor of
/// [`tolerance_factor`] for `data.length`, `coverage`, `confidence` and `sided` -- at least
/// `coverage` of the population lies inside, with probability `confidence`. It speaks about
/// individual values (a specification limit), where a confidence interval speaks about the
/// mean, and it assumes the population is normal.
///
/// A one-sided statement uses **one** of `lower` and `upper`: "at least `coverage` of the
/// population is above `lower`" or "... below `upper`", each with `confidence`. Both are
/// returned; quoting them as an interval is the two-sided statement with the larger
/// two-sided factor. Need >= 2 values that are not all equal.
#[wasm_bindgen(unchecked_return_type = "ToleranceIntervalDto")]
pub fn tolerance_interval(
    #[wasm_bindgen(unchecked_param_type = "number[] | Float64Array")] data: JsValue,
    coverage: f64,
    confidence: f64,
    #[wasm_bindgen(unchecked_param_type = "\"two_sided\" | \"one_sided\"")] sided: JsValue,
) -> Result<JsValue, JsValue> {
    let data = read_numbers(&data, "data").map_err(js_err)?;
    let sided = read_name(&sided, "sided").map_err(js_err)?;
    to_js(&tolerance_interval_dto(&data, coverage, confidence, &sided).map_err(js_err)?)
}

/// Type 1 gage study: the capability of a measurement device from repeated measurements of
/// one reference part whose true value is `reference`, against the specification limits
/// `lsl` and `usl` (tolerance `T = usl - lsl`). With the mean `x̄` and standard deviation
/// `s` of the `measurements`, `bias = x̄ - reference`, `Cg = tolerance_fraction * T /
/// (sigma_multiple * s)` and `Cgk = (tolerance_fraction * T / 2 - |bias|) / (sigma_multiple
/// * s / 2)`; `t_statistic`, `df` and `p_value` are those of the one-sample t test of
/// `bias = 0` (two-sided). `percent_var_repeatability` is `100 * sigma_multiple * s / T` and
/// `percent_var_repeatability_and_bias` is `100 * tolerance_fraction / Cgk` (`null` when
/// `Cgk <= 0`, the bias alone filling the band). `tolerance_fraction` (default 0.2, in (0, 1])
/// and `sigma_multiple` (default 6, above 0) are the conventions that differ between editions
/// and tools; the defaults are Minitab's, `Cg = 0.2 T / 6 s` and `Cgk = (0.1 T - |bias|) /
/// 3 s`. 1.33 is the customary benchmark for both. Need >= 2 measurements that are not all
/// equal and `lsl < usl`.
#[wasm_bindgen(unchecked_return_type = "Type1GageDto")]
pub fn type1_gage_study(
    #[wasm_bindgen(unchecked_param_type = "number[] | Float64Array")] measurements: JsValue,
    reference: f64,
    lsl: f64,
    usl: f64,
    #[wasm_bindgen(unchecked_optional_param_type = "number | null")] tolerance_fraction: Option<
        f64,
    >,
    #[wasm_bindgen(unchecked_optional_param_type = "number | null")] sigma_multiple: Option<f64>,
) -> Result<JsValue, JsValue> {
    let measurements = read_numbers(&measurements, "measurements").map_err(js_err)?;
    to_js(
        &type1_gage_study_dto(
            &measurements,
            reference,
            lsl,
            usl,
            tolerance_fraction,
            sigma_multiple,
        )
        .map_err(js_err)?,
    )
}

/// Chi-squared goodness of fit: `observed` counts (>= 0) against `expected` frequencies (> 0), same length >= 2.
#[wasm_bindgen(unchecked_return_type = "TestResultDto")]
pub fn chi_squared_goodness_of_fit(
    #[wasm_bindgen(unchecked_param_type = "number[] | Float64Array")] observed: JsValue,
    #[wasm_bindgen(unchecked_param_type = "number[] | Float64Array")] expected: JsValue,
) -> Result<JsValue, JsValue> {
    let observed = read_numbers(&observed, "observed").map_err(js_err)?;
    let expected = read_numbers(&expected, "expected").map_err(js_err)?;
    to_js(&chi_squared_gof_dto(&observed, &expected).map_err(js_err)?)
}

/// Jarque-Bera normality test (need >= 8 values).
#[wasm_bindgen(unchecked_return_type = "TestResultDto")]
pub fn jarque_bera_test(
    #[wasm_bindgen(unchecked_param_type = "number[] | Float64Array")] data: JsValue,
) -> Result<JsValue, JsValue> {
    let data = read_numbers(&data, "data").map_err(js_err)?;
    to_js(&jarque_bera_dto(&data).map_err(js_err)?)
}

/// Shapiro-Wilk normality test, Royston (1995) (3 to 5000 values). Returns `{ w, p_value }`.
#[wasm_bindgen(unchecked_return_type = "ShapiroWilkDto")]
pub fn shapiro_wilk_test(
    #[wasm_bindgen(unchecked_param_type = "number[] | Float64Array")] data: JsValue,
) -> Result<JsValue, JsValue> {
    let data = read_numbers(&data, "data").map_err(js_err)?;
    to_js(&shapiro_wilk_dto(&data).map_err(js_err)?)
}

/// Mann-Kendall trend test with Kendall's tau and Sen's slope (need >= 4 values).
#[wasm_bindgen(unchecked_return_type = "MannKendallDto")]
pub fn mann_kendall_test(
    #[wasm_bindgen(unchecked_param_type = "number[] | Float64Array")] data: JsValue,
) -> Result<JsValue, JsValue> {
    let data = read_numbers(&data, "data").map_err(js_err)?;
    to_js(&mann_kendall_dto(&data).map_err(js_err)?)
}

/// `observation` as sent: `truncation` checked by name (`unknown_option`),
/// then the rest read without changing a number.
fn read_observation(value: JsValue) -> Result<ObservationDto, JsValue> {
    let name = js_sys::Reflect::get(&value, &JsValue::from_str("truncation"))
        .ok()
        .and_then(|v| v.as_string())
        .unwrap_or_default();
    check_truncation(&name).map_err(js_err)?;
    from_js(value, "observation")
}

/// The optional settings argument of the counter functions.
fn read_counter_options(
    options: Option<JsValue>,
) -> Result<crate::wire::counter::CounterOptionsDto, JsValue> {
    match options {
        Some(o) if !o.is_undefined() && !o.is_null() => from_js(o, "options"),
        _ => Ok(Default::default()),
    }
}

/// How much a cumulative counter advanced between each pair of neighbouring
/// readings, and what explains it: `[{ start, end, increase, kind }]` with
/// `kind` `"normal"` (did not fall), `"declared"` (a declared reset or
/// replacement), `"wrap"` (fell, explained by `options.modulus`) or
/// `"unexplained"` (fell, nothing explains it -- `increase` is `null`).
///
/// `times` must increase strictly. `options`: `{ modulus?, max_rate?,
/// discontinuities?: [{ time, start, last? }] }`.
#[wasm_bindgen(unchecked_return_type = "CounterIncrementDto[]")]
pub fn counter_increments(
    #[wasm_bindgen(unchecked_param_type = "number[] | Float64Array")] times: JsValue,
    #[wasm_bindgen(unchecked_param_type = "number[] | Float64Array")] values: JsValue,
    #[wasm_bindgen(unchecked_optional_param_type = "CounterOptionsDto | null")] options: Option<
        JsValue,
    >,
) -> Result<JsValue, JsValue> {
    let times = read_numbers(&times, "times").map_err(js_err)?;
    let values = read_numbers(&values, "values").map_err(js_err)?;
    let options = read_counter_options(options)?;
    to_js(&crate::wire::counter::increments_dto(&times, &values, &options).map_err(js_err)?)
}

/// A cumulative counter's increments summed over the intervals
/// `[edges[j], edges[j+1])`: `[{ start, end, increase, pairs, unexplained }]`.
/// A pair counts in the interval its later reading falls in; `unexplained`
/// is how many of the interval's pairs have no known increase.
#[wasm_bindgen(unchecked_return_type = "CounterTotalDto[]")]
pub fn counter_totals(
    #[wasm_bindgen(unchecked_param_type = "number[] | Float64Array")] times: JsValue,
    #[wasm_bindgen(unchecked_param_type = "number[] | Float64Array")] values: JsValue,
    #[wasm_bindgen(unchecked_param_type = "number[] | Float64Array")] edges: JsValue,
    #[wasm_bindgen(unchecked_optional_param_type = "CounterOptionsDto | null")] options: Option<
        JsValue,
    >,
) -> Result<JsValue, JsValue> {
    let times = read_numbers(&times, "times").map_err(js_err)?;
    let values = read_numbers(&values, "values").map_err(js_err)?;
    let edges = read_numbers(&edges, "edges").map_err(js_err)?;
    let options = read_counter_options(options)?;
    to_js(&crate::wire::counter::totals_dto(&times, &values, &edges, &options).map_err(js_err)?)
}

/// Laplace trend test of event times against a constant rate.
///
/// `observation` is `{ truncation: "time", end }` (watched until `end`) or
/// `{ truncation: "failure" }` (stopped at the last event, which is then not
/// used). Returns `{ statistic, p_value, direction, events_used, df: null }`;
/// `p_value` is two-sided.
#[wasm_bindgen(unchecked_return_type = "TrendTestDto")]
pub fn laplace_trend_test(
    #[wasm_bindgen(unchecked_param_type = "number[] | Float64Array")] times: JsValue,
    #[wasm_bindgen(unchecked_param_type = "ObservationDto")] observation: JsValue,
) -> Result<JsValue, JsValue> {
    let times = read_numbers(&times, "times").map_err(js_err)?;
    let o = read_observation(observation)?;
    to_js(&laplace_dto(&times, o).map_err(js_err)?)
}

/// MIL-HDBK-189 trend test of event times against a constant rate:
/// `χ² = 2·Σ ln(T/tᵢ)` with `df = 2m`. `observation` as for
/// [`laplace_trend_test`].
#[wasm_bindgen(unchecked_return_type = "TrendTestDto")]
pub fn mil_hdbk_189_test(
    #[wasm_bindgen(unchecked_param_type = "number[] | Float64Array")] times: JsValue,
    #[wasm_bindgen(unchecked_param_type = "ObservationDto")] observation: JsValue,
) -> Result<JsValue, JsValue> {
    let times = read_numbers(&times, "times").map_err(js_err)?;
    let o = read_observation(observation)?;
    to_js(&mil_hdbk_189_dto(&times, o).map_err(js_err)?)
}

/// Power-law process (Crow-AMSAA) fit of event times:
/// `{ beta, beta_unbiased, lambda, intensity_at_end, end, events }`.
/// `observation` as for [`laplace_trend_test`].
#[wasm_bindgen(unchecked_return_type = "PowerLawFitDto")]
pub fn power_law_process_fit(
    #[wasm_bindgen(unchecked_param_type = "number[] | Float64Array")] times: JsValue,
    #[wasm_bindgen(unchecked_param_type = "ObservationDto")] observation: JsValue,
) -> Result<JsValue, JsValue> {
    let times = read_numbers(&times, "times").map_err(js_err)?;
    let o = read_observation(observation)?;
    to_js(&power_law_dto(&times, o).map_err(js_err)?)
}

/// One-way ANOVA across `groups` (>= 2 groups of >= 2 values).
#[wasm_bindgen(unchecked_return_type = "AnovaDto")]
pub fn one_way_anova(
    #[wasm_bindgen(unchecked_param_type = "number[][]")] groups: JsValue,
) -> Result<JsValue, JsValue> {
    let groups = read_number_rows(&groups, "groups").map_err(js_err)?;
    to_js(&one_way_anova_dto(&groups).map_err(js_err)?)
}

/// Kruskal-Wallis H test across `groups` (>= 2 groups of >= 2 values).
#[wasm_bindgen(unchecked_return_type = "TestResultDto")]
pub fn kruskal_wallis_test(
    #[wasm_bindgen(unchecked_param_type = "number[][]")] groups: JsValue,
) -> Result<JsValue, JsValue> {
    let groups = read_number_rows(&groups, "groups").map_err(js_err)?;
    to_js(&groups_test_dto("kruskal_wallis_test", &groups).map_err(js_err)?)
}

/// Levene's test (Brown-Forsythe, median-centred) for equal variances across `groups`.
#[wasm_bindgen(unchecked_return_type = "TestResultDto")]
pub fn levene_test(
    #[wasm_bindgen(unchecked_param_type = "number[][]")] groups: JsValue,
) -> Result<JsValue, JsValue> {
    let groups = read_number_rows(&groups, "groups").map_err(js_err)?;
    to_js(&groups_test_dto("levene_test", &groups).map_err(js_err)?)
}

/// Bartlett's test for equal variances across `groups` (each group must vary).
#[wasm_bindgen(unchecked_return_type = "TestResultDto")]
pub fn bartlett_test(
    #[wasm_bindgen(unchecked_param_type = "number[][]")] groups: JsValue,
) -> Result<JsValue, JsValue> {
    let groups = read_number_rows(&groups, "groups").map_err(js_err)?;
    to_js(&groups_test_dto("bartlett_test", &groups).map_err(js_err)?)
}

/// Chi-squared test of independence on a contingency `table` (>= 2 x 2, counts >= 0).
#[wasm_bindgen(unchecked_return_type = "TestResultDto")]
pub fn chi_squared_independence(
    #[wasm_bindgen(unchecked_param_type = "number[][]")] table: JsValue,
) -> Result<JsValue, JsValue> {
    let table = read_number_rows(&table, "table").map_err(js_err)?;
    to_js(&chi_squared_independence_dto(&table).map_err(js_err)?)
}

/// Fisher's exact test on a 2 x 2 `table` of whole counts `[[a, b], [c, d]]`.
#[wasm_bindgen(unchecked_return_type = "TestResultDto")]
pub fn fisher_exact_test(
    #[wasm_bindgen(unchecked_param_type = "[[number, number], [number, number]]")] table: JsValue,
) -> Result<JsValue, JsValue> {
    let table: serde_json::Value = from_js(table, "table")?;
    to_js(&fisher_exact_dto(&table).map_err(js_err)?)
}

/// Bonferroni-adjusted p-values, `min(p * m, 1)`, in input order.
#[wasm_bindgen]
pub fn bonferroni_correction(
    #[wasm_bindgen(unchecked_param_type = "number[] | Float64Array")] p_values: JsValue,
) -> Result<Vec<f64>, JsValue> {
    let p = read_numbers(&p_values, "p_values").map_err(js_err)?;
    p_adjust_dto("bonferroni_correction", &p).map_err(js_err)
}

/// Benjamini-Hochberg (1995) FDR-adjusted p-values, in input order.
#[wasm_bindgen]
pub fn benjamini_hochberg(
    #[wasm_bindgen(unchecked_param_type = "number[] | Float64Array")] p_values: JsValue,
) -> Result<Vec<f64>, JsValue> {
    let p = read_numbers(&p_values, "p_values").map_err(js_err)?;
    p_adjust_dto("benjamini_hochberg", &p).map_err(js_err)
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
#[wasm_bindgen(unchecked_return_type = "PeltResultDto")]
pub fn detect_changepoints(
    #[wasm_bindgen(unchecked_param_type = "PeltInputDto")] input: JsValue,
) -> Result<JsValue, JsValue> {
    let input: PeltInputDto = from_js(input, "input")?;
    to_js(&pelt_dto(input).map_err(js_err)?)
}

// ---------------------------------------------------------------------------
// Multi-signal PELT
// ---------------------------------------------------------------------------

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
#[wasm_bindgen(unchecked_return_type = "PeltResultDto")]
pub fn detect_changepoints_multi(
    #[wasm_bindgen(unchecked_param_type = "MultiPeltInputDto")] input: JsValue,
) -> Result<JsValue, JsValue> {
    let input: MultiPeltInputDto = from_js(input, "input")?;
    to_js(&multi_pelt_dto(input).map_err(js_err)?)
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
#[wasm_bindgen(unchecked_return_type = "GageRRResultDto")]
pub fn gage_rr_xbar_r(
    #[wasm_bindgen(unchecked_param_type = "GageRRInputDto")] input: JsValue,
) -> Result<JsValue, JsValue> {
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
#[wasm_bindgen(unchecked_return_type = "GageRRAnovaResultDto")]
pub fn gage_rr_anova(
    #[wasm_bindgen(unchecked_param_type = "GageRRInputDto")] input: JsValue,
) -> Result<JsValue, JsValue> {
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
/// `median`, `percentile_lower`, `percentile_upper`, `distribution`,
/// `parameters`, and `goodness_of_fit: { statistic, p_value, replicates }` —
/// whether the family describes the data (see `goodness_of_fit`; `p_value` is
/// `null` with `fit_replicates: 0`).
#[wasm_bindgen(unchecked_return_type = "PercentileCapabilityDto")]
pub fn percentile_capability(
    #[wasm_bindgen(unchecked_param_type = "PercentileCapabilityInputDto")] input: JsValue,
) -> Result<JsValue, JsValue> {
    let input: PercentileCapabilityInputDto = from_js(input, "input")?;
    to_js(&percentile_capability_dto(input).map_err(js_err)?)
}

// ---------------------------------------------------------------------------
// CUSUM / EWMA -- online (sequential) shift detection
// ---------------------------------------------------------------------------

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
#[wasm_bindgen(unchecked_return_type = "CusumDto")]
pub fn cusum(
    #[wasm_bindgen(unchecked_param_type = "CusumInputDto")] input: JsValue,
) -> Result<JsValue, JsValue> {
    let input: CusumInputDto = from_js(input, "input")?;
    to_js(&cusum_dto(input).map_err(js_err)?)
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
#[wasm_bindgen(unchecked_return_type = "EwmaDto")]
pub fn ewma(
    #[wasm_bindgen(unchecked_param_type = "EwmaInputDto")] input: JsValue,
) -> Result<JsValue, JsValue> {
    let input: EwmaInputDto = from_js(input, "input")?;
    to_js(&ewma_dto(input).map_err(js_err)?)
}

// ---------------------------------------------------------------------------
// Non-normal capability (Box-Cox) and sigma level <-> PPM
// ---------------------------------------------------------------------------

/// Process capability for non-normal data, via a Box-Cox transformation.
///
/// # Input JSON
///
/// ```json
/// { "data": [1.2, 3.4, 9.1, 22.0], "usl": 100.0, "lsl": 1.0, "lambda_range": [-5, 5] }
/// ```
///
/// - `data` (required): at least 4 observations, **all strictly positive**.
/// - `usl` / `lsl` (optional): each must be positive. With neither, only the
///   lambda estimate is returned and every index is `null`.
/// - `lambda_range` (optional): `[min, max]` search range, default `[-5, 5]`
///   (the range Minitab searches).
///
/// # Output JSON
///
/// ```json
/// { "lambda": 0.13, "lambda_at_bound": false,
///   "cp": null, "cpk": null, "cpu": null, "cpl": null,
///   "pp": 1.42, "ppk": 1.19, "ppu": 1.19, "ppl": 1.65, "cpm": null }
/// ```
///
/// The optimal lambda is estimated by maximum likelihood over `lambda_range`;
/// `lambda_at_bound: true` means the likelihood was still rising at an end of
/// the range, so `lambda` is that limit, not an interior optimum. The
/// specification limits are transformed with that same lambda, and the indices
/// are computed on the transformed scale. `cp`/`cpk`/`cpu`/`cpl` need a
/// short-term sigma, which a flat vector cannot carry, so they come back
/// `null` here -- exactly as they do from `process_capability` without
/// `sigma_within`.
#[wasm_bindgen(unchecked_return_type = "BoxcoxCapabilityDto")]
pub fn boxcox_capability(
    #[wasm_bindgen(unchecked_param_type = "BoxcoxCapabilityInputDto")] input: JsValue,
) -> Result<JsValue, JsValue> {
    let input: BoxcoxCapabilityInputDto = from_js(input, "input")?;
    let dto = boxcox_capability_dto(input).map_err(js_err)?;
    to_js(&dto)
}

// ---------------------------------------------------------------------------
// Seasonality
// ---------------------------------------------------------------------------

/// Estimate the dominant period of a univariate series.
///
/// Two stages (AutoPeriod, Vlachos et al. 2005): the linearly detrended,
/// zero-padded periodogram is searched for peaks above a permutation
/// threshold (99th percentile of the largest ordinate over 100 seeded
/// shuffles), and each peak's frequency band is refined on the
/// autocorrelation function to an integer lag that is a local maximum above
/// the `1.96/√n` white-noise bound. Deterministic for a given series.
///
/// # Input JSON
///
/// ```json
/// { "data": [0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1] }
/// ```
///
/// - `data` (required): at least 8 finite values.
///
/// # Output JSON
///
/// ```json
/// { "period": 7,
///   "candidates": [{ "period": 7, "acf": 0.71, "bin": 18, "power": 21.3, "power_share": 0.62 }],
///   "n": 16, "acf_threshold": 0.49, "power_threshold": 6.8 }
/// ```
///
/// `period` is `null` — explicitly, not an error — when no periodicity passes
/// both stages: a constant, a pure trend, white noise. Only periods from 2 to
/// `n / 2` are admissible. Each candidate carries the ACF at its lag (its
/// strength), the periodogram bin that produced it (1-based, of the padded
/// transform), that bin's power and share of the total.
/// Gaussian kernel density estimate of a sample (at least 2 values):
/// `{ x, density, bandwidth }` on `n_points` (default 512) evenly spaced
/// points from 3 bandwidths below the smallest value to 3 above the largest.
/// The bandwidth is Silverman's rule (0.9 · min(σ, IQR / 1.34) · n^−1/5), Scott's
/// (`rule: "scott"`, 1.06 · σ · n^−1/5) or your own (`bandwidth`).
#[wasm_bindgen(unchecked_return_type = "KdeDto")]
pub fn kde(
    #[wasm_bindgen(unchecked_param_type = "KdeInputDto")] input: JsValue,
) -> Result<JsValue, JsValue> {
    let input: KdeInputDto = from_js(input, "input")?;
    to_js(&kde_dto(input).map_err(js_err)?)
}

/// Sample autocorrelation of a series (at least 2 values) for lags 0 to
/// `max_lag` (absent = `min(⌊10·log₁₀ n⌋, n − 1)`): `{ acf, confidence_threshold,
/// bartlett_limits }`. `confidence_threshold` is the 95 % white-noise band
/// z / √n; `bartlett_limits[k − 1]` is the 95 % band for lag k by Bartlett's
/// formula (zero autocorrelation beyond lag k − 1) — the band an ACF plot draws.
/// Biased estimator (denominator n), as statsmodels' `acf`.
#[wasm_bindgen(unchecked_return_type = "AcfDto")]
pub fn acf(
    #[wasm_bindgen(unchecked_param_type = "AutocorrelationInputDto")] input: JsValue,
) -> Result<JsValue, JsValue> {
    let input: AutocorrelationInputDto = from_js(input, "input")?;
    to_js(&acf_dto(input).map_err(js_err)?)
}

/// Sample partial autocorrelation of a series (at least 4 values) for lags 1 to
/// `max_lag` (below n / 2; absent = `min(⌊10·log₁₀ n⌋, ⌊n/2⌋ − 1)`):
/// `{ pacf, confidence_threshold }`, by Durbin-Levinson on the biased ACF
/// (statsmodels' `pacf(method="ldb")`). `confidence_threshold` is z / √n.
#[wasm_bindgen(unchecked_return_type = "PacfDto")]
pub fn pacf(
    #[wasm_bindgen(unchecked_param_type = "AutocorrelationInputDto")] input: JsValue,
) -> Result<JsValue, JsValue> {
    let input: AutocorrelationInputDto = from_js(input, "input")?;
    to_js(&pacf_dto(input).map_err(js_err)?)
}

#[wasm_bindgen(unchecked_return_type = "SeasonalityDto")]
pub fn estimate_period(
    #[wasm_bindgen(unchecked_param_type = "SeasonalityInputDto")] input: JsValue,
) -> Result<JsValue, JsValue> {
    let input: SeasonalityInputDto = from_js(input, "input")?;
    to_js(&crate::wire::seasonality_dto(input).map_err(js_err)?)
}

/// Score every point of a series for anomalies by spectral residual saliency.
///
/// Ren et al. (2019): the log amplitude spectrum minus its moving average,
/// transformed back with the original phase, is large where the series has a
/// feature its regular structure does not explain — a spike, a step, a
/// dropout. No model is trained and no period is assumed.
///
/// # Input JSON
///
/// ```json
/// { "data": [1.0, 1.1, 0.9, 1.0, 6.0, 1.0, 1.1, 0.9, 1.0, 1.0, 1.1, 0.9],
///   "averaging_window": 3, "judgement_window": 40, "threshold": 3.0,
///   "min_zscore": 1.5, "sensitivity": 70, "batch_size": null }
/// ```
///
/// - `data` (required): at least 12 finite values.
/// - `averaging_window` (optional, default 3): moving average width on the log spectrum (`q`).
/// - `judgement_window` (optional, default 40): preceding saliencies a point is scored against (`z`).
/// - `threshold` (optional, default 3): score above which a point is an anomaly (`τ`).
/// - `min_zscore` (optional, default 1.5): the point must also stand this many
///   standard deviations from the level of the window before it; `0` disables.
/// - `sensitivity` (optional, default 70): coverage in percent of the band around
///   the expected value.
/// - `batch_size` (optional): score in consecutive batches of this size (>= 12).
///
/// # Output JSON
///
/// ```json
/// { "points": [{ "index": 4, "value": 6.0, "saliency": 2.1, "score": 5.3,
///                "expected": 1.0, "lower": 0.9, "upper": 1.1, "is_anomaly": true,
///                "near_edge": false }],
///   "anomalies": [4] }
/// ```
///
/// `expected` is the low-frequency reconstruction of the series with its
/// anomalies replaced by their neighbours; `lower`/`upper` is the band
/// around it. The band is chart information — the anomaly decision is the
/// `score`.
#[wasm_bindgen(unchecked_return_type = "SpectralResidualDto")]
pub fn spectral_residual(
    #[wasm_bindgen(unchecked_param_type = "SpectralResidualInputDto")] input: JsValue,
) -> Result<JsValue, JsValue> {
    let input: SpectralResidualInputDto = from_js(input, "input")?;
    to_js(&crate::wire::spectral_residual_dto(input).map_err(js_err)?)
}

/// Seasonal-trend decomposition by Loess (STL; Cleveland et al. 1990):
/// `data = trend + seasonal + residual` for a known `period` (at least 2, with
/// at least two full cycles of data — [`estimate_period`] finds one).
///
/// # Input JSON
///
/// `{ data, period, seasonal?, trend?, low_pass?, seasonal_degree?, trend_degree?,
/// low_pass_degree?, robust?, inner_iterations?, outer_iterations? }`. The spans
/// are odd (`seasonal` at least 3, default 7; `trend` and `low_pass` above the
/// period, defaults as statsmodels' `STL`), the degrees 0 or 1 (default 1).
/// `robust` adds the outer loop that down-weights outliers; the iteration
/// counts default to 5 inner / 0 outer, or 2 / 15 when robust. An even span is
/// `invalid_option`.
///
/// # Output JSON
///
/// `{ trend, seasonal, residual, weights, trend_strength, seasonal_strength,
/// period, seasonal_span, trend_span, low_pass_span }`: the three components,
/// the last pass's robustness weights (all 1 without the outer loop), the
/// strengths `max(0, 1 − Var(R)/Var(C + R))` of trend and seasonality, and the
/// spans used. As statsmodels' `STL`.
#[wasm_bindgen(unchecked_return_type = "StlDto")]
pub fn stl(
    #[wasm_bindgen(unchecked_param_type = "StlInputDto")] input: JsValue,
) -> Result<JsValue, JsValue> {
    let input: StlInputDto = from_js(input, "input")?;
    to_js(&crate::wire::stl_dto(input).map_err(js_err)?)
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
    sigma_to_ppm_value(sigma).map_err(js_err)
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
    ppm_to_sigma_value(ppm).map_err(js_err)
}

// ---------------------------------------------------------------------------
// Weibull
// ---------------------------------------------------------------------------

/// Weibull maximum-likelihood fit of failure times (each finite and `> 0`,
/// at least 2): `{ shape, scale, log_likelihood, iterations }`.
///
/// `suspension_times` (optional) are right-censored units -- still running, or
/// removed unfailed, at that time. They enter the likelihood through their
/// survival term; leaving them out of a field data set biases the scale (and
/// every B-life and MTBF from it) low.
///
/// `options.confidence` (in (0, 1), default 0.95) sets `intervals`: two-sided
/// `{ lower, upper }` for `shape` and `scale`, normal on the log scale from
/// the observed Fisher information (censored units included).
#[wasm_bindgen(unchecked_return_type = "WeibullMleDto")]
pub fn weibull_mle(
    #[wasm_bindgen(unchecked_param_type = "number[] | Float64Array")] failure_times: JsValue,
    #[wasm_bindgen(unchecked_optional_param_type = "number[] | Float64Array | null")]
    suspension_times: Option<JsValue>,
    #[wasm_bindgen(unchecked_optional_param_type = "WeibullMleOptions | null")] options: Option<
        JsValue,
    >,
) -> Result<JsValue, JsValue> {
    let times = read_numbers(&failure_times, "failure_times").map_err(js_err)?;
    let suspensions = match suspension_times {
        Some(s) if !s.is_undefined() && !s.is_null() => {
            read_numbers(&s, "suspension_times").map_err(js_err)?
        }
        _ => Vec::new(),
    };
    let options = match options {
        Some(o) if !o.is_undefined() && !o.is_null() => from_js::<WeibullMleOptions>(o, "options")?,
        _ => WeibullMleOptions::default(),
    };
    to_js(
        &weibull_mle_dto(
            &times,
            &suspensions,
            options.confidence,
            options.method.as_deref(),
            &options.fractions_failed,
        )
        .map_err(js_err)?,
    )
}

/// The optional third argument of [`weibull_mle`].
#[derive(serde::Deserialize, Default)]
#[cfg_attr(test, derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
struct WeibullMleOptions {
    #[serde(default)]
    confidence: Option<f64>,
    #[serde(default)]
    #[cfg_attr(test, schemars(schema_with = "card_schema::interval_method"))]
    method: Option<String>,
    #[serde(default)]
    fractions_failed: Vec<f64>,
}

#[wasm_bindgen(typescript_custom_section)]
const WEIBULL_OPTIONS_TS: &'static str = r#"
export interface WeibullMleOptions {
    /** Confidence level of `intervals`, in (0, 1). Omitted or `null`: 0.95. */
    confidence?: number | null;
    /** `"wald"` (default) or `"likelihood_ratio"` -- the profile-likelihood interval. */
    method?: "wald" | "likelihood_ratio" | null;
    /** Fractions failed (each in (0, 1)) whose B-life and interval to report; 0.1 is B10. */
    fractions_failed?: number[];
}
"#;

/// Weibull median-rank-regression fit (Bernard's ranks): `{ shape, scale,
/// r_squared }`. Same failures and suspensions as [`weibull_mle`]: a
/// suspension moves the ranks of the failures after it (Johnson's adjusted
/// ranks).
#[wasm_bindgen(unchecked_return_type = "WeibullMrrDto")]
pub fn weibull_mrr(
    #[wasm_bindgen(unchecked_param_type = "number[] | Float64Array")] failure_times: JsValue,
    #[wasm_bindgen(unchecked_optional_param_type = "number[] | Float64Array | null")]
    suspension_times: Option<JsValue>,
) -> Result<JsValue, JsValue> {
    let times = read_numbers(&failure_times, "failure_times").map_err(js_err)?;
    let suspensions = match suspension_times {
        Some(s) if !s.is_undefined() && !s.is_null() => {
            read_numbers(&s, "suspension_times").map_err(js_err)?
        }
        _ => Vec::new(),
    };
    to_js(&weibull_mrr_dto(&times, &suspensions).map_err(js_err)?)
}

/// Reliability metrics of a Weibull: `{ shape, scale, times?, fractions_failed? }`
/// → `{ mtbf, reliability, hazard_rate, b_life }`, the arrays aligned with
/// `times` (R(t), h(t)) and `fractions_failed` (B-life; `0.1` is B10, and the
/// time to reliability `p` is the B-life at `1 − p`).
#[wasm_bindgen(unchecked_return_type = "WeibullReliabilityDto")]
pub fn weibull_reliability(
    #[wasm_bindgen(unchecked_param_type = "WeibullReliabilityInputDto")] input: JsValue,
) -> Result<JsValue, JsValue> {
    let input: WeibullReliabilityInputDto = from_js(input, "input")?;
    to_js(&weibull_reliability_dto(input).map_err(js_err)?)
}

// ── Wire-schema strictness tests ─────────────────────────────────────

#[cfg(test)]
mod dto_strictness_tests {
    use serde_json::json;

    fn assert_rejects_unknown<T: serde::de::DeserializeOwned>(v: serde_json::Value) {
        match super::from_json::<T>(v, "input") {
            Ok(_) => panic!("unknown key must be rejected"),
            Err(e) => {
                assert!(e.message.contains("unknown field"), "{e}");
                assert_eq!(e.code, crate::wire::code::MALFORMED_INPUT);
            }
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
    use crate::wire::{each_within, Bound, Detail};

    // --- hypothesis tests ---

    #[test]
    fn a_t_test_reports_what_the_crate_computes() {
        let data = [5.1, 4.9, 5.3, 5.0, 5.2];
        let dto = one_sample_t_dto(&data, 5.0).unwrap();
        let direct = crate::testing::one_sample_t_test(&data, 5.0).unwrap();
        assert_eq!(
            (dto.statistic, dto.df, dto.p_value),
            (direct.statistic, direct.df, direct.p_value)
        );
    }

    #[test]
    fn hypothesis_refusals_name_the_argument_and_row() {
        let e = paired_t_dto(&[1.0, 2.0, 3.0], &[1.0, 2.0]).unwrap_err();
        assert_eq!(
            (e.code, e.parameter.as_deref()),
            ("dimension_mismatch", Some("y"))
        );

        let e = one_way_anova_dto(&[vec![1.0, 2.0], vec![3.0]]).unwrap_err();
        assert_eq!(
            (e.code, e.index),
            (crate::wire::code::INSUFFICIENT_DATA.name(), Some(1))
        );

        let e = chi_squared_independence_dto(&[vec![1.0, 2.0], vec![3.0]]).unwrap_err();
        assert_eq!((e.code, e.index), ("dimension_mismatch", Some(1)));

        let e = one_sample_t_dto(&[2.0, 2.0, 2.0], 1.0).unwrap_err();
        assert_eq!(
            (e.code, e.parameter.as_deref()),
            (crate::wire::code::DEGENERATE_DATA.name(), Some("data"))
        );

        let e = chi_squared_gof_dto(&[3.0, 4.0], &[3.5, 0.0]).unwrap_err();
        assert_eq!(
            (e.code, e.index),
            (crate::wire::code::PARAMETER_OUT_OF_RANGE.name(), Some(1))
        );
    }

    // --- equivalence tests (TOST) ---

    const FILL: [f64; 10] = [
        99.8, 100.1, 100.3, 99.9, 100.0, 100.2, 99.7, 100.1, 100.0, 99.9,
    ];
    const OTHER: [f64; 9] = [100.4, 99.6, 100.2, 100.0, 99.9, 100.3, 99.8, 100.1, 100.0];

    #[test]
    fn a_tost_reports_what_the_crate_computes() {
        let dto = tost_one_sample_dto(&FILL, 100.0, -0.5, 0.5, Some(0.1)).unwrap();
        let direct = crate::equivalence::tost_one_sample(&FILL, 100.0, -0.5, 0.5, 0.1).unwrap();
        assert_eq!(
            (dto.estimate, dto.df, dto.p_lower, dto.p_upper, dto.p_value),
            (
                direct.estimate,
                direct.df,
                direct.p_lower,
                direct.p_upper,
                direct.p_value
            )
        );
        assert_eq!(
            (dto.ci_lower, dto.ci_upper, dto.equivalent),
            (direct.ci_lower, direct.ci_upper, direct.equivalent)
        );

        let paired = tost_paired_dto(&FILL[..9], &OTHER, -0.5, 0.5, None).unwrap();
        let direct = crate::equivalence::tost_paired(&FILL[..9], &OTHER, -0.5, 0.5, 0.05).unwrap();
        assert_eq!(
            (paired.estimate, paired.p_value),
            (direct.estimate, direct.p_value)
        );

        let two = tost_two_sample_dto(&FILL, &OTHER, -0.5, 0.5, None, None).unwrap();
        let direct =
            crate::equivalence::tost_two_sample(&FILL, &OTHER, -0.5, 0.5, 0.05, false).unwrap();
        assert_eq!((two.df, two.p_value), (direct.df, direct.p_value));
    }

    #[test]
    fn a_tost_without_a_level_runs_at_five_percent_and_welch() {
        let omitted = tost_two_sample_dto(&FILL, &OTHER, -0.5, 0.5, None, None).unwrap();
        let explicit =
            tost_two_sample_dto(&FILL, &OTHER, -0.5, 0.5, Some(0.05), Some(false)).unwrap();
        assert_eq!(
            (omitted.ci_lower, omitted.ci_upper, omitted.df),
            (explicit.ci_lower, explicit.ci_upper, explicit.df)
        );
        let pooled = tost_two_sample_dto(&FILL, &OTHER, -0.5, 0.5, None, Some(true)).unwrap();
        assert_eq!(pooled.df, 17.0);
        assert_ne!(pooled.df, omitted.df);
        // The level moves the interval.
        let wide = tost_one_sample_dto(&FILL, 100.0, -0.5, 0.5, Some(0.01)).unwrap();
        let narrow = tost_one_sample_dto(&FILL, 100.0, -0.5, 0.5, Some(0.2)).unwrap();
        assert!(wide.ci_upper - wide.ci_lower > narrow.ci_upper - narrow.ci_lower);
    }

    #[test]
    fn tost_refusals_name_the_argument() {
        let at = |e: crate::wire::WireError| {
            (e.code, e.parameter.as_deref().map(str::to_owned), e.index)
        };

        // Margins: reversed or equal is a refusal of a combination, named on `high`.
        for (low, high) in [(1.0, -1.0), (0.5, 0.5)] {
            let e = tost_one_sample_dto(&FILL, 100.0, low, high, None).unwrap_err();
            assert_eq!(
                at(e),
                ("invalid_option", Some("high".to_string()), None),
                "{low}..{high}"
            );
        }
        // Not finite: the argument, not a statistic.
        let e = tost_one_sample_dto(&FILL, 100.0, f64::NAN, 0.5, None).unwrap_err();
        assert_eq!(at(e), ("value_not_finite", Some("low".to_string()), None));
        let e = tost_paired_dto(&FILL, &FILL, -0.5, f64::INFINITY, None).unwrap_err();
        assert_eq!(at(e), ("value_not_finite", Some("high".to_string()), None));
        let e = tost_one_sample_dto(&FILL, f64::NAN, -0.5, 0.5, None).unwrap_err();
        assert_eq!(
            at(e),
            ("value_not_finite", Some("target".to_string()), None)
        );
        let e = tost_two_sample_dto(&FILL, &OTHER, -0.5, 0.5, Some(f64::NAN), None).unwrap_err();
        assert_eq!(at(e), ("value_not_finite", Some("alpha".to_string()), None));

        // The level: open at both ends, the bounds and the value sent.
        for bad in [0.0, 0.5, 0.7, -0.1, 1.0] {
            let e = tost_one_sample_dto(&FILL, 100.0, -0.5, 0.5, Some(bad)).unwrap_err();
            assert_eq!(
                (e.code, e.parameter.as_deref()),
                ("parameter_out_of_range", Some("alpha")),
                "{bad}"
            );
            assert_eq!(
                e.details,
                vec![
                    ("min", Detail::Null),
                    ("exclusive_min", Detail::Num(0.0)),
                    ("max", Detail::Null),
                    ("exclusive_max", Detail::Num(0.5)),
                    ("got", Detail::Num(bad)),
                ]
            );
        }

        // Too few values, mismatched pairs, no variation.
        let e = tost_one_sample_dto(&[1.0], 0.0, -1.0, 1.0, None).unwrap_err();
        assert_eq!(
            (e.code, e.parameter.as_deref(), e.details.clone()),
            (
                "insufficient_data",
                Some("data"),
                vec![("min", Detail::Num(2.0)), ("got", Detail::Num(1.0))]
            )
        );
        let e = tost_two_sample_dto(&FILL, &[1.0], -1.0, 1.0, None, None).unwrap_err();
        assert_eq!(
            (e.code, e.parameter.as_deref()),
            ("insufficient_data", Some("b"))
        );
        let e = tost_paired_dto(&FILL, &OTHER, -1.0, 1.0, None).unwrap_err();
        assert_eq!(
            (e.code, e.parameter.as_deref()),
            ("dimension_mismatch", Some("y"))
        );
        let e = tost_one_sample_dto(&[2.0, 2.0, 2.0], 2.0, -1.0, 1.0, None).unwrap_err();
        assert_eq!(
            (e.code, e.parameter.as_deref()),
            ("degenerate_data", Some("data"))
        );
        let e = tost_paired_dto(&[1.0, 2.0, 3.0], &[0.5, 1.5, 2.5], -1.0, 1.0, None).unwrap_err();
        assert_eq!(
            (e.code, e.parameter.as_deref()),
            ("degenerate_data", Some("y"))
        );
        let e = tost_two_sample_dto(&[1.0, 1.0], &[2.0, 2.0], -2.0, 2.0, None, None).unwrap_err();
        assert_eq!(e.code, "degenerate_data");
        // One constant sample is still a test.
        assert!(tost_two_sample_dto(&[1.0, 1.0], &[2.0, 2.5, 3.0], -3.0, 3.0, None, None).is_ok());
    }

    // --- tolerance intervals ---

    #[test]
    fn a_tolerance_interval_reports_what_the_crate_computes() {
        use crate::tolerance::{tolerance_factor, tolerance_interval_normal, Sided};
        for (name, sided) in [
            ("two_sided", Sided::TwoSided),
            ("one_sided", Sided::OneSided),
        ] {
            let k = tolerance_factor_value(Whole(10.0), 0.95, 0.9, name).unwrap();
            assert_eq!(Some(k), tolerance_factor(10, 0.95, 0.9, sided));

            let dto = tolerance_interval_dto(&FILL, 0.95, 0.9, name).unwrap();
            let direct = tolerance_interval_normal(&FILL, 0.95, 0.9, sided).unwrap();
            assert_eq!(
                (dto.mean, dto.std_dev, dto.n, dto.k, dto.lower, dto.upper),
                (
                    direct.mean,
                    direct.std_dev,
                    direct.n,
                    direct.k,
                    direct.lower,
                    direct.upper
                )
            );
            // The factor of the interval is the factor of its sample size.
            assert_eq!(
                dto.k,
                tolerance_factor_value(Whole(FILL.len() as f64), 0.95, 0.9, name).unwrap()
            );
        }
        // The published factor: 95 % of the population, 95 % confidence, n = 10.
        let k = tolerance_factor_value(Whole(10.0), 0.95, 0.95, "two_sided").unwrap();
        assert!((k - 3.393429478712608).abs() < 1e-11);
        // The same whole number read from a float.
        assert_eq!(
            super::from_json::<Whole>(serde_json::json!(10.0), "n")
                .unwrap()
                .0,
            10.0
        );
    }

    #[test]
    fn tolerance_refusals_name_the_argument() {
        let at = |e: crate::wire::WireError| (e.code, e.parameter.as_deref().map(str::to_owned));

        // A count: a fraction is a shape error where it sits, a number outside
        // the range is a range error -- a negative one the same as a small one.
        let e = super::from_json::<Whole>(serde_json::json!(2.5), "n").unwrap_err();
        assert_eq!(at(e), ("malformed_input", Some("n".to_string())));
        let e = super::from_json::<Whole>(serde_json::json!(null), "n").unwrap_err();
        assert_eq!(at(e), ("malformed_input", Some("n".to_string())));
        for got in [1.0, 0.0, -1.0] {
            let e = tolerance_factor_value(Whole(got), 0.95, 0.95, "two_sided").unwrap_err();
            assert_eq!(
                (e.code, e.parameter.as_deref()),
                ("parameter_out_of_range", Some("n")),
                "{got}"
            );
            assert_eq!(
                e.details,
                vec![
                    ("min", Detail::Num(2.0)),
                    ("max", Detail::Null),
                    ("got", Detail::Num(got)),
                ]
            );
        }
        assert!(tolerance_factor_value(Whole(2.0), 0.95, 0.95, "one_sided").is_ok());

        // Proportions: open at both ends, the bounds and the value sent.
        for (parameter, coverage, confidence, bad) in [
            ("coverage", 0.0, 0.95, 0.0),
            ("coverage", 1.0, 0.95, 1.0),
            ("coverage", 1.5, 0.95, 1.5),
            ("confidence", 0.95, 0.0, 0.0),
            ("confidence", 0.95, 1.0, 1.0),
            ("confidence", 0.95, -0.1, -0.1),
        ] {
            let e =
                tolerance_factor_value(Whole(10.0), coverage, confidence, "two_sided").unwrap_err();
            assert_eq!(
                (e.code, e.parameter.as_deref()),
                ("parameter_out_of_range", Some(parameter))
            );
            assert_eq!(
                e.details,
                vec![
                    ("min", Detail::Null),
                    ("exclusive_min", Detail::Num(0.0)),
                    ("max", Detail::Null),
                    ("exclusive_max", Detail::Num(1.0)),
                    ("got", Detail::Num(bad)),
                ]
            );
            let e = tolerance_interval_dto(&FILL, coverage, confidence, "one_sided").unwrap_err();
            assert_eq!(
                (e.code, e.parameter.as_deref()),
                ("parameter_out_of_range", Some(parameter))
            );
        }
        let e = tolerance_factor_value(Whole(10.0), f64::NAN, 0.95, "two_sided").unwrap_err();
        assert_eq!(at(e), ("value_not_finite", Some("coverage".to_string())));
        let e = tolerance_interval_dto(&FILL, 0.95, f64::INFINITY, "two_sided").unwrap_err();
        assert_eq!(at(e), ("value_not_finite", Some("confidence".to_string())));

        // A name nobody knows is refused, never read as a default.
        for bad in ["both", "", "TwoSided", "two-sided"] {
            let e = tolerance_factor_value(Whole(10.0), 0.95, 0.95, bad).unwrap_err();
            assert_eq!(
                (e.code, e.parameter.as_deref()),
                ("unknown_option", Some("sided")),
                "{bad:?}"
            );
            assert_eq!(
                e.details,
                vec![
                    ("got", Detail::Text(bad.to_string())),
                    ("expected", Detail::List(vec!["two_sided", "one_sided"])),
                ]
            );
            let e = tolerance_interval_dto(&FILL, 0.95, 0.95, bad).unwrap_err();
            assert_eq!(
                (e.code, e.parameter.as_deref()),
                ("unknown_option", Some("sided"))
            );
        }

        // Too few values, no variation, a spread no double holds.
        let e = tolerance_interval_dto(&[1.0], 0.95, 0.95, "two_sided").unwrap_err();
        assert_eq!(
            (e.code, e.parameter.as_deref(), e.details.clone()),
            (
                "insufficient_data",
                Some("data"),
                vec![("min", Detail::Num(2.0)), ("got", Detail::Num(1.0))]
            )
        );
        let e = tolerance_interval_dto(&[2.0, 2.0, 2.0], 0.95, 0.95, "two_sided").unwrap_err();
        assert_eq!(at(e), ("degenerate_data", Some("data".to_string())));
        let e =
            tolerance_interval_dto(&[-f64::MAX, f64::MAX], 0.95, 0.95, "two_sided").unwrap_err();
        assert_eq!(e.code, "internal");
        // The data are checked before the settings, in the order of the arguments.
        let e = tolerance_interval_dto(&[1.0], 2.0, 0.95, "x").unwrap_err();
        assert_eq!(e.code, "insufficient_data");
    }

    // --- Type 1 gage study ---

    const READINGS: [f64; 8] = [10.02, 9.98, 10.05, 10.01, 9.97, 10.03, 10.0, 10.04];

    #[test]
    fn a_type1_study_reports_what_the_crate_computes() {
        use crate::msa::type1::{type1_gage_study, Type1Options};
        let dto = type1_gage_study_dto(&READINGS, 10.0, 9.0, 11.0, None, None).unwrap();
        let direct =
            type1_gage_study(&READINGS, 10.0, 9.0, 11.0, &Type1Options::default()).unwrap();
        assert_eq!(
            (dto.n, dto.mean, dto.std_dev, dto.bias, dto.study_variation),
            (
                direct.n,
                direct.mean,
                direct.std_dev,
                direct.bias,
                direct.study_variation
            )
        );
        assert_eq!(
            (dto.cg, dto.cgk, dto.t_statistic, dto.df, dto.p_value),
            (
                direct.cg,
                direct.cgk,
                direct.t_statistic,
                direct.df,
                direct.p_value
            )
        );
        assert_eq!(
            (
                dto.percent_var_repeatability,
                dto.percent_var_repeatability_and_bias
            ),
            (
                direct.percent_var_repeatability,
                direct.percent_var_repeatability_and_bias
            )
        );
    }

    #[test]
    fn a_type1_study_without_options_runs_at_twenty_percent_and_six_sigma() {
        let omitted = type1_gage_study_dto(&READINGS, 10.0, 9.0, 11.0, None, None).unwrap();
        let explicit =
            type1_gage_study_dto(&READINGS, 10.0, 9.0, 11.0, Some(0.2), Some(6.0)).unwrap();
        assert_eq!((omitted.cg, omitted.cgk), (explicit.cg, explicit.cgk));
        // Another edition is a configuration: K = 15 %, L = 4.
        let other =
            type1_gage_study_dto(&READINGS, 10.0, 9.0, 11.0, Some(0.15), Some(4.0)).unwrap();
        assert!((other.cg - omitted.cg * (0.15 / 0.2) * (6.0 / 4.0)).abs() < 1e-12);
        assert_ne!(other.cgk, omitted.cgk);
    }

    #[test]
    fn a_type1_study_whose_bias_fills_the_band_has_no_repeatability_and_bias_percentage() {
        let dto = type1_gage_study_dto(&READINGS, 9.5, 9.0, 11.0, None, None).unwrap();
        assert!(dto.cgk < 0.0);
        assert_eq!(dto.percent_var_repeatability_and_bias, None);
    }

    #[test]
    fn type1_refusals_name_the_argument() {
        let at = |e: crate::wire::WireError| {
            (e.code, e.parameter.as_deref().map(str::to_owned), e.index)
        };
        let study = |m: &[f64], r, lo, hi, k, l| type1_gage_study_dto(m, r, lo, hi, k, l);

        let e = study(&[1.0], 1.0, 0.0, 2.0, None, None).unwrap_err();
        assert_eq!(
            (e.code, e.parameter.as_deref(), e.details.clone()),
            (
                "insufficient_data",
                Some("measurements"),
                vec![("min", Detail::Num(2.0)), ("got", Detail::Num(1.0))]
            )
        );
        let e = study(&[1.0, f64::NAN, 2.0], 1.0, 0.0, 2.0, None, None).unwrap_err();
        assert_eq!(
            at(e),
            (
                "value_not_finite",
                Some("measurements".to_string()),
                Some(1)
            )
        );
        for (name, e) in [
            (
                "reference",
                study(&READINGS, f64::NAN, 9.0, 11.0, None, None),
            ),
            (
                "lsl",
                study(&READINGS, 10.0, f64::NEG_INFINITY, 11.0, None, None),
            ),
            (
                "usl",
                study(&READINGS, 10.0, 9.0, f64::INFINITY, None, None),
            ),
            (
                "tolerance_fraction",
                study(&READINGS, 10.0, 9.0, 11.0, Some(f64::NAN), None),
            ),
            (
                "sigma_multiple",
                study(&READINGS, 10.0, 9.0, 11.0, None, Some(f64::INFINITY)),
            ),
        ] {
            assert_eq!(
                at(e.unwrap_err()),
                ("value_not_finite", Some(name.to_string()), None)
            );
        }
        // Limits: reversed or equal is a refusal of a combination, named on `usl`.
        for (lsl, usl) in [(11.0, 9.0), (10.0, 10.0)] {
            let e = study(&READINGS, 10.0, lsl, usl, None, None).unwrap_err();
            assert_eq!(at(e), ("invalid_option", Some("usl".to_string()), None));
        }
        // The fraction: open at 0, closed at 1.
        for bad in [0.0, -0.2, 1.5, 20.0] {
            let e = study(&READINGS, 10.0, 9.0, 11.0, Some(bad), None).unwrap_err();
            assert_eq!(
                (e.code, e.parameter.as_deref()),
                ("parameter_out_of_range", Some("tolerance_fraction")),
                "{bad}"
            );
            assert_eq!(
                e.details,
                vec![
                    ("min", Detail::Null),
                    ("exclusive_min", Detail::Num(0.0)),
                    ("max", Detail::Num(1.0)),
                    ("got", Detail::Num(bad)),
                ]
            );
        }
        assert!(study(&READINGS, 10.0, 9.0, 11.0, Some(1.0), None).is_ok());
        // The multiple: above 0, no upper bound.
        for bad in [0.0, -6.0] {
            let e = study(&READINGS, 10.0, 9.0, 11.0, None, Some(bad)).unwrap_err();
            assert_eq!(
                (e.code, e.parameter.as_deref()),
                ("parameter_out_of_range", Some("sigma_multiple")),
                "{bad}"
            );
            assert_eq!(
                e.details,
                vec![
                    ("min", Detail::Null),
                    ("exclusive_min", Detail::Num(0.0)),
                    ("max", Detail::Null),
                    ("got", Detail::Num(bad)),
                ]
            );
        }
        // No variation, including a constant that does not sum exactly.
        for constant in [[2.0, 2.0, 2.0], [0.1, 0.1, 0.1]] {
            let e = study(&constant, constant[0], 0.0, 4.0, None, None).unwrap_err();
            assert_eq!(
                at(e),
                ("degenerate_data", Some("measurements".to_string()), None)
            );
        }
        // A spread so small that an index is not a number a double holds.
        let e = study(&[1.0, 1.0 + f64::EPSILON], 1.0, 0.0, f64::MAX, None, None).unwrap_err();
        assert_eq!(e.code, "degenerate_data");
        // Arguments are checked in order: the readings before the limits.
        let e = study(&[1.0], f64::NAN, 2.0, 1.0, Some(0.0), Some(0.0)).unwrap_err();
        assert_eq!(e.code, "insufficient_data");
    }

    #[test]
    fn extreme_settings_are_answered_or_refused_never_null_or_hung() {
        // The factor tends to the normal quantile as `n` grows; at the edge of
        // (0, 1) it may not be found. Either way the answer is a finite number
        // or a refusal, never `null` and never a loop.
        let started = std::time::Instant::now();
        let k = tolerance_factor_value(Whole(1e300), 0.95, 0.95, "two_sided").unwrap();
        assert!((k - 1.959963984540054).abs() < 1e-6, "{k}");
        for (n, coverage, confidence) in [
            (2.0, 1.0 - 1e-12, 1.0 - 1e-12),
            (2.0, 1.0 - 1e-15, 0.5),
            (2.0, f64::MIN_POSITIVE, 0.5),
        ] {
            for sided in ["two_sided", "one_sided"] {
                match tolerance_factor_value(Whole(n), coverage, confidence, sided) {
                    Ok(k) => assert!(k.is_finite(), "{n} {coverage} {confidence} {sided}"),
                    Err(e) => assert_eq!(e.code, "internal", "{n} {coverage} {confidence}"),
                }
            }
        }
        assert!(started.elapsed().as_secs() < 60, "{:?}", started.elapsed());
    }

    #[test]
    fn fisher_reads_a_two_by_two_table_of_whole_counts() {
        let ok = fisher_exact_dto(&serde_json::json!([[3, 1], [1, 3]])).unwrap();
        assert!(ok.p_value > 0.0 && ok.p_value <= 1.0);
        let e = fisher_exact_dto(&serde_json::json!([[3, 1.5], [1, 3]])).unwrap_err();
        assert_eq!(
            (e.code, e.index),
            (crate::wire::code::COUNT_NOT_WHOLE.name(), Some(1))
        );
        let e = fisher_exact_dto(&serde_json::json!([[3, 1, 2], [1, 3, 0]])).unwrap_err();
        assert_eq!(e.code, "dimension_mismatch");
    }

    // --- number arrays are read as sent ---

    #[test]
    fn null_in_a_number_array_is_refused_at_its_index() {
        let e = numbers_from(
            "data",
            [
                Element::Number(1.0),
                Element::Other("null".into()),
                Element::Number(3.0),
            ],
        )
        .unwrap_err();
        assert_eq!(e.code, crate::wire::code::MALFORMED_INPUT);
        assert_eq!(e.index, Some(1));
        assert_eq!(e.parameter.as_deref(), Some("data"));
    }

    #[test]
    fn a_non_finite_element_is_value_not_finite() {
        let e = numbers_from("gaps", [2.0, f64::NAN].map(Element::Number)).unwrap_err();
        assert_eq!(e.code, crate::wire::code::VALUE_NOT_FINITE);
        assert_eq!(e.index, Some(1));
    }

    #[test]
    fn rare_event_charts_need_three_values_and_say_so() {
        let e = crate::wire::at_least(vec![1.0, 2.0], 3, "times").unwrap_err();
        assert_eq!(e.code, crate::wire::code::INSUFFICIENT_DATA);
        assert!(e.message.contains("at least 3"), "{}", e.message);
        // The binding's minimum is the chart's: two values make no chart.
        assert!(crate::spc::t_chart(&[1.0, 2.0]).is_none());
        assert!(crate::spc::t_chart(&[1.0, 2.0, 3.0]).is_some());
        assert!(crate::spc::g_chart(&[1.0, 2.0]).is_none());
    }

    #[test]
    fn a_value_outside_the_domain_is_named_where_it_sits() {
        let e = each_within(
            &[1.0, 0.0, 2.0],
            "times",
            Bound::Open(0.0),
            Bound::Unbounded,
        )
        .unwrap_err();
        assert_eq!(e.code, crate::wire::code::PARAMETER_OUT_OF_RANGE);
        assert_eq!(e.index, Some(1));
        assert_eq!(e.parameter.as_deref(), Some("times"));
        // `> 0`: the open end is `exclusive_min`; `min` and `max` are null.
        assert_eq!(
            e.details,
            vec![
                ("min", Detail::Null),
                ("exclusive_min", Detail::Num(0.0)),
                ("max", Detail::Null),
                ("got", Detail::Num(0.0)),
            ]
        );
        assert_eq!(e.message, "times[1]: must be > 0, got 0");
        assert!(each_within(&[0.0, 2.0], "gaps", Bound::Closed(0.0), Bound::Unbounded).is_ok());
    }

    // --- rules option on xbar_r_chart ---

    /// Every way of not asking for a rule set means the 3-sigma test alone.
    #[test]
    fn rules_option_absent_means_the_shewhart_set() {
        use crate::spc::{RuleSet, ViolationType};
        for absent in [
            None,
            Some(serde_json::Value::Null),
            Some(serde_json::json!({})),
            Some(serde_json::json!({ "rules": null })),
        ] {
            let set = rules_from_json(absent).unwrap();
            assert_eq!(set, RuleSet::shewhart());
            assert_eq!(
                set.iter().collect::<Vec<_>>(),
                [ViolationType::BeyondLimits]
            );
        }
    }

    /// A set's name selects exactly that set; Western Electric's run rule is
    /// eight points, not Nelson's nine.
    #[test]
    fn rules_option_names_a_set() {
        use crate::spc::{RuleSet, ViolationType};
        for (name, set) in [
            ("shewhart", RuleSet::shewhart()),
            ("western_electric", RuleSet::western_electric()),
            ("nelson", RuleSet::nelson()),
        ] {
            assert_eq!(
                rules_from_json(Some(serde_json::json!({ "rules": name }))).unwrap(),
                set
            );
        }
        let we = RuleSet::western_electric();
        assert!(we.contains(ViolationType::EightOneSide));
        assert!(!we.contains(ViolationType::NineOneSide));
        assert_eq!(we.len(), 4);
        assert_eq!(RuleSet::nelson().len(), 8);
        assert!(!RuleSet::nelson().contains(ViolationType::EightOneSide));
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
        assert_eq!(
            (err.code, err.parameter.as_deref()),
            ("unknown_option", Some("rules"))
        );
        assert!(err.message.contains("Rule1"), "{err}");
        assert_eq!(
            err.details[0],
            ("got", crate::wire::Detail::Text("Rule1".into()))
        );
        let rule_names: Vec<&str> = crate::spc::ViolationType::ALL
            .iter()
            .map(|v| v.name())
            .collect();
        assert_eq!(
            err.details[1],
            ("expected", crate::wire::Detail::List(rule_names))
        );
        // A string names a set, not a test.
        let err =
            rules_from_json(Some(serde_json::json!({ "rules": "BeyondLimits" }))).unwrap_err();
        assert_eq!(
            (err.code, err.parameter.as_deref()),
            ("unknown_option", Some("rules"))
        );
        assert_eq!(
            err.details[1],
            (
                "expected",
                crate::wire::Detail::List(crate::spc::RuleSet::NAMES.to_vec())
            )
        );
        let err = rules_from_json(Some(serde_json::json!({ "rules": 3 }))).unwrap_err();
        assert_eq!(err.code, "malformed_input");
        assert!(err.message.contains("array"), "{err}");
    }

    use super::{capability_dto, from_json, CapabilityInputDto};
    use serde_json::json;

    /// A reported fixture: six subgroups of five, with real
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
        capability_dto(from_json::<CapabilityInputDto>(v, "input").map_err(|e| e.message)?)
            .map_err(|e| format!("{} [{}]", e.message, e.parameter.as_deref().unwrap_or("-")))
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
            assert!(e.ends_with("[sigma_within]"), "{bad} -> {e}");
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
                sigma_within_df: None,
                confidence: None,
            };
            let e = capability_dto(input).expect_err("non-finite sigma must be refused");
            assert_eq!(
                (e.code, e.parameter.as_deref()),
                ("parameter_out_of_range", Some("sigma_within")),
                "{bad} -> {e}"
            );
        }
    }

    /// `intervals` brackets each index; Pp/Ppk always (n - 1 df), Cp/Cpk only
    /// with the within sigma's degrees of freedom; the default level is 95 %.
    #[test]
    fn capability_intervals_on_the_wire() {
        let sigma = sigma_within_from_chart();
        let overall = dto(json!({ "data": flat(), "usl": 11.0, "lsl": 9.0 })).unwrap();
        let ci = &overall.intervals;
        assert_eq!(ci.confidence, 0.95);
        let (pp, ppk) = (ci.pp.as_ref().unwrap(), ci.ppk.as_ref().unwrap());
        assert!(pp.lower < overall.pp.unwrap() && overall.pp.unwrap() < pp.upper);
        assert!(ppk.lower < overall.ppk.unwrap() && overall.ppk.unwrap() < ppk.upper);
        assert!(ci.cp.is_none() && ci.cpk.is_none());

        let n = flat().len() as f64;
        let within = dto(json!({
            "data": flat(), "usl": 11.0, "lsl": 9.0,
            "sigma_within": sigma, "sigma_within_df": n - 5.0, "confidence": 0.90
        }))
        .unwrap();
        let ci = &within.intervals;
        assert_eq!(ci.confidence, 0.90);
        let cpk = ci.cpk.as_ref().unwrap();
        assert!(cpk.lower < within.cpk.unwrap() && within.cpk.unwrap() < cpk.upper);
        // The wire carries exactly the core's numbers.
        let core = crate::capability::ProcessCapability::new(Some(11.0), Some(9.0))
            .unwrap()
            .compute(&flat(), sigma)
            .unwrap()
            .intervals(0.90, Some(n - 5.0))
            .unwrap();
        let core_cpk = core.cpk.unwrap();
        assert_eq!((cpk.lower, cpk.upper), (core_cpk.lower, core_cpk.upper));
        // A narrower level, a narrower interval.
        assert!(
            within.intervals.ppk.as_ref().unwrap().upper
                < overall.intervals.ppk.as_ref().unwrap().upper
        );
    }

    /// The chart hands over both halves of what the short-term intervals need.
    #[test]
    fn chart_sigma_and_its_df_compose_into_cp_cpk_intervals() {
        let chart = xbar_r_dto(
            SUBGROUPS.iter().map(|g| g.to_vec()).collect(),
            crate::spc::RuleSet::default(),
            None,
        )
        .expect("valid");
        let cap = dto(json!({
            "data": flat(), "usl": 11.0, "lsl": 9.0,
            "sigma_within": chart.sigma_hat, "sigma_within_df": chart.sigma_hat_df
        }))
        .unwrap();
        let (cp, cpk) = (
            cap.intervals.cp.as_ref().unwrap(),
            cap.intervals.cpk.as_ref().unwrap(),
        );
        assert!(cp.lower < cap.cp.unwrap() && cap.cp.unwrap() < cp.upper);
        assert!(cpk.lower < cap.cpk.unwrap() && cap.cpk.unwrap() < cpk.upper);
        // The range spends degrees of freedom: fewer than m (n - 1).
        let pooled = (SUBGROUPS.len() * (SUBGROUPS[0].len() - 1)) as f64;
        let df = chart.sigma_hat_df.expect("estimated from the subgroups");
        assert!(df > 0.0 && df < pooled);
    }

    #[test]
    fn capability_interval_options_are_checked() {
        for bad in [0.0, 1.0, -0.1, 1.5] {
            let e = capability_dto(
                from_json::<CapabilityInputDto>(
                    json!({ "data": flat(), "usl": 11.0, "confidence": bad }),
                    "input",
                )
                .unwrap(),
            )
            .expect_err("confidence outside (0, 1)");
            assert_eq!(
                (e.code, e.parameter.as_deref()),
                ("parameter_out_of_range", Some("confidence")),
                "{bad}"
            );
        }
        let e = capability_dto(
            from_json::<CapabilityInputDto>(
                json!({ "data": flat(), "usl": 11.0, "sigma_within": 0.2, "sigma_within_df": 0.0 }),
                "input",
            )
            .unwrap(),
        )
        .expect_err("df must be positive");
        assert_eq!(
            (e.code, e.parameter.as_deref()),
            ("parameter_out_of_range", Some("sigma_within_df"))
        );
        let e = capability_dto(
            from_json::<CapabilityInputDto>(
                json!({ "data": flat(), "usl": 11.0, "sigma_within_df": 20.0 }),
                "input",
            )
            .unwrap(),
        )
        .expect_err("df without the sigma it belongs to");
        assert_eq!(
            (e.code, e.parameter.as_deref()),
            ("invalid_option", Some("sigma_within_df"))
        );
    }

    #[test]
    fn too_little_data_is_an_error_not_a_nan() {
        let e = dto(json!({ "data": [1.0], "usl": 11.0, "lsl": 9.0 }))
            .expect_err("one point has no dispersion");
        // insufficient_data on `data`, with min 2 and got 1
        assert!(
            e.contains("at least 2 values") && e.ends_with("[data]"),
            "{e}"
        );
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
        let d = xbar_s_dto(subgroups.clone(), RuleSet::default(), None).expect("valid input");

        let mut chart = XBarSChart::new(5).expect("5 is in range");
        for g in &subgroups {
            chart.add_sample(g).unwrap();
        }
        let x = chart.control_limits().expect("limits");
        let s = chart.s_limits().expect("limits");
        assert_eq!((d.xbar_ucl, d.xbar_cl, d.xbar_lcl), (x.ucl, x.cl, x.lcl));
        assert_eq!((d.s_ucl, d.s_cl, d.s_lcl), (s.ucl, s.cl, s.lcl));
        assert_eq!(Some(d.sigma_hat), chart.sigma_hat());
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
            xbar_s_dto(subgroups.clone(), RuleSet::default(), None).map(|_| ()),
            xbar_r_dto(subgroups, RuleSet::default(), None).map(|_| ()),
        ] {
            let e = result.expect_err("ragged subgroup");
            assert!(e.message.contains("subgroups[7]"), "{e}");
            assert_eq!(
                (e.code, e.index, e.parameter.as_deref()),
                (
                    crate::wire::code::DIMENSION_MISMATCH.name(),
                    Some(7),
                    Some("subgroups")
                )
            );
            assert_eq!(
                e.details,
                vec![("expected", Detail::Num(5.0)), ("got", Detail::Num(4.0))]
            );
        }
    }

    #[test]
    fn imr_chart_reports_the_crate_chart() {
        use crate::spc::{ControlChart, IndividualMRChart, RuleSet};
        let values: Vec<f64> = shifted_subgroups().iter().map(|g| g[0]).collect();
        let d = imr_dto(values.clone(), RuleSet::default(), None).expect("valid input");

        let mut chart = IndividualMRChart::new();
        for &x in &values {
            chart.add_sample(&[x]).unwrap();
        }
        let i = chart.control_limits().expect("limits");
        let mr = chart.mr_limits().expect("limits");
        assert_eq!((d.i_ucl, d.i_cl, d.i_lcl), (i.ucl, i.cl, i.lcl));
        assert_eq!((d.mr_ucl, d.mr_cl, d.mr_lcl), (mr.ucl, mr.cl, mr.lcl));
        assert_eq!(Some(d.sigma_hat), chart.sigma_hat());
        assert_eq!(d.i_points.len(), values.len());
        // MR_0 is undefined: the moving-range series starts at the second value.
        assert_eq!(d.mr_points.len(), values.len() - 1);
        assert_eq!(d.mr_points[0].index, 1);
    }

    #[test]
    fn imr_chart_needs_two_values() {
        use crate::spc::RuleSet;
        let e = imr_dto(vec![1.0], RuleSet::default(), None).expect_err("one value has no range");
        assert!(e.message.contains("two"), "{e}");
    }

    /// The standalone engine must be the engine the charts use: fed a chart's
    /// own points and limits, it has to find exactly what the chart found.
    #[test]
    fn run_rules_finds_what_the_chart_found() {
        use crate::spc::RuleSet;
        let chart = xbar_r_dto(shifted_subgroups(), RuleSet::default(), None).expect("valid");
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
        assert!(e.message.contains("lcl <= cl <= ucl"), "{e}");
        assert_eq!(e.code, "parameter_out_of_range");
    }

    #[test]
    fn limits_input_rejects_unknown_keys() {
        let e = from_json::<LimitsInputDto>(
            json!({ "ucl": 1.0, "cl": 0.0, "lcl": -1.0, "usl": 2.0 }),
            "limits",
        )
        .err()
        .expect("unknown key");
        assert!(e.message.contains("unknown field"), "{e}");
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
        use crate::wire::code;
        let e = np_chart_dto(&[5, 101, 3], 100).expect_err("more defectives than inspected");
        assert!(e.message.contains("defectives[1]"), "{e}");
        assert_eq!(
            (e.code, e.index),
            (code::DEFECTIVES_EXCEED_SAMPLE.name(), Some(1))
        );
        // A zero sample size is a value, not a panic that would trap the module.
        let e = np_chart_dto(&[0, 0], 0).expect_err("nothing inspected");
        assert!(e.message.contains("sample size"), "{e}");
        assert_eq!(
            (e.code, e.index),
            (code::SAMPLE_SIZE_NOT_WHOLE.name(), None)
        );
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
        let d = u_chart_dto(&samples, &Default::default()).expect("valid");
        let mut chart = UChart::new();
        for &(x, u) in &samples {
            chart.add_sample(x, u).unwrap();
        }
        assert_eq!(Some(d.u_bar), chart.u_bar());
        assert_eq!(d.points.len(), samples.len());

        let e = u_chart_dto(&[(3, 1.0), (5, 0.0), (2, 0.8)], &Default::default())
            .expect_err("zero units");
        assert!(e.message.contains("samples[1]"), "{e}");
        assert_eq!(
            (e.code, e.index),
            (crate::wire::code::UNITS_NOT_POSITIVE.name(), Some(1))
        );
    }

    #[test]
    fn laney_u_chart_reports_the_crate_chart() {
        let samples = [(3, 1.0), (5, 1.5), (2, 0.8), (4, 1.2), (6, 1.1)];
        let d = laney_u_dto(&samples, &Default::default()).expect("valid");
        let expected = crate::spc::laney_u_chart(&samples, None).expect("valid");
        assert_eq!((d.u_bar, d.phi), (expected.u_bar, expected.phi));
        let e = laney_u_dto(&samples[..2], &Default::default()).expect_err("fewer than 3 samples");
        assert_eq!(
            (e.code, e.index),
            (crate::wire::code::INSUFFICIENT_DATA.name(), None)
        );
        let e = laney_u_dto(&[(3, 1.0), (5, -1.0), (2, 0.8)], &Default::default())
            .expect_err("negative units");
        assert!(e.message.contains("samples[1]"), "{e}");
        assert_eq!(
            (e.code, e.index),
            (crate::wire::code::UNITS_NOT_POSITIVE.name(), Some(1))
        );
    }

    /// `p_chart` used to hand every pair to the chart, which dropped the ones
    /// it could not use -- shifting the index of every later point.
    #[test]
    fn p_chart_refuses_a_sample_with_no_proportion() {
        use crate::wire::code;
        let e = p_chart_dto(&[(3, 100), (12, 10), (4, 100)], &Default::default())
            .expect_err("12 of 10");
        assert!(e.message.contains("samples[1]"), "{e}");
        assert_eq!(
            (e.code, e.index),
            (code::DEFECTIVES_EXCEED_SAMPLE.name(), Some(1))
        );
        let e = p_chart_dto(&[(3, 100), (0, 0)], &Default::default()).expect_err("0 of 0");
        assert!(e.message.contains("samples[1]"), "{e}");
        assert_eq!(
            (e.code, e.index),
            (code::SAMPLE_SIZE_NOT_WHOLE.name(), Some(1))
        );
        let ok = p_chart_dto(&[(3, 100), (5, 120), (2, 80)], &Default::default()).expect("valid");
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

        let native = crate::capability::boxcox_capability(
            &data,
            Some(100.0),
            Some(1.0),
            crate::capability::DEFAULT_LAMBDA_RANGE,
        )
        .expect("same call");
        let native_i = native.indices.expect("limits given");
        assert!((dto.lambda - native.lambda).abs() < 1e-12);
        assert_eq!(dto.lambda_at_bound, native.lambda_at_bound);
        assert_eq!(dto.pp, native_i.pp);
        assert_eq!(dto.ppk, native_i.ppk);

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
            json!({ "data": [1.0, 2.0, 3.0], "usl": 9.0 }),      // fewer than four points
            json!({ "data": [1.0, 2.0, 3.0, 4.0], "lambda_range": [2.0, -2.0] }), // empty range
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
    fn boxcox_without_limits_and_with_a_narrow_range_reports_the_bound() {
        // Normal quantiles through the inverse transform at lambda = 4: the
        // likelihood peaks near 4, so a [-2, 2] range stops at its upper end.
        let n = 100;
        let data: Vec<f64> = (1..=n)
            .map(|i| {
                let p = (i as f64 - 0.5) / n as f64;
                let z = 10.0 + 2.0 * u_numflow::special::inverse_normal_cdf(p);
                (4.0 * z + 1.0).powf(0.25)
            })
            .collect();
        let narrow: super::BoxcoxCapabilityInputDto = super::from_json(
            json!({ "data": data, "lambda_range": [-2.0, 2.0] }),
            "input",
        )
        .expect("valid input");
        let dto = super::boxcox_capability_dto(narrow).expect("no limits is allowed");
        assert!(dto.lambda_at_bound);
        assert_eq!(dto.lambda, 2.0);
        assert!(
            dto.pp.is_none() && dto.ppk.is_none(),
            "no limits, no indices"
        );

        let default: super::BoxcoxCapabilityInputDto =
            super::from_json(json!({ "data": data }), "input").expect("valid input");
        let dto = super::boxcox_capability_dto(default).expect("default range");
        assert!(!dto.lambda_at_bound, "lambda={}", dto.lambda);
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

    // ── Where a request stopped deserializing ───────────────────────

    #[test]
    fn a_bad_value_inside_an_input_object_is_refused_where_it_sits() {
        let e = super::from_json::<super::CusumInputDto>(
            json!({ "data": [1.0, null, 3.0], "target": 1.0, "sigma": 1.0 }),
            "input",
        )
        .err()
        .expect("null is not a number");
        assert_eq!(
            (e.code, e.parameter.as_deref(), e.index),
            ("malformed_input", Some("data"), Some(1)),
            "{e}"
        );

        let e = super::from_json::<super::MultiPeltInputDto>(
            json!({ "signals": [[1.0, 2.0], [3.0, "4"]] }),
            "input",
        )
        .err()
        .expect("a string is not a number");
        assert_eq!(
            (e.parameter.as_deref(), e.index),
            (Some("signals[1]"), Some(1)),
            "{e}"
        );

        // Not inside an array: the field, no index.
        let e = super::from_json::<super::CusumInputDto>(
            json!({ "data": [1.0], "target": "ten", "sigma": 1.0 }),
            "input",
        )
        .err()
        .expect("target is a number");
        assert_eq!((e.parameter.as_deref(), e.index), (Some("target"), None));

        // A missing field is the argument's.
        let e = super::from_json::<super::CusumInputDto>(json!({ "data": [1.0] }), "input")
            .err()
            .expect("target is required");
        assert_eq!(e.parameter.as_deref(), Some("input"), "{e}");
    }

    // ── Counts: a whole number of either sign, range-checked after ──
    //
    // `from_js` hands `from_json` what serde-wasm-bindgen makes of the JS
    // value: a safe integer (`3`, `-1`) as an integer, anything else (`2.5`)
    // as a float -- the two `json!` spellings below.

    #[test]
    fn a_fractional_count_is_malformed_where_it_sits() {
        for (body, parameter) in [
            (
                json!({ "data": vec![1.0; 20], "max_lags": 2.5 }),
                "max_lags",
            ),
            (
                json!({ "data": vec![1.0; 20], "averaging_window": 2.5 }),
                "averaging_window",
            ),
            (
                json!({ "data": [1.0], "min_segment_len": 2.5 }),
                "min_segment_len",
            ),
        ] {
            let e = match parameter {
                "max_lags" => super::from_json::<super::AdfInputDto>(body, "input").map(|_| ()),
                "averaging_window" => {
                    super::from_json::<super::SpectralResidualInputDto>(body, "input").map(|_| ())
                }
                _ => super::from_json::<super::PeltInputDto>(body, "input").map(|_| ()),
            }
            .expect_err("a fraction is not a count");
            assert_eq!(
                (e.code, e.parameter.as_deref(), e.index),
                ("malformed_input", Some(parameter), None),
                "{e}"
            );
            assert!(e.message.contains("expected a whole number"), "{e}");
        }
    }

    #[test]
    fn a_negative_count_is_out_of_range_not_malformed() {
        let ten = [1.0, 3.0, 2.0, 5.0, 4.0, 6.0, 5.0, 8.0, 7.0, 9.0];
        let adf = |max_lags: serde_json::Value| {
            super::adf_dto(
                super::from_json(json!({ "data": ten, "max_lags": max_lags }), "input")
                    .expect("a whole number reads"),
            )
        };
        for got in [-1.0, 4.0] {
            let e = adf(json!(got as i64)).expect_err("outside [0, 3]");
            assert_eq!(
                (e.code, e.parameter.as_deref()),
                ("parameter_out_of_range", Some("max_lags"))
            );
            assert_eq!(
                e.details,
                vec![
                    ("min", Detail::Num(0.0)),
                    ("max", Detail::Num(3.0)),
                    ("got", Detail::Num(got)),
                ]
            );
        }
        // An integer is still read as one, and `2.0` is the same bound on the
        // AIC search.
        assert!(adf(json!(2)).expect("in range").n_lags <= 2);
        assert_eq!(
            adf(json!(2)).expect("in range").n_lags,
            adf(json!(2.0)).expect("in range").n_lags
        );

        let e = super::pelt_dto(
            super::from_json(
                json!({ "data": [1.0, 2.0], "min_segment_len": -1 }),
                "input",
            )
            .expect("a whole number reads"),
        )
        .err()
        .expect("below 2");
        assert_eq!(
            (e.code, e.parameter.as_deref(), &e.details[..]),
            (
                "parameter_out_of_range",
                Some("min_segment_len"),
                &[
                    ("min", Detail::Num(2.0)),
                    ("max", Detail::Null),
                    ("got", Detail::Num(-1.0)),
                ][..]
            )
        );
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
        let e = super::cusum_dto(empty).expect_err("empty");
        assert_eq!(
            (e.code, e.parameter.as_deref()),
            ("empty_input", Some("data"))
        );

        let bad: super::CusumInputDto = super::from_json(
            json!({ "data": [1.0], "target": 1.0, "sigma": 0.0 }),
            "input",
        )
        .expect("parses");
        let e = super::cusum_dto(bad).expect_err("sigma must be > 0");
        assert_eq!(
            (e.code, e.parameter.as_deref()),
            ("parameter_out_of_range", Some("sigma")),
            "{e}"
        );
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
        let e = super::ewma_dto(empty).expect_err("empty");
        assert_eq!(
            (e.code, e.parameter.as_deref()),
            ("empty_input", Some("data"))
        );

        for lambda in [0.0, 1.5] {
            let bad: super::EwmaInputDto = super::from_json(
                json!({ "data": [1.0, 2.0], "target": 1.0, "sigma": 1.0, "lambda": lambda }),
                "input",
            )
            .expect("parses");
            let e = super::ewma_dto(bad).expect_err("lambda out of range");
            assert_eq!(
                (e.code, e.parameter.as_deref()),
                ("parameter_out_of_range", Some("lambda")),
                "{e}"
            );
            // (0, 1]: open below, closed above.
            assert_eq!(
                e.details,
                vec![
                    ("min", crate::wire::Detail::Null),
                    ("exclusive_min", crate::wire::Detail::Num(0.0)),
                    ("max", crate::wire::Detail::Num(1.0)),
                    ("got", crate::wire::Detail::Num(lambda)),
                ]
            );
            assert_eq!(e.message, format!("lambda must be in (0, 1], got {lambda}"));
        }
    }

    #[test]
    fn laney_p_chart_refuses_a_sample_with_no_proportion() {
        let e = laney_p_dto(&[(3, 100), (0, 0), (4, 100), (2, 100)], &Default::default())
            .expect_err("0 of 0");
        assert!(e.message.contains("samples[1]"), "{e}");
        assert_eq!(e.index, Some(1));
        let e = laney_p_dto(&[(3, 100), (5, 120)], &Default::default())
            .expect_err("fewer than 3 samples");
        assert_eq!(e.code, crate::wire::code::INSUFFICIENT_DATA);
    }
}

#[cfg(test)]
mod code_catalog_tests {
    /// The README's code tables list exactly the published vocabulary, so the
    /// tables and `codes()` cannot drift apart.
    #[test]
    fn the_readme_documents_every_code_and_no_other() {
        let readme = include_str!("../README.md");
        let mut documented = std::collections::BTreeSet::new();
        let mut in_table = false;
        for line in readme.lines() {
            if line.starts_with("| `code` |") {
                in_table = true;
                continue;
            }
            if !line.starts_with('|') {
                in_table = false;
                continue;
            }
            if in_table && !line.starts_with("|---") {
                let first = line.split('|').nth(1).expect("a cell");
                for code in first.split('·') {
                    documented.insert(code.trim().trim_matches('`').to_string());
                }
            }
        }
        let catalog = crate::wire::code_catalog();
        let published: std::collections::BTreeSet<String> = catalog
            .refusals
            .iter()
            .map(|r| r.code.clone())
            .chain(catalog.violations.iter().map(|v| v.code.clone()))
            .collect();
        assert_eq!(documented, published);
    }
}

/// The option objects declared by hand in `OPTIONS_TS`, as JSON Schema built
/// from the tables the binding reads them with (the engine card's schemas;
/// tests only).
#[cfg(test)]
mod card_schema {
    use schemars::{Schema, SchemaGenerator};
    use serde_json::json;

    use crate::engine_card::string_enum;

    fn schema(value: serde_json::Value) -> Schema {
        Schema::try_from(value).expect("an object is a schema")
    }

    pub(super) fn interval_method(_: &mut SchemaGenerator) -> Schema {
        let names = string_enum(crate::wire::reliability::INTERVAL_METHODS).to_value();
        schema(json!({ "anyOf": [names, { "type": "null" }] }))
    }

    /// `{ rules? }`: a named rule set, the tests spelt out, or null.
    pub(super) fn rule_options() -> Schema {
        let sets = string_enum(crate::spc::RuleSet::NAMES).to_value();
        let rules = string_enum(crate::spc::ViolationType::ALL.iter().map(|v| v.name())).to_value();
        schema(json!({
            "description": "Which run tests to apply: a named set, or the tests spelt out. Omitted or null: \"shewhart\". []: none.",
            "type": ["object", "null"],
            "properties": {
                "rules": { "anyOf": [sets, { "type": "array", "items": rules }, { "type": "null" }] }
            },
            "additionalProperties": false
        }))
    }

    /// A Phase I standard: the named fields, each a number or null.
    pub(super) fn standard(fields: &[&str], description: &str) -> Schema {
        let properties: serde_json::Map<String, serde_json::Value> = fields
            .iter()
            .map(|f| (f.to_string(), json!({ "type": ["number", "null"] })))
            .collect();
        schema(json!({
            "description": description,
            "type": ["object", "null"],
            "properties": properties,
            "additionalProperties": false
        }))
    }
}

#[cfg(test)]
mod engine_card_tests {
    //! The engine card's operations are this binding's exports, and the
    //! schemas it points at are generated from the types they read and return.

    use super::*;
    use crate::engine_card::{arg, optional, string_enum, Schemas};
    use crate::wire::{
        counter, hypothesis, point_process, reliability, tolerance, type1_gage, AcfDto,
        CapabilityDto, ChartPointDto, CorrelationMatrixDto, CusumDto, DistributionFitDto, EwmaDto,
        FixedLimitChartDto, GChartDto, GageRRAnovaResultDto, GageRRResultDto,
        GoodnessOfFitResultDto, ImrChartDto, KdeDto, LaneyPChartDto, LaneyUChartDto, PChartDto,
        PacfDto, PeltResultDto, PercentileCapabilityDto, RegressionDto, SeasonalityDto,
        SpectralResidualDto, StlDto, TChartDto, UChartDto, XbarRChartDto, XbarSChartDto,
    };

    fn schemas() -> Schemas {
        let s = Schemas::new();
        let num = || s.of::<f64>();
        let numbers = || s.of::<Vec<f64>>();
        let rows = || s.of::<Vec<Vec<f64>>>();
        let pairs = || s.of::<Vec<(f64, f64)>>();
        let rules = || optional("options", card_schema::rule_options());
        let test = || s.of::<hypothesis::TestResultDto>();

        s.op(
            "xbar_r_chart",
            vec![arg("data", rows()), rules()],
            s.of::<XbarRChartDto>(),
        );
        s.op(
            "xbar_s_chart",
            vec![arg("data", rows()), rules()],
            s.of::<XbarSChartDto>(),
        );
        s.op(
            "imr_chart",
            vec![arg("values", numbers()), rules()],
            s.of::<ImrChartDto>(),
        );
        s.op(
            "run_rules",
            vec![
                arg("values", numbers()),
                arg("limits", s.of::<LimitsInputDto>()),
                rules(),
            ],
            s.of::<Vec<ChartPointDto>>(),
        );
        let standard = |fields: &[&str]| {
            optional(
                "options",
                card_schema::standard(fields, "A known standard from a Phase I study."),
            )
        };
        s.op(
            "p_chart",
            vec![arg("samples", pairs()), standard(&["p_bar"])],
            s.of::<PChartDto>(),
        );
        s.op(
            "laney_p_chart",
            vec![arg("samples", pairs()), standard(&["p_bar", "phi"])],
            s.of::<LaneyPChartDto>(),
        );
        s.op(
            "u_chart",
            vec![arg("samples", pairs()), standard(&["u_bar"])],
            s.of::<UChartDto>(),
        );
        s.op(
            "laney_u_chart",
            vec![arg("samples", pairs()), standard(&["u_bar", "phi"])],
            s.of::<LaneyUChartDto>(),
        );
        s.op(
            "np_chart",
            vec![arg("defectives", numbers()), arg("sample_size", num())],
            s.of::<FixedLimitChartDto>(),
        );
        s.op(
            "c_chart",
            vec![arg("defects", numbers())],
            s.of::<FixedLimitChartDto>(),
        );
        s.op("g_chart", vec![arg("gaps", numbers())], s.of::<GChartDto>());
        s.op(
            "t_chart",
            vec![arg("times", numbers())],
            s.of::<TChartDto>(),
        );
        s.op(
            "process_capability",
            vec![arg("input", s.of::<CapabilityInputDto>())],
            s.of::<CapabilityDto>(),
        );
        s.op(
            "anderson_darling_test",
            vec![arg("data", numbers())],
            s.of::<hypothesis::AndersonDarlingDto>(),
        );
        s.op(
            "adf_test",
            vec![arg("input", s.of::<AdfInputDto>())],
            s.of::<hypothesis::AdfDto>(),
        );
        s.op(
            "kpss_test",
            vec![arg("input", s.of::<KpssInputDto>())],
            s.of::<hypothesis::KpssDto>(),
        );
        s.op(
            "correlation_matrix",
            vec![
                arg("variables", rows()),
                optional("options", s.of::<Option<CorrelationOptions>>()),
            ],
            s.of::<CorrelationMatrixDto>(),
        );
        s.op(
            "simple_regression",
            vec![
                arg("x", numbers()),
                arg("y", numbers()),
                optional("options", s.of::<Option<RegressionOptions>>()),
            ],
            s.of::<RegressionDto>(),
        );
        s.op(
            "multiple_regression",
            vec![arg("input", s.of::<MultipleRegressionInputDto>())],
            s.of::<crate::wire::MultipleRegressionDto>(),
        );
        s.op(
            "exponential_smoothing",
            vec![arg("input", s.of::<SmoothingInputDto>())],
            s.of::<crate::wire::SmoothingDto>(),
        );
        s.op(
            "prediction_errors",
            vec![arg("input", s.of::<PredictionErrorsInputDto>())],
            s.of::<crate::wire::PredictionErrorsDto>(),
        );
        s.op(
            "fit_best",
            vec![arg("data", numbers())],
            s.of::<Vec<DistributionFitDto>>(),
        );
        s.op(
            "kde",
            vec![arg("input", s.of::<KdeInputDto>())],
            s.of::<KdeDto>(),
        );
        s.op(
            "acf",
            vec![arg("input", s.of::<AutocorrelationInputDto>())],
            s.of::<AcfDto>(),
        );
        s.op(
            "pacf",
            vec![arg("input", s.of::<AutocorrelationInputDto>())],
            s.of::<PacfDto>(),
        );
        s.op(
            "goodness_of_fit",
            vec![arg("input", s.of::<GoodnessOfFitInputDto>())],
            s.of::<GoodnessOfFitResultDto>(),
        );
        s.op(
            "one_sample_t_test",
            vec![arg("data", numbers()), arg("mu0", num())],
            test(),
        );
        let sided = || {
            arg(
                "sided",
                string_enum(crate::wire::tolerance::SIDED.iter().map(|(name, _)| *name)),
            )
        };
        s.op(
            "tolerance_factor",
            vec![
                arg("n", s.of::<Whole>()),
                arg("coverage", num()),
                arg("confidence", num()),
                sided(),
            ],
            num(),
        );
        s.op(
            "tolerance_interval",
            vec![
                arg("data", numbers()),
                arg("coverage", num()),
                arg("confidence", num()),
                sided(),
            ],
            s.of::<tolerance::ToleranceIntervalDto>(),
        );
        s.op(
            "type1_gage_study",
            vec![
                arg("measurements", numbers()),
                arg("reference", num()),
                arg("lsl", num()),
                arg("usl", num()),
                optional("tolerance_fraction", s.of::<Option<f64>>()),
                optional("sigma_multiple", s.of::<Option<f64>>()),
            ],
            s.of::<type1_gage::Type1GageDto>(),
        );
        let equivalence = || s.of::<hypothesis::EquivalenceDto>();
        let alpha = || optional("alpha", s.of::<Option<f64>>());
        s.op(
            "tost_one_sample",
            vec![
                arg("data", numbers()),
                arg("target", num()),
                arg("low", num()),
                arg("high", num()),
                alpha(),
            ],
            equivalence(),
        );
        s.op(
            "tost_paired",
            vec![
                arg("x", numbers()),
                arg("y", numbers()),
                arg("low", num()),
                arg("high", num()),
                alpha(),
            ],
            equivalence(),
        );
        s.op(
            "tost_two_sample",
            vec![
                arg("a", numbers()),
                arg("b", numbers()),
                arg("low", num()),
                arg("high", num()),
                alpha(),
                optional("equal_variance", s.of::<Option<bool>>()),
            ],
            equivalence(),
        );
        for (id, a, b) in [
            ("two_sample_t_test", "a", "b"),
            ("paired_t_test", "x", "y"),
            ("mann_whitney_u_test", "a", "b"),
            ("wilcoxon_signed_rank_test", "x", "y"),
            ("chi_squared_goodness_of_fit", "observed", "expected"),
        ] {
            s.op(id, vec![arg(a, numbers()), arg(b, numbers())], test());
        }
        s.op("jarque_bera_test", vec![arg("data", numbers())], test());
        s.op(
            "shapiro_wilk_test",
            vec![arg("data", numbers())],
            s.of::<hypothesis::ShapiroWilkDto>(),
        );
        s.op(
            "mann_kendall_test",
            vec![arg("data", numbers())],
            s.of::<hypothesis::MannKendallDto>(),
        );
        let counter_options = || optional("options", s.of::<Option<counter::CounterOptionsDto>>());
        s.op(
            "counter_increments",
            vec![
                arg("times", numbers()),
                arg("values", numbers()),
                counter_options(),
            ],
            s.of::<Vec<counter::CounterIncrementDto>>(),
        );
        s.op(
            "counter_totals",
            vec![
                arg("times", numbers()),
                arg("values", numbers()),
                arg("edges", numbers()),
                counter_options(),
            ],
            s.of::<Vec<counter::CounterTotalDto>>(),
        );
        let observation = || arg("observation", s.of::<point_process::ObservationDto>());
        for id in ["laplace_trend_test", "mil_hdbk_189_test"] {
            s.op(
                id,
                vec![arg("times", numbers()), observation()],
                s.of::<point_process::TrendTestDto>(),
            );
        }
        s.op(
            "power_law_process_fit",
            vec![arg("times", numbers()), observation()],
            s.of::<point_process::PowerLawFitDto>(),
        );
        s.op(
            "one_way_anova",
            vec![arg("groups", rows())],
            s.of::<hypothesis::AnovaDto>(),
        );
        for id in ["kruskal_wallis_test", "levene_test", "bartlett_test"] {
            s.op(id, vec![arg("groups", rows())], test());
        }
        s.op(
            "chi_squared_independence",
            vec![arg("table", rows())],
            test(),
        );
        s.op(
            "fisher_exact_test",
            vec![arg("table", s.of::<[[f64; 2]; 2]>())],
            test(),
        );
        for id in ["bonferroni_correction", "benjamini_hochberg"] {
            s.op(id, vec![arg("p_values", numbers())], numbers());
        }
        s.op(
            "detect_changepoints",
            vec![arg("input", s.of::<PeltInputDto>())],
            s.of::<PeltResultDto>(),
        );
        s.op(
            "detect_changepoints_multi",
            vec![arg("input", s.of::<MultiPeltInputDto>())],
            s.of::<PeltResultDto>(),
        );
        s.op(
            "gage_rr_xbar_r",
            vec![arg("input", s.of::<GageRRInputDto>())],
            s.of::<GageRRResultDto>(),
        );
        s.op(
            "gage_rr_anova",
            vec![arg("input", s.of::<GageRRInputDto>())],
            s.of::<GageRRAnovaResultDto>(),
        );
        s.op(
            "percentile_capability",
            vec![arg("input", s.of::<PercentileCapabilityInputDto>())],
            s.of::<PercentileCapabilityDto>(),
        );
        s.op(
            "cusum",
            vec![arg("input", s.of::<CusumInputDto>())],
            s.of::<CusumDto>(),
        );
        s.op(
            "ewma",
            vec![arg("input", s.of::<EwmaInputDto>())],
            s.of::<EwmaDto>(),
        );
        s.op(
            "boxcox_capability",
            vec![arg("input", s.of::<BoxcoxCapabilityInputDto>())],
            s.of::<reliability::BoxcoxCapabilityDto>(),
        );
        s.op(
            "estimate_period",
            vec![arg("input", s.of::<SeasonalityInputDto>())],
            s.of::<SeasonalityDto>(),
        );
        s.op(
            "spectral_residual",
            vec![arg("input", s.of::<SpectralResidualInputDto>())],
            s.of::<SpectralResidualDto>(),
        );
        s.op(
            "stl",
            vec![arg("input", s.of::<StlInputDto>())],
            s.of::<StlDto>(),
        );
        s.op("sigma_to_ppm", vec![arg("sigma", num())], num());
        s.op("ppm_to_sigma", vec![arg("ppm", num())], num());
        let suspensions = || optional("suspension_times", s.of::<Option<Vec<f64>>>());
        s.op(
            "weibull_mle",
            vec![
                arg("failure_times", numbers()),
                suspensions(),
                optional("options", s.of::<Option<WeibullMleOptions>>()),
            ],
            s.of::<reliability::WeibullMleDto>(),
        );
        s.op(
            "weibull_mrr",
            vec![arg("failure_times", numbers()), suspensions()],
            s.of::<reliability::WeibullMrrDto>(),
        );
        s.op(
            "weibull_reliability",
            vec![arg("input", s.of::<WeibullReliabilityInputDto>())],
            s.of::<reliability::WeibullReliabilityDto>(),
        );
        s
    }

    #[test]
    fn the_schemas_are_generated_from_the_binding_types() {
        schemas().check("u-analytics WebAssembly operations");
    }

    #[test]
    fn every_export_is_a_card_operation_with_its_arguments() {
        let s = schemas();
        s.check_exports(include_str!("wasm.rs"));
        s.check_card();
    }

    /// No operation reads a clock or draws a random number, so each returns
    /// the same result for the same arguments.
    #[test]
    fn the_crate_reads_no_clock_and_no_randomness() {
        assert_eq!(
            crate::engine_card::clock_and_entropy_uses(),
            Vec::<(String, &str)>::new()
        );
    }

    // ── The within sigma a variables chart can be given ────────────────

    fn phase_subgroups() -> Vec<Vec<f64>> {
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
    fn variables_charts_take_a_fixed_sigma_through_the_wire() {
        use crate::spc::RuleSet;
        let groups = phase_subgroups();
        let n = groups[0].len() as f64;
        let free = xbar_r_dto(groups.clone(), RuleSet::default(), None).unwrap();
        let fixed = xbar_r_dto(groups.clone(), RuleSet::default(), Some(0.5)).unwrap();
        assert_eq!(fixed.sigma_hat, 0.5);
        assert_eq!(fixed.sigma_hat_df, None);
        assert!(free.sigma_hat_df.is_some());
        assert_eq!(fixed.xbar_cl, free.xbar_cl);
        assert!((fixed.xbar_ucl - fixed.xbar_cl - 1.5 / n.sqrt()).abs() < 1e-12);
        assert_ne!(fixed.r_cl, free.r_cl);

        let s = xbar_s_dto(groups, RuleSet::default(), Some(0.5)).unwrap();
        assert_eq!((s.sigma_hat, s.sigma_hat_df), (0.5, None));
        let values: Vec<f64> = phase_subgroups().iter().map(|g| g[0]).collect();
        let i = imr_dto(values, RuleSet::default(), Some(0.5)).unwrap();
        assert!((i.i_ucl - i.i_cl - 1.5).abs() < 1e-12);
        assert_eq!((i.sigma_hat, i.sigma_hat_df), (0.5, None));
    }

    #[test]
    fn a_bad_fixed_sigma_is_refused_with_its_bounds() {
        use crate::spc::RuleSet;
        for bad in [0.0, -1.0] {
            let e = xbar_r_dto(phase_subgroups(), RuleSet::default(), Some(bad))
                .expect_err("sigma must exceed 0");
            assert_eq!(e.code, "parameter_out_of_range");
            assert_eq!(e.parameter.as_deref(), Some("sigma"));
        }
        let e = crate::wire::sigma_from_json(Some(&serde_json::json!({ "sigma": "wide" })))
            .expect_err("a string is not a number");
        assert_eq!(
            (e.code, e.parameter.as_deref()),
            ("malformed_input", Some("sigma"))
        );
        assert_eq!(
            crate::wire::sigma_from_json(Some(&serde_json::json!({}))).unwrap(),
            None
        );
        assert_eq!(
            crate::wire::sigma_from_json(Some(&serde_json::json!({ "sigma": null }))).unwrap(),
            None
        );
        assert_eq!(
            crate::wire::sigma_from_json(Some(&serde_json::json!({ "sigma": 0.25 }))).unwrap(),
            Some(0.25)
        );
    }
}
