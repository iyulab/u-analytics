//! The keys and names each C entry point reads, measured from the entry
//! points themselves and held in `request_keys.json` beside this file -- the
//! file a binding's contract tests check the requests they write against.
//!
//! Every request type refuses a key it does not know and names the keys it
//! does (`unknown field ..., expected one of ...`); every option refuses a name
//! it does not know and lists the names it does (`unknown_option` with
//! `expected`). So a valid request is sent with a key no request has at each
//! level, and with a name no option has at each value, and the refusals are
//! read. Each value is described as one of
//!
//! - `{"properties": {key: value, ...}}` -- an object;
//! - `{"tag": key, "variants": {name: {"properties": ...}, ...}}` -- an object
//!   whose `tag` key names which keys it has;
//! - `{"items": value}` -- an array of objects or of names;
//! - `{"enum": [name, ...]}` -- one of these names, with `"number": true` when
//!   a number is taken there too;
//! - `{"enum": [name, ...], "items": {"enum": [...]}}` -- one of these names,
//!   or an array of the other names (a named set, or the members spelt out);
//! - `{}` -- anything else: a number, an array of numbers.
//!
//! An object that does not refuse an unknown key cannot be measured, and fails
//! the measurement: a misspelt key it ignores would never reach the caller.
//!
//! `UPDATE_REQUEST_KEYS=1 cargo test --features ffi --lib request_keys`
//! rewrites the file; review its diff like any other change to the contract.

use super::*;
// What differs from one crate's copy of this module to another's.
use super::uanalytics_free_string as free_string;
const FREE_STRING: &str = "uanalytics_free_string";

use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::ffi::CString;
use std::path::PathBuf;

type Entry = unsafe extern "C" fn(*const libc::c_char, *mut *mut libc::c_char) -> i32;

/// A key and a name no request has.
const PROBE: &str = "__probe__";

#[derive(Clone)]
enum Step {
    Key(String),
    Index(usize),
}

fn call(entry: Entry, request: &Value) -> (i32, Value) {
    let request = CString::new(request.to_string()).expect("no interior NUL");
    let mut out: *mut libc::c_char = std::ptr::null_mut();
    let code = unsafe { entry(request.as_ptr(), &mut out) };
    assert!(!out.is_null(), "every status comes with a JSON body");
    let body = unsafe { CStr::from_ptr(out) }
        .to_str()
        .expect("body is UTF-8")
        .to_owned();
    unsafe { free_string(out) };
    (code, serde_json::from_str(&body).expect("body is JSON"))
}

/// `limits.ucl`, `customers[0].id`: how a refusal's `parameter` names a place.
fn parameter(path: &[Step]) -> String {
    let mut out = String::new();
    for step in path {
        match step {
            Step::Key(k) if out.is_empty() => out.push_str(k),
            Step::Key(k) => {
                out.push('.');
                out.push_str(k);
            }
            Step::Index(i) => out.push_str(&format!("[{i}]")),
        }
    }
    out
}

/// `base` with `value` at `path`; an object on the way is made when absent.
fn put(base: &Value, path: &[Step], value: Value) -> Value {
    let mut request = base.clone();
    let mut here = &mut request;
    for step in path {
        here = match step {
            Step::Key(k) => {
                if here.is_null() {
                    *here = json!({});
                }
                here.as_object_mut()
                    .unwrap_or_else(|| panic!("{} is not an object in {base}", parameter(path)))
                    .entry(k.clone())
                    .or_insert(Value::Null)
            }
            Step::Index(i) => here.get_mut(*i).unwrap_or_else(|| {
                panic!("the base request needs an element at {}", parameter(path))
            }),
        };
    }
    *here = value;
    request
}

fn get<'a>(base: &'a Value, path: &[Step]) -> Option<&'a Value> {
    path.iter().try_fold(base, |v, step| match step {
        Step::Key(k) => v.get(k),
        Step::Index(i) => v.get(*i),
    })
}

/// The keys an unknown-field refusal of the probe key lists.
fn known_keys(body: &Value) -> Option<Vec<String>> {
    let text = body["error"].as_str()?;
    let marker = format!("unknown field `{PROBE}`, ");
    let rest = &text[text.find(&marker)? + marker.len()..];
    let rest = rest.split(" at line ").next().unwrap_or(rest);
    if rest.starts_with("there are no fields") {
        return Some(Vec::new());
    }
    Some(
        rest.split('`')
            .skip(1)
            .step_by(2)
            .map(str::to_string)
            .collect(),
    )
}

/// The names an unknown-option refusal at `at` lists.
fn known_names(body: &Value, at: &str) -> Option<Vec<String>> {
    if body["code"] != "unknown_option" || body["parameter"] != at {
        return None;
    }
    let expected = body["expected"]
        .as_array()
        .unwrap_or_else(|| panic!("an unknown option without `expected`: {body}"));
    Some(
        expected
            .iter()
            .map(|n| n.as_str().expect("a name").to_string())
            .collect(),
    )
}

/// An object that took the probe key without a word cannot be measured.
fn refuse_silence(name: &str, at: &str, body: &Value) {
    let text = body["error"].as_str().unwrap_or_default();
    assert!(
        !text.contains("missing field"),
        "{name}: {at} took a key it does not know without refusing it \
         (or the base request lacks a key it needs): {body}"
    );
}

fn object(name: &str, entry: Entry, base: &Value, path: &[Step], keys: Vec<String>) -> Value {
    let properties: BTreeMap<String, Value> = keys
        .into_iter()
        .map(|key| {
            let mut at = path.to_vec();
            at.push(Step::Key(key.clone()));
            let value = describe(name, entry, base, &at);
            (key, value)
        })
        .collect();
    json!({ "properties": properties })
}

/// What the value at `path` of a valid `base` request is.
fn describe(name: &str, entry: Entry, base: &Value, path: &[Step]) -> Value {
    let at = parameter(path);

    // An object: the probe key is refused with the keys it has.
    let (_, body) = call(entry, &put(base, path, json!({ PROBE: 0 })));
    if let Some(keys) = known_keys(&body) {
        return object(name, entry, base, path, keys);
    }
    refuse_silence(name, &at, &body);
    // An object whose tag key names its variant: the object without the tag
    // is refused at that key, and a name no variant has with the names of
    // the variants.
    let tag = body["parameter"]
        .as_str()
        .and_then(|p| p.strip_prefix(&format!("{at}.")))
        .filter(|tag| !tag.contains(['.', '[']))
        .map(str::to_string);
    let variants = tag.as_ref().and_then(|tag| {
        let (_, body) = call(entry, &put(base, path, json!({ tag.as_str(): PROBE })));
        known_names(&body, &format!("{at}.{tag}"))
    });
    if let (Some(tag), Some(variants)) = (tag, variants) {
        let variants: BTreeMap<String, Value> = variants
            .into_iter()
            .map(|variant| {
                let (_, body) = call(
                    entry,
                    &put(base, path, json!({ tag.as_str(): variant, PROBE: 0 })),
                );
                let keys: Vec<String> = known_keys(&body)
                    .unwrap_or_else(|| panic!("{name}: {at} as {variant:?}: {body}"))
                    .into_iter()
                    .filter(|k| *k != tag)
                    .collect();
                let carried = get(base, path).and_then(|o| o.get(&tag)) == Some(&json!(variant));
                let variant_base = if carried {
                    base.clone()
                } else {
                    put(base, path, json!({ tag.as_str(): variant }))
                };
                let described = object(name, entry, &variant_base, path, keys);
                (variant, described)
            })
            .collect();
        return json!({ "tag": tag, "variants": variants });
    }

    // An array of objects.
    let (_, body) = call(entry, &put(base, path, json!([{ PROBE: 0 }])));
    if let Some(keys) = known_keys(&body) {
        let mut item = path.to_vec();
        item.push(Step::Index(0));
        return json!({ "items": object(name, entry, base, &item, keys) });
    }
    refuse_silence(name, &at, &body);

    // A name, or an array of names.
    let (_, body) = call(entry, &put(base, path, json!(PROBE)));
    if let Some(names) = known_names(&body, &at) {
        // A name, or an array of other names: the array is refused with those.
        let (_, listed) = call(entry, &put(base, path, json!([PROBE])));
        let items = known_names(&listed, &at)
            .or_else(|| known_names(&listed, &format!("{at}[0]")))
            .filter(|items| *items != names);
        if let Some(items) = items {
            return json!({ "enum": names, "items": { "enum": items } });
        }
        let (_, body) = call(entry, &put(base, path, json!(1)));
        let text = body["error"].as_str().unwrap_or_default();
        return if text.contains("invalid type") {
            json!({ "enum": names })
        } else {
            json!({ "enum": names, "number": true })
        };
    }
    let (_, body) = call(entry, &put(base, path, json!([PROBE])));
    if let Some(names) = known_names(&body, &at).or_else(|| known_names(&body, &format!("{at}[0]")))
    {
        return json!({ "items": { "enum": names } });
    }
    json!({})
}

fn measure(entry_points: &[(&str, Entry, Value)]) -> Value {
    let measured: BTreeMap<&str, Value> = entry_points
        .iter()
        .map(|(name, entry, base)| {
            let (code, body) = call(*entry, base);
            assert_eq!(code, 0, "{name}: the base request is refused: {body}");
            let (_, body) = call(*entry, &json!({ PROBE: 0 }));
            let keys = known_keys(&body)
                .unwrap_or_else(|| panic!("{name} does not refuse a key it does not know: {body}"));
            (*name, object(name, *entry, base, &[], keys))
        })
        .collect();
    json!({
        "$comment": "The keys and names each C entry point reads, measured from the entry points (src/ffi/request_keys.rs). Regenerate: UPDATE_REQUEST_KEYS=1 cargo test --features ffi --lib request_keys",
        "entry_points": measured,
    })
}

/// The JSON entry points `ffi.rs` exports, read from its source: each
/// `pub unsafe extern "C" fn` and each name an `...export!` macro is given.
fn exported() -> Vec<String> {
    let source = include_str!("../ffi.rs");
    let source = &source[..source.find("// ── Tests").expect("the tests marker")];
    let mut names = Vec::new();
    let mut in_macro_call = false;
    for line in source.lines().map(str::trim) {
        if let Some(rest) = line.strip_prefix("pub unsafe extern \"C\" fn ") {
            if let Some(name) = rest.split('(').next().filter(|n| !n.starts_with('$')) {
                names.push(name.to_string());
            }
        } else if line.ends_with("export!(") && !line.starts_with("macro_rules!") {
            in_macro_call = true;
        } else if in_macro_call && !line.starts_with("///") && !line.starts_with("#[") {
            let name = line.split([',', ' ']).next().expect("a name");
            names.push(name.to_string());
            in_macro_call = false;
        }
    }
    names.retain(|n| n != FREE_STRING);
    names.sort();
    names
}

// ── The entry points and a valid request to each ──────────────────────────

fn series() -> Value {
    json!([
        10.1, 9.8, 10.3, 10.0, 9.7, 10.4, 10.2, 9.9, 10.1, 10.6, 9.8, 10.0, 10.3, 9.6, 10.2, 10.1,
        9.9, 10.5, 10.0, 9.8
    ])
}

fn positive() -> Value {
    json!([
        12.0, 25.0, 33.0, 47.0, 58.0, 71.0, 90.0, 18.0, 29.0, 41.0, 64.0, 83.0, 22.0, 37.0, 52.0,
        77.0
    ])
}

fn subgroups() -> Value {
    json!([
        [10.1, 10.3, 9.8, 10.0],
        [10.0, 10.2, 10.1, 9.9],
        [9.9, 10.4, 10.2, 10.0],
        [10.2, 10.1, 9.7, 10.3],
        [10.0, 9.8, 10.1, 10.2]
    ])
}

fn step() -> Value {
    (0..40)
        .map(|i| (if i < 20 { 0.0 } else { 5.0 }) + 0.1 * f64::from((i * 7) % 5 - 2))
        .collect()
}

fn walk() -> Value {
    let mut v = 0.0;
    (0..60)
        .map(|i| {
            v += f64::from((i * 37) % 11 - 5) / 5.0;
            v
        })
        .collect()
}

fn seasonal(n: i32, period: i32) -> Value {
    (0..n)
        .map(|i| {
            (2.0 * std::f64::consts::PI * f64::from(i) / f64::from(period)).sin()
                + 0.05 * f64::from((i * 13) % 7 - 3)
        })
        .collect()
}

fn gage() -> Value {
    (0..10)
        .map(|p| {
            (0..3)
                .map(|o| {
                    let (p, o) = (f64::from(p), f64::from(o));
                    json!([
                        p * 0.5 + o * 0.05 + 0.01 * ((p + o) % 3.0),
                        p * 0.5 + o * 0.05
                    ])
                })
                .collect::<Value>()
        })
        .collect()
}

fn ep(name: &'static str, entry: Entry, base: Value) -> (&'static str, Entry, Value) {
    (name, entry, base)
}

fn entry_points() -> Vec<(&'static str, Entry, Value)> {
    let events = json!([5, 12, 20, 26, 31, 35, 38, 40]);
    let observed = json!({ "truncation": "time", "end": 42 });
    vec![
        ep(
            "uanalytics_xbar_r_chart",
            uanalytics_xbar_r_chart,
            json!({ "subgroups": subgroups() }),
        ),
        ep(
            "uanalytics_xbar_s_chart",
            uanalytics_xbar_s_chart,
            json!({ "subgroups": subgroups() }),
        ),
        ep(
            "uanalytics_imr_chart",
            uanalytics_imr_chart,
            json!({ "values": series() }),
        ),
        ep(
            "uanalytics_run_rules",
            uanalytics_run_rules,
            json!({ "values": [0, 1, 4, 0.5, -0.2, 3.5], "limits": { "ucl": 3, "cl": 0, "lcl": -3 } }),
        ),
        ep(
            "uanalytics_p_chart",
            uanalytics_p_chart,
            json!({ "samples": [[3, 50], [5, 50], [2, 40]] }),
        ),
        ep(
            "uanalytics_laney_p_chart",
            uanalytics_laney_p_chart,
            json!({ "samples": [[3, 50], [5, 50], [2, 40], [4, 50]] }),
        ),
        ep(
            "uanalytics_np_chart",
            uanalytics_np_chart,
            json!({ "defectives": [2, 3, 1, 4], "sample_size": 50 }),
        ),
        ep(
            "uanalytics_c_chart",
            uanalytics_c_chart,
            json!({ "defects": [3, 5, 2, 4] }),
        ),
        ep(
            "uanalytics_u_chart",
            uanalytics_u_chart,
            json!({ "samples": [[3, 1.5], [5, 2], [2, 1]] }),
        ),
        ep(
            "uanalytics_laney_u_chart",
            uanalytics_laney_u_chart,
            json!({ "samples": [[3, 1.5], [5, 2], [2, 1], [6, 2.5]] }),
        ),
        ep(
            "uanalytics_g_chart",
            uanalytics_g_chart,
            json!({ "gaps": [10, 20, 30, 15] }),
        ),
        ep(
            "uanalytics_t_chart",
            uanalytics_t_chart,
            json!({ "times": [1.5, 2.0, 0.5, 3.0] }),
        ),
        ep(
            "uanalytics_process_capability",
            uanalytics_process_capability,
            json!({ "data": series(), "usl": 11, "lsl": 9 }),
        ),
        ep(
            "uanalytics_percentile_capability",
            uanalytics_percentile_capability,
            json!({ "data": series(), "usl": 11, "lsl": 9 }),
        ),
        ep(
            "uanalytics_gage_rr_xbar_r",
            uanalytics_gage_rr_xbar_r,
            json!({ "measurements": gage() }),
        ),
        ep(
            "uanalytics_gage_rr_anova",
            uanalytics_gage_rr_anova,
            json!({ "measurements": gage() }),
        ),
        ep(
            "uanalytics_weibull_mle",
            uanalytics_weibull_mle,
            json!({ "failure_times": positive() }),
        ),
        ep(
            "uanalytics_weibull_mrr",
            uanalytics_weibull_mrr,
            json!({ "failure_times": positive() }),
        ),
        ep(
            "uanalytics_weibull_reliability",
            uanalytics_weibull_reliability,
            json!({ "shape": 2, "scale": 100 }),
        ),
        ep(
            "uanalytics_boxcox_capability",
            uanalytics_boxcox_capability,
            json!({ "data": positive(), "usl": 120, "lsl": 5 }),
        ),
        ep(
            "uanalytics_sigma_to_ppm",
            uanalytics_sigma_to_ppm,
            json!({ "sigma": 6 }),
        ),
        ep(
            "uanalytics_ppm_to_sigma",
            uanalytics_ppm_to_sigma,
            json!({ "ppm": 3.4 }),
        ),
        ep(
            "uanalytics_one_sample_t_test",
            uanalytics_one_sample_t_test,
            json!({ "data": series(), "mu0": 10 }),
        ),
        ep(
            "uanalytics_two_sample_t_test",
            uanalytics_two_sample_t_test,
            json!({ "a": [5.1, 4.9, 5.3, 5.0], "b": [5.6, 5.8, 5.5, 5.9] }),
        ),
        ep(
            "uanalytics_mann_whitney_u_test",
            uanalytics_mann_whitney_u_test,
            json!({ "a": [5.1, 4.9, 5.3, 5.0], "b": [5.6, 5.8, 5.5, 5.9] }),
        ),
        ep(
            "uanalytics_paired_t_test",
            uanalytics_paired_t_test,
            json!({ "x": [5.1, 4.9, 5.3, 5.0, 5.2], "y": [5.6, 5.8, 5.5, 5.9, 5.4] }),
        ),
        ep(
            "uanalytics_wilcoxon_signed_rank_test",
            uanalytics_wilcoxon_signed_rank_test,
            json!({ "x": [5.1, 4.9, 5.3, 5.0, 5.2, 4.8], "y": [5.6, 5.8, 5.5, 5.9, 5.4, 5.7] }),
        ),
        ep(
            "uanalytics_jarque_bera_test",
            uanalytics_jarque_bera_test,
            json!({ "data": series() }),
        ),
        ep(
            "uanalytics_shapiro_wilk_test",
            uanalytics_shapiro_wilk_test,
            json!({ "data": series() }),
        ),
        ep(
            "uanalytics_anderson_darling_test",
            uanalytics_anderson_darling_test,
            json!({ "data": series() }),
        ),
        ep(
            "uanalytics_adf_test",
            uanalytics_adf_test,
            json!({ "data": walk() }),
        ),
        ep(
            "uanalytics_mann_kendall_test",
            uanalytics_mann_kendall_test,
            json!({ "data": series() }),
        ),
        ep(
            "uanalytics_one_way_anova",
            uanalytics_one_way_anova,
            json!({ "groups": subgroups() }),
        ),
        ep(
            "uanalytics_kruskal_wallis_test",
            uanalytics_kruskal_wallis_test,
            json!({ "groups": subgroups() }),
        ),
        ep(
            "uanalytics_levene_test",
            uanalytics_levene_test,
            json!({ "groups": subgroups() }),
        ),
        ep(
            "uanalytics_bartlett_test",
            uanalytics_bartlett_test,
            json!({ "groups": subgroups() }),
        ),
        ep(
            "uanalytics_chi_squared_goodness_of_fit",
            uanalytics_chi_squared_goodness_of_fit,
            json!({ "observed": [10, 20, 30], "expected": [20, 20, 20] }),
        ),
        ep(
            "uanalytics_chi_squared_independence",
            uanalytics_chi_squared_independence,
            json!({ "table": [[10, 20], [30, 40]] }),
        ),
        ep(
            "uanalytics_fisher_exact_test",
            uanalytics_fisher_exact_test,
            json!({ "table": [[3, 1], [1, 3]] }),
        ),
        ep(
            "uanalytics_bonferroni_correction",
            uanalytics_bonferroni_correction,
            json!({ "p_values": [0.01, 0.04] }),
        ),
        ep(
            "uanalytics_benjamini_hochberg",
            uanalytics_benjamini_hochberg,
            json!({ "p_values": [0.01, 0.04, 0.03] }),
        ),
        ep(
            "uanalytics_counter_increments",
            uanalytics_counter_increments,
            json!({ "times": [0.0, 1.0, 2.0], "values": [1.0, 2.0, 0.5],
                    "modulus": 10.0, "max_rate": 100.0,
                    "discontinuities": [{ "time": 1.5, "start": 0.0, "last": 2.5 }] }),
        ),
        ep(
            "uanalytics_counter_totals",
            uanalytics_counter_totals,
            json!({ "times": [0.0, 1.0, 2.0], "values": [1.0, 2.0, 3.0],
                    "edges": [0.0, 2.0, 4.0], "modulus": 10.0, "max_rate": 100.0,
                    "discontinuities": [{ "time": 1.5, "start": 0.0, "last": 2.5 }] }),
        ),
        ep(
            "uanalytics_laplace_trend_test",
            uanalytics_laplace_trend_test,
            json!({ "times": events, "observation": observed }),
        ),
        ep(
            "uanalytics_mil_hdbk_189_test",
            uanalytics_mil_hdbk_189_test,
            json!({ "times": events, "observation": observed }),
        ),
        ep(
            "uanalytics_power_law_process_fit",
            uanalytics_power_law_process_fit,
            json!({ "times": events, "observation": observed }),
        ),
        ep(
            "uanalytics_detect_changepoints",
            uanalytics_detect_changepoints,
            json!({ "data": step() }),
        ),
        ep(
            "uanalytics_detect_changepoints_multi",
            uanalytics_detect_changepoints_multi,
            json!({ "signals": [step(), step()] }),
        ),
        ep(
            "uanalytics_cusum",
            uanalytics_cusum,
            json!({ "data": series(), "target": 10, "sigma": 0.25 }),
        ),
        ep(
            "uanalytics_ewma",
            uanalytics_ewma,
            json!({ "data": series(), "target": 10, "sigma": 0.25 }),
        ),
        ep(
            "uanalytics_estimate_period",
            uanalytics_estimate_period,
            json!({ "data": seasonal(84, 7) }),
        ),
        ep(
            "uanalytics_spectral_residual",
            uanalytics_spectral_residual,
            json!({ "data": seasonal(64, 8) }),
        ),
        ep(
            "uanalytics_correlation_matrix",
            uanalytics_correlation_matrix,
            json!({ "variables": [[1, 2, 3, 4, 5], [2, 1, 4, 3, 5]] }),
        ),
        ep(
            "uanalytics_simple_regression",
            uanalytics_simple_regression,
            json!({ "x": [1, 2, 3, 4], "y": [3, 5, 7.1, 9] }),
        ),
        ep(
            "uanalytics_fit_best",
            uanalytics_fit_best,
            json!({ "data": positive() }),
        ),
        ep(
            "uanalytics_acf",
            uanalytics_acf,
            json!({ "data": seasonal(84, 7), "max_lag": 10 }),
        ),
        ep(
            "uanalytics_pacf",
            uanalytics_pacf,
            json!({ "data": seasonal(84, 7), "max_lag": 10 }),
        ),
        ep(
            "uanalytics_goodness_of_fit",
            uanalytics_goodness_of_fit,
            json!({ "data": positive(), "distribution": "Weibull", "replicates": 9, "seed": 1 }),
        ),
    ]
}

fn held_file() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/ffi/request_keys.json")
}

#[test]
fn every_exported_entry_point_is_measured() {
    let mut measured: Vec<String> = entry_points()
        .iter()
        .map(|(n, _, _)| n.to_string())
        .collect();
    measured.sort();
    assert_eq!(measured, exported());
}

#[test]
fn the_held_request_keys_are_what_the_entry_points_read() {
    let text = serde_json::to_string_pretty(&measure(&entry_points())).expect("JSON") + "\n";
    let path = held_file();
    if std::env::var_os("UPDATE_REQUEST_KEYS").is_some() {
        std::fs::write(&path, &text).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        return;
    }
    let held = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
        .replace("\r\n", "\n");
    assert!(
        held == text,
        "{} is not what the entry points read. Rerun with UPDATE_REQUEST_KEYS=1 and review the \
         diff; the bindings' contract tests read this file.\nmeasured:\n{text}",
        path.display()
    );
}
