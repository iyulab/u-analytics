//! Every C entry point, fed the degenerate versions of its valid request:
//! each array of numbers -- or each numeric field across an array of objects,
//! such as every customer's `x` -- made constant, a straight line or all
//! zeros, and each array cut to one, two or three elements; one at a time and
//! all at once.
//!
//! A representative input does not reach the edges where a statistic divides
//! by a zero spread or takes the log of zero, or where every point sits in
//! the same place. There an entry point has to answer with a refusal or a
//! defined value; what it must not do is panic or write a non-finite number
//! (the test build's `finite_out` guard turns that into a panic, which the
//! entry point reports as `internal`). So every answer to these requests is
//! either a success or a refusal whose `code` is not `internal`.
//!
//! The same file serves every crate with a C ABI and a `request_keys` module.

use super::request_keys::{call, entry_points};
use serde_json::Value;
use std::cell::{Cell, RefCell};
use std::sync::Once;

thread_local! {
    /// The message of the last panic on this thread, while `QUIET`.
    static LAST_PANIC: RefCell<Option<String>> = const { RefCell::new(None) };
    /// This thread expects panics (an entry point catches them) and keeps
    /// their messages instead of printing them.
    static QUIET: Cell<bool> = const { Cell::new(false) };
}

/// Installs, once for the test binary, a panic hook that hands a quiet
/// thread's panic message to [`last_panic`] and prints every other panic as
/// before. Replacing the hook per test would race with the tests running
/// beside it.
pub(super) fn quiet_panics() {
    static HOOK: Once = Once::new();
    HOOK.call_once(|| {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            if QUIET.with(Cell::get) {
                LAST_PANIC.with(|l| *l.borrow_mut() = Some(info.to_string()));
            } else {
                previous(info);
            }
        }));
    });
    QUIET.with(|q| q.set(true));
}

/// The message of the last panic on this thread since the last call.
pub(super) fn last_panic() -> String {
    LAST_PANIC
        .with(|l| l.borrow_mut().take())
        .unwrap_or_default()
}

/// One way to make an array degenerate.
#[derive(Clone, Copy, Debug)]
enum Shape {
    Constant,
    Line,
    Zeros,
    Cut(usize),
}

const SHAPES: [Shape; 6] = [
    Shape::Constant,
    Shape::Line,
    Shape::Zeros,
    Shape::Cut(1),
    Shape::Cut(2),
    Shape::Cut(3),
];

/// Where a degenerate value goes: an array (of numbers, of rows of numbers,
/// or of objects -- the last only cut), or one numeric field of every object
/// of an array.
#[derive(Clone, Debug)]
enum Target {
    Array(String),
    Field(String, String),
}

fn is_numbers(v: &Value) -> bool {
    v.as_array()
        .is_some_and(|a| a.len() >= 2 && a.iter().all(Value::is_number))
}

fn is_rows(v: &Value) -> bool {
    v.as_array()
        .is_some_and(|a| a.len() >= 2 && a.iter().all(is_numbers))
}

fn is_objects(v: &Value) -> bool {
    v.as_array()
        .is_some_and(|a| a.len() >= 2 && a.iter().all(Value::is_object))
}

/// `values` made degenerate by `shape` (not a cut); integers stay integers,
/// so a count stays a whole number.
fn degenerate(values: &[Value], shape: Shape) -> Vec<Value> {
    let first = values[0].as_f64().expect("a number");
    let integral = values.iter().all(|x| x.is_i64() || x.is_u64());
    let num = |x: f64| {
        if integral {
            Value::from(x as i64)
        } else {
            Value::from(x)
        }
    };
    match shape {
        Shape::Constant => values.iter().map(|_| num(first)).collect(),
        Shape::Line => (0..values.len()).map(|i| num(first + i as f64)).collect(),
        Shape::Zeros => values.iter().map(|_| num(0.0)).collect(),
        Shape::Cut(_) => values.to_vec(),
    }
}

/// Every place in `v` a degenerate value can go.
fn targets(v: &Value, at: String, out: &mut Vec<Target>) {
    if is_numbers(v) || is_rows(v) {
        out.push(Target::Array(at));
        return;
    }
    if is_objects(v) {
        out.push(Target::Array(at.clone()));
        let a = v.as_array().expect("an array");
        let first = a[0].as_object().expect("an object");
        for key in first.keys() {
            if a.iter().all(|o| o.get(key).is_some_and(Value::is_number)) {
                out.push(Target::Field(at.clone(), key.clone()));
            }
        }
    }
    match v {
        Value::Object(m) => {
            for (k, x) in m {
                targets(x, format!("{at}/{k}"), out);
            }
        }
        Value::Array(a) => {
            for (i, x) in a.iter().enumerate() {
                targets(x, format!("{at}/{i}"), out);
            }
        }
        _ => {}
    }
}

/// `target` in `request` made degenerate by `shape`; `false` when the shape
/// does not apply there (a field is not cut).
fn apply(request: &mut Value, target: &Target, shape: Shape) -> bool {
    match (target, shape) {
        (Target::Array(p), Shape::Cut(n)) => {
            let a = request.pointer_mut(p).expect("read from it");
            let cut: Vec<Value> = a
                .as_array()
                .expect("an array")
                .iter()
                .take(n)
                .cloned()
                .collect();
            *a = Value::Array(cut);
            true
        }
        (Target::Array(p), _) => {
            let a = request.pointer_mut(p).expect("read from it");
            if is_numbers(a) {
                *a = Value::Array(degenerate(a.as_array().expect("an array"), shape));
                true
            } else if is_rows(a) {
                for row in a.as_array_mut().expect("an array") {
                    *row = Value::Array(degenerate(row.as_array().expect("a row"), shape));
                }
                true
            } else {
                false
            }
        }
        (Target::Field(_, _), Shape::Cut(_)) => false,
        (Target::Field(p, key), _) => {
            let a = request
                .pointer_mut(p)
                .and_then(Value::as_array_mut)
                .expect("read from it");
            let column: Vec<Value> = a.iter().map(|o| o[key.as_str()].clone()).collect();
            for (o, v) in a.iter_mut().zip(degenerate(&column, shape)) {
                o[key.as_str()] = v;
            }
            true
        }
    }
}

#[test]
fn degenerate_inputs_are_refused_or_answered_never_internal() {
    quiet_panics();
    let mut failures = Vec::new();
    let mut calls = 0;
    for (name, entry, base) in entry_points() {
        let mut found = Vec::new();
        targets(&base, String::new(), &mut found);
        let mut variants: Vec<(String, Value)> = Vec::new();
        for shape in SHAPES {
            let mut all = base.clone();
            let mut applied = 0;
            for t in &found {
                let mut one = base.clone();
                if apply(&mut one, t, shape) {
                    variants.push((format!("{t:?} {shape:?}"), one));
                }
                // Cuts are not combined: a cut array would move the pointers
                // of the others.
                if !matches!(shape, Shape::Cut(_)) && apply(&mut all, t, shape) {
                    applied += 1;
                }
            }
            if applied > 1 {
                variants.push((format!("all {shape:?}"), all));
            }
        }
        for (what, request) in variants {
            last_panic();
            let (_, body) = call(entry, &request);
            calls += 1;
            if body.get("code").and_then(Value::as_str) == Some("internal") {
                let why = last_panic();
                failures.push(format!("{name} [{what}]: {why}"));
            }
        }
    }
    assert!(calls > 0, "no degenerate request was made");
    assert!(
        failures.is_empty(),
        "{} of {calls} degenerate requests ended `internal`:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
