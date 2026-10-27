//! The JSON wire contract shared by the C FFI and the WASM bindings.
//!
//! Both transports carry the *same* analyses, so they must carry the same
//! shapes. They did not: the X-bar/R chart answered `x_bar_cl` over the FFI and
//! `xbar_cl` over WASM, the P chart took `[inspected, defective]` over one and
//! `[defectives, size]` over the other, and only the WASM side reported point
//! indices, rule violations or an in-control verdict. A consumer moving between
//! the two rewrote its parsing; a consumer reading the P chart pair in the wrong
//! order got a plausible number that was wrong.
//!
//! The types and functions here are the single definition. A transport module is
//! a thin shell over them: it decodes its own input representation, calls one of
//! these, and encodes the result. Nothing in this module names `JsValue` or a raw
//! pointer, which is what lets both sides compile it.
//!
//! **Where the two shapes disagreed, the crate's own model decided.** The pair
//! order follows `PChart::add_sample(defectives, sample_size)`; the field names
//! follow the chart types; the richer per-point payload is kept because dropping
//! it would lose the index a caller needs to line a violation up with its input
//! row.

use serde::{Deserialize, Serialize};

#[derive(Serialize, Debug)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(all(test, feature = "wasm"), derive(schemars::JsonSchema))]
#[cfg_attr(feature = "wasm", tsify(missing_as_null))]
pub(crate) struct XbarRChartDto {
    pub(crate) xbar_cl: f64,
    pub(crate) xbar_ucl: f64,
    pub(crate) xbar_lcl: f64,
    pub(crate) r_cl: f64,
    pub(crate) r_ucl: f64,
    pub(crate) r_lcl: f64,
    /// Short-term sigma implied by this chart (`R-bar / d2`). Feed it to
    /// `process_capability` as `sigma_within` -- it is the quantity a
    /// capability study needs and the one a flat measurement vector cannot
    /// carry.
    pub(crate) sigma_hat: f64,
    /// Degrees of freedom of `sigma_hat` -- `process_capability`'s
    /// `sigma_within_df`, for Cp/Cpk confidence intervals.
    pub(crate) sigma_hat_df: f64,
    pub(crate) xbar_points: Vec<ChartPointDto>,
    pub(crate) r_points: Vec<ChartPointDto>,
    pub(crate) in_control: bool,
}

#[derive(Serialize, Debug)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(all(test, feature = "wasm"), derive(schemars::JsonSchema))]
#[cfg_attr(feature = "wasm", tsify(missing_as_null))]
pub(crate) struct ChartPointDto {
    pub(crate) index: usize,
    pub(crate) value: f64,
    pub(crate) violations: Vec<String>,
}

#[derive(Serialize, Debug)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(all(test, feature = "wasm"), derive(schemars::JsonSchema))]
#[cfg_attr(feature = "wasm", tsify(missing_as_null))]
pub(crate) struct PChartDto {
    pub(crate) p_bar: f64,
    pub(crate) points: Vec<AttributeChartPointDto>,
    pub(crate) in_control: bool,
}

#[derive(Serialize, Debug)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(all(test, feature = "wasm"), derive(schemars::JsonSchema))]
#[cfg_attr(feature = "wasm", tsify(missing_as_null))]
pub(crate) struct AttributeChartPointDto {
    pub(crate) index: usize,
    pub(crate) value: f64,
    pub(crate) ucl: f64,
    pub(crate) cl: f64,
    pub(crate) lcl: f64,
    pub(crate) out_of_control: bool,
    /// Standardized value -- the scale on which every limit is +/-3, so zone
    /// run rules apply to charts whose limits vary with n.
    pub(crate) z: Option<f64>,
}

/// Known (Phase I) parameters an attributes chart can be given.
///
/// One type for all four charts, checked per chart: a key a chart does not
/// take is refused rather than ignored.
#[derive(Deserialize, Default, Debug)]
#[serde(deny_unknown_fields)]
pub(crate) struct AttributeStandardDto {
    #[serde(default)]
    pub(crate) p_bar: Option<f64>,
    #[serde(default)]
    pub(crate) u_bar: Option<f64>,
    #[serde(default)]
    pub(crate) phi: Option<f64>,
}

impl AttributeStandardDto {
    /// Refuses `forbidden` keys: `(name, present)`.
    pub(crate) fn refuse(&self, chart: &str, forbidden: &[(&str, bool)]) -> Result<(), WireError> {
        match forbidden.iter().find(|(_, present)| *present) {
            Some((name, _)) => Err(WireError::new(
                code::MALFORMED_INPUT,
                None,
                format!("options: {chart} does not take `{name}`"),
            )
            .about(name.to_string())),
            None => Ok(()),
        }
    }

    /// The Laney standard: both parts or neither.
    pub(crate) fn laney(
        &self,
        center: Option<f64>,
        center_name: &str,
    ) -> Result<Option<crate::spc::LaneyStandard>, WireError> {
        match (center, self.phi) {
            (Some(center), Some(phi)) => Ok(Some(crate::spc::LaneyStandard { center, phi })),
            (None, None) => Ok(None),
            // Named by the one that is missing.
            (Some(_), None) | (None, Some(_)) => Err(WireError::new(
                code::INVALID_OPTION,
                None,
                format!(
                    "options: `{center_name}` and `phi` are given together -- a Phase I \
                     standard fixes both, since phi scales the error about that centre"
                ),
            )
            .about(if center.is_some() {
                "phi".to_string()
            } else {
                center_name.to_string()
            })),
        }
    }
}

#[derive(Serialize, Debug)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(all(test, feature = "wasm"), derive(schemars::JsonSchema))]
#[cfg_attr(feature = "wasm", tsify(missing_as_null))]
pub(crate) struct LaneyPChartDto {
    pub(crate) p_bar: f64,
    pub(crate) phi: f64,
    pub(crate) points: Vec<AttributeChartPointDto>,
}

/// The value `table` gives `name`, or `unknown_option` at `parameter` listing
/// the table's names -- the parser and its refusal read one list.
pub(crate) fn named<T: Copy>(
    parameter: &'static str,
    name: &str,
    table: &[(&'static str, T)],
) -> Result<T, WireError> {
    table
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(_, v)| *v)
        .ok_or_else(|| {
            let expected: Vec<&'static str> = table.iter().map(|(n, _)| *n).collect();
            WireError::unknown_option(parameter, name, &expected)
        })
}

/// Parse an optional `{ rules }` options object into a rule set.
///
/// `rules` is the name of a set (`"shewhart"`, `"western_electric"`,
/// `"nelson"`) or an array of test names -- the values `violations` reports.
/// Absent, `undefined`, `null` or an object without `rules` all mean the
/// default set, [`RuleSet::shewhart`](crate::spc::RuleSet::shewhart). Split out
/// from the binding so it can be exercised without a `JsValue`.
pub(crate) fn rules_from_json(
    options: Option<serde_json::Value>,
) -> Result<crate::spc::RuleSet, WireError> {
    use crate::spc::{RuleSet, ViolationType};

    const EXPECTED: &str = "rules: expected the name of a rule set or an array of rule names";

    let Some(value) = options else {
        return Ok(RuleSet::default());
    };
    if value.is_null() {
        return Ok(RuleSet::default());
    }
    let Some(names) = value.get("rules") else {
        return Ok(RuleSet::default());
    };
    if names.is_null() {
        return Ok(RuleSet::default());
    }
    if let Some(set) = names.as_str() {
        return RuleSet::named(set)
            .ok_or_else(|| WireError::unknown_option("rules", set, &RuleSet::NAMES));
    }
    let names = names
        .as_array()
        .ok_or_else(|| WireError::new(code::MALFORMED_INPUT, None, EXPECTED).about("rules"))?;
    let rule_names: Vec<&str> = ViolationType::ALL.iter().map(|v| v.name()).collect();

    let mut set = RuleSet::none();
    for name in names {
        let name = name
            .as_str()
            .ok_or_else(|| WireError::new(code::MALFORMED_INPUT, None, EXPECTED).about("rules"))?;
        // The names are the values `violations` reports.
        let rule = ViolationType::from_name(name)
            .ok_or_else(|| WireError::unknown_option("rules", name, &rule_names))?;
        set = set.with(rule);
    }
    Ok(set)
}

pub(crate) fn point_dtos(points: &[crate::spc::ChartPoint]) -> Vec<ChartPointDto> {
    points
        .iter()
        .map(|p| ChartPointDto {
            index: p.index,
            value: p.value,
            violations: p.violations.iter().map(|&v| v.name().to_owned()).collect(),
        })
        .collect()
}

/// Checks the matrix a subgroup chart takes and returns its subgroup size.
///
/// A ragged subgroup is refused by its row: the chart would skip it, and every
/// later point would then carry an index one short of the row it came from.
///
/// The supported subgroup range is not restated here. It used to be, as a
/// literal `2..=10` alongside the same literal in the constructor, so widening
/// the crate's factor tables left the binding rejecting sizes the crate had
/// just learned to handle -- a disagreement no test in either crate could see,
/// because each one was right about its own copy.
pub(crate) fn subgroup_size(subgroups: &[Vec<f64>]) -> Result<usize, WireError> {
    subgroups.first().map(Vec::len).ok_or_else(|| {
        WireError::too_few(
            "subgroups",
            1,
            0,
            "subgroups: at least one subgroup required",
        )
    })
}

/// A refused input, in the one shape every transport reports.
///
/// A consumer with a data grid has to tell its user *which* row to fix and
/// *why*, in its own words. Free text carries neither in a form a program can
/// use -- and a count that failed to deserialize carried no row at all -- so
/// every consumer ended up re-validating the rules the crate already enforces.
/// `code` is stable across releases; `message` is for people and may change.
#[derive(Serialize, Debug, Clone, PartialEq)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(all(test, feature = "wasm"), derive(schemars::JsonSchema))]
#[cfg_attr(feature = "wasm", tsify(missing_as_null))]
pub(crate) struct WireError {
    /// Stable, machine-readable reason: the name of [`Self::reason`].
    pub(crate) code: &'static str,
    /// The reason as the table names it -- the refusal's declared fields.
    #[serde(skip)]
    #[cfg_attr(feature = "wasm", tsify(skip))]
    pub(crate) reason: Code,
    /// Position of the offending element in its input array, when there is one.
    pub(crate) index: Option<usize>,
    /// Name of the offending *option*, when the refusal is about one.
    ///
    /// `index` says where in the data; this says which knob. Without it the
    /// name lives only in `message`, which is the one field documented as free
    /// to change -- so a consumer wanting to point at the setting has to parse
    /// prose, or re-derive the rule itself.
    pub(crate) parameter: Option<std::borrow::Cow<'static, str>>,
    /// Human-readable description.
    pub(crate) message: String,
    /// The values behind the reason, by name: `min` / `max` (inclusive
    /// bounds, `null` on an unbounded or open side), `exclusive_min` /
    /// `exclusive_max` (only on an open side) and `got` for a value out of
    /// range, `got` and `expected`
    /// for an unknown option name, `min` and `got` for too few values. Set
    /// as properties next to `code` on the WebAssembly `Error`, and as keys
    /// next to `code` in the C error body.
    #[serde(skip)]
    pub(crate) details: Vec<(&'static str, Detail)>,
}

/// A value carried in [`WireError::details`].
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Detail {
    Num(f64),
    Text(String),
    List(Vec<&'static str>),
    Null,
}

impl Detail {
    /// The value as JSON — the C error body's rendering.
    #[cfg(feature = "ffi")]
    pub(crate) fn to_json(&self) -> serde_json::Value {
        match self {
            Detail::Num(n) => serde_json::json!(n),
            Detail::Text(s) => serde_json::json!(s),
            Detail::List(v) => serde_json::json!(v),
            Detail::Null => serde_json::Value::Null,
        }
    }
}

/// One end of the range a number must fall in.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Bound {
    /// The bound itself is accepted: sent as `min` / `max`.
    Closed(f64),
    /// The bound itself is refused: sent as `exclusive_min` / `exclusive_max`,
    /// with `min` / `max` `null`.
    Open(f64),
    /// No limit on this side: `min` / `max` `null`.
    Unbounded,
}

impl Bound {
    /// The end a public error type states as an inclusive / exclusive pair.
    pub(crate) fn from_pair(inclusive: Option<f64>, exclusive: Option<f64>) -> Self {
        match (inclusive, exclusive) {
            (Some(v), _) => Bound::Closed(v),
            (None, Some(v)) => Bound::Open(v),
            (None, None) => Bound::Unbounded,
        }
    }
}

/// Whether `v` lies in `low`..`high` (a NaN never does).
pub(crate) fn within(v: f64, low: Bound, high: Bound) -> bool {
    let above = match low {
        Bound::Closed(b) => v >= b,
        Bound::Open(b) => v > b,
        Bound::Unbounded => !v.is_nan(),
    };
    let below = match high {
        Bound::Closed(b) => v <= b,
        Bound::Open(b) => v < b,
        Bound::Unbounded => !v.is_nan(),
    };
    above && below
}

/// The range in words, for a message: `> 0`, `>= 0`, `in (0, 1]`.
pub(crate) fn describe(low: Bound, high: Bound) -> String {
    match (low, high) {
        (Bound::Unbounded, Bound::Unbounded) => "a finite number".to_owned(),
        (Bound::Closed(a), Bound::Unbounded) => format!(">= {a}"),
        (Bound::Open(a), Bound::Unbounded) => format!("> {a}"),
        (Bound::Unbounded, Bound::Closed(b)) => format!("<= {b}"),
        (Bound::Unbounded, Bound::Open(b)) => format!("< {b}"),
        (low, high) => {
            let (l, a) = match low {
                Bound::Closed(a) => ('[', a),
                Bound::Open(a) => ('(', a),
                Bound::Unbounded => unreachable!("matched above"),
            };
            let (h, b) = match high {
                Bound::Closed(b) => (']', b),
                Bound::Open(b) => (')', b),
                Bound::Unbounded => unreachable!("matched above"),
            };
            format!("in {l}{a}, {b}{h}")
        }
    }
}

/// A whole number as the caller sent it: any JSON integer, or a float with no
/// fractional part (`3.0`).
///
/// Count options are read as this rather than as `usize`. An unsigned type
/// refuses `-1` while deserializing -- as a shape error, before the range
/// check that refuses `0` or `1` for the same option ever runs -- so two
/// values wrong for the same reason left by different doors. A fraction is
/// still refused while deserializing (`malformed_input` at its path, "expected
/// a whole number"): whether a value is whole is a matter of its type, whether
/// it is in range a matter of its value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Whole(pub(crate) f64);

/// On the wire a [`Whole`] is a whole number -- its range is the field's to
/// check.
#[cfg(all(test, feature = "wasm"))]
impl schemars::JsonSchema for Whole {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        "Whole".into()
    }

    fn inline_schema() -> bool {
        true
    }

    fn json_schema(_: &mut schemars::SchemaGenerator) -> schemars::Schema {
        schemars::json_schema!({ "type": "integer" })
    }
}

impl Whole {
    /// The value as a count, once a range check has accepted it as `>= 0`.
    pub(crate) fn count(self) -> usize {
        debug_assert!(self.0 >= 0.0, "a count is checked non-negative first");
        // Saturates above `usize::MAX`, which no range a count option has
        // reaches.
        self.0 as usize
    }
}

impl From<usize> for Whole {
    fn from(n: usize) -> Self {
        Whole(n as f64)
    }
}

impl<'de> Deserialize<'de> for Whole {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct Visitor;
        impl serde::de::Visitor<'_> for Visitor {
            type Value = Whole;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a whole number")
            }
            fn visit_i64<E: serde::de::Error>(self, v: i64) -> Result<Whole, E> {
                Ok(Whole(v as f64))
            }
            fn visit_u64<E: serde::de::Error>(self, v: u64) -> Result<Whole, E> {
                Ok(Whole(v as f64))
            }
            fn visit_f64<E: serde::de::Error>(self, v: f64) -> Result<Whole, E> {
                if v.is_finite() && v.fract() == 0.0 {
                    Ok(Whole(v))
                } else {
                    Err(E::invalid_type(serde::de::Unexpected::Float(v), &self))
                }
            }
        }
        deserializer.deserialize_f64(Visitor)
    }
}

/// Where in the argument `param` a request stopped deserializing (both
/// transports read requests through `serde_path_to_error`), as the
/// refusal reports it: the field (`design[1]`, `points[0].y`) and, when the
/// failure sits in an array, its position there -- the same `parameter` /
/// `index` a number array read element by element reports. Without it a
/// `null` three levels down was "invalid type: null, expected f64" with no
/// way to say which row (the gap `read_numbers` closed for bare arrays).
pub(crate) fn failure_site(
    param: &str,
    path: &serde_path_to_error::Path,
) -> (String, Option<usize>) {
    use serde_path_to_error::Segment;
    let segments: Vec<&Segment> = path.iter().collect();
    let index = segments.iter().rev().find_map(|s| match s {
        Segment::Seq { index } => Some(*index),
        _ => None,
    });
    // A trailing `[i]` is the index, not part of the name.
    let named = match segments.last() {
        Some(Segment::Seq { .. }) => &segments[..segments.len() - 1],
        _ => &segments[..],
    };
    let mut name = String::new();
    for segment in named {
        match segment {
            Segment::Seq { index } => name.push_str(&format!("[{index}]")),
            Segment::Map { key } | Segment::Enum { variant: key } => {
                if !name.is_empty() {
                    name.push('.');
                }
                name.push_str(key);
            }
            Segment::Unknown => name.push_str(".?"),
        }
    }
    // A top-level array argument (`data[1][2]`) or a failure at the root
    // (a missing field) is named by the argument itself.
    if name.is_empty() || name.starts_with('[') {
        name.insert_str(0, param);
    }
    (name, index)
}

/// A refusal reason: the `code` a refusal carries (the WebAssembly `Error`'s,
/// the C error body's) and the fields that can
/// stand beside it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, serde::Deserialize)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(all(test, feature = "wasm"), derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub(crate) struct RefusalCode {
    pub(crate) code: String,
    pub(crate) fields: Vec<String>,
}

/// A violation a checker reports. This package has no checker; the type is
/// here so every engine's [`Codes`] has one shape.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, serde::Deserialize)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(all(test, feature = "wasm"), derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub(crate) struct ViolationCode {
    pub(crate) code: String,
    pub(crate) fields: Vec<String>,
    pub(crate) checkers: Vec<String>,
}

/// What this package can answer with besides a result: every refusal
/// `code`, and no violations -- it has no checker.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, serde::Deserialize)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(all(test, feature = "wasm"), derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub(crate) struct Codes {
    pub(crate) refusals: Vec<RefusalCode>,
    pub(crate) violations: Vec<ViolationCode>,
}

pub(crate) fn code_catalog() -> Codes {
    Codes {
        refusals: Code::ALL
            .into_iter()
            .map(|c| RefusalCode {
                code: c.name().to_string(),
                fields: c.fields().iter().map(|f| f.to_string()).collect(),
            })
            .collect(),
        violations: vec![],
    }
}

/// Every reason the crate's transports refuse with: the stable `code` a
/// refusal carries, and the fields that can stand beside it. A [`WireError`]
/// holds one of these (named through [`code`]), so the published list
/// ([`Code::ALL`]) is the list the crate refuses from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Code {
    InvalidInput,
    MalformedInput,
    CountNotWhole,
    SampleSizeNotWhole,
    DefectivesExceedSample,
    UnitsNotPositive,
    ValueNotFinite,
    InsufficientData,
    ParameterOutOfRange,
    EmptyInput,
    NonPositiveData,
    NoSpecification,
    InvalidOption,
    DimensionMismatch,
    UnknownOption,
    EventsUnordered,
    EventAfterEnd,
    /// The package failed, not the input.
    Internal,
}

impl Code {
    pub(crate) const ALL: [Code; 18] = [
        Code::InvalidInput,
        Code::MalformedInput,
        Code::CountNotWhole,
        Code::SampleSizeNotWhole,
        Code::DefectivesExceedSample,
        Code::UnitsNotPositive,
        Code::ValueNotFinite,
        Code::InsufficientData,
        Code::ParameterOutOfRange,
        Code::EmptyInput,
        Code::NonPositiveData,
        Code::NoSpecification,
        Code::InvalidOption,
        Code::DimensionMismatch,
        Code::UnknownOption,
        Code::EventsUnordered,
        Code::EventAfterEnd,
        Code::Internal,
    ];

    pub(crate) fn name(self) -> &'static str {
        match self {
            Code::InvalidInput => "invalid_input",
            Code::MalformedInput => "malformed_input",
            Code::CountNotWhole => "count_not_whole",
            Code::SampleSizeNotWhole => "sample_size_not_whole",
            Code::DefectivesExceedSample => "defectives_exceed_sample",
            Code::UnitsNotPositive => "units_not_positive",
            Code::ValueNotFinite => "value_not_finite",
            Code::InsufficientData => "insufficient_data",
            Code::ParameterOutOfRange => "parameter_out_of_range",
            Code::EmptyInput => "empty_input",
            Code::NonPositiveData => "non_positive_data",
            Code::NoSpecification => "no_specification",
            Code::InvalidOption => "invalid_option",
            Code::DimensionMismatch => "dimension_mismatch",
            Code::UnknownOption => "unknown_option",
            Code::EventsUnordered => "events_unordered",
            Code::EventAfterEnd => "event_after_end",
            Code::Internal => "internal",
        }
    }

    /// The fields a refusal with this code can carry besides `code` and
    /// `message` (`index` and `parameter` are always there, `null` when they
    /// do not apply).
    pub(crate) fn fields(self) -> &'static [&'static str] {
        match self {
            Code::InvalidInput => &["index", "parameter"],
            Code::MalformedInput => &["index", "parameter"],
            Code::CountNotWhole => &["index", "parameter"],
            Code::SampleSizeNotWhole => &["index", "parameter"],
            Code::DefectivesExceedSample => &["index", "parameter"],
            Code::UnitsNotPositive => &["index", "parameter"],
            Code::ValueNotFinite => &["index", "parameter"],
            Code::InsufficientData => &["index", "parameter", "min", "got"],
            Code::ParameterOutOfRange => &[
                "index",
                "parameter",
                "min",
                "max",
                "exclusive_min",
                "exclusive_max",
                "got",
            ],
            Code::EmptyInput => &["index", "parameter"],
            Code::NonPositiveData => &["index", "parameter", "got"],
            Code::NoSpecification => &["index", "parameter", "expected"],
            Code::InvalidOption => &["index", "parameter"],
            Code::DimensionMismatch => &["index", "parameter", "expected", "got"],
            Code::UnknownOption => &["index", "parameter", "got", "expected"],
            Code::EventsUnordered => &["index", "parameter"],
            Code::EventAfterEnd => &["index", "parameter"],
            Code::Internal => &["index", "parameter"],
        }
    }
}

impl PartialEq<&str> for Code {
    fn eq(&self, other: &&str) -> bool {
        self.name() == *other
    }
}

impl PartialEq<Code> for &str {
    fn eq(&self, other: &Code) -> bool {
        *self == other.name()
    }
}

/// The codes by their constant names, as the transports and their tests name
/// them.
pub(crate) mod code {
    /// Any refusal without a more specific code.
    pub(crate) const INVALID_INPUT: super::Code = super::Code::InvalidInput;
    /// The input is not the shape the function takes (not an array, a pair
    /// with the wrong arity, a value of the wrong type).
    pub(crate) const MALFORMED_INPUT: super::Code = super::Code::MalformedInput;
    /// A count that is not a whole number >= 0.
    pub(crate) const COUNT_NOT_WHOLE: super::Code = super::Code::CountNotWhole;
    /// A sample size that is not a whole number >= 1.
    ///
    /// One code for the whole field, not one per way of being wrong: a size of
    /// `0`, `-3`, `10.5` or a NaN are the same mistake to whoever typed the row,
    /// and splitting them made `code` insufficient on its own -- the consumer
    /// had to read `message` to learn whether a `count_not_whole` was about the
    /// defectives or the size beside them.
    pub(crate) const SAMPLE_SIZE_NOT_WHOLE: super::Code = super::Code::SampleSizeNotWhole;
    /// More defectives than the sample has items.
    pub(crate) const DEFECTIVES_EXCEED_SAMPLE: super::Code = super::Code::DefectivesExceedSample;
    /// Units inspected that are not a positive, finite number.
    pub(crate) const UNITS_NOT_POSITIVE: super::Code = super::Code::UnitsNotPositive;
    /// A NaN or an infinity where a measurement belongs.
    pub(crate) const VALUE_NOT_FINITE: super::Code = super::Code::ValueNotFinite;
    /// Fewer samples than the chart needs.
    pub(crate) const INSUFFICIENT_DATA: super::Code = super::Code::InsufficientData;
    /// A value outside its domain -- an option, a Phase I standard, a subgroup
    /// size the factor tables do not cover, an array element. `parameter`
    /// names which, `min` / `max` / `exclusive_*` the domain, `got` the value.
    pub(crate) const PARAMETER_OUT_OF_RANGE: super::Code = super::Code::ParameterOutOfRange;
    /// An argument with nothing in it.
    pub(crate) const EMPTY_INPUT: super::Code = super::Code::EmptyInput;
    /// A value that must be `> 0` is not (Box-Cox data, failure times).
    pub(crate) const NON_POSITIVE_DATA: super::Code = super::Code::NonPositiveData;
    /// A capability index with neither specification limit: there is nothing
    /// to measure the spread against. `expected` lists the two options.
    pub(crate) const NO_SPECIFICATION: super::Code = super::Code::NoSpecification;
    /// A setting refused for a reason a range cannot state (`[lo, hi]` with
    /// `lo >= hi`).
    pub(crate) const INVALID_OPTION: super::Code = super::Code::InvalidOption;
    /// Arrays that must have the same length do not.
    pub(crate) const DIMENSION_MISMATCH: super::Code = super::Code::DimensionMismatch;
    /// An option given a name the function does not know. `parameter` names it.
    pub(crate) const UNKNOWN_OPTION: super::Code = super::Code::UnknownOption;
    /// Event times that go backwards.
    pub(crate) const EVENTS_UNORDERED: super::Code = super::Code::EventsUnordered;
    /// An event time after the end of observation.
    pub(crate) const EVENT_AFTER_END: super::Code = super::Code::EventAfterEnd;
    /// The package failed, not the input: a result it could not hand back.
    pub(crate) const INTERNAL: super::Code = super::Code::Internal;
}

impl WireError {
    pub(crate) fn new(code: Code, index: Option<usize>, message: impl Into<String>) -> Self {
        WireError {
            code: code.name(),
            reason: code,
            index,
            parameter: None,
            message: message.into(),
            details: Vec::new(),
        }
    }

    /// Adds one of the values behind the reason.
    pub(crate) fn with(mut self, key: &'static str, value: Detail) -> Self {
        debug_assert!(
            self.reason.fields().contains(&key),
            "{} refusal with an undeclared field: {key}",
            self.code
        );
        self.details.push((key, value));
        self
    }

    /// An option given a name the function does not know: `parameter`, `got`
    /// and the names it does know as `expected`.
    pub(crate) fn unknown_option(
        parameter: &'static str,
        got: &str,
        expected: &[&'static str],
    ) -> Self {
        Self::new(
            code::UNKNOWN_OPTION,
            None,
            format!(
                "{parameter}: unknown name {got:?}; expected one of {}",
                expected.join(", ")
            ),
        )
        .about(parameter)
        .with("got", Detail::Text(got.to_string()))
        .with("expected", Detail::List(expected.to_vec()))
    }

    /// A number outside the range `low`..`high`. `min` and `max` are always
    /// sent -- the bound when that end is closed, `null` otherwise -- and an
    /// open end adds `exclusive_min` / `exclusive_max`.
    pub(crate) fn out_of_range(
        parameter: impl Into<std::borrow::Cow<'static, str>>,
        low: Bound,
        high: Bound,
        got: f64,
        message: impl Into<String>,
    ) -> Self {
        let mut e = Self::new(code::PARAMETER_OUT_OF_RANGE, None, message).about(parameter);
        for (closed, open, end) in [
            ("min", "exclusive_min", low),
            ("max", "exclusive_max", high),
        ] {
            match end {
                Bound::Closed(v) => e = e.with(closed, Detail::Num(v)),
                Bound::Open(v) => e = e.with(closed, Detail::Null).with(open, Detail::Num(v)),
                Bound::Unbounded => e = e.with(closed, Detail::Null),
            }
        }
        e.with("got", Detail::Num(got))
    }

    /// Fewer values than needed: `parameter`, `min` and `got`.
    pub(crate) fn too_few(
        parameter: impl Into<std::borrow::Cow<'static, str>>,
        min: usize,
        got: usize,
        message: impl Into<String>,
    ) -> Self {
        Self::new(code::INSUFFICIENT_DATA, None, message)
            .about(parameter)
            .with("min", Detail::Num(min as f64))
            .with("got", Detail::Num(got as f64))
    }

    /// An argument with nothing in it.
    pub(crate) fn empty_input(parameter: &'static str) -> Self {
        Self::new(
            code::EMPTY_INPUT,
            None,
            format!("{parameter} must not be empty"),
        )
        .about(parameter)
    }

    /// Names the option a refusal is about, alongside its code.
    pub(crate) fn about(mut self, parameter: impl Into<std::borrow::Cow<'static, str>>) -> Self {
        self.parameter = Some(parameter.into());
        self
    }

    /// A refusal without a more specific code.
    pub(crate) fn invalid_input(message: impl Into<String>) -> Self {
        Self::new(code::INVALID_INPUT, None, message)
    }

    /// Too few samples, with the message naming how many are needed.
    pub(crate) fn insufficient_data(message: impl Into<String>) -> Self {
        Self::new(code::INSUFFICIENT_DATA, None, message)
    }

    /// A chart's refusal of one element, placed at `label[index]`, or of the
    /// input as a whole when `index` is `None`.
    pub(crate) fn chart(
        label: &str,
        index: Option<usize>,
        error: &crate::spc::ControlChartError,
    ) -> Self {
        let message = match index {
            Some(i) => format!("{label}[{i}]: {error}"),
            None => error.to_string(),
        };
        use crate::spc::ControlChartError as E;
        let mut wire = match error {
            // The domain error names the parameter and carries the domain; the
            // wire record should not make a consumer read either back out of
            // the message.
            E::InvalidStandard {
                parameter,
                got,
                min,
                exclusive_min,
                max,
                exclusive_max,
                ..
            } => Self::out_of_range(
                *parameter,
                Bound::from_pair(*min, *exclusive_min),
                Bound::from_pair(*max, *exclusive_max),
                *got,
                message,
            ),
            // The size is the width of the subgroups given.
            E::SubgroupSizeOutOfRange { got, min, max } => Self::out_of_range(
                label.to_string(),
                Bound::Closed(*min as f64),
                Bound::Closed(*max as f64),
                *got as f64,
                message,
            ),
            // Same reason as every other pair of lengths that must agree.
            E::SampleLengthMismatch { expected, got } => {
                Self::new(code::DIMENSION_MISMATCH, index, message)
                    .about(label.to_string())
                    .with("expected", Detail::Num(*expected as f64))
                    .with("got", Detail::Num(*got as f64))
            }
            _ => Self::new(chart_error_code(error), index, message),
        };
        wire.index = index;
        wire
    }

    /// A whole-slice chart's refusal.
    pub(crate) fn chart_input(label: &str, error: &crate::spc::ChartInputError) -> Self {
        use crate::spc::ChartInputError;
        match error {
            ChartInputError::Sample { index, error } => Self::chart(label, Some(*index), error),
            ChartInputError::TooFewSamples { min, got } => {
                Self::new(code::INSUFFICIENT_DATA, None, format!("{label}: {error}"))
                    .about(label.to_string())
                    .with("min", Detail::Num(*min as f64))
                    .with("got", Detail::Num(*got as f64))
            }
            ChartInputError::Standard(error) => Self::chart(label, None, error),
        }
    }
}

impl std::fmt::Display for WireError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl From<String> for WireError {
    fn from(message: String) -> Self {
        Self::invalid_input(message)
    }
}

impl From<&String> for WireError {
    fn from(message: &String) -> Self {
        Self::invalid_input(message.as_str())
    }
}

impl From<&WireError> for WireError {
    fn from(error: &WireError) -> Self {
        error.clone()
    }
}

impl From<&str> for WireError {
    fn from(message: &str) -> Self {
        Self::invalid_input(message)
    }
}

pub(crate) fn chart_error_code(error: &crate::spc::ControlChartError) -> Code {
    use crate::spc::ControlChartError as E;
    match error {
        E::SampleLengthMismatch { .. } => code::DIMENSION_MISMATCH,
        E::NonFiniteValue => code::VALUE_NOT_FINITE,
        E::DefectivesExceedSampleSize { .. } => code::DEFECTIVES_EXCEED_SAMPLE,
        E::NonPositiveUnits => code::UNITS_NOT_POSITIVE,
        // A domain the error states as bounds: `parameter` (with the bounds)
        // tells the two apart.
        E::SubgroupSizeOutOfRange { .. } | E::InvalidStandard { .. } => {
            code::PARAMETER_OUT_OF_RANGE
        }
        E::ZeroSampleSize => code::SAMPLE_SIZE_NOT_WHOLE,
    }
}

/// Feeds rows to a chart, naming the row a rejection came from. The chart
/// knows why a sample is unusable but not where it sat in the caller's input.
pub(crate) fn add_rows<T>(
    rows: impl IntoIterator<Item = T>,
    label: &str,
    mut add: impl FnMut(T) -> Result<(), crate::spc::ControlChartError>,
) -> Result<(), WireError> {
    for (i, row) in rows.into_iter().enumerate() {
        add(row).map_err(|e| WireError::chart(label, Some(i), &e))?;
    }
    Ok(())
}

/// Largest count a JSON number carries exactly (2^53).
const MAX_EXACT_COUNT: f64 = 9_007_199_254_740_992.0;

/// What a whole number means for one input field: its floor, the name it goes
/// by in a message, and the code its refusal carries.
///
/// The two fields of a `[defectives, sample_size]` pair take the same *kind* of
/// value and different *domains*, and reporting both under one code left the
/// difference readable only in the message text.
#[derive(Clone, Copy)]
struct CountDomain {
    min: u64,
    noun: &'static str,
    code: Code,
}

/// A count of things observed: a whole number >= 0.
const COUNT: CountDomain = CountDomain {
    min: 0,
    noun: "a count",
    code: code::COUNT_NOT_WHOLE,
};

/// The size of a sample: a whole number >= 1. Zero is refused here rather than
/// downstream so that every bad size -- zero, negative, fractional, not a
/// number at all -- leaves by the same door.
const SAMPLE_SIZE: CountDomain = CountDomain {
    min: 1,
    noun: "a sample size",
    code: code::SAMPLE_SIZE_NOT_WHOLE,
};

/// A whole number in `domain`. Read from a JSON number of either kind, so a
/// fractional or negative value is refused *here*, with its position, rather
/// than by a deserializer that knows neither.
fn whole_count(
    value: &serde_json::Value,
    at: &str,
    index: Option<usize>,
    domain: CountDomain,
) -> Result<u64, WireError> {
    let CountDomain { min, noun, code } = domain;
    if let Some(n) = value.as_u64() {
        if n >= min {
            return Ok(n);
        }
    }
    let refuse = |shown: String| {
        WireError::new(
            code,
            index,
            format!("{at}: {noun} must be a whole number of at least {min}, got {shown}"),
        )
    };
    match value.as_f64() {
        Some(x) if x >= min as f64 && x.fract() == 0.0 && x <= MAX_EXACT_COUNT => Ok(x as u64),
        Some(x) => Err(refuse(x.to_string())),
        None => Err(refuse(value.to_string())),
    }
}

fn as_array<'a>(
    value: &'a serde_json::Value,
    at: &str,
    expected: &str,
) -> Result<&'a Vec<serde_json::Value>, WireError> {
    value.as_array().ok_or_else(|| {
        WireError::new(
            code::MALFORMED_INPUT,
            None,
            format!("{at}: expected {expected}"),
        )
    })
}

/// The `[a, b]` pair at `label[i]`.
fn pair<'a>(
    row: &'a serde_json::Value,
    label: &str,
    i: usize,
    expected: &str,
) -> Result<(&'a serde_json::Value, &'a serde_json::Value), WireError> {
    match row.as_array().map(Vec::as_slice) {
        Some([a, b]) => Ok((a, b)),
        _ => Err(WireError::new(
            code::MALFORMED_INPUT,
            Some(i),
            format!("{label}[{i}]: expected a pair {expected}, got {row}"),
        )),
    }
}

/// `[c1, c2, ...]` counts.
// Only the WASM binding carries the NP, C and U charts; the FFI has no caller.
#[cfg_attr(not(feature = "wasm"), allow(dead_code))]
pub(crate) fn count_rows(value: &serde_json::Value, label: &str) -> Result<Vec<u64>, WireError> {
    as_array(value, label, "an array of counts")?
        .iter()
        .enumerate()
        .map(|(i, v)| whole_count(v, &format!("{label}[{i}]"), Some(i), COUNT))
        .collect::<Result<Vec<_>, _>>()
        // Every refusal in the rows is about this argument.
        .map_err(|e: WireError| e.about(label.to_owned()))
}

/// The one sample size an NP chart applies to every row. Not an element of an
/// array, so its refusal carries no index.
// Only the WASM binding carries the NP, C and U charts; the FFI has no caller.
#[cfg_attr(not(feature = "wasm"), allow(dead_code))]
pub(crate) fn sample_size_value(value: &serde_json::Value, label: &str) -> Result<u64, WireError> {
    whole_count(value, label, None, SAMPLE_SIZE).map_err(|e| e.about(label.to_owned()))
}

/// `[[defectives, sample_size], ...]` pairs.
pub(crate) fn count_pairs(
    value: &serde_json::Value,
    label: &str,
) -> Result<Vec<(u64, u64)>, WireError> {
    const SHAPE: &str = "[defectives, sample_size]";
    as_array(value, label, &format!("an array of {SHAPE} pairs"))?
        .iter()
        .enumerate()
        .map(|(i, row)| {
            let (d, n) = pair(row, label, i, SHAPE)?;
            Ok((
                whole_count(d, &format!("{label}[{i}][0] (defectives)"), Some(i), COUNT)?,
                whole_count(
                    n,
                    &format!("{label}[{i}][1] (sample size)"),
                    Some(i),
                    SAMPLE_SIZE,
                )?,
            ))
        })
        .collect::<Result<Vec<_>, _>>()
        // Every refusal in the rows is about this argument.
        .map_err(|e: WireError| e.about(label.to_owned()))
}

/// `[[defects, units], ...]` pairs; `units` may be fractional.
// Only the WASM binding carries the NP, C and U charts; the FFI has no caller.
#[cfg_attr(not(feature = "wasm"), allow(dead_code))]
pub(crate) fn rate_pairs(
    value: &serde_json::Value,
    label: &str,
) -> Result<Vec<(u64, f64)>, WireError> {
    const SHAPE: &str = "[defects, units]";
    as_array(value, label, &format!("an array of {SHAPE} pairs"))?
        .iter()
        .enumerate()
        .map(|(i, row)| {
            let (d, u) = pair(row, label, i, SHAPE)?;
            let defects = whole_count(d, &format!("{label}[{i}][0] (defects)"), Some(i), COUNT)?;
            let units = u.as_f64().ok_or_else(|| {
                WireError::new(
                    code::UNITS_NOT_POSITIVE,
                    Some(i),
                    format!("{label}[{i}][1] (units): expected a positive number, got {u}"),
                )
            })?;
            Ok((defects, units))
        })
        .collect::<Result<Vec<_>, _>>()
        // Every refusal in the rows is about this argument.
        .map_err(|e: WireError| e.about(label.to_owned()))
}

pub(crate) fn attribute_point_dtos(
    points: &[crate::spc::AttributeChartPoint],
) -> Vec<AttributeChartPointDto> {
    points
        .iter()
        .map(|p| AttributeChartPointDto {
            index: p.index,
            value: p.value,
            ucl: p.ucl,
            cl: p.cl,
            lcl: p.lcl,
            out_of_control: p.out_of_control,
            z: p.z,
        })
        .collect()
}

pub(crate) fn laney_point_dtos(
    points: &[crate::spc::LaneyAttributePoint],
) -> Vec<AttributeChartPointDto> {
    points
        .iter()
        .map(|p| AttributeChartPointDto {
            index: p.index,
            value: p.value,
            ucl: p.ucl,
            cl: p.cl,
            lcl: p.lcl,
            out_of_control: p.out_of_control,
            z: p.z,
        })
        .collect()
}

pub(crate) fn xbar_r_dto(
    subgroups: Vec<Vec<f64>>,
    rules: crate::spc::RuleSet,
) -> Result<XbarRChartDto, WireError> {
    use crate::spc::{ControlChart, XBarRChart};

    let n = subgroup_size(&subgroups)?;
    let mut chart = XBarRChart::new(n)
        .map_err(|e| WireError::chart("subgroups", None, &e))?
        .with_rules(rules);
    add_rows(&subgroups, "subgroups", |g| chart.add_sample(g))?;
    let too_few = || {
        WireError::too_few(
            "subgroups",
            1,
            0,
            "subgroups: at least one subgroup is needed",
        )
    };
    let x = chart.control_limits().ok_or_else(too_few)?;
    let r = chart.r_limits().ok_or_else(too_few)?;
    Ok(XbarRChartDto {
        xbar_cl: x.cl,
        xbar_ucl: x.ucl,
        xbar_lcl: x.lcl,
        r_cl: r.cl,
        r_ucl: r.ucl,
        r_lcl: r.lcl,
        // Present whenever the limits are: it is their centre line over d2 / c4.
        sigma_hat: chart.sigma_hat().ok_or_else(too_few)?,
        sigma_hat_df: chart.sigma_hat_df().ok_or_else(too_few)?,
        xbar_points: point_dtos(chart.points()),
        r_points: point_dtos(chart.r_points()),
        in_control: chart.is_in_control(),
    })
}

pub(crate) fn p_chart_dto(
    samples: &[(u64, u64)],
    standard: &AttributeStandardDto,
) -> Result<PChartDto, WireError> {
    use crate::spc::PChart;

    standard.refuse(
        "p_chart",
        &[
            ("u_bar", standard.u_bar.is_some()),
            ("phi", standard.phi.is_some()),
        ],
    )?;
    let mut chart = match standard.p_bar {
        Some(p) => PChart::with_center(p).map_err(|e| WireError::chart("options", None, &e))?,
        None => PChart::new(),
    };
    add_rows(samples, "samples", |&(d, n)| chart.add_sample(d, n))?;
    let p_bar = chart.p_bar().ok_or_else(|| {
        WireError::too_few("samples", 1, 0, "samples: at least 1 sample is needed")
    })?;
    Ok(PChartDto {
        p_bar,
        points: attribute_point_dtos(chart.points()),
        in_control: chart.is_in_control(),
    })
}

pub(crate) fn laney_p_dto(
    samples: &[(u64, u64)],
    standard: &AttributeStandardDto,
) -> Result<LaneyPChartDto, WireError> {
    standard.refuse("laney_p_chart", &[("u_bar", standard.u_bar.is_some())])?;
    let laney = standard.laney(standard.p_bar, "p_bar")?;
    let chart = crate::spc::laney_p_chart(samples, laney)
        .map_err(|e| WireError::chart_input("samples", &e))?;
    Ok(LaneyPChartDto {
        p_bar: chart.p_bar,
        phi: chart.phi,
        points: laney_point_dtos(&chart.points),
    })
}

// ---------------------------------------------------------------------------
// Attributes and rare-event charts beyond P / Laney P'
// ---------------------------------------------------------------------------
//
// Shared by the WebAssembly and C transports, so a row refused over one is
// refused over the other with the same code and index.

/// NP and C charts: one set of limits for every point.
#[derive(Serialize, Debug)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(all(test, feature = "wasm"), derive(schemars::JsonSchema))]
#[cfg_attr(feature = "wasm", tsify(missing_as_null))]
pub(crate) struct FixedLimitChartDto {
    pub(crate) cl: f64,
    pub(crate) ucl: f64,
    pub(crate) lcl: f64,
    pub(crate) points: Vec<AttributeChartPointDto>,
    pub(crate) in_control: bool,
}

#[derive(Serialize, Debug)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(all(test, feature = "wasm"), derive(schemars::JsonSchema))]
#[cfg_attr(feature = "wasm", tsify(missing_as_null))]
pub(crate) struct UChartDto {
    pub(crate) u_bar: f64,
    pub(crate) points: Vec<AttributeChartPointDto>,
    pub(crate) in_control: bool,
}

#[derive(Serialize, Debug)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(all(test, feature = "wasm"), derive(schemars::JsonSchema))]
#[cfg_attr(feature = "wasm", tsify(missing_as_null))]
pub(crate) struct LaneyUChartDto {
    pub(crate) u_bar: f64,
    pub(crate) phi: f64,
    pub(crate) points: Vec<AttributeChartPointDto>,
}
#[derive(Serialize, Debug)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(all(test, feature = "wasm"), derive(schemars::JsonSchema))]
#[cfg_attr(feature = "wasm", tsify(missing_as_null))]
pub(crate) struct GChartDto {
    pub(crate) g_bar: f64,
    pub(crate) points: Vec<GChartPointDto>,
}

#[derive(Serialize, Debug)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(all(test, feature = "wasm"), derive(schemars::JsonSchema))]
#[cfg_attr(feature = "wasm", tsify(missing_as_null))]
pub(crate) struct GChartPointDto {
    pub(crate) index: usize,
    pub(crate) value: f64,
    pub(crate) ucl: f64,
    pub(crate) cl: f64,
    pub(crate) lcl: f64,
    pub(crate) out_of_control: bool,
}

#[derive(Serialize, Debug)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(all(test, feature = "wasm"), derive(schemars::JsonSchema))]
#[cfg_attr(feature = "wasm", tsify(missing_as_null))]
pub(crate) struct TChartDto {
    pub(crate) t_bar: f64,
    pub(crate) points: Vec<TChartPointDto>,
}

#[derive(Serialize, Debug)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(all(test, feature = "wasm"), derive(schemars::JsonSchema))]
#[cfg_attr(feature = "wasm", tsify(missing_as_null))]
pub(crate) struct TChartPointDto {
    pub(crate) index: usize,
    pub(crate) value: f64,
    pub(crate) ucl: f64,
    pub(crate) cl: f64,
    pub(crate) lcl: f64,
    pub(crate) out_of_control: bool,
}

/// `values` with at least `min` entries, or `insufficient_data` naming both.
pub(crate) fn at_least(
    values: Vec<f64>,
    min: usize,
    param: &'static str,
) -> Result<Vec<f64>, WireError> {
    if values.len() >= min {
        Ok(values)
    } else {
        let got = values.len();
        Err(WireError::too_few(
            param,
            min,
            got,
            format!("{param}: at least {min} values are needed, got {got}"),
        ))
    }
}

/// The first value outside `low`..`high`, as `parameter_out_of_range` at its
/// index with the bounds and the value.
pub(crate) fn each_within(
    values: &[f64],
    param: &'static str,
    low: Bound,
    high: Bound,
) -> Result<(), WireError> {
    match values.iter().position(|&v| !within(v, low, high)) {
        Some(i) => {
            let got = values[i];
            let mut e = WireError::out_of_range(
                param,
                low,
                high,
                got,
                format!("{param}[{i}]: must be {}, got {got}", describe(low, high)),
            );
            e.index = Some(i);
            Err(e)
        }
        None => Ok(()),
    }
}

pub(crate) fn np_chart_dto(
    defectives: &[u64],
    sample_size: u64,
) -> Result<FixedLimitChartDto, WireError> {
    use crate::spc::NPChart;

    let mut chart =
        NPChart::new(sample_size).map_err(|e| WireError::chart("sample_size", None, &e))?;
    add_rows(defectives, "defectives", |&d| chart.add_sample(d))?;
    let (ucl, cl, lcl) = chart
        .control_limits()
        .ok_or_else(|| WireError::too_few("defectives", 1, 0, "defectives must not be empty"))?;
    Ok(FixedLimitChartDto {
        cl,
        ucl,
        lcl,
        points: attribute_point_dtos(chart.points()),
        in_control: chart.is_in_control(),
    })
}

pub(crate) fn c_chart_dto(defects: &[u64]) -> Result<FixedLimitChartDto, WireError> {
    use crate::spc::CChart;

    let mut chart = CChart::new();
    for &c in defects {
        chart.add_sample(c);
    }
    let (ucl, cl, lcl) = chart
        .control_limits()
        .ok_or_else(|| WireError::too_few("defects", 1, 0, "defects must not be empty"))?;
    Ok(FixedLimitChartDto {
        cl,
        ucl,
        lcl,
        points: attribute_point_dtos(chart.points()),
        in_control: chart.is_in_control(),
    })
}

pub(crate) fn u_chart_dto(
    raw: &[(u64, f64)],
    standard: &AttributeStandardDto,
) -> Result<UChartDto, WireError> {
    use crate::spc::UChart;

    standard.refuse(
        "u_chart",
        &[
            ("p_bar", standard.p_bar.is_some()),
            ("phi", standard.phi.is_some()),
        ],
    )?;
    let mut chart = match standard.u_bar {
        Some(u) => UChart::with_center(u).map_err(|e| WireError::chart("options", None, &e))?,
        None => UChart::new(),
    };
    add_rows(raw, "samples", |&(d, u)| chart.add_sample(d, u))?;
    let u_bar = chart
        .u_bar()
        .ok_or_else(|| WireError::too_few("samples", 1, 0, "samples must not be empty"))?;
    Ok(UChartDto {
        u_bar,
        points: attribute_point_dtos(chart.points()),
        in_control: chart.is_in_control(),
    })
}

pub(crate) fn laney_u_dto(
    raw: &[(u64, f64)],
    standard: &AttributeStandardDto,
) -> Result<LaneyUChartDto, WireError> {
    standard.refuse("laney_u_chart", &[("p_bar", standard.p_bar.is_some())])?;
    let laney = standard.laney(standard.u_bar, "u_bar")?;
    let chart =
        crate::spc::laney_u_chart(raw, laney).map_err(|e| WireError::chart_input("samples", &e))?;
    Ok(LaneyUChartDto {
        u_bar: chart.u_bar,
        phi: chart.phi,
        points: laney_point_dtos(&chart.points),
    })
}

/// G chart: inter-event conforming counts, at least 3, each `>= 0`.
pub(crate) fn g_chart_dto(gaps: &[f64]) -> Result<GChartDto, WireError> {
    let gaps = at_least(gaps.to_vec(), 3, "gaps")?;
    each_within(&gaps, "gaps", Bound::Closed(0.0), Bound::Unbounded)?;
    let chart =
        crate::spc::g_chart(&gaps).expect("three or more finite counts >= 0 make a G chart");
    Ok(GChartDto {
        g_bar: chart.g_bar,
        points: chart
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
            .collect(),
    })
}

/// T chart: inter-event times, at least 3, each `> 0`.
pub(crate) fn t_chart_dto(times: &[f64]) -> Result<TChartDto, WireError> {
    let times = at_least(times.to_vec(), 3, "times")?;
    each_within(&times, "times", Bound::Open(0.0), Bound::Unbounded)?;
    let chart = crate::spc::t_chart(&times).expect("three or more finite times > 0 make a T chart");
    Ok(TChartDto {
        t_bar: chart.t_bar,
        points: chart
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
            .collect(),
    })
}

// ---------------------------------------------------------------------------
// Hypothesis tests (WebAssembly; the C transport does not carry them yet)
// ---------------------------------------------------------------------------
//
// Each core checks what it can name -- sizes, shapes, the domain of a count or
// an expected frequency -- and refuses with the argument and the row. What is
// left when the test still returns `None` is data with no variation, refused
// as `invalid_input` naming the argument.
pub(crate) mod hypothesis {
    use super::*;

    /// A test statistic, its degrees of freedom and two-sided p-value.
    #[derive(Serialize, Debug)]
    #[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
    #[cfg_attr(all(test, feature = "wasm"), derive(schemars::JsonSchema))]
    #[cfg_attr(feature = "wasm", tsify(missing_as_null))]
    pub(crate) struct TestResultDto {
        pub(crate) statistic: f64,
        pub(crate) df: f64,
        pub(crate) p_value: f64,
    }

    impl From<crate::testing::TestResult> for TestResultDto {
        fn from(r: crate::testing::TestResult) -> Self {
            TestResultDto {
                statistic: r.statistic,
                df: r.df,
                p_value: r.p_value,
            }
        }
    }

    /// One-way ANOVA table.
    #[derive(Serialize, Debug)]
    #[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
    #[cfg_attr(all(test, feature = "wasm"), derive(schemars::JsonSchema))]
    #[cfg_attr(feature = "wasm", tsify(missing_as_null))]
    pub(crate) struct AnovaDto {
        pub(crate) f_statistic: f64,
        pub(crate) df_between: usize,
        pub(crate) df_within: usize,
        pub(crate) p_value: f64,
        pub(crate) ss_between: f64,
        pub(crate) ss_within: f64,
    }

    /// Shapiro-Wilk W and its p-value.
    #[derive(Serialize, Debug)]
    #[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
    #[cfg_attr(all(test, feature = "wasm"), derive(schemars::JsonSchema))]
    #[cfg_attr(feature = "wasm", tsify(missing_as_null))]
    pub(crate) struct ShapiroWilkDto {
        pub(crate) w: f64,
        pub(crate) p_value: f64,
    }

    /// Mann-Kendall trend test with Sen's slope.
    #[derive(Serialize, Debug)]
    #[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
    #[cfg_attr(all(test, feature = "wasm"), derive(schemars::JsonSchema))]
    #[cfg_attr(feature = "wasm", tsify(missing_as_null))]
    pub(crate) struct MannKendallDto {
        pub(crate) s_statistic: i64,
        pub(crate) variance: f64,
        pub(crate) z_statistic: f64,
        pub(crate) p_value: f64,
        pub(crate) kendall_tau: f64,
        pub(crate) sen_slope: f64,
    }

    fn no_variation(test: &str, param: &'static str) -> WireError {
        WireError::invalid_input(format!(
            "{test}: {param} has no variation (every value is the same), so the statistic is undefined"
        ))
        .about(param)
    }

    fn both_constant(test: &str) -> WireError {
        WireError::invalid_input(format!(
            "{test}: a and b together have no variation, so the statistic is undefined"
        ))
    }

    fn same_length(
        x: &[f64],
        y: &[f64],
        xn: &'static str,
        yn: &'static str,
    ) -> Result<(), WireError> {
        if x.len() == y.len() {
            Ok(())
        } else {
            Err(WireError::new(
                code::DIMENSION_MISMATCH,
                None,
                format!(
                    "{yn} has {} values but {xn} has {}; the test pairs them",
                    y.len(),
                    x.len()
                ),
            )
            .about(yn))
        }
    }

    /// At least two groups, each with at least `min` values.
    fn groups_of(groups: &[Vec<f64>], min: usize) -> Result<(), WireError> {
        if groups.len() < 2 {
            return Err(WireError::too_few(
                "groups",
                2,
                groups.len(),
                format!("groups: at least 2 groups are needed, got {}", groups.len()),
            ));
        }
        if let Some(i) = groups.iter().position(|g| g.len() < min) {
            return Err(WireError::new(
                code::INSUFFICIENT_DATA,
                Some(i),
                format!(
                    "groups[{i}]: at least {min} values are needed, got {}",
                    groups[i].len()
                ),
            )
            .about("groups"));
        }
        Ok(())
    }

    fn as_slices(groups: &[Vec<f64>]) -> Vec<&[f64]> {
        groups.iter().map(Vec::as_slice).collect()
    }

    pub(crate) fn one_sample_t_dto(data: &[f64], mu0: f64) -> Result<TestResultDto, WireError> {
        let data = at_least(data.to_vec(), 2, "data")?;
        if !mu0.is_finite() {
            return Err(WireError::new(
                code::VALUE_NOT_FINITE,
                None,
                format!("mu0: expected a finite number, got {mu0}"),
            )
            .about("mu0"));
        }
        crate::testing::one_sample_t_test(&data, mu0)
            .map(Into::into)
            .ok_or_else(|| no_variation("one_sample_t_test", "data"))
    }

    pub(crate) fn two_sample_t_dto(a: &[f64], b: &[f64]) -> Result<TestResultDto, WireError> {
        let a = at_least(a.to_vec(), 2, "a")?;
        let b = at_least(b.to_vec(), 2, "b")?;
        crate::testing::two_sample_t_test(&a, &b)
            .map(Into::into)
            .ok_or_else(|| both_constant("two_sample_t_test"))
    }

    pub(crate) fn paired_t_dto(x: &[f64], y: &[f64]) -> Result<TestResultDto, WireError> {
        same_length(x, y, "x", "y")?;
        let x = at_least(x.to_vec(), 2, "x")?;
        crate::testing::paired_t_test(&x, y)
            .map(Into::into)
            .ok_or_else(|| {
                WireError::invalid_input(
                    "paired_t_test: every difference y - x is the same, so the statistic is undefined",
                )
                .about("y")
            })
    }

    pub(crate) fn mann_whitney_dto(a: &[f64], b: &[f64]) -> Result<TestResultDto, WireError> {
        let a = at_least(a.to_vec(), 2, "a")?;
        let b = at_least(b.to_vec(), 2, "b")?;
        crate::testing::mann_whitney_u_test(&a, &b)
            .map(Into::into)
            .ok_or_else(|| both_constant("mann_whitney_u_test"))
    }

    pub(crate) fn wilcoxon_dto(x: &[f64], y: &[f64]) -> Result<TestResultDto, WireError> {
        same_length(x, y, "x", "y")?;
        let x = at_least(x.to_vec(), 2, "x")?;
        crate::testing::wilcoxon_signed_rank_test(&x, y)
            .map(Into::into)
            .ok_or_else(|| {
                WireError::insufficient_data(
                    "wilcoxon_signed_rank_test: fewer than 2 pairs differ (pairs with x = y are dropped)",
                )
                .about("y")
            })
    }

    pub(crate) fn jarque_bera_dto(data: &[f64]) -> Result<TestResultDto, WireError> {
        let data = at_least(data.to_vec(), 8, "data")?;
        crate::testing::jarque_bera_test(&data)
            .map(Into::into)
            .ok_or_else(|| no_variation("jarque_bera_test", "data"))
    }

    pub(crate) fn shapiro_wilk_dto(data: &[f64]) -> Result<ShapiroWilkDto, WireError> {
        let data = at_least(data.to_vec(), 3, "data")?;
        if data.len() > 5000 {
            return Err(WireError::out_of_range(
                "data",
                Bound::Closed(3.0),
                Bound::Closed(5000.0),
                data.len() as f64,
                format!(
                    "data: Shapiro-Wilk covers 3 to 5000 values (Royston 1995), got {}",
                    data.len()
                ),
            ));
        }
        crate::testing::shapiro_wilk_test(&data)
            .map(|r| ShapiroWilkDto {
                w: r.w,
                p_value: r.p_value,
            })
            .ok_or_else(|| no_variation("shapiro_wilk_test", "data"))
    }

    /// Anderson-Darling normality: A², the sample-size-corrected A²* and its p-value.
    #[derive(Serialize, Debug)]
    #[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
    #[cfg_attr(all(test, feature = "wasm"), derive(schemars::JsonSchema))]
    #[cfg_attr(feature = "wasm", tsify(missing_as_null))]
    pub(crate) struct AndersonDarlingDto {
        pub(crate) statistic: f64,
        pub(crate) statistic_modified: f64,
        pub(crate) p_value: f64,
    }

    pub(crate) fn anderson_darling_dto(data: &[f64]) -> Result<AndersonDarlingDto, WireError> {
        let data = at_least(data.to_vec(), 8, "data")?;
        crate::testing::anderson_darling_test(&data)
            .map(|r| AndersonDarlingDto {
                statistic: r.statistic,
                statistic_modified: r.statistic_star,
                p_value: r.p_value,
            })
            .ok_or_else(|| no_variation("anderson_darling_test", "data"))
    }

    /// Augmented Dickey-Fuller unit-root test input.
    #[derive(Deserialize)]
    #[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
    #[cfg_attr(all(test, feature = "wasm"), derive(schemars::JsonSchema))]
    #[cfg_attr(feature = "wasm", tsify(missing_as_null))]
    #[serde(deny_unknown_fields)]
    pub(crate) struct AdfInputDto {
        pub(crate) data: Vec<f64>,
        /// Deterministic terms in the test regression.
        #[serde(default = "default_adf_model")]
        #[cfg_attr(feature = "wasm", tsify(optional))]
        #[cfg_attr(
            feature = "wasm",
            tsify(type = "\"none\" | \"constant\" | \"constant_trend\"")
        )]
        #[cfg_attr(
            all(test, feature = "wasm"),
            schemars(schema_with = "crate::wire::card_schema::adf_model")
        )]
        pub(crate) model: String,
        /// The largest number of lagged differences the AIC search considers;
        /// absent or `null`: Schwert's rule. Not with `lags`.
        #[serde(default)]
        #[cfg_attr(feature = "wasm", tsify(optional, type = "number | null"))]
        pub(crate) max_lags: Option<Whole>,
        /// Exactly this many lagged differences, no search. Not with `max_lags`.
        #[serde(default)]
        #[cfg_attr(feature = "wasm", tsify(optional, type = "number | null"))]
        pub(crate) lags: Option<Whole>,
    }

    fn default_adf_model() -> String {
        "constant".to_owned()
    }

    /// One significance level of the ADF test.
    #[derive(Serialize, Debug)]
    #[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
    #[cfg_attr(all(test, feature = "wasm"), derive(schemars::JsonSchema))]
    #[cfg_attr(feature = "wasm", tsify(missing_as_null))]
    pub(crate) struct AdfLevelDto {
        pub(crate) level: f64,
        pub(crate) critical_value: f64,
        /// `statistic <= critical_value`: the unit root is rejected at this level.
        pub(crate) rejected: bool,
    }

    #[derive(Serialize, Debug)]
    #[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
    #[cfg_attr(all(test, feature = "wasm"), derive(schemars::JsonSchema))]
    #[cfg_attr(feature = "wasm", tsify(missing_as_null))]
    pub(crate) struct AdfDto {
        pub(crate) statistic: f64,
        pub(crate) n_lags: usize,
        pub(crate) n_obs: usize,
        /// 1 %, 5 % and 10 %, in that order (MacKinnon 2010 response surfaces).
        pub(crate) levels: Vec<AdfLevelDto>,
    }

    /// The fewest rows the test regression is run on.
    const ADF_MIN_REGRESSION_ROWS: usize = 5;

    /// `adf_test`'s `model` names, each with the regression columns besides
    /// the lags: the lagged level, plus the deterministic terms.
    pub(crate) const ADF_MODELS: [(&str, (crate::testing::AdfModel, usize)); 3] = [
        ("none", (crate::testing::AdfModel::None, 1)),
        ("constant", (crate::testing::AdfModel::Constant, 2)),
        (
            "constant_trend",
            (crate::testing::AdfModel::ConstantTrend, 3),
        ),
    ];

    pub(crate) fn adf_dto(input: AdfInputDto) -> Result<AdfDto, WireError> {
        let (model, fixed_columns) = super::named("model", &input.model, &ADF_MODELS)?;
        let data = at_least(input.data, 10, "data")?;
        // With p lags the regression keeps n - 1 - p rows (one value is lost to
        // differencing, p to the lags) and has fixed_columns + p columns. It
        // needs ADF_MIN_REGRESSION_ROWS rows and a residual degree of freedom.
        let n = data.len();
        let max = (n - 1 - ADF_MIN_REGRESSION_ROWS).min((n - 2 - fixed_columns) / 2);
        let count = |parameter: &'static str, value: Option<Whole>| match value {
            Some(Whole(p)) if !(0.0..=max as f64).contains(&p) => Err(WireError::out_of_range(
                parameter,
                Bound::Closed(0.0),
                Bound::Closed(max as f64),
                p,
                format!(
                    "{parameter}: {} values leave room for 0 to {max} lags, got {p}",
                    data.len()
                ),
            )),
            p => Ok(p.map(Whole::count)),
        };
        let lags = match (
            count("max_lags", input.max_lags)?,
            count("lags", input.lags)?,
        ) {
            (Some(_), Some(_)) => {
                return Err(WireError::new(
                    code::INVALID_OPTION,
                    None,
                    "lags fixes the number of lagged differences and max_lags bounds the \
                     AIC search for it -- give one",
                )
                .about("lags"))
            }
            (_, Some(p)) => crate::testing::AdfLags::Fixed(p),
            (max, None) => crate::testing::AdfLags::Aic { max },
        };
        let r = crate::testing::adf_test(&data, model, lags).ok_or_else(|| {
            WireError::invalid_input(
                "adf_test: the test regression is degenerate -- singular, or an exact \
                 fit of the differences (a deterministic series), so the t-ratio is \
                 undefined",
            )
            .about("data")
        })?;
        Ok(AdfDto {
            statistic: r.statistic,
            n_lags: r.n_lags,
            n_obs: r.n_obs,
            levels: [0.01, 0.05, 0.10]
                .iter()
                .zip(r.critical_values.iter().zip(r.rejected.iter()))
                .map(|(&level, (&critical_value, &rejected))| AdfLevelDto {
                    level,
                    critical_value,
                    rejected,
                })
                .collect(),
        })
    }

    /// KPSS stationarity test input.
    #[derive(Deserialize)]
    #[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
    #[cfg_attr(all(test, feature = "wasm"), derive(schemars::JsonSchema))]
    #[cfg_attr(feature = "wasm", tsify(missing_as_null))]
    #[serde(deny_unknown_fields)]
    pub(crate) struct KpssInputDto {
        pub(crate) data: Vec<f64>,
        /// What the null says the series is stationary around.
        #[serde(default = "default_kpss_model")]
        #[cfg_attr(feature = "wasm", tsify(optional))]
        #[cfg_attr(feature = "wasm", tsify(type = "\"level\" | \"trend\""))]
        #[cfg_attr(
            all(test, feature = "wasm"),
            schemars(schema_with = "crate::wire::card_schema::kpss_model")
        )]
        pub(crate) model: String,
        /// How the long-run variance's truncation lag is chosen. Not with `lags`.
        #[serde(default)]
        #[cfg_attr(feature = "wasm", tsify(optional))]
        #[cfg_attr(feature = "wasm", tsify(type = "\"hobijn\" | \"schwert\" | null"))]
        #[cfg_attr(
            all(test, feature = "wasm"),
            schemars(schema_with = "crate::wire::card_schema::kpss_lag_rule_or_null")
        )]
        pub(crate) lag_rule: Option<String>,
        /// Exactly this truncation lag, 0 to n − 1. Not with `lag_rule`.
        #[serde(default)]
        #[cfg_attr(feature = "wasm", tsify(optional, type = "number | null"))]
        pub(crate) lags: Option<Whole>,
    }

    fn default_kpss_model() -> String {
        "level".to_owned()
    }

    /// One significance level of the KPSS test.
    #[derive(Serialize, Debug)]
    #[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
    #[cfg_attr(all(test, feature = "wasm"), derive(schemars::JsonSchema))]
    #[cfg_attr(feature = "wasm", tsify(missing_as_null))]
    pub(crate) struct KpssLevelDto {
        pub(crate) level: f64,
        pub(crate) critical_value: f64,
        /// `statistic > critical_value`: stationarity is rejected at this level.
        pub(crate) rejected: bool,
    }

    #[derive(Serialize, Debug)]
    #[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
    #[cfg_attr(all(test, feature = "wasm"), derive(schemars::JsonSchema))]
    #[cfg_attr(feature = "wasm", tsify(missing_as_null))]
    pub(crate) struct KpssDto {
        pub(crate) statistic: f64,
        pub(crate) n_lags: usize,
        /// Interpolated in the table of critical values, clamped to [0.01, 0.10].
        pub(crate) p_value: f64,
        /// `"interpolated"`, or `"at_most"` / `"at_least"` when the statistic
        /// is beyond the table and `p_value` is its bound.
        #[cfg_attr(
            feature = "wasm",
            tsify(type = "\"interpolated\" | \"at_most\" | \"at_least\"")
        )]
        #[cfg_attr(all(test, feature = "wasm"), schemars(extend("enum" = ["interpolated", "at_most", "at_least"])))]
        pub(crate) p_value_bound: &'static str,
        /// 1 %, 2.5 %, 5 % and 10 %, in that order (Kwiatkowski et al. 1992, Table 1).
        pub(crate) levels: Vec<KpssLevelDto>,
    }

    /// `kpss_test`'s `model` names.
    pub(crate) const KPSS_MODELS: [(&str, crate::testing::KpssModel); 2] = [
        ("level", crate::testing::KpssModel::Level),
        ("trend", crate::testing::KpssModel::Trend),
    ];

    /// `kpss_test`'s `lag_rule` names.
    pub(crate) const KPSS_LAG_RULES: [(&str, crate::testing::KpssLags); 2] = [
        ("hobijn", crate::testing::KpssLags::Hobijn),
        ("schwert", crate::testing::KpssLags::Schwert),
    ];

    pub(crate) fn kpss_dto(input: KpssInputDto) -> Result<KpssDto, WireError> {
        use crate::testing::KpssPValue as P;
        let model = super::named("model", &input.model, &KPSS_MODELS)?;
        let rule = input
            .lag_rule
            .as_deref()
            .map(|name| super::named("lag_rule", name, &KPSS_LAG_RULES))
            .transpose()?;
        let data = at_least(input.data, 10, "data")?;
        let n = data.len();
        let fixed = match input.lags {
            Some(Whole(l)) if !(0.0..=(n - 1) as f64).contains(&l) => {
                return Err(WireError::out_of_range(
                    "lags",
                    Bound::Closed(0.0),
                    Bound::Closed((n - 1) as f64),
                    l,
                    format!(
                        "lags: {n} values allow a truncation lag of 0 to {}, got {l}",
                        n - 1
                    ),
                ))
            }
            l => l.map(Whole::count),
        };
        let lags = match (rule, fixed) {
            (Some(_), Some(_)) => {
                return Err(WireError::new(
                    code::INVALID_OPTION,
                    None,
                    "lags fixes the truncation lag and lag_rule chooses it -- give one",
                )
                .about("lags"))
            }
            (_, Some(l)) => crate::testing::KpssLags::Fixed(l),
            (Some(rule), None) => rule,
            (None, None) => crate::testing::KpssLags::Hobijn,
        };
        let r = crate::testing::kpss_test(&data, model, lags).ok_or_else(|| {
            WireError::invalid_input(
                "kpss_test: the series has no variation about its level (or trend), so \
                 the statistic is undefined",
            )
            .about("data")
        })?;
        Ok(KpssDto {
            statistic: r.statistic,
            n_lags: r.n_lags,
            p_value: r.p_value,
            p_value_bound: match r.p_value_bound {
                P::Interpolated => "interpolated",
                P::AtMost => "at_most",
                P::AtLeast => "at_least",
            },
            levels: (0..4)
                .rev()
                .map(|k| KpssLevelDto {
                    level: crate::testing::KPSS_LEVELS[k],
                    critical_value: r.critical_values[k],
                    rejected: r.rejected[k],
                })
                .collect(),
        })
    }

    pub(crate) fn mann_kendall_dto(data: &[f64]) -> Result<MannKendallDto, WireError> {
        let data = at_least(data.to_vec(), 4, "data")?;
        crate::testing::mann_kendall_test(&data)
            .map(|r| MannKendallDto {
                s_statistic: r.s_statistic,
                variance: r.variance,
                z_statistic: r.z_statistic,
                p_value: r.p_value,
                kendall_tau: r.kendall_tau,
                sen_slope: r.sen_slope,
            })
            .ok_or_else(|| no_variation("mann_kendall_test", "data"))
    }

    pub(crate) fn one_way_anova_dto(groups: &[Vec<f64>]) -> Result<AnovaDto, WireError> {
        groups_of(groups, 2)?;
        crate::testing::one_way_anova(&as_slices(groups))
            .map(|r| AnovaDto {
                f_statistic: r.f_statistic,
                df_between: r.df_between,
                df_within: r.df_within,
                p_value: r.p_value,
                ss_between: r.ss_between,
                ss_within: r.ss_within,
            })
            .ok_or_else(|| no_variation("one_way_anova", "groups"))
    }

    /// Kruskal-Wallis, Levene or Bartlett: `test` names which.
    pub(crate) fn groups_test_dto(
        test: &str,
        groups: &[Vec<f64>],
    ) -> Result<TestResultDto, WireError> {
        groups_of(groups, 2)?;
        let slices = as_slices(groups);
        let result = match test {
            "kruskal_wallis_test" => crate::testing::kruskal_wallis_test(&slices),
            "levene_test" => crate::testing::levene_test(&slices),
            _ => crate::testing::bartlett_test(&slices),
        };
        result
            .map(Into::into)
            .ok_or_else(|| no_variation(test, "groups"))
    }

    pub(crate) fn chi_squared_gof_dto(
        observed: &[f64],
        expected: &[f64],
    ) -> Result<TestResultDto, WireError> {
        same_length(observed, expected, "observed", "expected")?;
        let observed = at_least(observed.to_vec(), 2, "observed")?;
        each_within(&observed, "observed", Bound::Closed(0.0), Bound::Unbounded)?;
        each_within(expected, "expected", Bound::Open(0.0), Bound::Unbounded)?;
        crate::testing::chi_squared_goodness_of_fit(&observed, expected)
            .map(Into::into)
            .ok_or_else(|| {
                WireError::invalid_input(
                    "chi_squared_goodness_of_fit: no statistic -- the observed counts sum to 0",
                )
                .about("observed")
            })
    }

    /// Rows of a contingency table, all the same length.
    fn rectangular(
        table: &[Vec<f64>],
        min_rows: usize,
        min_cols: usize,
    ) -> Result<usize, WireError> {
        if table.len() < min_rows {
            return Err(WireError::too_few(
                "table",
                min_rows,
                table.len(),
                format!(
                    "table: at least {min_rows} rows are needed, got {}",
                    table.len()
                ),
            ));
        }
        let cols = table[0].len();
        if cols < min_cols {
            return Err(WireError::too_few(
                "table",
                min_cols,
                cols,
                format!("table: at least {min_cols} columns are needed, got {cols}"),
            ));
        }
        if let Some(i) = table.iter().position(|r| r.len() != cols) {
            return Err(WireError::new(
                code::DIMENSION_MISMATCH,
                Some(i),
                format!(
                    "table[{i}] has {} cells but table[0] has {cols}; every row needs the same number",
                    table[i].len()
                ),
            )
            .about("table"));
        }
        Ok(cols)
    }

    pub(crate) fn chi_squared_independence_dto(
        table: &[Vec<f64>],
    ) -> Result<TestResultDto, WireError> {
        let cols = rectangular(table, 2, 2)?;
        let flat: Vec<f64> = table.iter().flatten().copied().collect();
        each_within(&flat, "table", Bound::Closed(0.0), Bound::Unbounded)?;
        crate::testing::chi_squared_independence(&flat, table.len(), cols)
            .map(Into::into)
            .ok_or_else(|| {
                WireError::invalid_input(
                    "chi_squared_independence: a row or a column sums to 0, so its expected counts are 0",
                )
                .about("table")
            })
    }

    /// Bonferroni or Benjamini-Hochberg adjusted p-values, in input order.
    pub(crate) fn p_adjust_dto(test: &str, p_values: &[f64]) -> Result<Vec<f64>, WireError> {
        let p = at_least(p_values.to_vec(), 1, "p_values")?;
        each_within(&p, "p_values", Bound::Closed(0.0), Bound::Closed(1.0))?;
        let adjusted = if test == "bonferroni_correction" {
            crate::testing::bonferroni_correction(&p)
        } else {
            crate::testing::benjamini_hochberg(&p)
        };
        Ok(adjusted.expect("probabilities in [0, 1] are adjusted"))
    }

    pub(crate) fn fisher_exact_dto(table: &serde_json::Value) -> Result<TestResultDto, WireError> {
        let rows = as_array(table, "table", "a 2 x 2 array of counts")?;
        if rows.len() != 2 {
            return Err(WireError::new(
                code::DIMENSION_MISMATCH,
                None,
                format!(
                    "table: Fisher's exact test takes 2 rows, got {}",
                    rows.len()
                ),
            )
            .about("table"));
        }
        let mut cells = Vec::with_capacity(4);
        for (i, row) in rows.iter().enumerate() {
            let row = count_rows(row, &format!("table[{i}]"))?;
            if row.len() != 2 {
                return Err(WireError::new(
                    code::DIMENSION_MISMATCH,
                    Some(i),
                    format!(
                        "table[{i}]: Fisher's exact test takes 2 columns, got {}",
                        row.len()
                    ),
                )
                .about("table"));
            }
            cells.extend(row);
        }
        crate::testing::fisher_exact_test(cells[0], cells[1], cells[2], cells[3])
            .map(Into::into)
            .ok_or_else(|| {
                WireError::invalid_input(
                    "fisher_exact_test: a row or a column of the table sums to 0",
                )
                .about("table")
            })
    }
}

/// Input for `process_capability`.
///
/// Every field except `data` is optional, but at least one of `usl`/`lsl` must
/// be present -- a capability index without a specification limit is undefined.
#[derive(Deserialize)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(all(test, feature = "wasm"), derive(schemars::JsonSchema))]
#[cfg_attr(feature = "wasm", tsify(missing_as_null))]
#[serde(deny_unknown_fields)]
pub(crate) struct CapabilityInputDto {
    pub(crate) data: Vec<f64>,
    #[serde(default)]
    #[cfg_attr(feature = "wasm", tsify(optional))]
    #[cfg_attr(feature = "wasm", tsify(type = "number | null"))]
    pub(crate) usl: Option<f64>,
    #[serde(default)]
    #[cfg_attr(feature = "wasm", tsify(optional))]
    #[cfg_attr(feature = "wasm", tsify(type = "number | null"))]
    pub(crate) lsl: Option<f64>,
    /// Short-term (within-subgroup) sigma, normally estimated from a control
    /// chart as R-bar/d2 or S-bar/c4. It cannot be recovered from `data`: the
    /// subgroup structure is not in a flat measurement vector.
    #[serde(default)]
    #[cfg_attr(feature = "wasm", tsify(optional))]
    #[cfg_attr(feature = "wasm", tsify(type = "number | null"))]
    pub(crate) sigma_within: Option<f64>,
    /// Process target for Cpm. Without it `cpm` is `null`.
    #[serde(default)]
    #[cfg_attr(feature = "wasm", tsify(optional))]
    #[cfg_attr(feature = "wasm", tsify(type = "number | null"))]
    pub(crate) target: Option<f64>,
    /// Degrees of freedom of `sigma_within` -- they depend on how it was
    /// estimated, so the short-term intervals are given only with them.
    #[serde(default)]
    #[cfg_attr(feature = "wasm", tsify(optional))]
    #[cfg_attr(feature = "wasm", tsify(type = "number | null"))]
    pub(crate) sigma_within_df: Option<f64>,
    /// Confidence level of `intervals`, in (0, 1). Omitted: 0.95.
    #[serde(default)]
    #[cfg_attr(feature = "wasm", tsify(optional))]
    #[cfg_attr(feature = "wasm", tsify(type = "number | null"))]
    pub(crate) confidence: Option<f64>,
}

/// A two-sided confidence interval.
#[derive(Serialize, Debug, PartialEq)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(all(test, feature = "wasm"), derive(schemars::JsonSchema))]
pub(crate) struct IntervalDto {
    pub(crate) lower: f64,
    pub(crate) upper: f64,
}

/// Confidence intervals of the capability indices; `null` where the index is,
/// and for `cp` / `cpk` unless `sigma_within_df` was given.
#[derive(Serialize, Debug)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(all(test, feature = "wasm"), derive(schemars::JsonSchema))]
#[cfg_attr(feature = "wasm", tsify(missing_as_null))]
pub(crate) struct CapabilityIntervalsDto {
    pub(crate) confidence: f64,
    pub(crate) cp: Option<IntervalDto>,
    pub(crate) cpk: Option<IntervalDto>,
    pub(crate) pp: Option<IntervalDto>,
    pub(crate) ppk: Option<IntervalDto>,
}

#[derive(Serialize, Debug)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(all(test, feature = "wasm"), derive(schemars::JsonSchema))]
#[cfg_attr(feature = "wasm", tsify(missing_as_null))]
pub(crate) struct CapabilityDto {
    pub(crate) mean: f64,
    /// `"within"` when `sigma_within` was supplied, `"overall"` otherwise.
    /// Without it the short-term indices are not computed at all rather than
    /// being filled with the long-term sigma -- a number under the wrong name
    /// is harder to notice than a null.
    pub(crate) sigma_source: &'static str,
    pub(crate) std_dev_within: Option<f64>,
    pub(crate) std_dev_overall: f64,
    pub(crate) cp: Option<f64>,
    pub(crate) cpk: Option<f64>,
    pub(crate) cpu: Option<f64>,
    pub(crate) cpl: Option<f64>,
    pub(crate) pp: Option<f64>,
    pub(crate) ppk: Option<f64>,
    pub(crate) ppu: Option<f64>,
    pub(crate) ppl: Option<f64>,
    pub(crate) cpm: Option<f64>,
    pub(crate) intervals: CapabilityIntervalsDto,
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
pub(crate) fn capability_dto(input: CapabilityInputDto) -> Result<CapabilityDto, WireError> {
    use crate::capability::ProcessCapability;

    check_specification(input.usl, input.lsl)?;
    let mut spec = ProcessCapability::new(input.usl, input.lsl)
        .map_err(|e| WireError::invalid_input(format!("invalid specification limits: {e}")))?;
    if let Some(target) = input.target {
        if !target.is_finite() {
            return Err(WireError::new(
                code::VALUE_NOT_FINITE,
                None,
                format!("target must be finite, got {target}"),
            )
            .about("target"));
        }
        spec = spec.with_target(target);
    }
    let data = at_least(input.data, 2, "data")?;
    let confidence = input.confidence.unwrap_or(0.95);
    if !(confidence > 0.0 && confidence < 1.0) {
        return Err(WireError::out_of_range(
            "confidence",
            Bound::Open(0.0),
            Bound::Open(1.0),
            confidence,
            format!("confidence must be strictly between 0 and 1, got {confidence}"),
        ));
    }
    if let Some(df) = input.sigma_within_df {
        if !(df.is_finite() && df > 0.0) {
            return Err(WireError::out_of_range(
                "sigma_within_df",
                Bound::Open(0.0),
                Bound::Unbounded,
                df,
                format!("sigma_within_df must be a positive, finite number, got {df}"),
            ));
        }
        if input.sigma_within.is_none() {
            return Err(WireError::new(
                code::INVALID_OPTION,
                None,
                "sigma_within_df is the degrees of freedom of sigma_within; give sigma_within with it",
            )
            .about("sigma_within_df"));
        }
    }

    // The crate decides what a missing within sigma means: `compute_overall`
    // reports the long-term indices only, so the short-term quartet and
    // `std_dev_within` arrive as `None` and are passed through as such.
    let indices = match input.sigma_within {
        Some(sigma_within) => {
            if !sigma_within.is_finite() || sigma_within <= 0.0 {
                return Err(WireError::out_of_range(
                    "sigma_within",
                    Bound::Open(0.0),
                    Bound::Unbounded,
                    sigma_within,
                    format!(
                        "sigma_within must be a positive, finite number (R-bar/d2 or \
                         S-bar/c4 from the control chart), got {sigma_within}"
                    ),
                ));
            }
            spec.compute(&data, sigma_within)
        }
        None => spec.compute_overall(&data),
    }
    .ok_or_else(|| {
        WireError::invalid_input("data: every value is the same, so no spread to measure against")
            .about("data")
    })?;

    let ci = indices
        .intervals(confidence, input.sigma_within_df)
        .expect("confidence and sigma_within_df were checked above");
    let interval = |i: Option<crate::capability::ConfidenceInterval>| {
        i.map(|i| IntervalDto {
            lower: i.lower,
            upper: i.upper,
        })
    };
    let dto = CapabilityDto {
        mean: indices.mean,
        sigma_source: if indices.std_dev_within.is_some() {
            "within"
        } else {
            "overall"
        },
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
        // Not a short-term index: Cpm is the spread about the target, the
        // same whichever sigma the caller could supply.
        cpm: indices.cpm,
        intervals: CapabilityIntervalsDto {
            confidence: ci.confidence,
            cp: interval(ci.cp),
            cpk: interval(ci.cpk),
            pp: interval(ci.pp),
            ppk: interval(ci.ppk),
        },
    };
    Ok(dto)
}

/// Input for `percentile_capability`.
#[derive(Deserialize)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(all(test, feature = "wasm"), derive(schemars::JsonSchema))]
#[cfg_attr(feature = "wasm", tsify(missing_as_null))]
#[serde(deny_unknown_fields)]
pub(crate) struct PercentileCapabilityInputDto {
    pub(crate) data: Vec<f64>,
    #[cfg_attr(feature = "wasm", tsify(optional))]
    #[cfg_attr(feature = "wasm", tsify(type = "number | null"))]
    pub(crate) lsl: Option<f64>,
    #[cfg_attr(feature = "wasm", tsify(optional))]
    #[cfg_attr(feature = "wasm", tsify(type = "number | null"))]
    pub(crate) usl: Option<f64>,
    /// The family to fit, by the names `fit_best` reports; absent = the one it
    /// ranks first.
    #[serde(default)]
    #[cfg_attr(feature = "wasm", tsify(optional))]
    #[cfg_attr(
        feature = "wasm",
        tsify(
            type = "\"Normal\" | \"LogNormal\" | \"Weibull\" | \"Gamma\" | \"Exponential\" | \"Beta\" | null"
        )
    )]
    #[cfg_attr(
        all(test, feature = "wasm"),
        schemars(schema_with = "crate::wire::card_schema::capability_model_or_null")
    )]
    pub(crate) distribution: Option<String>,
    /// Bootstrap replicates behind `goodness_of_fit.p_value`; absent = 199,
    /// 0 = skip the p-value (`null`). See `goodness_of_fit`.
    #[serde(default)]
    #[cfg_attr(feature = "wasm", tsify(optional, type = "number | null"))]
    pub(crate) fit_replicates: Option<Whole>,
    /// Seed of the bootstrap draws; absent = 0.
    #[serde(default)]
    #[cfg_attr(feature = "wasm", tsify(optional, type = "number | null"))]
    pub(crate) seed: Option<Whole>,
}

/// Input for `goodness_of_fit`.
#[derive(Deserialize)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(all(test, feature = "wasm"), derive(schemars::JsonSchema))]
#[cfg_attr(feature = "wasm", tsify(missing_as_null))]
#[serde(deny_unknown_fields)]
pub(crate) struct GoodnessOfFitInputDto {
    pub(crate) data: Vec<f64>,
    /// The family to fit and test, by the names `fit_best` reports.
    #[cfg_attr(
        feature = "wasm",
        tsify(
            type = "\"Normal\" | \"LogNormal\" | \"Weibull\" | \"Gamma\" | \"Exponential\" | \"Beta\""
        )
    )]
    #[cfg_attr(
        all(test, feature = "wasm"),
        schemars(extend("enum" = ["Normal", "LogNormal", "Weibull", "Gamma", "Exponential", "Beta"]))
    )]
    pub(crate) distribution: String,
    /// Bootstrap replicates, 1 to 9999; absent = 199.
    #[serde(default)]
    #[cfg_attr(feature = "wasm", tsify(optional, type = "number | null"))]
    pub(crate) replicates: Option<Whole>,
    /// Seed of the bootstrap draws; absent = 0.
    #[serde(default)]
    #[cfg_attr(feature = "wasm", tsify(optional, type = "number | null"))]
    pub(crate) seed: Option<Whole>,
}

/// Most bootstrap replicates a goodness-of-fit request may ask for.
const MAX_GOF_REPLICATES: f64 = 9999.0;

/// Fewest values `goodness_of_fit` tests (as `anderson_darling_test`).
const MIN_GOF_DATA: usize = 8;

/// `replicates` within `[min, MAX_GOF_REPLICATES]`, absent = the default.
fn gof_replicates(
    value: Option<Whole>,
    parameter: &'static str,
    min: f64,
) -> Result<usize, WireError> {
    match value {
        None => Ok(crate::distribution::DEFAULT_GOF_REPLICATES),
        Some(Whole(b)) if (min..=MAX_GOF_REPLICATES).contains(&b) => Ok(Whole(b).count()),
        Some(Whole(b)) => Err(WireError::out_of_range(
            parameter,
            Bound::Closed(min),
            Bound::Closed(MAX_GOF_REPLICATES),
            b,
            format!("{parameter}: {min} to {MAX_GOF_REPLICATES} replicates, got {b}"),
        )),
    }
}

/// A bootstrap seed: a whole number from 0 to 2^53 (exact in a JSON number).
fn gof_seed(value: Option<Whole>) -> Result<u64, WireError> {
    match value {
        None => Ok(0),
        Some(Whole(s)) if (0.0..=MAX_EXACT_COUNT).contains(&s) => Ok(s as u64),
        Some(Whole(s)) => Err(WireError::out_of_range(
            "seed",
            Bound::Closed(0.0),
            Bound::Closed(MAX_EXACT_COUNT),
            s,
            format!("seed: 0 to {MAX_EXACT_COUNT}, got {s}"),
        )),
    }
}

/// How well the fitted family describes the data.
#[derive(Serialize, Debug)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(all(test, feature = "wasm"), derive(schemars::JsonSchema))]
#[cfg_attr(feature = "wasm", tsify(missing_as_null))]
pub(crate) struct GoodnessOfFitDto {
    /// Anderson–Darling A² against the fitted distribution.
    pub(crate) statistic: f64,
    /// Parametric-bootstrap p-value; `null` when `replicates` is 0.
    pub(crate) p_value: Option<f64>,
    pub(crate) replicates: usize,
}

/// `goodness_of_fit`'s result: the family fitted, its parameters, and the test.
#[derive(Serialize, Debug)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(all(test, feature = "wasm"), derive(schemars::JsonSchema))]
#[cfg_attr(feature = "wasm", tsify(missing_as_null))]
pub(crate) struct GoodnessOfFitResultDto {
    #[cfg_attr(
        feature = "wasm",
        tsify(
            type = "\"Normal\" | \"LogNormal\" | \"Weibull\" | \"Gamma\" | \"Exponential\" | \"Beta\""
        )
    )]
    #[cfg_attr(all(test, feature = "wasm"), schemars(extend("enum" = ["Normal", "LogNormal", "Weibull", "Gamma", "Exponential", "Beta"])))]
    pub(crate) distribution: &'static str,
    #[cfg_attr(feature = "wasm", tsify(type = "Record<string, number>"))]
    pub(crate) parameters: std::collections::BTreeMap<String, f64>,
    pub(crate) statistic: f64,
    pub(crate) p_value: f64,
    pub(crate) replicates: usize,
}

/// The test of `fit` against `data`, with `replicates` (0 = statistic only).
fn goodness_dto(
    data: &[f64],
    fit: &crate::distribution::FitResult,
    replicates: usize,
    seed: u64,
) -> GoodnessOfFitDto {
    let statistic = crate::distribution::anderson_darling_fit(data, fit)
        .expect("the fit is one of fit_best's families and the data finite");
    let p_value = (replicates > 0).then(|| {
        crate::distribution::goodness_of_fit(data, fit, replicates, seed)
            .expect("the fit is one of fit_best's families and the data finite")
            .p_value
    });
    GoodnessOfFitDto {
        statistic,
        p_value,
        replicates,
    }
}

/// The refusal for a family `data` does not admit: the first value outside
/// its support when that is the reason, `invalid_option` otherwise.
fn family_refusal(
    model: crate::capability::CapabilityModel,
    data: &[f64],
    message: String,
) -> WireError {
    use crate::capability::CapabilityModel as M;
    match model {
        // The families on (0, inf) refuse data that reaches 0 or below; name
        // the first such value.
        M::LogNormal | M::Weibull | M::Gamma | M::Exponential if data.iter().any(|&v| v <= 0.0) => {
            let index = data.iter().position(|&v| v <= 0.0);
            let got = index.map_or(Detail::Null, |i| Detail::Num(data[i]));
            WireError::new(code::NON_POSITIVE_DATA, index, message)
                .about("data")
                .with("got", got)
        }
        _ => WireError::new(code::INVALID_OPTION, None, message).about("distribution"),
    }
}

pub(crate) fn goodness_of_fit_dto(
    input: GoodnessOfFitInputDto,
) -> Result<GoodnessOfFitResultDto, WireError> {
    let model = capability_model(&input.distribution).ok_or_else(|| {
        WireError::unknown_option("distribution", &input.distribution, CAPABILITY_MODELS)
    })?;
    let replicates = gof_replicates(input.replicates, "replicates", 1.0)?;
    let seed = gof_seed(input.seed)?;
    let data = at_least(input.data, MIN_GOF_DATA, "data")?;
    if first_constant(std::slice::from_ref(&data)).is_some() {
        return Err(WireError::invalid_input(
            "data: every value is the same, so no distribution can be fitted",
        )
        .about("data"));
    }
    let fit = model.fit(&data).ok_or_else(|| {
        family_refusal(
            model,
            &data,
            format!("the {model:?} distribution cannot be fitted to this data"),
        )
    })?;
    let gof = goodness_dto(&data, &fit, replicates, seed);
    Ok(GoodnessOfFitResultDto {
        distribution: capability_model_name(model),
        parameters: fit.parameters.into_iter().collect(),
        statistic: gof.statistic,
        p_value: gof.p_value.expect("replicates >= 1"),
        replicates,
    })
}

/// The families `percentile_capability` fits, by the names `fit_best` uses.
pub(crate) const CAPABILITY_MODELS: &[&str] = &[
    "Normal",
    "LogNormal",
    "Weibull",
    "Gamma",
    "Exponential",
    "Beta",
];

fn capability_model(name: &str) -> Option<crate::capability::CapabilityModel> {
    use crate::capability::CapabilityModel as M;
    Some(match name {
        "Normal" => M::Normal,
        "LogNormal" => M::LogNormal,
        "Weibull" => M::Weibull,
        "Gamma" => M::Gamma,
        "Exponential" => M::Exponential,
        "Beta" => M::Beta,
        _ => return None,
    })
}

fn capability_model_name(model: crate::capability::CapabilityModel) -> &'static str {
    use crate::capability::CapabilityModel as M;
    match model {
        M::Normal => "Normal",
        M::LogNormal => "LogNormal",
        M::Weibull => "Weibull",
        M::Gamma => "Gamma",
        M::Exponential => "Exponential",
        M::Beta => "Beta",
        // `percentile_capability` reports the family it fitted, never `Auto`.
        _ => "Auto",
    }
}

#[derive(Serialize)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(all(test, feature = "wasm"), derive(schemars::JsonSchema))]
#[cfg_attr(feature = "wasm", tsify(missing_as_null))]
pub(crate) struct PercentileCapabilityDto {
    pub(crate) cp_star: Option<f64>,
    pub(crate) cpk_star: Option<f64>,
    pub(crate) cpu_star: Option<f64>,
    pub(crate) cpl_star: Option<f64>,
    pub(crate) median: f64,
    pub(crate) percentile_lower: f64,
    pub(crate) percentile_upper: f64,
    /// The family the quantiles come from (`fit_best`'s names).
    #[cfg_attr(
        feature = "wasm",
        tsify(
            type = "\"Normal\" | \"LogNormal\" | \"Weibull\" | \"Gamma\" | \"Exponential\" | \"Beta\""
        )
    )]
    #[cfg_attr(all(test, feature = "wasm"), schemars(extend("enum" = ["Normal", "LogNormal", "Weibull", "Gamma", "Exponential", "Beta"])))]
    pub(crate) distribution: &'static str,
    /// Its fitted parameters by name, as `fit_best` reports them.
    #[cfg_attr(feature = "wasm", tsify(type = "Record<string, number>"))]
    pub(crate) parameters: std::collections::BTreeMap<String, f64>,
    /// Whether that family describes the data — the indices mean nothing when
    /// it does not.
    pub(crate) goodness_of_fit: GoodnessOfFitDto,
}

#[derive(Deserialize)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(all(test, feature = "wasm"), derive(schemars::JsonSchema))]
#[cfg_attr(feature = "wasm", tsify(missing_as_null))]
#[serde(deny_unknown_fields)]
pub(crate) struct GageRRInputDto {
    pub(crate) measurements: Vec<Vec<Vec<f64>>>,
    #[cfg_attr(feature = "wasm", tsify(optional))]
    #[cfg_attr(feature = "wasm", tsify(type = "number | null"))]
    pub(crate) tolerance: Option<f64>,
}

#[derive(Serialize)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(all(test, feature = "wasm"), derive(schemars::JsonSchema))]
#[cfg_attr(feature = "wasm", tsify(missing_as_null))]
pub(crate) struct GageChartLimitsDto {
    pub(crate) center: f64,
    pub(crate) ucl: f64,
    pub(crate) lcl: f64,
}

impl From<crate::msa::GageChartLimits> for GageChartLimitsDto {
    fn from(l: crate::msa::GageChartLimits) -> Self {
        GageChartLimitsDto {
            center: l.center,
            ucl: l.ucl,
            lcl: l.lcl,
        }
    }
}

#[derive(Serialize)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(all(test, feature = "wasm"), derive(schemars::JsonSchema))]
#[cfg_attr(feature = "wasm", tsify(missing_as_null))]
pub(crate) struct GageRRResultDto {
    /// The range chart the method plots, from the same R̄ the components use.
    pub(crate) range_chart: GageChartLimitsDto,
    /// The average chart the method plots.
    pub(crate) average_chart: GageChartLimitsDto,
    pub(crate) ev: f64,
    pub(crate) av: f64,
    pub(crate) grr: f64,
    pub(crate) pv: f64,
    pub(crate) tv: f64,
    pub(crate) percent_ev: f64,
    pub(crate) percent_av: f64,
    pub(crate) percent_grr: f64,
    pub(crate) percent_pv: f64,
    pub(crate) percent_tolerance: Option<f64>,
    pub(crate) ndc: u32,
    pub(crate) status: String,
}

#[derive(Serialize)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(all(test, feature = "wasm"), derive(schemars::JsonSchema))]
#[cfg_attr(feature = "wasm", tsify(missing_as_null))]
pub(crate) struct GageRRAnovaResultDto {
    pub(crate) anova_table: Vec<AnovaRowDto>,
    pub(crate) variance_components: VarianceComponentsDto,
    pub(crate) ev: f64,
    pub(crate) av: f64,
    pub(crate) grr: f64,
    pub(crate) pv: f64,
    pub(crate) tv: f64,
    pub(crate) percent_grr: f64,
    pub(crate) percent_tolerance: Option<f64>,
    pub(crate) ndc: u32,
    pub(crate) status: String,
    pub(crate) interaction_significant: bool,
    pub(crate) interaction_pooled: bool,
}

#[derive(Serialize)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(all(test, feature = "wasm"), derive(schemars::JsonSchema))]
#[cfg_attr(feature = "wasm", tsify(missing_as_null))]
pub(crate) struct AnovaRowDto {
    pub(crate) source: String,
    pub(crate) df: f64,
    pub(crate) ss: f64,
    pub(crate) ms: f64,
    pub(crate) f_value: Option<f64>,
    pub(crate) p_value: Option<f64>,
}

#[derive(Serialize)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(all(test, feature = "wasm"), derive(schemars::JsonSchema))]
#[cfg_attr(feature = "wasm", tsify(missing_as_null))]
pub(crate) struct VarianceComponentsDto {
    pub(crate) part: f64,
    pub(crate) operator: f64,
    pub(crate) interaction: f64,
    pub(crate) repeatability: f64,
    pub(crate) reproducibility: f64,
    pub(crate) total: f64,
}

pub(crate) fn grr_status_str(status: crate::msa::GrrStatus) -> &'static str {
    match status {
        crate::msa::GrrStatus::Acceptable => "Acceptable",
        crate::msa::GrrStatus::Marginal => "Marginal",
        crate::msa::GrrStatus::Unacceptable => "Unacceptable",
    }
}

#[derive(Deserialize)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(all(test, feature = "wasm"), derive(schemars::JsonSchema))]
#[cfg_attr(feature = "wasm", tsify(missing_as_null))]
#[serde(deny_unknown_fields)]
pub(crate) struct PeltInputDto {
    pub(crate) data: Vec<f64>,
    #[serde(default = "default_cost")]
    #[cfg_attr(feature = "wasm", tsify(optional))]
    #[cfg_attr(feature = "wasm", tsify(type = "\"l2\" | \"normal\""))]
    #[cfg_attr(
        all(test, feature = "wasm"),
        schemars(schema_with = "crate::wire::card_schema::pelt_cost")
    )]
    pub(crate) cost: String,
    #[serde(default = "default_penalty")]
    #[cfg_attr(feature = "wasm", tsify(optional))]
    #[cfg_attr(feature = "wasm", tsify(type = "\"bic\" | number"))]
    #[cfg_attr(
        all(test, feature = "wasm"),
        schemars(schema_with = "crate::wire::card_schema::penalty")
    )]
    pub(crate) penalty: PeltPenaltyDto,
    #[serde(default = "default_min_seg")]
    #[cfg_attr(feature = "wasm", tsify(optional, type = "number"))]
    pub(crate) min_segment_len: Whole,
}

#[derive(Deserialize)]
#[serde(untagged)]
pub(crate) enum PeltPenaltyDto {
    Named(String),
    Value(f64),
}

#[derive(Serialize)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(all(test, feature = "wasm"), derive(schemars::JsonSchema))]
#[cfg_attr(feature = "wasm", tsify(missing_as_null))]
pub(crate) struct PeltResultDto {
    pub(crate) changepoints: Vec<usize>,
    pub(crate) n_segments: usize,
}

pub(crate) fn default_cost() -> String {
    "l2".to_owned()
}

pub(crate) fn default_penalty() -> PeltPenaltyDto {
    PeltPenaltyDto::Named("bic".to_owned())
}

pub(crate) fn default_min_seg() -> Whole {
    Whole(2.0)
}

/// The specification limits a capability index is measured against, refused
/// by name: neither given (`no_specification`), or `usl` not above `lsl`
/// (`parameter_out_of_range` on `usl`, with `lsl` as its `exclusive_min`).
pub(crate) fn check_specification(usl: Option<f64>, lsl: Option<f64>) -> Result<(), WireError> {
    match (usl, lsl) {
        (None, None) => Err(WireError::new(
            code::NO_SPECIFICATION,
            None,
            "at least one specification limit (usl or lsl) is required",
        )
        .with("expected", Detail::List(vec!["usl", "lsl"]))),
        (Some(u), Some(l)) if u <= l => Err(WireError::out_of_range(
            "usl",
            Bound::Open(l),
            Bound::Unbounded,
            u,
            format!("usl must be greater than lsl ({l}), got {u}"),
        )),
        _ => Ok(()),
    }
}

/// The measurements of a Gage R&R study, `measurements[part][operator][trial]`,
/// refused where they go wrong: too few parts, operators or trials, or a cell
/// of another size than the first (`dimension_mismatch` naming the part).
fn check_gage_shape(measurements: &[Vec<Vec<f64>>]) -> Result<(usize, usize, usize), WireError> {
    let parts = measurements.len();
    if parts < 2 {
        return Err(WireError::too_few(
            "measurements",
            2,
            parts,
            format!("measurements: at least 2 parts are needed, got {parts}"),
        ));
    }
    let operators = measurements[0].len();
    if operators < 2 {
        return Err(WireError::too_few(
            "measurements[0]",
            2,
            operators,
            format!("measurements[0]: at least 2 operators are needed, got {operators}"),
        ));
    }
    let trials = measurements[0][0].len();
    if trials < 2 {
        return Err(WireError::too_few(
            "measurements[0][0]",
            2,
            trials,
            format!("measurements[0][0]: at least 2 trials are needed, got {trials}"),
        ));
    }
    for (p, part) in measurements.iter().enumerate() {
        if part.len() != operators {
            return Err(WireError::new(
                code::DIMENSION_MISMATCH,
                Some(p),
                format!(
                    "measurements[{p}] has {} operators; measurements[0] has {operators}",
                    part.len()
                ),
            )
            .about("measurements")
            .with("expected", Detail::Num(operators as f64))
            .with("got", Detail::Num(part.len() as f64)));
        }
        if let Some(o) = part.iter().position(|cell| cell.len() != trials) {
            return Err(WireError::new(
                code::DIMENSION_MISMATCH,
                Some(o),
                format!(
                    "measurements[{p}][{o}] has {} trials; measurements[0][0] has {trials}",
                    part[o].len()
                ),
            )
            .about(format!("measurements[{p}]"))
            .with("expected", Detail::Num(trials as f64))
            .with("got", Detail::Num(part[o].len() as f64)));
        }
    }
    Ok((parts, operators, trials))
}

pub(crate) fn percentile_capability_dto(
    input: PercentileCapabilityInputDto,
) -> Result<PercentileCapabilityDto, WireError> {
    use crate::capability::{CapabilityModel as M, PercentileCapabilityError as E};
    check_specification(input.usl, input.lsl)?;
    let model = match input.distribution.as_deref() {
        None => M::Auto,
        Some(name) => capability_model(name)
            .ok_or_else(|| WireError::unknown_option("distribution", name, CAPABILITY_MODELS))?,
    };
    let replicates = gof_replicates(input.fit_replicates, "fit_replicates", 0.0)?;
    let seed = gof_seed(input.seed)?;
    let data = at_least(input.data, crate::capability::MIN_PERCENTILE_DATA, "data")?;
    let result = crate::capability::percentile_capability(&data, input.lsl, input.usl, model)
        .map_err(|e| {
            let message = e.to_string();
            match e {
                E::NonFiniteData { index } => {
                    WireError::new(code::VALUE_NOT_FINITE, Some(index), message).about("data")
                }
                E::ModelNotFitted(model) => family_refusal(model, &data, message),
                _ => WireError::invalid_input(message).about("data"),
            }
        })?;
    let goodness_of_fit = goodness_dto(&data, &result.fit, replicates, seed);
    Ok(PercentileCapabilityDto {
        cp_star: result.cp_star,
        cpk_star: result.cpk_star,
        cpu_star: result.cpu_star,
        cpl_star: result.cpl_star,
        median: result.median,
        percentile_lower: result.percentile_lower,
        percentile_upper: result.percentile_upper,
        distribution: capability_model_name(result.model),
        parameters: result.fit.parameters.into_iter().collect(),
        goodness_of_fit,
    })
}

pub(crate) fn gage_rr_xbar_r_dto(dto: GageRRInputDto) -> Result<GageRRResultDto, WireError> {
    let (parts, operators, trials) = check_gage_shape(&dto.measurements)?;
    // The method's K-factor tables cover these sizes only.
    for (name, at, got, max) in [
        ("trials", "measurements[0][0]", trials, 3),
        ("operators", "measurements[0]", operators, 3),
        ("parts", "measurements", parts, 10),
    ] {
        if got > max {
            return Err(WireError::out_of_range(
                // `parameter` names the array whose length it is.
                match at {
                    "measurements" => "measurements",
                    "measurements[0]" => "measurements[0]",
                    _ => "measurements[0][0]",
                },
                Bound::Closed(2.0),
                Bound::Closed(max as f64),
                got as f64,
                format!("{at}: the X̄-R method supports 2 to {max} {name}, got {got}"),
            ));
        }
    }
    let input = crate::msa::GageRRInput {
        measurements: dto.measurements,
        tolerance: dto.tolerance,
    };
    let result = crate::msa::gage_rr_xbar_r(&input)
        .map_err(|e| WireError::invalid_input(e).about("measurements"))?;
    Ok(GageRRResultDto {
        range_chart: result.range_chart.into(),
        average_chart: result.average_chart.into(),
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
    })
}

pub(crate) fn gage_rr_anova_dto(dto: GageRRInputDto) -> Result<GageRRAnovaResultDto, WireError> {
    check_gage_shape(&dto.measurements)?;
    let input = crate::msa::GageRRInput {
        measurements: dto.measurements,
        tolerance: dto.tolerance,
    };
    let result = crate::msa::gage_rr_anova(&input)
        .map_err(|e| WireError::invalid_input(e).about("measurements"))?;
    let anova_table = result
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
    Ok(GageRRAnovaResultDto {
        anova_table,
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
    })
}

/// A PELT detector from the wire options, naming the option a refusal is about.
/// The PELT `cost` names.
pub(crate) const PELT_COSTS: [(&str, crate::detection::CostFunction); 2] = [
    ("l2", crate::detection::CostFunction::L2),
    ("normal", crate::detection::CostFunction::Normal),
];

pub(crate) fn pelt_from(
    cost: &str,
    penalty: &PeltPenaltyDto,
    min_segment_len: Whole,
) -> Result<crate::detection::Pelt, WireError> {
    use crate::detection::Penalty;
    let cost = named("cost", cost, &PELT_COSTS)?;
    let penalty = match *penalty {
        PeltPenaltyDto::Named(ref s) if s == "bic" => Penalty::Bic,
        PeltPenaltyDto::Named(ref s) => {
            return Err(WireError::unknown_option("penalty", s, &["bic"]))
        }
        PeltPenaltyDto::Value(v) if !(v.is_finite() && v > 0.0) => {
            return Err(WireError::out_of_range(
                "penalty",
                Bound::Open(0.0),
                Bound::Unbounded,
                v,
                format!("penalty must be \"bic\" or a finite number > 0, got {v}"),
            ))
        }
        PeltPenaltyDto::Value(v) => Penalty::Custom(v),
    };
    // One check for every value below the floor, `-1` and `1` alike.
    let Whole(len) = min_segment_len;
    if len < 2.0 {
        return Err(WireError::out_of_range(
            "min_segment_len",
            Bound::Closed(2.0),
            Bound::Unbounded,
            len,
            format!("min_segment_len must be at least 2, got {len}"),
        ));
    }
    let min_segment_len = min_segment_len.count();
    Ok(
        crate::detection::Pelt::with_min_segment_len(cost, penalty, min_segment_len)
            .expect("cost, penalty and min_segment_len were checked above"),
    )
}

pub(crate) fn pelt_dto(input: PeltInputDto) -> Result<PeltResultDto, WireError> {
    if input.data.is_empty() {
        return Err(WireError::empty_input("data"));
    }
    let pelt = pelt_from(&input.cost, &input.penalty, input.min_segment_len)?;
    let result = pelt.detect(&input.data);
    Ok(PeltResultDto {
        n_segments: result.changepoints.len() + 1,
        changepoints: result.changepoints,
    })
}

/// PELT over several aligned signals: one set of changepoints for all.
#[derive(Deserialize)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(all(test, feature = "wasm"), derive(schemars::JsonSchema))]
#[cfg_attr(feature = "wasm", tsify(missing_as_null))]
#[serde(deny_unknown_fields)]
pub(crate) struct MultiPeltInputDto {
    pub(crate) signals: Vec<Vec<f64>>,
    #[serde(default = "default_cost")]
    #[cfg_attr(feature = "wasm", tsify(optional))]
    #[cfg_attr(feature = "wasm", tsify(type = "\"l2\" | \"normal\""))]
    #[cfg_attr(
        all(test, feature = "wasm"),
        schemars(schema_with = "crate::wire::card_schema::pelt_cost")
    )]
    pub(crate) cost: String,
    #[serde(default = "default_penalty")]
    #[cfg_attr(feature = "wasm", tsify(optional))]
    #[cfg_attr(feature = "wasm", tsify(type = "\"bic\" | number"))]
    #[cfg_attr(
        all(test, feature = "wasm"),
        schemars(schema_with = "crate::wire::card_schema::penalty")
    )]
    pub(crate) penalty: PeltPenaltyDto,
    #[serde(default = "default_min_seg")]
    #[cfg_attr(feature = "wasm", tsify(optional, type = "number"))]
    pub(crate) min_segment_len: Whole,
}

pub(crate) fn multi_pelt_dto(input: MultiPeltInputDto) -> Result<PeltResultDto, WireError> {
    let Some(first) = input.signals.first() else {
        return Err(WireError::empty_input("signals"));
    };
    let expected = first.len();
    if let Some(i) = input.signals.iter().position(|s| s.len() != expected) {
        let got = input.signals[i].len();
        return Err(WireError::new(
            code::DIMENSION_MISMATCH,
            Some(i),
            format!("signals[{i}] has {got} values; signals[0] has {expected}"),
        )
        .about("signals")
        .with("expected", Detail::Num(expected as f64))
        .with("got", Detail::Num(got as f64)));
    }
    if expected == 0 {
        return Err(WireError::empty_input("signals"));
    }
    let pelt = pelt_from(&input.cost, &input.penalty, input.min_segment_len)?;
    let refs: Vec<&[f64]> = input.signals.iter().map(Vec::as_slice).collect();
    let result = pelt
        .detect_multi(&refs)
        .expect("signals were checked non-empty and of one length above");
    Ok(PeltResultDto {
        n_segments: result.changepoints.len() + 1,
        changepoints: result.changepoints,
    })
}

// ── CUSUM / EWMA: online (sequential) shift detection ──────────────────────

#[derive(Deserialize)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(all(test, feature = "wasm"), derive(schemars::JsonSchema))]
#[cfg_attr(feature = "wasm", tsify(missing_as_null))]
#[serde(deny_unknown_fields)]
pub(crate) struct CusumInputDto {
    pub(crate) data: Vec<f64>,
    pub(crate) target: f64,
    pub(crate) sigma: f64,
    #[serde(default = "default_cusum_k")]
    #[cfg_attr(feature = "wasm", tsify(optional))]
    pub(crate) k: f64,
    #[serde(default = "default_cusum_h")]
    #[cfg_attr(feature = "wasm", tsify(optional))]
    pub(crate) h: f64,
}

fn default_cusum_k() -> f64 {
    0.5
}

fn default_cusum_h() -> f64 {
    5.0
}

#[derive(Serialize, Debug)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(all(test, feature = "wasm"), derive(schemars::JsonSchema))]
#[cfg_attr(feature = "wasm", tsify(missing_as_null))]
pub(crate) struct CusumPointDto {
    pub(crate) index: usize,
    pub(crate) s_upper: f64,
    pub(crate) s_lower: f64,
    pub(crate) signal: bool,
}

#[derive(Serialize, Debug)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(all(test, feature = "wasm"), derive(schemars::JsonSchema))]
#[cfg_attr(feature = "wasm", tsify(missing_as_null))]
pub(crate) struct CusumDto {
    /// Decision interval actually used, echoed so a caller can draw the
    /// boundary without restating its own input. Unlike EWMA's widening
    /// limits this one is constant, so it belongs to the chart, not the point.
    pub(crate) h: f64,
    pub(crate) points: Vec<CusumPointDto>,
    pub(crate) signal_indices: Vec<usize>,
    pub(crate) in_control: bool,
}

#[derive(Deserialize)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(all(test, feature = "wasm"), derive(schemars::JsonSchema))]
#[cfg_attr(feature = "wasm", tsify(missing_as_null))]
#[serde(deny_unknown_fields)]
pub(crate) struct EwmaInputDto {
    pub(crate) data: Vec<f64>,
    pub(crate) target: f64,
    pub(crate) sigma: f64,
    #[serde(default = "default_ewma_lambda")]
    #[cfg_attr(feature = "wasm", tsify(optional))]
    pub(crate) lambda: f64,
    #[serde(default = "default_ewma_l_factor")]
    #[cfg_attr(feature = "wasm", tsify(optional))]
    pub(crate) l_factor: f64,
}

fn default_ewma_lambda() -> f64 {
    0.2
}

fn default_ewma_l_factor() -> f64 {
    3.0
}

#[derive(Serialize, Debug)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(all(test, feature = "wasm"), derive(schemars::JsonSchema))]
#[cfg_attr(feature = "wasm", tsify(missing_as_null))]
pub(crate) struct EwmaPointDto {
    pub(crate) index: usize,
    pub(crate) ewma: f64,
    pub(crate) ucl: f64,
    pub(crate) lcl: f64,
    pub(crate) signal: bool,
}

#[derive(Serialize, Debug)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(all(test, feature = "wasm"), derive(schemars::JsonSchema))]
#[cfg_attr(feature = "wasm", tsify(missing_as_null))]
pub(crate) struct EwmaDto {
    pub(crate) points: Vec<EwmaPointDto>,
    pub(crate) signal_indices: Vec<usize>,
    pub(crate) in_control: bool,
}

/// `value` must be a finite number in `low`..`high`; the refusal names it,
/// with the closed ends as `min` / `max` and the open ends as
/// `exclusive_min` / `exclusive_max`.
fn chart_parameter(
    name: &'static str,
    value: f64,
    low: Bound,
    high: Bound,
) -> Result<(), WireError> {
    if value.is_finite() && within(value, low, high) {
        return Ok(());
    }
    Err(WireError::out_of_range(
        name,
        low,
        high,
        value,
        format!("{name} must be {}, got {value}", describe(low, high)),
    ))
}

pub(crate) fn cusum_dto(input: CusumInputDto) -> Result<CusumDto, WireError> {
    if input.data.is_empty() {
        return Err(WireError::empty_input("data"));
    }
    let CusumInputDto {
        target,
        sigma,
        k,
        h,
        ..
    } = input;
    chart_parameter("target", target, Bound::Unbounded, Bound::Unbounded)?;
    chart_parameter("sigma", sigma, Bound::Open(0.0), Bound::Unbounded)?;
    chart_parameter("k", k, Bound::Closed(0.0), Bound::Unbounded)?;
    chart_parameter("h", h, Bound::Open(0.0), Bound::Unbounded)?;
    let chart = crate::detection::Cusum::with_params(target, sigma, k, h)
        .expect("every parameter was checked against the chart's own conditions above");

    let results = chart.analyze(&input.data);
    let signal_indices: Vec<usize> = results
        .iter()
        .filter(|r| r.signal)
        .map(|r| r.index)
        .collect();
    Ok(CusumDto {
        h,
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

pub(crate) fn ewma_dto(input: EwmaInputDto) -> Result<EwmaDto, WireError> {
    if input.data.is_empty() {
        return Err(WireError::empty_input("data"));
    }
    let EwmaInputDto {
        target,
        sigma,
        lambda,
        l_factor,
        ..
    } = input;
    chart_parameter("target", target, Bound::Unbounded, Bound::Unbounded)?;
    chart_parameter("sigma", sigma, Bound::Open(0.0), Bound::Unbounded)?;
    chart_parameter("lambda", lambda, Bound::Open(0.0), Bound::Closed(1.0))?;
    chart_parameter("l_factor", l_factor, Bound::Open(0.0), Bound::Unbounded)?;
    let chart = crate::detection::Ewma::with_params(target, sigma, lambda, l_factor)
        .expect("every parameter was checked against the chart's own conditions above");

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

// ── Correlation, simple regression and distribution fitting ────────────────

#[derive(Deserialize)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(all(test, feature = "wasm"), derive(schemars::JsonSchema))]
#[cfg_attr(feature = "wasm", tsify(missing_as_null))]
#[serde(deny_unknown_fields)]
pub(crate) struct CorrelationInputDto {
    /// Each inner array is one variable; all the same length.
    pub(crate) variables: Vec<Vec<f64>>,
    #[serde(default = "default_correlation_method")]
    #[cfg_attr(feature = "wasm", tsify(optional))]
    #[cfg_attr(
        feature = "wasm",
        tsify(type = "\"pearson\" | \"spearman\" | \"kendall\"")
    )]
    #[cfg_attr(
        all(test, feature = "wasm"),
        schemars(schema_with = "crate::wire::card_schema::correlation_method")
    )]
    pub(crate) method: String,
}

pub(crate) fn default_correlation_method() -> String {
    "pearson".to_owned()
}

#[derive(Serialize, Debug)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(all(test, feature = "wasm"), derive(schemars::JsonSchema))]
#[cfg_attr(feature = "wasm", tsify(missing_as_null))]
pub(crate) struct CorrelationMatrixDto {
    /// `matrix[i][j]` is the correlation of variables `i` and `j`.
    pub(crate) matrix: Vec<Vec<f64>>,
}

/// The index of the first variable whose values are all the same, for which
/// no correlation (and no regression slope) is defined.
fn first_constant(variables: &[Vec<f64>]) -> Option<usize> {
    variables.iter().position(|v| v.iter().all(|&x| x == v[0]))
}

/// A correlation matrix of variables, or `None` when one does not vary.
type MatrixFn = fn(&[&[f64]]) -> Option<u_numflow::matrix::Matrix>;

/// `correlation_matrix`'s `method` names.
pub(crate) const CORRELATION_METHODS: [(&str, MatrixFn); 3] = [
    ("pearson", crate::correlation::correlation_matrix),
    ("spearman", crate::correlation::spearman_matrix),
    ("kendall", crate::correlation::kendall_matrix),
];

pub(crate) fn correlation_matrix_dto(
    input: CorrelationInputDto,
) -> Result<CorrelationMatrixDto, WireError> {
    let compute = named("method", &input.method, &CORRELATION_METHODS)?;
    let variables = input.variables;
    if variables.len() < 2 {
        return Err(WireError::too_few(
            "variables",
            2,
            variables.len(),
            format!(
                "variables: a correlation matrix needs at least 2 variables, got {}",
                variables.len()
            ),
        ));
    }
    let n = variables[0].len();
    if let Some(i) = variables.iter().position(|v| v.len() != n) {
        let got = variables[i].len();
        return Err(WireError::new(
            code::DIMENSION_MISMATCH,
            Some(i),
            format!("variables[{i}] has {got} values; variables[0] has {n}"),
        )
        .about("variables")
        .with("expected", Detail::Num(n as f64))
        .with("got", Detail::Num(got as f64)));
    }
    if n < 3 {
        return Err(WireError::too_few(
            "variables",
            3,
            n,
            format!("variables: each variable needs at least 3 values, got {n}"),
        ));
    }
    if let Some(i) = first_constant(&variables) {
        return Err(WireError::new(
            code::INVALID_INPUT,
            Some(i),
            format!("variables[{i}]: every value is the same, so its correlation is undefined"),
        )
        .about("variables"));
    }
    let refs: Vec<&[f64]> = variables.iter().map(Vec::as_slice).collect();
    let m = compute(&refs).ok_or_else(|| {
        WireError::invalid_input(format!(
            "correlation_matrix: the {} correlation is undefined for these variables",
            input.method
        ))
        .about("variables")
    })?;
    Ok(CorrelationMatrixDto {
        matrix: (0..m.rows())
            .map(|i| (0..m.cols()).map(|j| m.get(i, j)).collect())
            .collect(),
    })
}

#[derive(Deserialize)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(all(test, feature = "wasm"), derive(schemars::JsonSchema))]
#[cfg_attr(feature = "wasm", tsify(missing_as_null))]
#[serde(deny_unknown_fields)]
pub(crate) struct RegressionInputDto {
    pub(crate) x: Vec<f64>,
    pub(crate) y: Vec<f64>,
}

/// Ordinary least squares `y = intercept + slope · x`.
#[derive(Serialize, Debug)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(all(test, feature = "wasm"), derive(schemars::JsonSchema))]
#[cfg_attr(feature = "wasm", tsify(missing_as_null))]
pub(crate) struct RegressionDto {
    pub(crate) slope: f64,
    pub(crate) intercept: f64,
    pub(crate) r_squared: f64,
    pub(crate) adjusted_r_squared: f64,
    pub(crate) slope_se: f64,
    pub(crate) intercept_se: f64,
    /// `null` when the fit is exact (a standard error of 0).
    pub(crate) slope_t: Option<f64>,
    /// `null` when the fit is exact (a standard error of 0).
    pub(crate) intercept_t: Option<f64>,
    pub(crate) slope_p: f64,
    pub(crate) intercept_p: f64,
    pub(crate) residual_se: f64,
    /// `null` when the fit is exact.
    pub(crate) f_statistic: Option<f64>,
    pub(crate) f_p_value: f64,
    /// `y[i]` minus its fitted value, in input order.
    pub(crate) residuals: Vec<f64>,
    pub(crate) fitted: Vec<f64>,
}

fn finite(v: f64) -> Option<f64> {
    v.is_finite().then_some(v)
}

pub(crate) fn simple_regression_dto(input: RegressionInputDto) -> Result<RegressionDto, WireError> {
    let RegressionInputDto { x, y } = input;
    if y.len() != x.len() {
        return Err(WireError::new(
            code::DIMENSION_MISMATCH,
            None,
            format!("y has {} values; x has {}", y.len(), x.len()),
        )
        .about("y")
        .with("expected", Detail::Num(x.len() as f64))
        .with("got", Detail::Num(y.len() as f64)));
    }
    let x = at_least(x, 3, "x")?;
    if first_constant(std::slice::from_ref(&x)).is_some() {
        return Err(
            WireError::invalid_input("x: every value is the same, so no slope is defined")
                .about("x"),
        );
    }
    let r = crate::regression::simple_linear_regression(&x, &y)
        .expect("lengths, size and variation of x were checked above; values are finite");
    Ok(RegressionDto {
        slope: r.slope,
        intercept: r.intercept,
        r_squared: r.r_squared,
        adjusted_r_squared: r.adjusted_r_squared,
        slope_se: r.slope_se,
        intercept_se: r.intercept_se,
        slope_t: finite(r.slope_t),
        intercept_t: finite(r.intercept_t),
        slope_p: r.slope_p,
        intercept_p: r.intercept_p,
        residual_se: r.residual_se,
        f_statistic: finite(r.f_statistic),
        f_p_value: r.f_p_value,
        residuals: r.residuals,
        fitted: r.fitted,
    })
}

#[derive(Serialize, Debug)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(all(test, feature = "wasm"), derive(schemars::JsonSchema))]
#[cfg_attr(feature = "wasm", tsify(missing_as_null))]
pub(crate) struct DistributionFitDto {
    /// `"Normal"`, `"Exponential"`, `"Gamma"`, `"LogNormal"`, `"Weibull"` or `"Beta"`.
    #[cfg_attr(
        feature = "wasm",
        tsify(
            type = "\"Normal\" | \"LogNormal\" | \"Weibull\" | \"Gamma\" | \"Exponential\" | \"Beta\""
        )
    )]
    #[cfg_attr(all(test, feature = "wasm"), schemars(extend("enum" = ["Normal", "LogNormal", "Weibull", "Gamma", "Exponential", "Beta"])))]
    pub(crate) distribution: String,
    /// Estimated parameters by name (`mu`/`sigma`, `lambda`, `alpha`/`beta`,
    /// `shape`/`scale`).
    #[cfg_attr(feature = "wasm", tsify(type = "Record<string, number>"))]
    pub(crate) parameters: std::collections::BTreeMap<String, f64>,
    pub(crate) log_likelihood: f64,
    pub(crate) aic: f64,
    pub(crate) bic: f64,
    /// Anderson–Darling A² against this fit — smaller is closer. Whether the
    /// family fits at all is `goodness_of_fit`'s p-value.
    pub(crate) anderson_darling: f64,
}

/// Every continuous family that fits `data`, best AIC first.
pub(crate) fn fit_best_dto(data: &[f64]) -> Result<Vec<DistributionFitDto>, WireError> {
    let data = at_least(data.to_vec(), 2, "data")?;
    if first_constant(std::slice::from_ref(&data)).is_some() {
        return Err(WireError::invalid_input(
            "data: every value is the same, so no distribution can be fitted",
        )
        .about("data"));
    }
    let fits = crate::distribution::fit_best(&data);
    debug_assert!(!fits.is_empty(), "Normal fits any non-constant finite data");
    Ok(fits
        .into_iter()
        .map(|f| DistributionFitDto {
            anderson_darling: crate::distribution::anderson_darling_fit(&data, &f)
                .expect("fit_best's families on finite data"),
            distribution: f.distribution,
            parameters: f.parameters.into_iter().collect(),
            log_likelihood: f.log_likelihood,
            aic: f.aic,
            bic: f.bic,
        })
        .collect())
}

#[derive(Serialize, Debug)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(all(test, feature = "wasm"), derive(schemars::JsonSchema))]
#[cfg_attr(feature = "wasm", tsify(missing_as_null))]
pub(crate) struct XbarSChartDto {
    pub(crate) xbar_cl: f64,
    pub(crate) xbar_ucl: f64,
    pub(crate) xbar_lcl: f64,
    pub(crate) s_cl: f64,
    pub(crate) s_ucl: f64,
    pub(crate) s_lcl: f64,
    /// Short-term sigma implied by this chart (`S-bar / c4`), for
    /// `process_capability`'s `sigma_within`.
    pub(crate) sigma_hat: f64,
    /// Degrees of freedom of `sigma_hat`, for `sigma_within_df`.
    pub(crate) sigma_hat_df: f64,
    pub(crate) xbar_points: Vec<ChartPointDto>,
    pub(crate) s_points: Vec<ChartPointDto>,
    pub(crate) in_control: bool,
}

#[derive(Serialize, Debug)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(all(test, feature = "wasm"), derive(schemars::JsonSchema))]
#[cfg_attr(feature = "wasm", tsify(missing_as_null))]
pub(crate) struct ImrChartDto {
    pub(crate) i_cl: f64,
    pub(crate) i_ucl: f64,
    pub(crate) i_lcl: f64,
    pub(crate) mr_cl: f64,
    pub(crate) mr_ucl: f64,
    pub(crate) mr_lcl: f64,
    /// Short-term sigma implied by this chart (`MR-bar / d2(2)`).
    pub(crate) sigma_hat: f64,
    /// Degrees of freedom of `sigma_hat` (`0.62 (n - 1)`), for `sigma_within_df`.
    pub(crate) sigma_hat_df: f64,
    pub(crate) i_points: Vec<ChartPointDto>,
    /// Starts at index 1: the first value has no moving range.
    pub(crate) mr_points: Vec<ChartPointDto>,
    pub(crate) in_control: bool,
}

/// Control limits a caller supplies to `run_rules`.
#[derive(Deserialize)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(all(test, feature = "wasm"), derive(schemars::JsonSchema))]
#[cfg_attr(feature = "wasm", tsify(missing_as_null))]
#[serde(deny_unknown_fields)]
pub(crate) struct LimitsInputDto {
    pub(crate) ucl: f64,
    pub(crate) cl: f64,
    pub(crate) lcl: f64,
}

pub(crate) fn xbar_s_dto(
    subgroups: Vec<Vec<f64>>,
    rules: crate::spc::RuleSet,
) -> Result<XbarSChartDto, WireError> {
    use crate::spc::{ControlChart, XBarSChart};

    let n = subgroup_size(&subgroups)?;
    let mut chart = XBarSChart::new(n)
        .map_err(|e| WireError::chart("subgroups", None, &e))?
        .with_rules(rules);
    add_rows(&subgroups, "subgroups", |g| chart.add_sample(g))?;
    let too_few = || {
        WireError::too_few(
            "subgroups",
            1,
            0,
            "subgroups: at least one subgroup is needed",
        )
    };
    let x = chart.control_limits().ok_or_else(too_few)?;
    let s = chart.s_limits().ok_or_else(too_few)?;
    Ok(XbarSChartDto {
        xbar_cl: x.cl,
        xbar_ucl: x.ucl,
        xbar_lcl: x.lcl,
        s_cl: s.cl,
        s_ucl: s.ucl,
        s_lcl: s.lcl,
        // Present whenever the limits are: it is their centre line over d2 / c4.
        sigma_hat: chart.sigma_hat().ok_or_else(too_few)?,
        sigma_hat_df: chart.sigma_hat_df().ok_or_else(too_few)?,
        xbar_points: point_dtos(chart.points()),
        s_points: point_dtos(chart.s_points()),
        in_control: chart.is_in_control(),
    })
}

pub(crate) fn imr_dto(
    values: Vec<f64>,
    rules: crate::spc::RuleSet,
) -> Result<ImrChartDto, WireError> {
    use crate::spc::{ControlChart, IndividualMRChart};

    let mut chart = IndividualMRChart::new().with_rules(rules);
    add_rows(&values, "values", |x| {
        chart.add_sample(std::slice::from_ref(x))
    })?;
    let n = values.len();
    let too_few = || {
        WireError::too_few(
            "values",
            2,
            n,
            format!("values: at least two values are needed for control limits, got {n}"),
        )
    };
    let i = chart.control_limits().ok_or_else(too_few)?;
    let mr = chart.mr_limits().ok_or_else(too_few)?;
    Ok(ImrChartDto {
        i_cl: i.cl,
        i_ucl: i.ucl,
        i_lcl: i.lcl,
        mr_cl: mr.cl,
        mr_ucl: mr.ucl,
        mr_lcl: mr.lcl,
        // Present whenever the limits are: it is their centre line over d2 / c4.
        sigma_hat: chart.sigma_hat().ok_or_else(too_few)?,
        sigma_hat_df: chart.sigma_hat_df().ok_or_else(too_few)?,
        i_points: point_dtos(chart.points()),
        mr_points: point_dtos(chart.mr_points()),
        in_control: chart.is_in_control(),
    })
}

pub(crate) fn run_rules_dto(
    values: Vec<f64>,
    limits: LimitsInputDto,
    rules: crate::spc::RuleSet,
) -> Result<Vec<ChartPointDto>, WireError> {
    use crate::spc::{ChartPoint, ControlLimits, RunRule};

    let LimitsInputDto { ucl, cl, lcl } = limits;
    // Written so that a NaN fails it too. The centre line is the one the
    // other two are ordered about, so the refusal names the limit out of place.
    if !(lcl <= cl && cl <= ucl) {
        let (parameter, min, max, got) = if lcl <= cl {
            ("limits.ucl", Bound::Closed(cl), Bound::Unbounded, ucl)
        } else {
            ("limits.lcl", Bound::Unbounded, Bound::Closed(cl), lcl)
        };
        return Err(WireError::out_of_range(
            parameter,
            min,
            max,
            got,
            format!("limits: need lcl <= cl <= ucl, got lcl={lcl} cl={cl} ucl={ucl}"),
        ));
    }
    if let Some(i) = values.iter().position(|x| !x.is_finite()) {
        return Err(WireError::new(
            code::VALUE_NOT_FINITE,
            Some(i),
            format!("values[{i}] is not a finite number"),
        )
        .about("values"));
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

// ── Autocorrelation ──────────────────────────────────────────────────────────

/// Input for `acf` and `pacf`.
#[derive(Deserialize)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(all(test, feature = "wasm"), derive(schemars::JsonSchema))]
#[cfg_attr(feature = "wasm", tsify(missing_as_null))]
#[serde(deny_unknown_fields)]
pub(crate) struct AutocorrelationInputDto {
    pub(crate) data: Vec<f64>,
    /// Largest lag; absent = `min(⌊10·log₁₀ n⌋, n − 1)` for `acf`, and below
    /// `n / 2` as well for `pacf` (statsmodels' defaults and limit).
    #[serde(default)]
    #[cfg_attr(feature = "wasm", tsify(optional, type = "number | null"))]
    pub(crate) max_lag: Option<Whole>,
}

#[derive(Serialize, Debug)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(all(test, feature = "wasm"), derive(schemars::JsonSchema))]
#[cfg_attr(feature = "wasm", tsify(missing_as_null))]
pub(crate) struct AcfDto {
    /// Lags 0 to `max_lag`; `acf[0]` is 1.
    pub(crate) acf: Vec<f64>,
    /// 95 % white-noise threshold, z / √n.
    pub(crate) confidence_threshold: f64,
    /// 95 % limits for lags 1 to `max_lag` by Bartlett's formula.
    pub(crate) bartlett_limits: Vec<f64>,
}

#[derive(Serialize, Debug)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(all(test, feature = "wasm"), derive(schemars::JsonSchema))]
#[cfg_attr(feature = "wasm", tsify(missing_as_null))]
pub(crate) struct PacfDto {
    /// Lags 1 to `max_lag`.
    pub(crate) pacf: Vec<f64>,
    /// 95 % threshold, z / √n.
    pub(crate) confidence_threshold: f64,
}

/// The lag range of an autocorrelation request: `max_lag` within
/// `[1, highest]`, absent = `min(⌊10·log₁₀ n⌋, highest)`.
fn max_lag(value: Option<Whole>, n: usize, highest: usize) -> Result<usize, WireError> {
    match value {
        None => Ok((((n as f64).log10() * 10.0).floor() as usize).clamp(1, highest)),
        Some(Whole(k)) if (1.0..=highest as f64).contains(&k) => Ok(Whole(k).count()),
        Some(Whole(k)) => Err(WireError::out_of_range(
            "max_lag",
            Bound::Closed(1.0),
            Bound::Closed(highest as f64),
            k,
            format!("max_lag: {n} values allow lags 1 to {highest}, got {k}"),
        )),
    }
}

fn no_autocorrelation() -> WireError {
    WireError::invalid_input("data: the series does not vary, so its autocorrelation is undefined")
        .about("data")
}

pub(crate) fn acf_dto(input: AutocorrelationInputDto) -> Result<AcfDto, WireError> {
    let data = at_least(input.data, 2, "data")?;
    let n = data.len();
    let lag = max_lag(input.max_lag, n, n - 1)?;
    let r = crate::correlation::acf(&data, lag).ok_or_else(no_autocorrelation)?;
    Ok(AcfDto {
        acf: r.acf,
        confidence_threshold: r.confidence_threshold,
        bartlett_limits: r.bartlett_limits,
    })
}

pub(crate) fn pacf_dto(input: AutocorrelationInputDto) -> Result<PacfDto, WireError> {
    // Lags below n / 2 (statsmodels' limit): beyond it the recursion runs on
    // autocorrelations estimated from too few pairs.
    let data = at_least(input.data, 4, "data")?;
    let n = data.len();
    let lag = max_lag(input.max_lag, n, n / 2 - 1)?;
    let r = crate::correlation::pacf(&data, lag).ok_or_else(no_autocorrelation)?;
    Ok(PacfDto {
        pacf: r.pacf,
        confidence_threshold: r.confidence_threshold,
    })
}

// ── Seasonality ──────────────────────────────────────────────────────────────

#[derive(Deserialize)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(all(test, feature = "wasm"), derive(schemars::JsonSchema))]
#[cfg_attr(feature = "wasm", tsify(missing_as_null))]
#[serde(deny_unknown_fields)]
pub(crate) struct SeasonalityInputDto {
    pub(crate) data: Vec<f64>,
}

#[derive(Serialize, Debug)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(all(test, feature = "wasm"), derive(schemars::JsonSchema))]
#[cfg_attr(feature = "wasm", tsify(missing_as_null))]
pub(crate) struct PeriodCandidateDto {
    pub(crate) period: usize,
    pub(crate) acf: f64,
    pub(crate) bin: usize,
    pub(crate) power: f64,
    pub(crate) power_share: f64,
}

#[derive(Serialize, Debug)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(all(test, feature = "wasm"), derive(schemars::JsonSchema))]
#[cfg_attr(feature = "wasm", tsify(missing_as_null))]
pub(crate) struct SeasonalityDto {
    /// `null` when no periodicity passed both stages.
    pub(crate) period: Option<usize>,
    pub(crate) candidates: Vec<PeriodCandidateDto>,
    pub(crate) n: usize,
    pub(crate) acf_threshold: f64,
    pub(crate) power_threshold: f64,
}

pub(crate) fn seasonality_dto(input: SeasonalityInputDto) -> Result<SeasonalityDto, WireError> {
    if let Some(i) = input.data.iter().position(|x| !x.is_finite()) {
        return Err(WireError::new(
            code::VALUE_NOT_FINITE,
            Some(i),
            format!("data[{i}] is not a finite number"),
        )
        .about("data"));
    }
    let n = input.data.len();
    let r = crate::seasonality::estimate_period(&input.data).ok_or_else(|| {
        let min = crate::seasonality::MIN_OBSERVATIONS;
        WireError::too_few(
            "data",
            min,
            n,
            format!("data must have at least {min} observations, got {n}"),
        )
    })?;
    Ok(SeasonalityDto {
        period: r.period,
        candidates: r
            .candidates
            .into_iter()
            .map(|c| PeriodCandidateDto {
                period: c.period,
                acf: c.acf,
                bin: c.bin,
                power: c.power,
                power_share: c.power_share,
            })
            .collect(),
        n: r.n,
        acf_threshold: r.acf_threshold,
        power_threshold: r.power_threshold,
    })
}

// ── Spectral residual ────────────────────────────────────────────────────────

#[derive(Deserialize)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(all(test, feature = "wasm"), derive(schemars::JsonSchema))]
#[cfg_attr(feature = "wasm", tsify(missing_as_null))]
#[serde(deny_unknown_fields)]
pub(crate) struct SpectralResidualInputDto {
    pub(crate) data: Vec<f64>,
    #[serde(default)]
    #[cfg_attr(feature = "wasm", tsify(optional))]
    #[cfg_attr(feature = "wasm", tsify(type = "number | null"))]
    pub(crate) averaging_window: Option<Whole>,
    #[serde(default)]
    #[cfg_attr(feature = "wasm", tsify(optional))]
    #[cfg_attr(feature = "wasm", tsify(type = "number | null"))]
    pub(crate) judgement_window: Option<Whole>,
    #[serde(default)]
    #[cfg_attr(feature = "wasm", tsify(optional))]
    #[cfg_attr(feature = "wasm", tsify(type = "number | null"))]
    pub(crate) threshold: Option<f64>,
    #[serde(default)]
    #[cfg_attr(feature = "wasm", tsify(optional))]
    #[cfg_attr(feature = "wasm", tsify(type = "number | null"))]
    pub(crate) min_zscore: Option<f64>,
    #[serde(default)]
    #[cfg_attr(feature = "wasm", tsify(optional))]
    #[cfg_attr(feature = "wasm", tsify(type = "number | null"))]
    pub(crate) sensitivity: Option<f64>,
    #[serde(default)]
    #[cfg_attr(feature = "wasm", tsify(optional))]
    #[cfg_attr(feature = "wasm", tsify(type = "number | null"))]
    pub(crate) batch_size: Option<Whole>,
}

#[derive(Serialize, Debug)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(all(test, feature = "wasm"), derive(schemars::JsonSchema))]
#[cfg_attr(feature = "wasm", tsify(missing_as_null))]
pub(crate) struct SrPointDto {
    pub(crate) index: usize,
    pub(crate) value: f64,
    pub(crate) saliency: f64,
    pub(crate) score: f64,
    pub(crate) expected: f64,
    pub(crate) lower: f64,
    pub(crate) upper: f64,
    pub(crate) is_anomaly: bool,
    /// Within kappa = 5 places of an end of its batch, where the transform's
    /// own boundary handling moves the saliency most.
    pub(crate) near_edge: bool,
}

#[derive(Serialize, Debug)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(all(test, feature = "wasm"), derive(schemars::JsonSchema))]
#[cfg_attr(feature = "wasm", tsify(missing_as_null))]
pub(crate) struct SpectralResidualDto {
    pub(crate) points: Vec<SrPointDto>,
    pub(crate) anomalies: Vec<usize>,
}

impl From<crate::detection::SpectralResidualError> for WireError {
    fn from(error: crate::detection::SpectralResidualError) -> Self {
        use crate::detection::SpectralResidualError as E;
        let message = error.to_string();
        match error {
            E::OptionOutOfRange {
                option,
                min,
                exclusive_min,
                max,
                exclusive_max,
                got,
                ..
            } => WireError::out_of_range(
                option,
                Bound::from_pair(min, exclusive_min),
                Bound::from_pair(max, exclusive_max),
                got,
                message,
            ),
            E::TooFewObservations { needed, got } => {
                WireError::too_few("data", needed, got, message)
            }
            E::ValueNotFinite { index } => {
                WireError::new(code::VALUE_NOT_FINITE, Some(index), message)
            }
        }
    }
}

pub(crate) fn spectral_residual_dto(
    input: SpectralResidualInputDto,
) -> Result<SpectralResidualDto, WireError> {
    use crate::detection::SrCountOption as Count;
    // A count the builder cannot hold (a negative one) is checked against the
    // floor the builder's own check applies, so it is refused the same way.
    let count = |option: Count, value: Whole| -> Result<usize, WireError> {
        option.check(value.0)?;
        Ok(value.count())
    };
    let mut sr = crate::detection::SpectralResidual::new();
    if let Some(q) = input.averaging_window {
        sr = sr.with_averaging_window(count(Count::AveragingWindow, q)?);
    }
    if let Some(z) = input.judgement_window {
        sr = sr.with_judgement_window(count(Count::JudgementWindow, z)?);
    }
    if let Some(t) = input.threshold {
        sr = sr.with_threshold(t);
    }
    if let Some(z) = input.min_zscore {
        sr = sr.with_min_zscore(z);
    }
    if let Some(s) = input.sensitivity {
        sr = sr.with_sensitivity(s);
    }
    if let Some(b) = input.batch_size {
        sr = sr.with_batch_size(Some(count(Count::BatchSize, b)?));
    }
    // The crate owns these rules, so it is the crate that says which one was
    // broken -- the transports only carry the answer.
    let points = sr.analyze(&input.data)?;
    let anomalies = points
        .iter()
        .filter(|p| p.is_anomaly)
        .map(|p| p.index)
        .collect();
    Ok(SpectralResidualDto {
        points: points
            .into_iter()
            .map(|p| SrPointDto {
                index: p.index,
                value: p.value,
                saliency: p.saliency,
                score: p.score,
                expected: p.expected,
                lower: p.lower,
                upper: p.upper,
                is_anomaly: p.is_anomaly,
                near_edge: p.near_edge,
            })
            .collect(),
        anomalies,
    })
}

/// Weibull fits and reliability metrics, Box-Cox capability, and the sigma
/// level ↔ PPM conversion — the reliability and non-normal capability surface
/// both transports carry.
pub(crate) mod reliability {
    use super::*;

    /// The first problem with values that must be finite and `> 0`, placed.
    fn positive_values(param: &'static str, values: &[f64], min: usize) -> Result<(), WireError> {
        for (i, &v) in values.iter().enumerate() {
            if !v.is_finite() {
                return Err(WireError::new(
                    code::VALUE_NOT_FINITE,
                    Some(i),
                    format!("{param}[{i}]: expected a finite number, got {v}"),
                )
                .about(param));
            }
            if v <= 0.0 {
                return Err(WireError::new(
                    code::NON_POSITIVE_DATA,
                    Some(i),
                    format!("{param}[{i}]: must be > 0, got {v}"),
                )
                .about(param)
                .with("got", Detail::Num(v)));
            }
        }
        if values.len() < min {
            return Err(WireError::too_few(
                param,
                min,
                values.len(),
                format!(
                    "{param}: at least {min} values are needed, got {}",
                    values.len()
                ),
            ));
        }
        Ok(())
    }

    /// A finite number `> 0`, or `parameter_out_of_range`.
    fn positive(param: &'static str, x: f64) -> Result<f64, WireError> {
        if x.is_finite() && x > 0.0 {
            Ok(x)
        } else {
            Err(WireError::out_of_range(
                param,
                Bound::Open(0.0),
                Bound::Unbounded,
                x,
                format!("{param} must be a finite number > 0, got {x}"),
            ))
        }
    }

    // ── Weibull fits ──────────────────────────────────────────────────────

    /// `{ failure_times }` — the C ABI request (WebAssembly takes the array).
    #[cfg(feature = "ffi")]
    /// `{ failure_times, suspension_times? }` — the C ABI request of the
    /// median-rank regression, which also takes right-censored units.
    #[cfg(feature = "ffi")]
    #[derive(Deserialize, Debug)]
    #[serde(deny_unknown_fields)]
    pub(crate) struct WeibullMrrRequestDto {
        pub(crate) failure_times: Vec<f64>,
        #[serde(default)]
        pub(crate) suspension_times: Vec<f64>,
    }

    /// `{ failure_times, suspension_times? }` — the C ABI request of the MLE,
    /// which also takes right-censored units.
    #[cfg(feature = "ffi")]
    #[derive(Deserialize, Debug)]
    #[serde(deny_unknown_fields)]
    pub(crate) struct WeibullMleRequestDto {
        pub(crate) failure_times: Vec<f64>,
        #[serde(default)]
        pub(crate) suspension_times: Vec<f64>,
        #[serde(default)]
        pub(crate) confidence: Option<f64>,
        #[serde(default)]
        pub(crate) method: Option<String>,
        #[serde(default)]
        pub(crate) fractions_failed: Vec<f64>,
    }

    /// The interval methods `weibull_mle` takes, as `method` names them.
    pub(crate) const INTERVAL_METHODS: [&str; 2] = ["wald", "likelihood_ratio"];

    /// Two-sided intervals of the Weibull MLE at `confidence`: `wald` --
    /// normal on the log scale from the observed Fisher information -- or
    /// `likelihood_ratio` -- the profile-likelihood interval.
    #[derive(Serialize, Debug)]
    #[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
    #[cfg_attr(all(test, feature = "wasm"), derive(schemars::JsonSchema))]
    #[cfg_attr(feature = "wasm", tsify(missing_as_null))]
    pub(crate) struct WeibullIntervalsDto {
        pub(crate) confidence: f64,
        #[cfg_attr(feature = "wasm", tsify(type = "\"wald\" | \"likelihood_ratio\""))]
        #[cfg_attr(
            all(test, feature = "wasm"),
            schemars(schema_with = "crate::wire::card_schema::interval_method")
        )]
        pub(crate) method: &'static str,
        pub(crate) shape: crate::wire::IntervalDto,
        pub(crate) scale: crate::wire::IntervalDto,
        /// The B-life intervals, aligned with `fractions_failed`.
        pub(crate) b_life: Vec<crate::wire::IntervalDto>,
    }

    /// Weibull maximum-likelihood fit.
    #[derive(Serialize, Debug)]
    #[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
    #[cfg_attr(all(test, feature = "wasm"), derive(schemars::JsonSchema))]
    #[cfg_attr(feature = "wasm", tsify(missing_as_null))]
    pub(crate) struct WeibullMleDto {
        pub(crate) shape: f64,
        pub(crate) scale: f64,
        pub(crate) log_likelihood: f64,
        pub(crate) iterations: usize,
        /// The B-life at each of the requested `fractions_failed`.
        pub(crate) b_life: Vec<f64>,
        pub(crate) intervals: WeibullIntervalsDto,
        /// Asymptotic covariance of `(shape, scale)` -- the inverse of the
        /// observed Fisher information, rows and columns shape, scale. Pass
        /// it to `weibull_reliability` for B-life and reliability intervals.
        #[cfg_attr(feature = "wasm", tsify(type = "[[number, number], [number, number]]"))]
        pub(crate) covariance: [[f64; 2]; 2],
    }

    /// Weibull median-rank-regression fit.
    #[derive(Serialize, Debug)]
    #[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
    #[cfg_attr(all(test, feature = "wasm"), derive(schemars::JsonSchema))]
    #[cfg_attr(feature = "wasm", tsify(missing_as_null))]
    pub(crate) struct WeibullMrrDto {
        pub(crate) shape: f64,
        pub(crate) scale: f64,
        pub(crate) r_squared: f64,
    }

    fn degenerate(method: &str) -> WireError {
        WireError::invalid_input(format!(
            "{method}: failure_times has no spread (every value is the same), so the shape is undefined"
        ))
        .about("failure_times")
    }

    /// `suspensions` are right-censored units -- still running, or removed
    /// unfailed, at that time; empty fits the failures alone.
    pub(crate) fn weibull_mle_dto(
        times: &[f64],
        suspensions: &[f64],
        confidence: Option<f64>,
        method: Option<&str>,
        fractions_failed: &[f64],
    ) -> Result<WeibullMleDto, WireError> {
        let method = match method.unwrap_or("wald") {
            m @ ("wald" | "likelihood_ratio") => INTERVAL_METHODS
                .into_iter()
                .find(|n| *n == m)
                .expect("listed"),
            other => {
                return Err(WireError::unknown_option(
                    "method",
                    other,
                    &INTERVAL_METHODS,
                ))
            }
        };
        positive_values("failure_times", times, 2)?;
        positive_values("suspension_times", suspensions, 0)?;
        fractions("fractions_failed", fractions_failed)?;
        let confidence = confidence.unwrap_or(0.95);
        if !(confidence > 0.0 && confidence < 1.0) {
            return Err(WireError::out_of_range(
                "confidence",
                Bound::Open(0.0),
                Bound::Open(1.0),
                confidence,
                format!("confidence must be strictly between 0 and 1, got {confidence}"),
            ));
        }
        let r = crate::weibull::weibull_mle_censored(times, suspensions)
            .ok_or_else(|| degenerate("weibull_mle"))?;
        let dto = |i: crate::weibull::WeibullInterval| crate::wire::IntervalDto {
            lower: i.lower,
            upper: i.upper,
        };
        let ra = crate::weibull::ReliabilityAnalysis::from_mle(&r);
        let b_life: Vec<f64> = fractions_failed
            .iter()
            .map(|&f| ra.b_life(f).expect("fraction checked inside (0, 1)"))
            .collect();
        let (shape, scale, b_life_intervals) = if method == "wald" {
            let checked = "confidence and fractions were checked above";
            (
                dto(r.shape_interval(confidence).expect(checked)),
                dto(r.scale_interval(confidence).expect(checked)),
                fractions_failed
                    .iter()
                    .map(|&f| dto(ra.b_life_interval(f, confidence).expect(checked).1))
                    .collect(),
            )
        } else {
            use crate::weibull::{weibull_likelihood_interval, WeibullQuantity};
            let profile = |q| {
                weibull_likelihood_interval(times, suspensions, q, confidence)
                    .map(|(_, i)| dto(i))
                    .ok_or_else(|| {
                        WireError::invalid_input(
                            "weibull_mle: the likelihood does not fall far enough on one side for a \
                             likelihood-ratio limit (too few failures for it); use method \"wald\"",
                        )
                        .about("method")
                    })
            };
            (
                profile(WeibullQuantity::Shape)?,
                profile(WeibullQuantity::Scale)?,
                fractions_failed
                    .iter()
                    .map(|&f| profile(WeibullQuantity::BLife(f)))
                    .collect::<Result<_, _>>()?,
            )
        };
        let intervals = WeibullIntervalsDto {
            confidence,
            method,
            shape,
            scale,
            b_life: b_life_intervals,
        };
        Ok(WeibullMleDto {
            shape: r.shape,
            scale: r.scale,
            log_likelihood: r.log_likelihood,
            iterations: r.iterations,
            b_life,
            intervals,
            covariance: r.covariance,
        })
    }

    /// `suspensions` move the failures' ranks (Johnson); empty fits the
    /// failures alone.
    pub(crate) fn weibull_mrr_dto(
        times: &[f64],
        suspensions: &[f64],
    ) -> Result<WeibullMrrDto, WireError> {
        positive_values("failure_times", times, 2)?;
        positive_values("suspension_times", suspensions, 0)?;
        let r = crate::weibull::weibull_mrr_censored(times, suspensions)
            .ok_or_else(|| degenerate("weibull_mrr"))?;
        Ok(WeibullMrrDto {
            shape: r.shape,
            scale: r.scale,
            r_squared: r.r_squared,
        })
    }

    /// Fractions failed, each strictly inside (0, 1) -- the B-lives to report.
    fn fractions(parameter: &'static str, values: &[f64]) -> Result<(), WireError> {
        for (i, &f) in values.iter().enumerate() {
            if !(f > 0.0 && f < 1.0) {
                let mut e = WireError::out_of_range(
                    parameter,
                    Bound::Open(0.0),
                    Bound::Open(1.0),
                    f,
                    format!("{parameter}[{i}]: must be strictly inside (0, 1), got {f}"),
                );
                e.index = Some(i);
                return Err(e);
            }
        }
        Ok(())
    }

    // ── Weibull reliability ───────────────────────────────────────────────

    /// `{ shape, scale, times?, fractions_failed?, covariance?, confidence? }`
    /// — a fitted (or known) Weibull and the points to evaluate it at; with
    /// the fit's `covariance`, intervals at `confidence`.
    #[derive(Deserialize, Debug)]
    #[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
    #[cfg_attr(all(test, feature = "wasm"), derive(schemars::JsonSchema))]
    #[serde(deny_unknown_fields)]
    pub(crate) struct WeibullReliabilityInputDto {
        pub(crate) shape: f64,
        pub(crate) scale: f64,
        /// Times at which to evaluate `R(t)` and `h(t)`.
        #[serde(default)]
        #[cfg_attr(feature = "wasm", tsify(optional))]
        pub(crate) times: Vec<f64>,
        /// Fractions failed (each in `(0, 1)`) whose B-life to report;
        /// `0.1` is B10. (The time to reliability `p` is the B-life at `1 − p`.)
        #[serde(default)]
        #[cfg_attr(feature = "wasm", tsify(optional))]
        pub(crate) fractions_failed: Vec<f64>,
        /// Covariance of the estimated `(shape, scale)` (`weibull_mle`'s
        /// `covariance`). With it the answer carries `intervals`.
        #[serde(default)]
        #[cfg_attr(feature = "wasm", tsify(optional))]
        #[cfg_attr(
            feature = "wasm",
            tsify(type = "[[number, number], [number, number]] | null")
        )]
        pub(crate) covariance: Option<[[f64; 2]; 2]>,
        /// Confidence level of `intervals`, in (0, 1). Omitted: 0.95. Given
        /// without `covariance` it is refused -- there is nothing to apply it to.
        #[serde(default)]
        #[cfg_attr(feature = "wasm", tsify(optional))]
        #[cfg_attr(feature = "wasm", tsify(type = "number | null"))]
        pub(crate) confidence: Option<f64>,
    }

    /// Two-sided intervals at `confidence`, aligned with the requested times
    /// and fractions: delta method on `ln(−ln R)` and `ln t_p`.
    #[derive(Serialize, Debug)]
    #[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
    #[cfg_attr(all(test, feature = "wasm"), derive(schemars::JsonSchema))]
    #[cfg_attr(feature = "wasm", tsify(missing_as_null))]
    pub(crate) struct WeibullReliabilityIntervalsDto {
        pub(crate) confidence: f64,
        pub(crate) reliability: Vec<crate::wire::IntervalDto>,
        pub(crate) b_life: Vec<crate::wire::IntervalDto>,
    }

    /// Reliability metrics of a Weibull, aligned with the input arrays.
    #[derive(Serialize, Debug)]
    #[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
    #[cfg_attr(all(test, feature = "wasm"), derive(schemars::JsonSchema))]
    #[cfg_attr(feature = "wasm", tsify(missing_as_null))]
    pub(crate) struct WeibullReliabilityDto {
        /// Mean time between (to) failure, `η·Γ(1 + 1/β)`.
        pub(crate) mtbf: f64,
        /// `R(t)` for each of `times`.
        pub(crate) reliability: Vec<f64>,
        /// `h(t)` for each of `times`.
        pub(crate) hazard_rate: Vec<f64>,
        /// The B-life for each of `fractions_failed`.
        pub(crate) b_life: Vec<f64>,
        /// Intervals for `reliability` and `b_life`; `null` without a
        /// `covariance`.
        pub(crate) intervals: Option<WeibullReliabilityIntervalsDto>,
    }

    pub(crate) fn weibull_reliability_dto(
        input: WeibullReliabilityInputDto,
    ) -> Result<WeibullReliabilityDto, WireError> {
        let shape = positive("shape", input.shape)?;
        let scale = positive("scale", input.scale)?;
        for (i, &t) in input.times.iter().enumerate() {
            if !t.is_finite() {
                return Err(WireError::new(
                    code::VALUE_NOT_FINITE,
                    Some(i),
                    format!("times[{i}]: expected a finite number, got {t}"),
                )
                .about("times"));
            }
        }
        fractions("fractions_failed", &input.fractions_failed)?;
        let mut ra = crate::weibull::ReliabilityAnalysis::new(shape, scale)
            .expect("shape and scale were checked finite and > 0");
        let confidence = match (input.covariance, input.confidence) {
            (None, None) => None,
            (None, Some(_)) => {
                return Err(WireError::new(
                    code::INVALID_OPTION,
                    None,
                    "confidence sets the level of the intervals, which need the fit's covariance",
                )
                .about("confidence"))
            }
            (Some(covariance), confidence) => {
                for (i, v) in covariance.iter().flatten().enumerate() {
                    if !v.is_finite() {
                        return Err(WireError::new(
                            code::VALUE_NOT_FINITE,
                            Some(i),
                            format!("covariance: expected finite numbers, got {v}"),
                        )
                        .about("covariance"));
                    }
                }
                ra = ra.with_covariance(covariance).ok_or_else(|| {
                    WireError::new(
                        code::INVALID_OPTION,
                        None,
                        "covariance must be symmetric and positive definite",
                    )
                    .about("covariance")
                })?;
                let confidence = confidence.unwrap_or(0.95);
                if !(confidence > 0.0 && confidence < 1.0) {
                    return Err(WireError::out_of_range(
                        "confidence",
                        Bound::Open(0.0),
                        Bound::Open(1.0),
                        confidence,
                        format!("confidence must be strictly between 0 and 1, got {confidence}"),
                    ));
                }
                Some(confidence)
            }
        };
        let interval = |(_, i): (f64, crate::weibull::WeibullInterval)| crate::wire::IntervalDto {
            lower: i.lower,
            upper: i.upper,
        };
        let intervals = confidence.map(|confidence| WeibullReliabilityIntervalsDto {
            confidence,
            reliability: input
                .times
                .iter()
                .map(|&t| {
                    interval(
                        ra.reliability_interval(t, confidence)
                            .expect("covariance, confidence and time were checked"),
                    )
                })
                .collect(),
            b_life: input
                .fractions_failed
                .iter()
                .map(|&f| {
                    interval(
                        ra.b_life_interval(f, confidence)
                            .expect("covariance, confidence and fraction were checked"),
                    )
                })
                .collect(),
        });
        Ok(WeibullReliabilityDto {
            mtbf: ra.mtbf(),
            reliability: input.times.iter().map(|&t| ra.reliability(t)).collect(),
            hazard_rate: input.times.iter().map(|&t| ra.hazard_rate(t)).collect(),
            b_life: input
                .fractions_failed
                .iter()
                .map(|&f| ra.b_life(f).expect("fraction checked inside (0, 1)"))
                .collect(),
            intervals,
        })
    }

    // ── Sigma level ↔ PPM ─────────────────────────────────────────────────

    /// PPM at a sigma level (1.5σ shift convention).
    pub(crate) fn sigma_to_ppm_value(sigma: f64) -> Result<f64, WireError> {
        if !sigma.is_finite() {
            return Err(WireError::new(
                code::VALUE_NOT_FINITE,
                None,
                format!("sigma must be a finite number, got {sigma}"),
            )
            .about("sigma"));
        }
        Ok(crate::capability::sigma_to_ppm(sigma))
    }

    /// Sigma level at a PPM, for `ppm` strictly inside `(0, 1_000_000)`.
    pub(crate) fn ppm_to_sigma_value(ppm: f64) -> Result<f64, WireError> {
        crate::capability::ppm_to_sigma(ppm).ok_or_else(|| {
            WireError::out_of_range(
                "ppm",
                Bound::Open(0.0),
                Bound::Open(1_000_000.0),
                ppm,
                format!("ppm must be finite and strictly inside (0, 1000000), got {ppm}"),
            )
        })
    }

    // ── Box-Cox capability ────────────────────────────────────────────────

    #[derive(Deserialize, Debug)]
    #[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
    #[cfg_attr(all(test, feature = "wasm"), derive(schemars::JsonSchema))]
    #[cfg_attr(feature = "wasm", tsify(missing_as_null))]
    #[serde(deny_unknown_fields)]
    pub(crate) struct BoxcoxCapabilityInputDto {
        pub(crate) data: Vec<f64>,
        #[serde(default)]
        #[cfg_attr(feature = "wasm", tsify(optional, type = "number | null"))]
        pub(crate) usl: Option<f64>,
        #[serde(default)]
        #[cfg_attr(feature = "wasm", tsify(optional, type = "number | null"))]
        pub(crate) lsl: Option<f64>,
        /// `[min, max]` lambda search range; defaults to `[-5, 5]`.
        #[serde(default)]
        #[cfg_attr(feature = "wasm", tsify(optional, type = "[number, number] | null"))]
        pub(crate) lambda_range: Option<[f64; 2]>,
    }

    #[derive(Serialize, Debug)]
    #[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
    #[cfg_attr(all(test, feature = "wasm"), derive(schemars::JsonSchema))]
    #[cfg_attr(feature = "wasm", tsify(missing_as_null))]
    pub(crate) struct BoxcoxCapabilityDto {
        /// Estimated optimal Box-Cox parameter. `0` is a log transform, `1` the
        /// identity, `0.5` approximately a square root.
        pub(crate) lambda: f64,
        /// `true` when the likelihood maximum is on an end of `lambda_range`, so
        /// `lambda` is that limit rather than an interior estimate.
        pub(crate) lambda_at_bound: bool,
        /// Every index below is on the **transformed** scale, which is where the
        /// normal-theory formulas are valid -- they are not comparable to indices
        /// computed on the raw non-normal data.
        pub(crate) cp: Option<f64>,
        pub(crate) cpk: Option<f64>,
        pub(crate) cpu: Option<f64>,
        pub(crate) cpl: Option<f64>,
        pub(crate) pp: Option<f64>,
        pub(crate) ppk: Option<f64>,
        pub(crate) ppu: Option<f64>,
        pub(crate) ppl: Option<f64>,
        pub(crate) cpm: Option<f64>,
    }

    pub(crate) fn boxcox_capability_dto(
        input: BoxcoxCapabilityInputDto,
    ) -> Result<BoxcoxCapabilityDto, WireError> {
        use crate::capability::NonNormalCapabilityError as E;
        positive_values("data", &input.data, 4)?;
        for (name, limit) in [("usl", input.usl), ("lsl", input.lsl)] {
            if let Some(v) = limit {
                positive(name, v)?;
            }
        }
        let range = input
            .lambda_range
            .map_or(crate::capability::DEFAULT_LAMBDA_RANGE, |[lo, hi]| (lo, hi));
        let result = crate::capability::boxcox_capability(&input.data, input.usl, input.lsl, range)
            .map_err(|e| {
                let message = format!("boxcox_capability: {e}");
                match e {
                    E::InvalidLambdaRange => {
                        WireError::new(code::INVALID_OPTION, None, message).about("lambda_range")
                    }
                    E::CapabilityError => WireError::invalid_input(message).about("data"),
                    // Checked above with their place; kept for completeness.
                    E::NonPositiveData => {
                        WireError::new(code::NON_POSITIVE_DATA, None, message).about("data")
                    }
                    E::NonFiniteData => {
                        WireError::new(code::VALUE_NOT_FINITE, None, message).about("data")
                    }
                    E::InsufficientData => {
                        WireError::new(code::INSUFFICIENT_DATA, None, message).about("data")
                    }
                    E::SpecTransformError => WireError::invalid_input(message),
                }
            })?;
        let i = result.indices.as_ref();
        Ok(BoxcoxCapabilityDto {
            lambda: result.lambda,
            lambda_at_bound: result.lambda_at_bound,
            cp: i.and_then(|i| i.cp),
            cpk: i.and_then(|i| i.cpk),
            cpu: i.and_then(|i| i.cpu),
            cpl: i.and_then(|i| i.cpl),
            pp: i.and_then(|i| i.pp),
            ppk: i.and_then(|i| i.ppk),
            ppu: i.and_then(|i| i.ppu),
            ppl: i.and_then(|i| i.ppl),
            cpm: i.and_then(|i| i.cpm),
        })
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn weibull_fits_refuse_with_the_place() {
            let e = weibull_mle_dto(&[10.0, 20.0], &[30.0, -1.0], None, None, &[]).unwrap_err();
            assert_eq!(
                (e.code, e.parameter.as_deref(), e.index),
                ("non_positive_data", Some("suspension_times"), Some(1))
            );
            let fit =
                weibull_mle_dto(&[55.0, 187.0, 216.0], &[361.0, 373.0], None, None, &[]).unwrap();
            let alone = weibull_mle_dto(&[55.0, 187.0, 216.0], &[], None, None, &[]).unwrap();
            assert!(fit.scale > alone.scale);
            assert_eq!(fit.intervals.confidence, 0.95);
            assert!(fit.intervals.shape.lower < fit.shape && fit.shape < fit.intervals.shape.upper);
            assert!(fit.intervals.scale.lower < fit.scale && fit.scale < fit.intervals.scale.upper);
            let e = weibull_mle_dto(&[55.0, 187.0, 216.0], &[], Some(1.0), None, &[]).unwrap_err();
            assert_eq!(
                (e.code, e.parameter.as_deref()),
                ("parameter_out_of_range", Some("confidence"))
            );
            let e = weibull_mle_dto(&[10.0, -1.0], &[], None, None, &[]).unwrap_err();
            assert_eq!(
                (e.code, e.index, e.parameter.as_deref()),
                ("non_positive_data", Some(1), Some("failure_times"))
            );
            let e = weibull_mrr_dto(&[10.0, f64::NAN], &[]).unwrap_err();
            assert_eq!((e.code, e.index), ("value_not_finite", Some(1)));
            let e = weibull_mrr_dto(&[10.0], &[]).unwrap_err();
            assert_eq!(
                e.details,
                vec![("min", Detail::Num(2.0)), ("got", Detail::Num(1.0))]
            );
            let e = weibull_mle_dto(&[5.0, 5.0, 5.0], &[], None, None, &[]).unwrap_err();
            assert_eq!(e.code, "invalid_input");
            let ok = weibull_mrr_dto(&[150.0, 200.0, 250.0, 300.0, 350.0, 400.0], &[]).unwrap();
            assert!(ok.shape > 0.0 && ok.r_squared > 0.9);
            let censored =
                weibull_mrr_dto(&[150.0, 200.0, 250.0, 300.0, 350.0, 400.0], &[450.0, 450.0])
                    .unwrap();
            assert!(censored.scale > ok.scale);
            let e = weibull_mrr_dto(&[150.0, 200.0], &[0.0]).unwrap_err();
            assert_eq!(e.parameter.as_deref(), Some("suspension_times"));
        }

        #[test]
        fn reliability_is_aligned_with_its_inputs() {
            let r = weibull_reliability_dto(WeibullReliabilityInputDto {
                shape: 2.0,
                scale: 100.0,
                times: vec![0.0, 100.0],
                fractions_failed: vec![0.1, 0.5],
                covariance: None,
                confidence: None,
            })
            .unwrap();
            assert!((r.reliability[1] - (-1.0f64).exp()).abs() < 1e-12);
            assert_eq!(r.reliability[0], 1.0);
            assert_eq!((r.hazard_rate.len(), r.b_life.len()), (2, 2));
            // B10 = η·(−ln 0.9)^(1/β)
            assert!((r.b_life[0] - 100.0 * (-(0.9f64).ln()).sqrt()).abs() < 1e-9);
            assert!((r.mtbf - 100.0 * 0.886_226_925_452_758).abs() < 1e-9);

            let e = weibull_reliability_dto(WeibullReliabilityInputDto {
                shape: 0.0,
                scale: 1.0,
                times: vec![],
                fractions_failed: vec![],
                covariance: None,
                confidence: None,
            })
            .unwrap_err();
            assert_eq!(
                (e.code, e.parameter.as_deref()),
                ("parameter_out_of_range", Some("shape"))
            );
            let e = weibull_reliability_dto(WeibullReliabilityInputDto {
                shape: 1.0,
                scale: 1.0,
                times: vec![],
                fractions_failed: vec![0.5, 1.0],
                covariance: None,
                confidence: None,
            })
            .unwrap_err();
            assert_eq!(
                (e.code, e.index, e.parameter.as_deref()),
                ("parameter_out_of_range", Some(1), Some("fractions_failed"))
            );
        }

        #[test]
        fn the_mle_s_intervals_take_a_method() {
            let (f, s) = (
                &[55.0, 187.0, 216.0, 240.0, 244.0, 335.0][..],
                &[361.0, 373.0][..],
            );
            let wald = weibull_mle_dto(f, s, Some(0.9), None, &[]).unwrap();
            assert_eq!(wald.intervals.method, "wald");
            let lr = weibull_mle_dto(f, s, Some(0.9), Some("likelihood_ratio"), &[]).unwrap();
            assert_eq!(lr.intervals.method, "likelihood_ratio");
            let (_, core) = crate::weibull::weibull_likelihood_interval(
                f,
                s,
                crate::weibull::WeibullQuantity::Scale,
                0.9,
            )
            .unwrap();
            assert_eq!(
                (lr.intervals.scale.lower, lr.intervals.scale.upper),
                (core.lower, core.upper)
            );
            // Same fit, same covariance: only the intervals differ.
            assert_eq!((lr.shape, lr.covariance), (wald.shape, wald.covariance));
            assert_ne!(lr.intervals.shape, wald.intervals.shape);
            let e = weibull_mle_dto(f, s, None, Some("bootstrap"), &[]).unwrap_err();
            assert_eq!(
                (e.code, e.parameter.as_deref()),
                ("unknown_option", Some("method"))
            );
            // B-lives at the requested fractions, by the same method.
            let wald = weibull_mle_dto(f, s, Some(0.9), None, &[0.1, 0.5]).unwrap();
            let lr =
                weibull_mle_dto(f, s, Some(0.9), Some("likelihood_ratio"), &[0.1, 0.5]).unwrap();
            assert_eq!(wald.b_life, lr.b_life);
            assert_eq!(
                (wald.intervals.b_life.len(), lr.intervals.b_life.len()),
                (2, 2)
            );
            let core = crate::weibull::weibull_mle_censored(f, s).unwrap();
            let (b10, ci) = core.b_life_interval(0.1, 0.9).unwrap();
            assert_eq!(
                (
                    wald.b_life[0],
                    wald.intervals.b_life[0].lower,
                    wald.intervals.b_life[0].upper
                ),
                (b10, ci.lower, ci.upper)
            );
            let (_, lr10) = crate::weibull::weibull_likelihood_interval(
                f,
                s,
                crate::weibull::WeibullQuantity::BLife(0.1),
                0.9,
            )
            .unwrap();
            assert_eq!(
                (lr.intervals.b_life[0].lower, lr.intervals.b_life[0].upper),
                (lr10.lower, lr10.upper)
            );
            let e = weibull_mle_dto(f, s, None, None, &[0.1, 1.0]).unwrap_err();
            assert_eq!(
                (e.code, e.parameter.as_deref(), e.index),
                ("parameter_out_of_range", Some("fractions_failed"), Some(1))
            );
        }

        /// The fit's covariance, passed through, gives the intervals the
        /// core gives the fit -- one definition.
        #[test]
        fn reliability_intervals_come_from_the_fit_s_covariance() {
            let times = [55.0, 187.0, 216.0, 240.0, 244.0, 335.0, 361.0, 373.0];
            let fit = weibull_mle_dto(&times, &[400.0, 400.0], None, None, &[]).unwrap();
            let input = |covariance, confidence| WeibullReliabilityInputDto {
                shape: fit.shape,
                scale: fit.scale,
                times: vec![0.0, 150.0],
                fractions_failed: vec![0.1],
                covariance,
                confidence,
            };
            let r = weibull_reliability_dto(input(Some(fit.covariance), Some(0.9))).unwrap();
            let i = r.intervals.as_ref().expect("covariance given");
            assert_eq!(i.confidence, 0.9);
            assert_eq!((i.reliability.len(), i.b_life.len()), (2, 1));
            assert_eq!((i.reliability[0].lower, i.reliability[0].upper), (1.0, 1.0));
            assert!(
                i.reliability[1].lower < r.reliability[1]
                    && r.reliability[1] < i.reliability[1].upper
            );
            let core = crate::weibull::weibull_mle_censored(&times, &[400.0, 400.0]).unwrap();
            let (b10, ci) = core.b_life_interval(0.1, 0.9).unwrap();
            assert_eq!(
                (r.b_life[0], i.b_life[0].lower, i.b_life[0].upper),
                (b10, ci.lower, ci.upper)
            );
            let r = weibull_reliability_dto(input(Some(fit.covariance), None)).unwrap();
            assert_eq!(r.intervals.expect("default level").confidence, 0.95);
            assert!(weibull_reliability_dto(input(None, None))
                .unwrap()
                .intervals
                .is_none());

            let refused = |covariance, confidence| {
                let e = weibull_reliability_dto(input(covariance, confidence)).unwrap_err();
                (e.code, e.parameter, e.index)
            };
            assert_eq!(
                refused(None, Some(0.9)),
                ("invalid_option", Some("confidence".into()), None)
            );
            assert_eq!(
                refused(Some([[0.04, 2.0], [2.0, 25.0]]), None),
                ("invalid_option", Some("covariance".into()), None)
            );
            assert_eq!(
                refused(Some([[0.04, 0.1], [0.1, f64::INFINITY]]), None),
                ("value_not_finite", Some("covariance".into()), Some(3))
            );
            assert_eq!(
                refused(Some(fit.covariance), Some(1.0)),
                ("parameter_out_of_range", Some("confidence".into()), None)
            );
        }

        #[test]
        fn sigma_and_ppm_refuse_by_code() {
            assert!((ppm_to_sigma_value(3.4).unwrap() - 6.0).abs() < 1e-2);
            let e = ppm_to_sigma_value(0.0).unwrap_err();
            assert_eq!(
                (e.code, e.parameter.as_deref()),
                ("parameter_out_of_range", Some("ppm"))
            );
            // (0, 1 000 000): both ends open.
            assert_eq!(
                e.details,
                vec![
                    ("min", Detail::Null),
                    ("exclusive_min", Detail::Num(0.0)),
                    ("max", Detail::Null),
                    ("exclusive_max", Detail::Num(1_000_000.0)),
                    ("got", Detail::Num(0.0)),
                ]
            );
            let e = sigma_to_ppm_value(f64::NAN).unwrap_err();
            assert_eq!(
                (e.code, e.parameter.as_deref()),
                ("value_not_finite", Some("sigma"))
            );
        }

        #[test]
        fn boxcox_refusals_name_their_input() {
            let input = |data: Vec<f64>, usl: Option<f64>, range: Option<[f64; 2]>| {
                BoxcoxCapabilityInputDto {
                    data,
                    usl,
                    lsl: None,
                    lambda_range: range,
                }
            };
            let e = boxcox_capability_dto(input(vec![1.0, 2.0, 0.0, 3.0], Some(9.0), None))
                .unwrap_err();
            assert_eq!((e.code, e.index), ("non_positive_data", Some(2)));
            let e = boxcox_capability_dto(input(vec![1.0, 2.0, 3.0], Some(9.0), None)).unwrap_err();
            assert_eq!(
                e.details,
                vec![("min", Detail::Num(4.0)), ("got", Detail::Num(3.0))]
            );
            let e = boxcox_capability_dto(input(vec![1.0, 2.0, 3.0, 4.0], Some(-1.0), None))
                .unwrap_err();
            assert_eq!(
                (e.code, e.parameter.as_deref()),
                ("parameter_out_of_range", Some("usl"))
            );
            let e =
                boxcox_capability_dto(input(vec![1.0, 2.0, 3.0, 4.0], Some(9.0), Some([2.0, 1.0])))
                    .unwrap_err();
            assert_eq!(
                (e.code, e.parameter.as_deref()),
                ("invalid_option", Some("lambda_range"))
            );
        }
    }
}

/// Increments of a cumulative counter (`crate::counter`).
pub(crate) mod counter {
    use super::*;
    use crate::counter::{
        counter_increments, counter_totals, CounterError, CounterIncrement, CounterOptions,
        Discontinuity, IncrementKind,
    };

    /// A reset or replacement the caller declares: when, the register's value
    /// right after it, and (when read) the old register's last value.
    #[derive(Deserialize, Debug, Clone, Copy, PartialEq)]
    #[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
    #[cfg_attr(all(test, feature = "wasm"), derive(schemars::JsonSchema))]
    #[serde(deny_unknown_fields)]
    pub(crate) struct DiscontinuityDto {
        pub(crate) time: f64,
        pub(crate) start: f64,
        #[serde(default)]
        #[cfg_attr(feature = "wasm", tsify(optional, type = "number | null"))]
        pub(crate) last: Option<f64>,
    }

    /// The settings of a counter's increments; every one may be left out.
    #[derive(Deserialize, Debug, Clone, Default, PartialEq)]
    #[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
    #[cfg_attr(all(test, feature = "wasm"), derive(schemars::JsonSchema))]
    #[serde(deny_unknown_fields)]
    pub(crate) struct CounterOptionsDto {
        /// The value at which the register returns to 0.
        #[serde(default)]
        #[cfg_attr(feature = "wasm", tsify(optional, type = "number | null"))]
        pub(crate) modulus: Option<f64>,
        /// The most the counter can advance per unit of time.
        #[serde(default)]
        #[cfg_attr(feature = "wasm", tsify(optional, type = "number | null"))]
        pub(crate) max_rate: Option<f64>,
        /// Resets and replacements, in time order.
        #[serde(default)]
        #[cfg_attr(feature = "wasm", tsify(optional))]
        pub(crate) discontinuities: Vec<DiscontinuityDto>,
    }

    impl From<&CounterOptionsDto> for CounterOptions {
        fn from(o: &CounterOptionsDto) -> Self {
            CounterOptions {
                modulus: o.modulus,
                max_rate: o.max_rate,
                discontinuities: o
                    .discontinuities
                    .iter()
                    .map(|d| Discontinuity {
                        time: d.time,
                        start: d.start,
                        last: d.last,
                    })
                    .collect(),
            }
        }
    }

    /// How much the counter advanced between two neighbouring readings.
    #[derive(Serialize, Debug, PartialEq)]
    #[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
    #[cfg_attr(all(test, feature = "wasm"), derive(schemars::JsonSchema))]
    #[cfg_attr(feature = "wasm", tsify(missing_as_null))]
    pub(crate) struct CounterIncrementDto {
        pub(crate) start: f64,
        pub(crate) end: f64,
        /// `null` when `kind` is `"unexplained"`.
        pub(crate) increase: Option<f64>,
        #[cfg_attr(
            feature = "wasm",
            tsify(type = "\"normal\" | \"declared\" | \"wrap\" | \"unexplained\"")
        )]
        #[cfg_attr(all(test, feature = "wasm"), schemars(extend("enum" = ["normal", "declared", "wrap", "unexplained"])))]
        pub(crate) kind: &'static str,
    }

    impl From<CounterIncrement> for CounterIncrementDto {
        fn from(i: CounterIncrement) -> Self {
            CounterIncrementDto {
                start: i.start,
                end: i.end,
                increase: i.increase,
                kind: match i.kind {
                    IncrementKind::Normal => "normal",
                    IncrementKind::Declared => "declared",
                    IncrementKind::Wrap => "wrap",
                    IncrementKind::Unexplained => "unexplained",
                },
            }
        }
    }

    /// The increments that end in one interval `[start, end)`.
    #[derive(Serialize, Debug, PartialEq)]
    #[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
    #[cfg_attr(all(test, feature = "wasm"), derive(schemars::JsonSchema))]
    pub(crate) struct CounterTotalDto {
        pub(crate) start: f64,
        pub(crate) end: f64,
        pub(crate) increase: f64,
        pub(crate) pairs: usize,
        pub(crate) unexplained: usize,
    }

    fn refusal(e: CounterError) -> WireError {
        let message = e.to_string();
        match e {
            CounterError::LengthMismatch { times, values } => {
                WireError::new(code::DIMENSION_MISMATCH, None, message)
                    .about("values")
                    .with("expected", Detail::Num(times as f64))
                    .with("got", Detail::Num(values as f64))
            }
            CounterError::TooFewReadings { got } => {
                // Readings, or interval edges: both need two.
                WireError::too_few("times", 2, got, message)
            }
            CounterError::NotFinite { parameter, index } => {
                WireError::new(code::VALUE_NOT_FINITE, index, message).about(parameter)
            }
            CounterError::Unordered { parameter, index } => {
                WireError::new(code::EVENTS_UNORDERED, Some(index), message).about(parameter)
            }
            CounterError::NotPositive { parameter, got } => {
                WireError::out_of_range(parameter, Bound::Open(0.0), Bound::Unbounded, got, message)
            }
            CounterError::OutsideModulus {
                parameter,
                index,
                got,
                modulus,
            } => {
                let mut e = WireError::out_of_range(
                    parameter,
                    Bound::Closed(0.0),
                    Bound::Open(modulus),
                    got,
                    message,
                );
                e.index = Some(index);
                e
            }
            CounterError::OutsideReadings {
                index,
                got,
                first,
                last,
            } => {
                let mut e = WireError::out_of_range(
                    "discontinuities.time",
                    Bound::Open(first),
                    Bound::Closed(last),
                    got,
                    message,
                );
                e.index = Some(index);
                e
            }
            CounterError::TwoInOnePair { index } => {
                WireError::new(code::INVALID_OPTION, Some(index), message).about("discontinuities")
            }
        }
    }

    /// The C ABI's request for the increments: the readings and, flat beside
    /// them, the settings.
    #[cfg(feature = "ffi")]
    #[derive(Deserialize, Debug)]
    #[serde(deny_unknown_fields)]
    pub(crate) struct CounterRequestDto {
        pub(crate) times: Vec<f64>,
        pub(crate) values: Vec<f64>,
        #[serde(default)]
        pub(crate) modulus: Option<f64>,
        #[serde(default)]
        pub(crate) max_rate: Option<f64>,
        #[serde(default)]
        pub(crate) discontinuities: Vec<DiscontinuityDto>,
    }

    /// The C ABI's request for the totals: the increments' request and the
    /// interval `edges`.
    #[cfg(feature = "ffi")]
    #[derive(Deserialize, Debug)]
    #[serde(deny_unknown_fields)]
    pub(crate) struct CounterTotalsRequestDto {
        pub(crate) times: Vec<f64>,
        pub(crate) values: Vec<f64>,
        pub(crate) edges: Vec<f64>,
        #[serde(default)]
        pub(crate) modulus: Option<f64>,
        #[serde(default)]
        pub(crate) max_rate: Option<f64>,
        #[serde(default)]
        pub(crate) discontinuities: Vec<DiscontinuityDto>,
    }

    #[cfg(feature = "ffi")]
    pub(crate) fn options_of(
        modulus: Option<f64>,
        max_rate: Option<f64>,
        discontinuities: &[DiscontinuityDto],
    ) -> CounterOptionsDto {
        CounterOptionsDto {
            modulus,
            max_rate,
            discontinuities: discontinuities.to_vec(),
        }
    }

    pub(crate) fn increments_dto(
        times: &[f64],
        values: &[f64],
        options: &CounterOptionsDto,
    ) -> Result<Vec<CounterIncrementDto>, WireError> {
        counter_increments(times, values, &options.into())
            .map(|v| v.into_iter().map(Into::into).collect())
            .map_err(refusal)
    }

    pub(crate) fn totals_dto(
        times: &[f64],
        values: &[f64],
        edges: &[f64],
        options: &CounterOptionsDto,
    ) -> Result<Vec<CounterTotalDto>, WireError> {
        let increments = counter_increments(times, values, &options.into()).map_err(refusal)?;
        let totals = counter_totals(&increments, edges).map_err(|e| match e {
            CounterError::TooFewReadings { got } => WireError::too_few(
                "edges",
                2,
                got,
                format!("at least 2 edges are needed for one interval, got {got}"),
            ),
            other => refusal(other),
        })?;
        Ok(totals
            .into_iter()
            .map(|t| CounterTotalDto {
                start: t.start,
                end: t.end,
                increase: t.increase,
                pairs: t.pairs,
                unexplained: t.unexplained,
            })
            .collect())
    }
}

#[cfg(test)]
mod counter_wire_tests {
    use super::counter::*;

    #[test]
    fn increments_carry_their_kind_on_the_wire() {
        let o = CounterOptionsDto {
            modulus: Some(1000.0),
            ..Default::default()
        };
        let out = increments_dto(&[0.0, 1.0, 2.0], &[990.0, 995.0, 4.0], &o).unwrap();
        assert_eq!(out[0].kind, "normal");
        assert_eq!((out[1].kind, out[1].increase), ("wrap", Some(9.0)));
        let out = increments_dto(&[0.0, 1.0], &[5.0, 4.0], &CounterOptionsDto::default()).unwrap();
        assert_eq!((out[0].kind, out[0].increase), ("unexplained", None));
    }

    #[test]
    fn counter_refusals_use_the_shared_reasons() {
        let none = CounterOptionsDto::default();
        let e = increments_dto(&[0.0, 1.0], &[1.0], &none).unwrap_err();
        assert_eq!(
            (e.code, e.parameter.as_deref()),
            ("dimension_mismatch", Some("values"))
        );
        let e = increments_dto(&[1.0, 1.0], &[1.0, 2.0], &none).unwrap_err();
        assert_eq!((e.code, e.index), ("events_unordered", Some(1)));
        let o = CounterOptionsDto {
            modulus: Some(10.0),
            ..Default::default()
        };
        let e = increments_dto(&[0.0, 1.0], &[1.0, 12.0], &o).unwrap_err();
        assert_eq!((e.code, e.index), ("parameter_out_of_range", Some(1)));
        let o = CounterOptionsDto {
            discontinuities: vec![DiscontinuityDto {
                time: 5.0,
                start: 0.0,
                last: None,
            }],
            ..Default::default()
        };
        let e = increments_dto(&[0.0, 1.0], &[1.0, 2.0], &o).unwrap_err();
        assert_eq!(
            (e.code, e.parameter.as_deref(), e.index),
            (
                "parameter_out_of_range",
                Some("discontinuities.time"),
                Some(0)
            )
        );
        let e = totals_dto(&[0.0, 1.0], &[1.0, 2.0], &[0.0], &none).unwrap_err();
        assert_eq!(
            (e.code, e.parameter.as_deref()),
            ("insufficient_data", Some("edges"))
        );
    }

    #[test]
    fn totals_sum_the_pairs_ending_in_each_interval() {
        let t = totals_dto(
            &[0.0, 6.0, 12.0, 24.0, 30.0],
            &[0.0, 1.0, 3.0, 4.0, 2.0],
            &[0.0, 24.0, 48.0],
            &CounterOptionsDto::default(),
        )
        .unwrap();
        assert_eq!((t[0].increase, t[0].pairs, t[0].unexplained), (3.0, 2, 0));
        assert_eq!((t[1].increase, t[1].pairs, t[1].unexplained), (1.0, 2, 1));
    }
}

/// Trend tests and the power-law fit on event times (`crate::point_process`).
pub(crate) mod point_process {
    use super::*;
    use crate::point_process::{PointProcessError, TrendDirection, TrendTestResult, Truncation};

    /// How observation ended: `{ truncation: "time", end }` — watched until
    /// `end` — or `{ truncation: "failure" }` — stopped at the last event.
    #[derive(Deserialize, Debug, Clone, Copy, PartialEq)]
    #[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
    #[cfg_attr(all(test, feature = "wasm"), derive(schemars::JsonSchema))]
    #[serde(tag = "truncation", rename_all = "snake_case", deny_unknown_fields)]
    pub(crate) enum ObservationDto {
        Time { end: f64 },
        Failure {},
    }

    /// The `truncation` names [`ObservationDto`] accepts.
    pub(crate) const TRUNCATIONS: &[&str] = &["time", "failure"];

    /// `truncation` is a name this function knows, or `unknown_option`.
    pub(crate) fn check_truncation(name: &str) -> Result<(), WireError> {
        if TRUNCATIONS.contains(&name) {
            return Ok(());
        }
        Err(WireError::unknown_option(
            "observation.truncation",
            name,
            TRUNCATIONS,
        ))
    }

    impl From<ObservationDto> for Truncation {
        fn from(o: ObservationDto) -> Self {
            match o {
                ObservationDto::Time { end } => Truncation::Time(end),
                ObservationDto::Failure {} => Truncation::Failure,
            }
        }
    }

    /// A trend test against a constant event rate.
    #[derive(Serialize, Debug)]
    #[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
    #[cfg_attr(all(test, feature = "wasm"), derive(schemars::JsonSchema))]
    #[cfg_attr(feature = "wasm", tsify(missing_as_null))]
    pub(crate) struct TrendTestDto {
        pub(crate) statistic: f64,
        /// Two-sided; the one-sided p-value in `direction` is half of it.
        pub(crate) p_value: f64,
        #[cfg_attr(
            feature = "wasm",
            tsify(type = "\"increasing\" | \"decreasing\" | \"flat\"")
        )]
        #[cfg_attr(all(test, feature = "wasm"), schemars(extend("enum" = ["increasing", "decreasing", "flat"])))]
        pub(crate) direction: &'static str,
        pub(crate) events_used: usize,
        /// χ² degrees of freedom (MIL-HDBK-189); `null` for Laplace.
        pub(crate) df: Option<f64>,
    }

    impl From<TrendTestResult> for TrendTestDto {
        fn from(r: TrendTestResult) -> Self {
            TrendTestDto {
                statistic: r.statistic,
                p_value: r.p_value,
                direction: match r.direction {
                    TrendDirection::Increasing => "increasing",
                    TrendDirection::Decreasing => "decreasing",
                    TrendDirection::Flat => "flat",
                },
                events_used: r.events_used,
                df: r.degrees_of_freedom,
            }
        }
    }

    /// A power-law process (Crow-AMSAA) fit.
    #[derive(Serialize, Debug)]
    #[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
    #[cfg_attr(all(test, feature = "wasm"), derive(schemars::JsonSchema))]
    #[cfg_attr(feature = "wasm", tsify(missing_as_null))]
    pub(crate) struct PowerLawFitDto {
        pub(crate) beta: f64,
        pub(crate) beta_unbiased: f64,
        pub(crate) lambda: f64,
        pub(crate) intensity_at_end: f64,
        pub(crate) end: f64,
        pub(crate) events: usize,
    }

    fn refusal(e: PointProcessError) -> WireError {
        let message = format!("times: {e}");
        match e {
            PointProcessError::TooFewEvents { min, got } => {
                WireError::too_few("times", min, got, message)
            }
            PointProcessError::NotFinite { index: Some(i) } => {
                WireError::new(code::VALUE_NOT_FINITE, Some(i), message).about("times")
            }
            PointProcessError::NotFinite { index: None } => {
                WireError::new(code::VALUE_NOT_FINITE, None, message).about("observation.end")
            }
            PointProcessError::NotPositive { index, got } => {
                let mut e = WireError::out_of_range(
                    if index.is_some() {
                        "times"
                    } else {
                        "observation.end"
                    },
                    Bound::Open(0.0),
                    Bound::Unbounded,
                    got,
                    message,
                );
                e.index = index;
                e
            }
            PointProcessError::Unordered { index } => {
                WireError::new(code::EVENTS_UNORDERED, Some(index), message).about("times")
            }
            PointProcessError::AfterEnd { index } => {
                WireError::new(code::EVENT_AFTER_END, Some(index), message).about("times")
            }
        }
    }

    pub(crate) fn laplace_dto(times: &[f64], o: ObservationDto) -> Result<TrendTestDto, WireError> {
        crate::point_process::laplace_trend_test(times, o.into())
            .map(Into::into)
            .map_err(refusal)
    }

    pub(crate) fn mil_hdbk_189_dto(
        times: &[f64],
        o: ObservationDto,
    ) -> Result<TrendTestDto, WireError> {
        crate::point_process::mil_hdbk_189_test(times, o.into())
            .map(Into::into)
            .map_err(refusal)
    }

    pub(crate) fn power_law_dto(
        times: &[f64],
        o: ObservationDto,
    ) -> Result<PowerLawFitDto, WireError> {
        crate::point_process::power_law_process_fit(times, o.into())
            .map(|f| PowerLawFitDto {
                beta: f.beta,
                beta_unbiased: f.beta_unbiased,
                lambda: f.lambda,
                intensity_at_end: f.intensity_at_end,
                end: f.end,
                events: f.events,
            })
            .map_err(refusal)
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use serde_json::json;

        #[test]
        fn observation_reads_both_shapes_and_refuses_unknown_keys() {
            let t: ObservationDto =
                serde_json::from_value(json!({ "truncation": "time", "end": 60 })).unwrap();
            assert_eq!(t, ObservationDto::Time { end: 60.0 });
            let f: ObservationDto =
                serde_json::from_value(json!({ "truncation": "failure" })).unwrap();
            assert_eq!(f, ObservationDto::Failure {});
            assert!(serde_json::from_value::<ObservationDto>(
                json!({ "truncation": "failure", "end": 3 })
            )
            .is_err());
            let e = check_truncation("timed").unwrap_err();
            assert_eq!(
                (e.code, e.parameter.as_deref()),
                ("unknown_option", Some("observation.truncation"))
            );
        }

        #[test]
        fn refusals_place_the_event() {
            let o = ObservationDto::Time { end: 10.0 };
            let e = laplace_dto(&[2.0, 1.0], o).unwrap_err();
            assert_eq!(
                (e.code, e.index, e.parameter.as_deref()),
                ("events_unordered", Some(1), Some("times"))
            );
            let e = laplace_dto(&[2.0, 11.0], o).unwrap_err();
            assert_eq!((e.code, e.index), ("event_after_end", Some(1)));
            let e = mil_hdbk_189_dto(&[2.0], ObservationDto::Time { end: 0.0 }).unwrap_err();
            assert_eq!(
                (e.code, e.parameter.as_deref()),
                ("parameter_out_of_range", Some("observation.end"))
            );
            let e = power_law_dto(&[1.0, 2.0], ObservationDto::Failure {}).unwrap_err();
            assert_eq!(e.code, "insufficient_data");
        }

        #[test]
        fn results_carry_the_direction_by_name() {
            let o = ObservationDto::Time { end: 60.0 };
            let r = mil_hdbk_189_dto(&[12.0, 15.0, 27.0, 34.0, 44.0, 53.0], o).unwrap();
            assert_eq!((r.direction, r.df), ("increasing", Some(12.0)));
            assert!(laplace_dto(&[12.0, 15.0], o).unwrap().df.is_none());
        }
    }
}

#[cfg(test)]
mod input_error_tests {
    /// Every refusal names what it is about (C464): these were strings that
    /// left `parameter` empty, several naming two settings in one sentence.
    #[test]
    fn refusals_name_their_parameter() {
        use super::*;
        let cap = |v: serde_json::Value| {
            capability_dto(serde_json::from_value(v).expect("valid shape")).expect_err("refused")
        };
        let e = cap(serde_json::json!({ "data": [1.0, 2.0, 3.0] }));
        assert_eq!((e.code, e.parameter.as_deref()), ("no_specification", None));
        let e = cap(serde_json::json!({ "data": [1.0, 2.0, 3.0], "usl": 1.0, "lsl": 2.0 }));
        assert_eq!(
            (e.code, e.parameter.as_deref()),
            ("parameter_out_of_range", Some("usl"))
        );
        let e = cap(serde_json::json!({ "data": [1.0], "usl": 2.0 }));
        assert_eq!(
            (e.code, e.parameter.as_deref()),
            ("insufficient_data", Some("data"))
        );

        let e = percentile_capability_dto(
            serde_json::from_value(
                serde_json::json!({ "data": [1.0, 2.0, 3.0, 4.0, 5.0], "usl": 2.0 }),
            )
            .unwrap(),
        )
        .err()
        .expect("5 < 20");
        assert_eq!(
            (e.code, e.parameter.as_deref()),
            ("insufficient_data", Some("data"))
        );

        // The family: named, refused by name, or refused at the first value its
        // support excludes -- and the result says which one it fitted.
        let pct = |extra: serde_json::Value| {
            let mut body = serde_json::json!({
                "data": (1..=30).map(|i| 10.0 + (i % 7) as f64 * 0.3).collect::<Vec<_>>(),
                "usl": 14.0, "lsl": 8.0,
            });
            for (k, v) in extra.as_object().unwrap() {
                body[k] = v.clone();
            }
            percentile_capability_dto(serde_json::from_value(body).unwrap())
        };
        let ok = pct(serde_json::json!({ "distribution": "Normal" })).expect("normal fits");
        assert_eq!(ok.distribution, "Normal");
        assert_eq!(
            ok.parameters.keys().map(String::as_str).collect::<Vec<_>>(),
            ["mu", "sigma"]
        );
        assert_ne!(
            pct(serde_json::json!({})).expect("auto").distribution,
            "Auto"
        );
        let e = pct(serde_json::json!({ "distribution": "normal" }))
            .err()
            .expect("unknown");
        assert_eq!(
            (e.code, e.parameter.as_deref()),
            ("unknown_option", Some("distribution"))
        );
        let mut low: Vec<f64> = (1..=30).map(|i| i as f64).collect();
        low[4] = -2.0;
        let e = pct(serde_json::json!({ "data": low, "lsl": -5.0, "distribution": "Weibull" }))
            .err()
            .expect("weibull needs positive data");
        assert_eq!(
            (e.code, e.parameter.as_deref(), e.index),
            ("non_positive_data", Some("data"), Some(4))
        );

        let grr = |m: serde_json::Value| {
            gage_rr_xbar_r_dto(
                serde_json::from_value(serde_json::json!({ "measurements": m })).unwrap(),
            )
            .err()
            .expect("refused")
        };
        let e = grr(serde_json::json!([
            [[1.0, 2.0], [1.0, 2.0]],
            [[1.0, 2.0], [1.0]]
        ]));
        assert_eq!(
            (e.code, e.parameter.as_deref(), e.index),
            ("dimension_mismatch", Some("measurements[1]"), Some(1))
        );
        let e = grr(serde_json::json!([
            [[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]],
            [[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]]
        ]));
        assert_eq!(
            (e.code, e.parameter.as_deref()),
            ("parameter_out_of_range", Some("measurements[0][0]"))
        );

        let e = imr_dto(vec![1.0], crate::spc::RuleSet::default()).expect_err("one value");
        assert_eq!(
            (e.code, e.parameter.as_deref()),
            ("insufficient_data", Some("values"))
        );

        let e = run_rules_dto(
            vec![1.0],
            LimitsInputDto {
                ucl: 1.0,
                cl: 2.0,
                lcl: 0.0,
            },
            crate::spc::RuleSet::default(),
        )
        .expect_err("ucl below cl");
        assert_eq!(
            (e.code, e.parameter.as_deref()),
            ("parameter_out_of_range", Some("limits.ucl"))
        );

        let e = seasonality_dto(SeasonalityInputDto { data: vec![1.0; 3] }).expect_err("too short");
        assert_eq!(
            (e.code, e.parameter.as_deref()),
            ("insufficient_data", Some("data"))
        );
    }

    /// Row walkers reported the row (`index`) but not whose rows they were.
    #[test]
    fn count_rows_name_their_argument_alongside_the_row() {
        let e = super::count_pairs(&serde_json::json!([[1, 10], [2]]), "samples")
            .expect_err("a row that is not a pair");
        assert_eq!(
            (e.code, e.parameter.as_deref(), e.index),
            ("malformed_input", Some("samples"), Some(1))
        );
        let e = super::rate_pairs(&serde_json::json!([[1, 2.0], [1.5, 2.0]]), "samples")
            .expect_err("1.5 defects");
        assert_eq!(
            (e.parameter.as_deref(), e.index),
            (Some("samples"), Some(1))
        );
        let e = super::count_rows(&serde_json::json!([3, -1]), "defects").expect_err("negative");
        assert_eq!(
            (e.parameter.as_deref(), e.index),
            (Some("defects"), Some(1))
        );
    }

    #[test]
    fn pelt_options_are_refused_by_name_with_their_values() {
        use super::{pelt_dto, Detail, PeltInputDto, PeltPenaltyDto, Whole};
        let input = |cost: &str, penalty: PeltPenaltyDto, min_segment_len: f64| PeltInputDto {
            data: vec![0.0, 0.0, 5.0, 5.0],
            cost: cost.to_string(),
            penalty,
            min_segment_len: Whole(min_segment_len),
        };
        let bic = || PeltPenaltyDto::Named("bic".into());

        let e = pelt_dto(input("l1", bic(), 2.0)).err().unwrap();
        assert_eq!(
            (e.code, e.parameter.as_deref()),
            ("unknown_option", Some("cost"))
        );
        assert_eq!(
            e.details[1],
            ("expected", Detail::List(vec!["l2", "normal"]))
        );

        let e = pelt_dto(input("l2", PeltPenaltyDto::Named("aic".into()), 2.0))
            .err()
            .unwrap();
        assert_eq!(
            (e.code, e.parameter.as_deref()),
            ("unknown_option", Some("penalty"))
        );
        assert_eq!(e.details[0], ("got", Detail::Text("aic".into())));

        let e = pelt_dto(input("l2", PeltPenaltyDto::Value(-1.0), 2.0))
            .err()
            .unwrap();
        assert_eq!(
            (e.code, e.parameter.as_deref()),
            ("parameter_out_of_range", Some("penalty"))
        );
        assert_eq!(
            e.details,
            vec![
                ("min", Detail::Null),
                ("exclusive_min", Detail::Num(0.0)),
                ("max", Detail::Null),
                ("got", Detail::Num(-1.0))
            ]
        );

        // A closed bound keeps its shape: `min`, no `exclusive_min`. A
        // negative length is refused by the same check, in the same shape.
        for got in [1.0, 0.0, -1.0] {
            let e = pelt_dto(input("l2", bic(), got)).err().unwrap();
            assert_eq!(
                (e.code, e.parameter.as_deref()),
                ("parameter_out_of_range", Some("min_segment_len"))
            );
            assert_eq!(
                e.details,
                vec![
                    ("min", Detail::Num(2.0)),
                    ("max", Detail::Null),
                    ("got", Detail::Num(got))
                ]
            );
        }
        assert!(pelt_dto(input("l2", bic(), 2.0)).is_ok());

        let mut empty = input("l2", bic(), 2.0);
        empty.data.clear();
        let e = pelt_dto(empty).err().unwrap();
        assert_eq!(
            (e.code, e.parameter.as_deref()),
            ("empty_input", Some("data"))
        );
    }

    #[test]
    fn too_few_values_carry_min_and_got() {
        let e = super::at_least(vec![1.0], 3, "data").unwrap_err();
        assert_eq!(
            (e.code, e.parameter.as_deref()),
            ("insufficient_data", Some("data"))
        );
        assert_eq!(
            e.details,
            vec![
                ("min", super::Detail::Num(3.0)),
                ("got", super::Detail::Num(1.0))
            ]
        );
    }

    use super::code;
    use super::*;
    use serde_json::json;

    fn err<T: std::fmt::Debug>(r: Result<T, WireError>) -> (&'static str, Option<usize>, String) {
        let e = r.expect_err("must be refused");
        (e.code, e.index, e.message)
    }

    /// A count that failed to deserialize used to carry no row at all
    /// (`invalid type: floating point 1.5, expected u64`): the count is now
    /// read as a JSON number and refused with its position.
    #[test]
    fn a_count_that_is_not_a_whole_number_is_refused_with_its_row() {
        for bad in [json!(1.5), json!(-1), json!("3"), json!(null), json!(1e300)] {
            let (c, i, m) = err(count_rows(&json!([2, 0, bad]), "defects"));
            assert_eq!((c, i), (code::COUNT_NOT_WHOLE.name(), Some(2)), "{m}");
            assert!(m.starts_with("defects[2]"), "{m}");
        }
        let (c, i, _) = err(count_pairs(&json!([[1, 10], [2.5, 10]]), "samples"));
        assert_eq!((c, i), (code::COUNT_NOT_WHOLE.name(), Some(1)));
        let (c, i, _) = err(rate_pairs(&json!([[1, 1.0], [0.5, 2.0]]), "samples"));
        assert_eq!((c, i), (code::COUNT_NOT_WHOLE.name(), Some(1)));
    }

    /// The other half of a `[defectives, sample_size]` pair has its own domain,
    /// so it has its own code: `count_not_whole` on a pair row used to mean
    /// either member, and which one was readable only in the message.
    #[test]
    fn every_way_a_sample_size_can_be_wrong_carries_one_code() {
        for bad in [
            json!(0),
            json!(-3),
            json!(10.5),
            json!("10"),
            json!(null),
            json!(1e300),
        ] {
            let (c, i, m) = err(count_pairs(&json!([[1, 10], [1, bad]]), "samples"));
            assert_eq!((c, i), (code::SAMPLE_SIZE_NOT_WHOLE.name(), Some(1)), "{m}");
            assert!(m.contains("sample size"), "{m}");
            assert!(m.contains("at least 1"), "{m}");
        }
        // Defectives are checked first, and keep the count code.
        let (c, i, m) = err(count_pairs(&json!([[1, 10], [1.5, 0]]), "samples"));
        assert_eq!((c, i), (code::COUNT_NOT_WHOLE.name(), Some(1)), "{m}");
        assert!(m.contains("defectives"), "{m}");
        // An NP chart's single size is not an element, so it carries no position.
        for bad in [json!(0), json!(-3), json!(10.5)] {
            let (c, i, _) = err(sample_size_value(&bad, "sample_size"));
            assert_eq!((c, i), (code::SAMPLE_SIZE_NOT_WHOLE.name(), None));
        }
    }

    #[test]
    fn whole_counts_written_as_floats_are_accepted() {
        // A JS number is a double; 3 and 3.0 are the same value.
        assert_eq!(
            count_rows(&json!([3.0, 0, 7]), "c").expect("whole"),
            vec![3, 0, 7]
        );
        assert_eq!(sample_size_value(&json!(100.0), "n").expect("whole"), 100);
        assert_eq!(
            count_pairs(&json!([[0, 1], [3.0, 10.0]]), "s").expect("whole"),
            vec![(0, 1), (3, 10)]
        );
    }

    #[test]
    fn a_row_that_is_not_a_pair_is_malformed_at_its_row() {
        let (c, i, _) = err(count_pairs(&json!([[1, 10], [1, 2, 3]]), "samples"));
        assert_eq!((c, i), (code::MALFORMED_INPUT.name(), Some(1)));
        let (c, i, _) = err(rate_pairs(&json!([[1, 1.0], 4]), "samples"));
        assert_eq!((c, i), (code::MALFORMED_INPUT.name(), Some(1)));
        let (c, i, _) = err(count_pairs(&json!({ "samples": [] }), "samples"));
        assert_eq!((c, i), (code::MALFORMED_INPUT.name(), None));
        let (c, i, _) = err(rate_pairs(&json!([[1, 1.0], [2, "x"]]), "samples"));
        assert_eq!((c, i), (code::UNITS_NOT_POSITIVE.name(), Some(1)));
    }

    /// The code and `parameter` together tell every chart refusal apart. The
    /// two that state a domain share `parameter_out_of_range` and differ by
    /// `parameter` (the subgroups, or the standard named) and carry the
    /// domain; every other reason has its own code.
    #[test]
    fn every_chart_error_is_told_apart_by_code_and_parameter() {
        use crate::spc::ControlChartError as E;
        let all = [
            E::SampleLengthMismatch {
                expected: 2,
                got: 3,
            },
            E::NonFiniteValue,
            E::DefectivesExceedSampleSize {
                defectives: 2,
                sample_size: 1,
            },
            E::NonPositiveUnits,
            E::SubgroupSizeOutOfRange {
                got: 1,
                min: 2,
                max: 25,
            },
            E::ZeroSampleSize,
            E::dispersion_standard("phi", -1.0),
            E::proportion_standard("p_bar", 1.5),
        ];
        let refusals: Vec<_> = all
            .iter()
            .map(|e| WireError::chart("subgroups", None, e))
            .collect();
        let reasons: std::collections::HashSet<_> = refusals
            .iter()
            .map(|w| (w.code, w.parameter.clone()))
            .collect();
        assert_eq!(reasons.len(), all.len(), "reasons must be told apart");
        assert!(!reasons.iter().any(|(c, _)| *c == code::INVALID_INPUT));
        // No range reason keeps a code of its own.
        assert!(refusals
            .iter()
            .all(|w| !w.code.ends_with("_out_of_range") || w.code == code::PARAMETER_OUT_OF_RANGE));

        let fields = |w: &WireError| -> std::collections::HashMap<&'static str, Detail> {
            w.details.iter().cloned().collect()
        };
        let num = Detail::Num;
        let size = fields(&refusals[4]);
        assert_eq!(
            (
                refusals[4].code,
                refusals[4].parameter.as_deref(),
                &size["min"],
                &size["max"],
                &size["got"]
            ),
            (
                code::PARAMETER_OUT_OF_RANGE.name(),
                Some("subgroups"),
                &num(2.0),
                &num(25.0),
                &num(1.0)
            )
        );
        let phi = fields(&refusals[6]);
        assert_eq!(refusals[6].parameter.as_deref(), Some("phi"));
        assert_eq!(
            (&phi["min"], &phi["max"], &phi["got"]),
            (&num(0.0), &Detail::Null, &num(-1.0))
        );
        assert!(!phi.contains_key("exclusive_min"));
        let p_bar = fields(&refusals[7]);
        assert_eq!(
            (
                &p_bar["min"],
                &p_bar["exclusive_min"],
                &p_bar["max"],
                &p_bar["exclusive_max"],
                &p_bar["got"]
            ),
            (
                &Detail::Null,
                &num(0.0),
                &Detail::Null,
                &num(1.0),
                &num(1.5)
            )
        );
    }

    #[test]
    fn the_error_serializes_as_code_index_parameter_message() {
        // Both locators are always present: `index` says where in the data,
        // `parameter` says which option, and each is null when it does not
        // apply -- so a consumer reads a field rather than testing for one.
        let e = WireError::new(code::COUNT_NOT_WHOLE, Some(4), "defects[4]: nope");
        assert_eq!(
            serde_json::to_value(&e).expect("serializable"),
            json!({
                "code": "count_not_whole", "index": 4,
                "parameter": null, "message": "defects[4]: nope"
            })
        );

        let e = WireError::new(code::PARAMETER_OUT_OF_RANGE, None, "threshold must be > 0")
            .about("threshold");
        assert_eq!(
            serde_json::to_value(&e).expect("serializable"),
            json!({
                "code": "parameter_out_of_range", "index": null,
                "parameter": "threshold", "message": "threshold must be > 0"
            })
        );
    }

    /// Each refused option arrives naming itself, with the value given and
    /// the bounds of its domain.
    #[test]
    fn a_refused_option_names_itself_on_the_wire() {
        let request = |extra: serde_json::Value| {
            let mut body = json!({ "data": vec![1.0_f64; 20] });
            let (map, extra) = (body.as_object_mut().expect("object"), extra);
            for (k, v) in extra.as_object().expect("object") {
                map.insert(k.clone(), v.clone());
            }
            let input: SpectralResidualInputDto =
                serde_json::from_value(body).expect("valid request");
            spectral_residual_dto(input)
        };

        for (option, body, details) in [
            (
                "threshold",
                json!({ "threshold": 0.0 }),
                vec![
                    ("min", Detail::Null),
                    ("exclusive_min", Detail::Num(0.0)),
                    ("max", Detail::Null),
                    ("got", Detail::Num(0.0)),
                ],
            ),
            (
                "sensitivity",
                json!({ "sensitivity": 100.0 }),
                vec![
                    ("min", Detail::Null),
                    ("exclusive_min", Detail::Num(0.0)),
                    ("max", Detail::Null),
                    ("exclusive_max", Detail::Num(100.0)),
                    ("got", Detail::Num(100.0)),
                ],
            ),
            (
                "sensitivity",
                json!({ "sensitivity": 0.0 }),
                vec![
                    ("min", Detail::Null),
                    ("exclusive_min", Detail::Num(0.0)),
                    ("max", Detail::Null),
                    ("exclusive_max", Detail::Num(100.0)),
                    ("got", Detail::Num(0.0)),
                ],
            ),
            (
                "batch_size",
                json!({ "batch_size": 5 }),
                vec![
                    ("min", Detail::Num(12.0)),
                    ("max", Detail::Null),
                    ("got", Detail::Num(5.0)),
                ],
            ),
            // A negative count is refused by the check a too-small one is.
            (
                "batch_size",
                json!({ "batch_size": -1 }),
                vec![
                    ("min", Detail::Num(12.0)),
                    ("max", Detail::Null),
                    ("got", Detail::Num(-1.0)),
                ],
            ),
            (
                "averaging_window",
                json!({ "averaging_window": 0 }),
                vec![
                    ("min", Detail::Num(1.0)),
                    ("max", Detail::Null),
                    ("got", Detail::Num(0.0)),
                ],
            ),
            (
                "averaging_window",
                json!({ "averaging_window": -1 }),
                vec![
                    ("min", Detail::Num(1.0)),
                    ("max", Detail::Null),
                    ("got", Detail::Num(-1.0)),
                ],
            ),
            (
                "judgement_window",
                json!({ "judgement_window": -1 }),
                vec![
                    ("min", Detail::Num(1.0)),
                    ("max", Detail::Null),
                    ("got", Detail::Num(-1.0)),
                ],
            ),
        ] {
            let e = request(body).expect_err("refused");
            assert_eq!(e.code, code::PARAMETER_OUT_OF_RANGE, "{}", e.message);
            assert_eq!(e.parameter.as_deref(), Some(option), "{}", e.message);
            assert!(e.message.starts_with(option), "{}", e.message);
            assert_eq!(e.details, details, "{}", e.message);
        }

        // Data problems keep their own codes and the position. A NaN has no
        // JSON spelling, so this request is built directly.
        let mut data = vec![1.0_f64; 20];
        data[7] = f64::NAN;
        let e = spectral_residual_dto(SpectralResidualInputDto {
            data,
            averaging_window: None,
            judgement_window: None,
            threshold: None,
            min_zscore: None,
            sensitivity: None,
            batch_size: None,
        })
        .expect_err("refused");
        assert_eq!(
            (e.code, e.index, e.parameter),
            (code::VALUE_NOT_FINITE.name(), Some(7), None)
        );

        let input: SpectralResidualInputDto =
            serde_json::from_value(json!({ "data": vec![1.0_f64; 11] })).expect("valid request");
        let e = spectral_residual_dto(input).expect_err("refused");
        assert_eq!(e.code, code::INSUFFICIENT_DATA);

        // And a valid request is still served, whole counts as given.
        assert!(request(json!({})).is_ok());
        assert!(request(json!({ "averaging_window": 3, "batch_size": 12 })).is_ok());
    }

    #[test]
    fn a_count_is_read_as_a_whole_number_of_either_sign() {
        let read = |v: serde_json::Value| serde_json::from_value::<Whole>(v);
        assert_eq!(read(json!(3)).expect("integer"), Whole(3.0));
        assert_eq!(read(json!(-1)).expect("negative"), Whole(-1.0));
        assert_eq!(read(json!(3.0)).expect("whole float"), Whole(3.0));
        assert_eq!(read(json!(u64::MAX)).expect("large").0, u64::MAX as f64);
        // From text too, as the C ABI reads it.
        let text = serde_json::from_str::<Whole>("-1").expect("negative text");
        assert_eq!(text, Whole(-1.0));
        for refused in [json!(2.5), json!(null), json!("3")] {
            let e = read(refused.clone()).expect_err("not a whole number");
            assert!(
                e.to_string().contains("expected a whole number"),
                "{refused}: {e}"
            );
        }
    }

    fn standard(v: serde_json::Value) -> AttributeStandardDto {
        serde_json::from_value(v).expect("valid options")
    }

    #[test]
    fn phase_two_options_reach_the_chart_and_are_checked() {
        let samples = [(11_u64, 170_u64)];
        let d = laney_p_dto(
            &samples,
            &standard(json!({ "p_bar": 0.0514, "phi": 0.236 })),
        )
        .expect("one sample with a standard");
        assert_eq!((d.p_bar, d.phi), (0.0514, 0.236));
        assert!(d.points[0].out_of_control);
        assert!(d.points[0].z.expect("scaled") > 3.0);

        let d = p_chart_dto(&[(2, 100), (20, 100)], &standard(json!({ "p_bar": 0.05 })))
            .expect("valid");
        assert_eq!(d.p_bar, 0.05);

        // Half a Laney standard is refused, not completed by estimation.
        let (c, _, m) = err(laney_p_dto(&samples, &standard(json!({ "p_bar": 0.05 }))));
        assert_eq!(c, code::INVALID_OPTION, "{m}");
        // A key the chart does not take is refused, not ignored.
        let (c, _, m) = err(p_chart_dto(&samples, &standard(json!({ "phi": 1.0 }))));
        assert_eq!(c, code::MALFORMED_INPUT, "{m}");
        assert!(m.contains("phi"), "{m}");
        // Outside its domain.
        let (c, i, _) = err(p_chart_dto(&samples, &standard(json!({ "p_bar": 1.5 }))));
        assert_eq!((c, i), (code::PARAMETER_OUT_OF_RANGE.name(), None));
        let (c, _, _) = err(laney_p_dto(
            &samples,
            &standard(json!({ "p_bar": 0.05, "phi": -2.0 })),
        ));
        assert_eq!(c, code::PARAMETER_OUT_OF_RANGE);
        // Unknown option keys are refused by the schema.
        assert!(serde_json::from_value::<AttributeStandardDto>(json!({ "pbar": 0.1 })).is_err());
    }
}

/// Wire shapes JSON Schema cannot read off a Rust type (the engine card's
/// schemas; tests only).
#[cfg(all(test, feature = "wasm"))]
pub(crate) mod card_schema {
    use schemars::{Schema, SchemaGenerator};

    fn names<T>(table: &[(&'static str, T)]) -> Schema {
        crate::engine_card::string_enum(table.iter().map(|(n, _)| *n))
    }

    fn or_null(schema: Schema) -> Schema {
        Schema::try_from(serde_json::json!({ "anyOf": [schema.to_value(), { "type": "null" }] }))
            .expect("an object is a schema")
    }

    pub(crate) fn adf_model(_: &mut SchemaGenerator) -> Schema {
        names(&super::hypothesis::ADF_MODELS)
    }

    pub(crate) fn kpss_model(_: &mut SchemaGenerator) -> Schema {
        names(&super::hypothesis::KPSS_MODELS)
    }

    pub(crate) fn kpss_lag_rule_or_null(_: &mut SchemaGenerator) -> Schema {
        or_null(crate::engine_card::string_enum(
            super::hypothesis::KPSS_LAG_RULES.iter().map(|(n, _)| *n),
        ))
    }

    pub(crate) fn capability_model_or_null(_: &mut SchemaGenerator) -> Schema {
        or_null(crate::engine_card::string_enum(
            super::CAPABILITY_MODELS.iter().copied(),
        ))
    }

    pub(crate) fn pelt_cost(_: &mut SchemaGenerator) -> Schema {
        names(&super::PELT_COSTS)
    }

    pub(crate) fn correlation_method(_: &mut SchemaGenerator) -> Schema {
        names(&super::CORRELATION_METHODS)
    }

    /// `correlation_matrix`'s optional `method`.
    pub(crate) fn correlation_method_or_null(_: &mut SchemaGenerator) -> Schema {
        or_null(names(&super::CORRELATION_METHODS))
    }

    pub(crate) fn interval_method(_: &mut SchemaGenerator) -> Schema {
        crate::engine_card::string_enum(super::reliability::INTERVAL_METHODS)
    }

    /// A PELT penalty: `"bic"` or a number.
    pub(crate) fn penalty(_: &mut SchemaGenerator) -> Schema {
        schemars::json_schema!({
            "anyOf": [{ "type": "string", "enum": ["bic"] }, { "type": "number" }]
        })
    }
}
