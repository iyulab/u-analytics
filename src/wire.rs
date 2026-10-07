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
    pub(crate) sigma_hat: Option<f64>,
    pub(crate) xbar_points: Vec<ChartPointDto>,
    pub(crate) r_points: Vec<ChartPointDto>,
    pub(crate) in_control: bool,
}

#[derive(Serialize, Debug)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(feature = "wasm", tsify(missing_as_null))]
pub(crate) struct ChartPointDto {
    pub(crate) index: usize,
    pub(crate) value: f64,
    pub(crate) violations: Vec<String>,
}

#[derive(Serialize, Debug)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(feature = "wasm", tsify(missing_as_null))]
pub(crate) struct PChartDto {
    pub(crate) p_bar: f64,
    pub(crate) points: Vec<AttributeChartPointDto>,
    pub(crate) in_control: bool,
}

#[derive(Serialize, Debug)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
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
            )),
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
            _ => Err(WireError::invalid_input(format!(
                "options: `{center_name}` and `phi` are given together -- a Phase I standard \
                 fixes both, since phi scales the error about that centre"
            ))),
        }
    }
}

#[derive(Serialize, Debug)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(feature = "wasm", tsify(missing_as_null))]
pub(crate) struct LaneyPChartDto {
    pub(crate) p_bar: f64,
    pub(crate) phi: f64,
    pub(crate) points: Vec<AttributeChartPointDto>,
}

pub(crate) fn violation_name(v: crate::spc::ViolationType) -> &'static str {
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

/// Every run-rule name `rules` accepts — the values `violations` reports.
pub(crate) const RULE_NAMES: &[&str] = &[
    "BeyondLimits",
    "NineOneSide",
    "SixTrend",
    "FourteenAlternating",
    "TwoOfThreeBeyond2Sigma",
    "FourOfFiveBeyond1Sigma",
    "FifteenWithin1Sigma",
    "EightBeyond1Sigma",
];

/// Parse an optional `{ rules: [...] }` options object into a rule set.
///
/// Absent, `undefined`, `null` or an object without `rules` all mean "the
/// default set", so a caller that never passes options keeps the behaviour it
/// had. Split out from the binding so it can be exercised without a `JsValue`.
pub(crate) fn rules_from_json(
    options: Option<serde_json::Value>,
) -> Result<crate::spc::RuleSet, WireError> {
    use crate::spc::{RuleSet, ViolationType};

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
    let names = names.as_array().ok_or_else(|| {
        WireError::new(
            code::MALFORMED_INPUT,
            None,
            "rules: expected an array of rule names",
        )
        .about("rules")
    })?;

    let mut set = RuleSet::none();
    for name in names {
        let name = name.as_str().ok_or_else(|| {
            WireError::new(
                code::MALFORMED_INPUT,
                None,
                "rules: expected an array of rule names",
            )
            .about("rules")
        })?;
        let rule = match name {
            "BeyondLimits" => ViolationType::BeyondLimits,
            "NineOneSide" => ViolationType::NineOneSide,
            "SixTrend" => ViolationType::SixTrend,
            "FourteenAlternating" => ViolationType::FourteenAlternating,
            "TwoOfThreeBeyond2Sigma" => ViolationType::TwoOfThreeBeyond2Sigma,
            "FourOfFiveBeyond1Sigma" => ViolationType::FourOfFiveBeyond1Sigma,
            "FifteenWithin1Sigma" => ViolationType::FifteenWithin1Sigma,
            "EightBeyond1Sigma" => ViolationType::EightBeyond1Sigma,
            // The names are the values `violations` reports.
            other => return Err(WireError::unknown_option("rules", other, RULE_NAMES)),
        };
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
            violations: p
                .violations
                .iter()
                .map(|&v| violation_name(v).to_owned())
                .collect(),
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
#[cfg_attr(feature = "wasm", tsify(missing_as_null))]
pub(crate) struct WireError {
    /// Stable, machine-readable reason (see the constants below).
    pub(crate) code: &'static str,
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
    /// The values behind the reason, by name: `min` / `max` (`null` on an
    /// open side) and `got` for a value out of range, `got` and `expected`
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

/// Error codes. Kept as constants so the transports and their tests name the
/// same strings.
pub(crate) mod code {
    /// Any refusal without a more specific code.
    pub(crate) const INVALID_INPUT: &str = "invalid_input";
    /// The input is not the shape the function takes (not an array, a pair
    /// with the wrong arity, a value of the wrong type).
    pub(crate) const MALFORMED_INPUT: &str = "malformed_input";
    /// A count that is not a whole number >= 0.
    pub(crate) const COUNT_NOT_WHOLE: &str = "count_not_whole";
    /// A sample size that is not a whole number >= 1.
    ///
    /// One code for the whole field, not one per way of being wrong: a size of
    /// `0`, `-3`, `10.5` or a NaN are the same mistake to whoever typed the row,
    /// and splitting them made `code` insufficient on its own -- the consumer
    /// had to read `message` to learn whether a `count_not_whole` was about the
    /// defectives or the size beside them.
    pub(crate) const SAMPLE_SIZE_NOT_WHOLE: &str = "sample_size_not_whole";
    /// More defectives than the sample has items.
    pub(crate) const DEFECTIVES_EXCEED_SAMPLE: &str = "defectives_exceed_sample";
    /// Units inspected that are not a positive, finite number.
    pub(crate) const UNITS_NOT_POSITIVE: &str = "units_not_positive";
    /// A subgroup whose length differs from the first subgroup's.
    pub(crate) const SUBGROUP_LENGTH_MISMATCH: &str = "subgroup_length_mismatch";
    /// A subgroup size outside the factor tables.
    pub(crate) const SUBGROUP_SIZE_OUT_OF_RANGE: &str = "subgroup_size_out_of_range";
    /// A NaN or an infinity where a measurement belongs.
    pub(crate) const VALUE_NOT_FINITE: &str = "value_not_finite";
    /// Fewer samples than the chart needs.
    pub(crate) const INSUFFICIENT_DATA: &str = "insufficient_data";
    /// A known (Phase I) parameter outside its domain.
    pub(crate) const STANDARD_OUT_OF_RANGE: &str = "standard_out_of_range";
    /// An option outside its domain. `parameter` names which one.
    pub(crate) const PARAMETER_OUT_OF_RANGE: &str = "parameter_out_of_range";
    /// An argument with nothing in it.
    pub(crate) const EMPTY_INPUT: &str = "empty_input";
    /// A value that must be `> 0` is not (Box-Cox data, failure times).
    pub(crate) const NON_POSITIVE_DATA: &str = "non_positive_data";
    /// A setting refused for a reason a range cannot state (`[lo, hi]` with
    /// `lo >= hi`).
    pub(crate) const INVALID_OPTION: &str = "invalid_option";
    /// Arrays that must have the same length do not.
    pub(crate) const DIMENSION_MISMATCH: &str = "dimension_mismatch";
    /// An option given a name the function does not know. `parameter` names it.
    pub(crate) const UNKNOWN_OPTION: &str = "unknown_option";
    /// Event times that go backwards.
    pub(crate) const EVENTS_UNORDERED: &str = "events_unordered";
    /// An event time after the end of observation.
    pub(crate) const EVENT_AFTER_END: &str = "event_after_end";
}

impl WireError {
    pub(crate) fn new(
        code: &'static str,
        index: Option<usize>,
        message: impl Into<String>,
    ) -> Self {
        WireError {
            code,
            index,
            parameter: None,
            message: message.into(),
            details: Vec::new(),
        }
    }

    /// Adds one of the values behind the reason.
    pub(crate) fn with(mut self, key: &'static str, value: Detail) -> Self {
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

    /// A number outside `[min, max]`; `None` is an open side (sent as `null`).
    pub(crate) fn out_of_range(
        parameter: &'static str,
        min: Option<f64>,
        max: Option<f64>,
        got: f64,
        message: impl Into<String>,
    ) -> Self {
        let side = |b: Option<f64>| b.map_or(Detail::Null, Detail::Num);
        Self::new(code::PARAMETER_OUT_OF_RANGE, None, message)
            .about(parameter)
            .with("min", side(min))
            .with("max", side(max))
            .with("got", Detail::Num(got))
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
        let wire = Self::new(chart_error_code(error), index, message);
        match error {
            // The domain error already names the parameter; the wire record
            // should not make a consumer read it back out of the message.
            crate::spc::ControlChartError::InvalidStandard { parameter } => wire.about(*parameter),
            _ => wire,
        }
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

pub(crate) fn chart_error_code(error: &crate::spc::ControlChartError) -> &'static str {
    use crate::spc::ControlChartError as E;
    match error {
        E::SampleLengthMismatch { .. } => code::SUBGROUP_LENGTH_MISMATCH,
        E::NonFiniteValue => code::VALUE_NOT_FINITE,
        E::DefectivesExceedSampleSize { .. } => code::DEFECTIVES_EXCEED_SAMPLE,
        E::NonPositiveUnits => code::UNITS_NOT_POSITIVE,
        E::SubgroupSizeOutOfRange { .. } => code::SUBGROUP_SIZE_OUT_OF_RANGE,
        E::ZeroSampleSize => code::SAMPLE_SIZE_NOT_WHOLE,
        E::InvalidStandard { .. } => code::STANDARD_OUT_OF_RANGE,
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
    code: &'static str,
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
        .collect()
}

/// The one sample size an NP chart applies to every row. Not an element of an
/// array, so its refusal carries no index.
// Only the WASM binding carries the NP, C and U charts; the FFI has no caller.
#[cfg_attr(not(feature = "wasm"), allow(dead_code))]
pub(crate) fn sample_size_value(value: &serde_json::Value, label: &str) -> Result<u64, WireError> {
    whole_count(value, label, None, SAMPLE_SIZE)
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
        .collect()
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
        .collect()
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
    let x = chart
        .control_limits()
        .ok_or("insufficient data for control limits")?;
    let r = chart
        .r_limits()
        .ok_or("insufficient data for R chart limits")?;
    Ok(XbarRChartDto {
        xbar_cl: x.cl,
        xbar_ucl: x.ucl,
        xbar_lcl: x.lcl,
        r_cl: r.cl,
        r_ucl: r.ucl,
        r_lcl: r.lcl,
        sigma_hat: chart.sigma_hat(),
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
#[cfg_attr(feature = "wasm", tsify(missing_as_null))]
pub(crate) struct UChartDto {
    pub(crate) u_bar: f64,
    pub(crate) points: Vec<AttributeChartPointDto>,
    pub(crate) in_control: bool,
}

#[derive(Serialize, Debug)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(feature = "wasm", tsify(missing_as_null))]
pub(crate) struct LaneyUChartDto {
    pub(crate) u_bar: f64,
    pub(crate) phi: f64,
    pub(crate) points: Vec<AttributeChartPointDto>,
}
#[derive(Serialize, Debug)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(feature = "wasm", tsify(missing_as_null))]
pub(crate) struct GChartDto {
    pub(crate) g_bar: f64,
    pub(crate) points: Vec<GChartPointDto>,
}

#[derive(Serialize, Debug)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
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
#[cfg_attr(feature = "wasm", tsify(missing_as_null))]
pub(crate) struct TChartDto {
    pub(crate) t_bar: f64,
    pub(crate) points: Vec<TChartPointDto>,
}

#[derive(Serialize, Debug)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
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

/// The first value that `allowed` rejects, as `parameter_out_of_range` at its
/// index; `bound` states the domain in the message.
pub(crate) fn each_within(
    values: &[f64],
    param: &'static str,
    bound: &str,
    allowed: impl Fn(f64) -> bool,
) -> Result<(), WireError> {
    match values.iter().position(|&v| !allowed(v)) {
        Some(i) => Err(WireError::new(
            crate::wire::code::PARAMETER_OUT_OF_RANGE,
            Some(i),
            format!("{param}[{i}]: must be {bound}, got {}", values[i]),
        )
        .about(param)),
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
    each_within(&gaps, "gaps", ">= 0", |v| v >= 0.0)?;
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
    each_within(&times, "times", "> 0", |v| v > 0.0)?;
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
    #[cfg_attr(feature = "wasm", tsify(missing_as_null))]
    pub(crate) struct ShapiroWilkDto {
        pub(crate) w: f64,
        pub(crate) p_value: f64,
    }

    /// Mann-Kendall trend test with Sen's slope.
    #[derive(Serialize, Debug)]
    #[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
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
            return Err(WireError::new(
                code::PARAMETER_OUT_OF_RANGE,
                None,
                format!(
                    "data: Shapiro-Wilk covers 3 to 5000 values (Royston 1995), got {}",
                    data.len()
                ),
            )
            .about("data"));
        }
        crate::testing::shapiro_wilk_test(&data)
            .map(|r| ShapiroWilkDto {
                w: r.w,
                p_value: r.p_value,
            })
            .ok_or_else(|| no_variation("shapiro_wilk_test", "data"))
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
        each_within(&observed, "observed", ">= 0", |v| v >= 0.0)?;
        each_within(expected, "expected", "> 0", |v| v > 0.0)?;
        crate::testing::chi_squared_goodness_of_fit(&observed, expected)
            .map(Into::into)
            .ok_or_else(|| WireError::invalid_input("chi_squared_goodness_of_fit: no statistic"))
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
        each_within(&flat, "table", ">= 0", |v| v >= 0.0)?;
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
        each_within(&p, "p_values", "a probability in [0, 1]", |v| {
            (0.0..=1.0).contains(&v)
        })?;
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
}

#[derive(Serialize, Debug)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
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
pub(crate) fn capability_dto(input: CapabilityInputDto) -> Result<CapabilityDto, String> {
    use crate::capability::ProcessCapability;

    let mut spec = ProcessCapability::new(input.usl, input.lsl)
        .map_err(|e| format!("invalid specification limits: {e}"))?;
    if let Some(target) = input.target {
        if !target.is_finite() {
            return Err("target must be finite".to_string());
        }
        spec = spec.with_target(target);
    }

    // The crate decides what a missing within sigma means: `compute_overall`
    // reports the long-term indices only, so the short-term quartet and
    // `std_dev_within` arrive as `None` and are passed through as such.
    let indices = match input.sigma_within {
        Some(sigma_within) => {
            if !sigma_within.is_finite() || sigma_within <= 0.0 {
                return Err("sigma_within must be a positive, finite number \
                     (R-bar/d2 or S-bar/c4 from the control chart)"
                    .to_string());
            }
            spec.compute(&input.data, sigma_within)
        }
        None => spec.compute_overall(&input.data),
    }
    .ok_or("insufficient or invalid data (need >= 2 finite values)")?;

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
    };
    Ok(dto)
}

/// Input for `percentile_capability`.
#[derive(Deserialize)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
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
}

#[derive(Serialize)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(feature = "wasm", tsify(missing_as_null))]
pub(crate) struct PercentileCapabilityDto {
    pub(crate) cp_star: Option<f64>,
    pub(crate) cpk_star: Option<f64>,
    pub(crate) cpu_star: Option<f64>,
    pub(crate) cpl_star: Option<f64>,
    pub(crate) median: f64,
    pub(crate) percentile_lower: f64,
    pub(crate) percentile_upper: f64,
}

#[derive(Deserialize)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
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
#[cfg_attr(feature = "wasm", tsify(missing_as_null))]
#[serde(deny_unknown_fields)]
pub(crate) struct PeltInputDto {
    pub(crate) data: Vec<f64>,
    #[serde(default = "default_cost")]
    #[cfg_attr(feature = "wasm", tsify(optional))]
    #[cfg_attr(feature = "wasm", tsify(type = "\"l2\" | \"normal\""))]
    pub(crate) cost: String,
    #[serde(default = "default_penalty")]
    #[cfg_attr(feature = "wasm", tsify(optional))]
    #[cfg_attr(feature = "wasm", tsify(type = "\"bic\" | number"))]
    pub(crate) penalty: PeltPenaltyDto,
    #[serde(default = "default_min_seg")]
    #[cfg_attr(feature = "wasm", tsify(optional))]
    pub(crate) min_segment_len: usize,
}

#[derive(Deserialize)]
#[serde(untagged)]
pub(crate) enum PeltPenaltyDto {
    Named(String),
    Value(f64),
}

#[derive(Serialize)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
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

pub(crate) fn default_min_seg() -> usize {
    2
}

pub(crate) fn percentile_capability_dto(
    input: PercentileCapabilityInputDto,
) -> Result<PercentileCapabilityDto, String> {
    let result = crate::capability::percentile_capability(&input.data, input.lsl, input.usl)
        .map_err(|e| e.to_string())?;
    Ok(PercentileCapabilityDto {
        cp_star: result.cp_star,
        cpk_star: result.cpk_star,
        cpu_star: result.cpu_star,
        cpl_star: result.cpl_star,
        median: result.median,
        percentile_lower: result.percentile_lower,
        percentile_upper: result.percentile_upper,
    })
}

pub(crate) fn gage_rr_xbar_r_dto(dto: GageRRInputDto) -> Result<GageRRResultDto, String> {
    let input = crate::msa::GageRRInput {
        measurements: dto.measurements,
        tolerance: dto.tolerance,
    };
    let result = crate::msa::gage_rr_xbar_r(&input).map_err(|e| e.to_string())?;
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

pub(crate) fn gage_rr_anova_dto(dto: GageRRInputDto) -> Result<GageRRAnovaResultDto, String> {
    let input = crate::msa::GageRRInput {
        measurements: dto.measurements,
        tolerance: dto.tolerance,
    };
    let result = crate::msa::gage_rr_anova(&input).map_err(|e| e.to_string())?;
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
pub(crate) fn pelt_from(
    cost: &str,
    penalty: &PeltPenaltyDto,
    min_segment_len: usize,
) -> Result<crate::detection::Pelt, WireError> {
    use crate::detection::{CostFunction, Penalty};
    let cost = match cost {
        "l2" => CostFunction::L2,
        "normal" => CostFunction::Normal,
        other => return Err(WireError::unknown_option("cost", other, &["l2", "normal"])),
    };
    let penalty = match *penalty {
        PeltPenaltyDto::Named(ref s) if s == "bic" => Penalty::Bic,
        PeltPenaltyDto::Named(ref s) => {
            return Err(WireError::unknown_option("penalty", s, &["bic"]))
        }
        PeltPenaltyDto::Value(v) if !(v.is_finite() && v > 0.0) => {
            return Err(WireError::out_of_range(
                "penalty",
                Some(0.0),
                None,
                v,
                format!("penalty must be \"bic\" or a finite number > 0, got {v}"),
            ))
        }
        PeltPenaltyDto::Value(v) => Penalty::Custom(v),
    };
    if min_segment_len < 2 {
        return Err(WireError::out_of_range(
            "min_segment_len",
            Some(2.0),
            None,
            min_segment_len as f64,
            format!("min_segment_len must be at least 2, got {min_segment_len}"),
        ));
    }
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

#[derive(Serialize, Debug)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
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
    pub(crate) sigma_hat: Option<f64>,
    pub(crate) xbar_points: Vec<ChartPointDto>,
    pub(crate) s_points: Vec<ChartPointDto>,
    pub(crate) in_control: bool,
}

#[derive(Serialize, Debug)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(feature = "wasm", tsify(missing_as_null))]
pub(crate) struct ImrChartDto {
    pub(crate) i_cl: f64,
    pub(crate) i_ucl: f64,
    pub(crate) i_lcl: f64,
    pub(crate) mr_cl: f64,
    pub(crate) mr_ucl: f64,
    pub(crate) mr_lcl: f64,
    /// Short-term sigma implied by this chart (`MR-bar / d2(2)`).
    pub(crate) sigma_hat: Option<f64>,
    pub(crate) i_points: Vec<ChartPointDto>,
    /// Starts at index 1: the first value has no moving range.
    pub(crate) mr_points: Vec<ChartPointDto>,
    pub(crate) in_control: bool,
}

/// Control limits a caller supplies to `run_rules`.
#[derive(Deserialize)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
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
    let x = chart
        .control_limits()
        .ok_or("insufficient data for control limits")?;
    let s = chart
        .s_limits()
        .ok_or("insufficient data for S chart limits")?;
    Ok(XbarSChartDto {
        xbar_cl: x.cl,
        xbar_ucl: x.ucl,
        xbar_lcl: x.lcl,
        s_cl: s.cl,
        s_ucl: s.ucl,
        s_lcl: s.lcl,
        sigma_hat: chart.sigma_hat(),
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
    let i = chart
        .control_limits()
        .ok_or("at least two values are needed for control limits")?;
    let mr = chart
        .mr_limits()
        .ok_or("at least two values are needed for control limits")?;
    Ok(ImrChartDto {
        i_cl: i.cl,
        i_ucl: i.ucl,
        i_lcl: i.lcl,
        mr_cl: mr.cl,
        mr_ucl: mr.ucl,
        mr_lcl: mr.lcl,
        sigma_hat: chart.sigma_hat(),
        i_points: point_dtos(chart.points()),
        mr_points: point_dtos(chart.mr_points()),
        in_control: chart.is_in_control(),
    })
}

pub(crate) fn run_rules_dto(
    values: Vec<f64>,
    limits: LimitsInputDto,
    rules: crate::spc::RuleSet,
) -> Result<Vec<ChartPointDto>, String> {
    use crate::spc::{ChartPoint, ControlLimits, RunRule};

    let LimitsInputDto { ucl, cl, lcl } = limits;
    // Written so that a NaN fails it too.
    if !(lcl <= cl && cl <= ucl) {
        return Err(format!(
            "limits: need lcl <= cl <= ucl, got lcl={lcl} cl={cl} ucl={ucl}"
        ));
    }
    if let Some(i) = values.iter().position(|x| !x.is_finite()) {
        return Err(format!("values[{i}] is not a finite number"));
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

// ── Seasonality ──────────────────────────────────────────────────────────────

#[derive(Deserialize)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(feature = "wasm", tsify(missing_as_null))]
#[serde(deny_unknown_fields)]
pub(crate) struct SeasonalityInputDto {
    pub(crate) data: Vec<f64>,
}

#[derive(Serialize, Debug)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
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
#[cfg_attr(feature = "wasm", tsify(missing_as_null))]
pub(crate) struct SeasonalityDto {
    /// `null` when no periodicity passed both stages.
    pub(crate) period: Option<usize>,
    pub(crate) candidates: Vec<PeriodCandidateDto>,
    pub(crate) n: usize,
    pub(crate) acf_threshold: f64,
    pub(crate) power_threshold: f64,
}

pub(crate) fn seasonality_dto(input: SeasonalityInputDto) -> Result<SeasonalityDto, String> {
    if let Some(i) = input.data.iter().position(|x| !x.is_finite()) {
        return Err(format!("data[{i}] is not a finite number"));
    }
    let r = crate::seasonality::estimate_period(&input.data).ok_or_else(|| {
        format!(
            "data must have at least {} observations, got {}",
            crate::seasonality::MIN_OBSERVATIONS,
            input.data.len()
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
#[cfg_attr(feature = "wasm", tsify(missing_as_null))]
#[serde(deny_unknown_fields)]
pub(crate) struct SpectralResidualInputDto {
    pub(crate) data: Vec<f64>,
    #[serde(default)]
    #[cfg_attr(feature = "wasm", tsify(optional))]
    #[cfg_attr(feature = "wasm", tsify(type = "number | null"))]
    pub(crate) averaging_window: Option<usize>,
    #[serde(default)]
    #[cfg_attr(feature = "wasm", tsify(optional))]
    #[cfg_attr(feature = "wasm", tsify(type = "number | null"))]
    pub(crate) judgement_window: Option<usize>,
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
    pub(crate) batch_size: Option<usize>,
}

#[derive(Serialize, Debug)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
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
            E::OptionOutOfRange { option, .. } => {
                WireError::new(code::PARAMETER_OUT_OF_RANGE, None, message).about(option)
            }
            E::TooFewObservations { .. } => WireError::insufficient_data(message),
            E::ValueNotFinite { index } => {
                WireError::new(code::VALUE_NOT_FINITE, Some(index), message)
            }
        }
    }
}

pub(crate) fn spectral_residual_dto(
    input: SpectralResidualInputDto,
) -> Result<SpectralResidualDto, WireError> {
    let mut sr = crate::detection::SpectralResidual::new();
    if let Some(q) = input.averaging_window {
        sr = sr.with_averaging_window(q);
    }
    if let Some(z) = input.judgement_window {
        sr = sr.with_judgement_window(z);
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
    if input.batch_size.is_some() {
        sr = sr.with_batch_size(input.batch_size);
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
                Some(0.0),
                None,
                x,
                format!("{param} must be a finite number > 0, got {x}"),
            ))
        }
    }

    // ── Weibull fits ──────────────────────────────────────────────────────

    /// `{ failure_times }` — the C ABI request (WebAssembly takes the array).
    #[cfg(feature = "ffi")]
    #[derive(Deserialize, Debug)]
    #[serde(deny_unknown_fields)]
    pub(crate) struct FailureTimesDto {
        pub(crate) failure_times: Vec<f64>,
    }

    /// Weibull maximum-likelihood fit.
    #[derive(Serialize, Debug)]
    #[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
    #[cfg_attr(feature = "wasm", tsify(missing_as_null))]
    pub(crate) struct WeibullMleDto {
        pub(crate) shape: f64,
        pub(crate) scale: f64,
        pub(crate) log_likelihood: f64,
        pub(crate) iterations: usize,
    }

    /// Weibull median-rank-regression fit.
    #[derive(Serialize, Debug)]
    #[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
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

    pub(crate) fn weibull_mle_dto(times: &[f64]) -> Result<WeibullMleDto, WireError> {
        positive_values("failure_times", times, 2)?;
        let r = crate::weibull::weibull_mle(times).ok_or_else(|| degenerate("weibull_mle"))?;
        Ok(WeibullMleDto {
            shape: r.shape,
            scale: r.scale,
            log_likelihood: r.log_likelihood,
            iterations: r.iterations,
        })
    }

    pub(crate) fn weibull_mrr_dto(times: &[f64]) -> Result<WeibullMrrDto, WireError> {
        positive_values("failure_times", times, 2)?;
        let r = crate::weibull::weibull_mrr(times).ok_or_else(|| degenerate("weibull_mrr"))?;
        Ok(WeibullMrrDto {
            shape: r.shape,
            scale: r.scale,
            r_squared: r.r_squared,
        })
    }

    // ── Weibull reliability ───────────────────────────────────────────────

    /// `{ shape, scale, times?, fractions_failed? }` — a fitted (or known)
    /// Weibull and the points to evaluate it at.
    #[derive(Deserialize, Debug)]
    #[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
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
    }

    /// Reliability metrics of a Weibull, aligned with the input arrays.
    #[derive(Serialize, Debug)]
    #[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
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
        for (i, &f) in input.fractions_failed.iter().enumerate() {
            if !(f > 0.0 && f < 1.0) {
                return Err(WireError::new(
                    code::PARAMETER_OUT_OF_RANGE,
                    Some(i),
                    format!("fractions_failed[{i}]: must be strictly inside (0, 1), got {f}"),
                )
                .about("fractions_failed")
                .with("min", Detail::Num(0.0))
                .with("max", Detail::Num(1.0))
                .with("got", Detail::Num(f)));
            }
        }
        let ra = crate::weibull::ReliabilityAnalysis::new(shape, scale)
            .expect("shape and scale were checked finite and > 0");
        Ok(WeibullReliabilityDto {
            mtbf: ra.mtbf(),
            reliability: input.times.iter().map(|&t| ra.reliability(t)).collect(),
            hazard_rate: input.times.iter().map(|&t| ra.hazard_rate(t)).collect(),
            b_life: input
                .fractions_failed
                .iter()
                .map(|&f| ra.b_life(f).expect("fraction checked inside (0, 1)"))
                .collect(),
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
                Some(0.0),
                Some(1_000_000.0),
                ppm,
                format!("ppm must be finite and strictly inside (0, 1000000), got {ppm}"),
            )
        })
    }

    // ── Box-Cox capability ────────────────────────────────────────────────

    #[derive(Deserialize, Debug)]
    #[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
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
            let e = weibull_mle_dto(&[10.0, -1.0]).unwrap_err();
            assert_eq!(
                (e.code, e.index, e.parameter.as_deref()),
                ("non_positive_data", Some(1), Some("failure_times"))
            );
            let e = weibull_mrr_dto(&[10.0, f64::NAN]).unwrap_err();
            assert_eq!((e.code, e.index), ("value_not_finite", Some(1)));
            let e = weibull_mrr_dto(&[10.0]).unwrap_err();
            assert_eq!(
                e.details,
                vec![("min", Detail::Num(2.0)), ("got", Detail::Num(1.0))]
            );
            let e = weibull_mle_dto(&[5.0, 5.0, 5.0]).unwrap_err();
            assert_eq!(e.code, "invalid_input");
            let ok = weibull_mrr_dto(&[150.0, 200.0, 250.0, 300.0, 350.0, 400.0]).unwrap();
            assert!(ok.shape > 0.0 && ok.r_squared > 0.9);
        }

        #[test]
        fn reliability_is_aligned_with_its_inputs() {
            let r = weibull_reliability_dto(WeibullReliabilityInputDto {
                shape: 2.0,
                scale: 100.0,
                times: vec![0.0, 100.0],
                fractions_failed: vec![0.1, 0.5],
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
            })
            .unwrap_err();
            assert_eq!(
                (e.code, e.index, e.parameter.as_deref()),
                ("parameter_out_of_range", Some(1), Some("fractions_failed"))
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
            assert_eq!(e.details[1], ("max", Detail::Num(1_000_000.0)));
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

/// Trend tests and the power-law fit on event times (`crate::point_process`).
pub(crate) mod point_process {
    use super::*;
    use crate::point_process::{PointProcessError, TrendDirection, TrendTestResult, Truncation};

    /// How observation ended: `{ truncation: "time", end }` — watched until
    /// `end` — or `{ truncation: "failure" }` — stopped at the last event.
    #[derive(Deserialize, Debug, Clone, Copy, PartialEq)]
    #[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
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
        Err(WireError::new(
            code::UNKNOWN_OPTION,
            None,
            format!(
                "observation.truncation: unknown truncation {name:?}; expected one of {}",
                TRUNCATIONS.join(", ")
            ),
        )
        .about("observation.truncation"))
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
    #[cfg_attr(feature = "wasm", tsify(missing_as_null))]
    pub(crate) struct TrendTestDto {
        pub(crate) statistic: f64,
        /// Two-sided; the one-sided p-value in `direction` is half of it.
        pub(crate) p_value: f64,
        #[cfg_attr(
            feature = "wasm",
            tsify(type = "\"increasing\" | \"decreasing\" | \"flat\"")
        )]
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
            PointProcessError::NotPositive { index: Some(i), .. } => {
                WireError::new(code::PARAMETER_OUT_OF_RANGE, Some(i), message).about("times")
            }
            PointProcessError::NotPositive { index: None, .. } => {
                WireError::new(code::PARAMETER_OUT_OF_RANGE, None, message).about("observation.end")
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
    #[test]
    fn pelt_options_are_refused_by_name_with_their_values() {
        use super::{pelt_dto, Detail, PeltInputDto, PeltPenaltyDto};
        let input = |cost: &str, penalty: PeltPenaltyDto, min_segment_len: usize| PeltInputDto {
            data: vec![0.0, 0.0, 5.0, 5.0],
            cost: cost.to_string(),
            penalty,
            min_segment_len,
        };
        let bic = || PeltPenaltyDto::Named("bic".into());

        let e = pelt_dto(input("l1", bic(), 2)).err().unwrap();
        assert_eq!(
            (e.code, e.parameter.as_deref()),
            ("unknown_option", Some("cost"))
        );
        assert_eq!(
            e.details[1],
            ("expected", Detail::List(vec!["l2", "normal"]))
        );

        let e = pelt_dto(input("l2", PeltPenaltyDto::Named("aic".into()), 2))
            .err()
            .unwrap();
        assert_eq!(
            (e.code, e.parameter.as_deref()),
            ("unknown_option", Some("penalty"))
        );
        assert_eq!(e.details[0], ("got", Detail::Text("aic".into())));

        let e = pelt_dto(input("l2", PeltPenaltyDto::Value(-1.0), 2))
            .err()
            .unwrap();
        assert_eq!(
            (e.code, e.parameter.as_deref()),
            ("parameter_out_of_range", Some("penalty"))
        );
        assert_eq!(
            e.details,
            vec![
                ("min", Detail::Num(0.0)),
                ("max", Detail::Null),
                ("got", Detail::Num(-1.0))
            ]
        );

        let e = pelt_dto(input("l2", bic(), 1)).err().unwrap();
        assert_eq!(
            (e.code, e.parameter.as_deref()),
            ("parameter_out_of_range", Some("min_segment_len"))
        );

        let mut empty = input("l2", bic(), 2);
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
            assert_eq!((c, i), (code::COUNT_NOT_WHOLE, Some(2)), "{m}");
            assert!(m.starts_with("defects[2]"), "{m}");
        }
        let (c, i, _) = err(count_pairs(&json!([[1, 10], [2.5, 10]]), "samples"));
        assert_eq!((c, i), (code::COUNT_NOT_WHOLE, Some(1)));
        let (c, i, _) = err(rate_pairs(&json!([[1, 1.0], [0.5, 2.0]]), "samples"));
        assert_eq!((c, i), (code::COUNT_NOT_WHOLE, Some(1)));
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
            assert_eq!((c, i), (code::SAMPLE_SIZE_NOT_WHOLE, Some(1)), "{m}");
            assert!(m.contains("sample size"), "{m}");
            assert!(m.contains("at least 1"), "{m}");
        }
        // Defectives are checked first, and keep the count code.
        let (c, i, m) = err(count_pairs(&json!([[1, 10], [1.5, 0]]), "samples"));
        assert_eq!((c, i), (code::COUNT_NOT_WHOLE, Some(1)), "{m}");
        assert!(m.contains("defectives"), "{m}");
        // An NP chart's single size is not an element, so it carries no position.
        for bad in [json!(0), json!(-3), json!(10.5)] {
            let (c, i, _) = err(sample_size_value(&bad, "sample_size"));
            assert_eq!((c, i), (code::SAMPLE_SIZE_NOT_WHOLE, None));
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
        assert_eq!((c, i), (code::MALFORMED_INPUT, Some(1)));
        let (c, i, _) = err(rate_pairs(&json!([[1, 1.0], 4]), "samples"));
        assert_eq!((c, i), (code::MALFORMED_INPUT, Some(1)));
        let (c, i, _) = err(count_pairs(&json!({ "samples": [] }), "samples"));
        assert_eq!((c, i), (code::MALFORMED_INPUT, None));
        let (c, i, _) = err(rate_pairs(&json!([[1, 1.0], [2, "x"]]), "samples"));
        assert_eq!((c, i), (code::UNITS_NOT_POSITIVE, Some(1)));
    }

    #[test]
    fn every_chart_error_has_its_own_code() {
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
            E::InvalidStandard { parameter: "phi" },
        ];
        let codes: std::collections::HashSet<_> = all.iter().map(chart_error_code).collect();
        assert_eq!(codes.len(), all.len(), "codes must tell the reasons apart");
        assert!(!codes.contains(code::INVALID_INPUT));
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

    /// The report that prompted this: four different option mistakes reached
    /// the consumer as one indistinguishable message, so it re-validated the
    /// options itself. Each now arrives naming its own option.
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

        for (option, body) in [
            ("threshold", json!({ "threshold": 0.0 })),
            ("sensitivity", json!({ "sensitivity": 100.0 })),
            ("sensitivity", json!({ "sensitivity": 0.0 })),
            ("batch_size", json!({ "batch_size": 5 })),
        ] {
            let e = request(body).expect_err("refused");
            assert_eq!(e.code, code::PARAMETER_OUT_OF_RANGE, "{}", e.message);
            assert_eq!(e.parameter.as_deref(), Some(option), "{}", e.message);
            assert!(e.message.starts_with(option), "{}", e.message);
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
            (code::VALUE_NOT_FINITE, Some(7), None)
        );

        let input: SpectralResidualInputDto =
            serde_json::from_value(json!({ "data": vec![1.0_f64; 11] })).expect("valid request");
        let e = spectral_residual_dto(input).expect_err("refused");
        assert_eq!(e.code, code::INSUFFICIENT_DATA);

        // And a valid request is still served.
        assert!(request(json!({})).is_ok());
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
        assert_eq!(c, code::INVALID_INPUT, "{m}");
        // A key the chart does not take is refused, not ignored.
        let (c, _, m) = err(p_chart_dto(&samples, &standard(json!({ "phi": 1.0 }))));
        assert_eq!(c, code::MALFORMED_INPUT, "{m}");
        assert!(m.contains("phi"), "{m}");
        // Outside its domain.
        let (c, i, _) = err(p_chart_dto(&samples, &standard(json!({ "p_bar": 1.5 }))));
        assert_eq!((c, i), (code::STANDARD_OUT_OF_RANGE, None));
        let (c, _, _) = err(laney_p_dto(
            &samples,
            &standard(json!({ "p_bar": 0.05, "phi": -2.0 })),
        ));
        assert_eq!(c, code::STANDARD_OUT_OF_RANGE);
        // Unknown option keys are refused by the schema.
        assert!(serde_json::from_value::<AttributeStandardDto>(json!({ "pbar": 0.1 })).is_err());
    }
}
