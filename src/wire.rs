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

/// Parse an optional `{ rules: [...] }` options object into a rule set.
///
/// Absent, `undefined`, `null` or an object without `rules` all mean "the
/// default set", so a caller that never passes options keeps the behaviour it
/// had. Split out from the binding so it can be exercised without a `JsValue`.
pub(crate) fn rules_from_json(
    options: Option<serde_json::Value>,
) -> Result<crate::spc::RuleSet, String> {
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
    let names = names
        .as_array()
        .ok_or_else(|| "rules: expected an array of rule names".to_string())?;

    let mut set = RuleSet::none();
    for name in names {
        let name = name
            .as_str()
            .ok_or_else(|| "rules: expected an array of rule names".to_string())?;
        let rule = match name {
            "BeyondLimits" => ViolationType::BeyondLimits,
            "NineOneSide" => ViolationType::NineOneSide,
            "SixTrend" => ViolationType::SixTrend,
            "FourteenAlternating" => ViolationType::FourteenAlternating,
            "TwoOfThreeBeyond2Sigma" => ViolationType::TwoOfThreeBeyond2Sigma,
            "FourOfFiveBeyond1Sigma" => ViolationType::FourOfFiveBeyond1Sigma,
            "FifteenWithin1Sigma" => ViolationType::FifteenWithin1Sigma,
            "EightBeyond1Sigma" => ViolationType::EightBeyond1Sigma,
            other => {
                return Err(format!(
                    "rules: unknown rule {other:?} -- the names are the values `violations` reports"
                ))
            }
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
    subgroups
        .first()
        .map(Vec::len)
        .ok_or_else(|| WireError::too_few_samples("subgroups: at least one subgroup required"))
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
    pub(crate) parameter: Option<&'static str>,
    /// Human-readable description.
    pub(crate) message: String,
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
    pub(crate) const TOO_FEW_SAMPLES: &str = "too_few_samples";
    /// A known (Phase I) parameter outside its domain.
    pub(crate) const STANDARD_OUT_OF_RANGE: &str = "standard_out_of_range";
    /// An option outside its domain. `parameter` names which one.
    pub(crate) const OPTION_OUT_OF_RANGE: &str = "option_out_of_range";
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
        }
    }

    /// Names the option a refusal is about, alongside its code.
    pub(crate) fn about(mut self, parameter: &'static str) -> Self {
        self.parameter = Some(parameter);
        self
    }

    /// A refusal without a more specific code.
    pub(crate) fn invalid_input(message: impl Into<String>) -> Self {
        Self::new(code::INVALID_INPUT, None, message)
    }

    /// Too few samples, with the message naming how many are needed.
    pub(crate) fn too_few_samples(message: impl Into<String>) -> Self {
        Self::new(code::TOO_FEW_SAMPLES, None, message)
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
            crate::spc::ControlChartError::InvalidStandard { parameter } => wire.about(parameter),
            _ => wire,
        }
    }

    /// A whole-slice chart's refusal.
    pub(crate) fn chart_input(label: &str, error: &crate::spc::ChartInputError) -> Self {
        use crate::spc::ChartInputError;
        match error {
            ChartInputError::Sample { index, error } => Self::chart(label, Some(*index), error),
            ChartInputError::TooFewSamples { .. } => {
                Self::too_few_samples(format!("{label}: {error}"))
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
    let p_bar = chart
        .p_bar()
        .ok_or_else(|| WireError::too_few_samples("samples: at least 1 sample is needed"))?;
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

/// Input for `process_capability`.
///
/// Every field except `data` is optional, but at least one of `usl`/`lsl` must
/// be present -- a capability index without a specification limit is undefined.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct CapabilityInputDto {
    pub(crate) data: Vec<f64>,
    #[serde(default)]
    pub(crate) usl: Option<f64>,
    #[serde(default)]
    pub(crate) lsl: Option<f64>,
    /// Short-term (within-subgroup) sigma, normally estimated from a control
    /// chart as R-bar/d2 or S-bar/c4. It cannot be recovered from `data`: the
    /// subgroup structure is not in a flat measurement vector.
    #[serde(default)]
    pub(crate) sigma_within: Option<f64>,
    /// Process target for Cpm. Without it `cpm` is `null`.
    #[serde(default)]
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
#[serde(deny_unknown_fields)]
pub(crate) struct PercentileCapabilityInputDto {
    pub(crate) data: Vec<f64>,
    pub(crate) lsl: Option<f64>,
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
#[serde(deny_unknown_fields)]
pub(crate) struct GageRRInputDto {
    pub(crate) measurements: Vec<Vec<Vec<f64>>>,
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
#[serde(deny_unknown_fields)]
pub(crate) struct PeltInputDto {
    pub(crate) data: Vec<f64>,
    #[serde(default = "default_cost")]
    pub(crate) cost: String,
    #[serde(default = "default_penalty")]
    pub(crate) penalty: PeltPenaltyDto,
    #[serde(default = "default_min_seg")]
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

pub(crate) fn pelt_dto(input: PeltInputDto) -> Result<PeltResultDto, String> {
    if input.data.is_empty() {
        return Err("data must not be empty".to_owned());
    }
    let cost = match input.cost.as_str() {
        "l2" => crate::detection::CostFunction::L2,
        "normal" => crate::detection::CostFunction::Normal,
        other => return Err(format!("unknown cost function: {other}")),
    };
    let penalty = match input.penalty {
        PeltPenaltyDto::Named(ref s) if s == "bic" => crate::detection::Penalty::Bic,
        PeltPenaltyDto::Named(ref s) => return Err(format!("unknown penalty: {s}")),
        PeltPenaltyDto::Value(v) => crate::detection::Penalty::Custom(v),
    };
    let pelt = crate::detection::Pelt::with_min_segment_len(cost, penalty, input.min_segment_len)
        .ok_or("invalid parameters (penalty must be positive, min_segment_len >= 2)")?;
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
#[serde(deny_unknown_fields)]
pub(crate) struct SpectralResidualInputDto {
    pub(crate) data: Vec<f64>,
    #[serde(default)]
    pub(crate) averaging_window: Option<usize>,
    #[serde(default)]
    pub(crate) judgement_window: Option<usize>,
    #[serde(default)]
    pub(crate) threshold: Option<f64>,
    #[serde(default)]
    pub(crate) min_zscore: Option<f64>,
    #[serde(default)]
    pub(crate) sensitivity: Option<f64>,
    #[serde(default)]
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
                WireError::new(code::OPTION_OUT_OF_RANGE, None, message).about(option)
            }
            E::TooFewObservations { .. } => WireError::too_few_samples(message),
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

#[cfg(test)]
mod input_error_tests {
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

        let e = WireError::new(code::OPTION_OUT_OF_RANGE, None, "threshold must be > 0")
            .about("threshold");
        assert_eq!(
            serde_json::to_value(&e).expect("serializable"),
            json!({
                "code": "option_out_of_range", "index": null,
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
            assert_eq!(e.code, code::OPTION_OUT_OF_RANGE, "{}", e.message);
            assert_eq!(e.parameter, Some(option), "{}", e.message);
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
        assert_eq!(e.code, code::TOO_FEW_SAMPLES);

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
