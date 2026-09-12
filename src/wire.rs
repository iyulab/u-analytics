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

use serde::Serialize;

#[derive(Serialize, Debug)]
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
pub(crate) struct ChartPointDto {
    pub(crate) index: usize,
    pub(crate) value: f64,
    pub(crate) violations: Vec<String>,
}

#[derive(Serialize, Debug)]
pub(crate) struct PChartDto {
    pub(crate) p_bar: f64,
    pub(crate) points: Vec<AttributeChartPointDto>,
    pub(crate) in_control: bool,
}

#[derive(Serialize, Debug)]
pub(crate) struct AttributeChartPointDto {
    pub(crate) index: usize,
    pub(crate) value: f64,
    pub(crate) ucl: f64,
    pub(crate) cl: f64,
    pub(crate) lcl: f64,
    pub(crate) out_of_control: bool,
}

#[derive(Serialize, Debug)]
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
pub(crate) fn subgroup_size(subgroups: &[Vec<f64>]) -> Result<usize, String> {
    let n = subgroups
        .first()
        .map(Vec::len)
        .ok_or("at least one subgroup required")?;
    Ok(n)
}

/// Feeds rows to a chart, naming the row a rejection came from. The chart
/// knows why a sample is unusable but not where it sat in the caller's input.
pub(crate) fn add_rows<T>(
    rows: impl IntoIterator<Item = T>,
    label: &str,
    mut add: impl FnMut(T) -> Result<(), crate::spc::ControlChartError>,
) -> Result<(), String> {
    for (i, row) in rows.into_iter().enumerate() {
        add(row).map_err(|e| format!("{label}[{i}]: {e}"))?;
    }
    Ok(())
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
        })
        .collect()
}

/// `[defectives, sample_size]` pairs as `(defectives, sample_size)`, refusing
/// a pair that has no proportion.
pub(crate) fn proportion_samples(raw: &[[u64; 2]]) -> Result<Vec<(u64, u64)>, String> {
    raw.iter()
        .enumerate()
        .map(|(i, &[d, n])| {
            if n == 0 || d > n {
                Err(format!(
                    "samples[{i}]: {d} defectives out of {n} -- each sample needs a size \
                     above 0 and at most that many defectives"
                ))
            } else {
                Ok((d, n))
            }
        })
        .collect()
}

pub(crate) fn xbar_r_dto(
    subgroups: Vec<Vec<f64>>,
    rules: crate::spc::RuleSet,
) -> Result<XbarRChartDto, String> {
    use crate::spc::{ControlChart, XBarRChart};

    let n = subgroup_size(&subgroups)?;
    let mut chart = XBarRChart::new(n)
        .map_err(|e| e.to_string())?
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

pub(crate) fn p_chart_dto(raw: &[[u64; 2]]) -> Result<PChartDto, String> {
    use crate::spc::PChart;

    let samples = proportion_samples(raw)?;
    let mut chart = PChart::new();
    add_rows(&samples, "samples", |&(d, n)| chart.add_sample(d, n))?;
    let p_bar = chart.p_bar().ok_or("no samples provided")?;
    Ok(PChartDto {
        p_bar,
        points: attribute_point_dtos(chart.points()),
        in_control: chart.is_in_control(),
    })
}

pub(crate) fn laney_p_dto(raw: &[[u64; 2]]) -> Result<LaneyPChartDto, String> {
    let samples = proportion_samples(raw)?;
    let chart = crate::spc::laney_p_chart(&samples).ok_or("at least 3 samples are needed")?;
    Ok(LaneyPChartDto {
        p_bar: chart.p_bar,
        phi: chart.phi,
        points: laney_point_dtos(&chart.points),
    })
}
