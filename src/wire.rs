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

    let dto = match input.sigma_within {
        Some(sigma_within) => {
            if !sigma_within.is_finite() || sigma_within <= 0.0 {
                return Err("sigma_within must be a positive, finite number \
                     (R-bar/d2 or S-bar/c4 from the control chart)"
                    .to_string());
            }
            let indices = spec
                .compute(&input.data, sigma_within)
                .ok_or("insufficient or invalid data (need >= 2 finite values)")?;
            CapabilityDto {
                mean: indices.mean,
                sigma_source: "within",
                std_dev_within: Some(indices.std_dev_within),
                std_dev_overall: indices.std_dev_overall,
                cp: indices.cp,
                cpk: indices.cpk,
                cpu: indices.cpu,
                cpl: indices.cpl,
                pp: indices.pp,
                ppk: indices.ppk,
                ppu: indices.ppu,
                ppl: indices.ppl,
                cpm: indices.cpm,
            }
        }
        None => {
            // No short-term sigma: report the long-term indices only. The
            // crate computes both from the same sigma in this mode, so
            // carrying the short-term names through would publish Pp under the
            // name Cp for every input.
            let indices = spec
                .compute_overall(&input.data)
                .ok_or("insufficient or invalid data (need >= 2 finite values)")?;
            CapabilityDto {
                mean: indices.mean,
                sigma_source: "overall",
                std_dev_within: None,
                std_dev_overall: indices.std_dev_overall,
                cp: None,
                cpk: None,
                cpu: None,
                cpl: None,
                pp: indices.pp,
                ppk: indices.ppk,
                ppu: indices.ppu,
                ppl: indices.ppl,
                // Not a short-term index: Cpm is the spread about the target,
                // the same whichever sigma the caller could supply.
                cpm: indices.cpm,
            }
        }
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
pub(crate) struct GageRRResultDto {
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
pub(crate) struct AnovaRowDto {
    pub(crate) source: String,
    pub(crate) df: f64,
    pub(crate) ss: f64,
    pub(crate) ms: f64,
    pub(crate) f_value: Option<f64>,
    pub(crate) p_value: Option<f64>,
}

#[derive(Serialize)]
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
) -> Result<XbarSChartDto, String> {
    use crate::spc::{ControlChart, XBarSChart};

    let n = subgroup_size(&subgroups)?;
    let mut chart = XBarSChart::new(n)
        .map_err(|e| e.to_string())?
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

pub(crate) fn imr_dto(values: Vec<f64>, rules: crate::spc::RuleSet) -> Result<ImrChartDto, String> {
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
