//! Measurement System Analysis (MSA).
//!
//! Implements Gage R&R studies using both the X̄-R (Average & Range) method
//! and the ANOVA method, following AIAG MSA 4th Edition.
//!
//! # Overview
//!
//! A Gage R&R study decomposes total measurement variation into:
//! - **Repeatability (EV)**: Equipment variation — same operator, same part, multiple trials
//! - **Reproducibility (AV)**: Appraiser variation — different operators, same part
//! - **Part Variation (PV)**: True part-to-part variation
//!
//! # References
//!
//! - AIAG (2010). *Measurement Systems Analysis*, 4th ed.
//! - Montgomery, D.C. (2019). *Introduction to Statistical Quality Control*, 8th ed., §5.

use u_numflow::special;
use u_numflow::stats;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Input data for a Gage R&R study.
///
/// The `measurements` field is a 3D array indexed as `[part][operator][trial]`.
/// All parts must have the same number of operators, and all operator×part cells
/// must have the same number of trials.
pub struct GageRRInput {
    /// 3D measurement data: `measurements[part][operator][trial]`.
    pub measurements: Vec<Vec<Vec<f64>>>,
    /// Process tolerance (USL − LSL), optional for %Tolerance calculation.
    pub tolerance: Option<f64>,
}

/// Results from a Gage R&R study (X̄-R or ANOVA method).
#[derive(Debug, Clone)]
pub struct GageRRResult {
    /// Equipment Variation (Repeatability).
    pub ev: f64,
    /// Appraiser Variation (Reproducibility).
    pub av: f64,
    /// Gage R&R = √(EV² + AV²).
    pub grr: f64,
    /// Part Variation.
    pub pv: f64,
    /// Total Variation = √(GRR² + PV²).
    pub tv: f64,

    /// %EV = EV / TV × 100.
    pub percent_ev: f64,
    /// %AV = AV / TV × 100.
    pub percent_av: f64,
    /// %GRR = GRR / TV × 100.
    pub percent_grr: f64,
    /// %PV = PV / TV × 100.
    pub percent_pv: f64,
    /// %Tolerance = 6 × GRR / tolerance × 100 (if tolerance provided).
    pub percent_tolerance: Option<f64>,

    /// Number of Distinct Categories = floor(1.41 × PV / GRR), minimum 1.
    pub ndc: u32,
    /// Acceptability status based on %GRR.
    pub status: GrrStatus,
}

/// ANOVA-based Gage R&R result with full ANOVA table and variance components.
#[derive(Debug, Clone)]
pub struct GageRRAnovaResult {
    /// Two-factor crossed ANOVA table.
    pub anova_table: AnovaTable,
    /// Variance components extracted from expected mean squares.
    pub variance_components: VarianceComponents,
    /// Equipment Variation (Repeatability) = √σ²_repeatability.
    pub ev: f64,
    /// Appraiser Variation (Reproducibility) = √σ²_reproducibility.
    pub av: f64,
    /// Gage R&R = √(EV² + AV²).
    pub grr: f64,
    /// Part Variation = √σ²_part.
    pub pv: f64,
    /// Total Variation = √σ²_total.
    pub tv: f64,
    /// %GRR = GRR / TV × 100.
    pub percent_grr: f64,
    /// %Tolerance = 6 × GRR / tolerance × 100 (if tolerance provided).
    pub percent_tolerance: Option<f64>,
    /// Number of Distinct Categories.
    pub ndc: u32,
    /// Acceptability status.
    pub status: GrrStatus,
    /// Whether the Part×Operator interaction is significant (p ≤ 0.25).
    pub interaction_significant: bool,
    /// Whether the interaction term was pooled into error.
    pub interaction_pooled: bool,
}

/// ANOVA table for a two-factor crossed design.
#[derive(Debug, Clone)]
pub struct AnovaTable {
    /// Rows of the ANOVA table.
    pub rows: Vec<AnovaRow>,
}

/// A single row in the ANOVA table.
#[derive(Debug, Clone)]
pub struct AnovaRow {
    /// Source of variation: "Part", "Operator", "Part×Operator", "Repeatability", "Total".
    pub source: String,
    /// Degrees of freedom.
    pub df: f64,
    /// Sum of squares.
    pub ss: f64,
    /// Mean square (SS / DF).
    pub ms: f64,
    /// F statistic (None for Error/Total rows).
    pub f_value: Option<f64>,
    /// p-value from F distribution (None for Error/Total rows).
    pub p_value: Option<f64>,
}

/// Variance components from ANOVA expected mean squares.
#[derive(Debug, Clone)]
pub struct VarianceComponents {
    /// σ²_part — true part-to-part variation.
    pub part: f64,
    /// σ²_operator — operator main effect.
    pub operator: f64,
    /// σ²_interaction — part×operator interaction.
    pub interaction: f64,
    /// σ²_repeatability — within-cell (equipment) variation.
    pub repeatability: f64,
    /// σ²_reproducibility = σ²_operator + σ²_interaction.
    pub reproducibility: f64,
    /// σ²_total = σ²_part + σ²_grr.
    pub total: f64,
}

/// Acceptability status based on %GRR (AIAG guidelines).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GrrStatus {
    /// %GRR ≤ 10% — measurement system is acceptable.
    Acceptable,
    /// 10% < %GRR ≤ 30% — may be acceptable depending on application.
    Marginal,
    /// %GRR > 30% — measurement system needs improvement.
    Unacceptable,
}

// ---------------------------------------------------------------------------
// K-factor constants (AIAG MSA 4th Ed.)
// ---------------------------------------------------------------------------

/// K1 constants: 1/d2* for the number of trials.
/// Index: K1[trials - 2] (trials = 2..=3).
///
/// Values from AIAG MSA 4th Edition, Table III-B-1.
#[allow(clippy::approx_constant)]
const K1: [f64; 2] = [0.8862, 0.5908];

/// K2 constants: 1/d2* for the number of operators.
/// Index: K2[operators - 2] (operators = 2..=3).
///
/// Values from AIAG MSA 4th Edition, Table III-B-1.
#[allow(clippy::approx_constant)]
const K2: [f64; 2] = [0.7071, 0.5231];

/// K3 constants: 1/d2* for the number of parts.
/// Index: K3[parts - 2] (parts = 2..=10).
///
/// Values from AIAG MSA 4th Edition, Table III-B-1.
#[allow(clippy::approx_constant)]
const K3: [f64; 9] = [
    0.7071, 0.5231, 0.4467, 0.4030, 0.3742, 0.3534, 0.3375, 0.3249, 0.3146,
];

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Determine GRR status from %GRR.
fn grr_status(percent_grr: f64) -> GrrStatus {
    if percent_grr <= 10.0 {
        GrrStatus::Acceptable
    } else if percent_grr <= 30.0 {
        GrrStatus::Marginal
    } else {
        GrrStatus::Unacceptable
    }
}

/// Compute NDC = floor(1.41 × PV / GRR), minimum 1.
fn compute_ndc(pv: f64, grr: f64) -> u32 {
    if grr < 1e-300 {
        return 1;
    }
    let ndc = (1.41 * pv / grr).floor() as i64;
    ndc.max(1) as u32
}

/// Validate the 3D measurement array. Returns `(n_parts, n_operators, n_trials)`
/// or an error message.
fn validate_measurements(
    measurements: &[Vec<Vec<f64>>],
) -> Result<(usize, usize, usize), &'static str> {
    let n_parts = measurements.len();
    if n_parts < 2 {
        return Err("at least 2 parts are required");
    }

    let n_operators = measurements[0].len();
    if n_operators < 2 {
        return Err("at least 2 operators are required");
    }

    let n_trials = measurements[0][0].len();
    if n_trials < 2 {
        return Err("at least 2 trials are required");
    }

    for part in measurements {
        if part.len() != n_operators {
            return Err("all parts must have the same number of operators");
        }
        for trials in part {
            if trials.len() != n_trials {
                return Err("all operator×part cells must have the same number of trials");
            }
            for &v in trials {
                if !v.is_finite() {
                    return Err("all measurements must be finite");
                }
            }
        }
    }

    Ok((n_parts, n_operators, n_trials))
}

// ---------------------------------------------------------------------------
// X̄-R Method
// ---------------------------------------------------------------------------

/// Gage R&R using the X̄-R (Average & Range) method (AIAG MSA 4th Edition).
///
/// # Algorithm
///
/// 1. For each operator×part cell, compute the range across trials.
/// 2. R̄ = grand mean of all ranges.
/// 3. For each operator, compute part averages → operator grand averages → X̄_diff.
/// 4. EV = R̄ × K1(trials), AV = √((X̄_diff × K2)² − EV²/(n×r)), GRR = √(EV² + AV²).
/// 5. Part means → Rp → PV = Rp × K3(parts), TV = √(GRR² + PV²).
/// 6. NDC = floor(1.41 × PV / GRR).
///
/// # Arguments
///
/// * `input` — Measurement data and optional tolerance.
///
/// # Returns
///
/// `Err` if input dimensions are invalid or out of supported range
/// (parts 2..=10, operators 2..=3, trials 2..=3).
///
/// # References
///
/// AIAG (2010). *Measurement Systems Analysis*, 4th ed., Chapter III.
pub fn gage_rr_xbar_r(input: &GageRRInput) -> Result<GageRRResult, &'static str> {
    let (n_parts, n_operators, n_trials) = validate_measurements(&input.measurements)?;

    // Validate supported ranges for K-factor tables
    if !(2..=3).contains(&n_trials) {
        return Err("X̄-R method supports 2 or 3 trials");
    }
    if !(2..=3).contains(&n_operators) {
        return Err("X̄-R method supports 2 or 3 operators");
    }
    if !(2..=10).contains(&n_parts) {
        return Err("X̄-R method supports 2 to 10 parts");
    }

    // Step 1: Compute range for each operator×part cell
    let mut ranges: Vec<f64> = Vec::with_capacity(n_parts * n_operators);
    for part in &input.measurements {
        for trials in part {
            let min = trials.iter().copied().fold(f64::INFINITY, f64::min);
            let max = trials.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            ranges.push(max - min);
        }
    }

    // Step 2: R̄ = grand mean of all ranges
    let r_bar = stats::mean(&ranges).expect("ranges is non-empty");

    // Step 3: Operator averages and X̄_diff
    // Compute the average measurement for each operator across all parts and trials
    let mut operator_avgs: Vec<f64> = Vec::with_capacity(n_operators);
    for op in 0..n_operators {
        let mut sum = 0.0;
        let mut count = 0usize;
        for part in &input.measurements {
            for &v in &part[op] {
                sum += v;
                count += 1;
            }
        }
        operator_avgs.push(sum / count as f64);
    }

    let x_diff = operator_avgs
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max)
        - operator_avgs.iter().copied().fold(f64::INFINITY, f64::min);

    // Step 4: EV and AV
    let k1 = K1[n_trials - 2];
    let k2 = K2[n_operators - 2];

    let ev = r_bar * k1;

    let av_squared = (x_diff * k2).powi(2) - ev.powi(2) / (n_parts * n_trials) as f64;
    let av = if av_squared > 0.0 {
        av_squared.sqrt()
    } else {
        0.0
    };

    // Step 5: GRR
    let grr = (ev.powi(2) + av.powi(2)).sqrt();

    // Step 6: Part means across all operators/trials → PV
    let mut part_means: Vec<f64> = Vec::with_capacity(n_parts);
    for part in &input.measurements {
        let mut sum = 0.0;
        let mut count = 0usize;
        for trials in part {
            for &v in trials {
                sum += v;
                count += 1;
            }
        }
        part_means.push(sum / count as f64);
    }

    let rp = part_means.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        - part_means.iter().copied().fold(f64::INFINITY, f64::min);

    let k3 = K3[n_parts - 2];
    let pv = rp * k3;

    // Step 7: TV
    let tv = (grr.powi(2) + pv.powi(2)).sqrt();

    // Percentages
    let (percent_ev, percent_av, percent_grr, percent_pv) = if tv > 1e-300 {
        (
            ev / tv * 100.0,
            av / tv * 100.0,
            grr / tv * 100.0,
            pv / tv * 100.0,
        )
    } else {
        (0.0, 0.0, 0.0, 0.0)
    };

    let percent_tolerance = input.tolerance.and_then(|tol| {
        if tol > 1e-300 {
            Some(grr / tol * 600.0)
        } else {
            None
        }
    });

    let ndc = compute_ndc(pv, grr);
    let status = grr_status(percent_grr);

    Ok(GageRRResult {
        ev,
        av,
        grr,
        pv,
        tv,
        percent_ev,
        percent_av,
        percent_grr,
        percent_pv,
        percent_tolerance,
        ndc,
        status,
    })
}

// ---------------------------------------------------------------------------
// ANOVA Method
// ---------------------------------------------------------------------------

/// Gage R&R using the two-factor crossed ANOVA method (AIAG MSA 4th Edition).
///
/// # Algorithm
///
/// Performs a Part × Operator crossed ANOVA with replications (trials).
/// Computes SS for Part, Operator, Interaction, and Error (Repeatability).
/// Extracts variance components from expected mean squares.
/// If the interaction p-value > 0.25, pools the interaction into error.
///
/// # Arguments
///
/// * `input` — Measurement data and optional tolerance.
///
/// # Returns
///
/// `Err` if input dimensions are invalid (need ≥ 2 parts, ≥ 2 operators, ≥ 2 trials).
///
/// # References
///
/// - AIAG (2010). *Measurement Systems Analysis*, 4th ed., Chapter III, Section D.
/// - Montgomery (2019). *Introduction to Statistical Quality Control*, 8th ed., §5.4.
pub fn gage_rr_anova(input: &GageRRInput) -> Result<GageRRAnovaResult, &'static str> {
    let (p, o, r) = validate_measurements(&input.measurements)?;
    let n_total = p * o * r;

    // Compute grand mean
    let mut grand_sum = 0.0;
    for part in &input.measurements {
        for trials in part {
            for &v in trials {
                grand_sum += v;
            }
        }
    }
    let grand_mean = grand_sum / n_total as f64;

    // Part means (across all operators and trials)
    let mut part_means: Vec<f64> = Vec::with_capacity(p);
    for part in &input.measurements {
        let mut sum = 0.0;
        for trials in part {
            for &v in trials {
                sum += v;
            }
        }
        part_means.push(sum / (o * r) as f64);
    }

    // Operator means (across all parts and trials)
    let mut operator_means: Vec<f64> = Vec::with_capacity(o);
    for op in 0..o {
        let mut sum = 0.0;
        for part in &input.measurements {
            for &v in &part[op] {
                sum += v;
            }
        }
        operator_means.push(sum / (p * r) as f64);
    }

    // Cell means (part × operator, averaged over trials)
    let mut cell_means: Vec<Vec<f64>> = Vec::with_capacity(p);
    for part in &input.measurements {
        let mut row: Vec<f64> = Vec::with_capacity(o);
        for trials in part {
            let cell_sum: f64 = trials.iter().sum();
            row.push(cell_sum / r as f64);
        }
        cell_means.push(row);
    }

    // SS_Part = o * r * Σ(part_mean - grand_mean)²
    let ss_part: f64 = part_means
        .iter()
        .map(|&pm| (pm - grand_mean).powi(2))
        .sum::<f64>()
        * (o * r) as f64;

    // SS_Operator = p * r * Σ(operator_mean - grand_mean)²
    let ss_operator: f64 = operator_means
        .iter()
        .map(|&om| (om - grand_mean).powi(2))
        .sum::<f64>()
        * (p * r) as f64;

    // SS_Interaction = r * Σ_ij (cell_mean_ij - part_mean_i - operator_mean_j + grand_mean)²
    let mut ss_interaction = 0.0;
    for (i, row) in cell_means.iter().enumerate() {
        for (j, &cm) in row.iter().enumerate() {
            let residual = cm - part_means[i] - operator_means[j] + grand_mean;
            ss_interaction += residual.powi(2);
        }
    }
    ss_interaction *= r as f64;

    // SS_Total = Σ(x_ijk - grand_mean)²
    let mut ss_total = 0.0;
    for part in &input.measurements {
        for trials in part {
            for &v in trials {
                ss_total += (v - grand_mean).powi(2);
            }
        }
    }

    // SS_Error = SS_Total - SS_Part - SS_Operator - SS_Interaction
    let ss_error = ss_total - ss_part - ss_operator - ss_interaction;

    // Degrees of freedom
    let df_part = (p - 1) as f64;
    let df_operator = (o - 1) as f64;
    let df_interaction = ((p - 1) * (o - 1)) as f64;
    let df_error = (p * o * (r - 1)) as f64;
    let df_total = (n_total - 1) as f64;

    // Mean squares
    let ms_part = ss_part / df_part;
    let ms_operator = ss_operator / df_operator;
    let ms_interaction = if df_interaction > 0.0 {
        ss_interaction / df_interaction
    } else {
        0.0
    };
    let ms_error = if df_error > 0.0 {
        ss_error / df_error
    } else {
        0.0
    };

    // A mean-square denominator is numerically *degenerate* — indistinguishable
    // from zero variance — when it collapses to floating-point noise relative to
    // the total variation (e.g. every trial within each cell identical, so the
    // within-cell repeatability SS is 0 up to rounding: ms ≈ ±1e-17). A *relative*
    // floor, with a small absolute backstop for the all-identical case where the
    // total MS is itself ~0, keeps the boundary sign-independent: a repeatability
    // MS of +2e-17 and one that dips to −4e-18 under rounding are BOTH reported as
    // "not computable" (F/p = None), rather than flipping between a spurious ~1e12
    // finite F ("very significant") and None depending on the noise sign.
    let ms_total = ss_total / df_total;
    let degenerate_floor = (1e-9 * ms_total.abs()).max(1e-12);
    let is_valid_denom = |ms: f64| ms > degenerate_floor;

    // F statistics and p-values
    let (f_interaction, p_interaction) = if is_valid_denom(ms_error) && df_interaction > 0.0 {
        let f_val = ms_interaction / ms_error;
        let p_val = 1.0 - special::f_distribution_cdf(f_val, df_interaction, df_error);
        (Some(f_val), Some(p_val))
    } else {
        (None, None)
    };

    // Determine if interaction should be pooled (p > 0.25)
    let interaction_significant = p_interaction.is_some_and(|p| p <= 0.25);
    let interaction_pooled = !interaction_significant;

    // Pooled error term (AIAG "without interaction" model): interaction SS/df is
    // folded into the error term, yielding a single pooled MS that serves as BOTH
    // the common F-test denominator AND the repeatability variance component.
    // Computed once here so every consumer (F-tests, ANOVA table, variance
    // components) uses the identical value.
    let pooled_ss = ss_error + ss_interaction;
    let pooled_df = df_error + df_interaction;
    let ms_pooled = if pooled_df > 0.0 {
        pooled_ss / pooled_df
    } else {
        ms_error
    };

    // Determine the denominator for Part and Operator F-tests
    let (denom_ms, denom_df) = if interaction_pooled {
        (ms_pooled, pooled_df)
    } else {
        (ms_interaction, df_interaction)
    };

    let (f_part, p_part) = if is_valid_denom(denom_ms) {
        let f_val = ms_part / denom_ms;
        let p_val = 1.0 - special::f_distribution_cdf(f_val, df_part, denom_df);
        (Some(f_val), Some(p_val))
    } else {
        (None, None)
    };

    let (f_operator, p_operator) = if is_valid_denom(denom_ms) {
        let f_val = ms_operator / denom_ms;
        let p_val = 1.0 - special::f_distribution_cdf(f_val, df_operator, denom_df);
        (Some(f_val), Some(p_val))
    } else {
        (None, None)
    };

    // Build ANOVA table. The error/repeatability row must reflect the SAME model
    // the F-tests and variance components use. When the interaction is pooled we
    // report the pooled error term (df = df_error + df_interaction) and drop the
    // Part×Operator row (it has been folded into error), so that (a) the row df's
    // sum to df_total and (b) the returned f_value for Part/Operator is
    // reproducible from the table's own MS values (f_part = ms_part / ms_pooled).
    // Previously the flag said "pooled" while the table still showed the unpooled
    // Error row, making the reported F un-reproducible from the displayed numbers.
    let mut rows = vec![
        AnovaRow {
            source: "Part".to_owned(),
            df: df_part,
            ss: ss_part,
            ms: ms_part,
            f_value: f_part,
            p_value: p_part,
        },
        AnovaRow {
            source: "Operator".to_owned(),
            df: df_operator,
            ss: ss_operator,
            ms: ms_operator,
            f_value: f_operator,
            p_value: p_operator,
        },
    ];
    if !interaction_pooled {
        rows.push(AnovaRow {
            source: "Part×Operator".to_owned(),
            df: df_interaction,
            ss: ss_interaction,
            ms: ms_interaction,
            f_value: f_interaction,
            p_value: p_interaction,
        });
    }
    rows.push(AnovaRow {
        source: "Repeatability".to_owned(),
        df: if interaction_pooled { pooled_df } else { df_error },
        ss: if interaction_pooled { pooled_ss } else { ss_error },
        ms: if interaction_pooled { ms_pooled } else { ms_error },
        f_value: None,
        p_value: None,
    });
    rows.push(AnovaRow {
        source: "Total".to_owned(),
        df: df_total,
        ss: ss_total,
        ms: ss_total / df_total,
        f_value: None,
        p_value: None,
    });
    let anova_table = AnovaTable { rows };

    // Variance components from expected mean squares. When the interaction is
    // pooled, EVERY component — repeatability included — is derived from the
    // single pooled error MS. Previously repeatability alone leaked the un-pooled
    // raw MS_error while operator/part used the pooled MS, mixing two models in
    // one result and inflating GRR / %GRR. `.max(0.0)` guards against a negative
    // std-dev (NaN) when rounding pushes the (already ~0) MS slightly below zero.
    let sigma2_repeatability = (if interaction_pooled { ms_pooled } else { ms_error }).max(0.0);

    let sigma2_interaction = if interaction_pooled {
        0.0
    } else {
        let val = (ms_interaction - ms_error) / r as f64;
        val.max(0.0)
    };

    let sigma2_operator = if interaction_pooled {
        let val = (ms_operator - ms_pooled) / (p * r) as f64;
        val.max(0.0)
    } else {
        let val = (ms_operator - ms_interaction) / (p * r) as f64;
        val.max(0.0)
    };

    let sigma2_part = if interaction_pooled {
        let val = (ms_part - ms_pooled) / (o * r) as f64;
        val.max(0.0)
    } else {
        let val = (ms_part - ms_interaction) / (o * r) as f64;
        val.max(0.0)
    };

    let sigma2_reproducibility = sigma2_operator + sigma2_interaction;
    let sigma2_grr = sigma2_repeatability + sigma2_reproducibility;
    let sigma2_total = sigma2_part + sigma2_grr;

    let variance_components = VarianceComponents {
        part: sigma2_part,
        operator: sigma2_operator,
        interaction: sigma2_interaction,
        repeatability: sigma2_repeatability,
        reproducibility: sigma2_reproducibility,
        total: sigma2_total,
    };

    // Convert variance components to standard deviations (study variation)
    let ev = sigma2_repeatability.sqrt();
    let av = sigma2_reproducibility.sqrt();
    let grr = sigma2_grr.sqrt();
    let pv = sigma2_part.sqrt();
    let tv = sigma2_total.sqrt();

    let percent_grr = if tv > 1e-300 { grr / tv * 100.0 } else { 0.0 };

    let percent_tolerance = input.tolerance.and_then(|tol| {
        if tol > 1e-300 {
            Some(grr / tol * 600.0)
        } else {
            None
        }
    });

    let ndc = compute_ndc(pv, grr);
    let status = grr_status(percent_grr);

    Ok(GageRRAnovaResult {
        anova_table,
        variance_components,
        ev,
        av,
        grr,
        pv,
        tv,
        percent_grr,
        percent_tolerance,
        ndc,
        status,
        interaction_significant,
        interaction_pooled,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Generate a simple balanced dataset for testing.
    /// 3 operators, 10 parts, 3 trials.
    /// Based on AIAG MSA 4th Edition reference data.
    fn aiag_reference_data() -> Vec<Vec<Vec<f64>>> {
        // measurements[part][operator][trial]
        vec![
            // Part 1
            vec![
                vec![0.29, 0.41, 0.64],  // Operator A
                vec![0.08, 0.25, 0.07],  // Operator B
                vec![0.04, -0.11, 0.75], // Operator C
            ],
            // Part 2
            vec![
                vec![-0.56, -0.68, -0.58],
                vec![-0.47, -1.22, -0.68],
                vec![-0.49, -0.56, -0.49],
            ],
            // Part 3
            vec![
                vec![1.34, 1.17, 1.27],
                vec![1.19, 0.94, 1.34],
                vec![1.02, 0.82, 0.90],
            ],
            // Part 4
            vec![
                vec![0.47, 0.50, 0.64],
                vec![0.01, 0.14, 0.43],
                vec![0.12, 0.22, 0.31],
            ],
            // Part 5
            vec![
                vec![-0.80, -0.92, -0.84],
                vec![-0.56, -1.20, -1.28],
                vec![-0.44, -0.21, -0.17],
            ],
            // Part 6
            vec![
                vec![0.02, 0.16, -0.10],
                vec![0.01, -0.10, 0.07],
                vec![-0.14, -0.46, 0.18],
            ],
            // Part 7
            vec![
                vec![0.59, 0.75, 0.66],
                vec![0.55, 0.36, 0.51],
                vec![0.47, 0.63, 0.34],
            ],
            // Part 8
            vec![
                vec![-0.31, -0.20, 0.17],
                vec![0.02, -0.09, 0.12],
                vec![-0.24, 0.04, -0.19],
            ],
            // Part 9
            vec![
                vec![2.26, 1.99, 2.01],
                vec![1.80, 2.12, 2.19],
                vec![1.80, 1.71, 2.29],
            ],
            // Part 10
            vec![
                vec![-1.36, -1.14, -1.30],
                vec![-1.34, -1.11, -1.42],
                vec![-1.13, -1.13, -0.96],
            ],
        ]
    }

    // -----------------------------------------------------------------------
    // X̄-R method tests
    // -----------------------------------------------------------------------

    #[test]
    fn xbar_r_basic_computation() {
        let data = aiag_reference_data();
        let input = GageRRInput {
            measurements: data,
            tolerance: Some(4.0),
        };
        let result = gage_rr_xbar_r(&input).expect("should compute");

        // EV, AV, GRR should be positive
        assert!(result.ev > 0.0, "EV should be positive: {}", result.ev);
        assert!(result.grr > 0.0, "GRR should be positive: {}", result.grr);
        assert!(result.pv > 0.0, "PV should be positive: {}", result.pv);
        assert!(result.tv > 0.0, "TV should be positive: {}", result.tv);

        // GRR = sqrt(EV² + AV²)
        let expected_grr = (result.ev.powi(2) + result.av.powi(2)).sqrt();
        assert!(
            (result.grr - expected_grr).abs() < 1e-10,
            "GRR identity failed: {} vs {}",
            result.grr,
            expected_grr
        );

        // TV = sqrt(GRR² + PV²)
        let expected_tv = (result.grr.powi(2) + result.pv.powi(2)).sqrt();
        assert!(
            (result.tv - expected_tv).abs() < 1e-10,
            "TV identity failed: {} vs {}",
            result.tv,
            expected_tv
        );

        // Percentages should sum close to 100% (via Pythagorean: %EV² + %AV² + %PV² ≈ 10000)
        // Actually: %GRR² + %PV² = 10000 since TV is the hypotenuse
        let pct_check = result.percent_grr.powi(2) + result.percent_pv.powi(2);
        assert!(
            (pct_check - 10000.0).abs() < 1.0,
            "percentage identity: {} should be ~10000",
            pct_check
        );

        // %Tolerance should be present when tolerance is provided
        assert!(result.percent_tolerance.is_some());

        // NDC should be at least 1
        assert!(result.ndc >= 1);
    }

    #[test]
    fn xbar_r_ndc_minimum_one() {
        // Create data where GRR >> PV (bad measurement system)
        let data = vec![
            vec![vec![1.0, 5.0], vec![0.0, 6.0]],
            vec![vec![1.5, 4.5], vec![0.5, 5.5]],
        ];
        let input = GageRRInput {
            measurements: data,
            tolerance: None,
        };
        let result = gage_rr_xbar_r(&input).expect("should compute");
        assert!(result.ndc >= 1, "NDC should be at least 1");
    }

    #[test]
    fn xbar_r_status_classification() {
        let data = aiag_reference_data();
        let input = GageRRInput {
            measurements: data,
            tolerance: None,
        };
        let result = gage_rr_xbar_r(&input).expect("should compute");

        // The status should match the percent_grr
        match result.status {
            GrrStatus::Acceptable => assert!(result.percent_grr <= 10.0),
            GrrStatus::Marginal => {
                assert!(result.percent_grr > 10.0 && result.percent_grr <= 30.0)
            }
            GrrStatus::Unacceptable => assert!(result.percent_grr > 30.0),
        }
    }

    #[test]
    fn xbar_r_rejects_invalid_dimensions() {
        // Only 1 part
        let data = vec![vec![vec![1.0, 2.0], vec![1.0, 2.0]]];
        let input = GageRRInput {
            measurements: data,
            tolerance: None,
        };
        assert!(gage_rr_xbar_r(&input).is_err());

        // Only 1 operator
        let data = vec![vec![vec![1.0, 2.0]], vec![vec![3.0, 4.0]]];
        let input = GageRRInput {
            measurements: data,
            tolerance: None,
        };
        assert!(gage_rr_xbar_r(&input).is_err());

        // Only 1 trial
        let data = vec![vec![vec![1.0], vec![2.0]], vec![vec![3.0], vec![4.0]]];
        let input = GageRRInput {
            measurements: data,
            tolerance: None,
        };
        assert!(gage_rr_xbar_r(&input).is_err());
    }

    #[test]
    fn xbar_r_rejects_non_finite() {
        let data = vec![
            vec![vec![1.0, f64::NAN], vec![1.0, 2.0]],
            vec![vec![1.0, 2.0], vec![3.0, 4.0]],
        ];
        let input = GageRRInput {
            measurements: data,
            tolerance: None,
        };
        assert!(gage_rr_xbar_r(&input).is_err());
    }

    #[test]
    fn xbar_r_two_operators_two_trials() {
        // Minimal case: 2 parts, 2 operators, 2 trials
        let data = vec![
            vec![vec![10.0, 10.2], vec![10.1, 10.3]],
            vec![vec![20.0, 20.1], vec![19.9, 20.2]],
        ];
        let input = GageRRInput {
            measurements: data,
            tolerance: Some(2.0),
        };
        let result = gage_rr_xbar_r(&input).expect("should compute");
        assert!(result.ev > 0.0);
        assert!(result.pv > 0.0);
        assert!(result.percent_tolerance.is_some());
    }

    // -----------------------------------------------------------------------
    // ANOVA method tests
    // -----------------------------------------------------------------------

    #[test]
    fn anova_basic_computation() {
        let data = aiag_reference_data();
        let input = GageRRInput {
            measurements: data,
            tolerance: Some(4.0),
        };
        let result = gage_rr_anova(&input).expect("should compute");

        // Variance components should be non-negative
        assert!(
            result.variance_components.repeatability >= 0.0,
            "σ²_repeatability should be non-negative"
        );
        assert!(
            result.variance_components.part >= 0.0,
            "σ²_part should be non-negative"
        );
        assert!(
            result.variance_components.operator >= 0.0,
            "σ²_operator should be non-negative"
        );
        assert!(
            result.variance_components.interaction >= 0.0,
            "σ²_interaction should be non-negative"
        );

        // σ²_reproducibility = σ²_operator + σ²_interaction
        let expected_repro =
            result.variance_components.operator + result.variance_components.interaction;
        assert!(
            (result.variance_components.reproducibility - expected_repro).abs() < 1e-10,
            "σ²_reproducibility identity failed"
        );

        // σ²_total = σ²_part + σ²_repeatability + σ²_reproducibility
        let expected_total = result.variance_components.part
            + result.variance_components.repeatability
            + result.variance_components.reproducibility;
        assert!(
            (result.variance_components.total - expected_total).abs() < 1e-10,
            "σ²_total identity failed"
        );

        // ANOVA table should have 5 rows
        assert_eq!(result.anova_table.rows.len(), 5);

        // Check source names
        assert_eq!(result.anova_table.rows[0].source, "Part");
        assert_eq!(result.anova_table.rows[1].source, "Operator");
        assert_eq!(result.anova_table.rows[2].source, "Part×Operator");
        assert_eq!(result.anova_table.rows[3].source, "Repeatability");
        assert_eq!(result.anova_table.rows[4].source, "Total");

        // EV, GRR, PV, TV should be consistent with variance components
        assert!(
            (result.ev - result.variance_components.repeatability.sqrt()).abs() < 1e-10,
            "EV should be sqrt(σ²_repeatability)"
        );
        assert!(
            (result.pv - result.variance_components.part.sqrt()).abs() < 1e-10,
            "PV should be sqrt(σ²_part)"
        );
    }

    #[test]
    fn anova_ss_decomposition() {
        let data = aiag_reference_data();
        let input = GageRRInput {
            measurements: data,
            tolerance: None,
        };
        let result = gage_rr_anova(&input).expect("should compute");

        // SS_Part + SS_Operator + SS_Interaction + SS_Error = SS_Total
        let rows = &result.anova_table.rows;
        let ss_sum = rows[0].ss + rows[1].ss + rows[2].ss + rows[3].ss;
        let ss_total = rows[4].ss;
        assert!(
            (ss_sum - ss_total).abs() < 1e-8,
            "SS decomposition: {} + {} + {} + {} = {} vs total {}",
            rows[0].ss,
            rows[1].ss,
            rows[2].ss,
            rows[3].ss,
            ss_sum,
            ss_total
        );

        // DF decomposition
        let df_sum = rows[0].df + rows[1].df + rows[2].df + rows[3].df;
        let df_total = rows[4].df;
        assert!(
            (df_sum - df_total).abs() < 1e-10,
            "DF decomposition failed: {} vs {}",
            df_sum,
            df_total
        );
    }

    #[test]
    fn anova_interaction_pooling() {
        // Create data with negligible interaction (operators measure similarly)
        let data = vec![
            vec![
                vec![10.0, 10.1, 10.0],
                vec![10.0, 10.0, 10.1],
                vec![10.1, 10.0, 10.0],
            ],
            vec![
                vec![20.0, 20.1, 20.0],
                vec![20.0, 20.0, 20.1],
                vec![20.1, 20.0, 20.0],
            ],
            vec![
                vec![15.0, 15.1, 15.0],
                vec![15.0, 15.0, 15.1],
                vec![15.1, 15.0, 15.0],
            ],
        ];
        let input = GageRRInput {
            measurements: data,
            tolerance: None,
        };
        let result = gage_rr_anova(&input).expect("should compute");

        // With no real interaction, it should likely be pooled
        if result.interaction_pooled {
            assert_eq!(result.variance_components.interaction, 0.0);
        }
    }

    /// Regression (upstream-013): when the interaction is pooled, EVERY variance
    /// component — repeatability included — must be derived from the single pooled
    /// error MS, and the returned ANOVA table's Repeatability row must report that
    /// same pooled term so the Part/Operator F is reproducible from the table.
    ///
    /// The discriminating invariant: the denominator actually used for the Part
    /// F-test is `ms_part / f_part`. Before the fix σ²_repeatability leaked the raw
    /// (un-pooled) MS_error while f_part used the pooled MS, so they disagreed.
    #[test]
    fn anova_pooled_repeatability_uses_pooled_ms() {
        // 3 parts × 3 operators × 3 trials with a dominant part effect and
        // negligible operator/interaction — this pools (interaction p > 0.25).
        let data = vec![
            vec![
                vec![10.0, 10.1, 10.0],
                vec![10.0, 10.0, 10.1],
                vec![10.1, 10.0, 10.0],
            ],
            vec![
                vec![20.0, 20.1, 20.0],
                vec![20.0, 20.0, 20.1],
                vec![20.1, 20.0, 20.0],
            ],
            vec![
                vec![15.0, 15.1, 15.0],
                vec![15.0, 15.0, 15.1],
                vec![15.1, 15.0, 15.0],
            ],
        ];
        let input = GageRRInput {
            measurements: data,
            tolerance: None,
        };
        let result = gage_rr_anova(&input).expect("should compute");
        assert!(result.interaction_pooled, "this dataset should pool");

        let rep_row = result
            .anova_table
            .rows
            .iter()
            .find(|r| r.source == "Repeatability")
            .expect("Repeatability row present");
        let part = result
            .anova_table
            .rows
            .iter()
            .find(|r| r.source == "Part")
            .expect("Part row");

        // The pooled denominator actually used for f_part.
        let pooled_denom = part.ms / part.f_value.expect("Part F present");

        // (a) σ²_repeatability equals that pooled denominator — NOT the raw MS.
        assert!(
            (result.variance_components.repeatability - pooled_denom).abs() < 1e-10,
            "σ²_repeatability ({}) must equal the pooled denominator ({pooled_denom})",
            result.variance_components.repeatability
        );
        // (b) The Repeatability row MS equals it too — F reproducible from table.
        assert!(
            (rep_row.ms - pooled_denom).abs() < 1e-10,
            "Repeatability row MS ({}) must equal the pooled denominator ({pooled_denom})",
            rep_row.ms
        );
        // (c) When pooled, no standalone interaction row; df's sum to df_total.
        assert!(
            !result
                .anova_table
                .rows
                .iter()
                .any(|r| r.source == "Part×Operator"),
            "pooled table must not carry a separate interaction row"
        );
        let df_sum: f64 = result
            .anova_table
            .rows
            .iter()
            .filter(|r| r.source != "Total")
            .map(|r| r.df)
            .sum();
        let df_total = result
            .anova_table
            .rows
            .iter()
            .find(|r| r.source == "Total")
            .expect("Total row")
            .df;
        assert!(
            (df_sum - df_total).abs() < 1e-9,
            "component df ({df_sum}) must sum to total df ({df_total})"
        );
    }

    /// Regression (upstream-011): a degenerate denominator (every trial within
    /// each cell identical → repeatability MS ≈ floating-point noise) must yield
    /// F/p = None, not a spurious ~1e12 finite F, and must do so regardless of the
    /// sign of the rounding noise.
    #[test]
    fn anova_degenerate_denominator_returns_none() {
        // Take the AIAG reference layout (10×3×3) and collapse every trial within
        // a cell to that cell's first value, so the within-cell repeatability SS is
        // 0. At this scale/count the SS *decomposition* residual does not cancel to
        // exact 0 — it lands on surviving floating-point noise (ms_error ≈ 1.85e-16,
        // reproducing the report's ~1e-17 case). The OLD absolute `> 1e-300` guard
        // divides the (real, ~0.08) interaction MS by that noise, yielding a
        // spurious F ≈ 4.4e14 that renders as "very significant"; the relative
        // degeneracy floor must instead report F/p as None.
        let mut data = aiag_reference_data();
        for part in data.iter_mut() {
            for op in part.iter_mut() {
                let first = op[0];
                for trial in op.iter_mut() {
                    *trial = first;
                }
            }
        }
        let input = GageRRInput {
            measurements: data,
            tolerance: None,
        };
        let result = gage_rr_anova(&input).expect("should compute");

        for row in &result.anova_table.rows {
            if let Some(f) = row.f_value {
                assert!(
                    f.is_finite() && f < 1e6,
                    "degenerate denominator must not produce a divergent F for {}: {}",
                    row.source,
                    f
                );
            }
        }
        // Repeatability is ~0, so GRR-derived std devs must stay finite (no NaN).
        assert!(result.ev.is_finite(), "EV must be finite, got {}", result.ev);
        assert!(
            result.percent_grr.is_finite(),
            "%GRR must be finite, got {}",
            result.percent_grr
        );
    }

    #[test]
    fn anova_rejects_invalid_input() {
        let data = vec![vec![vec![1.0, 2.0]]];
        let input = GageRRInput {
            measurements: data,
            tolerance: None,
        };
        assert!(gage_rr_anova(&input).is_err());
    }

    #[test]
    fn anova_status_matches_percent_grr() {
        let data = aiag_reference_data();
        let input = GageRRInput {
            measurements: data,
            tolerance: None,
        };
        let result = gage_rr_anova(&input).expect("should compute");

        match result.status {
            GrrStatus::Acceptable => assert!(result.percent_grr <= 10.0),
            GrrStatus::Marginal => {
                assert!(result.percent_grr > 10.0 && result.percent_grr <= 30.0)
            }
            GrrStatus::Unacceptable => assert!(result.percent_grr > 30.0),
        }
    }

    #[test]
    fn anova_p_values_bounded() {
        let data = aiag_reference_data();
        let input = GageRRInput {
            measurements: data,
            tolerance: None,
        };
        let result = gage_rr_anova(&input).expect("should compute");

        for row in &result.anova_table.rows {
            if let Some(p) = row.p_value {
                assert!(
                    (0.0..=1.0).contains(&p),
                    "p-value for {} out of range: {}",
                    row.source,
                    p
                );
            }
            if let Some(f) = row.f_value {
                assert!(
                    f >= 0.0,
                    "F-value for {} should be non-negative: {}",
                    row.source,
                    f
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // Consistency between methods
    // -----------------------------------------------------------------------

    #[test]
    fn both_methods_detect_same_dominant_variation() {
        let data = aiag_reference_data();
        let input_xr = GageRRInput {
            measurements: data.clone(),
            tolerance: None,
        };
        let input_anova = GageRRInput {
            measurements: data,
            tolerance: None,
        };

        let xr = gage_rr_xbar_r(&input_xr).expect("X̄-R should compute");
        let anova = gage_rr_anova(&input_anova).expect("ANOVA should compute");

        // Both methods should agree on whether PV dominates over GRR
        let xr_pv_dominant = xr.pv > xr.grr;
        let anova_pv_dominant = anova.pv > anova.grr;
        assert_eq!(
            xr_pv_dominant, anova_pv_dominant,
            "X̄-R and ANOVA should agree on PV vs GRR dominance"
        );
    }
}
