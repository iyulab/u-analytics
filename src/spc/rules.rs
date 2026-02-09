//! Run rules for detecting non-random patterns in control charts.
//!
//! Implements Western Electric (4 rules) and Nelson (8 rules) run tests
//! for identifying special causes of variation in control chart data.
//!
//! # References
//!
//! - Nelson, L.S. (1984). "The Shewhart Control Chart — Tests for Special Causes",
//!   *Journal of Quality Technology* 16(4), pp. 237-239.
//! - Western Electric (1956). *Statistical Quality Control Handbook*.
//! - Montgomery, D.C. (2019). *Introduction to Statistical Quality Control*, 8th ed.

use super::chart::{ChartPoint, ControlLimits, ViolationType};

/// Trait for applying run rules to chart data.
///
/// Run rules detect non-random patterns that indicate special causes of
/// variation even when individual points remain within control limits.
pub trait RunRule {
    /// Check points against this rule set and return violations per point index.
    ///
    /// Returns a vector of `(point_index, violation_type)` pairs. A single
    /// point may appear multiple times if it triggers multiple rules.
    fn check(&self, points: &[ChartPoint], limits: &ControlLimits) -> Vec<(usize, ViolationType)>;
}

/// Western Electric rules (4 rules).
///
/// A subset of Nelson's rules, these are the original run tests from the
/// Western Electric *Statistical Quality Control Handbook* (1956):
///
/// 1. Any point beyond 3 sigma (Nelson Rule 1)
/// 2. 2 of 3 consecutive points beyond 2 sigma, same side (Nelson Rule 5)
/// 3. 4 of 5 consecutive points beyond 1 sigma, same side (Nelson Rule 6)
/// 4. 9 consecutive points on the same side of center line (Nelson Rule 2)
pub struct WesternElectricRules;

/// Nelson rules (8 rules, superset of Western Electric).
///
/// The full set of eight tests for special causes as defined by
/// Nelson (1984). Rules 1, 2, 5, and 6 correspond to the Western
/// Electric rules.
pub struct NelsonRules;

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Compute 1-sigma and 2-sigma zone boundaries from control limits.
///
/// Returns `(one_sigma, two_sigma)` where `one_sigma = (UCL - CL) / 3`.
fn zone_widths(limits: &ControlLimits) -> (f64, f64) {
    let sigma = (limits.ucl - limits.cl) / 3.0;
    (sigma, 2.0 * sigma)
}

/// Nelson Rule 1: Point beyond control limits.
///
/// A single point falls outside the UCL or LCL.
fn check_rule1(points: &[ChartPoint], limits: &ControlLimits) -> Vec<(usize, ViolationType)> {
    let mut violations = Vec::new();
    for point in points {
        if point.value > limits.ucl || point.value < limits.lcl {
            violations.push((point.index, ViolationType::BeyondLimits));
        }
    }
    violations
}

/// Nelson Rule 2: 9 consecutive points on the same side of center line.
///
/// Indicates a sustained shift in the process mean.
fn check_rule2(points: &[ChartPoint], limits: &ControlLimits) -> Vec<(usize, ViolationType)> {
    let mut violations = Vec::new();
    if points.len() < 9 {
        return violations;
    }

    let cl = limits.cl;
    // Track consecutive runs above/below center line.
    // +1 = above, -1 = below, 0 = exactly on center (counted as neither side).
    let sides: Vec<i8> = points
        .iter()
        .map(|p| {
            if p.value > cl {
                1
            } else if p.value < cl {
                -1
            } else {
                0
            }
        })
        .collect();

    let mut run_length = 1_usize;
    for i in 1..sides.len() {
        if sides[i] != 0 && sides[i] == sides[i - 1] {
            run_length += 1;
        } else {
            run_length = 1;
        }
        if run_length >= 9 {
            violations.push((points[i].index, ViolationType::NineOneSide));
        }
    }
    violations
}

/// Nelson Rule 3: 6 consecutive points steadily increasing or decreasing.
///
/// Indicates a trend in the process.
fn check_rule3(points: &[ChartPoint], limits: &ControlLimits) -> Vec<(usize, ViolationType)> {
    let _ = limits; // Not used for trend detection
    let mut violations = Vec::new();
    if points.len() < 6 {
        return violations;
    }

    // Direction: +1 = increasing, -1 = decreasing, 0 = equal
    let dirs: Vec<i8> = points
        .windows(2)
        .map(|w| {
            if w[1].value > w[0].value {
                1
            } else if w[1].value < w[0].value {
                -1
            } else {
                0
            }
        })
        .collect();

    let mut run_length = 1_usize;
    for i in 1..dirs.len() {
        if dirs[i] != 0 && dirs[i] == dirs[i - 1] {
            run_length += 1;
        } else {
            run_length = 1;
        }
        // 5 consecutive same-direction changes = 6 points forming a trend.
        // The violation is reported at the last point of the trend.
        if run_length >= 5 {
            // dirs[i] corresponds to the transition between points[i] and points[i+1],
            // so the last point of the 6-point trend is points[i+1].
            violations.push((points[i + 1].index, ViolationType::SixTrend));
        }
    }
    violations
}

/// Nelson Rule 4: 14 consecutive points alternating up and down.
///
/// Indicates systematic variation (e.g., two alternating streams).
fn check_rule4(points: &[ChartPoint], limits: &ControlLimits) -> Vec<(usize, ViolationType)> {
    let _ = limits;
    let mut violations = Vec::new();
    if points.len() < 14 {
        return violations;
    }

    // Direction changes: +1 = up, -1 = down, 0 = equal
    let dirs: Vec<i8> = points
        .windows(2)
        .map(|w| {
            if w[1].value > w[0].value {
                1
            } else if w[1].value < w[0].value {
                -1
            } else {
                0
            }
        })
        .collect();

    // Count consecutive alternations
    let mut alt_length = 1_usize;
    for i in 1..dirs.len() {
        if dirs[i] != 0 && dirs[i - 1] != 0 && dirs[i] == -dirs[i - 1] {
            alt_length += 1;
        } else {
            alt_length = 1;
        }
        // 13 consecutive alternating directions = 14 points alternating.
        // dirs[i] corresponds to points[i]→points[i+1],
        // so the last point is points[i+1].
        if alt_length >= 13 {
            violations.push((points[i + 1].index, ViolationType::FourteenAlternating));
        }
    }
    violations
}

/// Nelson Rule 5: 2 out of 3 consecutive points beyond 2 sigma, same side.
///
/// An early warning of a potential shift.
fn check_rule5(points: &[ChartPoint], limits: &ControlLimits) -> Vec<(usize, ViolationType)> {
    let mut violations = Vec::new();
    if points.len() < 3 {
        return violations;
    }

    let (_, two_sigma) = zone_widths(limits);
    let upper_2s = limits.cl + two_sigma;
    let lower_2s = limits.cl - two_sigma;

    for i in 2..points.len() {
        let window = &points[i - 2..=i];

        // Check upper side: count points above CL + 2σ
        let above_count = window.iter().filter(|p| p.value > upper_2s).count();
        if above_count >= 2 {
            violations.push((points[i].index, ViolationType::TwoOfThreeBeyond2Sigma));
            continue;
        }

        // Check lower side: count points below CL - 2σ
        let below_count = window.iter().filter(|p| p.value < lower_2s).count();
        if below_count >= 2 {
            violations.push((points[i].index, ViolationType::TwoOfThreeBeyond2Sigma));
        }
    }
    violations
}

/// Nelson Rule 6: 4 out of 5 consecutive points beyond 1 sigma, same side.
///
/// Indicates a small sustained shift.
fn check_rule6(points: &[ChartPoint], limits: &ControlLimits) -> Vec<(usize, ViolationType)> {
    let mut violations = Vec::new();
    if points.len() < 5 {
        return violations;
    }

    let (one_sigma, _) = zone_widths(limits);
    let upper_1s = limits.cl + one_sigma;
    let lower_1s = limits.cl - one_sigma;

    for i in 4..points.len() {
        let window = &points[i - 4..=i];

        // Check upper side: count points above CL + σ
        let above_count = window.iter().filter(|p| p.value > upper_1s).count();
        if above_count >= 4 {
            violations.push((points[i].index, ViolationType::FourOfFiveBeyond1Sigma));
            continue;
        }

        // Check lower side: count points below CL - σ
        let below_count = window.iter().filter(|p| p.value < lower_1s).count();
        if below_count >= 4 {
            violations.push((points[i].index, ViolationType::FourOfFiveBeyond1Sigma));
        }
    }
    violations
}

/// Nelson Rule 7: 15 consecutive points within 1 sigma of center line.
///
/// Indicates stratification — reduced variation suggesting mixed streams.
fn check_rule7(points: &[ChartPoint], limits: &ControlLimits) -> Vec<(usize, ViolationType)> {
    let mut violations = Vec::new();
    if points.len() < 15 {
        return violations;
    }

    let (one_sigma, _) = zone_widths(limits);
    let upper_1s = limits.cl + one_sigma;
    let lower_1s = limits.cl - one_sigma;

    let mut run_length = 0_usize;
    for point in points {
        if point.value >= lower_1s && point.value <= upper_1s {
            run_length += 1;
        } else {
            run_length = 0;
        }
        if run_length >= 15 {
            violations.push((point.index, ViolationType::FifteenWithin1Sigma));
        }
    }
    violations
}

/// Nelson Rule 8: 8 consecutive points beyond 1 sigma on either side.
///
/// Indicates a mixture pattern — points avoid the center zone.
fn check_rule8(points: &[ChartPoint], limits: &ControlLimits) -> Vec<(usize, ViolationType)> {
    let mut violations = Vec::new();
    if points.len() < 8 {
        return violations;
    }

    let (one_sigma, _) = zone_widths(limits);
    let upper_1s = limits.cl + one_sigma;
    let lower_1s = limits.cl - one_sigma;

    let mut run_length = 0_usize;
    for point in points {
        // Point is beyond 1σ on either side (not within CL ± σ)
        if point.value > upper_1s || point.value < lower_1s {
            run_length += 1;
        } else {
            run_length = 0;
        }
        if run_length >= 8 {
            violations.push((point.index, ViolationType::EightBeyond1Sigma));
        }
    }
    violations
}

// ---------------------------------------------------------------------------
// RunRule implementations
// ---------------------------------------------------------------------------

impl RunRule for WesternElectricRules {
    /// Apply the 4 Western Electric run rules.
    ///
    /// These correspond to Nelson Rules 1, 2, 5, and 6.
    fn check(&self, points: &[ChartPoint], limits: &ControlLimits) -> Vec<(usize, ViolationType)> {
        let mut results = Vec::new();
        results.extend(check_rule1(points, limits));
        results.extend(check_rule2(points, limits));
        results.extend(check_rule5(points, limits));
        results.extend(check_rule6(points, limits));
        results.sort_by_key(|&(idx, _)| idx);
        results
    }
}

impl RunRule for NelsonRules {
    /// Apply all 8 Nelson run rules.
    fn check(&self, points: &[ChartPoint], limits: &ControlLimits) -> Vec<(usize, ViolationType)> {
        let mut results = Vec::new();
        results.extend(check_rule1(points, limits));
        results.extend(check_rule2(points, limits));
        results.extend(check_rule3(points, limits));
        results.extend(check_rule4(points, limits));
        results.extend(check_rule5(points, limits));
        results.extend(check_rule6(points, limits));
        results.extend(check_rule7(points, limits));
        results.extend(check_rule8(points, limits));
        results.sort_by_key(|&(idx, _)| idx);
        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create chart points from a slice of values.
    fn make_points(values: &[f64]) -> Vec<ChartPoint> {
        values
            .iter()
            .enumerate()
            .map(|(i, &v)| ChartPoint {
                value: v,
                index: i,
                violations: Vec::new(),
            })
            .collect()
    }

    // --- Rule 1: Beyond limits ---

    #[test]
    fn test_rule1_point_above_ucl() {
        let limits = ControlLimits {
            ucl: 30.0,
            cl: 25.0,
            lcl: 20.0,
        };
        let points = make_points(&[25.0, 31.0, 25.0]);
        let violations = check_rule1(&points, &limits);
        assert_eq!(violations.len(), 1);
        assert_eq!(violations[0].0, 1);
        assert_eq!(violations[0].1, ViolationType::BeyondLimits);
    }

    #[test]
    fn test_rule1_point_below_lcl() {
        let limits = ControlLimits {
            ucl: 30.0,
            cl: 25.0,
            lcl: 20.0,
        };
        let points = make_points(&[25.0, 19.0, 25.0]);
        let violations = check_rule1(&points, &limits);
        assert_eq!(violations.len(), 1);
        assert_eq!(violations[0].0, 1);
    }

    #[test]
    fn test_rule1_on_limit_is_not_violation() {
        let limits = ControlLimits {
            ucl: 30.0,
            cl: 25.0,
            lcl: 20.0,
        };
        let points = make_points(&[30.0, 20.0]);
        let violations = check_rule1(&points, &limits);
        assert!(violations.is_empty());
    }

    // --- Rule 2: 9 on same side ---

    #[test]
    fn test_rule2_nine_above() {
        let limits = ControlLimits {
            ucl: 30.0,
            cl: 25.0,
            lcl: 20.0,
        };
        // 9 points above center line
        let values: Vec<f64> = (0..9).map(|_| 26.0).collect();
        let points = make_points(&values);
        let violations = check_rule2(&points, &limits);
        assert_eq!(violations.len(), 1);
        assert_eq!(violations[0].1, ViolationType::NineOneSide);
    }

    #[test]
    fn test_rule2_eight_not_enough() {
        let limits = ControlLimits {
            ucl: 30.0,
            cl: 25.0,
            lcl: 20.0,
        };
        let values: Vec<f64> = (0..8).map(|_| 26.0).collect();
        let points = make_points(&values);
        let violations = check_rule2(&points, &limits);
        assert!(violations.is_empty());
    }

    #[test]
    fn test_rule2_nine_below() {
        let limits = ControlLimits {
            ucl: 30.0,
            cl: 25.0,
            lcl: 20.0,
        };
        let values: Vec<f64> = (0..9).map(|_| 24.0).collect();
        let points = make_points(&values);
        let violations = check_rule2(&points, &limits);
        assert_eq!(violations.len(), 1);
    }

    // --- Rule 3: 6 trending ---

    #[test]
    fn test_rule3_six_increasing() {
        let limits = ControlLimits {
            ucl: 30.0,
            cl: 25.0,
            lcl: 20.0,
        };
        let points = make_points(&[20.0, 21.0, 22.0, 23.0, 24.0, 25.0]);
        let violations = check_rule3(&points, &limits);
        assert_eq!(violations.len(), 1);
        assert_eq!(violations[0].1, ViolationType::SixTrend);
    }

    #[test]
    fn test_rule3_six_decreasing() {
        let limits = ControlLimits {
            ucl: 30.0,
            cl: 25.0,
            lcl: 20.0,
        };
        let points = make_points(&[30.0, 29.0, 28.0, 27.0, 26.0, 25.0]);
        let violations = check_rule3(&points, &limits);
        assert_eq!(violations.len(), 1);
    }

    #[test]
    fn test_rule3_five_not_enough() {
        let limits = ControlLimits {
            ucl: 30.0,
            cl: 25.0,
            lcl: 20.0,
        };
        let points = make_points(&[20.0, 21.0, 22.0, 23.0, 24.0]);
        let violations = check_rule3(&points, &limits);
        assert!(violations.is_empty());
    }

    // --- Rule 4: 14 alternating ---

    #[test]
    fn test_rule4_fourteen_alternating() {
        let limits = ControlLimits {
            ucl: 30.0,
            cl: 25.0,
            lcl: 20.0,
        };
        // Create alternating pattern: 24, 26, 24, 26, ...
        let values: Vec<f64> = (0..14).map(|i| if i % 2 == 0 { 24.0 } else { 26.0 }).collect();
        let points = make_points(&values);
        let violations = check_rule4(&points, &limits);
        assert_eq!(violations.len(), 1);
        assert_eq!(violations[0].1, ViolationType::FourteenAlternating);
    }

    #[test]
    fn test_rule4_thirteen_not_enough() {
        let limits = ControlLimits {
            ucl: 30.0,
            cl: 25.0,
            lcl: 20.0,
        };
        let values: Vec<f64> = (0..13).map(|i| if i % 2 == 0 { 24.0 } else { 26.0 }).collect();
        let points = make_points(&values);
        let violations = check_rule4(&points, &limits);
        assert!(violations.is_empty());
    }

    // --- Rule 5: 2 of 3 beyond 2σ ---

    #[test]
    fn test_rule5_two_of_three_above() {
        let limits = ControlLimits {
            ucl: 28.0, // σ = (28-25)/3 ≈ 1.0, 2σ = 2.0
            cl: 25.0,
            lcl: 22.0,
        };
        // 2σ line upper = 25 + 2 = 27, lower = 25 - 2 = 23
        // Two of three points above 27
        let points = make_points(&[27.5, 25.0, 27.5]);
        let violations = check_rule5(&points, &limits);
        assert_eq!(violations.len(), 1);
        assert_eq!(violations[0].1, ViolationType::TwoOfThreeBeyond2Sigma);
    }

    #[test]
    fn test_rule5_two_of_three_below() {
        let limits = ControlLimits {
            ucl: 28.0,
            cl: 25.0,
            lcl: 22.0,
        };
        // Two of three below CL - 2σ = 23
        let points = make_points(&[22.5, 25.0, 22.5]);
        let violations = check_rule5(&points, &limits);
        assert_eq!(violations.len(), 1);
    }

    // --- Rule 6: 4 of 5 beyond 1σ ---

    #[test]
    fn test_rule6_four_of_five_above() {
        let limits = ControlLimits {
            ucl: 28.0, // σ = 1.0
            cl: 25.0,
            lcl: 22.0,
        };
        // 1σ upper = 26. Four of five points above 26.
        let points = make_points(&[26.5, 26.5, 25.0, 26.5, 26.5]);
        let violations = check_rule6(&points, &limits);
        assert_eq!(violations.len(), 1);
        assert_eq!(violations[0].1, ViolationType::FourOfFiveBeyond1Sigma);
    }

    // --- Rule 7: 15 within 1σ ---

    #[test]
    fn test_rule7_fifteen_within() {
        let limits = ControlLimits {
            ucl: 28.0, // σ = 1.0
            cl: 25.0,
            lcl: 22.0,
        };
        // 15 points within CL ± σ = [24, 26]
        let values: Vec<f64> = (0..15).map(|i| 24.5 + (i as f64 % 3.0) * 0.25).collect();
        let points = make_points(&values);
        let violations = check_rule7(&points, &limits);
        assert_eq!(violations.len(), 1);
        assert_eq!(violations[0].1, ViolationType::FifteenWithin1Sigma);
    }

    // --- Rule 8: 8 beyond 1σ on either side ---

    #[test]
    fn test_rule8_eight_beyond() {
        let limits = ControlLimits {
            ucl: 28.0, // σ = 1.0
            cl: 25.0,
            lcl: 22.0,
        };
        // 8 points beyond CL ± σ (alternating sides is fine)
        let points = make_points(&[27.0, 23.0, 27.0, 23.0, 27.0, 23.0, 27.0, 23.0]);
        let violations = check_rule8(&points, &limits);
        assert_eq!(violations.len(), 1);
        assert_eq!(violations[0].1, ViolationType::EightBeyond1Sigma);
    }

    #[test]
    fn test_rule8_seven_not_enough() {
        let limits = ControlLimits {
            ucl: 28.0,
            cl: 25.0,
            lcl: 22.0,
        };
        let points = make_points(&[27.0, 23.0, 27.0, 23.0, 27.0, 23.0, 27.0]);
        let violations = check_rule8(&points, &limits);
        assert!(violations.is_empty());
    }

    // --- Western Electric Rules ---

    #[test]
    fn test_western_electric_combines_four_rules() {
        let limits = ControlLimits {
            ucl: 30.0,
            cl: 25.0,
            lcl: 20.0,
        };
        // A point beyond limits triggers WE rules
        let points = make_points(&[25.0, 31.0, 25.0]);
        let we = WesternElectricRules;
        let violations = we.check(&points, &limits);
        assert!(!violations.is_empty());
        assert!(violations
            .iter()
            .any(|(_, v)| *v == ViolationType::BeyondLimits));
    }

    // --- Nelson Rules ---

    #[test]
    fn test_nelson_detects_trend() {
        let limits = ControlLimits {
            ucl: 30.0,
            cl: 25.0,
            lcl: 20.0,
        };
        let points = make_points(&[20.0, 21.0, 22.0, 23.0, 24.0, 25.0]);
        let nelson = NelsonRules;
        let violations = nelson.check(&points, &limits);
        assert!(violations
            .iter()
            .any(|(_, v)| *v == ViolationType::SixTrend));
    }

    #[test]
    fn test_nelson_no_violations_in_random_data() {
        let limits = ControlLimits {
            ucl: 28.0,
            cl: 25.0,
            lcl: 22.0,
        };
        // A short random-looking sequence should not trigger
        let points = make_points(&[25.5, 24.8, 25.2, 24.9, 25.1]);
        let nelson = NelsonRules;
        let violations = nelson.check(&points, &limits);
        assert!(violations.is_empty());
    }

    #[test]
    fn test_nelson_rule2_continuation() {
        // If 10 points on same side, rules fire at point 8 (index) and 9
        let limits = ControlLimits {
            ucl: 30.0,
            cl: 25.0,
            lcl: 20.0,
        };
        let values: Vec<f64> = (0..10).map(|_| 26.0).collect();
        let points = make_points(&values);
        let violations = check_rule2(&points, &limits);
        // Should fire at index 8 and 9
        assert_eq!(violations.len(), 2);
        assert_eq!(violations[0].0, 8);
        assert_eq!(violations[1].0, 9);
    }

    #[test]
    fn test_rule5_not_triggered_mixed_sides() {
        let limits = ControlLimits {
            ucl: 28.0,
            cl: 25.0,
            lcl: 22.0,
        };
        // One point above 2σ upper (>27), one below 2σ lower (<23) — different sides
        // Rule 5 requires 2 of 3 on the SAME side, so this should not trigger
        let points = make_points(&[27.5, 25.0, 22.5]);
        let violations = check_rule5(&points, &limits);
        assert!(violations.is_empty());
    }

    #[test]
    fn test_rule7_fourteen_not_enough() {
        let limits = ControlLimits {
            ucl: 28.0,
            cl: 25.0,
            lcl: 22.0,
        };
        let values: Vec<f64> = (0..14).map(|_| 25.5).collect();
        let points = make_points(&values);
        let violations = check_rule7(&points, &limits);
        assert!(violations.is_empty());
    }
}
