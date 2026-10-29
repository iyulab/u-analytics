//! A measurement system study gives the same verdict for the same parts
//! measured in another unit: a gage in metres reads micrometres of variation,
//! so the variance the study works with is far below any fixed number.

use u_analytics::msa::{gage_rr_anova, GageRRInput};

const SCALES: [f64; 7] = [1e-12, 1e-9, 1e-6, 1e-3, 1e3, 1e6, 1e9];

/// 5 parts x 3 operators x 3 trials: part differences, a small operator
/// effect, small repeatability noise. Deterministic.
fn study() -> Vec<Vec<Vec<f64>>> {
    let mut state = 20261029u64;
    let mut noise = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 0.2
    };
    (0..5)
        .map(|p| {
            (0..3)
                .map(|o| {
                    (0..3)
                        .map(|_| 10.0 + 1.5 * p as f64 + 0.1 * o as f64 + noise())
                        .collect()
                })
                .collect()
        })
        .collect()
}

fn scaled(k: f64) -> GageRRInput {
    GageRRInput {
        measurements: study()
            .into_iter()
            .map(|p| {
                p.into_iter()
                    .map(|o| o.into_iter().map(|v| v * k).collect())
                    .collect()
            })
            .collect(),
        tolerance: Some(12.0 * k),
    }
}

#[test]
fn the_anova_study_gives_the_same_verdict_in_any_unit() {
    let base = gage_rr_anova(&scaled(1.0)).expect("a study");
    for k in SCALES {
        let got = gage_rr_anova(&scaled(k)).expect("a study");
        for (a, b) in got.anova_table.rows.iter().zip(&base.anova_table.rows) {
            assert_eq!(
                a.f_value.is_some(),
                b.f_value.is_some(),
                "{} F is computable at scale {k:e}",
                a.source
            );
            if let (Some(x), Some(y)) = (a.p_value, b.p_value) {
                assert!((x - y).abs() <= 1e-6, "{} p at {k:e}: {x} vs {y}", a.source);
            }
        }
        assert!(
            (got.percent_grr - base.percent_grr).abs() <= 1e-6 * base.percent_grr.max(1.0),
            "%GRR at {k:e}: {} vs {}",
            got.percent_grr,
            base.percent_grr
        );
        assert_eq!(got.ndc, base.ndc, "ndc at {k:e}");
        assert_eq!(
            got.interaction_significant, base.interaction_significant,
            "interaction at {k:e}"
        );
    }
}
