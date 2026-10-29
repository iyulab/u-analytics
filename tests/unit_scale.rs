//! The same observations written in another unit give the same verdict: every
//! hypothesis test of `u_analytics::testing` is judged by a p-value (or a
//! statistic) that does not depend on the scale of the data, from picovolts to
//! megavolts.

use u_analytics::testing::{
    adf_test, anderson_darling_test, bartlett_test, jarque_bera_test, kpss_test,
    kruskal_wallis_test, levene_test, mann_kendall_test, mann_whitney_u_test, one_sample_t_test,
    one_way_anova, paired_t_test, shapiro_wilk_test, two_sample_t_test, wilcoxon_signed_rank_test,
    AdfLags, AdfModel, KpssLags, KpssModel,
};

/// A deterministic spread of `n` values around `centre`, with a few repeats.
fn sample(seed: u64, n: usize, centre: f64, spread: f64) -> Vec<f64> {
    let mut state = seed;
    (0..n)
        .map(|i| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = (state >> 11) as f64 / (1u64 << 53) as f64;
            // Rounded to a coarse grid so that some values are equal.
            let raw = centre + spread * (u - 0.5) * 2.0 + 0.05 * (i as f64).sin();
            (raw * 8.0).round() / 8.0
        })
        .collect()
}

fn scaled(v: &[f64], k: f64) -> Vec<f64> {
    v.iter().map(|x| x * k).collect()
}

fn same(got: f64, want: f64, what: &str, k: f64) {
    let tolerance = 1e-7 * want.abs().max(1e-3);
    assert!(
        (got - want).abs() <= tolerance,
        "{what} at scale {k:e}: {got} vs {want}"
    );
}

#[test]
fn the_tests_give_the_same_verdict_in_any_unit() {
    let a = sample(1, 24, 10.0, 3.0);
    let b = sample(2, 24, 11.0, 3.5);
    let c = sample(3, 20, 9.5, 2.5);
    let series: Vec<f64> = (0..60)
        .map(|i| 0.05 * i as f64 + sample(4, 60, 0.0, 1.0)[i])
        .collect();

    let base = (
        one_sample_t_test(&a, 10.5).expect("t").p_value,
        two_sample_t_test(&a, &b).expect("t2").p_value,
        paired_t_test(&a, &b).expect("paired").p_value,
        one_way_anova(&[&a, &b, &c]).expect("anova").p_value,
        jarque_bera_test(&a).expect("jb").p_value,
        anderson_darling_test(&a).expect("ad").p_value,
        shapiro_wilk_test(&a).expect("sw").p_value,
        mann_whitney_u_test(&a, &b).expect("mw").p_value,
        wilcoxon_signed_rank_test(&a, &b).expect("wx").p_value,
        kruskal_wallis_test(&[&a, &b, &c]).expect("kw").p_value,
        levene_test(&[&a, &b, &c]).expect("levene").p_value,
        bartlett_test(&[&a, &b, &c]).expect("bartlett").p_value,
        mann_kendall_test(&series).expect("mk").p_value,
        kpss_test(&series, KpssModel::Level, KpssLags::Hobijn)
            .expect("kpss")
            .statistic,
        adf_test(&series, AdfModel::Constant, AdfLags::Aic { max: None })
            .expect("adf")
            .statistic,
    );

    for k in [1e-12, 1e-9, 1e-6, 1e6, 1e9, 1e12] {
        let (sa, sb, sc, ss) = (
            scaled(&a, k),
            scaled(&b, k),
            scaled(&c, k),
            scaled(&series, k),
        );
        same(
            one_sample_t_test(&sa, 10.5 * k).expect("t").p_value,
            base.0,
            "one-sample t",
            k,
        );
        same(
            two_sample_t_test(&sa, &sb).expect("t2").p_value,
            base.1,
            "two-sample t",
            k,
        );
        same(
            paired_t_test(&sa, &sb).expect("paired").p_value,
            base.2,
            "paired t",
            k,
        );
        same(
            one_way_anova(&[&sa, &sb, &sc]).expect("anova").p_value,
            base.3,
            "ANOVA",
            k,
        );
        same(
            jarque_bera_test(&sa).expect("jb").p_value,
            base.4,
            "Jarque-Bera",
            k,
        );
        same(
            anderson_darling_test(&sa).expect("ad").p_value,
            base.5,
            "Anderson-Darling",
            k,
        );
        same(
            shapiro_wilk_test(&sa).expect("sw").p_value,
            base.6,
            "Shapiro-Wilk",
            k,
        );
        same(
            mann_whitney_u_test(&sa, &sb).expect("mw").p_value,
            base.7,
            "Mann-Whitney",
            k,
        );
        same(
            wilcoxon_signed_rank_test(&sa, &sb).expect("wx").p_value,
            base.8,
            "Wilcoxon",
            k,
        );
        same(
            kruskal_wallis_test(&[&sa, &sb, &sc]).expect("kw").p_value,
            base.9,
            "Kruskal-Wallis",
            k,
        );
        same(
            levene_test(&[&sa, &sb, &sc]).expect("levene").p_value,
            base.10,
            "Levene",
            k,
        );
        same(
            bartlett_test(&[&sa, &sb, &sc]).expect("bartlett").p_value,
            base.11,
            "Bartlett",
            k,
        );
        same(
            mann_kendall_test(&ss).expect("mk").p_value,
            base.12,
            "Mann-Kendall",
            k,
        );
        same(
            kpss_test(&ss, KpssModel::Level, KpssLags::Hobijn)
                .expect("kpss")
                .statistic,
            base.13,
            "KPSS",
            k,
        );
        same(
            adf_test(&ss, AdfModel::Constant, AdfLags::Aic { max: None })
                .unwrap_or_else(|e| panic!("ADF at scale {k:e}: {e:?}"))
                .statistic,
            base.14,
            "ADF",
            k,
        );
    }
}
