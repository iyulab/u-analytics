//! Type 1 gage study (`Cg`, `Cgk`, bias and its t test) against an independent
//! computation, and against the example of a published study.
//!
//! Two tiers:
//!
//! 1. the figures printed in the published example (BPI Consulting, "Type 1
//!    Gage Study", SPC for Excel: 50 readings of a 12.306 reference, limits
//!    12.285 and 12.335, K = 20 %, L = 6), compared at the precision they are
//!    printed to;
//! 2. full-precision values from NumPy 2 / SciPy (`std(ddof=1)`, `ttest_1samp`)
//!    on seeded samples, the two-sided p-value cross-checked against a 50-digit
//!    `mpmath` incomplete beta (so the `1e-40` and `1e-74` tails are verified,
//!    not just produced), for the default multipliers and for other editions'
//!    (K = 15 %, L = 4; K = 10 %, L = 5.15; K = 25 %).
//!
//! The expected values were computed from the definitions, not from this crate.
#![allow(clippy::excessive_precision, clippy::approx_constant)]

use u_analytics::msa::type1::{type1_gage_study, Type1GageResult, Type1Options};

struct Want {
    mean: f64,
    std_dev: f64,
    bias: f64,
    study_variation: f64,
    cg: f64,
    cgk: f64,
    pv_rep: f64,
    pv_rep_bias: Option<f64>,
    t: f64,
    df: f64,
    p: f64,
}

struct Case {
    name: &'static str,
    data: &'static [f64],
    reference: f64,
    limits: (f64, f64),
    fraction: f64,
    multiple: f64,
    want: Want,
}

const CASES: &[Case] = &[
    // the 50-value example of the published Type 1 study (reference 12.306)
    Case {
        name: "published example, Minitab defaults",
        data: &[
            12.3075, 12.3018, 12.3116, 12.3074, 12.3077, 12.3127, 12.3035, 12.2985, 12.3059,
            12.3127, 12.3096, 12.3092, 12.3036, 12.319, 12.3086, 12.3154, 12.3009, 12.3054,
            12.3132, 12.2995, 12.3142, 12.3014, 12.3055, 12.3028, 12.3096, 12.3023, 12.2979,
            12.3091, 12.3091, 12.301, 12.3102, 12.3068, 12.3116, 12.3029, 12.3045, 12.303, 12.3068,
            12.3153, 12.3072, 12.3097, 12.3104, 12.3076, 12.3071, 12.3162, 12.3117, 12.3104,
            12.3066, 12.3019, 12.3124, 12.3061,
        ],
        reference: 12.306,
        limits: (12.285, 12.335),
        fraction: 0.2,
        multiple: 6.0,
        want: Want {
            mean: 12.3075,
            std_dev: 0.004854453036719426,
            bias: 0.0015000000000000568,
            study_variation: 0.029126718220316556,
            cg: 0.34332738499268733,
            cgk: 0.24032916949487867,
            pv_rep: 58.253436440632285,
            t: 2.184922098858415,
            df: 49.0,
            p: 0.03370650522772363,
            pv_rep_bias: Some(83.21919491518982),
        },
    },
    // seeded normal, bias about +0.002
    Case {
        name: "shaft, 50 readings, small bias",
        data: &[
            20.00398, 20.00433, 20.00427, 20.00385, 20.00514, 20.00098, 20.00395, 20.00397,
            20.00358, 20.00441, 20.00512, 20.0033, 20.00334, 20.00433, 20.00572, 20.00583,
            20.00452, 20.00431, 20.00478, 20.00357, 20.00395, 20.00349, 20.00573, 20.00336,
            20.00338, 20.00224, 20.00482, 20.00249, 20.00012, 20.00586, 20.00426, 20.00179,
            20.00256, 20.00339, 20.00564, 20.00484, 20.0046, 20.00355, 20.00386, 20.00247,
            20.00282, 20.00499, 20.00366, 20.00394, 20.00514, 20.00642, 20.00361, 20.00403,
            20.00484, 20.00358,
        ],
        reference: 20.002,
        limits: (19.95, 20.05),
        fraction: 0.2,
        multiple: 6.0,
        want: Want {
            mean: 20.003974200000002,
            std_dev: 0.0012264701132827308,
            bias: 0.0019742000000029236,
            study_variation: 0.0073588206796963845,
            cg: 2.717826791890988,
            cgk: 2.1812734266350824,
            pv_rep: 7.35882067969628,
            t: 11.382015691227405,
            df: 49.0,
            p: 2.3056456580183135e-15,
            pv_rep_bias: Some(9.168955966631282),
        },
    },
    // K = 15 %, L = 4: a different edition is a configuration
    Case {
        name: "fifteen percent, four sigma",
        data: &[
            4.9912, 4.964, 5.0002, 4.9832, 5.0151, 4.9926, 4.9927, 5.0342, 4.9787, 4.978, 5.0068,
            4.9633, 5.004, 4.993, 5.0392, 5.0157, 4.9979, 5.0074, 5.0137, 4.9791, 4.983, 5.0232,
            5.0074, 5.0237, 5.0122,
        ],
        reference: 5.01,
        limits: (4.9, 5.1),
        fraction: 0.15,
        multiple: 4.0,
        want: Want {
            mean: 4.999980000000001,
            std_dev: 0.020017242567346768,
            bias: -0.01001999999999903,
            study_variation: 0.08006897026938707,
            cg: 0.3746769803466531,
            cgk: 0.12439275747511216,
            pv_rep: 40.03448513469368,
            t: -2.5028422287154095,
            df: 24.0,
            p: 0.019530188461093632,
            pv_rep_bias: Some(120.58579859845233),
        },
    },
    // K = 10 %, L = 5.15
    Case {
        name: "five point one five, ten percent",
        data: &[
            100.23, 99.911, 100.947, 100.074, 99.378, 99.78, 100.319, 100.545, 100.432, 100.932,
            99.966, 99.732, 100.037, 100.482, 100.476, 100.564, 99.719, 99.525, 99.94, 99.969,
            98.835, 100.743, 100.2, 100.087, 100.165, 100.153, 98.597, 99.472, 100.335, 100.021,
        ],
        reference: 100.3,
        limits: (97.0, 103.0),
        fraction: 0.1,
        multiple: 5.15,
        want: Want {
            mean: 100.05220000000001,
            std_dev: 0.5349349557624301,
            bias: -0.2477999999999838,
            study_variation: 2.7549150221765153,
            cg: 0.21779256171973366,
            cgk: 0.037895905739245436,
            pv_rep: 45.91525036960859,
            t: -2.5372365048816965,
            df: 29.0,
            p: 0.016809673483291283,
            pv_rep_bias: Some(263.880749250542),
        },
    },
    // |bias| above K*T/2: Cgk negative, no repeatability-and-bias percentage
    Case {
        name: "bias beyond the band",
        data: &[
            10.0096, 10.0204, 9.9919, 10.0032, 10.0128, 10.0026, 9.9997, 10.0103, 10.0058, 9.9909,
            10.0058, 9.9989, 9.9901, 9.9977, 9.9998, 10.0262, 9.9965, 9.995, 10.0071, 9.9985,
            10.0033, 9.9916, 9.9852, 9.9992, 10.0069, 10.0177, 10.0073, 10.0103, 9.999, 9.9805,
            9.9917, 10.0005, 10.017, 9.9841, 9.9854, 9.9902, 10.0055, 10.0029, 9.9915, 10.003,
        ],
        reference: 9.7,
        limits: (9.0, 11.0),
        fraction: 0.2,
        multiple: 6.0,
        want: Want {
            mean: 10.000890000000002,
            std_dev: 0.01025692511528255,
            bias: 0.30089000000000254,
            study_variation: 0.0615415506916953,
            cg: 6.4996737245683,
            cgk: -3.278760410358561,
            pv_rep: 3.077077534584765,
            t: 185.53274289784278,
            df: 39.0,
            p: 4.4835941230676805e-59,
            pv_rep_bias: None,
        },
    },
    // n = 200 and a bias of several standard errors: p far below 1e-16
    Case {
        name: "far tail p",
        data: &[
            1.0534, 0.9684, 0.9915, 0.9513, 0.9732, 0.962, 1.0702, 0.9897, 1.0069, 1.0464, 1.1715,
            1.023, 1.0622, 0.9918, 0.908, 1.0314, 0.926, 1.0182, 0.9612, 1.0245, 0.9244, 0.9809,
            1.1092, 1.0446, 0.9796, 1.0222, 1.0547, 1.0242, 0.9259, 1.0414, 0.9949, 0.9826, 0.9632,
            1.0187, 1.0002, 1.0126, 0.9707, 1.006, 0.9766, 0.8887, 0.9359, 0.9793, 1.0402, 1.103,
            1.0168, 0.9674, 0.9951, 0.9938, 1.0179, 0.9574, 1.0759, 0.9694, 0.9878, 0.9723, 1.0156,
            0.9886, 1.0909, 0.9527, 0.9882, 1.0328, 0.9753, 1.0137, 1.0198, 0.9665, 1.0166, 0.8774,
            1.0372, 0.8675, 1.0197, 0.9469, 0.968, 0.9696, 0.9999, 1.0083, 0.9657, 1.001, 1.0447,
            1.0384, 1.0312, 0.9774, 1.0354, 0.9964, 1.0595, 1.0901, 0.9963, 0.9056, 1.0007, 1.0417,
            0.9661, 0.9532, 1.0115, 0.9233, 0.9616, 1.0167, 0.9726, 0.9365, 0.9761, 0.9854, 0.9942,
            1.0023, 0.9647, 1.114, 1.0211, 0.9088, 1.0299, 0.9488, 1.0454, 1.0803, 0.9347, 1.0025,
            0.9415, 1.0491, 0.9842, 0.9575, 1.0526, 0.9839, 0.9903, 1.0995, 1.0437, 1.0175, 1.024,
            1.062, 1.009, 1.0249, 0.9666, 0.9635, 0.9201, 0.9294, 1.0127, 0.9644, 1.0233, 0.9726,
            1.0316, 1.0057, 0.9473, 0.9999, 0.9823, 0.9752, 1.0443, 1.0206, 0.8993, 1.017, 0.97,
            1.0234, 0.9362, 1.0523, 1.0204, 1.0362, 1.0633, 1.0976, 1.0319, 1.0496, 0.9203, 1.0052,
            1.0372, 1.0276, 0.9765, 0.9767, 1.0179, 1.0883, 0.9674, 0.9624, 0.9868, 1.0178, 1.0199,
            0.9945, 1.0254, 1.0116, 0.9194, 1.0747, 0.9912, 1.0898, 0.9602, 1.067, 0.9408, 0.9451,
            1.0523, 0.9691, 0.9999, 0.9773, 0.9675, 0.9143, 1.0302, 1.0034, 1.0237, 1.0612, 1.0038,
            0.9668, 0.9959, 1.0536, 1.0186, 1.0541, 1.0184, 1.0218, 0.9978, 0.9429, 0.9449, 1.0103,
            1.0014, 0.9958,
        ],
        reference: 0.9,
        limits: (0.0, 2.0),
        fraction: 0.2,
        multiple: 6.0,
        want: Want {
            mean: 0.999992,
            std_dev: 0.048003895738056,
            bias: 0.09999199999999997,
            study_variation: 0.28802337442833603,
            cg: 1.3887761741348712,
            cgk: 0.6944436381144012,
            pv_rep: 14.401168721416802,
            t: 29.458034677111627,
            df: 199.0,
            p: 1.7334388534166026e-74,
            pv_rep_bias: Some(28.80003344015838),
        },
    },
    // the smallest sizes
    Case {
        name: "five readings",
        data: &[3.225, 2.554, 3.279, 2.453, 2.707],
        reference: 3.1,
        limits: (0.0, 6.0),
        fraction: 0.2,
        multiple: 6.0,
        want: Want {
            mean: 2.8436,
            std_dev: 0.3841012887247322,
            bias: -0.2564000000000002,
            study_variation: 2.3046077323483933,
            cg: 0.5206959879359604,
            cgk: 0.2981852357579932,
            pv_rep: 38.41012887247322,
            t: -1.4926475027836323,
            df: 4.0,
            p: 0.20981494087800087,
            pv_rep_bias: Some(67.07240198918493),
        },
    },
    // n = 2, bias exactly zero: t = 0, p = 1
    Case {
        name: "two readings",
        data: &[10.0, 10.2],
        reference: 10.1,
        limits: (9.0, 11.0),
        fraction: 0.2,
        multiple: 6.0,
        want: Want {
            mean: 10.1,
            std_dev: 0.141421356237309,
            bias: 0.0,
            study_variation: 0.848528137423854,
            cg: 0.4714045207910334,
            cgk: 0.4714045207910334,
            pv_rep: 42.426406871192704,
            t: 0.0,
            df: 1.0,
            p: 1.0,
            pv_rep_bias: Some(42.4264068711927),
        },
    },
    // K = 25 %
    Case {
        name: "large values",
        data: &[
            250.75, 245.82, 252.12, 253.04, 251.82, 248.93, 247.82, 250.52, 251.27, 253.54, 247.6,
            248.13, 246.26, 248.41, 259.8, 245.94, 246.73, 246.78, 255.39, 253.54, 249.2, 251.47,
            251.44, 249.97, 250.33, 254.71, 253.77, 246.36, 247.71, 243.36, 253.15, 247.64, 249.31,
            249.32, 245.01, 250.71, 250.97, 246.02, 253.33, 249.54, 248.27, 248.97, 250.09, 253.7,
            252.56, 251.36, 255.68, 250.72, 253.98, 252.76, 251.02, 248.41, 250.73, 246.86, 259.82,
            251.76, 245.93, 250.23, 247.61, 252.48,
        ],
        reference: 249.0,
        limits: (200.0, 300.0),
        fraction: 0.25,
        multiple: 6.0,
        want: Want {
            mean: 250.34116666666668,
            std_dev: 3.3130441179802617,
            bias: 1.3411666666666804,
            study_variation: 19.87826470788157,
            cg: 1.2576550502462975,
            cgk: 1.1227170477218702,
            pv_rep: 19.87826470788157,
            t: 3.1356758192856122,
            df: 59.0,
            p: 0.002672761691725454,
            pv_rep_bias: Some(22.267409273536952),
        },
    },
];

fn run(case: &Case) -> Type1GageResult {
    let options = Type1Options {
        tolerance_fraction: case.fraction,
        sigma_multiple: case.multiple,
    };
    type1_gage_study(
        case.data,
        case.reference,
        case.limits.0,
        case.limits.1,
        &options,
    )
    .unwrap_or_else(|| panic!("{}: refused", case.name))
}

fn close(name: &str, what: &str, got: f64, want: f64, rel: f64) {
    let scale = want.abs().max(1e-300);
    assert!(
        (got - want).abs() <= rel * scale,
        "{name}: {what} = {got:e}, expected {want:e}"
    );
}

#[test]
fn matches_the_independent_computation() {
    for case in CASES {
        let r = run(case);
        let w = &case.want;
        let n = case.name;
        assert_eq!(r.n, case.data.len(), "{n}");
        close(n, "mean", r.mean, w.mean, 1e-13);
        close(n, "std_dev", r.std_dev, w.std_dev, 1e-12);
        // The bias is a difference of two close numbers: absolute to the data's ulp.
        assert!(
            (r.bias - w.bias).abs() <= 1e-12 * w.mean.abs().max(1.0),
            "{n}: bias"
        );
        close(
            n,
            "study_variation",
            r.study_variation,
            w.study_variation,
            1e-12,
        );
        close(n, "cg", r.cg, w.cg, 1e-12);
        close(n, "cgk", r.cgk, w.cgk, 1e-10);
        close(
            n,
            "%var repeatability",
            r.percent_var_repeatability,
            w.pv_rep,
            1e-12,
        );
        match (r.percent_var_repeatability_and_bias, w.pv_rep_bias) {
            (Some(got), Some(want)) => close(n, "%var repeatability and bias", got, want, 1e-10),
            (None, None) => {}
            (got, want) => panic!("{n}: %var repeatability and bias {got:?}, expected {want:?}"),
        }
        close(n, "t", r.t_statistic, w.t, 1e-10);
        assert_eq!(r.df, w.df, "{n}");
        // Far tails are compared in relative terms: 1e-74 keeps its digits.
        close(n, "p", r.p_value, w.p, 1e-9);
    }
}

/// The printed figures of the published example: what the document states,
/// to the digits it states.
#[test]
fn reproduces_the_published_example() {
    let r = run(&CASES[0]);
    let within = |got: f64, printed: f64, half_unit: f64, what: &str| {
        assert!(
            (got - printed).abs() <= half_unit,
            "{what}: {got} against the printed {printed}"
        );
    };
    within(r.mean, 12.3075, 5e-5, "average");
    within(r.std_dev, 0.00485, 5e-6, "standard deviation");
    within(r.study_variation, 0.0291, 5e-5, "study variation");
    within(r.bias, 0.0015, 5e-5, "bias");
    within(r.t_statistic, 2.185, 5e-4, "t");
    within(r.df, 49.0, 0.0, "df");
    within(r.p_value, 0.0337, 5e-5, "p-value");
    within(r.cg, 0.343, 5e-4, "Cg");
    within(r.cgk, 0.240, 5e-4, "Cgk");
    // The document prints 58.2 from the rounded study variation (0.0291 / 0.05);
    // from the unrounded one it is 58.25.
    within(
        r.percent_var_repeatability,
        58.2,
        6e-2,
        "%Var (repeatability)",
    );
    // The document prints 83.22 from the rounded Cgk of 0.240 (20 / 0.240 =
    // 83.33); from the unrounded Cgk it is 83.219.
    within(
        r.percent_var_repeatability_and_bias.unwrap(),
        83.22,
        5e-3,
        "%Var (repeatability + bias)",
    );
    // The same document counts 16 readings outside reference ± 0.1 · tolerance.
    let outside = CASES[0]
        .data
        .iter()
        .filter(|&&v| (v - 12.306).abs() > 0.1 * 0.05)
        .count();
    assert_eq!(outside, 16);
}

/// Cg and Cgk of the default parameters in the closed form quoted for them,
/// `0.2·T / 6s` and `(0.1·T − |x̄ − x_ref|) / 3s`.
#[test]
fn the_defaults_are_the_quoted_closed_forms() {
    for case in CASES
        .iter()
        .filter(|c| c.fraction == 0.2 && c.multiple == 6.0)
    {
        let r = run(case);
        let t = case.limits.1 - case.limits.0;
        close(case.name, "Cg", r.cg, 0.2 * t / (6.0 * r.std_dev), 1e-13);
        close(
            case.name,
            "Cgk",
            r.cgk,
            (0.1 * t - r.bias.abs()) / (3.0 * r.std_dev),
            1e-10,
        );
    }
}

#[test]
fn cgk_is_cg_less_the_bias_in_half_spreads_and_never_above() {
    for case in CASES {
        let r = run(case);
        let half_spread = r.study_variation / 2.0;
        close(
            case.name,
            "Cgk = Cg - |bias| / (SV/2)",
            r.cgk,
            r.cg - r.bias.abs() / half_spread,
            1e-9,
        );
        // Equal for zero bias, up to the last place.
        assert!(r.cgk <= r.cg * (1.0 + 1e-12), "{}", case.name);
        // The two percentages are K over the two indices.
        close(
            case.name,
            "K/Cg",
            r.percent_var_repeatability,
            100.0 * case.fraction / r.cg,
            1e-12,
        );
        if let Some(both) = r.percent_var_repeatability_and_bias {
            assert!(
                r.cgk > 0.0 && both >= r.percent_var_repeatability * (1.0 - 1e-12),
                "{}",
                case.name
            );
        } else {
            assert!(r.cgk <= 0.0, "{}", case.name);
        }
    }
}

/// Multiplying data, reference and limits by one positive constant changes
/// nothing dimensionless. A power of two scales every double exactly.
#[test]
fn the_indices_are_scale_equivariant() {
    for case in CASES {
        let base = run(case);
        for c in [0.25, 2.0, 1024.0, 1.0 / 4096.0] {
            let data: Vec<f64> = case.data.iter().map(|v| v * c).collect();
            let options = Type1Options {
                tolerance_fraction: case.fraction,
                sigma_multiple: case.multiple,
            };
            let r = type1_gage_study(
                &data,
                case.reference * c,
                case.limits.0 * c,
                case.limits.1 * c,
                &options,
            )
            .unwrap();
            for (what, a, b) in [
                ("cg", r.cg, base.cg),
                ("cgk", r.cgk, base.cgk),
                ("t", r.t_statistic, base.t_statistic),
                (
                    "%var",
                    r.percent_var_repeatability,
                    base.percent_var_repeatability,
                ),
                ("mean", r.mean / c, base.mean),
                ("std_dev", r.std_dev / c, base.std_dev),
                ("bias", r.bias / c, base.bias),
            ] {
                close(case.name, &format!("{what} at scale {c}"), a, b, 1e-12);
            }
            close(case.name, "p", r.p_value, base.p_value, 1e-10);
        }
    }
}

/// Adding one constant to data, reference and both limits moves nothing the
/// study measures. (Exactly representable shift; rounding of the sum limits
/// the agreement, hence the looser bound.)
#[test]
fn the_indices_are_shift_invariant() {
    for case in CASES
        .iter()
        .filter(|c| c.data.iter().all(|v| v.abs() < 1000.0))
    {
        let base = run(case);
        for shift in [-16.0, 8.0, 64.0] {
            let data: Vec<f64> = case.data.iter().map(|v| v + shift).collect();
            let options = Type1Options {
                tolerance_fraction: case.fraction,
                sigma_multiple: case.multiple,
            };
            let r = type1_gage_study(
                &data,
                case.reference + shift,
                case.limits.0 + shift,
                case.limits.1 + shift,
                &options,
            )
            .unwrap();
            close(case.name, "cg", r.cg, base.cg, 1e-8);
            close(case.name, "std_dev", r.std_dev, base.std_dev, 1e-8);
            // Cgk and the bias carry the cancellation of the shifted means.
            assert!(
                (r.cgk - base.cgk).abs() <= 1e-7 * base.cgk.abs().max(1.0),
                "{}",
                case.name
            );
            assert!(
                (r.t_statistic - base.t_statistic).abs() <= 1e-7 * base.t_statistic.abs().max(1.0)
            );
        }
    }
}

/// The order of the measurements does not matter.
#[test]
fn the_order_of_the_readings_is_irrelevant() {
    let case = &CASES[1];
    let base = run(case);
    let mut data = case.data.to_vec();
    data.reverse();
    let options = Type1Options {
        tolerance_fraction: case.fraction,
        sigma_multiple: case.multiple,
    };
    let r = type1_gage_study(
        &data,
        case.reference,
        case.limits.0,
        case.limits.1,
        &options,
    )
    .unwrap();
    close(case.name, "cg", r.cg, base.cg, 1e-12);
    close(case.name, "cgk", r.cgk, base.cgk, 1e-10);
}

/// A wider spread is a lower Cg; a larger bias a lower Cgk and a smaller p.
#[test]
fn the_indices_move_the_right_way() {
    let case = &CASES[1];
    let base = run(case);
    let options = Type1Options {
        tolerance_fraction: case.fraction,
        sigma_multiple: case.multiple,
    };
    let wide: Vec<f64> = case.data.iter().map(|v| 20.0 + (v - 20.0) * 3.0).collect();
    let r = type1_gage_study(
        &wide,
        case.reference,
        case.limits.0,
        case.limits.1,
        &options,
    )
    .unwrap();
    assert!(r.cg < base.cg / 2.9 && r.cg > base.cg / 3.1);
    let (reference_far, reference_near) = (case.reference + 0.01, case.reference - 0.001);
    let far = type1_gage_study(
        case.data,
        reference_far,
        case.limits.0,
        case.limits.1,
        &options,
    )
    .unwrap();
    let near = type1_gage_study(
        case.data,
        reference_near,
        case.limits.0,
        case.limits.1,
        &options,
    )
    .unwrap();
    assert!(far.cgk < near.cgk);
    assert!(far.p_value < near.p_value);
    // A larger K or a smaller L is a larger index.
    let loose = Type1Options {
        tolerance_fraction: 0.4,
        ..options
    };
    let dense = Type1Options {
        sigma_multiple: 4.0,
        ..options
    };
    let a = type1_gage_study(
        case.data,
        case.reference,
        case.limits.0,
        case.limits.1,
        &loose,
    )
    .unwrap();
    let b = type1_gage_study(
        case.data,
        case.reference,
        case.limits.0,
        case.limits.1,
        &dense,
    )
    .unwrap();
    close("K", "Cg doubles with K", a.cg, base.cg * 2.0, 1e-12);
    close("L", "Cg scales with 6/L", b.cg, base.cg * 1.5, 1e-12);
}

#[test]
fn refuses_what_has_no_study() {
    let o = Type1Options::default();
    let ok = [10.02, 9.98, 10.05, 10.01];
    assert!(type1_gage_study(&ok, 10.0, 9.0, 11.0, &o).is_some());
    // Too few, or not finite.
    assert!(type1_gage_study(&[10.0], 10.0, 9.0, 11.0, &o).is_none());
    assert!(type1_gage_study(&[], 10.0, 9.0, 11.0, &o).is_none());
    assert!(type1_gage_study(&[10.0, f64::NAN], 10.0, 9.0, 11.0, &o).is_none());
    assert!(type1_gage_study(&ok, f64::INFINITY, 9.0, 11.0, &o).is_none());
    // Limits.
    assert!(type1_gage_study(&ok, 10.0, 11.0, 9.0, &o).is_none());
    assert!(type1_gage_study(&ok, 10.0, 10.0, 10.0, &o).is_none());
    assert!(type1_gage_study(&ok, 10.0, f64::NAN, 11.0, &o).is_none());
    // No spread: neither index exists.
    assert!(type1_gage_study(&[5.0; 7], 5.0, 4.0, 6.0, &o).is_none());
    assert!(type1_gage_study(&[0.1; 3], 0.1, 0.0, 1.0, &o).is_none());
    // Multipliers.
    for fraction in [0.0, -1.0, 1.5, f64::NAN] {
        let bad = Type1Options {
            tolerance_fraction: fraction,
            ..o
        };
        assert!(
            type1_gage_study(&ok, 10.0, 9.0, 11.0, &bad).is_none(),
            "{fraction}"
        );
    }
    for multiple in [0.0, -6.0, f64::NAN, f64::INFINITY] {
        let bad = Type1Options {
            sigma_multiple: multiple,
            ..o
        };
        assert!(
            type1_gage_study(&ok, 10.0, 9.0, 11.0, &bad).is_none(),
            "{multiple}"
        );
    }
    // A spread so small that the index is not a number a double holds.
    assert!(type1_gage_study(&[1.0, 1.0 + f64::EPSILON], 1.0, 0.0, f64::MAX, &o).is_none());
}
