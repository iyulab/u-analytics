//! Normal tolerance factors against an independent computation, published
//! tables and Monte Carlo, plus the properties a tolerance factor must have.
//!
//! The reference factors come from `mpmath` 1.4.1 at 40 digits: the exact
//! confidence as an integral over the chi variable `w = s/σ` (one-sided:
//! `Φ(√n·(k·w − z_P))`; two-sided: `2Φ(√n·A(k·w)) − 1` in `u = √(k·w − z)`, with
//! the centre offset `A` found by Newton at 40 digits), Gauss–Legendre
//! quadrature and `findroot` on `k`. The library computes the same integrals
//! in double precision with adaptive Gauss–Kronrod panels and a Brent root, so
//! the two share no code. For the one-sided factor
//! `scipy.stats.nct.ppf(γ, n − 1, z_P·√n) / √n` (SciPy 1.17.1) was compared
//! with the `mpmath` values wherever its series holds (90 rows, 2e-14). Five
//! two-sided rows (`n` from 300 to 100 000) come from a second `mpmath` oracle
//! that integrates in the other order — over the normal variable, with the
//! regularized incomplete gamma function. The two oracles overlap at `n = 10`
//! and at `n = 100 000` (`P = γ = 0.9999`) and agree to all 25 digits kept.
#![allow(clippy::excessive_precision, clippy::approx_constant)]

use proptest::prelude::*;
use u_analytics::tolerance::{tolerance_factor, tolerance_interval_normal, Sided};
use u_numflow::special;

use Sided::{OneSided as One, TwoSided as Two};

/// `(n, coverage, confidence, sided, k)`.
type Row = (usize, f64, f64, Sided, f64);

const ROWS: &[Row] = &[
    (2, 0.3, 0.5, One, -0.6789912049520631),
    (2, 0.5, 0.5, One, 0.0),
    (2, 0.5, 0.9999, One, 2250.7907163449636),
    (2, 0.75, 0.9, One, 5.842408607235056),
    (2, 0.9, 0.95, One, 20.581467624244965),
    (2, 0.95, 0.95, One, 26.25967398303447),
    (2, 0.99, 0.99, One, 185.61695860321123),
    (2, 0.999, 0.5, One, 4.526691889564526),
    (2, 0.9999, 0.5, One, 5.46829354100839),
    (2, 0.9999, 0.9999, One, 29673.4583377463),
    (3, 0.5, 0.5, One, 0.0),
    (3, 0.5, 0.9999, One, 40.81870506685855),
    (3, 0.75, 0.9, One, 2.6028061347907463),
    (3, 0.9, 0.95, One, 6.1552811033262556),
    (3, 0.95, 0.95, One, 7.655900133153337),
    (3, 0.99, 0.99, One, 23.895563349392766),
    (3, 0.999, 0.5, One, 3.688210374994423),
    (3, 0.9999, 0.5, One, 4.447053349324689),
    (3, 0.9999, 0.9999, One, 376.3461549896256),
    (4, 0.5, 0.5, One, 0.0),
    (4, 0.5, 0.9999, One, 11.101871136602501),
    (4, 0.75, 0.9, One, 1.9722378186779186),
    (4, 0.9, 0.95, One, 4.161933184729743),
    (4, 0.95, 0.95, One, 5.143874860989299),
    (4, 0.99, 0.99, One, 12.387283601295314),
    (4, 0.999, 0.5, One, 3.4645781270467557),
    (4, 0.9999, 0.5, One, 4.174807312803496),
    (4, 0.9999, 0.9999, One, 90.7807284337674),
    (5, 0.5, 0.5, One, 0.0),
    (5, 0.5, 0.9999, One, 5.828835192868392),
    (5, 0.75, 0.9, One, 1.6977905318916957),
    (5, 0.9, 0.95, One, 3.406633262800809),
    (5, 0.95, 0.95, One, 4.202680741256177),
    (5, 0.99, 0.99, One, 8.939024938893022),
    (5, 0.999, 0.5, One, 3.3621440887199725),
    (5, 0.9999, 0.5, One, 4.050110600668103),
    (5, 0.9999, 0.9999, One, 45.04742931943273),
    (7, 0.5, 0.5, One, 0.0),
    (7, 0.5, 0.9999, One, 3.033086571499246),
    (7, 0.75, 0.9, One, 1.4352569648561915),
    (7, 0.9, 0.95, One, 2.7554319798309557),
    (7, 0.95, 0.95, One, 3.399468980315697),
    (7, 0.99, 0.99, One, 6.411943350759197),
    (7, 0.999, 0.5, One, 3.2657190011741064),
    (7, 0.9999, 0.5, One, 3.9327212260490922),
    (7, 0.9999, 0.9999, One, 22.47310382552874),
    (10, 0.3, 0.95, One, -0.004368895531841532),
    (10, 0.5, 0.5, One, 0.0),
    (10, 0.5, 0.9999, One, 1.9005706566454743),
    (10, 0.75, 0.9, One, 1.2567178701462702),
    (10, 0.9, 0.95, One, 2.3546401318290604),
    (10, 0.95, 0.95, One, 2.9109634130781665),
    (10, 0.99, 0.99, One, 5.0737253480452775),
    (10, 0.999, 0.5, One, 3.204691692705887),
    (10, 0.9999, 0.5, One, 3.858416351140112),
    (10, 0.9999, 0.9999, One, 14.0761682916929),
    (20, 0.5, 0.5, One, 0.0),
    (20, 0.5, 0.9999, One, 1.0263249245864023),
    (20, 0.75, 0.9, One, 1.0458986644819848),
    (20, 0.9, 0.95, One, 1.9259909722621964),
    (20, 0.95, 0.95, One, 2.3960016837521696),
    (20, 0.99, 0.99, One, 3.8315579855990776),
    (20, 0.999, 0.5, One, 3.1432046900623405),
    (20, 0.9999, 0.5, One, 3.783539264515978),
    (20, 0.9999, 0.9999, One, 8.335943340828518),
    (30, 0.1, 0.99, One, -0.7968765632000466),
    (35, 0.5, 0.5, One, 0.0),
    (35, 0.5, 0.9999, One, 0.7043892277777798),
    (35, 0.75, 0.9, One, 0.9420763764874519),
    (35, 0.9, 0.95, One, 1.732252088168926),
    (35, 0.95, 0.95, One, 2.1667200001608697),
    (35, 0.99, 0.99, One, 3.3343504778488042),
    (35, 0.999, 0.5, One, 3.119561066727777),
    (35, 0.9999, 0.5, One, 3.754742204134659),
    (35, 0.9999, 0.9999, One, 6.549867658870064),
    (50, 0.5, 0.5, One, 0.0),
    (50, 0.5, 0.9999, One, 0.5685831568516609),
    (50, 0.75, 0.9, One, 0.893605285406783),
    (50, 0.9, 0.95, One, 1.6455648561425475),
    (50, 0.95, 0.95, One, 2.0649934177439753),
    (50, 0.99, 0.99, One, 3.124607242021686),
    (50, 0.999, 0.5, One, 3.1105096131960557),
    (50, 0.9999, 0.5, One, 3.743717067152663),
    (50, 0.9999, 0.9999, One, 5.877415518700387),
    (100, 0.2, 0.9, One, -0.7002023048576578),
    (100, 0.5, 0.5, One, 0.0),
    (100, 0.5, 0.9999, One, 0.38630891071273094),
    (100, 0.75, 0.9, One, 0.8249641662649905),
    (100, 0.9, 0.95, One, 1.5267487478503063),
    (100, 0.95, 0.95, One, 1.926538850512315),
    (100, 0.99, 0.99, One, 2.849648117597919),
    (100, 0.999, 0.5, One, 3.100227146495577),
    (100, 0.9999, 0.5, One, 3.7311919098057333),
    (100, 0.9999, 0.9999, One, 5.064747316415692),
    (300, 0.5, 0.5, One, 0.0),
    (300, 0.5, 0.9999, One, 0.2174099261712758),
    (300, 0.75, 0.9, One, 0.7590926448650271),
    (300, 0.9, 0.95, One, 1.4169111139422714),
    (300, 0.95, 0.95, One, 1.799641930474128),
    (300, 0.99, 0.99, One, 2.60804545845495),
    (300, 0.999, 0.5, One, 3.093532669344885),
    (300, 0.9999, 0.5, One, 3.723036962554015),
    (300, 0.9999, 0.9999, One, 4.412045377606972),
    (1000, 0.5, 0.5, One, 0.0),
    (1000, 0.5, 0.9999, One, 0.11804357177824643),
    (1000, 0.75, 0.9, One, 0.7201435010751297),
    (1000, 0.9, 0.95, One, 1.353817471225263),
    (1000, 0.95, 0.95, One, 1.7272632696712737),
    (1000, 0.99, 0.99, One, 2.4745797060384955),
    (1000, 0.999, 0.5, One, 3.091219173036118),
    (1000, 0.9999, 0.5, One, 3.7202186865039137),
    (1000, 0.9999, 0.9999, One, 4.074873288531248),
    (10000, 0.5, 0.9999, One, 0.03720396007115605),
    (10000, 0.75, 0.9, One, 0.6887609754848131),
    (10000, 0.9, 0.95, One, 1.303949320039131),
    (10000, 0.95, 0.95, One, 1.670337590419526),
    (10000, 0.99, 0.99, One, 2.3717681837029936),
    (10000, 0.999, 0.5, One, 3.0903308683391195),
    (10000, 0.9999, 0.5, One, 3.719136554161203),
    (10000, 0.9999, 0.9999, One, 3.8260211973698337),
    (100000, 0.5, 0.9999, One, 0.011760998823305013),
    (100000, 0.75, 0.9, One, 0.6789868924788408),
    (100000, 0.9, 0.95, One, 1.288590853452739),
    (100000, 0.95, 0.95, One, 1.6528571890460575),
    (100000, 0.99, 0.99, One, 2.3405727268348846),
    (100000, 0.999, 0.5, One, 3.0902421611414237),
    (100000, 0.9999, 0.5, One, 3.7190284908142535),
    (100000, 0.9999, 0.9999, One, 3.7523377638068864),
    (2, 0.5, 0.5, Two, 1.2427213636035679),
    (2, 0.9, 0.95, Two, 31.092225599749874),
    (2, 0.95, 0.95, Two, 36.5192146120607),
    (3, 0.75, 0.9, Two, 4.134480763733478),
    (3, 0.99, 0.99, Two, 28.585695168916835),
    (3, 0.9999, 0.9999, Two, 422.37103908228005),
    (4, 0.5, 0.9999, Two, 18.960516707622144),
    (4, 0.999, 0.5, Two, 3.9784718818325895),
    (4, 0.9999, 0.5, Two, 4.673638335466032),
    (5, 0.1, 0.5, Two, 0.15122107442249494),
    (5, 0.5, 0.5, Two, 0.8082622673636243),
    (5, 0.9, 0.95, Two, 4.290604070686457),
    (5, 0.95, 0.95, Two, 5.0768745320594),
    (7, 0.75, 0.9, Two, 2.052685759639394),
    (7, 0.99, 0.99, Two, 7.190701310880392),
    (7, 0.9999, 0.9999, Two, 24.136764662787343),
    (10, 0.3, 0.9, Two, 0.6024882613852924),
    (10, 0.5, 0.9999, Two, 2.699280751537605),
    (10, 0.95, 0.95, Two, 3.3934294787126076),
    (10, 0.999, 0.5, Two, 3.543474268081425),
    (10, 0.9999, 0.5, Two, 4.179334367397475),
    (20, 0.5, 0.5, Two, 0.7035305517964616),
    (20, 0.9, 0.95, Two, 2.3187910746015326),
    (20, 0.95, 0.95, Two, 2.7603461784456993),
    (25, 0.99, 0.95, Two, 3.462148912059184),
    (35, 0.75, 0.9, Two, 1.3926677053386158),
    (35, 0.99, 0.99, Two, 3.6184878193728327),
    (35, 0.9999, 0.9999, Two, 6.8122026897832155),
    (50, 0.5, 0.9999, Two, 1.0609158877866727),
    (50, 0.95, 0.95, Two, 2.3815597421152948),
    (50, 0.999, 0.5, Two, 3.343341576305593),
    (50, 0.9999, 0.5, Two, 3.9520152031040694),
    (100, 0.5, 0.5, Two, 0.680143541904078),
    (100, 0.9, 0.95, Two, 1.8748075437924703),
    (100, 0.95, 0.95, Two, 2.2338820230442487),
    (100, 0.999, 0.99, Two, 3.9564579066069006),
    (300, 0.75, 0.9, Two, 1.216915922729487),
    (300, 0.9, 0.95, Two, 1.767222088994278),
    (300, 0.99, 0.99, Two, 2.8503243921580856),
    (300, 0.9999, 0.9999, Two, 4.580926412737119),
    (1000, 0.5, 0.9999, Two, 0.7354129833723538),
    (1000, 0.95, 0.95, Two, 2.036114277875938),
    (1000, 0.999, 0.5, Two, 3.2932619918686568),
    (1000, 0.9999, 0.5, Two, 3.893821902381688),
    (10000, 0.5, 0.5, Two, 0.6745459608584954),
    (10000, 0.9, 0.95, Two, 1.6643128987065792),
    (10000, 0.95, 0.95, Two, 1.9831511310948804),
    (10000, 0.99, 0.99, Two, 2.619012772885556),
    (10000, 0.9999, 0.5, Two, 3.8909159782661598),
    (100000, 0.5, 0.5, Two, 0.674495370975821),
    (100000, 0.75, 0.9, Two, 1.153663408259294),
    (100000, 0.99, 0.99, Two, 2.5893084937790722),
    (100000, 0.9999, 0.9999, Two, 3.9232046448131315),
];

fn relative(got: f64, want: f64) -> f64 {
    if want == 0.0 {
        got.abs()
    } else {
        ((got - want) / want).abs()
    }
}

#[test]
fn matches_the_independent_computation() {
    for &(n, p, g, sided, want) in ROWS {
        let got = tolerance_factor(n, p, g, sided).expect("a valid case");
        // Observed: about 3e-15 for both; the bound leaves room for another
        // libm.
        let tolerance = 1e-12;
        assert!(
            relative(got, want) <= tolerance,
            "n={n} P={p} γ={g} {sided:?}: got {got:e}, want {want:e}, relative {:e}",
            relative(got, want)
        );
    }
}

/// The commonly quoted values. The one-sided factors (2.911, 2.396 at 95 % /
/// 95 %) are exact to the digits printed. The two-sided ones (3.379, 2.752,
/// 2.379 at 95 % / 95 %) equal the Wald–Wolfowitz approximation to the digits
/// printed and are slightly *low*: the exact factor is larger, and the
/// confidence that the quoted factor reaches is below the 95 % it is quoted for
/// (a Monte Carlo of 4 million samples: 94.86 % at n = 10, 94.84 % at n = 20,
/// 94.91 % at n = 50; the factors here: 95.00 %, 95.00 %, 95.01 %).
#[test]
fn published_tables() {
    for (n, printed) in [(10, 2.911), (20, 2.396)] {
        let k = tolerance_factor(n, 0.95, 0.95, One).unwrap();
        assert!((k - printed).abs() < 5e-4, "one-sided n={n}: {k}");
    }
    for (n, printed) in [(10, 3.379), (20, 2.752), (50, 2.379)] {
        let k = tolerance_factor(n, 0.95, 0.95, Two).unwrap();
        assert!(k > printed, "two-sided n={n}: {k} vs {printed}");
        assert!(k < printed * 1.005, "two-sided n={n}: {k} vs {printed}");
    }
}

/// A Monte Carlo of the *definition*: the share of samples whose interval holds
/// at least `P` of the population is the confidence the factor was asked for.
/// (A small SplitMix64 + Box–Muller generator, so the test needs no new
/// dependency and does not change.)
#[test]
fn the_interval_has_the_requested_confidence() {
    struct Rng(u64);
    impl Rng {
        fn next_f64(&mut self) -> f64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
        }
        fn normal(&mut self) -> f64 {
            let (u1, u2) = (1.0 - self.next_f64(), self.next_f64());
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        }
    }
    let mut rng = Rng(20_261_110);
    let (n, p, g) = (10, 0.9, 0.9);
    let draws = 40_000;
    for sided in [One, Two] {
        let k = tolerance_factor(n, p, g, sided).unwrap();
        let mut hits = 0;
        for _ in 0..draws {
            let x: Vec<f64> = (0..n).map(|_| rng.normal()).collect();
            let mean = x.iter().sum::<f64>() / n as f64;
            let sd = (x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1) as f64).sqrt();
            let holds = if sided == One {
                // the lower bound leaves at least P above it
                special::standard_normal_sf(mean - k * sd) >= p
            } else {
                special::standard_normal_cdf(mean + k * sd)
                    - special::standard_normal_cdf(mean - k * sd)
                    >= p
            };
            hits += usize::from(holds);
        }
        let share = hits as f64 / draws as f64;
        let standard_error = (g * (1.0 - g) / draws as f64).sqrt();
        assert!(
            (share - g).abs() < 4.5 * standard_error,
            "{sided:?}: confidence {share} for a requested {g}"
        );
    }
}

/// With `n → ∞` the interval is the population's own: the factor tends to
/// `z_P` (one-sided) and `z_{(1+P)/2}` (two-sided).
#[test]
fn tends_to_the_normal_quantile() {
    for (p, g) in [(0.9, 0.95), (0.99, 0.99), (0.9999, 0.9)] {
        let z_one = special::inverse_normal_cdf(p);
        let z_two = special::inverse_normal_cdf(0.5 * (1.0 + p));
        let mut last = (f64::INFINITY, f64::INFINITY);
        for n in [1_000, 10_000, 100_000, 1_000_000] {
            let one = tolerance_factor(n, p, g, One).unwrap();
            let two = tolerance_factor(n, p, g, Two).unwrap();
            // from above, closing in
            assert!(one > z_one && two > z_two, "n={n}");
            assert!(one < last.0 && two < last.1, "n={n}");
            last = (one, two);
        }
        // the delta method: k − z is of the order of z_γ·√(1/n + z²/(2n))
        let z_g = special::inverse_normal_cdf(g).abs().max(1.0);
        assert!(last.0 - z_one < 3.0 * z_g * (1.0 + z_one * z_one / 2.0).sqrt() / 1e3);
        assert!(last.1 - z_two < 3.0 * z_g * (1.0 + z_two * z_two / 2.0).sqrt() / 1e3);
    }
}

#[test]
fn refuses_what_it_cannot_answer() {
    for sided in [One, Two] {
        assert!(tolerance_factor(10, 0.95, 0.95, sided).is_some());
        assert!(tolerance_factor(2, 0.95, 0.95, sided).is_some());
        assert!(tolerance_factor(1, 0.95, 0.95, sided).is_none());
        assert!(tolerance_factor(0, 0.95, 0.95, sided).is_none());
        for bad in [0.0, 1.0, -0.1, 1.5, f64::NAN, f64::INFINITY] {
            assert!(tolerance_factor(10, bad, 0.95, sided).is_none(), "{bad}");
            assert!(tolerance_factor(10, 0.95, bad, sided).is_none(), "{bad}");
        }
        let data = [1.0, 2.0, 3.0, 4.0];
        assert!(tolerance_interval_normal(&data, 0.95, 0.95, sided).is_some());
        assert!(tolerance_interval_normal(&data[..1], 0.95, 0.95, sided).is_none());
        assert!(tolerance_interval_normal(&[], 0.95, 0.95, sided).is_none());
        assert!(tolerance_interval_normal(&[5.0, 5.0, 5.0], 0.95, 0.95, sided).is_none());
        assert!(tolerance_interval_normal(&[1.0, f64::NAN, 3.0], 0.95, 0.95, sided).is_none());
        assert!(tolerance_interval_normal(&[1.0, f64::INFINITY, 3.0], 0.95, 0.95, sided).is_none());
        assert!(tolerance_interval_normal(&data, 1.0, 0.95, sided).is_none());
        assert!(tolerance_interval_normal(&data, 0.95, 0.0, sided).is_none());
    }
}

/// The interval's fields are the sample's statistics and the factor.
#[test]
fn the_interval_is_the_mean_plus_or_minus_k_s() {
    let data = [
        10.02, 9.98, 10.01, 10.0, 9.99, 10.03, 9.97, 10.01, 10.0, 9.99,
    ];
    for sided in [One, Two] {
        let t = tolerance_interval_normal(&data, 0.95, 0.95, sided).unwrap();
        let k = tolerance_factor(data.len(), 0.95, 0.95, sided).unwrap();
        assert_eq!(t.n, 10);
        assert_eq!(t.k, k);
        assert!((t.mean - 10.0).abs() < 1e-12);
        assert!((t.std_dev - 0.018_257_418_583_505_54).abs() < 1e-12);
        assert_eq!(t.lower, t.mean - k * t.std_dev);
        assert_eq!(t.upper, t.mean + k * t.std_dev);
    }
}

/// The one-sided factor at coverage ½ is the (central) t quantile over √n, so
/// at confidence ½ it is exactly 0.
#[test]
fn the_median_case_is_zero() {
    for n in [2, 5, 100, 100_000] {
        assert_eq!(tolerance_factor(n, 0.5, 0.5, One), Some(0.0));
    }
    // coverage ½, confidence γ: k = t_{γ, n−1}/√n
    for (n, g) in [(5, 0.95), (30, 0.99), (200, 0.9)] {
        let t = special::t_distribution_quantile(g, (n - 1) as f64);
        let k = tolerance_factor(n, 0.5, g, One).unwrap();
        assert!(relative(k, t / (n as f64).sqrt()) < 1e-12, "n={n}");
    }
}

/// The one-sided factor is `t'_{γ; n−1, z_P√n}/√n` (noncentral t); where the
/// library's series holds (`|δ| < 37`) its distribution function must return γ.
#[test]
fn the_one_sided_factor_inverts_the_noncentral_t() {
    for (n, p, g) in [
        (5, 0.9, 0.95),
        (10, 0.95, 0.95),
        (30, 0.99, 0.9),
        (100, 0.9, 0.99),
    ] {
        let k = tolerance_factor(n, p, g, One).unwrap();
        let sqrt_n = (n as f64).sqrt();
        let delta = special::inverse_normal_cdf(p) * sqrt_n;
        assert!(delta.abs() < 37.0);
        let confidence = special::noncentral_t_cdf(k * sqrt_n, (n - 1) as f64, delta);
        assert!((confidence - g).abs() < 1e-9, "n={n}: {confidence}");
    }
}

fn levels() -> impl Strategy<Value = (usize, f64, f64)> {
    (2_usize..120, 0.5_f64..0.9995, 0.55_f64..0.9995)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(24))]

    /// More observations narrow the interval.
    #[test]
    fn decreasing_in_n((n, p, g) in levels()) {
        for sided in [One, Two] {
            let a = tolerance_factor(n, p, g, sided).unwrap();
            let b = tolerance_factor(n + 1, p, g, sided).unwrap();
            prop_assert!(b < a, "{sided:?} n={n}: {a} then {b}");
        }
    }

    /// A larger proportion or a larger confidence needs a wider interval.
    #[test]
    fn increasing_in_coverage_and_confidence((n, p, g) in levels()) {
        for sided in [One, Two] {
            let base = tolerance_factor(n, p, g, sided).unwrap();
            let more_coverage = tolerance_factor(n, p + 0.5 * (1.0 - p), g, sided).unwrap();
            let more_confidence = tolerance_factor(n, p, g + 0.5 * (1.0 - g), sided).unwrap();
            prop_assert!(more_coverage > base, "{sided:?}: coverage");
            prop_assert!(more_confidence > base, "{sided:?}: confidence");
        }
    }

    /// An interval that holds `P` of the population is also a bound that does,
    /// so the two-sided factor is the larger at the same `P` and `γ`.
    #[test]
    fn two_sided_is_wider_than_one_sided((n, p, g) in levels()) {
        let one = tolerance_factor(n, p, g, One).unwrap();
        let two = tolerance_factor(n, p, g, Two).unwrap();
        prop_assert!(two > one, "n={n} P={p} γ={g}: {two} vs {one}");
    }

    /// The interval of `a·x + b` is `a·(interval of x) + b`; a negative `a`
    /// swaps the ends.
    #[test]
    fn scale_and_shift_equivariance(
        data in proptest::collection::vec(-1e3_f64..1e3, 3..30),
        scale in prop_oneof![0.01_f64..100.0, -100.0_f64..-0.01],
        shift in -1e4_f64..1e4,
    ) {
        for sided in [One, Two] {
            let Some(t) = tolerance_interval_normal(&data, 0.95, 0.9, sided) else {
                continue; // no variation
            };
            let moved: Vec<f64> = data.iter().map(|v| scale * v + shift).collect();
            let m = tolerance_interval_normal(&moved, 0.95, 0.9, sided).unwrap();
            prop_assert_eq!(m.k, t.k);
            let (lo, hi) = if scale > 0.0 {
                (scale * t.lower + shift, scale * t.upper + shift)
            } else {
                (scale * t.upper + shift, scale * t.lower + shift)
            };
            let width = (hi - lo).abs();
            prop_assert!((m.lower - lo).abs() <= 1e-9 * width.max(1.0), "{} vs {lo}", m.lower);
            prop_assert!((m.upper - hi).abs() <= 1e-9 * width.max(1.0), "{} vs {hi}", m.upper);
        }
    }
}
