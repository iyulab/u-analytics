//! Tolerance intervals for a normal population.
//!
//! A *(P, γ) tolerance interval* is computed from a sample and contains at
//! least a proportion `P` of the population with confidence `γ`. Unlike a
//! confidence interval of the mean it speaks about individual values, which is
//! what a specification limit is about: "95 % of the parts are inside, and we
//! are 95 % sure of that statement".
//!
//! For a sample of `n` observations from a normal population with mean `x̄` and
//! standard deviation `s`, the interval is `x̄ ± k·s` (two-sided) or one of the
//! bounds `x̄ − k·s` / `x̄ + k·s` (one-sided), where the factor `k` depends on
//! `n`, `P` and `γ` only.
//!
//! # Examples
//!
//! ```
//! use u_analytics::tolerance::{tolerance_factor, tolerance_interval_normal, Sided};
//!
//! // The factor alone: 95 % of the population, 95 % confidence, n = 10.
//! let two_sided = tolerance_factor(10, 0.95, 0.95, Sided::TwoSided).unwrap();
//! assert!((two_sided - 3.393429478712608).abs() < 1e-11);
//! let one_sided = tolerance_factor(10, 0.95, 0.95, Sided::OneSided).unwrap();
//! assert!((one_sided - 2.9109634130781664).abs() < 1e-11); // tables: 2.911
//!
//! // The interval of a sample.
//! let diameters = [10.02, 9.98, 10.01, 10.00, 9.99, 10.03, 9.97, 10.01, 10.00, 9.99];
//! let t = tolerance_interval_normal(&diameters, 0.95, 0.95, Sided::TwoSided).unwrap();
//! assert!((t.mean - 10.0).abs() < 1e-12);
//! assert!((t.lower - 9.938).abs() < 1e-3 && (t.upper - 10.062).abs() < 1e-3);
//! // a one-sided statement uses ONE bound: "at least 95 % of the parts are
//! // below `upper`, with 95 % confidence".
//! let t = tolerance_interval_normal(&diameters, 0.95, 0.95, Sided::OneSided).unwrap();
//! assert!((t.upper - 10.053).abs() < 1e-3);
//! ```
//!
//! # Algorithm
//!
//! The factor is the root `k` of the exact confidence equation, not a
//! closed-form approximation. Write `x̄ = μ + σ·Y/√n` with `Y ~ N(0, 1)` and
//! `s = σ·w` with `w² ~ χ²_ν / ν`, `ν = n − 1`, independent.
//!
//! - **One-sided.** The bound `x̄ − k·s` leaves at least `P` of the population
//!   above it exactly when `Y/√n ≤ k·w − z_P`, so
//!   `γ = ∫ f_ν(w) · Φ(√n·(k·w − z_P)) dw`, which is the distribution function
//!   of the noncentral `t` at `k√n` with non-centrality `z_P√n` (Owen 1968). It
//!   is integrated directly — the noncentral-`t` series loses its terms to
//!   underflow once `z_P√n` is beyond ≈ 37 — in the form of the *failure*
//!   probability `1 − γ`, so a confidence close to 1 keeps its digits.
//! - **Two-sided.** The interval `x̄ ± k·s` covers
//!   `Φ(a + k·w) − Φ(a − k·w)` of the population, `a = Y/√n`. For a given `w`
//!   the coverage reaches `P` exactly when `|a| ≤ A(k·w)`, where `A(ρ)` is the
//!   offset at which an interval of half-width `ρ` holds `P` (Newton on the
//!   normal distribution function; it exists only for `ρ` above `z_{(1+P)/2}`).
//!   Hence `γ = ∫ f_ν(w) · [2Φ(√n·A(k·w)) − 1] dw` — the condition of
//!   Eisenhart, Hastay and Wallis (1947) and Wald and Wolfowitz (1946), solved
//!   exactly rather than by their approximation (which fixes `a = 1/√n`) or
//!   Howe's (1969). `A` starts like a square root at `ρ = z_{(1+P)/2}`, so the
//!   integral is taken in `u = √(k·w − z_{(1+P)/2})`, where it is smooth. Taken
//!   in the other order the same condition is
//!   `γ = ∫ 2φ(Y) · Q_ν(ν·r(Y/√n)² / (2k²)) dY` with `Q_ν` the regularized upper
//!   incomplete gamma function; the implementation avoids that form because the
//!   incomplete gamma function loses digits at large `n`.
//!
//! The integrals use adaptive Gauss–Kronrod (7, 15) quadrature, broken at the
//! points where the integrand changes fastest (the mode of the chi density and
//! the location of the step of `Φ`), and `k` is found by Brent's method on a
//! bracket grown from a closed-form approximation (Howe's for the two-sided
//! factor, Natrella's for the one-sided one) — used only as a starting point,
//! never as a result.
//!
//! # Accuracy
//!
//! Against two independent 40-digit `mpmath` computations (the confidence
//! integral taken over the chi variable, and for the two-sided factor also over
//! the normal variable) and, for the one-sided factor, `scipy.stats.nct`, the
//! factor agrees to a relative `3e-15` (and to `1e-16` absolutely where `|k| <
//! 0.01`) for `n` from 2 to 100 000 and `P`, `γ` from 0.5 to
//! 0.9999 (`tests/tolerance_reference.rs`). Larger `n` (tried to 10⁷) and `P`,
//! `γ` up to `1 − 10⁻⁹` are computed without a refusal but are not compared.
//! A `P` or `γ` below 0.5 is computed by the same equations; the two-sided
//! digits are then limited by the difference of two nearby normal
//! probabilities (about `10⁻¹⁶/P`) and the quadrature is slower.
//!
//! # References
//!
//! - Eisenhart, C., Hastay, M. W. and Wallis, W. A. (1947). *Techniques of
//!   Statistical Analysis*, ch. 2. McGraw-Hill.
//! - Wald, A. and Wolfowitz, J. (1946). "Tolerance limits for a normal
//!   distribution". *Annals of Mathematical Statistics* 17(2), 208–215.
//! - Howe, W. G. (1969). "Two-sided tolerance limits for normal populations —
//!   some improvements". *J. American Statistical Association* 64, 610–620.
//! - Owen, D. B. (1968). "A survey of properties and applications of the
//!   noncentral t-distribution". *Technometrics* 10(3), 445–478.
//! - Odeh, R. E. and Owen, D. B. (1980). *Tables for Normal Tolerance Limits,
//!   Sampling Plans and Screening*. Marcel Dekker.
//! - NIST/SEMATECH e-Handbook of Statistical Methods, §7.2.6.3 "Tolerance
//!   intervals for a normal distribution".
//! - ISO 16269-6:2014, *Statistical interpretation of data — Part 6:
//!   Determination of statistical tolerance intervals*.

use u_numflow::special;
use u_numflow::stats;

/// Whether a tolerance interval has two ends or one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Sided {
    /// The symmetric interval `x̄ ± k·s` that contains at least `P` of the
    /// population with confidence `γ`.
    TwoSided,
    /// A single bound: at least `P` of the population lies above `x̄ − k·s`
    /// (a lower bound), or below `x̄ + k·s` (an upper bound), with confidence
    /// `γ`. The same factor serves both bounds; a statement uses **one** of
    /// them, not the pair.
    OneSided,
}

/// A normal tolerance interval computed from a sample.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ToleranceInterval {
    /// Sample mean `x̄`.
    pub mean: f64,
    /// Sample standard deviation `s` (divisor `n − 1`).
    pub std_dev: f64,
    /// Number of observations `n`.
    pub n: usize,
    /// The tolerance factor `k` of this `n`, coverage, confidence and sidedness.
    pub k: f64,
    /// `x̄ − k·s`: the lower end of the two-sided interval, or the lower bound
    /// of a one-sided statement.
    pub lower: f64,
    /// `x̄ + k·s`: the upper end of the two-sided interval, or the upper bound
    /// of a one-sided statement.
    pub upper: f64,
}

/// Relative tolerance of every quadrature.
const QUAD_TOL: f64 = 1e-13;
/// Upper limit of panels of one quadrature. It is reached when the rounding of
/// the integrand keeps the error estimate above `QUAD_TOL` (seen at large `n`);
/// the value is then as good as that rounding allows.
const MAX_PANELS: usize = 500;
/// `ln` of the density, relative to its peak, at which the chi density is cut
/// off (`e⁻⁷⁰ ≈ 4e-31`).
const DENSITY_CUTOFF: f64 = -70.0;

/// Nodes and weights of the Gauss–Kronrod (7, 15) pair on `[-1, 1]`.
const XGK: [f64; 8] = [
    0.991_455_371_120_812_6,
    0.949_107_912_342_758_5,
    0.864_864_423_359_769_1,
    0.741_531_185_599_394_4,
    0.586_087_235_467_691_1,
    0.405_845_151_377_397_2,
    0.207_784_955_007_898_47,
    0.0,
];
const WGK: [f64; 8] = [
    0.022_935_322_010_529_225,
    0.063_092_092_629_978_55,
    0.104_790_010_322_250_18,
    0.140_653_259_715_525_92,
    0.169_004_726_639_267_9,
    0.190_350_578_064_785_4,
    0.204_432_940_075_298_9,
    0.209_482_141_084_727_82,
];
/// Weights of the embedded 7-point Gauss rule (nodes `XGK[1]`, `[3]`, `[5]`, `0`).
const WG: [f64; 4] = [
    0.129_484_966_168_869_7,
    0.279_705_391_489_276_7,
    0.381_830_050_505_118_9,
    0.417_959_183_673_469_4,
];

/// One Gauss–Kronrod 15 panel on `[a, b]`: the integral and an error estimate
/// (QUADPACK's `qk15`).
fn gk15<F: FnMut(f64) -> f64>(f: &mut F, a: f64, b: f64) -> (f64, f64) {
    let centre = 0.5 * (a + b);
    let half = 0.5 * (b - a);
    let fc = f(centre);
    let mut gauss = fc * WG[3];
    let mut kronrod = fc * WGK[7];
    let mut abs_integral = kronrod.abs();
    let mut left = [0.0; 7];
    let mut right = [0.0; 7];
    for (j, (&x, &weight)) in XGK.iter().zip(&WGK).take(7).enumerate() {
        let dx = half * x;
        let (f1, f2) = (f(centre - dx), f(centre + dx));
        left[j] = f1;
        right[j] = f2;
        kronrod += weight * (f1 + f2);
        abs_integral += weight * (f1.abs() + f2.abs());
        if j % 2 == 1 {
            gauss += WG[j / 2] * (f1 + f2);
        }
    }
    let mean = 0.5 * kronrod;
    let mut deviation = WGK[7] * (fc - mean).abs();
    for ((&l, &r), &weight) in left.iter().zip(&right).zip(&WGK) {
        deviation += weight * ((l - mean).abs() + (r - mean).abs());
    }
    let h = half.abs();
    let value = kronrod * half;
    let (abs_integral, deviation) = (abs_integral * h, deviation * h);
    let mut error = ((kronrod - gauss) * half).abs();
    if deviation != 0.0 && error != 0.0 {
        error = deviation * (200.0 * error / deviation).powf(1.5).min(1.0);
    }
    error = error.max(10.0 * f64::EPSILON * abs_integral);
    (value, error)
}

/// A quadrature panel: its ends, its integral and the error estimate of it.
struct Panel {
    a: f64,
    b: f64,
    value: f64,
    error: f64,
}

/// `∫ f` over the span of the sorted `breaks`, by adaptive Gauss–Kronrod
/// bisection that always splits the panel with the largest error estimate
/// until the total error is below `QUAD_TOL` of the total.
fn integrate<F: FnMut(f64) -> f64>(f: &mut F, breaks: &[f64]) -> f64 {
    let mut panels: Vec<Panel> = Vec::with_capacity(64);
    for w in breaks.windows(2) {
        if w[1] > w[0] {
            let (value, error) = gk15(f, w[0], w[1]);
            panels.push(Panel {
                a: w[0],
                b: w[1],
                value,
                error,
            });
        }
    }
    loop {
        let total: f64 = panels.iter().map(|p| p.value).sum();
        let error: f64 = panels.iter().map(|p| p.error).sum();
        if !total.is_finite() || error <= QUAD_TOL * total.abs() || panels.len() >= MAX_PANELS {
            return total;
        }
        let worst = panels
            .iter()
            .enumerate()
            .max_by(|x, y| x.1.error.total_cmp(&y.1.error))
            .map(|(i, _)| i)
            .expect("panels is not empty: the breaks span an interval");
        let Panel { a, b, .. } = panels.swap_remove(worst);
        let mid = 0.5 * (a + b);
        if !(mid > a && mid < b) {
            // the panel cannot be split in floating point; its error is final
            return total;
        }
        for (lo, hi) in [(a, mid), (mid, b)] {
            let (value, error) = gk15(f, lo, hi);
            panels.push(Panel {
                a: lo,
                b: hi,
                value,
                error,
            });
        }
    }
}

/// `Φ⁻¹(p)` for `p ∈ (0, 1)`, keeping the digits of the upper tail.
fn normal_quantile(p: f64) -> f64 {
    if p >= 0.5 {
        special::inverse_normal_sf(1.0 - p)
    } else {
        special::inverse_normal_cdf(p)
    }
}

/// `Φ⁻¹((1 + p) / 2)`: the half-width of the central interval of mass `p`.
fn central_quantile(p: f64) -> f64 {
    if p >= 0.5 {
        special::inverse_normal_sf((1.0 - p) / 2.0)
    } else {
        special::inverse_normal_cdf((1.0 + p) / 2.0)
    }
}

/// The density of `w = s/σ`, `w² ~ χ²_ν/ν`, up to a constant: `w^(ν−1)·e^(−νw²/2)`.
/// The constant is never formed — the density is divided by its own integral —
/// so `ln Γ(ν/2)` and its rounding at large `ν` do not enter.
struct ScaledChi {
    nu: f64,
    /// The peak, `√((ν − 1)/ν)`.
    mode: f64,
    /// The support kept: the density is below `e⁻⁷⁰` of its peak outside it.
    lo: f64,
    hi: f64,
}

impl ScaledChi {
    fn new(nu: f64) -> ScaledChi {
        let mut chi = ScaledChi {
            nu,
            mode: ((nu - 1.0) / nu).sqrt(),
            lo: 0.0,
            hi: 0.0,
        };
        // right of the peak the density falls monotonically
        let mut step = 1.0 / (2.0 * nu).sqrt();
        while chi.ln_relative(chi.mode + step) > DENSITY_CUTOFF {
            step *= 2.0;
        }
        let (mut inside, mut outside) = (chi.mode + 0.5 * step, chi.mode + step);
        for _ in 0..100 {
            let mid = 0.5 * (inside + outside);
            if chi.ln_relative(mid) > DENSITY_CUTOFF {
                inside = mid;
            } else {
                outside = mid;
            }
        }
        chi.hi = outside;
        if nu > 1.0 {
            // left of the peak, geometrically toward 0 where the density vanishes
            let mut outside = chi.mode;
            let mut inside = chi.mode;
            while chi.ln_relative(outside) > DENSITY_CUTOFF && outside > 0.0 {
                inside = outside;
                outside *= 0.5;
            }
            for _ in 0..100 {
                let mid = 0.5 * (inside + outside);
                if chi.ln_relative(mid) > DENSITY_CUTOFF {
                    inside = mid;
                } else {
                    outside = mid;
                }
            }
            chi.lo = outside;
        }
        chi
    }

    /// `ln(density(w)/density(mode))`, `−∞` where the density is zero.
    ///
    /// With `u = w/mode` and `t = u − 1`, `ln f(w) − ln f(mode) =
    /// (ν − 1)·(ln u − (u² − 1)/2) = (ν − 1)·(ln(1 + t) − t − t²/2)`. The
    /// bracket is `−t² + t³/3 − t⁴/4 + …` and is summed as that series for small
    /// `t`: at large `ν` the three terms of the direct form cancel to the
    /// size of `t²`, and the product with `ν − 1` would turn that into a
    /// relative error of the density of `ν·ε·t`.
    fn ln_relative(&self, w: f64) -> f64 {
        if self.nu == 1.0 {
            // the half-normal, whose peak is at 0
            return -0.5 * w * w;
        }
        if w <= 0.0 {
            return f64::NEG_INFINITY;
        }
        let t = w / self.mode - 1.0;
        let bracket = if t.abs() < 0.25 {
            let mut sum = -t * t;
            let mut power = t * t;
            for m in 3..80 {
                power *= t;
                let term = power / f64::from(m);
                sum += if m % 2 == 1 { term } else { -term };
            }
            sum
        } else {
            t.ln_1p() - t - 0.5 * t * t
        };
        (self.nu - 1.0) * bracket
    }

    fn density(&self, w: f64) -> f64 {
        if w < self.lo || w > self.hi {
            0.0
        } else {
            self.ln_relative(w).exp()
        }
    }

    /// Breakpoints that resolve the peak of the density, sorted, inside the support.
    fn breaks(&self) -> Vec<f64> {
        let sigma = 1.0 / (2.0 * self.nu).sqrt();
        let mut points = vec![self.lo, self.hi, self.mode];
        for c in [-8.0, -4.0, -2.0, -1.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0] {
            points.push(self.mode + c * sigma);
        }
        self.sorted_inside(points)
    }

    fn sorted_inside(&self, mut points: Vec<f64>) -> Vec<f64> {
        points.retain(|x| *x >= self.lo && *x <= self.hi);
        points.sort_by(f64::total_cmp);
        points.dedup();
        points
    }
}

/// The root `k` of an increasing function `h`, bracketed from `seed`.
///
/// The bracket grows geometrically (`multiplicative`, for a positive factor) or
/// by steps of the size of the answer plus `floor`, doubling until `h` changes
/// sign; then Brent's method.
fn solve_increasing<F: FnMut(f64) -> f64>(
    h: &mut F,
    seed: f64,
    multiplicative: bool,
    floor: f64,
) -> Option<f64> {
    let f_seed = h(seed);
    if !f_seed.is_finite() {
        return None;
    }
    if f_seed == 0.0 {
        return Some(seed);
    }
    let up = f_seed < 0.0;
    let mut step = 0.05;
    let (mut a, mut fa) = (seed, f_seed);
    let (b, fb) = loop {
        let candidate = match (multiplicative, up) {
            (true, true) => seed * (1.0 + step),
            (true, false) => seed / (1.0 + step),
            (false, true) => seed + step * (seed.abs() + floor),
            (false, false) => seed - step * (seed.abs() + floor),
        };
        let fc = h(candidate);
        if !fc.is_finite() || step > 1e12 {
            return None;
        }
        if (fc >= 0.0) == up {
            break (candidate, fc);
        }
        a = candidate;
        fa = fc;
        step *= 2.0;
    };
    // orient so that `a` is the end where h < 0 and `b` where h >= 0
    let (a, fa, b, fb) = if up { (a, fa, b, fb) } else { (b, fb, a, fa) };
    brent(h, a, fa, b, fb)
}

/// Brent's method on `[a, b]` with `h(a) < 0 <= h(b)` (or the ends swapped).
fn brent<F: FnMut(f64) -> f64>(h: &mut F, a: f64, fa: f64, b: f64, fb: f64) -> Option<f64> {
    let (mut a, mut fa, mut b, mut fb) = (a, fa, b, fb);
    let (mut c, mut fc) = (b, fb);
    let mut d = b - a;
    let mut e = d;
    for _ in 0..200 {
        if (fb > 0.0) == (fc > 0.0) && fb != 0.0 {
            c = a;
            fc = fa;
            d = b - a;
            e = d;
        }
        if fc.abs() < fb.abs() {
            a = b;
            b = c;
            c = a;
            fa = fb;
            fb = fc;
            fc = fa;
        }
        let tol = 2.0 * f64::EPSILON * b.abs() + 5e-15 * b.abs();
        let xm = 0.5 * (c - b);
        if xm.abs() <= tol || fb == 0.0 {
            return Some(b);
        }
        if e.abs() >= tol && fa.abs() > fb.abs() {
            let s = fb / fa;
            let (mut p, mut q);
            if a == c {
                p = 2.0 * xm * s;
                q = 1.0 - s;
            } else {
                let q0 = fa / fc;
                let r = fb / fc;
                p = s * (2.0 * xm * q0 * (q0 - r) - (b - a) * (r - 1.0));
                q = (q0 - 1.0) * (r - 1.0) * (s - 1.0);
            }
            if p > 0.0 {
                q = -q;
            }
            p = p.abs();
            if 2.0 * p < (3.0 * xm * q - (tol * q).abs()).min((e * q).abs()) {
                e = d;
                d = p / q;
            } else {
                d = xm;
                e = d;
            }
        } else {
            d = xm;
            e = d;
        }
        a = b;
        fa = fb;
        b += if d.abs() > tol { d } else { tol.copysign(xm) };
        fb = h(b);
        if !fb.is_finite() {
            return None;
        }
    }
    Some(b)
}

/// Offsets, in widths of the step, at which the integration is broken around
/// the step of a normal distribution function.
const STEP_OFFSETS: [f64; 13] = [
    -32.0, -16.0, -8.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0,
];

/// The one-sided factor: the `k` with
/// `γ = ∫ f_ν(w)·Φ(√n·(k·w − z_P)) dw`.
fn one_sided_factor(n: usize, coverage: f64, confidence: f64) -> Option<f64> {
    let n_f = n as f64;
    let sqrt_n = n_f.sqrt();
    let z_p = normal_quantile(coverage);
    let chi = ScaledChi::new(n_f - 1.0);
    let density_breaks = chi.breaks();
    let norm = integrate(&mut |w| chi.density(w), &density_breaks);
    // The smaller of γ and 1 − γ is formed directly, as a tail, so a
    // confidence near either end keeps its digits.
    let failure_form = confidence >= 0.5;
    let target = if failure_form {
        1.0 - confidence
    } else {
        confidence
    };
    let mut residual = |k: f64| -> f64 {
        let mut points = density_breaks.clone();
        if k != 0.0 {
            let (step_at, width) = (z_p / k, 1.0 / (sqrt_n * k.abs()));
            points.extend(STEP_OFFSETS.iter().map(|c| step_at + c * width));
        }
        let points = chi.sorted_inside(points);
        let value = integrate(
            &mut |w| {
                let x = sqrt_n * (k * w - z_p);
                let tail = if failure_form {
                    special::standard_normal_sf(x)
                } else {
                    special::standard_normal_cdf(x)
                };
                chi.density(w) * tail
            },
            &points,
        ) / norm;
        // increasing in k in both forms
        if failure_form {
            target - value
        } else {
            value - target
        }
    };
    solve_increasing(
        &mut residual,
        one_sided_seed(n_f, z_p, confidence),
        false,
        1.0 / sqrt_n,
    )
}

/// A starting point for the one-sided factor: the closed-form approximation
/// (Natrella; the NIST handbook's), or the delta-method value where it does
/// not exist. Only the bracket search starts here.
fn one_sided_seed(n: f64, z_p: f64, confidence: f64) -> f64 {
    let z_g = normal_quantile(confidence);
    let nu = n - 1.0;
    let a = 1.0 - z_g * z_g / (2.0 * nu);
    let b = z_p * z_p - z_g * z_g / n;
    let disc = z_p * z_p - a * b;
    if a > 0.1 && disc >= 0.0 {
        (z_p + disc.sqrt()) / a
    } else {
        z_p + z_g * (1.0 / n + z_p * z_p / (2.0 * nu)).sqrt()
    }
}

/// The offset `a ≥ 0` at which the interval `[a − ρ, a + ρ]` holds exactly a
/// proportion `p` of the standard normal distribution, for `ρ > z`, where
/// `z = Φ⁻¹((1 + p)/2)` is the half-width at which the centred interval does.
///
/// Safeguarded Newton on the mass *outside* the interval (for `p ≥ ½`, so a
/// coverage near 1 keeps its digits) or inside it. The mass outside increases
/// with `a`; `a ∈ [ρ − z, ρ + z]` because an interval holding `[−z, z]` holds
/// at least `p`.
fn centre_offset(rho: f64, p: f64, z: f64) -> f64 {
    let miss_form = p >= 0.5;
    // Increasing in `a`, and its derivative (non-negative for `a ≥ 0`).
    let evaluate = |a: f64| -> (f64, f64) {
        let slope = special::standard_normal_pdf(a - rho) - special::standard_normal_pdf(a + rho);
        let value = if miss_form {
            special::standard_normal_sf(a + rho) + special::standard_normal_cdf(a - rho) - (1.0 - p)
        } else {
            p - (special::standard_normal_cdf(a + rho) - special::standard_normal_cdf(a - rho))
        };
        (value, slope)
    };
    let mut lo = (rho - z).max(0.0);
    let mut hi = rho + z;
    for _ in 0..100 {
        if evaluate(hi).0 >= 0.0 {
            break;
        }
        hi = 2.0 * hi + 1.0;
    }
    // near the start, a ≈ √(2(ρ − z)/z)
    let mut a = (2.0 * (rho - z) / z).sqrt().clamp(lo, hi);
    for _ in 0..100 {
        let (value, slope) = evaluate(a);
        if value < 0.0 {
            lo = a;
        } else {
            hi = a;
        }
        let newton = a - value / slope;
        let next = if newton > lo && newton < hi {
            newton
        } else {
            0.5 * (lo + hi)
        };
        if (next - a).abs() <= 2e-15 * next.abs() || hi - lo <= 4e-16 * hi {
            return next;
        }
        a = next;
    }
    a
}

/// The two-sided factor: the `k` with
/// `γ = ∫ f_ν(w)·[2Φ(√n·A(k·w)) − 1] dw`, where `A(ρ)` is [`centre_offset`] and
/// the integrand is 0 for `k·w ≤ z`.
///
/// This is the integral of the one-sided kind taken over the chi variable with
/// the normal one done in closed form — the order that needs no incomplete
/// gamma function, so it keeps its digits at any `n`. The coverage of the
/// interval starts like a square root at `k·w = z`; the integral is taken in
/// `u = √(k·w − z)`, in which it is smooth.
fn two_sided_factor(n: usize, coverage: f64, confidence: f64) -> Option<f64> {
    let n_f = n as f64;
    let sqrt_n = n_f.sqrt();
    let z = central_quantile(coverage);
    let chi = ScaledChi::new(n_f - 1.0);
    let density_breaks = chi.breaks();
    let norm = integrate(&mut |w| chi.density(w), &density_breaks);
    // Near `u = 0` the offset is `A ≈ u·√(2/z)`, so `Φ(√n·A)` rises over this width of `u`.
    let rise = (z / (2.0 * n_f)).sqrt();
    // The smaller of γ and 1 − γ is formed directly, so a confidence near
    // either end keeps its digits. The failure includes all the mass of `w`
    // too small for the interval to reach the coverage at all.
    let failure_form = confidence >= 0.5;
    let target = if failure_form {
        1.0 - confidence
    } else {
        confidence
    };
    let mut residual = |k: f64| -> f64 {
        let w_c = z / k;
        let mut below = 0.0;
        if failure_form {
            let mut points = density_breaks.clone();
            points.retain(|w| *w < w_c);
            points.push(w_c.min(chi.hi));
            below = integrate(&mut |w| chi.density(w), &points);
        }
        let mut above = 0.0;
        let u_hi_squared = k * chi.hi - z;
        if u_hi_squared > 0.0 {
            let u_lo = (k * chi.lo - z).max(0.0).sqrt();
            let u_hi = u_hi_squared.sqrt();
            let mut points = vec![u_lo, u_hi];
            points.extend([0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0].map(|c| c * rise));
            points.extend(
                density_breaks
                    .iter()
                    .filter(|w| k * **w > z)
                    .map(|w| (k * w - z).sqrt()),
            );
            points.retain(|u| *u >= u_lo && *u <= u_hi);
            points.sort_by(f64::total_cmp);
            points.dedup();
            above = integrate(
                &mut |u| {
                    let rho = z + u * u;
                    let x = sqrt_n * centre_offset(rho, coverage, z);
                    let share = if failure_form {
                        2.0 * special::standard_normal_sf(x)
                    } else {
                        special::erf(x / std::f64::consts::SQRT_2)
                    };
                    chi.density(rho / k) * share * (2.0 * u / k)
                },
                &points,
            );
        }
        let value = (below + above) / norm;
        if failure_form {
            target - value
        } else {
            value - target
        }
    };
    solve_increasing(&mut residual, two_sided_seed(n_f, z, confidence), true, 0.0)
}

/// Howe's (1969) approximation `z·√(ν(1 + 1/n)/χ²_{1−γ,ν})`, used only to start
/// the bracket search of the exact factor.
fn two_sided_seed(n: f64, z: f64, confidence: f64) -> f64 {
    let nu = n - 1.0;
    let chi = special::chi_squared_quantile(1.0 - confidence, nu);
    let seed = z * (nu * (1.0 + 1.0 / n) / chi).sqrt();
    if seed.is_finite() && seed > 0.0 {
        seed
    } else {
        z * (1.0 + 1.0 / n.sqrt())
    }
}

/// The tolerance factor `k` of a normal population.
///
/// With `n` observations, mean `x̄` and standard deviation `s`, the interval
/// `x̄ ± k·s` ([`Sided::TwoSided`]) or the bound `x̄ − k·s` / `x̄ + k·s`
/// ([`Sided::OneSided`]) contains at least a proportion `coverage` of the
/// population with probability `confidence`.
///
/// The factor is exact (see the [module documentation](self) for the equations
/// and the accuracy), not Howe's approximation: it is the root of the
/// confidence integral. It decreases in `n` toward `z_P` (one-sided) or
/// `z_{(1+P)/2}` (two-sided) for a `confidence` of ½ or more — below ½ it
/// approaches them from below and increases in `n` — and it increases in
/// `coverage` and `confidence`. For a coverage below ½ the one-sided factor can
/// be negative: the bound then lies on the far side of the mean.
///
/// # Published tables
///
/// The one-sided factors are the commonly quoted ones (2.911 for `n = 10`, 2.396
/// for `n = 20` at 95 % / 95 %). The two-sided factors commonly quoted for
/// 95 % / 95 % (3.379, 2.752 and 2.379 for `n = 10`, 20 and 50) are, to the
/// digits printed, the Wald–Wolfowitz approximation (`r·√(ν/χ²_{1−γ,ν})`, with
/// `r` the half-width at which an interval centred at `1/√n` holds `P`), which
/// is slightly low: the exact factors are 3.3934, 2.7603 and 2.3816, and a
/// Monte Carlo of four million samples gives the quoted factors a confidence of
/// 94.86 %, 94.84 % and 94.91 %, against 95.00 %, 95.00 % and 95.01 % for these.
///
/// # Returns
///
/// `None` if `n < 2`, or `coverage` or `confidence` is not in `(0, 1)`.
///
/// # Examples
///
/// ```
/// use u_analytics::tolerance::{tolerance_factor, Sided};
///
/// // 95 % of the population, 95 % confidence, n = 20.
/// let k = tolerance_factor(20, 0.95, 0.95, Sided::TwoSided).unwrap();
/// assert!((k - 2.7603461784456993).abs() < 1e-11);
/// assert!(tolerance_factor(1, 0.95, 0.95, Sided::TwoSided).is_none());
/// ```
pub fn tolerance_factor(n: usize, coverage: f64, confidence: f64, sided: Sided) -> Option<f64> {
    let open_unit = |p: f64| p.is_finite() && p > 0.0 && p < 1.0;
    if n < 2 || !open_unit(coverage) || !open_unit(confidence) {
        return None;
    }
    let k = match sided {
        Sided::TwoSided => two_sided_factor(n, coverage, confidence),
        Sided::OneSided => one_sided_factor(n, coverage, confidence),
    }?;
    k.is_finite().then_some(k)
}

/// The normal tolerance interval of a sample: `x̄ ± k·s`, with the factor of
/// [`tolerance_factor`].
///
/// For [`Sided::OneSided`] both `lower = x̄ − k·s` and `upper = x̄ + k·s` are
/// returned, but a one-sided statement uses **one** of them: "at least
/// `coverage` of the population is above `lower`" or "… is below `upper`", each
/// with `confidence`. Quoting both as an interval is the two-sided statement,
/// with the larger two-sided `k`.
///
/// The interval is valid for a sample drawn from a normal population; it is
/// not robust to departures from normality, which a tolerance interval of a
/// high proportion probes in the tails.
///
/// # Returns
///
/// `None` if there are fewer than 2 observations, a value is not finite, the
/// sample has no variation, or `coverage` or `confidence` is not in `(0, 1)`.
pub fn tolerance_interval_normal(
    data: &[f64],
    coverage: f64,
    confidence: f64,
    sided: Sided,
) -> Option<ToleranceInterval> {
    if data.len() < 2 || data.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let mean = stats::mean(data)?;
    let std_dev = stats::std_dev(data)?;
    if !(std_dev.is_finite() && std_dev > 0.0) {
        return None;
    }
    let k = tolerance_factor(data.len(), coverage, confidence, sided)?;
    Some(ToleranceInterval {
        mean,
        std_dev,
        n: data.len(),
        k,
        lower: mean - k * std_dev,
        upper: mean + k * std_dev,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_kronrod_rule_is_exact_for_polynomials_of_degree_22() {
        for degree in 0..=22_i32 {
            let mut f = |x: f64| x.powi(degree);
            let (value, _) = gk15(&mut f, -1.0, 1.0);
            let exact = if degree % 2 == 1 {
                0.0
            } else {
                2.0 / f64::from(degree + 1)
            };
            assert!((value - exact).abs() < 1e-15, "degree {degree}: {value}");
        }
        // the embedded Gauss rule is exact to degree 13
        let weights: f64 = 2.0 * (WG[0] + WG[1] + WG[2]) + WG[3];
        assert!((weights - 2.0).abs() < 1e-15);
        let weights: f64 = 2.0 * WGK[..7].iter().sum::<f64>() + WGK[7];
        assert!((weights - 2.0).abs() < 1e-15);
    }

    #[test]
    fn quadrature_resolves_a_narrow_peak() {
        let mut f = |x: f64| (-0.5 * ((x - 0.3) / 1e-3).powi(2)).exp();
        // a panel that no node falls into would see nothing, which is why the callers
        // break the range at the places the integrand changes quickly
        let got = integrate(&mut f, &[-1.0, 0.28, 0.32, 1.0]);
        let want = 1e-3 * (2.0 * std::f64::consts::PI).sqrt();
        assert!(((got - want) / want).abs() < 1e-12, "{got} vs {want}");
    }

    #[test]
    fn the_chi_density_integrates_to_the_closed_form() {
        // the peak-relative integral tends to √(π/ν) (a normal of variance 1/(2ν))
        let chi = ScaledChi::new(1.0e5 - 1.0);
        let got = integrate(&mut |w| chi.density(w), &chi.breaks());
        let want = (std::f64::consts::PI / (1.0e5 - 1.0)).sqrt();
        assert!(((got - want) / want).abs() < 1e-4, "{got} vs {want}");
        // ν = 1: the half-normal
        let chi = ScaledChi::new(1.0);
        let got = integrate(&mut |w| chi.density(w), &chi.breaks());
        let want = (0.5 * std::f64::consts::PI).sqrt();
        assert!(((got - want) / want).abs() < 1e-12);
        // ν = 2: ∫ w e^(−w²) dw = ½, relative to the peak value
        // (w·e^(−w²) at w = 1/√2 is e^(−½)/√2)
        let chi = ScaledChi::new(2.0);
        let peak = std::f64::consts::FRAC_1_SQRT_2 * (-0.5_f64).exp();
        let got = integrate(&mut |w| chi.density(w), &chi.breaks());
        assert!(((got - 0.5 / peak) / (0.5 / peak)).abs() < 1e-12);
    }

    #[test]
    fn the_centre_offset_holds_the_requested_mass() {
        for p in [0.3, 0.5, 0.9, 0.95, 0.9999] {
            let z = central_quantile(p);
            for s in [1e-4, 0.1, 1.0, 5.0, 40.0] {
                let rho = z + s;
                let a = centre_offset(rho, p, z);
                let mass =
                    special::standard_normal_cdf(a + rho) - special::standard_normal_cdf(a - rho);
                assert!((mass - p).abs() < 1e-13, "p {p} rho {rho}: {mass}");
            }
        }
    }

    #[test]
    fn brent_finds_a_root_of_an_increasing_function() {
        let mut h = |x: f64| x * x * x - 2.0;
        let root = solve_increasing(&mut h, 5.0, true, 0.0).unwrap();
        assert!((root - 2.0_f64.cbrt()).abs() < 1e-13, "{root}");
        let root = solve_increasing(&mut h, 0.1, false, 1.0).unwrap();
        assert!((root - 2.0_f64.cbrt()).abs() < 1e-13, "{root}");
    }
}
