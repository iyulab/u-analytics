//! Time of PELT on long series without changes — the worst case of
//! inequality pruning. Not part of the regular run:
//! `cargo test --release --test pelt_scale -- --ignored --nocapture`.
//! Functional pruning (one or two channels) should stay near-linear; three
//! channels or more fall back to inequality pruning, quadratic here.
//! (`ROWS=…` sets the length, default 50 000).

use std::time::Instant;
use u_analytics::detection::{CostFunction, Pelt, Penalty};

/// Seeded standard normal noise (SplitMix64 + Box-Muller).
fn noise(seed: u64, n: usize) -> Vec<f64> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1B5_4A32_D192_ED03;
    let mut uniform = move || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    };
    (0..n)
        .map(|_| {
            let u1 = uniform().max(1e-300);
            let u2 = uniform();
            (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
        })
        .collect()
}

fn rows() -> usize {
    std::env::var("ROWS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(50_000)
}

#[test]
#[ignore = "timing — run by hand"]
fn l2_without_changes_by_channel_count() {
    let n = rows();
    let pelt = Pelt::new(CostFunction::L2, Penalty::Bic).expect("valid");
    for p in [1usize, 2, 3, 4] {
        let cols: Vec<Vec<f64>> = (1..=p as u64).map(|s| noise(s, n)).collect();
        let refs: Vec<&[f64]> = cols.iter().map(Vec::as_slice).collect();
        let start = Instant::now();
        let r = pelt.detect_multi(&refs).expect("same length");
        println!(
            "p {p} n {n}: {:?} ({} changes)",
            start.elapsed(),
            r.changepoints.len()
        );
    }
}
