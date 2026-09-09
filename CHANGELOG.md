# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Maintained from 0.5.0 onward; earlier entries list release dates only (see git history).

## [Unreleased]

## [0.8.0]

### Changed

- **Breaking:** the WebAssembly `process_capability` binding takes an object
  instead of three positional arguments:
  `{ data, usl?, lsl?, sigma_within?, target? }`. The old signature
  (`data, usl, lsl`) could not express three things the crate already supports.
  - **`sigma_within` is now reachable.** The binding always called
    `ProcessCapability::compute_overall`, which uses the long-term sigma for
    both index families — so `cp === pp` and `cpk === ppk` held for every
    input, and the returned `std_dev_within` field carried the *overall*
    standard deviation. Short-term sigma is estimated from a control chart and
    cannot be derived from a flat measurement vector, so no caller could
    recover it. It is now an input, and the response carries `sigma_source`
    (`"within"` | `"overall"`) saying which family was actually computed. When
    it is omitted, `std_dev_within`, `cp`, `cpk`, `cpu`, `cpl` and `cpm` come
    back `null` rather than repeating the long-term numbers under short-term
    names.
  - **One-sided specifications are now expressible.** `usl` and `lsl` are each
    optional (at least one required), matching `ProcessCapability::new`. The
    previous signature took two bare numbers, so "no lower limit" could not be
    stated — while the doc comment already promised `null` for inapplicable
    one-sided indices.
  - **`target` is now reachable**, so Cpm is computed against a declared target
    instead of always against the specification midpoint. Against the wrong
    target Cpm is not a less precise index; it is a different quantity.
- `xbar_r_chart` additionally returns `sigma_hat` (`R-bar / d2`). Feeding it
  back as `process_capability`'s `sigma_within` is what makes the two bindings
  compose instead of each being independently incomplete; without it a
  JavaScript caller has to reimplement the d2 table.

### Added

- `spc::XBarRChart::sigma_hat` and `spc::XBarSChart::sigma_hat` — the
  within-subgroup sigma implied by the chart (`R-bar / d2`, `S-bar / c4`). The
  charts already held every input; the estimate was simply not exposed.


## [0.7.0] - 2026-09-07

### Fixed

- **`spc` module now re-exports `AttributeChartPoint`.** `PChart::points()`,
  `NPChart::points()`, `CChart::points()`, and `UChart::points()` all already
  returned `&[AttributeChartPoint]` publicly, but the type itself wasn't
  re-exported — a consumer could get a slice of this type back but had no
  way to name it (e.g. to write a function taking `&[AttributeChartPoint]`
  as a parameter). Purely additive; no behavior change.

### Changed

- **`rand` is now 0.10** and **`getrandom` 0.4** on WebAssembly targets. This
  crate does not name `rand` types in its public signatures, so the change is
  internal and the API is unaffected. The
  `RUSTFLAGS --cfg getrandom_backend="wasm_js"` that `getrandom` 0.3 required is
  no longer needed.
- **`u-numflow` is now required at 0.4** (previously 0.3), following that crate's
  own `rand` 0.10 break.
- **The minimum supported Rust version is now declared as 1.85** and is verified
  by building on that exact toolchain; 1.84 and below fail. The crate previously
  declared no `rust-version` at all.

## [0.6.3] - 2026-07-15

### Fixed

- **`gage_rr_anova` — pooled repeatability consistency.** When the operator×part
  interaction is pooled into error (AIAG p > 0.25), the repeatability variance
  component now uses the pooled error MS like the operator/part components,
  instead of leaking the un-pooled raw `MS_error`. Previously this internal
  inconsistency inflated `GRR` / `%GRR` and could tip the AIAG acceptability grade
  near threshold. The returned `anova_table` is now consistent with the pooled
  model too: it reports the pooled error row (so the Part/Operator `f_value` is
  reproducible from the table's own MS values) and drops the folded-in interaction
  row, and the row degrees of freedom sum to the total.
- **`gage_rr_anova` — degenerate F-denominator.** A mean-square denominator that
  collapses to floating-point noise (e.g. every trial within each cell identical)
  now yields `f_value`/`p_value` = `None` via a sign-independent *relative*
  degeneracy floor, instead of a spurious ~1e12 finite F (rendered as "very
  significant") or a result that flipped between a huge value and `None` depending
  on the sign of the rounding noise.
- **`anderson_darling_test` / `anderson_darling_normality` — large-n p-value
  overflow.** For clearly non-normal data at large n the p-value no longer
  overflows to exactly `1.0` ("perfectly normal") while the A² statistic is large
  and still growing. The upper-tail approximation's evaluation point is now clamped
  to its polynomial vertex, keeping the p-value monotonically non-increasing in A².

## [0.6.2] - 2026-07-05

### Fixed

- npm: expose the `./package.json` subpath in the `exports` map so tools
  that `require('<pkg>/package.json')` (license scanners, version
  reporters) keep working alongside the conditional exports introduced in
  the previous release (`ERR_PACKAGE_PATH_NOT_EXPORTED`).

## [0.6.1] - 2026-07-05

### Fixed

- **npm packaging — Node-compatible entry.** The npm package previously
  shipped only the wasm-bindgen *bundler*-target output, whose static
  `.wasm` import fails on Node's CJS path (`tsx`/`ts-node` in non-ESM
  packages) with an opaque `SyntaxError: Invalid or unexpected token`.
  The package now additionally ships the *nodejs*-target CJS glue under
  `node/` and routes Node consumers to it via a conditional `exports`
  map (`node` → CJS with filesystem wasm loading, `default` → bundler
  ESM). `require()`, native ESM `import`, and CJS TS runners all work
  without loader hooks. A pre-publish smoke test (CJS `require` + ESM
  `import`) now guards this path in CI. Rust API unchanged.

### Changed

- `u-numflow` dependency `^0.2` → `^0.3` (compatible; 0.3.0 publishes the
  previously-unreleased `wasm` feature and input-validation hardening —
  no API used by this crate changed).

## [0.6.0] - 2026-06-12

### Changed — BREAKING (WASM)

- WASM input objects (`detect_changepoints`, `detect_changepoints_multi`,
  `gage_rr_xbar_r`, `gage_rr_anova`, `percentile_capability`) now **reject
  unknown keys** with an explicit `unknown field` error instead of silently
  ignoring them (`serde(deny_unknown_fields)`, enforced through a
  `serde_json::Value` round-trip at the JS boundary). Typos and unsupported
  options previously failed silently; remove any extra keys when upgrading.

## [0.5.0] - 2026-06-11

### Added

- `distribution::BinMethod::Fixed(usize)` — caller-specified histogram bin
  count. `histogram_bins` honors the given `k` exactly (the automatic rules'
  minimum-2 floor does not apply); `Fixed(0)` returns `None` like other
  invalid inputs. Input guards (< 2 points, zero range) are shared with the
  automatic rules.

### Changed — BREAKING

- Adding a variant to the public `BinMethod` enum breaks exhaustive `match`
  expressions. Add a `Fixed(k)` arm (or a wildcard arm) when upgrading.

## [0.4.1] - 2026-06-10

### Changed

- WASM: dropped legacy `*_json` parameter-name suffixes from 8 exported
  functions (`xbar_r_chart`, `p_chart`, `laney_p_chart`, `detect_changepoints`,
  `detect_changepoints_multi`, `gage_rr_xbar_r`, `gage_rr_anova`,
  `percentile_capability`) — they take native JS objects/arrays, and JSON-string
  arguments are now rejected early with a descriptive error.

## Earlier releases

- 0.4.0 — 2026-04-28
- 0.3.0 — 2026-04-02
- 0.2.0 — 2026-03-18
- 0.1.0 — 2026-02-09
