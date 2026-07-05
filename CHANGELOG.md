# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Maintained from 0.5.0 onward; earlier entries list release dates only (see git history).

## [Unreleased]

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
