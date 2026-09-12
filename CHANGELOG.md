# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Maintained from 0.5.0 onward; earlier entries list release dates only (see git history).

## [Unreleased]

### Changed

- **The C FFI and the WASM binding now share one JSON contract** (`src/wire.rs`).
  Both transports carry the same analyses, but their shapes had drifted apart.
  Every entry point the two have in common -- the three SPC charts, both
  capability analyses, both Gage R&R methods and changepoint detection -- now
  parses the same request type and serialises the same response type.
  **Breaking for C FFI callers** (`UAnalytics` NuGet 0.4.0 → 0.5.0):
  - `uanalytics_xbar_r_chart` answers `xbar_cl`/`xbar_ucl`/`xbar_lcl` (was
    `x_bar_*`), and carries the per-point `xbar_points`/`r_points` (each with
    `index`, `value`, `violations`) and `in_control` that only the WASM side
    reported before. The request accepts an optional `rules` array, exactly as
    the WASM binding's options do.
  - `uanalytics_p_chart` and `uanalytics_laney_p_chart` take `[defectives,
    sample_size]` pairs -- the crate's own order, and the WASM binding's. They
    used to take the pair **reversed**, so a caller reading the two transports'
    docs against each other got a plausible chart from the wrong numbers.
    Responses now carry per-point `points` (`index`, `value`, `ucl`, `cl`,
    `lcl`, `out_of_control`) and `in_control` instead of parallel
    `proportions`/`ucls`/`lcls` arrays.
  - `uanalytics_process_capability` no longer estimates a within sigma from
    the moving range when `sigma_within` is omitted. That estimate is right
    for individual observations and wrong for data flattened out of
    subgroups, and the entry point could not tell which it had; the WASM
    binding never guessed. Without `sigma_within` the short-term indices are
    `null` and `sigma_source` is `"overall"`. A caller with individual
    observations gets the former number by running `imr_chart` first and
    passing its `sigma_hat` -- the assumption becomes the caller's, and
    visible.
  - `uanalytics_detect_changepoints` defaults to the `l2` cost (was `normal`
    -- same input, different changepoints across the two transports), takes
    a numeric `penalty` as a JSON number rather than a numeric string, accepts
    an optional `cost` of `"l2"`/`"normal"`, and reports `n_segments`.
  - `uanalytics_percentile_capability` reports `percentile_lower` and
    `percentile_upper`; the Gage R&R `status` uses the same spelling as WASM.
  - **Unknown request fields are rejected** over the FFI as they always were
    over WASM. A misspelt `sigmaWithin` used to be ignored silently, yielding
    a long-term-only answer with no hint that the caller's sigma never
    arrived; it is now a parse error naming the field.
  - A test asserts, for all eight shared entry points, that the FFI body is
    the `serde_json` rendering of the wire value the WASM binding emits, so
    the two cannot drift silently again.
- The C# client exposes `LaneyPChart` (the native entry point already existed),
  an optional `rules` argument on `XbarRChart`, and takes the changepoint
  `penalty` as `double?` plus a `cost` argument.
- **Three C FFI entry points that only WASM had**: `uanalytics_xbar_s_chart`,
  `uanalytics_imr_chart` and `uanalytics_run_rules`, with `XbarSChart`,
  `ImrChart` and `RunRules` on the C# client. `imr_chart` is what makes the
  `process_capability` change above whole for individual data: it returns the
  moving-range `sigma_hat` that the capability entry point used to compute on
  its own, and a test pins that feeding it back yields the same number.

### Fixed

- The `rules: unknown rule …` error message carried a run of literal spaces in
  the middle of a sentence (a line-continuation lost at some point).

- **`boxcox_capability` no longer reports short-term capability indices.** It
  computed `cp`/`cpk`/`cpu`/`cpl` from the overall sigma of the transformed
  data, which made `cp` equal `pp` and `cpk` equal `ppk` for every input — a
  long-term number under a short-term name. A Box-Cox analysis starts from a
  flat vector, so there is no rational subgrouping and no within-subgroup sigma
  to estimate; the four are now always `None` (`NaN` across the C FFI, `null`
  in WASM), the same choice `process_capability` already makes when
  `sigma_within` is omitted. **Breaking** for callers reading those four
  fields; the values they were reading were not short-term indices.
  A crate test that asserted `cp.is_some()` had been pinning the old behaviour.

### Added

- **WASM bindings for `boxcox_capability`, `sigma_to_ppm` and `ppm_to_sigma`**.
  `#[wasm_bindgen]` exports go from 21 to 24.
  - `boxcox_capability({ data, usl?, lsl? })` → `{ lambda, pp, ppk, ppu, ppl, cpm, … }`
  - `sigma_to_ppm(sigma)` / `ppm_to_sigma(ppm)` — scalar in, scalar out.
  - Both sigma-level functions use the Motorola convention **including the
    1.5-sigma shift**, which the docs now name explicitly rather than leaving to
    be inferred from a table: six sigma is ~3.4 PPM, not ~0.002.
  - `sigma_to_ppm` rejects a non-finite input instead of returning `NaN`;
    `ppm_to_sigma` rejects both ends of `(0, 1_000_000)`, which the sigma scale
    does not reach.
- **WASM bindings for `cusum` and `ewma`** (docket #220 group ②). The crate has
  had both charts since before the bindings existed; only the exposure was
  missing, so JS/TS consumers reimplemented sequential shift detection that the
  crate already performs. `#[wasm_bindgen]` exports go from 19 to 21.
  - `cusum({ data, target, sigma, k?, h? })` → `{ h, points, signal_indices, in_control }`
  - `ewma({ data, target, sigma, lambda?, l_factor? })` → `{ points, signal_indices, in_control }`
  - Defaults follow the sources rather than the binding's convenience: `k = 0.5` /
    `h = 5.0` (Page 1954), `lambda = 0.2` / `l_factor = 3.0` (Roberts 1959).
  - `h` is echoed on the CUSUM result because its decision interval is constant,
    while EWMA's limits widen with the index and are therefore per point.
  - Both reject an empty `data` and out-of-domain parameters with a message
    naming the offending parameter, instead of returning an empty or silently
    degraded series.


## [0.9.0] - 2026-09-12

### Changed

- **Breaking:** variables control charts accept subgroup sizes up to 25, not 10.
  The factor tables (A2, A3, D3, D4, B3, B4, d2, c4) stopped at n=10, which is
  the range a published table conventionally prints -- not a limit of the
  method. A study running larger subgroups could not use the chart at all.

  The whole range is now computed from the definitions rather than transcribed,
  because no single published table covers n=25 in every factor:

  ```
  d2(n) = E[W],  d3(n) = sd[W]   for W the range of n iid standard normals
  c4(n) = sqrt(2/(n-1)) * Gamma(n/2) / Gamma((n-1)/2)
  A2 = 3/(d2*sqrt(n))                  A3 = 3/(c4*sqrt(n))
  D3 = max(0, 1 - 3*d3/d2)             D4 = 1 + 3*d3/d2
  B3 = max(0, 1 - 3*sqrt(1-c4^2)/c4)   B4 = 1 + 3*sqrt(1-c4^2)/c4
  ```

  The computation reproduces all 72 ASTM E2587 published values over n=2..=10
  across the eight tables, and a test pins that agreement so a later edit to the
  tables cannot quietly diverge from the standard.
- **Breaking:** `XBarRChart::new` and `XBarSChart::new` return
  `Result<Self, ControlChartError>` instead of panicking on an unsupported
  subgroup size. The size usually comes from the measurements a caller was
  handed, so rejecting it is an ordinary outcome rather than a contract
  violation -- and a boundary that cannot unwind, such as the WebAssembly
  entry points, needs it as a value.
- **Breaking:** `NPChart::new` returns `Result<Self, ControlChartError>`
  instead of panicking on a sample size of zero, for the same reason as the
  subgroup charts: the size comes from data. `ControlChartError` gains
  `ZeroSampleSize` and is now `#[non_exhaustive]`, so a later variant is not a
  breaking change.
- **Breaking:** `add_sample` on the X-bar-R, X-bar-S, Individual-MR, P, NP and
  U charts returns `Result<(), ControlChartError>` and rejects a sample it
  cannot use -- the wrong length, a non-finite value, a sample size of zero,
  more defectives than items, units that are not positive -- instead of
  dropping it. A dropped sample left every later point one position off the
  input it came from, and nothing told the caller. `ControlChartError` gains
  `SampleLengthMismatch`, `NonFiniteValue`, `DefectivesExceedSampleSize` and
  `NonPositiveUnits`. `CChart::add_sample` accepts any count and still returns
  nothing.
- The WebAssembly `xbar_r_chart` binding no longer states the supported
  subgroup range itself. It carried a second copy of the same literal bound, so
  widening the tables would have left the binding rejecting sizes the crate had
  just learned to handle -- a disagreement neither side's tests could see,
  because each was right about its own copy. It now reports whatever the
  constructor rejects.

### Added

- `RuleSet` selects which of the eight run tests a chart applies, and
  `XBarRChart`, `XBarSChart` and `IndividualMRChart` take one through
  `with_rules`. `RuleSet::nelson()` -- all eight -- stays the default, so a
  chart built without it behaves as before. A process that trips a test for a
  benign reason previously had to have that signal filtered out downstream,
  after it had already been counted as out of control.

  A rule is named by the `ViolationType` it reports rather than by a parallel
  enum, because the two are one-to-one and a second list would have to be kept
  in step with the first.
- The WebAssembly `xbar_r_chart` binding takes an optional second argument,
  `{ rules?: string[] }`, naming the tests to apply. Omitted, `undefined`,
  `null`, or an object without `rules` all mean all eight, so existing calls
  are unchanged; `{ rules: [] }` leaves control limits only.

- `ControlChartError`, `MIN_SUBGROUP_SIZE` and `MAX_SUBGROUP_SIZE` are public,
  so a caller can validate a subgroup size before building a chart and can match
  on the rejection rather than parsing a message.
- `Debug` and `Clone` on `XBarRChart`, `XBarSChart` and `IndividualMRChart`.
- `IndividualMRChart::sigma_hat` -- `MR-bar / d2(2)`, the within sigma for
  individual observations. The subgroup charts already had theirs.
- WebAssembly `xbar_s_chart` and `imr_chart`, with the same `{ rules? }` option
  and `sigma_hat` as `xbar_r_chart`. Both charts existed in the crate and could
  not be reached from JavaScript.
- WebAssembly `run_rules(values, limits, options?)` -- the run-test engine on
  its own, against caller-supplied limits. Given a chart's points and limits it
  reports exactly what the chart reported.
- WebAssembly `np_chart`, `c_chart`, `u_chart` and `laney_u_chart`.
- C FFI `uanalytics_xbar_r_chart` returns `sigma_hat`, and
  `uanalytics_process_capability` accepts `sigma_within` and reports
  `sigma_source` (`"within"` or `"moving_range"`).

### Fixed

- **Breaking:** Cpm is computed against a declared target only, and from the
  spread of the data about that target. It was derived from Cp, so it inherited
  Cp's within-subgroup sigma, and without a target it was measured against the
  specification midpoint. Chan, Cheng & Spiring (1988) define it through
  `sqrt(sum((x_i - T)^2) / (n - 1))`, and Minitab reports it only when a target
  is given. Now:
  - `cpm` is `None` without `with_target`, when the target lies outside the
    limits, or when every observation equals the target;
  - the numerator is `min(T - LSL, USL - T) / 3`, which is `(USL - LSL) / 6`
    when the target is the midpoint;
  - `compute` and `compute_overall` give the same `cpm`, and the WebAssembly
    `process_capability` returns it whether or not `sigma_within` is supplied.

  For `[9.8, 10.1, 10.3, 9.9, 10.2, 10.4]` with limits 8 and 12 and a within
  sigma of 0.2, 0.8.0 reported 2.8793 with no target; it now reports none, and
  2.5198 against a declared target of 10.
- **Breaking (C FFI):** a rejected request now returns its failure status
  (`-2` malformed JSON, `-3` rejected input). It returned `0` with an
  `{"error": ...}` body, so a caller that branches on the status -- the C#
  client does -- received the error as a successful result.
- The C FFI X-bar-R, P and capability entry points carried their own
  arithmetic instead of calling the crate. They now delegate:
  - `uanalytics_xbar_r_chart` accepted only n <= 10 with three-decimal
    constants after the crate reached n = 25. It now rejects ragged or
    non-finite subgroups rather than letting the chart skip them, since its
    response arrays carry no index and a skipped subgroup would shift every
    later value onto the wrong row.
  - `uanalytics_p_chart` rejects a sample with nothing inspected or more
    defectives than inspected; it used to report a proportion above 1 or NaN.
  - `uanalytics_process_capability` still estimates the within sigma from the
    moving range when none is given, but through the crate's I-MR chart rather
    than its own `d2` literal, and it now says so in `sigma_source`.
- C FFI `uanalytics_laney_p_chart` passed its `[inspected, defective]` pairs to
  a function that takes `(defective, sample_size)`, so every request was
  computed in reverse.
- C FFI `uanalytics_detect_changepoints` documented `"AIC"` and `"MBIC"`
  penalties and computed `BIC` for both, as it did for any unrecognised string.
  It now accepts `"BIC"` or a number and rejects anything else.
- The README's X-bar-R example had stopped compiling when `XBarRChart::new`
  began returning `Result`. The README's Rust examples now run with the
  doc-tests.
- `laney_p_chart` returns `None` for a sample with nothing inspected or more
  defectives than inspected. It checked only the total, so one such sample made
  phi and every limit NaN -- and with NaN limits no point compares as out of
  control, so the chart read as in control. Its documentation also said a
  degenerate `p_bar` of 0 or 1 returns `None`; it returns a zero-width chart,
  and now says so.
- The WebAssembly `p_chart` and `laney_p_chart` reject a sample with nothing
  inspected or more defectives than inspected, naming its index. `p_chart`
  dropped such a sample, which shifted the index of every point after it.

## [0.8.0] - 2026-09-10

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
