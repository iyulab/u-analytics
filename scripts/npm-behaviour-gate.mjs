// Binding behaviour gate: the published bindings actually answer.
//
// An import-only smoke test proves the package loads. It cannot prove any
// binding works -- it would pass unchanged if every export threw. The
// host-target tests cover the wire contract, but they run on the pure core
// with no `JsValue` anywhere; what is only observable against the built
// package is whether a real JS object survives the boundary and comes back
// with the documented shape.
//
//   PKG_NAME=<package-name> node scripts/npm-behaviour-gate.mjs
//
// Run it from a directory where the package is installed -- the publish
// workflow's smoke directory, or any directory with a local build under
// node_modules. The package is resolved from the working directory, not from
// this file. The publish workflow runs it after the README examples and before
// `npm publish`; keeping the assertions in a file rather than in the workflow
// means the same checks can run against a local build before the push that
// publishes, so an assertion the engine no longer satisfies is found there.

import { createRequire } from 'node:module'
import { join } from 'node:path'
import { pathToFileURL } from 'node:url'

const pkg = process.env.PKG_NAME
if (!pkg) {
  console.error('usage: PKG_NAME=<package-name> node npm-behaviour-gate.mjs')
  process.exit(2)
}
const requireFromCwd = createRequire(join(process.cwd(), 'noop.js'))
const m = await import(pathToFileURL(requireFromCwd.resolve(pkg)).href)
let checks = 0
const ok = (c, msg) => { checks++; if (!c) throw new Error(msg) }

const subgroups = [
  [9.9, 10.1, 10.0, 9.8, 10.2], [10.3, 9.7, 10.0, 10.1, 9.9],
  [9.8, 10.2, 10.1, 9.9, 10.0], [10.5, 9.5, 10.0, 10.2, 9.8],
  [9.6, 10.4, 10.0, 9.9, 10.1], [10.1, 9.9, 10.0, 10.3, 9.7],
]
const data = subgroups.flat()

// A native JS object must cross the boundary and come back shaped as
// documented -- omitting sigma_within reports the long-term family
// only, with the short-term one null rather than borrowed.
const overall = m.process_capability({ data, usl: 11, lsl: 9 })
ok(overall.sigma_source === 'overall', `sigma_source=${overall.sigma_source}`)
ok(overall.cp === null || overall.cp === undefined, 'cp must be null without sigma_within')
ok(typeof overall.pp === 'number', 'pp must be present')

// sigma_hat is what makes the two bindings compose: the chart hands
// over the quantity a flat vector cannot carry.
const chart = m.xbar_r_chart(subgroups)
ok(typeof chart.sigma_hat === 'number', 'xbar_r_chart must expose sigma_hat')

const within = m.process_capability({ data, usl: 11, lsl: 9, sigma_within: chart.sigma_hat })
ok(within.sigma_source === 'within', `sigma_source=${within.sigma_source}`)
ok(within.cp !== within.pp, 'cp and pp must come from different sigmas')

// A one-sided specification is expressible.
const upper = m.process_capability({ data, usl: 11 })
ok(upper.pp === null || upper.pp === undefined, 'Pp needs both limits')
ok(typeof upper.ppu === 'number', 'an upper limit supports Ppu')

// The wire schema is strict, and a stringified payload is refused
// with an actionable message rather than a serde error.
let threw = false
try { m.process_capability({ data, usl: 11, sigmaWithin: 0.3 }) } catch { threw = true }
ok(threw, 'unknown key must be rejected')

threw = false
try { m.process_capability(JSON.stringify({ data, usl: 11 })) } catch { threw = true }
ok(threw, 'a JSON string must be rejected, not parsed')

// A refusal is an Error carrying a stable code and the offending row,
// so a caller can say "row 2: not a whole number" without parsing text.
// Only observable here: the host tests never build a JS Error.
const refusal = (f) => { try { f() } catch (e) { return e } return null }
let e = refusal(() => m.c_chart([1, 2, 2.5]))
ok(e instanceof Error, 'a refusal must be an Error')
ok(e.code === 'count_not_whole' && e.index === 2, `code=${e.code} index=${e.index}`)
ok(typeof e.message === 'string' && e.message.includes('defects[2]'), `message=${e.message}`)
// On a [defectives, sample_size] pair the code says which member failed:
// every bad size is one code, so a consumer never reads `message` to find out.
for (const bad of [0, -3, 10.5]) {
  e = refusal(() => m.p_chart([[3, 100], [1, bad]]))
  ok(e.code === 'sample_size_not_whole' && e.index === 1, `size=${bad} code=${e.code} index=${e.index}`)
}
e = refusal(() => m.p_chart([[3, 100], [1.5, 100]]))
ok(e.code === 'count_not_whole' && e.index === 1, `code=${e.code} index=${e.index}`)
// Lengths that must agree are one code everywhere: a ragged subgroup names the
// array, its row, and both sizes.
e = refusal(() => m.xbar_r_chart([[1, 2, 3], [4, 5], [6, 7, 8]]))
ok(e.code === 'dimension_mismatch' && e.parameter === 'subgroups' && e.index === 1
   && e.expected === 3 && e.got === 2, `ragged: ${e.code} ${e.parameter} ${e.index}`)
e = refusal(() => m.np_chart([1, 2], 10.5))
ok(e.code === 'sample_size_not_whole' && e.index === null, `code=${e.code} index=${e.index}`)

// A Gage R&R study comes back with the charts its method plots.
const meas = [[[10,10.2,10.1],[10.1,10,10.2]],[[12,12.1,12.3],[12.2,12,12.1]],[[14,14.1,14.2],[14.1,14.3,14]]]
const grr = m.gage_rr_xbar_r({ measurements: meas })
const ranges = meas.flatMap((p) => p.map((tr) => Math.max(...tr) - Math.min(...tr)))
const rbar = ranges.reduce((a, b) => a + b, 0) / ranges.length
ok(Math.abs(grr.range_chart.center - rbar) < 1e-12, `R-bar=${grr.range_chart.center}`)
ok(Math.abs(grr.range_chart.ucl - 2.575 * rbar) < 1e-12, `UCL_R=${grr.range_chart.ucl}`)
ok(grr.range_chart.lcl === 0, `LCL_R=${grr.range_chart.lcl}`)
ok(typeof grr.average_chart.center === 'number', `X-double-bar=${grr.average_chart.center}`)

// Phase II: a Phase I standard crosses as options and fixes the limits.
const p2 = m.laney_p_chart([[11, 170]], { p_bar: 0.0514, phi: 0.236 })
ok(p2.p_bar === 0.0514 && p2.points[0].out_of_control, 'Phase II uses the given standard')
ok(p2.points[0].z > 3, `z=${p2.points[0].z} must exceed the +3 limit`)

// A number array is read as sent: null is refused at its index,
// not turned into 0 by the glue's typed-array copy.
let refused = null
try { m.t_chart([1, null, 2]) } catch (err) { refused = err }
ok(refused && refused.code === 'malformed_input' && refused.index === 1,
  `t_chart([1, null, 2]) gave ${refused ? refused.code + '@' + refused.index : 'no error'}`)
ok(m.g_chart(new Float64Array([10, 20, 30])).points.length === 3, 'g_chart(Float64Array) changed')

// Hypothesis tests refuse at the row of a number[][] argument.
let rowErr = null
try { m.kruskal_wallis_test([[1, 2, 3], [2, 3, null]]) } catch (err) { rowErr = err }
ok(rowErr && rowErr.code === 'malformed_input' && rowErr.index === 2 && rowErr.parameter === 'groups[1]',
  `kruskal_wallis_test null row: ${rowErr ? rowErr.code + '@' + rowErr.index : 'no error'}`)
const anova = m.one_way_anova([[1, 2, 3], [2, 3, 4], [5, 6, 7]])
ok(anova.df_between === 2 && anova.df_within === 6 && anova.f_statistic === 13, `anova ${JSON.stringify(anova)}`)

// Run-test sets by name: eight points above the centre line are Western
// Electric's run (eight), not Nelson's (nine), and the default -- the 3-sigma
// test alone -- sees nothing.
const eight = [1, 1, 1, 1, 1, 1, 1, 1]
const limits = { ucl: 3, cl: 0, lcl: -3 }
const fired = (opts) => m.run_rules(eight, limits, opts).flatMap((p) => p.violations)
ok(JSON.stringify(fired({ rules: 'western_electric' })) === '["EightOneSide"]',
  `western_electric: ${JSON.stringify(fired({ rules: 'western_electric' }))}`)
ok(fired({ rules: 'nelson' }).length === 0, `nelson: ${JSON.stringify(fired({ rules: 'nelson' }))}`)
ok(fired(undefined).length === 0, `default: ${JSON.stringify(fired(undefined))}`)
let setErr = null
try { m.run_rules(eight, limits, { rules: 'weco' }) } catch (err) { setErr = err }
ok(setErr && setErr.code === 'unknown_option' && setErr.parameter === 'rules' &&
  JSON.stringify(setErr.expected) === '["shewhart","western_electric","nelson"]',
  `unknown set: ${setErr ? setErr.code + ' ' + JSON.stringify(setErr.expected) : 'no error'}`)

// Equivalence (TOST): the trailing `alpha` and `equal_variance` are optional
// -- omitted, undefined and null all mean the default -- and a refusal carries
// its code and the fields behind it.
const fill = [99.8, 100.1, 100.3, 99.9, 100.0, 100.2, 99.7, 100.1, 100.0, 99.9]
const tost = m.tost_one_sample(fill, 100, -0.5, 0.5)
ok(tost.equivalent === true && tost.df === 9 && tost.p_value === Math.max(tost.p_lower, tost.p_upper),
  `tost_one_sample ${JSON.stringify(tost)}`)
for (const absent of [undefined, null]) {
  ok(JSON.stringify(m.tost_one_sample(fill, 100, -0.5, 0.5, absent)) === JSON.stringify(tost),
    `tost_one_sample alpha=${absent} differs from the omitted default`)
}
ok(m.tost_one_sample(fill, 100, -0.5, 0.5, 0.2).ci_upper < tost.ci_upper, 'alpha 0.2 must narrow the interval')
const other = [100.4, 99.6, 100.2, 100.0, 99.9, 100.3, 99.8, 100.1, 100.0]
ok(m.tost_two_sample(fill, other, -0.5, 0.5, null, true).df === 17, 'equal_variance true is the pooled df')
ok(m.tost_two_sample(fill, other, -0.5, 0.5, undefined, null).df !== 17, 'equal_variance null is Welch')
ok(m.tost_paired(fill.slice(0, 9), other, -0.5, 0.5).df === 8, 'tost_paired df = n - 1')
const badLevel = refusal(() => m.tost_one_sample(fill, 100, -0.5, 0.5, 0.5))
ok(badLevel && badLevel.code === 'parameter_out_of_range' && badLevel.parameter === 'alpha' &&
  badLevel.exclusive_min === 0 && badLevel.exclusive_max === 0.5 && badLevel.got === 0.5,
  `alpha 0.5: ${badLevel ? badLevel.code + ' ' + badLevel.parameter : 'no error'}`)
const badMargins = refusal(() => m.tost_one_sample(fill, 100, 0.5, -0.5))
ok(badMargins && badMargins.code === 'invalid_option' && badMargins.parameter === 'high',
  `margins reversed: ${badMargins ? badMargins.code + ' ' + badMargins.parameter : 'no error'}`)
const nanMargin = refusal(() => m.tost_paired(fill, fill, Number.NaN, 0.5))
ok(nanMargin && nanMargin.code === 'value_not_finite' && nanMargin.parameter === 'low',
  `NaN margin: ${nanMargin ? nanMargin.code + ' ' + nanMargin.parameter : 'no error'}`)

// Tolerance intervals: `n` is a whole number read from the JS value, `sided` a
// closed vocabulary read from a JS string -- neither is coerced by the glue.
const toleranceK = m.tolerance_factor(10, 0.95, 0.95, 'two_sided')
ok(typeof toleranceK === 'number' && Math.abs(toleranceK - 3.393429478712608) < 1e-9, `tolerance_factor ${toleranceK}`)
ok(Math.abs(m.tolerance_factor(10, 0.95, 0.95, 'one_sided') - 2.9109634130781664) < 1e-9, 'one-sided factor')
const diameters = [10.02, 9.98, 10.01, 10.0, 9.99, 10.03, 9.97, 10.01, 10.0, 9.99]
const tolerance = m.tolerance_interval(diameters, 0.95, 0.95, 'two_sided')
ok(tolerance.n === 10 && tolerance.k === toleranceK && tolerance.lower < tolerance.mean && tolerance.mean < tolerance.upper &&
  Math.abs(tolerance.upper - (tolerance.mean + tolerance.k * tolerance.std_dev)) < 1e-12,
  `tolerance_interval ${JSON.stringify(tolerance)}`)
ok(m.tolerance_interval(new Float64Array(diameters), 0.95, 0.95, 'one_sided').k < tolerance.k, 'one-sided k is the smaller')
const badSided = refusal(() => m.tolerance_factor(10, 0.95, 0.95, 'both'))
ok(badSided && badSided.code === 'unknown_option' && badSided.parameter === 'sided' && badSided.got === 'both' &&
  JSON.stringify(badSided.expected) === '["two_sided","one_sided"]',
  `unknown sided: ${badSided ? badSided.code + ' ' + JSON.stringify(badSided.expected) : 'no error'}`)
for (const notAName of [null, undefined, 1]) {
  const e2 = refusal(() => m.tolerance_interval(diameters, 0.95, 0.95, notAName))
  ok(e2 && e2.code === 'malformed_input' && e2.parameter === 'sided', `sided=${notAName}: ${e2 ? e2.code : 'no error'}`)
}
const smallN = refusal(() => m.tolerance_factor(1, 0.95, 0.95, 'two_sided'))
ok(smallN && smallN.code === 'parameter_out_of_range' && smallN.parameter === 'n' && smallN.min === 2 && smallN.got === 1,
  `n=1: ${smallN ? smallN.code + ' ' + smallN.parameter : 'no error'}`)
const negativeN = refusal(() => m.tolerance_factor(-1, 0.95, 0.95, 'two_sided'))
ok(negativeN && negativeN.code === 'parameter_out_of_range' && negativeN.parameter === 'n' && negativeN.got === -1,
  `n=-1: ${negativeN ? negativeN.code + ' ' + negativeN.parameter : 'no error'}`)
const fractionalN = refusal(() => m.tolerance_factor(2.5, 0.95, 0.95, 'two_sided'))
ok(fractionalN && fractionalN.code === 'malformed_input' && fractionalN.parameter === 'n',
  `n=2.5: ${fractionalN ? fractionalN.code + ' ' + fractionalN.parameter : 'no error'}`)
const edgeCoverage = refusal(() => m.tolerance_factor(10, 1, 0.95, 'two_sided'))
ok(edgeCoverage && edgeCoverage.code === 'parameter_out_of_range' && edgeCoverage.parameter === 'coverage' &&
  edgeCoverage.exclusive_min === 0 && edgeCoverage.exclusive_max === 1 && edgeCoverage.got === 1,
  `coverage=1: ${edgeCoverage ? edgeCoverage.code + ' ' + edgeCoverage.parameter : 'no error'}`)
const flat = refusal(() => m.tolerance_interval([2, 2, 2], 0.95, 0.95, 'two_sided'))
ok(flat && flat.code === 'degenerate_data' && flat.parameter === 'data', `constant data: ${flat ? flat.code : 'no error'}`)
const nanData = refusal(() => m.tolerance_interval([1, Number.NaN, 3], 0.95, 0.95, 'two_sided'))
ok(nanData && nanData.code === 'value_not_finite' && nanData.index === 1, `NaN datum: ${nanData ? nanData.code : 'no error'}`)

// Type 1 gage study: the trailing `tolerance_fraction` and `sigma_multiple` are optional
// (omitted, undefined and null all mean the default), Cg and Cgk are the quoted closed
// forms at the defaults, and a refusal carries its code and the fields behind it.
const readings = [10.02, 9.98, 10.05, 10.01, 9.97, 10.03, 10.0, 10.04]
const t1 = m.type1_gage_study(readings, 10, 9, 11)
ok(t1.n === 8 && t1.df === 7 && Math.abs(t1.cg - 0.4 / (6 * t1.std_dev)) < 1e-12 &&
  Math.abs(t1.cgk - (0.2 - Math.abs(t1.bias)) / (3 * t1.std_dev)) < 1e-12 && t1.cgk < t1.cg,
  `type1_gage_study ${JSON.stringify(t1)}`)
for (const absent of [undefined, null]) {
  ok(JSON.stringify(m.type1_gage_study(readings, 10, 9, 11, absent, absent)) === JSON.stringify(t1),
    `type1_gage_study options=${absent} differ from the omitted defaults`)
}
const t1b = m.type1_gage_study(readings, 10, 9, 11, 0.15, 4)
ok(Math.abs(t1b.cg - t1.cg * 0.75 * 1.5) < 1e-12, 'tolerance_fraction 0.15 and sigma_multiple 4 scale Cg by K and 1/L')
const t1c = m.type1_gage_study(readings, 9.5, 9, 11)
ok(t1c.cgk < 0 && t1c.percent_var_repeatability_and_bias === null,
  `a bias that fills the band has no repeatability-and-bias percentage: ${JSON.stringify(t1c)}`)
const badLimits = refusal(() => m.type1_gage_study(readings, 10, 11, 9))
ok(badLimits && badLimits.code === 'invalid_option' && badLimits.parameter === 'usl',
  `limits reversed: ${badLimits ? badLimits.code + ' ' + badLimits.parameter : 'no error'}`)
const badFraction = refusal(() => m.type1_gage_study(readings, 10, 9, 11, 0))
ok(badFraction && badFraction.code === 'parameter_out_of_range' && badFraction.parameter === 'tolerance_fraction' &&
  badFraction.exclusive_min === 0 && badFraction.max === 1 && badFraction.got === 0,
  `tolerance_fraction 0: ${badFraction ? badFraction.code + ' ' + badFraction.parameter : 'no error'}`)
const flat1 = refusal(() => m.type1_gage_study([2, 2, 2], 2, 0, 4))
ok(flat1 && flat1.code === 'degenerate_data' && flat1.parameter === 'measurements',
  `equal readings: ${flat1 ? flat1.code + ' ' + flat1.parameter : 'no error'}`)
const nanRef = refusal(() => m.type1_gage_study(readings, Number.NaN, 9, 11))
ok(nanRef && nanRef.code === 'value_not_finite' && nanRef.parameter === 'reference',
  `NaN reference: ${nanRef ? nanRef.code + ' ' + nanRef.parameter : 'no error'}`)

// Hotelling T²: a plain JS array of rows crosses the boundary, the pair that leaves the
// correlation (inside both ranges) is the one flagged, and the `estimates` of a Phase I
// result go back in as the `reference` of Phase II (`m` stays a whole number on the way).
const line = [[1, 1.1], [2, 1.9], [3, 3.2], [4, 3.9], [5, 5.1], [6, 5.8], [7, 7.2], [8, 7.9], [4.5, 1.5]]
const h1 = m.hotelling_phase_i(line)
ok(JSON.stringify(h1.out_of_control) === '[8]' && h1.in_control === false && h1.t2.length === 9 &&
  h1.estimates.m === 9 && h1.estimates.mean.length === 2, `hotelling_phase_i ${JSON.stringify(h1)}`)
for (const absent of [undefined, null]) {
  ok(JSON.stringify(m.hotelling_phase_i(line, absent)) === JSON.stringify(h1),
    `hotelling_phase_i alpha=${absent} differs from the default 0.0027`)
}
ok(m.hotelling_phase_i(line, 0.05).ucl < h1.ucl, 'a larger alpha lowers the limit')
const h2 = m.hotelling_phase_ii([[4.5, 4.4], [4.5, 1.5]], m.hotelling_phase_i(line.slice(0, 8)).estimates)
ok(JSON.stringify(h2.out_of_control) === '[1]' && h2.estimates.m === 8, `hotelling_phase_ii ${JSON.stringify(h2)}`)
ok(m.hotelling_phase_ii([], h1.estimates).t2.length === 0, 'no observations: the limit alone')
const badAlpha = refusal(() => m.hotelling_phase_i(line, 1))
ok(badAlpha && badAlpha.code === 'parameter_out_of_range' && badAlpha.parameter === 'alpha' &&
  badAlpha.exclusive_min === 0 && badAlpha.exclusive_max === 1 && badAlpha.got === 1,
  `alpha 1: ${badAlpha ? badAlpha.code + ' ' + badAlpha.parameter : 'no error'}`)
const fewRows = refusal(() => m.hotelling_phase_i(line.slice(0, 3)))
ok(fewRows && fewRows.code === 'insufficient_data' && fewRows.parameter === 'data' && fewRows.min === 4 && fewRows.got === 3,
  `three rows of two: ${fewRows ? fewRows.code + ' ' + fewRows.parameter : 'no error'}`)
const nullCell = refusal(() => m.hotelling_phase_i([[1, 2], [3, null], [5, 6], [7, 9]]))
ok(nullCell && nullCell.code === 'malformed_input' && nullCell.index === 1 && nullCell.parameter === 'data[1]',
  `null cell: ${nullCell ? nullCell.code + ' ' + nullCell.parameter + ' ' + nullCell.index : 'no error'}`)
const fractionM = refusal(() => m.hotelling_phase_ii([[1, 2]], { ...h1.estimates, m: 9.5 }))
ok(fractionM && fractionM.code === 'malformed_input' && fractionM.parameter === 'reference.m',
  `fractional m: ${fractionM ? fractionM.code + ' ' + fractionM.parameter : 'no error'}`)
const flatColumn = refusal(() => m.hotelling_phase_i([[1, 5], [2, 5], [3, 5], [4, 5], [5, 5]]))
ok(flatColumn && flatColumn.code === 'degenerate_data', `constant column: ${flatColumn ? flatColumn.code : 'no error'}`)

// Hotelling T² for subgroups: a nested JS array crosses the boundary, the estimates of Phase I
// are the reference of Phase II (`m` and `n` stay whole), and a subgroup of another size names itself.
const sub = [
  [[1, 1.1], [1.2, 1], [0.9, 1.3]], [[2, 2.2], [2.1, 1.9], [1.8, 2.1]], [[3.1, 3], [2.9, 3.3], [3, 3.1]],
  [[4, 4.2], [4.2, 3.9], [3.8, 4.1]], [[5, 5.2], [5.1, 4.9], [4.9, 5.1]],
]
const s1 = m.hotelling_subgroup_phase_i(sub)
ok(s1.t2.length === 5 && s1.estimates.m === 5 && s1.estimates.n === 3 && s1.estimates.mean.length === 2,
  `hotelling_subgroup_phase_i ${JSON.stringify(s1)}`)
for (const absent of [undefined, null]) {
  ok(JSON.stringify(m.hotelling_subgroup_phase_i(sub, absent)) === JSON.stringify(s1),
    `hotelling_subgroup_phase_i alpha=${absent} differs from the default`)
}
const s2 = m.hotelling_subgroup_phase_ii(sub.slice(0, 2), s1.estimates)
ok(JSON.stringify(s2.t2) === JSON.stringify(s1.t2.slice(0, 2)) && s2.ucl > s1.ucl, `hotelling_subgroup_phase_ii ${JSON.stringify(s2)}`)
ok(m.hotelling_subgroup_phase_ii([], s1.estimates).t2.length === 0, 'no subgroups: the limit alone')
const oddSize = refusal(() => m.hotelling_subgroup_phase_i([sub[0], sub[1].slice(0, 2), sub[2], sub[3], sub[4]]))
ok(oddSize && oddSize.code === 'dimension_mismatch' && oddSize.parameter === 'subgroups' && oddSize.index === 1 &&
  oddSize.expected === 3 && oddSize.got === 2, `subgroup of another size: ${oddSize ? oddSize.code + ' ' + oddSize.index : 'no error'}`)
const fractionN = refusal(() => m.hotelling_subgroup_phase_ii(sub.slice(0, 1), { ...s1.estimates, n: 2.5 }))
ok(fractionN && fractionN.code === 'malformed_input' && fractionN.parameter === 'reference.n',
  `fractional n: ${fractionN ? fractionN.code + ' ' + fractionN.parameter : 'no error'}`)
const oneRow = refusal(() => m.hotelling_subgroup_phase_i(sub.map(g => g.slice(0, 1))))
ok(oneRow && oneRow.code === 'insufficient_data', `one row per subgroup: ${oneRow ? oneRow.code : 'no error'}`)

// Which variable moved: a plain array in (the observation), the estimates of a Phase I chart as the
// reference, one entry per variable out; the point that broke the correlation has its first variable
// at the mean (unconditional 0) yet a large drop_one, and the signs say which way each departs.
const hc = m.hotelling_contributions(line[8], h1.estimates)
ok(Math.abs(hc.t2 - h1.t2[8]) < 1e-9 && hc.drop_one.length === 2 && hc.z.length === 2 && hc.unconditional.length === 2,
  `hotelling_contributions ${JSON.stringify(hc)}`)
ok(hc.unconditional[0] < 1e-9 && hc.drop_one[0] > 5 && hc.z[0] > 0 && hc.z[1] < 0,
  `hotelling_contributions reading ${JSON.stringify(hc)}`)
ok(Math.abs(hc.drop_one[0] - hc.z[0] * hc.z[0]) < 1e-12, 'drop_one is z squared')
ok(hc.drop_one[0] + hc.drop_one[1] > hc.t2, 'the entries are not shares of t2')
ok(JSON.stringify(m.hotelling_contributions(new Float64Array(line[8]), h1.estimates)) === JSON.stringify(hc),
  'a Float64Array observation reads like an array')
const hs = m.hotelling_subgroup_contributions([1, -1], { mean: [0, 0], covariance: [[1, 0.9], [0.9, 1]], m: 20, n: 4 })
ok(Math.abs(hs.t2 - 80) < 1e-9 && Math.abs(hs.drop_one[0] - 76) < 1e-9, `hotelling_subgroup_contributions ${JSON.stringify(hs)}`)
const shortObs = refusal(() => m.hotelling_contributions([1], h1.estimates))
ok(shortObs && shortObs.code === 'dimension_mismatch' && shortObs.parameter === 'observation' && shortObs.expected === 2 && shortObs.got === 1,
  `a short observation: ${shortObs ? shortObs.code + ' ' + shortObs.parameter : 'no error'}`)
const nullObs = refusal(() => m.hotelling_contributions([1, null], h1.estimates))
ok(nullObs && nullObs.code === 'malformed_input' && nullObs.parameter === 'observation' && nullObs.index === 1,
  `null in an observation: ${nullObs ? nullObs.code + ' ' + nullObs.parameter + ' ' + nullObs.index : 'no error'}`)
const oneEach = refusal(() => m.hotelling_subgroup_contributions([1, 2], { ...s1.estimates, n: 1 }))
ok(oneEach && oneEach.code === 'insufficient_data' && oneEach.parameter === 'reference.n',
  `a subgroup size of one: ${oneEach ? oneEach.code + ' ' + oneEach.parameter : 'no error'}`)

console.log(`binding behaviour gate: ok (${checks} checks)`)
