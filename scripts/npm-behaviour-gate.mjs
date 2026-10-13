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

console.log(`binding behaviour gate: ok (${checks} checks)`)
