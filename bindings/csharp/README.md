# UAnalytics

.NET bindings for the u-analytics statistical process control and reliability engine.

## Features

- **Control charts**: X̄-R, X̄-S, I-MR for variables; p, np, c, u and Laney p′ for
  attributes, with Phase I baselines for Phase II limits
- **Run rules**: applied to any series against limits you supply, or to a chart's own
- **Process capability**: Cp, Cpk, Pp, Ppk, Cpm, and a percentile method for
  non-normal data
- **Measurement systems analysis**: Gage R&R by the Average & Range method and by
  ANOVA, each returning the charts the method plots
- **Reliability**: Weibull fitting by maximum likelihood (`WeibullMle`) and median-rank
  regression (`WeibullMrr`); `WeibullReliability` — MTBF, R(t), h(t) and B-lives
- **Non-normal capability**: `BoxCoxCapability`, and `SigmaToPpm` / `PpmToSigma` on the
  1.5σ-shift convention
- **Event-time trend**: whether one unit's events (failures of a repairable system,
  incidents) arrive at a changing rate — `LaplaceTrendTest`, `MilHdbk189Test`, and the
  power-law process (Crow-AMSAA) fit `PowerLawProcessFit`. Pass the end of observation,
  or `null` when it stopped at the last event
- **Hypothesis tests**: t (one-sample, Welch, paired), Mann-Whitney, Wilcoxon,
  Jarque-Bera, Shapiro-Wilk, Anderson-Darling, Mann-Kendall, ANOVA, Kruskal-Wallis, Levene, Bartlett,
  χ² goodness of fit and independence, Fisher's exact; Bonferroni and
  Benjamini-Hochberg adjustment
- **Stationarity**: `AdfTest` (augmented Dickey-Fuller)
- **Change detection**: changepoints in one series or several aligned ones
  (`DetectChangepointsMulti`), `Cusum` and `Ewma` charts for small sustained shifts,
  period estimation, spectral residual scoring
- **Correlation and regression**: correlation matrices, simple linear regression
- **Distribution fitting**: best-fit selection across candidate distributions

## Installation

```bash
dotnet add package UAnalytics
```

## Usage

```csharp
using UAnalytics;

using var analytics = new AnalyticsClient();

XbarRChartResult chart = analytics.XbarRChart(
[
    [10.1, 10.3, 9.8],
    [10.0, 10.2, 10.1],
    [9.9,  10.4, 10.2],
]);

Console.WriteLine($"UCL {chart.XbarUcl:F3}, sigma {chart.SigmaHat:F3}, in control: {chart.InControl}");

TrendTestResult trend = analytics.MilHdbk189Test([10, 19, 27, 34, 40, 45, 49, 52, 54, 55], end: null);
if (trend.Direction == TrendDirection.Increasing && trend.PValue < 0.05)
    Console.WriteLine("Failures are coming more often.");
```

Every method returns a record (`XbarRChartResult`, `CapabilityResult`, `TrendTestResult`, …)
whose properties are the engine's result fields. A value the inputs do not define — a
short-term index without a within-subgroup sigma, a t ratio of an exact fit — is `null`
on a nullable property. Closed vocabularies are enums: `RunRule` (the run tests a chart
applies and reports), `TrendDirection`, `SigmaSource`, `GageStatus`, and the options
`AdfModel`, `PeltCost` and `CorrelationMethod`.

Input that cannot be analysed is refused rather than approximated: `AnalyticsException`
carries a stable `Reason` (`insufficient_data`, `parameter_out_of_range`,
`value_not_finite`, …), the `Parameter` it is about, the `Index` of the offending
element, and the whole error body in `Details`.

## Trimming and NativeAOT

The client uses no reflection: requests are built as JSON nodes and results are read
through source-generated serialization, and the package is marked `IsAotCompatible`.
It runs unchanged in trimmed and NativeAOT applications, and in .NET file-based apps.

## Platforms

The package carries the native library for `win-x64`, `linux-x64` (glibc 2.39 or
later), `osx-x64` and `osx-arm64`.

## License

MIT
