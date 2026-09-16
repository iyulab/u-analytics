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
- **Reliability**: Weibull fitting by maximum likelihood
- **Change detection**: changepoints, period estimation, spectral residual scoring
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

var chart = analytics.XbarRChart(new[]
{
    new[] { 10.1, 10.3, 9.8 },
    new[] { 10.0, 10.2, 10.1 },
    new[] { 9.9,  10.4, 10.2 },
});

Console.WriteLine(chart.GetProperty("ucl").GetDouble());
```

Every method returns a `System.Text.Json.JsonElement`, so a result can be read
field by field or deserialized into your own type.

Input that cannot be analysed is refused rather than approximated, with a code
naming what was wrong and the index of the row it was wrong in.

## Platforms

Windows, Linux and macOS (x64 and arm64). The native library ships inside the
package; no separate install is needed.

## License

MIT
