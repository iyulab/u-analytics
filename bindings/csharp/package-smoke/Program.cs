// Run by scripts/nuget_package_smoke.sh against the freshly packed package:
// the native library loads, answers, and refuses with the documented surface.
using UAnalytics;

using var analytics = new AnalyticsClient();
Console.WriteLine($"native version: {analytics.GetVersion()}");

var chart = analytics.ImrChart([10.0, 10.2, 9.9, 10.1, 10.3, 9.8, 10.0, 10.1]);
if (!(chart.SigmaHat > 0) || chart.IPoints.Count != 8)
    throw new Exception($"I-MR chart: sigma {chart.SigmaHat}, {chart.IPoints.Count} points");

try
{
    analytics.XbarRChart([[1, 2, 3], [4, 5], [6, 7, 8]]);
    throw new Exception("a ragged subgroup was accepted");
}
catch (AnalyticsException e) when (e.ErrorCode == "dimension_mismatch" && e.Index == 1)
{
}

Console.WriteLine("nuget package smoke test passed");
