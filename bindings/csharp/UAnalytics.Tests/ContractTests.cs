using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.Json.Serialization.Metadata;
using Xunit;

namespace UAnalytics.Tests;

/// <summary>
/// Every method's record carries exactly what the engine sent: the raw response and the
/// record written back out must be the same JSON — no key dropped, renamed or added, no
/// value changed. Runs with reflection-based serialization disabled.
/// </summary>
public class ContractTests
{
    private static readonly double[][] Subgroups =
    [
        [10.1, 10.3, 9.8, 10.0], [10.0, 10.2, 10.1, 9.9], [9.9, 10.4, 10.2, 10.0],
        [10.2, 10.1, 9.7, 10.3], [10.0, 9.8, 10.1, 10.2], [10.4, 10.0, 9.9, 10.1],
    ];

    private static readonly double[] Series =
        [10.1, 9.8, 10.3, 10.0, 9.7, 10.4, 10.2, 9.9, 10.1, 10.6, 9.8, 10.0, 10.3, 9.6, 10.2, 10.1, 9.9, 10.5, 10.0, 9.8];

    private static readonly double[] Positive =
        [12.0, 25.0, 33.0, 47.0, 58.0, 71.0, 90.0, 18.0, 29.0, 41.0, 64.0, 83.0, 22.0, 37.0, 52.0, 77.0];

    private static readonly double[] Events = [5, 12, 20, 26, 31, 35, 38, 40];

    private static double[][][] Gage()
    {
        // 10 parts × 3 operators × 2 trials, deterministic.
        var parts = new double[10][][];
        for (var p = 0; p < 10; p++)
        {
            parts[p] = new double[3][];
            for (var o = 0; o < 3; o++)
                parts[p][o] = [p * 0.5 + o * 0.05 + 0.01 * ((p + o) % 3), p * 0.5 + o * 0.05 - 0.01 * ((p * o) % 2)];
        }
        return parts;
    }

    private static double[] Step()
    {
        var x = new double[40];
        for (var i = 0; i < 40; i++)
            x[i] = (i < 20 ? 0.0 : 5.0) + 0.1 * ((i * 7) % 5 - 2);
        return x;
    }

    private static double[] Walk()
    {
        var x = new double[60];
        var v = 0.0;
        for (var i = 0; i < 60; i++)
        {
            v += ((i * 37) % 11 - 5) / 5.0;
            x[i] = v;
        }
        return x;
    }

    private static double[] Seasonal(int n, int period)
    {
        var x = new double[n];
        for (var i = 0; i < n; i++)
            x[i] = Math.Sin(2 * Math.PI * i / period) + 0.05 * ((i * 13) % 7 - 3);
        return x;
    }

    public static TheoryData<string> Methods() => [.. Cases.Keys];

    private static readonly Dictionary<string, Func<AnalyticsClient, JsonNode?>> Cases = new()
    {
        ["XbarRChart"] = c => Out(c.XbarRChart(Subgroups), AnalyticsJson.Default.XbarRChartResult),
        ["XbarRChart(rules)"] = c => Out(c.XbarRChart(Subgroups, [RunRule.BeyondLimits, RunRule.SixTrend]), AnalyticsJson.Default.XbarRChartResult),
        ["XbarSChart"] = c => Out(c.XbarSChart(Subgroups), AnalyticsJson.Default.XbarSChartResult),
        ["ImrChart"] = c => Out(c.ImrChart(Series), AnalyticsJson.Default.ImrChartResult),
        ["RunRules"] = c => Out(c.RunRules([0, 1, 4, 0.5, -0.2, 3.5], 3, 0, -3), AnalyticsJson.Default.IReadOnlyListChartPoint),
        ["PChart"] = c => Out(c.PChart([[3, 50], [5, 50], [2, 40], [4, 50], [6, 60]]), AnalyticsJson.Default.PChartResult),
        ["PChart(pBar)"] = c => Out(c.PChart([[3, 50], [5, 50]], pBar: 0.08), AnalyticsJson.Default.PChartResult),
        ["LaneyPChart"] = c => Out(c.LaneyPChart([[3, 50], [5, 50], [2, 40], [4, 50], [6, 60]]), AnalyticsJson.Default.LaneyPChartResult),
        ["NpChart"] = c => Out(c.NpChart([3, 5, 2, 4, 6], 50), AnalyticsJson.Default.FixedLimitChartResult),
        ["CChart"] = c => Out(c.CChart([3, 5, 2, 4, 6]), AnalyticsJson.Default.FixedLimitChartResult),
        ["UChart"] = c => Out(c.UChart([[3, 1.0], [5, 1.5], [2, 1.0], [4, 2.0]]), AnalyticsJson.Default.UChartResult),
        ["LaneyUChart"] = c => Out(c.LaneyUChart([[3, 1.0], [5, 1.5], [2, 1.0], [4, 2.0]]), AnalyticsJson.Default.LaneyUChartResult),
        ["GChart"] = c => Out(c.GChart([10, 20, 15, 30, 5]), AnalyticsJson.Default.GChartResult),
        ["TChart"] = c => Out(c.TChart([1.5, 2.0, 0.5, 3.0]), AnalyticsJson.Default.TChartResult),
        ["ProcessCapability"] = c => Out(c.ProcessCapability(Series, 11, 9, 10, 0.25), AnalyticsJson.Default.CapabilityResult),
        ["ProcessCapability(overall)"] = c => Out(c.ProcessCapability(Series, 11, null), AnalyticsJson.Default.CapabilityResult),
        ["PercentileCapability"] = c => Out(c.PercentileCapability(Series, 11, 9), AnalyticsJson.Default.PercentileCapabilityResult),
        ["GageRRXbarR"] = c => Out(c.GageRRXbarR(Gage(), 10), AnalyticsJson.Default.GageRRResult),
        ["GageRRAnova"] = c => Out(c.GageRRAnova(Gage()), AnalyticsJson.Default.GageRRAnovaResult),
        ["WeibullMle"] = c => Out(c.WeibullMle(Positive), AnalyticsJson.Default.WeibullMleResult),
        ["WeibullMrr"] = c => Out(c.WeibullMrr(Positive), AnalyticsJson.Default.WeibullMrrResult),
        ["WeibullReliability"] = c => Out(c.WeibullReliability(2, 100, [50, 100], [0.1]), AnalyticsJson.Default.WeibullReliabilityResult),
        ["BoxCoxCapability"] = c => Out(c.BoxCoxCapability(Positive, 120, 5), AnalyticsJson.Default.BoxCoxCapabilityResult),
        ["OneSampleTTest"] = c => Out(c.OneSampleTTest(Series, 10), AnalyticsJson.Default.TestResult),
        ["TwoSampleTTest"] = c => Out(c.TwoSampleTTest(Series[..10], Series[10..]), AnalyticsJson.Default.TestResult),
        ["PairedTTest"] = c => Out(c.PairedTTest(Series[..10], Series[10..]), AnalyticsJson.Default.TestResult),
        ["MannWhitneyUTest"] = c => Out(c.MannWhitneyUTest(Series[..10], Series[10..]), AnalyticsJson.Default.TestResult),
        ["WilcoxonSignedRankTest"] = c => Out(c.WilcoxonSignedRankTest(Series[..10], Series[10..]), AnalyticsJson.Default.TestResult),
        ["JarqueBeraTest"] = c => Out(c.JarqueBeraTest(Series), AnalyticsJson.Default.TestResult),
        ["ShapiroWilkTest"] = c => Out(c.ShapiroWilkTest(Series), AnalyticsJson.Default.ShapiroWilkResult),
        ["AndersonDarlingTest"] = c => Out(c.AndersonDarlingTest(Series), AnalyticsJson.Default.AndersonDarlingResult),
        ["AdfTest"] = c => Out(c.AdfTest(Walk(), AdfModel.ConstantTrend, 2), AnalyticsJson.Default.AdfResult),
        ["MannKendallTest"] = c => Out(c.MannKendallTest(Series), AnalyticsJson.Default.MannKendallResult),
        ["OneWayAnova"] = c => Out(c.OneWayAnova(Subgroups), AnalyticsJson.Default.AnovaResult),
        ["KruskalWallisTest"] = c => Out(c.KruskalWallisTest(Subgroups), AnalyticsJson.Default.TestResult),
        ["LeveneTest"] = c => Out(c.LeveneTest(Subgroups), AnalyticsJson.Default.TestResult),
        ["BartlettTest"] = c => Out(c.BartlettTest(Subgroups), AnalyticsJson.Default.TestResult),
        ["ChiSquaredGoodnessOfFit"] = c => Out(c.ChiSquaredGoodnessOfFit([10, 20, 30], [20, 20, 20]), AnalyticsJson.Default.TestResult),
        ["ChiSquaredIndependence"] = c => Out(c.ChiSquaredIndependence([[10, 20], [30, 40]]), AnalyticsJson.Default.TestResult),
        ["FisherExactTest"] = c => Out(c.FisherExactTest([[3, 1], [1, 3]]), AnalyticsJson.Default.TestResult),
        ["BonferroniCorrection"] = c => Out(new ValuesBody(c.BonferroniCorrection([0.01, 0.04])), AnalyticsJson.Default.ValuesBody),
        ["BenjaminiHochberg"] = c => Out(new ValuesBody(c.BenjaminiHochberg([0.01, 0.04, 0.03])), AnalyticsJson.Default.ValuesBody),
        ["SigmaToPpm"] = c => Out(new ValueBody(c.SigmaToPpm(6)), AnalyticsJson.Default.ValueBody),
        ["PpmToSigma"] = c => Out(new ValueBody(c.PpmToSigma(3.4)), AnalyticsJson.Default.ValueBody),
        ["LaplaceTrendTest"] = c => Out(c.LaplaceTrendTest(Events, 42), AnalyticsJson.Default.TrendTestResult),
        ["MilHdbk189Test"] = c => Out(c.MilHdbk189Test(Events, null), AnalyticsJson.Default.TrendTestResult),
        ["PowerLawProcessFit"] = c => Out(c.PowerLawProcessFit(Events, 42), AnalyticsJson.Default.PowerLawFitResult),
        ["DetectChangepoints"] = c => Out(c.DetectChangepoints(Step()), AnalyticsJson.Default.ChangepointResult),
        ["DetectChangepoints(normal)"] = c => Out(c.DetectChangepoints(Step(), 10, 3, PeltCost.Normal), AnalyticsJson.Default.ChangepointResult),
        ["DetectChangepointsMulti"] = c => Out(c.DetectChangepointsMulti([Step(), Step()]), AnalyticsJson.Default.ChangepointResult),
        ["Cusum"] = c => Out(c.Cusum(Series, 10, 0.25), AnalyticsJson.Default.CusumResult),
        ["Ewma"] = c => Out(c.Ewma(Series, 10, 0.25, 0.3, 2.8), AnalyticsJson.Default.EwmaResult),
        ["EstimatePeriod"] = c => Out(c.EstimatePeriod(Seasonal(84, 7)), AnalyticsJson.Default.PeriodResult),
        ["EstimatePeriod(none)"] = c => Out(c.EstimatePeriod(Series), AnalyticsJson.Default.PeriodResult),
        ["SpectralResidual"] = c => Out(c.SpectralResidual(Seasonal(64, 8)), AnalyticsJson.Default.SpectralResidualResult),
        ["CorrelationMatrix"] = c => Out(c.CorrelationMatrix([Series, Positive.Concat(Positive).Take(20).ToArray(), Step()[..20]], CorrelationMethod.Kendall), AnalyticsJson.Default.CorrelationMatrixResult),
        ["SimpleRegression"] = c => Out(c.SimpleRegression(Positive, Positive.Select((v, i) => 2 * v + i % 3).ToArray()), AnalyticsJson.Default.RegressionResult),
        ["SimpleRegression(exact)"] = c => Out(c.SimpleRegression([1, 2, 3, 4], [3, 5, 7, 9]), AnalyticsJson.Default.RegressionResult),
        ["FitBest"] = c => Out(c.FitBest(Positive), AnalyticsJson.Default.IReadOnlyListDistributionFit),
    };

    private static JsonNode? Out<T>(T value, JsonTypeInfo<T> info) => JsonSerializer.SerializeToNode(value, info);

    [Theory]
    [MemberData(nameof(Methods))]
    public void Record_carries_exactly_what_the_engine_sent(string method)
    {
        using var client = new AnalyticsClient();
        string? raw = null;
        client.ResponseObserver = body => raw = body;

        var written = Cases[method](client);

        Assert.NotNull(raw);
        var difference = Difference(JsonNode.Parse(raw!), written, "$");
        Assert.True(difference is null, $"{method}: {difference}\nengine: {raw}\nrecord: {written?.ToJsonString()}");
    }

    [Fact]
    public void Every_public_method_has_a_contract_case()
    {
        var methods = typeof(AnalyticsClient)
            .GetMethods(System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.DeclaredOnly)
            .Select(m => m.Name)
            .Where(n => n is not ("GetVersion" or "Dispose") && !n.StartsWith("get_") && !n.StartsWith("set_"))
            .ToHashSet();
        var covered = Cases.Keys.Select(k => k.Split('(')[0]).ToHashSet();
        Assert.Empty(methods.Except(covered));
    }

    [Fact]
    public void The_suite_runs_as_a_trimmed_host_does()
    {
        Assert.False(JsonSerializer.IsReflectionEnabledByDefault);
        // What the client did up to 0.13.0, and what a trimmed or NativeAOT consumer saw.
        var e = Assert.Throws<InvalidOperationException>(() =>
            JsonSerializer.Serialize(new { subgroups = new[] { 1.0 } }, new JsonSerializerOptions()));
        Assert.Contains("Reflection-based serialization has been disabled", e.Message);
    }

    /// <summary>Where two JSON trees differ, or null when they are the same.</summary>
    private static string? Difference(JsonNode? expected, JsonNode? actual, string path)
    {
        switch (expected, actual)
        {
            case (null, null):
                return null;
            case (null, _) or (_, null):
                return $"{path}: {expected?.ToJsonString() ?? "null"} vs {actual?.ToJsonString() ?? "null"}";
            case (JsonObject e, JsonObject a):
            {
                var missing = e.Select(p => p.Key).Except(a.Select(p => p.Key)).ToList();
                var extra = a.Select(p => p.Key).Except(e.Select(p => p.Key)).ToList();
                if (missing.Count > 0 || extra.Count > 0)
                    return $"{path}: record lacks [{string.Join(", ", missing)}], adds [{string.Join(", ", extra)}]";
                foreach (var (key, value) in e)
                {
                    if (Difference(value, a[key], $"{path}.{key}") is { } d)
                        return d;
                }
                return null;
            }
            case (JsonArray e, JsonArray a):
            {
                if (e.Count != a.Count)
                    return $"{path}: {e.Count} elements vs {a.Count}";
                for (var i = 0; i < e.Count; i++)
                {
                    if (Difference(e[i], a[i], $"{path}[{i}]") is { } d)
                        return d;
                }
                return null;
            }
            case (JsonValue e, JsonValue a) when e.GetValueKind() == JsonValueKind.Number && a.GetValueKind() == JsonValueKind.Number:
                return e.GetValue<double>() == a.GetValue<double>() ? null : $"{path}: {e} vs {a}";
            default:
                return JsonNode.DeepEquals(expected, actual) ? null : $"{path}: {expected.ToJsonString()} vs {actual.ToJsonString()}";
        }
    }
}
