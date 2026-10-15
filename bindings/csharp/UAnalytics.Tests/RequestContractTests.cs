using System.Reflection;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.Json.Serialization.Metadata;
using UAnalytics.Interop;
using Xunit;

namespace UAnalytics.Tests;

/// <summary>
/// The requests the client writes and the ones the engine reads are the same, both ways, against
/// the real native library with reflection-based serialization disabled: every key and name the
/// client can write is one the entry point reads (<c>request_keys.json</c>, which the engine's
/// tests measure from the entry points), every one the entry point reads can be written but for
/// the few listed with a reason, and a request with every setting given runs.
/// </summary>
public class RequestContractTests
{
    /// <summary>Places of a request the client does not write, each with why.</summary>
    private static readonly Dictionary<string, string> NotWritten = new()
    {
        ["uanalytics_detect_changepoints $.penalty=\"bic\""] =
            "BIC is the default penalty: the client asks for it by leaving the penalty out.",
        ["uanalytics_detect_changepoints_multi $.penalty=\"bic\""] =
            "BIC is the default penalty: the client asks for it by leaving the penalty out.",
    };

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

    private static readonly RunRule[] AllRules = Enum.GetValues<RunRule>();

    private static double[][][] Gage()
    {
        var parts = new double[10][][];
        for (var p = 0; p < 10; p++)
        {
            parts[p] = new double[3][];
            for (var o = 0; o < 3; o++)
                parts[p][o] = [p * 0.5 + o * 0.05 + 0.01 * ((p + o) % 3), p * 0.5 + o * 0.05 - 0.01 * ((p * o) % 2)];
        }
        return parts;
    }

    private static double[] Step() => [.. Enumerable.Range(0, 40).Select(i => (i < 20 ? 0.0 : 5.0) + 0.1 * ((i * 7) % 5 - 2))];

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
        => [.. Enumerable.Range(0, n).Select(i => Math.Sin(2 * Math.PI * i / period) + 0.05 * ((i * 13) % 7 - 3))];

    /// <summary>Every call with every setting it has given, and each option under each of its names.</summary>
    private static readonly Dictionary<string, Action<AnalyticsClient>> Full = new()
    {
        ["XbarRChart"] = c => c.XbarRChart(Subgroups, AllRules),
        ["XbarSChart"] = c => c.XbarSChart(Subgroups, AllRules),
        ["ImrChart"] = c => c.ImrChart(Series, AllRules),
        ["RunRules"] = c => c.RunRules([0, 1, 4, 0.5, -0.2, 3.5], 3, 0, -3, AllRules),
        ["XbarRChart(Shewhart)"] = c => c.XbarRChart(Subgroups, RunRuleSet.Shewhart),
        ["XbarSChart(WesternElectric)"] = c => c.XbarSChart(Subgroups, RunRuleSet.WesternElectric),
        ["ImrChart(Nelson)"] = c => c.ImrChart(Series, RunRuleSet.Nelson),
        ["RunRules(Shewhart)"] = c => c.RunRules([0, 1, 4, 0.5, -0.2, 3.5], 3, 0, -3, RunRuleSet.Shewhart),
        ["RunRules(WesternElectric)"] = c => c.RunRules([0, 1, 4, 0.5, -0.2, 3.5], 3, 0, -3, RunRuleSet.WesternElectric),
        ["RunRules(Nelson)"] = c => c.RunRules([0, 1, 4, 0.5, -0.2, 3.5], 3, 0, -3, RunRuleSet.Nelson),
        ["XbarRChart(WesternElectric)"] = c => c.XbarRChart(Subgroups, RunRuleSet.WesternElectric),
        ["XbarRChart(Nelson)"] = c => c.XbarRChart(Subgroups, RunRuleSet.Nelson),
        ["XbarSChart(Shewhart)"] = c => c.XbarSChart(Subgroups, RunRuleSet.Shewhart),
        ["XbarSChart(Nelson)"] = c => c.XbarSChart(Subgroups, RunRuleSet.Nelson),
        ["ImrChart(Shewhart)"] = c => c.ImrChart(Series, RunRuleSet.Shewhart),
        ["ImrChart(WesternElectric)"] = c => c.ImrChart(Series, RunRuleSet.WesternElectric),
        ["PChart"] = c => c.PChart([[3, 50], [5, 50], [2, 40]], pBar: 0.08),
        ["LaneyPChart"] = c => c.LaneyPChart([[3, 50], [5, 50], [2, 40]], pBar: 0.08, phi: 1.2),
        ["NpChart"] = c => c.NpChart([3, 5, 2, 4, 6], 50),
        ["CChart"] = c => c.CChart([3, 5, 2, 4, 6]),
        ["UChart"] = c => c.UChart([[3, 1.0], [5, 1.5], [2, 1.0]], uBar: 2.5),
        ["LaneyUChart"] = c => c.LaneyUChart([[3, 1.0], [5, 1.5], [2, 1.0]], uBar: 2.5, phi: 1.2),
        ["GChart"] = c => c.GChart([10, 20, 15, 30, 5]),
        ["TChart"] = c => c.TChart([1.5, 2.0, 0.5, 3.0]),
        ["ProcessCapability"] = c => c.ProcessCapability(Series, 11, 9, target: 10, sigmaWithin: 0.25, sigmaWithinDf: 15, confidence: 0.9),
        ["PercentileCapability"] = c => c.PercentileCapability(Series, 11, 9),
        ["PercentileCapability(Normal)"] = c => c.PercentileCapability(Series, 11, 9, DistributionFamily.Normal),
        ["PercentileCapability(LogNormal)"] = c => c.PercentileCapability(Series, 11, 9, DistributionFamily.LogNormal),
        ["PercentileCapability(Weibull)"] = c => c.PercentileCapability(Series, 11, 9, DistributionFamily.Weibull),
        ["PercentileCapability(Gamma)"] = c => c.PercentileCapability(Series, 11, 9, DistributionFamily.Gamma),
        ["PercentileCapability(Exponential)"] = c => c.PercentileCapability(Series, 11, 9, DistributionFamily.Exponential),
        ["PercentileCapability(Beta)"] = c => c.PercentileCapability([.. Series.Select(v => v / 20)], 0.55, 0.45, DistributionFamily.Beta),
        ["GageRRXbarR"] = c => c.GageRRXbarR(Gage(), tolerance: 10),
        ["GageRRAnova"] = c => c.GageRRAnova(Gage(), tolerance: 10),
        ["WeibullMle"] = c => c.WeibullMle(Positive, [95.0, 95.0, 100.0], 0.9),
        ["WeibullMrr"] = c => c.WeibullMrr(Positive),
        ["WeibullReliability"] = c => c.WeibullReliability(2, 100, [50, 100], [0.1], [[0.04, 0.1], [0.1, 25]], 0.9),
        ["BoxCoxCapability"] = c => c.BoxCoxCapability(Positive, 120, 5, (-2, 2)),
        ["SigmaToPpm"] = c => c.SigmaToPpm(6),
        ["PpmToSigma"] = c => c.PpmToSigma(3.4),
        ["OneSampleTTest"] = c => c.OneSampleTTest(Series, 10),
        ["TwoSampleTTest"] = c => c.TwoSampleTTest(Series[..10], Series[10..]),
        ["PairedTTest"] = c => c.PairedTTest(Series[..10], Series[10..]),
        ["MannWhitneyUTest"] = c => c.MannWhitneyUTest(Series[..10], Series[10..]),
        ["WilcoxonSignedRankTest"] = c => c.WilcoxonSignedRankTest(Series[..10], Series[10..]),
        ["JarqueBeraTest"] = c => c.JarqueBeraTest(Series),
        ["ShapiroWilkTest"] = c => c.ShapiroWilkTest(Series),
        ["AndersonDarlingTest"] = c => c.AndersonDarlingTest(Series),
        ["AdfTest(none)"] = c => c.AdfTest(Walk(), AdfModel.None, maxLags: 2),
        ["AdfTest(constant)"] = c => c.AdfTest(Walk(), AdfModel.Constant, maxLags: 2),
        ["AdfTest(constant_trend)"] = c => c.AdfTest(Walk(), AdfModel.ConstantTrend, maxLags: 2),
        ["MannKendallTest"] = c => c.MannKendallTest(Series),
        ["OneWayAnova"] = c => c.OneWayAnova(Subgroups),
        ["KruskalWallisTest"] = c => c.KruskalWallisTest(Subgroups),
        ["LeveneTest"] = c => c.LeveneTest(Subgroups),
        ["BartlettTest"] = c => c.BartlettTest(Subgroups),
        ["ChiSquaredGoodnessOfFit"] = c => c.ChiSquaredGoodnessOfFit([10, 20, 30], [20, 20, 20]),
        ["ChiSquaredIndependence"] = c => c.ChiSquaredIndependence([[10, 20], [30, 40]]),
        ["FisherExactTest"] = c => c.FisherExactTest([[3, 1], [1, 3]]),
        ["BonferroniCorrection"] = c => c.BonferroniCorrection([0.01, 0.04]),
        ["BenjaminiHochberg"] = c => c.BenjaminiHochberg([0.01, 0.04, 0.03]),
        ["CounterIncrements"] = c => c.CounterIncrements([0, 1, 2], [1, 2, 0.5], modulus: 10, maxRate: 100,
            discontinuities: [new CounterDiscontinuity(1.5, 0, 2.5)]),
        ["CounterTotals"] = c => c.CounterTotals([0, 1, 2], [1, 2, 3], [0, 2, 4], modulus: 10, maxRate: 100,
            discontinuities: [new CounterDiscontinuity(1.5, 0, 2.5)]),
        ["LaplaceTrendTest(time)"] = c => c.LaplaceTrendTest(Events, 42),
        ["LaplaceTrendTest(failure)"] = c => c.LaplaceTrendTest(Events, null),
        ["MilHdbk189Test(time)"] = c => c.MilHdbk189Test(Events, 42),
        ["MilHdbk189Test(failure)"] = c => c.MilHdbk189Test(Events, null),
        ["PowerLawProcessFit(time)"] = c => c.PowerLawProcessFit(Events, 42),
        ["PowerLawProcessFit(failure)"] = c => c.PowerLawProcessFit(Events, null),
        ["DetectChangepoints(l2)"] = c => c.DetectChangepoints(Step(), penalty: 10, minSegmentLen: 3, cost: PeltCost.L2),
        ["DetectChangepoints(normal)"] = c => c.DetectChangepoints(Step(), penalty: 10, minSegmentLen: 3, cost: PeltCost.Normal),
        ["DetectChangepointsMulti(l2)"] = c => c.DetectChangepointsMulti([Step(), Step()], penalty: 10, minSegmentLen: 3, cost: PeltCost.L2),
        ["DetectChangepointsMulti(normal)"] = c => c.DetectChangepointsMulti([Step(), Step()], penalty: 10, minSegmentLen: 3, cost: PeltCost.Normal),
        ["Cusum"] = c => c.Cusum(Series, 10, 0.25, k: 0.5, h: 5),
        ["Ewma"] = c => c.Ewma(Series, 10, 0.25, lambda: 0.3, lFactor: 2.8),
        ["EstimatePeriod"] = c => c.EstimatePeriod(Seasonal(84, 7)),
        ["SpectralResidual"] = c => c.SpectralResidual(Seasonal(64, 8), averagingWindow: 3, judgementWindow: 21,
            threshold: 3, minZscore: 1.5, sensitivity: 0.7, batchSize: 32),
        ["CorrelationMatrix(pearson)"] = c => c.CorrelationMatrix([Series, Step()[..20]], CorrelationMethod.Pearson),
        ["CorrelationMatrix(spearman)"] = c => c.CorrelationMatrix([Series, Step()[..20]], CorrelationMethod.Spearman),
        ["CorrelationMatrix(kendall)"] = c => c.CorrelationMatrix([Series, Step()[..20]], CorrelationMethod.Kendall),
        ["SimpleRegression"] = c => c.SimpleRegression([1, 2, 3, 4], [3, 5, 7.1, 9]),
        ["FitBest"] = c => c.FitBest(Positive),
    };

    /// <summary>
    /// Each call's request: (call, entry point, body). Whether the engine accepts it is
    /// <see cref="A_request_with_every_setting_is_accepted_and_runs"/>'s to say; a refused one is
    /// still a request the client wrote.
    /// </summary>
    private static readonly Lazy<List<(string Call, string Entry, string Body)>> Written = new(() =>
    {
        var written = new List<(string, string, string)>();
        foreach (var (call, run) in Full)
        {
            using var client = new AnalyticsClient();
            client.RequestObserver = (entry, body) => written.Add((call, entry, body));
            try
            {
                run(client);
            }
            catch (AnalyticsException)
            {
            }
        }
        return written;
    });

    public static TheoryData<string> Calls() => [.. Full.Keys];

    [Theory]
    [MemberData(nameof(Calls))]
    public void A_request_with_every_setting_is_accepted_and_runs(string call)
    {
        using var client = new AnalyticsClient();
        var sent = 0;
        client.RequestObserver = (_, _) => sent++;
        Full[call](client);
        Assert.Equal(1, sent);
    }

    [Fact]
    public void Every_key_and_name_the_client_writes_is_one_the_engine_reads()
    {
        var problems = new List<string>();
        foreach (var (call, entry, body) in Written.Value)
        {
            var found = new List<string>();
            RequestKeys.Read(RequestKeys.Schema(entry), JsonNode.Parse(body), "$", [], found);
            problems.AddRange(found.Select(p => $"{call} → {entry} {p}"));
        }
        Assert.True(problems.Count == 0, string.Join("\n", problems.Distinct()));
    }

    [Fact]
    public void Every_key_and_name_the_engine_reads_is_one_the_client_writes()
    {
        var written = new HashSet<string>();
        foreach (var (_, entry, body) in Written.Value)
        {
            var places = new HashSet<string>();
            RequestKeys.Read(RequestKeys.Schema(entry), JsonNode.Parse(body), "$", places, []);
            written.UnionWith(places.Select(p => $"{entry} {p}"));
        }
        var engine = RequestKeys.EntryPoints
            .SelectMany(e => RequestKeys.Places(e.Key).Select(p => $"{e.Key} {p}"))
            .ToHashSet();

        var unwritable = engine.Except(written).Except(NotWritten.Keys).Order(StringComparer.Ordinal).ToList();
        Assert.True(unwritable.Count == 0, $"the engine reads what the client cannot write:\n{string.Join("\n", unwritable)}");

        // A reason kept for a place the client does write, or the engine no longer has, is stale.
        var stale = NotWritten.Keys.Where(p => written.Contains(p) || !engine.Contains(p)).ToList();
        Assert.True(stale.Count == 0, $"stale reasons: {string.Join(", ", stale)}");
    }

    [Fact]
    public void Every_entry_point_the_engine_measures_is_bound()
    {
        var bound = typeof(NativeInterop)
            .GetMethods(BindingFlags.Public | BindingFlags.Static | BindingFlags.DeclaredOnly)
            .Select(m => m.Name)
            .Where(n => n is not ("uanalytics_version" or "uanalytics_codes" or "uanalytics_free_string"))
            .Order(StringComparer.Ordinal);
        Assert.Equal(RequestKeys.EntryPoints.Select(e => e.Key).Order(StringComparer.Ordinal), bound);
    }

    [Fact]
    public void Each_option_is_a_name_the_engine_reads()
    {
        Assert.Equal(RequestKeys.Names("uanalytics_xbar_r_chart", "$.rules[]"), Names(AllRules, AnalyticsJson.Default.RunRule));
        Assert.Equal(RequestKeys.Names("uanalytics_xbar_r_chart", "$.rules"), Names(Enum.GetValues<RunRuleSet>(), AnalyticsJson.Default.RunRuleSet));
        Assert.Equal(RequestKeys.Names("uanalytics_adf_test", "$.model"), Names(Enum.GetValues<AdfModel>(), AnalyticsJson.Default.AdfModel));
        Assert.Equal(RequestKeys.Names("uanalytics_detect_changepoints", "$.cost"), Names(Enum.GetValues<PeltCost>(), AnalyticsJson.Default.PeltCost));
        Assert.Equal(RequestKeys.Names("uanalytics_correlation_matrix", "$.method"), Names(Enum.GetValues<CorrelationMethod>(), AnalyticsJson.Default.CorrelationMethod));
        Assert.Equal(RequestKeys.Names("uanalytics_percentile_capability", "$.distribution"), Names(Enum.GetValues<DistributionFamily>(), AnalyticsJson.Default.DistributionFamily));
    }

    private static string[] Names<T>(T[] values, JsonTypeInfo<T> info)
        => [.. values.Select(v => JsonSerializer.SerializeToNode(v, info)!.GetValue<string>()).Order(StringComparer.Ordinal)];
}
