using System.Text.Json;
using Xunit;

namespace UAnalytics.Tests;

public class BehaviourTests
{
    private readonly AnalyticsClient _client = new();

    [Fact]
    public void A_non_finite_value_is_refused_where_it_sits()
    {
        var e = Assert.Throws<AnalyticsException>(() => _client.XbarRChart([[1, 2], [3, double.NaN]]));
        Assert.Equal("value_not_finite", e.ErrorCode);
        Assert.Equal("subgroups[1]", e.Parameter);
        Assert.Equal(1, e.Index);

        var limit = Assert.Throws<AnalyticsException>(() => _client.RunRules([1, 2, 3], double.PositiveInfinity, 0, -3));
        Assert.Equal("limits.ucl", limit.Parameter);
        Assert.Null(limit.Index);

        var end = Assert.Throws<AnalyticsException>(() => _client.LaplaceTrendTest([1, 2, 3], double.NaN));
        Assert.Equal("observation.end", end.Parameter);

        var gage = Assert.Throws<AnalyticsException>(() => _client.GageRRAnova([[[1, double.NegativeInfinity]]]));
        Assert.Equal("measurements[0][0]", gage.Parameter);
        Assert.Equal(1, gage.Index);
    }

    [Fact]
    public void A_ragged_subgroup_is_a_dimension_mismatch_at_its_row()
    {
        var e = Assert.Throws<AnalyticsException>(() => _client.XbarRChart([[1, 2, 3], [4, 5], [6, 7, 8]]));
        Assert.Equal("dimension_mismatch", e.ErrorCode);
        Assert.Equal("subgroups", e.Parameter);
        Assert.Equal(1, e.Index);
        Assert.Equal(3.0, e.Details!.Value.GetProperty("expected").GetDouble());
        Assert.Equal(2.0, e.Details!.Value.GetProperty("got").GetDouble());
    }

    [Fact]
    public void A_fitted_family_goes_back_into_percentile_capability_as_is()
    {
        double[] data = [.. Enumerable.Range(1, 40).Select(i => 10 + Math.Log(i))];
        var best = _client.FitBest(data)[0];
        var capability = _client.PercentileCapability(data, usl: 20, lsl: 5, distribution: best.Distribution);
        Assert.Equal(best.Distribution, capability.Distribution);
    }

    [Fact]
    public void Percentile_capability_says_when_its_family_does_not_fit()
    {
        double[] uniform = [.. Enumerable.Range(1, 1000).Select(i => (double)i)];
        var capability = _client.PercentileCapability(uniform, usl: 1100, lsl: 0);
        Assert.Equal(199, capability.GoodnessOfFit.Replicates);
        Assert.Equal(1.0 / 200, capability.GoodnessOfFit.PValue);
        var same = _client.GoodnessOfFit(uniform, capability.Distribution);
        Assert.Equal(capability.GoodnessOfFit.Statistic, same.Statistic, 12);
        Assert.Null(_client.PercentileCapability(uniform, usl: 1100, lsl: 0, fitReplicates: 0).GoodnessOfFit.PValue);
    }

    [Fact]
    public void The_engine_still_names_what_it_refuses()
    {
        var e = Assert.Throws<AnalyticsException>(() => _client.TChart([1.0, -2.0, 3.0]));
        Assert.Equal(-3, e.Status);
        Assert.NotNull(e.ErrorCode);
        Assert.Equal(1, e.Index);
    }

    [Fact]
    public void Run_rules_round_trip_through_the_engine_vocabulary()
    {
        // A point beyond the limit and nothing else: only BeyondLimits can fire.
        var points = _client.RunRules([0, 0.5, 4, 0.2], 3, 0, -3, [RunRule.BeyondLimits]);
        Assert.Equal([RunRule.BeyondLimits], points[2].Violations);
        Assert.All(points.Where(p => p.Index != 2), p => Assert.Empty(p.Violations));

        // An empty rule set applies the limits only... and no rule at all fires.
        Assert.All(_client.RunRules([0, 0.5, 4, 0.2], 3, 0, -3, []), p => Assert.Empty(p.Violations));
    }

    [Fact]
    public void Survivors_lengthen_the_weibull_scale()
    {
        double[] failures = [55, 187, 216, 240, 244, 335];
        var alone = _client.WeibullMle(failures);
        var withSurvivors = _client.WeibullMle(failures, [361, 373, 375, 386]);
        Assert.True(withSurvivors.Scale > alone.Scale);
        Assert.InRange(withSurvivors.Shape, withSurvivors.Intervals.Shape.Lower, withSurvivors.Intervals.Shape.Upper);
        Assert.Equal(0.95, withSurvivors.Intervals.Confidence);
        var e = Assert.Throws<AnalyticsException>(() => _client.WeibullMle(failures, [10, -1]));
        Assert.Equal("suspension_times", e.Parameter);
        Assert.Equal(1, e.Index);
    }

    [Fact]
    public void Capability_intervals_bracket_the_indices()
    {
        double[] data = [9.8, 10.1, 10.0, 9.9, 10.2, 10.3, 9.7, 10.0, 10.1, 9.9, 10.4, 9.6, 10.0, 10.2, 9.8];
        var overall = _client.ProcessCapability(data, 11, 9);
        Assert.Equal(0.95, overall.Intervals.Confidence);
        Assert.InRange(overall.Ppk!.Value, overall.Intervals.Ppk!.Lower, overall.Intervals.Ppk.Upper);
        Assert.Null(overall.Intervals.Cpk);

        var within = _client.ProcessCapability(data, 11, 9, sigmaWithin: 0.2, sigmaWithinDf: 12, confidence: 0.90);
        Assert.Equal(0.90, within.Intervals.Confidence);
        Assert.InRange(within.Cpk!.Value, within.Intervals.Cpk!.Lower, within.Intervals.Cpk.Upper);

        double[][] groups = [[9.8, 10.1, 10.0, 9.9, 10.2], [10.3, 9.7, 10.0, 10.1, 9.9], [10.4, 9.6, 10.0, 10.2, 9.8]];
        var chart = _client.XbarRChart(groups);
        Assert.InRange(chart.SigmaHatDf, 0.1, 3 * 4);
        var composed = _client.ProcessCapability([.. groups.SelectMany(g => g)], 11, 9,
            sigmaWithin: chart.SigmaHat, sigmaWithinDf: chart.SigmaHatDf);
        Assert.NotNull(composed.Intervals.Cp);

        var e = Assert.Throws<AnalyticsException>(() => _client.ProcessCapability(data, 11, 9, sigmaWithinDf: 12));
        Assert.Equal("invalid_option", e.ErrorCode);
        Assert.Equal("sigma_within_df", e.Parameter);
    }

    [Fact]
    public void Named_sets_are_the_engines_and_the_default_is_the_three_sigma_test()
    {
        // Eight points above the centre line: Western Electric's run (eight) fires at the
        // eighth, Nelson's (nine) does not, and the default -- the 3-sigma test -- sees nothing.
        double[] eight = [1, 1, 1, 1, 1, 1, 1, 1];
        var we = _client.RunRules(eight, 3, 0, -3, RunRuleSet.WesternElectric);
        Assert.Equal([RunRule.EightOneSide], we[7].Violations);
        Assert.All(_client.RunRules(eight, 3, 0, -3, RunRuleSet.Nelson), p => Assert.Empty(p.Violations));
        Assert.All(_client.RunRules(eight, 3, 0, -3), p => Assert.Empty(p.Violations));
        Assert.Equal(
            _client.RunRules(eight, 3, 0, -3, RunRuleSet.Shewhart).Select(p => p.Violations.Count),
            _client.RunRules(eight, 3, 0, -3).Select(p => p.Violations.Count));
    }

    [Fact]
    public void Options_reach_the_engine()
    {
        double[][] variables = [[1, 2, 3, 4, 5, 6], [1, 3, 2, 5, 4, 6]];
        var pearson = _client.CorrelationMatrix(variables).Matrix[0][1];
        var kendall = _client.CorrelationMatrix(variables, CorrelationMethod.Kendall).Matrix[0][1];
        Assert.Equal(11.0 / 15.0, kendall, 12); // (C − D) / (n(n−1)/2) = (13 − 2) / 15
        Assert.NotEqual(pearson, kendall);

        var series = Enumerable.Range(0, 40).Select(i => Math.Sin(i * 0.7) + ((i * 37) % 11 - 5) / 4.0).ToArray();
        Assert.NotEqual(
            _client.AdfTest(series, AdfModel.None, 1).Statistic,
            _client.AdfTest(series, AdfModel.ConstantTrend, 1).Statistic);
    }

    [Fact]
    public void A_counter_classifies_each_increment_rather_than_assuming()
    {
        // A six-digit register passing 999 999, then a fall nothing explains.
        var steps = _client.CounterIncrements([0, 1, 2, 3], [999_995, 999_998, 4, 1], modulus: 1_000_000, maxRate: 100);
        Assert.Equal([IncrementKind.Normal, IncrementKind.Wrap, IncrementKind.Unexplained], steps.Select(s => s.Kind));
        Assert.Equal(6, steps[1].Increase);
        Assert.Null(steps[2].Increase);

        // A replacement between the readings: counted from the new register's start.
        var replaced = _client.CounterIncrements([0, 1, 2], [100, 120, 7],
            discontinuities: [new CounterDiscontinuity(1.5, 0, Last: 126)]);
        Assert.Equal(IncrementKind.Declared, replaced[1].Kind);
        Assert.Equal(13, replaced[1].Increase);

        var days = _client.CounterTotals([0, 6, 12, 24, 30], [0, 1, 3, 4, 2], [0, 24, 48]);
        Assert.Equal((3.0, 2, 0), (days[0].Increase, days[0].Pairs, days[0].Unexplained));
        Assert.Equal((1.0, 2, 1), (days[1].Increase, days[1].Pairs, days[1].Unexplained));

        var outside = Assert.Throws<AnalyticsException>(() => _client.CounterIncrements([0, 1], [1, 2],
            discontinuities: [new CounterDiscontinuity(5, 0)]));
        Assert.Equal("parameter_out_of_range", outside.ErrorCode);
        Assert.Equal("discontinuities.time", outside.Parameter);
        Assert.Equal(0, outside.Index);
    }

    [Fact]
    public void Trend_direction_is_typed()
    {
        // Gaps shrinking: events coming more often.
        double[] times = [10, 19, 27, 34, 40, 45, 49, 52, 54, 55];
        var laplace = _client.LaplaceTrendTest(times, 56);
        Assert.Equal(TrendDirection.Increasing, laplace.Direction);
        Assert.Null(laplace.Df);
        Assert.Equal(times.Length, laplace.EventsUsed);

        var mil = _client.MilHdbk189Test(times, null);
        Assert.Equal(TrendDirection.Increasing, mil.Direction);
        Assert.Equal(2.0 * (times.Length - 1), mil.Df);
    }

    [Fact]
    public void Capability_without_a_within_sigma_says_so()
    {
        var result = _client.ProcessCapability([9.8, 10.1, 10.0, 10.3, 9.9, 10.2], 11, 9);
        Assert.Equal(SigmaSource.Overall, result.SigmaSource);
        Assert.Null(result.Cp);
        Assert.NotNull(result.Pp);
    }

    [Fact]
    public void A_missing_key_fails_rather_than_reading_as_zero()
    {
        Assert.ThrowsAny<JsonException>(() =>
            JsonSerializer.Deserialize("""{"statistic": 1.5, "df": 2}""", AnalyticsJson.Default.TestResult));
        Assert.ThrowsAny<JsonException>(() =>
            JsonSerializer.Deserialize("""{"statistic": 1.5, "df": null, "p_value": 0.2}""", AnalyticsJson.Default.TestResult));
        Assert.ThrowsAny<JsonException>(() =>
            JsonSerializer.Deserialize("""{"statistic": 1, "p_value": 0.2, "direction": "sideways", "events_used": 3, "df": null}""",
                AnalyticsJson.Default.TrendTestResult));
    }
}
