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
        Assert.Equal("value_not_finite", e.Reason);
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
    public void The_engine_still_names_what_it_refuses()
    {
        var e = Assert.Throws<AnalyticsException>(() => _client.TChart([1.0, -2.0, 3.0]));
        Assert.Equal(-3, e.Code);
        Assert.NotNull(e.Reason);
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
