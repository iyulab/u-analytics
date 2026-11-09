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
        Assert.InRange(chart.SigmaHatDf!.Value, 0.1, 3 * 4);
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
    public void An_equivalence_test_reads_its_margins_and_level_and_refuses_with_the_engines_codes()
    {
        double[] fill = [99.8, 100.1, 100.3, 99.9, 100.0, 100.2, 99.7, 100.1, 100.0, 99.9];
        var result = _client.TostOneSample(fill, 100, -0.5, 0.5);
        Assert.True(result.Equivalent);
        Assert.Equal(Math.Max(result.PLower, result.PUpper), result.PValue);
        Assert.Equal(9, result.Df);
        // A larger alpha is a narrower 1 - 2 alpha interval.
        var tight = _client.TostOneSample(fill, 100, -0.5, 0.5, 0.2);
        Assert.True(tight.CiUpper - tight.CiLower < result.CiUpper - result.CiLower);
        // Pooled variance is the pooled degrees of freedom.
        double[] other = [100.4, 99.6, 100.2, 100.0, 99.9, 100.3, 99.8, 100.1, 100.0];
        Assert.Equal(17, _client.TostTwoSample(fill, other, -0.5, 0.5, equalVariance: true).Df);
        Assert.NotEqual(17, _client.TostTwoSample(fill, other, -0.5, 0.5).Df);

        var margins = Assert.Throws<AnalyticsException>(() => _client.TostOneSample(fill, 100, 0.5, -0.5));
        Assert.Equal(("invalid_option", "high"), (margins.ErrorCode, margins.Parameter));
        var level = Assert.Throws<AnalyticsException>(() => _client.TostPaired(fill, fill[..10], -0.5, 0.5, 0.5));
        Assert.Equal(("parameter_out_of_range", "alpha"), (level.ErrorCode, level.Parameter));
        Assert.Equal(0.0, level.Details!.Value.GetProperty("exclusive_min").GetDouble());
        Assert.Equal(0.5, level.Details!.Value.GetProperty("exclusive_max").GetDouble());
        var flat = Assert.Throws<AnalyticsException>(() => _client.TostOneSample([2, 2, 2], 2, -1, 1));
        Assert.Equal(("degenerate_data", "data"), (flat.ErrorCode, flat.Parameter));
        var nan = Assert.Throws<AnalyticsException>(() => _client.TostTwoSample(fill, other, double.NaN, 0.5));
        Assert.Equal(("value_not_finite", "low"), (nan.ErrorCode, nan.Parameter));
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

    [Fact]
    public void A_forecast_total_and_when_it_reaches_a_level()
    {
        // About 760 a day, 4 820 in stock: the total crosses it in the 7th day.
        double[] use = [.. Enumerable.Range(0, 30).Select(i => 760.0 + 15 * Math.Sin(i))];
        var r = _client.ExponentialSmoothing(use, SmoothingMethod.Simple, horizon: 20,
            crossing: new Crossing(4820, [0.1, 0.5, 0.9], Paths: 2000, Seed: 7));

        var day7 = r.Forecasts[6];
        Assert.Equal(r.Forecasts.Take(7).Sum(f => f.Mean), day7.Cumulative.Mean, 9);
        var p = day7.Cumulative.Prediction!;
        Assert.True(p.Lower < day7.Cumulative.Mean && day7.Cumulative.Mean < p.Upper);

        var fp = r.FirstPassage!;
        Assert.Equal(4820, fp.Threshold);
        Assert.Equal(2000, fp.Paths);
        Assert.Equal(1.0, fp.Probability);
        var times = fp.Quantiles.Select(q => q.Time!.Value).ToArray();
        Assert.True(times[0] < times[1] && times[1] < times[2]);
        Assert.InRange(times[1], 6.0, 7.0);

        // The same seed, the same times; without crossing, no first passage.
        var again = _client.ExponentialSmoothing(use, SmoothingMethod.Simple, horizon: 20,
            crossing: new Crossing(4820, [0.1, 0.5, 0.9], Paths: 2000, Seed: 7));
        Assert.Equal(times, again.FirstPassage!.Quantiles.Select(q => q.Time!.Value).ToArray());
        Assert.Null(_client.ExponentialSmoothing(use, SmoothingMethod.Simple, horizon: 3).FirstPassage);
    }

    [Fact]
    public void A_crossing_the_model_cannot_follow_is_refused()
    {
        double[] use = [.. Enumerable.Range(0, 30).Select(i => 760.0 + 15 * Math.Sin(i))];
        var zero = Assert.Throws<AnalyticsException>(() => _client.ExponentialSmoothing(use, SmoothingMethod.Simple, horizon: 5, crossing: new Crossing(0)));
        Assert.Equal("invalid_option", zero.ErrorCode);
        Assert.Equal("crossing.threshold", zero.Parameter);

        var noHorizon = Assert.Throws<AnalyticsException>(() => _client.ExponentialSmoothing(use, SmoothingMethod.Simple, crossing: new Crossing(100)));
        Assert.Equal("parameter_out_of_range", noHorizon.ErrorCode);
        Assert.Equal("horizon", noHorizon.Parameter);

        var q = Assert.Throws<AnalyticsException>(() => _client.ExponentialSmoothing(use, SmoothingMethod.Simple, horizon: 5, crossing: new Crossing(100, [0.5, 1.5])));
        Assert.Equal("parameter_out_of_range", q.ErrorCode);
        Assert.Equal("crossing.quantiles", q.Parameter);
        Assert.Equal(1, q.Index);

        var mult = Assert.Throws<AnalyticsException>(() => _client.ExponentialSmoothing(use, SmoothingMethod.HoltWinters, period: 4,
            seasonality: Seasonality.Multiplicative, horizon: 5, crossing: new Crossing(100)));
        Assert.Equal("crossing", mult.Parameter);

        // At most 1e9 simulated steps: 100 000 steps take at most 10 000 futures.
        var many = Assert.Throws<AnalyticsException>(() => _client.ExponentialSmoothing(use, SmoothingMethod.Simple, horizon: 100_000, crossing: new Crossing(100, Paths: 20_000)));
        Assert.Equal("crossing.paths", many.Parameter);
        Assert.Equal(10_000, many.Details!.Value.GetProperty("max").GetDouble());
    }

    [Fact]
    public void A_variables_chart_given_its_sigma_builds_its_limits_on_it()
    {
        double[][] groups = [[9.8, 10.1, 10.0, 9.9, 10.2], [10.3, 9.7, 10.0, 10.1, 9.9], [10.4, 9.6, 10.0, 10.2, 9.8]];
        var free = _client.XbarRChart(groups);
        var fixedSigma = _client.XbarRChart(groups, sigma: 0.2);

        // The centre line is still the grand mean; the half-width is 3 sigma / sqrt(n).
        Assert.Equal(free.XbarCl, fixedSigma.XbarCl, 12);
        Assert.Equal(3 * 0.2 / Math.Sqrt(5), fixedSigma.XbarUcl - fixedSigma.XbarCl, 12);
        Assert.Equal(0.2, fixedSigma.SigmaHat);
        Assert.Null(fixedSigma.SigmaHatDf);
        Assert.NotNull(free.SigmaHatDf);

        var s = _client.XbarSChart(groups, sigma: 0.2);
        Assert.Equal(0.2, s.SigmaHat);
        var i = _client.ImrChart([1.0, 1.4, 0.9, 1.2], sigma: 0.2);
        Assert.Equal(3 * 0.2, i.IUcl - i.ICl, 12);

        var e = Assert.Throws<AnalyticsException>(() => _client.ImrChart([1.0, 1.4], sigma: 0.0));
        Assert.Equal("parameter_out_of_range", e.ErrorCode);
        Assert.Equal("sigma", e.Parameter);
    }
}
