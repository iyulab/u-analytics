using System.Text.Json.Serialization;

namespace UAnalytics;

// The engine's results, one record per shape. Property names map to the engine's
// snake_case keys; a key the engine does not send fails the call rather than
// reading as zero, and a null where the record has no `?` fails it too.

// ── Run rules ──

/// <summary>
/// A run test: the vocabulary <see cref="ChartPoint.Violations"/> reports and the
/// <c>rules</c> argument of the charts and <see cref="AnalyticsClient.RunRules(double[], double, double, double, IReadOnlyList{RunRule}?)"/>
/// takes. The first eight are Nelson's tests 1–8 in his numbering; <see cref="EightOneSide"/>
/// is the Western Electric run rule, which asks for eight points on one side where Nelson's
/// test 2 asks for nine. The named sets are <see cref="RunRuleSet"/>.
/// </summary>
[JsonConverter(typeof(JsonStringEnumConverter<RunRule>))]
public enum RunRule
{
    /// <summary>One point beyond a control limit (Nelson 1).</summary>
    BeyondLimits,
    /// <summary>Nine points in a row on one side of the centre line (Nelson 2).</summary>
    NineOneSide,
    /// <summary>Six points in a row steadily increasing or decreasing (Nelson 3).</summary>
    SixTrend,
    /// <summary>Fourteen points in a row alternating up and down (Nelson 4).</summary>
    FourteenAlternating,
    /// <summary>Two of three points beyond 2σ on one side (Nelson 5).</summary>
    TwoOfThreeBeyond2Sigma,
    /// <summary>Four of five points beyond 1σ on one side (Nelson 6).</summary>
    FourOfFiveBeyond1Sigma,
    /// <summary>Fifteen points in a row within 1σ (Nelson 7).</summary>
    FifteenWithin1Sigma,
    /// <summary>Eight points in a row beyond 1σ on either side (Nelson 8).</summary>
    EightBeyond1Sigma,
    /// <summary>Eight points in a row on one side of the centre line (Western Electric 4).</summary>
    EightOneSide,
}

/// <summary>
/// A named set of run tests, as the engine defines it. Each added test raises the false alarm
/// rate of a process that is in control, so a chart applies <see cref="Shewhart"/> unless told
/// otherwise; the supplementary sets are for bringing a process into control (ISO 7870-2:2023, 8.2.7).
/// </summary>
[JsonConverter(typeof(JsonStringEnumConverter<RunRuleSet>))]
public enum RunRuleSet
{
    /// <summary>The 3-sigma test alone (<see cref="RunRule.BeyondLimits"/>) — the default.</summary>
    [JsonStringEnumMemberName("shewhart")] Shewhart,
    /// <summary>Western Electric's four: Nelson 1, 5 and 6, and <see cref="RunRule.EightOneSide"/>.</summary>
    [JsonStringEnumMemberName("western_electric")] WesternElectric,
    /// <summary>Nelson's eight tests.</summary>
    [JsonStringEnumMemberName("nelson")] Nelson,
}

// ── Variables charts ──

/// <summary>A point on a variables chart and the run tests it fails.</summary>
public sealed record ChartPoint(int Index, double Value, IReadOnlyList<RunRule> Violations);

/// <summary>
/// X̄-R chart. <see cref="SigmaHat"/> is R̄ / d2; <see cref="SigmaHatDf"/> its degrees of freedom
/// (David's approximation) — pass both to <see cref="AnalyticsClient.ProcessCapability"/> for Cp/Cpk intervals.
/// </summary>
public sealed record XbarRChartResult(
    double XbarCl, double XbarUcl, double XbarLcl,
    double RCl, double RUcl, double RLcl,
    double SigmaHat,
    double SigmaHatDf,
    IReadOnlyList<ChartPoint> XbarPoints,
    IReadOnlyList<ChartPoint> RPoints,
    bool InControl);

/// <summary>X̄-S chart. <see cref="SigmaHat"/> is S̄ / c4; <see cref="SigmaHatDf"/> its degrees of freedom.</summary>
public sealed record XbarSChartResult(
    double XbarCl, double XbarUcl, double XbarLcl,
    double SCl, double SUcl, double SLcl,
    double SigmaHat,
    double SigmaHatDf,
    IReadOnlyList<ChartPoint> XbarPoints,
    IReadOnlyList<ChartPoint> SPoints,
    bool InControl);

/// <summary>Individual / moving-range chart. <see cref="SigmaHat"/> is MR̄ / d2(2); <see cref="SigmaHatDf"/> is 0.62 (n − 1).</summary>
public sealed record ImrChartResult(
    double ICl, double IUcl, double ILcl,
    double MrCl, double MrUcl, double MrLcl,
    double SigmaHat,
    double SigmaHatDf,
    IReadOnlyList<ChartPoint> IPoints,
    IReadOnlyList<ChartPoint> MrPoints,
    bool InControl);

// ── Attributes charts ──

/// <summary>
/// A point on an attributes chart with its own limits. <see cref="Z"/> is the point in its
/// own standard errors (every limit is ±3 on that scale); <c>null</c> on charts whose limits
/// do not vary.
/// </summary>
public sealed record AttributeChartPoint(
    int Index, double Value, double Ucl, double Cl, double Lcl, bool OutOfControl, double? Z);

/// <summary>P chart. <see cref="PBar"/> is the centre line used.</summary>
public sealed record PChartResult(double PBar, IReadOnlyList<AttributeChartPoint> Points, bool InControl);

/// <summary>Laney P′ chart. <see cref="PBar"/> and <see cref="Phi"/> are the values used.</summary>
public sealed record LaneyPChartResult(double PBar, double Phi, IReadOnlyList<AttributeChartPoint> Points);

/// <summary>A chart with one set of limits for every point: NP and C charts.</summary>
public sealed record FixedLimitChartResult(
    double Cl, double Ucl, double Lcl, IReadOnlyList<AttributeChartPoint> Points, bool InControl);

/// <summary>U chart. <see cref="UBar"/> is the centre line used.</summary>
public sealed record UChartResult(double UBar, IReadOnlyList<AttributeChartPoint> Points, bool InControl);

/// <summary>Laney U′ chart. <see cref="UBar"/> and <see cref="Phi"/> are the values used.</summary>
public sealed record LaneyUChartResult(double UBar, double Phi, IReadOnlyList<AttributeChartPoint> Points);

/// <summary>A point on a rare-event (G or T) chart.</summary>
public sealed record RareEventChartPoint(
    int Index, double Value, double Ucl, double Cl, double Lcl, bool OutOfControl);

/// <summary>G chart: conforming counts between events.</summary>
public sealed record GChartResult(double GBar, IReadOnlyList<RareEventChartPoint> Points);

/// <summary>T chart: times between events.</summary>
public sealed record TChartResult(double TBar, IReadOnlyList<RareEventChartPoint> Points);

// ── Capability ──

/// <summary>Which sigma the short-term indices were computed from.</summary>
[JsonConverter(typeof(JsonStringEnumConverter<SigmaSource>))]
public enum SigmaSource
{
    /// <summary>The <c>sigmaWithin</c> the caller supplied; short-term indices are present.</summary>
    [JsonStringEnumMemberName("within")] Within,
    /// <summary>No within-subgroup sigma was supplied; short-term indices are <c>null</c>.</summary>
    [JsonStringEnumMemberName("overall")] Overall,
}

/// <summary>
/// Capability indices. Short-term indices (<see cref="Cp"/>, <see cref="Cpk"/>, <see cref="Cpu"/>,
/// <see cref="Cpl"/>) use the within-subgroup sigma and are <c>null</c> without one; an index
/// the given limits do not define is <c>null</c>.
/// </summary>
public sealed record CapabilityResult(
    double Mean,
    SigmaSource SigmaSource,
    double? StdDevWithin,
    double StdDevOverall,
    double? Cp, double? Cpk, double? Cpu, double? Cpl,
    double? Pp, double? Ppk, double? Ppu, double? Ppl,
    double? Cpm,
    CapabilityIntervals Intervals);

/// <summary>A two-sided confidence interval.</summary>
public sealed record ConfidenceInterval(double Lower, double Upper);

/// <summary>
/// Confidence intervals of the capability indices at <see cref="Confidence"/>: Cp/Pp from the
/// chi-square distribution of the variance, Cpk/Ppk by Bissell's approximation. <c>null</c> where
/// the index is, and for <see cref="Cp"/>/<see cref="Cpk"/> unless the within sigma's degrees of
/// freedom were given. A one-sided 95 % lower bound is the <c>Lower</c> at confidence 0.90.
/// </summary>
public sealed record CapabilityIntervals(
    double Confidence,
    ConfidenceInterval? Cp, ConfidenceInterval? Cpk,
    ConfidenceInterval? Pp, ConfidenceInterval? Ppk);

/// <summary>Percentile (ISO 22514-2) capability for non-normal data.</summary>
public sealed record PercentileCapabilityResult(
    double? CpStar, double? CpkStar, double? CpuStar, double? CplStar,
    double Median, double PercentileLower, double PercentileUpper,
    DistributionFamily Distribution, IReadOnlyDictionary<string, double> Parameters);

/// <summary>
/// A continuous distribution family, by the names the engine uses: the families
/// <see cref="AnalyticsClient.FitBest"/> fits and a percentile capability's quantiles come from.
/// </summary>
[JsonConverter(typeof(JsonStringEnumConverter<DistributionFamily>))]
public enum DistributionFamily
{
    /// <summary>Normal (<c>mu</c>, <c>sigma</c>).</summary>
    [JsonStringEnumMemberName("Normal")] Normal,
    /// <summary>Log-normal (<c>mu</c>, <c>sigma</c> of the log); positive data.</summary>
    [JsonStringEnumMemberName("LogNormal")] LogNormal,
    /// <summary>Two-parameter Weibull (<c>shape</c>, <c>scale</c>); positive data.</summary>
    [JsonStringEnumMemberName("Weibull")] Weibull,
    /// <summary>Gamma (<c>alpha</c>, rate <c>beta</c>); positive data.</summary>
    [JsonStringEnumMemberName("Gamma")] Gamma,
    /// <summary>Exponential (<c>lambda</c>); positive data.</summary>
    [JsonStringEnumMemberName("Exponential")] Exponential,
    /// <summary>Beta (<c>alpha</c>, <c>beta</c>); data within (0, 1).</summary>
    [JsonStringEnumMemberName("Beta")] Beta,
}

/// <summary>
/// Capability on the Box-Cox transformed scale. <see cref="LambdaAtBound"/> says the
/// likelihood's best λ lies at the edge of the searched range.
/// </summary>
public sealed record BoxCoxCapabilityResult(
    double Lambda, bool LambdaAtBound,
    double? Cp, double? Cpk, double? Cpu, double? Cpl,
    double? Pp, double? Ppk, double? Ppu, double? Ppl,
    double? Cpm);

// ── MSA ──

/// <summary>AIAG verdict on a measurement system's %GRR.</summary>
[JsonConverter(typeof(JsonStringEnumConverter<GageStatus>))]
public enum GageStatus
{
    /// <summary>%GRR ≤ 10 %.</summary>
    Acceptable,
    /// <summary>10 % &lt; %GRR ≤ 30 %.</summary>
    Marginal,
    /// <summary>%GRR &gt; 30 %.</summary>
    Unacceptable,
}

/// <summary>Centre line and limits of one of the Average &amp; Range method's charts.</summary>
public sealed record GageChartLimits(double Center, double Ucl, double Lcl);

/// <summary>Gage R&amp;R by the Average &amp; Range method, with the range and average charts it reads.</summary>
public sealed record GageRRResult(
    GageChartLimits RangeChart,
    GageChartLimits AverageChart,
    double Ev, double Av, double Grr, double Pv, double Tv,
    double PercentEv, double PercentAv, double PercentGrr, double PercentPv,
    double? PercentTolerance,
    int Ndc,
    GageStatus Status);

/// <summary>One row of a Gage R&amp;R ANOVA table. F and p are <c>null</c> on the rows that have none.</summary>
public sealed record AnovaTableRow(string Source, double Df, double Ss, double Ms, double? FValue, double? PValue);

/// <summary>Variance components of a Gage R&amp;R ANOVA.</summary>
public sealed record VarianceComponents(
    double Part, double Operator, double Interaction,
    double Repeatability, double Reproducibility, double Total);

/// <summary>Gage R&amp;R by the ANOVA method.</summary>
public sealed record GageRRAnovaResult(
    IReadOnlyList<AnovaTableRow> AnovaTable,
    VarianceComponents VarianceComponents,
    double Ev, double Av, double Grr, double Pv, double Tv,
    double PercentGrr,
    double? PercentTolerance,
    int Ndc,
    GageStatus Status,
    bool InteractionSignificant,
    bool InteractionPooled);

// ── Reliability ──

/// <summary>Weibull maximum-likelihood fit.</summary>
public sealed record WeibullMleResult(double Shape, double Scale, double LogLikelihood, int Iterations,
    WeibullIntervals Intervals);

/// <summary>
/// Two-sided intervals of a Weibull MLE at <see cref="Confidence"/>, normal on the log scale from
/// the observed Fisher information.
/// </summary>
public sealed record WeibullIntervals(double Confidence, ConfidenceInterval Shape, ConfidenceInterval Scale);

/// <summary>Weibull median-rank-regression fit.</summary>
public sealed record WeibullMrrResult(double Shape, double Scale, double RSquared);

/// <summary>
/// Reliability metrics of a Weibull. <see cref="Reliability"/> and <see cref="HazardRate"/> are
/// aligned with the requested times, <see cref="BLife"/> with the requested fractions failed.
/// </summary>
public sealed record WeibullReliabilityResult(
    double Mtbf, IReadOnlyList<double> Reliability, IReadOnlyList<double> HazardRate, IReadOnlyList<double> BLife);

// ── Event-time trend ──

/// <summary>Which way an event rate is moving.</summary>
[JsonConverter(typeof(JsonStringEnumConverter<TrendDirection>))]
public enum TrendDirection
{
    /// <summary>Events are coming more often.</summary>
    [JsonStringEnumMemberName("increasing")] Increasing,
    /// <summary>Events are coming less often.</summary>
    [JsonStringEnumMemberName("decreasing")] Decreasing,
    /// <summary>No direction in the statistic.</summary>
    [JsonStringEnumMemberName("flat")] Flat,
}

/// <summary>
/// A trend test of event times. <see cref="PValue"/> is two-sided; <see cref="EventsUsed"/> is
/// how many events entered the statistic; <see cref="Df"/> is the χ² degrees of freedom where
/// the test has them (MIL-HDBK-189), <c>null</c> otherwise (Laplace).
/// </summary>
public sealed record TrendTestResult(
    double Statistic, double PValue, TrendDirection Direction, int EventsUsed, double? Df);

/// <summary>
/// Power-law process (Crow-AMSAA) fit: expected events by t are λ·t^β, so β &lt; 1 means a
/// decreasing rate and β &gt; 1 an increasing one.
/// </summary>
public sealed record PowerLawFitResult(
    double Beta, double BetaUnbiased, double Lambda, double IntensityAtEnd, double End, int Events);

// ── Cumulative counters ──

/// <summary>What explains an increment of a cumulative counter.</summary>
[JsonConverter(typeof(JsonStringEnumConverter<IncrementKind>))]
public enum IncrementKind
{
    /// <summary>The value did not fall.</summary>
    [JsonStringEnumMemberName("normal")] Normal,
    /// <summary>A declared reset or replacement lies between the readings.</summary>
    [JsonStringEnumMemberName("declared")] Declared,
    /// <summary>The value fell and the register's modulus explains it.</summary>
    [JsonStringEnumMemberName("wrap")] Wrap,
    /// <summary>The value fell and nothing given explains it; the increase is unknown.</summary>
    [JsonStringEnumMemberName("unexplained")] Unexplained,
}

/// <summary>
/// How much a cumulative counter advanced between two neighbouring readings, at
/// <c>Start</c> and <c>End</c>. <c>Increase</c> is <c>null</c> when <c>Kind</c> is
/// <see cref="IncrementKind.Unexplained"/>.
/// </summary>
public sealed record CounterIncrement(double Start, double End, double? Increase, IncrementKind Kind);

/// <summary>
/// The increments that end in one interval [<c>Start</c>, <c>End</c>): the sum of the known
/// increases, the number of pairs, and how many of them are unexplained.
/// </summary>
public sealed record CounterTotal(double Start, double End, double Increase, int Pairs, int Unexplained);

/// <summary>
/// A reset or replacement of a counter, declared by the caller: when it happened, the
/// register's value right after it (<c>Start</c>), and the old register's last value when it
/// was read (<c>Last</c>) — without it, what the counter advanced before the change is not
/// counted.
/// </summary>
public sealed record CounterDiscontinuity(double Time, double Start, double? Last = null);

// ── Hypothesis tests ──

/// <summary>A test statistic, its degrees of freedom (0 where the test has none) and its p value.</summary>
public sealed record TestResult(double Statistic, double Df, double PValue);

/// <summary>Shapiro-Wilk W and its p value.</summary>
public sealed record ShapiroWilkResult(double W, double PValue);

/// <summary>Anderson-Darling A², the small-sample A²* the p value uses, and the p value.</summary>
public sealed record AndersonDarlingResult(double Statistic, double StatisticModified, double PValue);

/// <summary>One significance level of an ADF test.</summary>
public sealed record AdfLevel(double Level, double CriticalValue, bool Rejected);

/// <summary>Augmented Dickey-Fuller test.</summary>
public sealed record AdfResult(double Statistic, int NLags, int NObs, IReadOnlyList<AdfLevel> Levels);

/// <summary>Deterministic terms of an ADF regression.</summary>
[JsonConverter(typeof(JsonStringEnumConverter<AdfModel>))]
public enum AdfModel
{
    /// <summary>An intercept (the usual choice).</summary>
    [JsonStringEnumMemberName("constant")] Constant,
    /// <summary>No deterministic terms.</summary>
    [JsonStringEnumMemberName("none")] None,
    /// <summary>An intercept and a linear trend.</summary>
    [JsonStringEnumMemberName("constant_trend")] ConstantTrend,
}

/// <summary>Mann-Kendall trend test with Kendall's τ and Sen's slope.</summary>
public sealed record MannKendallResult(
    long SStatistic, double Variance, double ZStatistic, double PValue, double KendallTau, double SenSlope);

/// <summary>One-way ANOVA.</summary>
public sealed record AnovaResult(
    double FStatistic, int DfBetween, int DfWithin, double PValue, double SsBetween, double SsWithin);

// ── Detection ──

/// <summary>Cost a PELT segment is scored by.</summary>
[JsonConverter(typeof(JsonStringEnumConverter<PeltCost>))]
public enum PeltCost
{
    /// <summary>Squared error: a change in the mean.</summary>
    [JsonStringEnumMemberName("l2")] L2,
    /// <summary>Gaussian likelihood: a change in the mean or the variance.</summary>
    [JsonStringEnumMemberName("normal")] Normal,
}

/// <summary>PELT changepoints (the first index of each new segment).</summary>
public sealed record ChangepointResult(IReadOnlyList<int> Changepoints, int NSegments);

/// <summary>A CUSUM point, on the standardized scale.</summary>
public sealed record CusumPoint(int Index, double SUpper, double SLower, bool Signal);

/// <summary>CUSUM chart. <see cref="H"/> is the decision interval used, in sigmas.</summary>
public sealed record CusumResult(
    double H, IReadOnlyList<CusumPoint> Points, IReadOnlyList<int> SignalIndices, bool InControl);

/// <summary>An EWMA point with its exact limits.</summary>
public sealed record EwmaPoint(int Index, double Ewma, double Ucl, double Lcl, bool Signal);

/// <summary>EWMA chart.</summary>
public sealed record EwmaResult(IReadOnlyList<EwmaPoint> Points, IReadOnlyList<int> SignalIndices, bool InControl);

/// <summary>A period that passed the periodogram threshold and the ACF check.</summary>
public sealed record PeriodCandidate(int Period, double Acf, int Bin, double Power, double PowerShare);

/// <summary>Dominant period of a series; <see cref="Period"/> is <c>null</c> when none is found.</summary>
public sealed record PeriodResult(
    int? Period, IReadOnlyList<PeriodCandidate> Candidates, int N, double AcfThreshold, double PowerThreshold);

/// <summary>A point scored by spectral residual, with its expected value and band.</summary>
public sealed record SpectralResidualPoint(
    int Index, double Value, double Saliency, double Score, double Expected,
    double Lower, double Upper, bool IsAnomaly, bool NearEdge);

/// <summary>Spectral residual scores; <see cref="Anomalies"/> lists the flagged indices.</summary>
public sealed record SpectralResidualResult(IReadOnlyList<SpectralResidualPoint> Points, IReadOnlyList<int> Anomalies);

// ── Correlation, regression, distributions ──

/// <summary>Correlation coefficient.</summary>
[JsonConverter(typeof(JsonStringEnumConverter<CorrelationMethod>))]
public enum CorrelationMethod
{
    /// <summary>Pearson's r.</summary>
    [JsonStringEnumMemberName("pearson")] Pearson,
    /// <summary>Spearman's ρ.</summary>
    [JsonStringEnumMemberName("spearman")] Spearman,
    /// <summary>Kendall's τ-b.</summary>
    [JsonStringEnumMemberName("kendall")] Kendall,
}

/// <summary>Correlation matrix: <c>Matrix[i][j]</c> is the correlation of variables i and j.</summary>
public sealed record CorrelationMatrixResult(IReadOnlyList<IReadOnlyList<double>> Matrix);

/// <summary>
/// Simple linear regression. A t or F with no finite value (an exact fit) is <c>null</c>.
/// </summary>
public sealed record RegressionResult(
    double Slope, double Intercept,
    double RSquared, double AdjustedRSquared,
    double SlopeSe, double InterceptSe,
    double? SlopeT, double? InterceptT,
    double SlopeP, double InterceptP,
    double ResidualSe,
    double? FStatistic,
    [property: JsonPropertyName("f_p_value")] double FPValue,
    IReadOnlyList<double> Residuals,
    IReadOnlyList<double> Fitted);

/// <summary>
/// One fitted continuous family: which one, its parameters by name, and its log-likelihood,
/// AIC and BIC. <see cref="Distribution"/> can be passed back to
/// <see cref="AnalyticsClient.PercentileCapability"/> as is.
/// </summary>
public sealed record DistributionFit(
    DistributionFamily Distribution, IReadOnlyDictionary<string, double> Parameters,
    double LogLikelihood, double Aic, double Bic);

// ── Bodies that carry one value ──

internal sealed record ValueBody(double Value);

internal sealed record ValuesBody(IReadOnlyList<double> Values);
