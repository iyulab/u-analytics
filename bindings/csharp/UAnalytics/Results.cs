using System.Text.Json.Serialization;

namespace UAnalytics;

// The engine's results, one record per shape. Property names map to the engine's
// snake_case keys; a key the engine does not send fails the call rather than
// reading as zero, and a null where the record has no `?` fails it too.

// ── Run rules ──

/// <summary>
/// A run test: the vocabulary <see cref="ChartPoint.Violations"/> reports and the
/// <c>rules</c> argument of the charts and <see cref="AnalyticsClient.RunRules"/> takes.
/// The first four are Nelson's tests 1–4 (and Western Electric's), the rest Nelson's 5–8.
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
}

// ── Variables charts ──

/// <summary>A point on a variables chart and the run tests it fails.</summary>
public sealed record ChartPoint(int Index, double Value, IReadOnlyList<RunRule> Violations);

/// <summary>X̄-R chart. <see cref="SigmaHat"/> is R̄ / d2.</summary>
public sealed record XbarRChartResult(
    double XbarCl, double XbarUcl, double XbarLcl,
    double RCl, double RUcl, double RLcl,
    double? SigmaHat,
    IReadOnlyList<ChartPoint> XbarPoints,
    IReadOnlyList<ChartPoint> RPoints,
    bool InControl);

/// <summary>X̄-S chart. <see cref="SigmaHat"/> is S̄ / c4.</summary>
public sealed record XbarSChartResult(
    double XbarCl, double XbarUcl, double XbarLcl,
    double SCl, double SUcl, double SLcl,
    double? SigmaHat,
    IReadOnlyList<ChartPoint> XbarPoints,
    IReadOnlyList<ChartPoint> SPoints,
    bool InControl);

/// <summary>Individual / moving-range chart. <see cref="SigmaHat"/> is MR̄ / d2(2).</summary>
public sealed record ImrChartResult(
    double ICl, double IUcl, double ILcl,
    double MrCl, double MrUcl, double MrLcl,
    double? SigmaHat,
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
    double? Cpm);

/// <summary>Percentile (ISO 22514-2) capability for non-normal data.</summary>
public sealed record PercentileCapabilityResult(
    double? CpStar, double? CpkStar, double? CpuStar, double? CplStar,
    double Median, double PercentileLower, double PercentileUpper);

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
public sealed record WeibullMleResult(double Shape, double Scale, double LogLikelihood, int Iterations);

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
/// One fitted continuous family: its name (<c>normal</c>, <c>weibull</c>, …), its parameters by
/// name, and its log-likelihood, AIC and BIC.
/// </summary>
public sealed record DistributionFit(
    string Distribution, IReadOnlyDictionary<string, double> Parameters,
    double LogLikelihood, double Aic, double Bic);

// ── Bodies that carry one value ──

internal sealed record ValueBody(double Value);

internal sealed record ValuesBody(IReadOnlyList<double> Values);
