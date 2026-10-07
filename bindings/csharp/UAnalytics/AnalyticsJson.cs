using System.Text.Json.Serialization;

namespace UAnalytics;

/// <summary>
/// Source-generated (de)serialization of the engine's results: no reflection, so the
/// client works in trimmed and NativeAOT hosts. A missing key, or a null where the record
/// does not allow one, fails the call instead of reading as a default.
/// </summary>
[JsonSourceGenerationOptions(
    PropertyNamingPolicy = JsonKnownNamingPolicy.SnakeCaseLower,
    RespectNullableAnnotations = true,
    RespectRequiredConstructorParameters = true)]
[JsonSerializable(typeof(XbarRChartResult))]
[JsonSerializable(typeof(XbarSChartResult))]
[JsonSerializable(typeof(ImrChartResult))]
[JsonSerializable(typeof(IReadOnlyList<ChartPoint>))]
[JsonSerializable(typeof(PChartResult))]
[JsonSerializable(typeof(LaneyPChartResult))]
[JsonSerializable(typeof(FixedLimitChartResult))]
[JsonSerializable(typeof(UChartResult))]
[JsonSerializable(typeof(LaneyUChartResult))]
[JsonSerializable(typeof(GChartResult))]
[JsonSerializable(typeof(TChartResult))]
[JsonSerializable(typeof(CapabilityResult))]
[JsonSerializable(typeof(PercentileCapabilityResult))]
[JsonSerializable(typeof(BoxCoxCapabilityResult))]
[JsonSerializable(typeof(GageRRResult))]
[JsonSerializable(typeof(GageRRAnovaResult))]
[JsonSerializable(typeof(WeibullMleResult))]
[JsonSerializable(typeof(WeibullMrrResult))]
[JsonSerializable(typeof(WeibullReliabilityResult))]
[JsonSerializable(typeof(TrendTestResult))]
[JsonSerializable(typeof(PowerLawFitResult))]
[JsonSerializable(typeof(TestResult))]
[JsonSerializable(typeof(ShapiroWilkResult))]
[JsonSerializable(typeof(AndersonDarlingResult))]
[JsonSerializable(typeof(AdfResult))]
[JsonSerializable(typeof(MannKendallResult))]
[JsonSerializable(typeof(AnovaResult))]
[JsonSerializable(typeof(ChangepointResult))]
[JsonSerializable(typeof(CusumResult))]
[JsonSerializable(typeof(EwmaResult))]
[JsonSerializable(typeof(PeriodResult))]
[JsonSerializable(typeof(SpectralResidualResult))]
[JsonSerializable(typeof(CorrelationMatrixResult))]
[JsonSerializable(typeof(RegressionResult))]
[JsonSerializable(typeof(IReadOnlyList<DistributionFit>))]
[JsonSerializable(typeof(ValueBody))]
[JsonSerializable(typeof(ValuesBody))]
[JsonSerializable(typeof(RunRule))]
[JsonSerializable(typeof(AdfModel))]
[JsonSerializable(typeof(PeltCost))]
[JsonSerializable(typeof(CorrelationMethod))]
internal sealed partial class AnalyticsJson : JsonSerializerContext;
