using System.Runtime.InteropServices;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.Json.Serialization.Metadata;
using UAnalytics.Interop;

namespace UAnalytics;

public sealed class AnalyticsClient : IDisposable
{
    private bool _disposed;

    /// <summary>Sees each raw response body before it is read — the contract tests compare the two.</summary>
    internal Action<string>? ResponseObserver { get; set; }

    public string GetVersion()
    {
        var ptr = NativeInterop.uanalytics_version();
        var version = Marshal.PtrToStringUTF8(ptr) ?? "unknown";
        NativeInterop.uanalytics_free_string(ptr);
        return version;
    }

    // ── SPC ──

    /// <summary>
    /// X-bar/R chart. Returns the same JSON the WASM binding does:
    /// <c>xbar_cl</c>/<c>xbar_ucl</c>/<c>xbar_lcl</c>, <c>r_cl</c>/<c>r_ucl</c>/<c>r_lcl</c>,
    /// <c>sigma_hat</c>, per-point <c>xbar_points</c>/<c>r_points</c> (each with
    /// <c>index</c>, <c>value</c>, <c>violations</c>) and <c>in_control</c>.
    /// <paramref name="rules"/> names the run tests to apply, in the vocabulary
    /// <c>violations</c> reports; <c>null</c> applies all eight Nelson tests, an
    /// empty array applies the control limits only.
    /// </summary>
    public XbarRChartResult XbarRChart(double[][] subgroups, IReadOnlyList<RunRule>? rules = null)
        => Call(NativeInterop.uanalytics_xbar_r_chart, AnalyticsJson.Default.XbarRChartResult,
            ("subgroups", Request.Rows("subgroups", subgroups)), ("rules", Request.Rules(rules)));

    /// <summary>
    /// P chart. Each sample is <c>[defectives, sampleSize]</c> -- the order the
    /// chart's own <c>add_sample</c> takes, and the order the WASM binding uses.
    /// Returns <c>p_bar</c> (the centre line used), per-point <c>points</c>
    /// (<c>index</c>, <c>value</c>, <c>ucl</c>, <c>cl</c>, <c>lcl</c>, <c>out_of_control</c>,
    /// and <c>z</c> -- the point in its own standard errors, on which every limit is ±3,
    /// so <see cref="RunRules"/> with limits (3, 0, −3) applies zone tests when sizes vary)
    /// and <c>in_control</c>. A sample with more defectives than its size, or a size of
    /// zero, is rejected by its row (<see cref="AnalyticsException.Index"/>) rather than dropped.
    /// </summary>
    /// <param name="samples"><c>[defectives, sampleSize]</c> pairs.</param>
    /// <param name="pBar">A known centre line from a Phase I study (Phase II): every limit
    /// uses it with each sample's own size instead of re-estimating p-bar from
    /// <paramref name="samples"/>. Strictly between 0 and 1.</param>
    public PChartResult PChart(ulong[][] samples, double? pBar = null)
        => Call(NativeInterop.uanalytics_p_chart, AnalyticsJson.Default.PChartResult,
            ("samples", Request.Counts(samples)), ("p_bar", Request.Num("p_bar", pBar)));

    /// <summary>
    /// Laney P' chart for over- or under-dispersed proportions. Same
    /// <c>[defectives, sampleSize]</c> samples as <see cref="PChart"/>; at least three
    /// when p-bar and phi are estimated, one when they are given.
    /// Returns <c>p_bar</c>, <c>phi</c> (the values used) and per-point <c>points</c>
    /// (with <c>z</c>, as for <see cref="PChart"/>).
    /// </summary>
    /// <param name="samples"><c>[defectives, sampleSize]</c> pairs.</param>
    /// <param name="pBar">Phase I p-bar; given together with <paramref name="phi"/> or not at all.</param>
    /// <param name="phi">Phase I sigma-inflation factor (≥ 0); given together with <paramref name="pBar"/>.</param>
    public LaneyPChartResult LaneyPChart(ulong[][] samples, double? pBar = null, double? phi = null)
        => Call(NativeInterop.uanalytics_laney_p_chart, AnalyticsJson.Default.LaneyPChartResult,
            ("samples", Request.Counts(samples)), ("p_bar", Request.Num("p_bar", pBar)),
            ("phi", Request.Num("phi", phi)));

    /// <summary>
    /// NP chart: defectives per subgroup when every subgroup has the same
    /// <paramref name="sampleSize"/>. Returns <c>cl</c>, <c>ucl</c>, <c>lcl</c>,
    /// per-point <c>points</c> and <c>in_control</c>.
    /// </summary>
    public FixedLimitChartResult NpChart(ulong[] defectives, ulong sampleSize)
        => Call(NativeInterop.uanalytics_np_chart, AnalyticsJson.Default.FixedLimitChartResult,
            ("defectives", Request.Counts(defectives)), ("sample_size", JsonValue.Create(sampleSize)));

    /// <summary>
    /// C chart: defects per inspection unit of one size. Returns <c>cl</c>, <c>ucl</c>,
    /// <c>lcl</c>, per-point <c>points</c> and <c>in_control</c>.
    /// </summary>
    public FixedLimitChartResult CChart(ulong[] defects)
        => Call(NativeInterop.uanalytics_c_chart, AnalyticsJson.Default.FixedLimitChartResult,
            ("defects", Request.Counts(defects)));

    /// <summary>
    /// U chart: defects per unit when the quantity inspected varies. Each sample is
    /// <c>[defects, units]</c> (<c>units</c> may be fractional, and must be positive).
    /// Returns <c>u_bar</c>, per-point <c>points</c> (with <c>z</c>) and <c>in_control</c>.
    /// </summary>
    /// <param name="samples"><c>[defects, units]</c> pairs.</param>
    /// <param name="uBar">A known centre line from a Phase I study (Phase II); positive.</param>
    public UChartResult UChart(double[][] samples, double? uBar = null)
        => Call(NativeInterop.uanalytics_u_chart, AnalyticsJson.Default.UChartResult,
            ("samples", Request.Rows("samples", samples)), ("u_bar", Request.Num("u_bar", uBar)));

    /// <summary>
    /// Laney U' chart for over- or under-dispersed rates. Same samples as
    /// <see cref="UChart"/>; at least three when u-bar and phi are estimated.
    /// Returns <c>u_bar</c>, <c>phi</c> and per-point <c>points</c> (with <c>z</c>).
    /// </summary>
    /// <param name="samples"><c>[defects, units]</c> pairs.</param>
    /// <param name="uBar">Phase I u-bar; given together with <paramref name="phi"/> or not at all.</param>
    /// <param name="phi">Phase I sigma-inflation factor (&#8805; 0); given together with <paramref name="uBar"/>.</param>
    public LaneyUChartResult LaneyUChart(double[][] samples, double? uBar = null, double? phi = null)
        => Call(NativeInterop.uanalytics_laney_u_chart, AnalyticsJson.Default.LaneyUChartResult,
            ("samples", Request.Rows("samples", samples)), ("u_bar", Request.Num("u_bar", uBar)),
            ("phi", Request.Num("phi", phi)));

    /// <summary>
    /// G chart for rare events: conforming counts between events (at least three,
    /// each &#8805; 0). Returns <c>g_bar</c> and per-point <c>points</c>.
    /// </summary>
    public GChartResult GChart(double[] gaps)
        => Call(NativeInterop.uanalytics_g_chart, AnalyticsJson.Default.GChartResult,
            ("gaps", Request.Nums("gaps", gaps)));

    /// <summary>
    /// T chart for rare events: times between events (at least three, each &gt; 0).
    /// Returns <c>t_bar</c> and per-point <c>points</c>.
    /// </summary>
    public TChartResult TChart(double[] times)
        => Call(NativeInterop.uanalytics_t_chart, AnalyticsJson.Default.TChartResult,
            ("times", Request.Nums("times", times)));

    /// <summary>
    /// X-bar/S chart. Same request and response shape as <see cref="XbarRChart"/> with
    /// <c>s_*</c> limits in place of <c>r_*</c>; the usual choice once subgroups exceed about
    /// ten values. Returns <c>sigma_hat</c> (<c>S-bar / c4</c>).
    /// </summary>
    public XbarSChartResult XbarSChart(double[][] subgroups, IReadOnlyList<RunRule>? rules = null)
        => Call(NativeInterop.uanalytics_xbar_s_chart, AnalyticsJson.Default.XbarSChartResult,
            ("subgroups", Request.Rows("subgroups", subgroups)), ("rules", Request.Rules(rules)));

    /// <summary>
    /// Individual / Moving-Range chart for a series of single observations. Returns
    /// <c>sigma_hat</c> (<c>MR-bar / d2(2)</c>) -- pass it as <c>sigmaWithin</c> to
    /// <see cref="ProcessCapability"/> for individual data, which no longer estimates one itself.
    /// </summary>
    public ImrChartResult ImrChart(double[] values, IReadOnlyList<RunRule>? rules = null)
        => Call(NativeInterop.uanalytics_imr_chart, AnalyticsJson.Default.ImrChartResult,
            ("values", Request.Nums("values", values)), ("rules", Request.Rules(rules)));

    /// <summary>
    /// Applies the run tests to a series against one set of limits -- the engine the charts
    /// use, callable on its own. One point per value, in order, each with its
    /// <c>violations</c>.
    /// </summary>
    public IReadOnlyList<ChartPoint> RunRules(double[] values, double ucl, double cl, double lcl,
        IReadOnlyList<RunRule>? rules = null)
        => Call(NativeInterop.uanalytics_run_rules, AnalyticsJson.Default.IReadOnlyListChartPoint,
            ("values", Request.Nums("values", values)),
            ("limits", Request.Body(
                ("ucl", Request.Num("limits.ucl", ucl)),
                ("cl", Request.Num("limits.cl", cl)),
                ("lcl", Request.Num("limits.lcl", lcl)))),
            ("rules", Request.Rules(rules)));

    // ── Capability ──

    /// <summary>
    /// Capability indices. <paramref name="sigmaWithin"/> is the short-term sigma from a
    /// control chart (the <c>sigma_hat</c> that <see cref="XbarRChart"/> returns). Without it
    /// the short-term indices (<c>cp</c>, <c>cpk</c>, <c>cpu</c>, <c>cpl</c>) are <c>null</c> and
    /// <c>sigma_source</c> is <c>"overall"</c> -- a flat vector carries no subgroup structure to
    /// estimate one from, and this client no longer guesses one from the moving range.
    /// Same JSON as the WASM binding.
    /// </summary>
    public CapabilityResult ProcessCapability(double[] data, double? usl, double? lsl, double? target = null,
        double? sigmaWithin = null)
        => Call(NativeInterop.uanalytics_process_capability, AnalyticsJson.Default.CapabilityResult,
            ("data", Request.Nums("data", data)), ("usl", Request.Num("usl", usl)),
            ("lsl", Request.Num("lsl", lsl)), ("target", Request.Num("target", target)),
            ("sigma_within", Request.Num("sigma_within", sigmaWithin)));

    /// <summary>
    /// Percentile capability (ISO 22514-2) for non-normal data: the 0.135 % and 99.865 %
    /// percentiles and the median stand in for μ ± 3σ and μ.
    /// </summary>
    public PercentileCapabilityResult PercentileCapability(double[] data, double? usl, double? lsl)
        => Call(NativeInterop.uanalytics_percentile_capability, AnalyticsJson.Default.PercentileCapabilityResult,
            ("data", Request.Nums("data", data)), ("usl", Request.Num("usl", usl)),
            ("lsl", Request.Num("lsl", lsl)));

    // ── MSA ──

    /// <summary>
    /// Gage R&amp;R by the Average &amp; Range method. <paramref name="measurements"/> is
    /// <c>[part][operator][trial]</c>; <paramref name="tolerance"/> (USL − LSL) adds
    /// <see cref="GageRRResult.PercentTolerance"/>.
    /// </summary>
    public GageRRResult GageRRXbarR(double[][][] measurements, double? tolerance = null)
        => Call(NativeInterop.uanalytics_gage_rr_xbar_r, AnalyticsJson.Default.GageRRResult,
            ("measurements", Request.Rows("measurements", measurements)),
            ("tolerance", Request.Num("tolerance", tolerance)));

    /// <summary>Gage R&amp;R by the ANOVA method; same input as <see cref="GageRRXbarR"/>.</summary>
    public GageRRAnovaResult GageRRAnova(double[][][] measurements, double? tolerance = null)
        => Call(NativeInterop.uanalytics_gage_rr_anova, AnalyticsJson.Default.GageRRAnovaResult,
            ("measurements", Request.Rows("measurements", measurements)),
            ("tolerance", Request.Num("tolerance", tolerance)));

    // ── Weibull ──

    /// <summary>
    /// Weibull maximum-likelihood fit of <paramref name="failureTimes"/> (each finite and &gt; 0,
    /// at least 2). Returns <c>shape</c>, <c>scale</c>, <c>log_likelihood</c>, <c>iterations</c>.
    /// </summary>
    public WeibullMleResult WeibullMle(double[] failureTimes)
        => Call(NativeInterop.uanalytics_weibull_mle, AnalyticsJson.Default.WeibullMleResult,
            ("failure_times", Request.Nums("failure_times", failureTimes)));

    /// <summary>
    /// Weibull median-rank-regression fit (Bernard's ranks). Returns <c>shape</c>,
    /// <c>scale</c>, <c>r_squared</c>.
    /// </summary>
    public WeibullMrrResult WeibullMrr(double[] failureTimes)
        => Call(NativeInterop.uanalytics_weibull_mrr, AnalyticsJson.Default.WeibullMrrResult,
            ("failure_times", Request.Nums("failure_times", failureTimes)));

    /// <summary>
    /// Reliability metrics of a Weibull: <c>mtbf</c>, and arrays aligned with the inputs —
    /// <c>reliability</c> and <c>hazard_rate</c> at each of <paramref name="times"/>,
    /// <c>b_life</c> at each of <paramref name="fractionsFailed"/> (0.1 is B10; the time to
    /// reliability p is the B-life at 1 − p).
    /// </summary>
    public WeibullReliabilityResult WeibullReliability(double shape, double scale, double[]? times = null,
        double[]? fractionsFailed = null)
        => Call(NativeInterop.uanalytics_weibull_reliability, AnalyticsJson.Default.WeibullReliabilityResult,
            ("shape", Request.Num("shape", shape)), ("scale", Request.Num("scale", scale)),
            ("times", Request.Nums("times", times ?? [])),
            ("fractions_failed", Request.Nums("fractions_failed", fractionsFailed ?? [])));

    // ── Non-normal capability and sigma level ──

    /// <summary>
    /// Process capability for non-normal data via a Box-Cox transformation of
    /// <paramref name="data"/> (each &gt; 0, at least 4). <paramref name="lambdaRange"/> bounds
    /// the λ search (default [-5, 5]). Returns <c>lambda</c>, <c>lambda_at_bound</c> and the
    /// indices on the transformed scale (<c>null</c> where the limits do not define them).
    /// </summary>
    public BoxCoxCapabilityResult BoxCoxCapability(double[] data, double? usl = null, double? lsl = null,
        (double Min, double Max)? lambdaRange = null)
        => Call(NativeInterop.uanalytics_boxcox_capability, AnalyticsJson.Default.BoxCoxCapabilityResult,
            ("data", Request.Nums("data", data)), ("usl", Request.Num("usl", usl)),
            ("lsl", Request.Num("lsl", lsl)),
            ("lambda_range", lambdaRange is { } r ? Request.Nums("lambda_range", [r.Min, r.Max]) : null));

    /// <summary>Defect rate in PPM at a sigma level, with the conventional 1.5σ shift (6σ → 3.4 PPM).</summary>
    public double SigmaToPpm(double sigma)
        => Call(NativeInterop.uanalytics_sigma_to_ppm, AnalyticsJson.Default.ValueBody,
            ("sigma", Request.Num("sigma", sigma))).Value;

    /// <summary>Sigma level at a defect rate in PPM strictly inside (0, 1 000 000) — the inverse of <see cref="SigmaToPpm"/>.</summary>
    public double PpmToSigma(double ppm)
        => Call(NativeInterop.uanalytics_ppm_to_sigma, AnalyticsJson.Default.ValueBody,
            ("ppm", Request.Num("ppm", ppm))).Value;

    // ── Hypothesis tests (same results as the WASM binding) ──

    /// <summary>One-sample t test of <paramref name="data"/> against <paramref name="mu0"/>: <c>statistic</c>, <c>df</c>, <c>p_value</c>.</summary>
    public TestResult OneSampleTTest(double[] data, double mu0)
        => Call(NativeInterop.uanalytics_one_sample_t_test, AnalyticsJson.Default.TestResult,
            ("data", Request.Nums("data", data)), ("mu0", Request.Num("mu0", mu0)));

    /// <summary>Welch two-sample t test: <c>statistic</c>, <c>df</c>, <c>p_value</c>.</summary>
    public TestResult TwoSampleTTest(double[] a, double[] b)
        => Call(NativeInterop.uanalytics_two_sample_t_test, AnalyticsJson.Default.TestResult,
            ("a", Request.Nums("a", a)), ("b", Request.Nums("b", b)));

    /// <summary>Paired t test (same length): <c>statistic</c>, <c>df</c>, <c>p_value</c>.</summary>
    public TestResult PairedTTest(double[] x, double[] y)
        => Call(NativeInterop.uanalytics_paired_t_test, AnalyticsJson.Default.TestResult,
            ("x", Request.Nums("x", x)), ("y", Request.Nums("y", y)));

    /// <summary>Mann-Whitney U test: <c>statistic</c>, <c>df</c>, <c>p_value</c>.</summary>
    public TestResult MannWhitneyUTest(double[] a, double[] b)
        => Call(NativeInterop.uanalytics_mann_whitney_u_test, AnalyticsJson.Default.TestResult,
            ("a", Request.Nums("a", a)), ("b", Request.Nums("b", b)));

    /// <summary>Wilcoxon signed-rank test (pairs with x = y dropped): <c>statistic</c>, <c>df</c>, <c>p_value</c>.</summary>
    public TestResult WilcoxonSignedRankTest(double[] x, double[] y)
        => Call(NativeInterop.uanalytics_wilcoxon_signed_rank_test, AnalyticsJson.Default.TestResult,
            ("x", Request.Nums("x", x)), ("y", Request.Nums("y", y)));

    /// <summary>Jarque-Bera normality test: <c>statistic</c>, <c>df</c>, <c>p_value</c>.</summary>
    public TestResult JarqueBeraTest(double[] data)
        => Call(NativeInterop.uanalytics_jarque_bera_test, AnalyticsJson.Default.TestResult,
            ("data", Request.Nums("data", data)));

    /// <summary>Shapiro-Wilk normality test (Royston 1995): <c>w</c>, <c>p_value</c>.</summary>
    public ShapiroWilkResult ShapiroWilkTest(double[] data)
        => Call(NativeInterop.uanalytics_shapiro_wilk_test, AnalyticsJson.Default.ShapiroWilkResult,
            ("data", Request.Nums("data", data)));

    /// <summary>
    /// Anderson-Darling normality test: <c>statistic</c> (A²), <c>statistic_modified</c>
    /// (A²* = A²·(1 + 0.75/n + 2.25/n²), which the p-value uses) and <c>p_value</c>.
    /// At least 8 values, not all equal.
    /// </summary>
    public AndersonDarlingResult AndersonDarlingTest(double[] data)
        => Call(NativeInterop.uanalytics_anderson_darling_test, AnalyticsJson.Default.AndersonDarlingResult,
            ("data", Request.Nums("data", data)));

    /// <summary>
    /// Augmented Dickey-Fuller unit-root test (H₀: the series has a unit root). At least
    /// 10 values. <paramref name="model"/> is <c>"constant"</c> (default), <c>"none"</c> or
    /// <c>"constant_trend"</c>; <paramref name="maxLags"/> fixes the number of lagged
    /// differences, <c>null</c> selects it by AIC. Returns <c>statistic</c>, <c>n_lags</c>,
    /// <c>n_obs</c> and <c>levels</c> — <c>level</c>, <c>critical_value</c>, <c>rejected</c>
    /// for 1 %, 5 % and 10 %.
    /// </summary>
    public AdfResult AdfTest(double[] data, AdfModel model = AdfModel.Constant, int? maxLags = null)
        => Call(NativeInterop.uanalytics_adf_test, AnalyticsJson.Default.AdfResult,
            ("data", Request.Nums("data", data)),
            ("model", Request.Option(model, AnalyticsJson.Default.AdfModel)),
            ("max_lags", maxLags is { } l ? JsonValue.Create(l) : null));

    /// <summary>Mann-Kendall trend test with Kendall's tau and Sen's slope.</summary>
    public MannKendallResult MannKendallTest(double[] data)
        => Call(NativeInterop.uanalytics_mann_kendall_test, AnalyticsJson.Default.MannKendallResult,
            ("data", Request.Nums("data", data)));

    /// <summary>One-way ANOVA across <paramref name="groups"/>.</summary>
    public AnovaResult OneWayAnova(double[][] groups)
        => Call(NativeInterop.uanalytics_one_way_anova, AnalyticsJson.Default.AnovaResult,
            ("groups", Request.Rows("groups", groups)));

    /// <summary>Kruskal-Wallis test across <paramref name="groups"/>.</summary>
    public TestResult KruskalWallisTest(double[][] groups)
        => Call(NativeInterop.uanalytics_kruskal_wallis_test, AnalyticsJson.Default.TestResult,
            ("groups", Request.Rows("groups", groups)));

    /// <summary>Levene test of equal variances across <paramref name="groups"/>.</summary>
    public TestResult LeveneTest(double[][] groups)
        => Call(NativeInterop.uanalytics_levene_test, AnalyticsJson.Default.TestResult,
            ("groups", Request.Rows("groups", groups)));

    /// <summary>Bartlett test of equal variances across <paramref name="groups"/>.</summary>
    public TestResult BartlettTest(double[][] groups)
        => Call(NativeInterop.uanalytics_bartlett_test, AnalyticsJson.Default.TestResult,
            ("groups", Request.Rows("groups", groups)));

    /// <summary>Chi-squared goodness of fit of <paramref name="observed"/> counts against <paramref name="expected"/>.</summary>
    public TestResult ChiSquaredGoodnessOfFit(double[] observed, double[] expected)
        => Call(NativeInterop.uanalytics_chi_squared_goodness_of_fit, AnalyticsJson.Default.TestResult,
            ("observed", Request.Nums("observed", observed)), ("expected", Request.Nums("expected", expected)));

    /// <summary>Chi-squared test of independence on a contingency <paramref name="table"/>.</summary>
    public TestResult ChiSquaredIndependence(double[][] table)
        => Call(NativeInterop.uanalytics_chi_squared_independence, AnalyticsJson.Default.TestResult,
            ("table", Request.Rows("table", table)));

    /// <summary>Fisher's exact test on a 2×2 <paramref name="table"/> of whole counts.</summary>
    public TestResult FisherExactTest(long[][] table)
        => Call(NativeInterop.uanalytics_fisher_exact_test, AnalyticsJson.Default.TestResult,
            ("table", Request.Counts(table)));

    /// <summary>Bonferroni-adjusted p-values, in input order.</summary>
    public IReadOnlyList<double> BonferroniCorrection(double[] pValues)
        => Call(NativeInterop.uanalytics_bonferroni_correction, AnalyticsJson.Default.ValuesBody,
            ("p_values", Request.Nums("p_values", pValues))).Values;

    /// <summary>Benjamini-Hochberg (false discovery rate) adjusted p-values, in input order.</summary>
    public IReadOnlyList<double> BenjaminiHochberg(double[] pValues)
        => Call(NativeInterop.uanalytics_benjamini_hochberg, AnalyticsJson.Default.ValuesBody,
            ("p_values", Request.Nums("p_values", pValues))).Values;

    // ── Event-time trend (one unit's events: failures of a repairable system, incidents, ...) ──

    /// <summary>
    /// Laplace trend test of <paramref name="times"/> (ascending, &gt; 0) against a constant
    /// event rate. <paramref name="end"/> is when observation ended (at or after the last
    /// event); <c>null</c> means it stopped at the last event, which is then not used.
    /// Returns <c>statistic</c>, two-sided <c>p_value</c>, <c>direction</c>
    /// (<c>increasing</c> / <c>decreasing</c> / <c>flat</c>), <c>events_used</c>, <c>df</c> (null).
    /// </summary>
    public TrendTestResult LaplaceTrendTest(double[] times, double? end)
        => Call(NativeInterop.uanalytics_laplace_trend_test, AnalyticsJson.Default.TrendTestResult,
            PointProcessRequest(times, end));

    /// <summary>
    /// MIL-HDBK-189 trend test (χ² = 2·Σ ln(T/tᵢ), df = 2m) — <paramref name="end"/> as for
    /// <see cref="LaplaceTrendTest"/>. A small statistic means the rate is increasing.
    /// </summary>
    public TrendTestResult MilHdbk189Test(double[] times, double? end)
        => Call(NativeInterop.uanalytics_mil_hdbk_189_test, AnalyticsJson.Default.TrendTestResult,
            PointProcessRequest(times, end));

    /// <summary>
    /// Power-law process (Crow-AMSAA) fit — <paramref name="end"/> as for
    /// <see cref="LaplaceTrendTest"/>. Returns <c>beta</c> (&lt; 1 rate decreasing, &gt; 1
    /// increasing), <c>beta_unbiased</c>, <c>lambda</c> (expected events by t = λ·t^β),
    /// <c>intensity_at_end</c>, <c>end</c>, <c>events</c>.
    /// </summary>
    public PowerLawFitResult PowerLawProcessFit(double[] times, double? end)
        => Call(NativeInterop.uanalytics_power_law_process_fit, AnalyticsJson.Default.PowerLawFitResult,
            PointProcessRequest(times, end));

    private static (string, JsonNode?)[] PointProcessRequest(double[] times, double? end)
        =>
        [
            ("times", Request.Nums("times", times)),
            ("observation", end is { } t
                ? Request.Body(("truncation", JsonValue.Create("time")), ("end", Request.Num("observation.end", t)))
                : Request.Body(("truncation", JsonValue.Create("failure")))),
        ];

    // ── Detection ──

    /// <summary>
    /// PELT changepoint detection. <paramref name="penalty"/> is a positive number, or
    /// <c>null</c> for BIC. <paramref name="cost"/> is <c>"l2"</c> (mean change, the default)
    /// or <c>"normal"</c> (mean and variance). Returns <c>changepoints</c> and
    /// <c>n_segments</c>. Same JSON as the WASM binding; the penalty was formerly a string.
    /// </summary>
    public ChangepointResult DetectChangepoints(double[] data, double? penalty = null, int? minSegmentLen = null,
        PeltCost cost = PeltCost.L2)
        => Call(NativeInterop.uanalytics_detect_changepoints, AnalyticsJson.Default.ChangepointResult,
            ("data", Request.Nums("data", data)),
            ("cost", Request.Option(cost, AnalyticsJson.Default.PeltCost)),
            ("penalty", Request.Num("penalty", penalty)),
            ("min_segment_len", minSegmentLen is { } m ? JsonValue.Create(m) : null));

    /// <summary>
    /// PELT over several aligned <paramref name="signals"/> (each the same length): one set
    /// of <c>changepoints</c> for all of them, and <c>n_segments</c>. Options as for
    /// <see cref="DetectChangepoints"/>. A signal whose length differs from the first is
    /// refused as <c>dimension_mismatch</c> at its <see cref="AnalyticsException.Index"/>.
    /// </summary>
    public ChangepointResult DetectChangepointsMulti(double[][] signals, double? penalty = null,
        int? minSegmentLen = null, PeltCost cost = PeltCost.L2)
        => Call(NativeInterop.uanalytics_detect_changepoints_multi, AnalyticsJson.Default.ChangepointResult,
            ("signals", Request.Rows("signals", signals)),
            ("cost", Request.Option(cost, AnalyticsJson.Default.PeltCost)),
            ("penalty", Request.Num("penalty", penalty)),
            ("min_segment_len", minSegmentLen is { } m ? JsonValue.Create(m) : null));

    /// <summary>
    /// CUSUM chart (Page 1954) for small sustained shifts of the mean away from
    /// <paramref name="target"/>, with known process <paramref name="sigma"/> (&gt; 0).
    /// <paramref name="k"/> is the allowance (≥ 0, default 0.5) and <paramref name="h"/> the
    /// decision interval (&gt; 0, default 5), both in sigmas. Returns <c>h</c>, per-point
    /// <c>points</c> (<c>index</c>, <c>s_upper</c>, <c>s_lower</c>, <c>signal</c>) on the
    /// standardized scale, <c>signal_indices</c> and <c>in_control</c>.
    /// </summary>
    public CusumResult Cusum(double[] data, double target, double sigma, double? k = null, double? h = null)
        => Call(NativeInterop.uanalytics_cusum, AnalyticsJson.Default.CusumResult,
            ("data", Request.Nums("data", data)), ("target", Request.Num("target", target)),
            ("sigma", Request.Num("sigma", sigma)), ("k", Request.Num("k", k)), ("h", Request.Num("h", h)));

    /// <summary>
    /// EWMA chart (Roberts 1959) about <paramref name="target"/> with known process
    /// <paramref name="sigma"/> (&gt; 0). <paramref name="lambda"/> is the smoothing constant
    /// in (0, 1] (default 0.2) and <paramref name="lFactor"/> the limit width in sigmas
    /// (&gt; 0, default 3). Returns per-point <c>points</c> (<c>index</c>, <c>ewma</c>,
    /// <c>ucl</c>, <c>lcl</c>, <c>signal</c>) -- the limits are exact, so they widen with the
    /// index -- plus <c>signal_indices</c> and <c>in_control</c>.
    /// </summary>
    public EwmaResult Ewma(double[] data, double target, double sigma, double? lambda = null,
        double? lFactor = null)
        => Call(NativeInterop.uanalytics_ewma, AnalyticsJson.Default.EwmaResult,
            ("data", Request.Nums("data", data)), ("target", Request.Num("target", target)),
            ("sigma", Request.Num("sigma", sigma)), ("lambda", Request.Num("lambda", lambda)),
            ("l_factor", Request.Num("l_factor", lFactor)));

    // ── Seasonality ──

    /// <summary>
    /// Estimates the dominant period of a univariate series (AutoPeriod:
    /// permutation-thresholded periodogram peaks refined on the ACF). The
    /// response's <c>period</c> is <c>null</c>, not an error, when no
    /// periodicity is found; <c>candidates</c> lists every validated period.
    /// </summary>
    public PeriodResult EstimatePeriod(double[] data)
        => Call(NativeInterop.uanalytics_estimate_period, AnalyticsJson.Default.PeriodResult,
            ("data", Request.Nums("data", data)));

    /// <summary>
    /// Scores every point for anomalies by spectral residual saliency (Ren et
    /// al. 2019). Optional arguments left <c>null</c> take the paper's
    /// defaults (q = 3, z = 40, τ = 3, z-score gate 1.5, 70% band, no
    /// batching). The response's <c>anomalies</c> lists the flagged indices.
    /// </summary>
    public SpectralResidualResult SpectralResidual(double[] data, int? averagingWindow = null, int? judgementWindow = null,
        double? threshold = null, double? minZscore = null, double? sensitivity = null, int? batchSize = null)
        => Call(NativeInterop.uanalytics_spectral_residual, AnalyticsJson.Default.SpectralResidualResult,
            ("data", Request.Nums("data", data)),
            ("averaging_window", averagingWindow is { } a ? JsonValue.Create(a) : null),
            ("judgement_window", judgementWindow is { } j ? JsonValue.Create(j) : null),
            ("threshold", Request.Num("threshold", threshold)),
            ("min_zscore", Request.Num("min_zscore", minZscore)),
            ("sensitivity", Request.Num("sensitivity", sensitivity)),
            ("batch_size", batchSize is { } b ? JsonValue.Create(b) : null));

    // ── Correlation ──

    /// <summary>
    /// Correlation matrix of <paramref name="variables"/> (at least 2, each with the same
    /// number of values, at least 3). <paramref name="method"/> is <c>"pearson"</c> (default),
    /// <c>"spearman"</c> or <c>"kendall"</c> (tau-b). Returns <c>matrix</c>, an array of rows:
    /// <c>matrix[i][j]</c> is the correlation of variables i and j. A constant variable is
    /// refused at its <see cref="AnalyticsException.Index"/>.
    /// </summary>
    public CorrelationMatrixResult CorrelationMatrix(double[][] variables, CorrelationMethod method = CorrelationMethod.Pearson)
        => Call(NativeInterop.uanalytics_correlation_matrix, AnalyticsJson.Default.CorrelationMatrixResult,
            ("variables", Request.Rows("variables", variables)),
            ("method", Request.Option(method, AnalyticsJson.Default.CorrelationMethod)));

    // ── Regression ──

    /// <summary>
    /// Simple linear regression of <paramref name="y"/> on <paramref name="x"/> (same length,
    /// at least 3, <paramref name="x"/> not constant). Returns <c>slope</c>, <c>intercept</c>,
    /// <c>r_squared</c>, <c>adjusted_r_squared</c>, their standard errors, <c>slope_t</c> /
    /// <c>intercept_t</c> and p values, <c>residual_se</c>, <c>f_statistic</c> /
    /// <c>f_p_value</c>, <c>residuals</c> and <c>fitted</c>. A t or F with no finite value
    /// (an exact fit) is <c>null</c>.
    /// </summary>
    public RegressionResult SimpleRegression(double[] x, double[] y)
        => Call(NativeInterop.uanalytics_simple_regression, AnalyticsJson.Default.RegressionResult,
            ("x", Request.Nums("x", x)), ("y", Request.Nums("y", y)));

    // ── Distribution ──

    /// <summary>
    /// Every continuous family that fits <paramref name="data"/> (at least 2 values, not all
    /// equal), best AIC first: each with <c>distribution</c>, <c>parameters</c> (an object of
    /// name → value), <c>log_likelihood</c>, <c>aic</c>, <c>bic</c>. Normal always;
    /// Exponential, Gamma, LogNormal and Weibull for positive data; Beta for data in (0, 1).
    /// </summary>
    public IReadOnlyList<DistributionFit> FitBest(double[] data)
        => Call(NativeInterop.uanalytics_fit_best, AnalyticsJson.Default.IReadOnlyListDistributionFit,
            ("data", Request.Nums("data", data)));

    // ── Internal ──

    private delegate int NativeFunc(string json, out IntPtr result);

    private T Call<T>(NativeFunc func, JsonTypeInfo<T> result, params (string Key, JsonNode? Value)[] members)
    {
        var body = Invoke(func, Request.Body(members).ToJsonString());
        ResponseObserver?.Invoke(body);
        return JsonSerializer.Deserialize(body, result)
               ?? throw new AnalyticsException(-4, "The engine returned null.");
    }

    private static string Invoke(NativeFunc func, string requestJson)
    {
        var code = func(requestJson, out var resultPtr);

        try
        {
            if (resultPtr == IntPtr.Zero)
                throw new AnalyticsException(code, "Null result from engine");

            var resultJson = Marshal.PtrToStringUTF8(resultPtr);
            if (string.IsNullOrEmpty(resultJson))
                throw new AnalyticsException(code, "Empty result from engine");

            if (code != 0)
                throw AnalyticsException.FromErrorBody(code, resultJson);

            return resultJson;
        }
        finally
        {
            if (resultPtr != IntPtr.Zero)
                NativeInterop.uanalytics_free_string(resultPtr);
        }
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            GC.SuppressFinalize(this);
        }
    }
}

/// <summary>
/// A call the engine refused. <see cref="Exception.Message"/> is human-readable;
/// <see cref="Reason"/>, <see cref="Parameter"/>, <see cref="Index"/> and <see cref="Details"/>
/// are for programs.
/// </summary>
public class AnalyticsException : Exception
{
    /// <summary>Native status: -1 null pointer, -2 malformed request, -3 refused input, -4 internal panic.</summary>
    public int Code { get; }

    /// <summary>
    /// Stable, machine-readable reason, e.g. <c>count_not_whole</c>,
    /// <c>sample_size_not_whole</c>, <c>defectives_exceed_sample</c>,
    /// <c>units_not_positive</c>, <c>insufficient_data</c>, <c>empty_input</c>,
    /// <c>parameter_out_of_range</c>, <c>unknown_option</c>, <c>value_not_finite</c>,
    /// <c>malformed_input</c>, <c>invalid_input</c>. <c>null</c> when the engine returned no body.
    /// </summary>
    public string? Reason { get; }

    /// <summary>Zero-based position of the offending element in its input array, when there is one.</summary>
    public int? Index { get; }

    /// <summary>The argument or option the refusal is about (<c>penalty</c>, <c>times</c>, …), when there is one.</summary>
    public string? Parameter { get; }

    /// <summary>
    /// The whole error body: <c>error</c>, <c>code</c>, <c>index</c>, <c>parameter</c> and the
    /// values behind the reason — <c>min</c> / <c>max</c> / <c>got</c> for a value out of range,
    /// <c>got</c> / <c>expected</c> for an unknown option name, <c>min</c> / <c>got</c> for too
    /// few values. <c>null</c> when there is no body.
    /// </summary>
    public JsonElement? Details { get; }

    public AnalyticsException(int code, string message) : base(message)
    {
        Code = code;
    }

    public AnalyticsException(int code, string message, string? reason, int? index) : base(message)
    {
        Code = code;
        Reason = reason;
        Index = index;
    }

    public AnalyticsException(int code, string message, string? reason, int? index, string? parameter,
        JsonElement? details) : this(code, message, reason, index)
    {
        Parameter = parameter;
        Details = details;
    }

    /// <summary>Reads the engine's <c>{"error", "code", "index", "parameter", ...}</c> error body.</summary>
    internal static AnalyticsException FromErrorBody(int code, string body)
    {
        try
        {
            using var doc = JsonDocument.Parse(body);
            var root = doc.RootElement;
            var message = root.TryGetProperty("error", out var e) && e.ValueKind == JsonValueKind.String
                ? e.GetString()!
                : body;
            string? reason = root.TryGetProperty("code", out var c) && c.ValueKind == JsonValueKind.String
                ? c.GetString()
                : null;
            int? index = root.TryGetProperty("index", out var i) && i.ValueKind == JsonValueKind.Number
                ? i.GetInt32()
                : null;
            string? parameter = root.TryGetProperty("parameter", out var p) && p.ValueKind == JsonValueKind.String
                ? p.GetString()
                : null;
            return new AnalyticsException(code, message, reason, index, parameter, root.Clone());
        }
        catch (JsonException)
        {
            return new AnalyticsException(code, body);
        }
    }
}
