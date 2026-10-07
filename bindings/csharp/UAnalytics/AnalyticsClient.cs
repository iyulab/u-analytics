using System.Runtime.InteropServices;
using System.Text.Json;
using System.Text.Json.Serialization;
using UAnalytics.Interop;

namespace UAnalytics;

public sealed class AnalyticsClient : IDisposable
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
    };

    private bool _disposed;

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
    public JsonElement XbarRChart(double[][] subgroups, string[]? rules = null)
        => CallNative(NativeInterop.uanalytics_xbar_r_chart,
            rules is null ? new { subgroups } : (object)new { subgroups, rules });

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
    public JsonElement PChart(ulong[][] samples, double? pBar = null)
        => CallNative(NativeInterop.uanalytics_p_chart, new { samples, pBar });

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
    public JsonElement LaneyPChart(ulong[][] samples, double? pBar = null, double? phi = null)
        => CallNative(NativeInterop.uanalytics_laney_p_chart, new { samples, pBar, phi });

    /// <summary>
    /// NP chart: defectives per subgroup when every subgroup has the same
    /// <paramref name="sampleSize"/>. Returns <c>cl</c>, <c>ucl</c>, <c>lcl</c>,
    /// per-point <c>points</c> and <c>in_control</c>.
    /// </summary>
    public JsonElement NpChart(ulong[] defectives, ulong sampleSize)
        => CallNative(NativeInterop.uanalytics_np_chart, new { defectives, sampleSize });

    /// <summary>
    /// C chart: defects per inspection unit of one size. Returns <c>cl</c>, <c>ucl</c>,
    /// <c>lcl</c>, per-point <c>points</c> and <c>in_control</c>.
    /// </summary>
    public JsonElement CChart(ulong[] defects)
        => CallNative(NativeInterop.uanalytics_c_chart, new { defects });

    /// <summary>
    /// U chart: defects per unit when the quantity inspected varies. Each sample is
    /// <c>[defects, units]</c> (<c>units</c> may be fractional, and must be positive).
    /// Returns <c>u_bar</c>, per-point <c>points</c> (with <c>z</c>) and <c>in_control</c>.
    /// </summary>
    /// <param name="samples"><c>[defects, units]</c> pairs.</param>
    /// <param name="uBar">A known centre line from a Phase I study (Phase II); positive.</param>
    public JsonElement UChart(double[][] samples, double? uBar = null)
        => CallNative(NativeInterop.uanalytics_u_chart, new { samples, uBar });

    /// <summary>
    /// Laney U' chart for over- or under-dispersed rates. Same samples as
    /// <see cref="UChart"/>; at least three when u-bar and phi are estimated.
    /// Returns <c>u_bar</c>, <c>phi</c> and per-point <c>points</c> (with <c>z</c>).
    /// </summary>
    /// <param name="samples"><c>[defects, units]</c> pairs.</param>
    /// <param name="uBar">Phase I u-bar; given together with <paramref name="phi"/> or not at all.</param>
    /// <param name="phi">Phase I sigma-inflation factor (&#8805; 0); given together with <paramref name="uBar"/>.</param>
    public JsonElement LaneyUChart(double[][] samples, double? uBar = null, double? phi = null)
        => CallNative(NativeInterop.uanalytics_laney_u_chart, new { samples, uBar, phi });

    /// <summary>
    /// G chart for rare events: conforming counts between events (at least three,
    /// each &#8805; 0). Returns <c>g_bar</c> and per-point <c>points</c>.
    /// </summary>
    public JsonElement GChart(double[] gaps)
        => CallNative(NativeInterop.uanalytics_g_chart, new { gaps });

    /// <summary>
    /// T chart for rare events: times between events (at least three, each &gt; 0).
    /// Returns <c>t_bar</c> and per-point <c>points</c>.
    /// </summary>
    public JsonElement TChart(double[] times)
        => CallNative(NativeInterop.uanalytics_t_chart, new { times });

    /// <summary>
    /// X-bar/S chart. Same request and response shape as <see cref="XbarRChart"/> with
    /// <c>s_*</c> limits in place of <c>r_*</c>; the usual choice once subgroups exceed about
    /// ten values. Returns <c>sigma_hat</c> (<c>S-bar / c4</c>).
    /// </summary>
    public JsonElement XbarSChart(double[][] subgroups, string[]? rules = null)
        => CallNative(NativeInterop.uanalytics_xbar_s_chart,
            rules is null ? new { subgroups } : (object)new { subgroups, rules });

    /// <summary>
    /// Individual / Moving-Range chart for a series of single observations. Returns
    /// <c>sigma_hat</c> (<c>MR-bar / d2(2)</c>) -- pass it as <c>sigmaWithin</c> to
    /// <see cref="ProcessCapability"/> for individual data, which no longer estimates one itself.
    /// </summary>
    public JsonElement ImrChart(double[] values, string[]? rules = null)
        => CallNative(NativeInterop.uanalytics_imr_chart,
            rules is null ? new { values } : (object)new { values, rules });

    /// <summary>
    /// Applies the run tests to a series against one set of limits -- the engine the charts
    /// use, callable on its own. One point per value, in order, each with its
    /// <c>violations</c>.
    /// </summary>
    public JsonElement RunRules(double[] values, double ucl, double cl, double lcl,
        string[]? rules = null)
        => CallNative(NativeInterop.uanalytics_run_rules,
            rules is null
                ? new { values, limits = new { ucl, cl, lcl } }
                : (object)new { values, limits = new { ucl, cl, lcl }, rules });

    // ── Capability ──

    /// <summary>
    /// Capability indices. <paramref name="sigmaWithin"/> is the short-term sigma from a
    /// control chart (the <c>sigma_hat</c> that <see cref="XbarRChart"/> returns). Without it
    /// the short-term indices (<c>cp</c>, <c>cpk</c>, <c>cpu</c>, <c>cpl</c>) are <c>null</c> and
    /// <c>sigma_source</c> is <c>"overall"</c> -- a flat vector carries no subgroup structure to
    /// estimate one from, and this client no longer guesses one from the moving range.
    /// Same JSON as the WASM binding.
    /// </summary>
    public JsonElement ProcessCapability(double[] data, double? usl, double? lsl, double? target = null,
        double? sigmaWithin = null)
        => CallNative(NativeInterop.uanalytics_process_capability,
            new { data, usl, lsl, target, sigma_within = sigmaWithin });

    public JsonElement PercentileCapability(double[] data, double? usl, double? lsl)
        => CallNative(NativeInterop.uanalytics_percentile_capability,
            new { data, usl, lsl });

    // ── MSA ──

    public JsonElement GageRRXbarR(double[][][] measurements, double? tolerance = null)
        => CallNative(NativeInterop.uanalytics_gage_rr_xbar_r,
            new { measurements, tolerance });

    public JsonElement GageRRAnova(double[][][] measurements, double? tolerance = null)
        => CallNative(NativeInterop.uanalytics_gage_rr_anova,
            new { measurements, tolerance });

    // ── Weibull ──

    /// <summary>
    /// Weibull maximum-likelihood fit of <paramref name="failureTimes"/> (each finite and &gt; 0,
    /// at least 2). Returns <c>shape</c>, <c>scale</c>, <c>log_likelihood</c>, <c>iterations</c>.
    /// </summary>
    public JsonElement WeibullMle(double[] failureTimes)
        => CallNative(NativeInterop.uanalytics_weibull_mle,
            new { failure_times = failureTimes });

    /// <summary>
    /// Weibull median-rank-regression fit (Bernard's ranks). Returns <c>shape</c>,
    /// <c>scale</c>, <c>r_squared</c>.
    /// </summary>
    public JsonElement WeibullMrr(double[] failureTimes)
        => CallNative(NativeInterop.uanalytics_weibull_mrr,
            new { failure_times = failureTimes });

    /// <summary>
    /// Reliability metrics of a Weibull: <c>mtbf</c>, and arrays aligned with the inputs —
    /// <c>reliability</c> and <c>hazard_rate</c> at each of <paramref name="times"/>,
    /// <c>b_life</c> at each of <paramref name="fractionsFailed"/> (0.1 is B10; the time to
    /// reliability p is the B-life at 1 − p).
    /// </summary>
    public JsonElement WeibullReliability(double shape, double scale, double[]? times = null,
        double[]? fractionsFailed = null)
        => CallNative(NativeInterop.uanalytics_weibull_reliability,
            new { shape, scale, times = times ?? [], fractions_failed = fractionsFailed ?? [] });

    // ── Non-normal capability and sigma level ──

    /// <summary>
    /// Process capability for non-normal data via a Box-Cox transformation of
    /// <paramref name="data"/> (each &gt; 0, at least 4). <paramref name="lambdaRange"/> bounds
    /// the λ search (default [-5, 5]). Returns <c>lambda</c>, <c>lambda_at_bound</c> and the
    /// indices on the transformed scale (<c>null</c> where the limits do not define them).
    /// </summary>
    public JsonElement BoxCoxCapability(double[] data, double? usl = null, double? lsl = null,
        (double Min, double Max)? lambdaRange = null)
        => CallNative(NativeInterop.uanalytics_boxcox_capability,
            new
            {
                data,
                usl,
                lsl,
                lambda_range = lambdaRange is { } r ? new[] { r.Min, r.Max } : null,
            });

    /// <summary>Defect rate in PPM at a sigma level, with the conventional 1.5σ shift (6σ → 3.4 PPM).</summary>
    public double SigmaToPpm(double sigma)
        => CallNative(NativeInterop.uanalytics_sigma_to_ppm, new { sigma }).GetProperty("value").GetDouble();

    /// <summary>Sigma level at a defect rate in PPM strictly inside (0, 1 000 000) — the inverse of <see cref="SigmaToPpm"/>.</summary>
    public double PpmToSigma(double ppm)
        => CallNative(NativeInterop.uanalytics_ppm_to_sigma, new { ppm }).GetProperty("value").GetDouble();

    // ── Hypothesis tests (same results as the WASM binding) ──

    /// <summary>One-sample t test of <paramref name="data"/> against <paramref name="mu0"/>: <c>statistic</c>, <c>df</c>, <c>p_value</c>.</summary>
    public JsonElement OneSampleTTest(double[] data, double mu0)
        => CallNative(NativeInterop.uanalytics_one_sample_t_test, new { data, mu0 });

    /// <summary>Welch two-sample t test: <c>statistic</c>, <c>df</c>, <c>p_value</c>.</summary>
    public JsonElement TwoSampleTTest(double[] a, double[] b)
        => CallNative(NativeInterop.uanalytics_two_sample_t_test, new { a, b });

    /// <summary>Paired t test (same length): <c>statistic</c>, <c>df</c>, <c>p_value</c>.</summary>
    public JsonElement PairedTTest(double[] x, double[] y)
        => CallNative(NativeInterop.uanalytics_paired_t_test, new { x, y });

    /// <summary>Mann-Whitney U test: <c>statistic</c>, <c>df</c>, <c>p_value</c>.</summary>
    public JsonElement MannWhitneyUTest(double[] a, double[] b)
        => CallNative(NativeInterop.uanalytics_mann_whitney_u_test, new { a, b });

    /// <summary>Wilcoxon signed-rank test (pairs with x = y dropped): <c>statistic</c>, <c>df</c>, <c>p_value</c>.</summary>
    public JsonElement WilcoxonSignedRankTest(double[] x, double[] y)
        => CallNative(NativeInterop.uanalytics_wilcoxon_signed_rank_test, new { x, y });

    /// <summary>Jarque-Bera normality test: <c>statistic</c>, <c>df</c>, <c>p_value</c>.</summary>
    public JsonElement JarqueBeraTest(double[] data)
        => CallNative(NativeInterop.uanalytics_jarque_bera_test, new { data });

    /// <summary>Shapiro-Wilk normality test (Royston 1995): <c>w</c>, <c>p_value</c>.</summary>
    public JsonElement ShapiroWilkTest(double[] data)
        => CallNative(NativeInterop.uanalytics_shapiro_wilk_test, new { data });

    /// <summary>
    /// Anderson-Darling normality test (Stephens 1974): <c>statistic</c> (A²),
    /// <c>statistic_modified</c> (A²*, the small-sample correction the p-value uses), <c>p_value</c>.
    /// At least 3 values, not all equal.
    /// </summary>
    public JsonElement AndersonDarlingNormality(double[] data)
        => CallNative(NativeInterop.uanalytics_anderson_darling_normality, new { data });

    /// <summary>Mann-Kendall trend test with Kendall's tau and Sen's slope.</summary>
    public JsonElement MannKendallTest(double[] data)
        => CallNative(NativeInterop.uanalytics_mann_kendall_test, new { data });

    /// <summary>One-way ANOVA across <paramref name="groups"/>.</summary>
    public JsonElement OneWayAnova(double[][] groups)
        => CallNative(NativeInterop.uanalytics_one_way_anova, new { groups });

    /// <summary>Kruskal-Wallis test across <paramref name="groups"/>.</summary>
    public JsonElement KruskalWallisTest(double[][] groups)
        => CallNative(NativeInterop.uanalytics_kruskal_wallis_test, new { groups });

    /// <summary>Levene test of equal variances across <paramref name="groups"/>.</summary>
    public JsonElement LeveneTest(double[][] groups)
        => CallNative(NativeInterop.uanalytics_levene_test, new { groups });

    /// <summary>Bartlett test of equal variances across <paramref name="groups"/>.</summary>
    public JsonElement BartlettTest(double[][] groups)
        => CallNative(NativeInterop.uanalytics_bartlett_test, new { groups });

    /// <summary>Chi-squared goodness of fit of <paramref name="observed"/> counts against <paramref name="expected"/>.</summary>
    public JsonElement ChiSquaredGoodnessOfFit(double[] observed, double[] expected)
        => CallNative(NativeInterop.uanalytics_chi_squared_goodness_of_fit, new { observed, expected });

    /// <summary>Chi-squared test of independence on a contingency <paramref name="table"/>.</summary>
    public JsonElement ChiSquaredIndependence(double[][] table)
        => CallNative(NativeInterop.uanalytics_chi_squared_independence, new { table });

    /// <summary>Fisher's exact test on a 2×2 <paramref name="table"/> of whole counts.</summary>
    public JsonElement FisherExactTest(long[][] table)
        => CallNative(NativeInterop.uanalytics_fisher_exact_test, new { table });

    /// <summary>Bonferroni-adjusted p-values, in input order.</summary>
    public double[] BonferroniCorrection(double[] pValues)
        => Values(CallNative(NativeInterop.uanalytics_bonferroni_correction, new { p_values = pValues }));

    /// <summary>Benjamini-Hochberg (false discovery rate) adjusted p-values, in input order.</summary>
    public double[] BenjaminiHochberg(double[] pValues)
        => Values(CallNative(NativeInterop.uanalytics_benjamini_hochberg, new { p_values = pValues }));

    private static double[] Values(JsonElement body)
        => body.GetProperty("values").EnumerateArray().Select(v => v.GetDouble()).ToArray();

    // ── Event-time trend (one unit's events: failures of a repairable system, incidents, ...) ──

    /// <summary>
    /// Laplace trend test of <paramref name="times"/> (ascending, &gt; 0) against a constant
    /// event rate. <paramref name="end"/> is when observation ended (at or after the last
    /// event); <c>null</c> means it stopped at the last event, which is then not used.
    /// Returns <c>statistic</c>, two-sided <c>p_value</c>, <c>direction</c>
    /// (<c>increasing</c> / <c>decreasing</c> / <c>flat</c>), <c>events_used</c>, <c>df</c> (null).
    /// </summary>
    public JsonElement LaplaceTrendTest(double[] times, double? end)
        => CallNative(NativeInterop.uanalytics_laplace_trend_test, PointProcessRequest(times, end));

    /// <summary>
    /// MIL-HDBK-189 trend test (χ² = 2·Σ ln(T/tᵢ), df = 2m) — <paramref name="end"/> as for
    /// <see cref="LaplaceTrendTest"/>. A small statistic means the rate is increasing.
    /// </summary>
    public JsonElement MilHdbk189Test(double[] times, double? end)
        => CallNative(NativeInterop.uanalytics_mil_hdbk_189_test, PointProcessRequest(times, end));

    /// <summary>
    /// Power-law process (Crow-AMSAA) fit — <paramref name="end"/> as for
    /// <see cref="LaplaceTrendTest"/>. Returns <c>beta</c> (&lt; 1 rate decreasing, &gt; 1
    /// increasing), <c>beta_unbiased</c>, <c>lambda</c> (expected events by t = λ·t^β),
    /// <c>intensity_at_end</c>, <c>end</c>, <c>events</c>.
    /// </summary>
    public JsonElement PowerLawProcessFit(double[] times, double? end)
        => CallNative(NativeInterop.uanalytics_power_law_process_fit, PointProcessRequest(times, end));

    private static object PointProcessRequest(double[] times, double? end)
        => end is { } t
            ? new { times, observation = (object)new { truncation = "time", end = t } }
            : new { times, observation = (object)new { truncation = "failure" } };

    // ── Detection ──

    /// <summary>
    /// PELT changepoint detection. <paramref name="penalty"/> is a positive number, or
    /// <c>null</c> for BIC. <paramref name="cost"/> is <c>"l2"</c> (mean change, the default)
    /// or <c>"normal"</c> (mean and variance). Returns <c>changepoints</c> and
    /// <c>n_segments</c>. Same JSON as the WASM binding; the penalty was formerly a string.
    /// </summary>
    public JsonElement DetectChangepoints(double[] data, double? penalty = null, int? minSegmentLen = null,
        string cost = "l2")
        => CallNative(NativeInterop.uanalytics_detect_changepoints,
            penalty is null
                ? new { data, cost, min_segment_len = minSegmentLen }
                : (object)new { data, cost, penalty, min_segment_len = minSegmentLen });

    /// <summary>
    /// PELT over several aligned <paramref name="signals"/> (each the same length): one set
    /// of <c>changepoints</c> for all of them, and <c>n_segments</c>. Options as for
    /// <see cref="DetectChangepoints"/>. A signal whose length differs from the first is
    /// refused as <c>dimension_mismatch</c> at its <see cref="AnalyticsException.Index"/>.
    /// </summary>
    public JsonElement DetectChangepointsMulti(double[][] signals, double? penalty = null,
        int? minSegmentLen = null, string cost = "l2")
        => CallNative(NativeInterop.uanalytics_detect_changepoints_multi,
            penalty is null
                ? new { signals, cost, min_segment_len = minSegmentLen }
                : (object)new { signals, cost, penalty, min_segment_len = minSegmentLen });

    /// <summary>
    /// CUSUM chart (Page 1954) for small sustained shifts of the mean away from
    /// <paramref name="target"/>, with known process <paramref name="sigma"/> (&gt; 0).
    /// <paramref name="k"/> is the allowance (≥ 0, default 0.5) and <paramref name="h"/> the
    /// decision interval (&gt; 0, default 5), both in sigmas. Returns <c>h</c>, per-point
    /// <c>points</c> (<c>index</c>, <c>s_upper</c>, <c>s_lower</c>, <c>signal</c>) on the
    /// standardized scale, <c>signal_indices</c> and <c>in_control</c>.
    /// </summary>
    public JsonElement Cusum(double[] data, double target, double sigma, double? k = null, double? h = null)
        => CallNative(NativeInterop.uanalytics_cusum, new { data, target, sigma, k, h });

    /// <summary>
    /// EWMA chart (Roberts 1959) about <paramref name="target"/> with known process
    /// <paramref name="sigma"/> (&gt; 0). <paramref name="lambda"/> is the smoothing constant
    /// in (0, 1] (default 0.2) and <paramref name="lFactor"/> the limit width in sigmas
    /// (&gt; 0, default 3). Returns per-point <c>points</c> (<c>index</c>, <c>ewma</c>,
    /// <c>ucl</c>, <c>lcl</c>, <c>signal</c>) -- the limits are exact, so they widen with the
    /// index -- plus <c>signal_indices</c> and <c>in_control</c>.
    /// </summary>
    public JsonElement Ewma(double[] data, double target, double sigma, double? lambda = null,
        double? lFactor = null)
        => CallNative(NativeInterop.uanalytics_ewma, new { data, target, sigma, lambda, l_factor = lFactor });

    // ── Seasonality ──

    /// <summary>
    /// Estimates the dominant period of a univariate series (AutoPeriod:
    /// permutation-thresholded periodogram peaks refined on the ACF). The
    /// response's <c>period</c> is <c>null</c>, not an error, when no
    /// periodicity is found; <c>candidates</c> lists every validated period.
    /// </summary>
    public JsonElement EstimatePeriod(double[] data)
        => CallNative(NativeInterop.uanalytics_estimate_period, new { data });

    /// <summary>
    /// Scores every point for anomalies by spectral residual saliency (Ren et
    /// al. 2019). Optional arguments left <c>null</c> take the paper's
    /// defaults (q = 3, z = 40, τ = 3, z-score gate 1.5, 70% band, no
    /// batching). The response's <c>anomalies</c> lists the flagged indices.
    /// </summary>
    public JsonElement SpectralResidual(double[] data, int? averagingWindow = null, int? judgementWindow = null,
        double? threshold = null, double? minZscore = null, double? sensitivity = null, int? batchSize = null)
        => CallNative(NativeInterop.uanalytics_spectral_residual,
            new
            {
                data,
                averaging_window = averagingWindow,
                judgement_window = judgementWindow,
                threshold,
                min_zscore = minZscore,
                sensitivity,
                batch_size = batchSize
            });

    // ── Correlation ──

    public JsonElement CorrelationMatrix(double[][] variables)
        => CallNative(NativeInterop.uanalytics_correlation_matrix,
            new { variables });

    // ── Regression ──

    public JsonElement SimpleRegression(double[] x, double[] y)
        => CallNative(NativeInterop.uanalytics_simple_regression,
            new { x, y });

    // ── Distribution ──

    public JsonElement FitBest(double[] data)
        => CallNative(NativeInterop.uanalytics_fit_best,
            new { data });

    // ── Internal ──

    private delegate int NativeFunc(string json, out IntPtr result);

    private JsonElement CallNative(NativeFunc func, object request)
    {
        NonFinite.Check(request, JsonOptions.PropertyNamingPolicy!);
        var requestJson = JsonSerializer.Serialize(request, JsonOptions);
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

            return JsonDocument.Parse(resultJson).RootElement.Clone();
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
