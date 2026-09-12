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
    /// Returns <c>p_bar</c>, per-point <c>points</c> (<c>index</c>, <c>value</c>,
    /// <c>ucl</c>, <c>cl</c>, <c>lcl</c>, <c>out_of_control</c>) and <c>in_control</c>.
    /// A sample with more defectives than its size, or a size of zero, is rejected
    /// by its row rather than dropped.
    /// </summary>
    public JsonElement PChart(ulong[][] samples)
        => CallNative(NativeInterop.uanalytics_p_chart, new { samples });

    /// <summary>
    /// Laney P' chart for over- or under-dispersed proportions. Same
    /// <c>[defectives, sampleSize]</c> samples as <see cref="PChart"/>; at least three.
    /// Returns <c>p_bar</c>, <c>phi</c> and per-point <c>points</c>.
    /// </summary>
    public JsonElement LaneyPChart(ulong[][] samples)
        => CallNative(NativeInterop.uanalytics_laney_p_chart, new { samples });

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

    public JsonElement WeibullMle(double[] failureTimes)
        => CallNative(NativeInterop.uanalytics_weibull_mle,
            new { failure_times = failureTimes });

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
                throw new AnalyticsException(code, resultJson);

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

public class AnalyticsException : Exception
{
    public int Code { get; }

    public AnalyticsException(int code, string message) : base(message)
    {
        Code = code;
    }
}
