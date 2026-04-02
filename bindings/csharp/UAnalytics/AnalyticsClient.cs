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

    public JsonElement XbarRChart(double[][] subgroups)
        => CallNative(NativeInterop.uanalytics_xbar_r_chart, new { subgroups });

    public JsonElement PChart(ulong[][] samples)
        => CallNative(NativeInterop.uanalytics_p_chart, new { samples });

    // ── Capability ──

    public JsonElement ProcessCapability(double[] data, double? usl, double? lsl, double? target = null)
        => CallNative(NativeInterop.uanalytics_process_capability,
            new { data, usl, lsl, target });

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

    public JsonElement DetectChangepoints(double[] data, string? penalty = null, int? minSegmentLen = null)
        => CallNative(NativeInterop.uanalytics_detect_changepoints,
            new { data, penalty, min_segment_len = minSegmentLen });

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
