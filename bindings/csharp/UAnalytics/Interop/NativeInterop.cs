using System.Runtime.InteropServices;

namespace UAnalytics.Interop;

internal static partial class NativeInterop
{
    private const string DllName = "u_analytics";

    [LibraryImport(DllName, StringMarshalling = StringMarshalling.Utf8)]
    public static partial int uanalytics_xbar_r_chart(string requestJson, out IntPtr resultPtr);

    [LibraryImport(DllName, StringMarshalling = StringMarshalling.Utf8)]
    public static partial int uanalytics_p_chart(string requestJson, out IntPtr resultPtr);

    [LibraryImport(DllName, StringMarshalling = StringMarshalling.Utf8)]
    public static partial int uanalytics_laney_p_chart(string requestJson, out IntPtr resultPtr);

    [LibraryImport(DllName, StringMarshalling = StringMarshalling.Utf8)]
    public static partial int uanalytics_process_capability(string requestJson, out IntPtr resultPtr);

    [LibraryImport(DllName, StringMarshalling = StringMarshalling.Utf8)]
    public static partial int uanalytics_percentile_capability(string requestJson, out IntPtr resultPtr);

    [LibraryImport(DllName, StringMarshalling = StringMarshalling.Utf8)]
    public static partial int uanalytics_gage_rr_xbar_r(string requestJson, out IntPtr resultPtr);

    [LibraryImport(DllName, StringMarshalling = StringMarshalling.Utf8)]
    public static partial int uanalytics_gage_rr_anova(string requestJson, out IntPtr resultPtr);

    [LibraryImport(DllName, StringMarshalling = StringMarshalling.Utf8)]
    public static partial int uanalytics_weibull_mle(string requestJson, out IntPtr resultPtr);

    [LibraryImport(DllName, StringMarshalling = StringMarshalling.Utf8)]
    public static partial int uanalytics_detect_changepoints(string requestJson, out IntPtr resultPtr);

    [LibraryImport(DllName, StringMarshalling = StringMarshalling.Utf8)]
    public static partial int uanalytics_correlation_matrix(string requestJson, out IntPtr resultPtr);

    [LibraryImport(DllName, StringMarshalling = StringMarshalling.Utf8)]
    public static partial int uanalytics_simple_regression(string requestJson, out IntPtr resultPtr);

    [LibraryImport(DllName, StringMarshalling = StringMarshalling.Utf8)]
    public static partial int uanalytics_fit_best(string requestJson, out IntPtr resultPtr);

    [LibraryImport(DllName)]
    public static partial void uanalytics_free_string(IntPtr ptr);

    [LibraryImport(DllName)]
    public static partial IntPtr uanalytics_version();
}
