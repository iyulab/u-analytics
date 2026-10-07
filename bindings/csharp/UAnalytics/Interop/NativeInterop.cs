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
    public static partial int uanalytics_np_chart(string requestJson, out IntPtr resultPtr);

    [LibraryImport(DllName, StringMarshalling = StringMarshalling.Utf8)]
    public static partial int uanalytics_c_chart(string requestJson, out IntPtr resultPtr);

    [LibraryImport(DllName, StringMarshalling = StringMarshalling.Utf8)]
    public static partial int uanalytics_u_chart(string requestJson, out IntPtr resultPtr);

    [LibraryImport(DllName, StringMarshalling = StringMarshalling.Utf8)]
    public static partial int uanalytics_laney_u_chart(string requestJson, out IntPtr resultPtr);

    [LibraryImport(DllName, StringMarshalling = StringMarshalling.Utf8)]
    public static partial int uanalytics_g_chart(string requestJson, out IntPtr resultPtr);

    [LibraryImport(DllName, StringMarshalling = StringMarshalling.Utf8)]
    public static partial int uanalytics_t_chart(string requestJson, out IntPtr resultPtr);

    [LibraryImport(DllName, StringMarshalling = StringMarshalling.Utf8)]
    public static partial int uanalytics_xbar_s_chart(string requestJson, out IntPtr resultPtr);

    [LibraryImport(DllName, StringMarshalling = StringMarshalling.Utf8)]
    public static partial int uanalytics_imr_chart(string requestJson, out IntPtr resultPtr);

    [LibraryImport(DllName, StringMarshalling = StringMarshalling.Utf8)]
    public static partial int uanalytics_run_rules(string requestJson, out IntPtr resultPtr);

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
    public static partial int uanalytics_weibull_mrr(string requestJson, out IntPtr resultPtr);

    [LibraryImport(DllName, StringMarshalling = StringMarshalling.Utf8)]
    public static partial int uanalytics_weibull_reliability(string requestJson, out IntPtr resultPtr);

    [LibraryImport(DllName, StringMarshalling = StringMarshalling.Utf8)]
    public static partial int uanalytics_boxcox_capability(string requestJson, out IntPtr resultPtr);

    [LibraryImport(DllName, StringMarshalling = StringMarshalling.Utf8)]
    public static partial int uanalytics_sigma_to_ppm(string requestJson, out IntPtr resultPtr);

    [LibraryImport(DllName, StringMarshalling = StringMarshalling.Utf8)]
    public static partial int uanalytics_ppm_to_sigma(string requestJson, out IntPtr resultPtr);

    [LibraryImport(DllName, StringMarshalling = StringMarshalling.Utf8)]
    public static partial int uanalytics_one_sample_t_test(string requestJson, out IntPtr resultPtr);

    [LibraryImport(DllName, StringMarshalling = StringMarshalling.Utf8)]
    public static partial int uanalytics_two_sample_t_test(string requestJson, out IntPtr resultPtr);

    [LibraryImport(DllName, StringMarshalling = StringMarshalling.Utf8)]
    public static partial int uanalytics_mann_whitney_u_test(string requestJson, out IntPtr resultPtr);

    [LibraryImport(DllName, StringMarshalling = StringMarshalling.Utf8)]
    public static partial int uanalytics_paired_t_test(string requestJson, out IntPtr resultPtr);

    [LibraryImport(DllName, StringMarshalling = StringMarshalling.Utf8)]
    public static partial int uanalytics_wilcoxon_signed_rank_test(string requestJson, out IntPtr resultPtr);

    [LibraryImport(DllName, StringMarshalling = StringMarshalling.Utf8)]
    public static partial int uanalytics_jarque_bera_test(string requestJson, out IntPtr resultPtr);

    [LibraryImport(DllName, StringMarshalling = StringMarshalling.Utf8)]
    public static partial int uanalytics_shapiro_wilk_test(string requestJson, out IntPtr resultPtr);

    [LibraryImport(DllName, StringMarshalling = StringMarshalling.Utf8)]
    public static partial int uanalytics_mann_kendall_test(string requestJson, out IntPtr resultPtr);

    [LibraryImport(DllName, StringMarshalling = StringMarshalling.Utf8)]
    public static partial int uanalytics_one_way_anova(string requestJson, out IntPtr resultPtr);

    [LibraryImport(DllName, StringMarshalling = StringMarshalling.Utf8)]
    public static partial int uanalytics_kruskal_wallis_test(string requestJson, out IntPtr resultPtr);

    [LibraryImport(DllName, StringMarshalling = StringMarshalling.Utf8)]
    public static partial int uanalytics_levene_test(string requestJson, out IntPtr resultPtr);

    [LibraryImport(DllName, StringMarshalling = StringMarshalling.Utf8)]
    public static partial int uanalytics_bartlett_test(string requestJson, out IntPtr resultPtr);

    [LibraryImport(DllName, StringMarshalling = StringMarshalling.Utf8)]
    public static partial int uanalytics_chi_squared_goodness_of_fit(string requestJson, out IntPtr resultPtr);

    [LibraryImport(DllName, StringMarshalling = StringMarshalling.Utf8)]
    public static partial int uanalytics_chi_squared_independence(string requestJson, out IntPtr resultPtr);

    [LibraryImport(DllName, StringMarshalling = StringMarshalling.Utf8)]
    public static partial int uanalytics_fisher_exact_test(string requestJson, out IntPtr resultPtr);

    [LibraryImport(DllName, StringMarshalling = StringMarshalling.Utf8)]
    public static partial int uanalytics_bonferroni_correction(string requestJson, out IntPtr resultPtr);

    [LibraryImport(DllName, StringMarshalling = StringMarshalling.Utf8)]
    public static partial int uanalytics_benjamini_hochberg(string requestJson, out IntPtr resultPtr);

    [LibraryImport(DllName, StringMarshalling = StringMarshalling.Utf8)]
    public static partial int uanalytics_laplace_trend_test(string requestJson, out IntPtr resultPtr);

    [LibraryImport(DllName, StringMarshalling = StringMarshalling.Utf8)]
    public static partial int uanalytics_mil_hdbk_189_test(string requestJson, out IntPtr resultPtr);

    [LibraryImport(DllName, StringMarshalling = StringMarshalling.Utf8)]
    public static partial int uanalytics_power_law_process_fit(string requestJson, out IntPtr resultPtr);

    [LibraryImport(DllName, StringMarshalling = StringMarshalling.Utf8)]
    public static partial int uanalytics_detect_changepoints(string requestJson, out IntPtr resultPtr);

    [LibraryImport(DllName, StringMarshalling = StringMarshalling.Utf8)]
    public static partial int uanalytics_detect_changepoints_multi(string requestJson, out IntPtr resultPtr);

    [LibraryImport(DllName, StringMarshalling = StringMarshalling.Utf8)]
    public static partial int uanalytics_cusum(string requestJson, out IntPtr resultPtr);

    [LibraryImport(DllName, StringMarshalling = StringMarshalling.Utf8)]
    public static partial int uanalytics_ewma(string requestJson, out IntPtr resultPtr);

    [LibraryImport(DllName, StringMarshalling = StringMarshalling.Utf8)]
    public static partial int uanalytics_anderson_darling_normality(string requestJson, out IntPtr resultPtr);

    [LibraryImport(DllName, StringMarshalling = StringMarshalling.Utf8)]
    public static partial int uanalytics_estimate_period(string requestJson, out IntPtr resultPtr);

    [LibraryImport(DllName, StringMarshalling = StringMarshalling.Utf8)]
    public static partial int uanalytics_spectral_residual(string requestJson, out IntPtr resultPtr);

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
