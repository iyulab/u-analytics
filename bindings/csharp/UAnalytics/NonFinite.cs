using System.Collections;
using System.Globalization;
using System.Reflection;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace UAnalytics;

/// <summary>
/// Finds a NaN or infinity in a request before it is serialized. JSON has no
/// such numbers, so one cannot reach the engine; the request is refused here
/// with the engine's own reason (<c>value_not_finite</c>), the argument's path
/// as <c>parameter</c> and the element's position as <c>index</c> — the shape
/// the WebAssembly binding reports for the same input.
/// </summary>
internal static class NonFinite
{
    internal static void Check(object request, JsonNamingPolicy naming)
    {
        if (Find(request, null, naming) is { } found)
        {
            var (parameter, index, value) = found;
            var where = index is { } i ? $"{parameter}[{i}]" : parameter;
            var body = new JsonObject
            {
                ["error"] = $"{where}: expected a finite number, got {value.ToString(CultureInfo.InvariantCulture)}",
                ["code"] = "value_not_finite",
                ["index"] = index,
                ["parameter"] = parameter,
            };
            throw AnalyticsException.FromErrorBody(-3, body.ToJsonString());
        }
    }

    /// <summary>The first non-finite number under <paramref name="value"/>: its path, its index in the innermost array, and the value.</summary>
    private static (string Parameter, int? Index, double Value)? Find(object? value, string? path, JsonNamingPolicy naming)
    {
        switch (value)
        {
            case null or string:
                return null;
            case double d:
                return double.IsFinite(d) ? null : (path ?? "request", null, d);
            case float f:
                return float.IsFinite(f) ? null : (path ?? "request", null, f);
            case IEnumerable items:
            {
                var i = 0;
                foreach (var item in items)
                {
                    switch (item)
                    {
                        case double d when !double.IsFinite(d):
                            return (path ?? "request", i, d);
                        case float f when !float.IsFinite(f):
                            return (path ?? "request", i, f);
                        case double or float or null:
                            break;
                        default:
                            if (Find(item, $"{path}[{i}]", naming) is { } inner)
                                return inner;
                            break;
                    }
                    i++;
                }
                return null;
            }
        }
        var type = value.GetType();
        if (type.IsPrimitive || type.IsEnum || value is decimal)
            return null;
        foreach (var property in type.GetProperties(BindingFlags.Public | BindingFlags.Instance))
        {
            if (property.GetIndexParameters().Length > 0)
                continue;
            var name = naming.ConvertName(property.Name);
            var child = path is null ? name : $"{path}.{name}";
            if (Find(property.GetValue(value), child, naming) is { } found)
                return found;
        }
        return null;
    }
}
