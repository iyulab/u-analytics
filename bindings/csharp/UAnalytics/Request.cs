using System.Globalization;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.Json.Serialization.Metadata;

namespace UAnalytics;

/// <summary>
/// Builds request bodies as JSON nodes — no reflection, so trimmed and NativeAOT hosts can
/// call the engine. JSON has no NaN or infinity, so such a value cannot reach the engine;
/// it is refused here with the engine's own reason (<c>value_not_finite</c>), the argument's
/// path as <c>parameter</c> and the element's position in its innermost array as
/// <c>index</c> — the shape the WebAssembly binding reports for the same input.
/// </summary>
internal static class Request
{
    internal static JsonNode Num(string parameter, double value, int? index = null)
    {
        if (double.IsFinite(value))
            return JsonValue.Create(value);
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

    internal static JsonNode? Num(string parameter, double? value)
        => value is { } v ? Num(parameter, v) : null;

    internal static JsonArray Nums(string parameter, IReadOnlyList<double> values)
    {
        var array = new JsonArray();
        for (var i = 0; i < values.Count; i++)
            array.Add(Num(parameter, values[i], i));
        return array;
    }

    /// <summary>Rows of numbers; a bad value is at <c>parameter[row]</c>, <c>index</c> its column.</summary>
    internal static JsonArray Rows(string parameter, IReadOnlyList<IReadOnlyList<double>> rows)
    {
        var array = new JsonArray();
        for (var r = 0; r < rows.Count; r++)
            array.Add((JsonNode)Nums($"{parameter}[{r}]", rows[r]));
        return array;
    }

    internal static JsonArray Rows(string parameter, IReadOnlyList<IReadOnlyList<IReadOnlyList<double>>> blocks)
    {
        var array = new JsonArray();
        for (var b = 0; b < blocks.Count; b++)
            array.Add((JsonNode)Rows($"{parameter}[{b}]", blocks[b]));
        return array;
    }

    internal static JsonArray Counts(IReadOnlyList<ulong> values)
    {
        var array = new JsonArray();
        foreach (var v in values)
            array.Add((JsonNode)JsonValue.Create(v));
        return array;
    }

    internal static JsonArray Counts(IReadOnlyList<IReadOnlyList<ulong>> rows)
    {
        var array = new JsonArray();
        foreach (var row in rows)
            array.Add((JsonNode)Counts(row));
        return array;
    }

    internal static JsonArray Counts(IReadOnlyList<IReadOnlyList<long>> rows)
    {
        var array = new JsonArray();
        foreach (var row in rows)
        {
            var inner = new JsonArray();
            foreach (var v in row)
                inner.Add((JsonNode)JsonValue.Create(v));
            array.Add((JsonNode)inner);
        }
        return array;
    }

    /// <summary>An option in the engine's own vocabulary, written by the same mapping that reads it.</summary>
    internal static JsonNode Option<T>(T value, JsonTypeInfo<T> info)
        => JsonSerializer.SerializeToNode(value, info)
           ?? throw new InvalidOperationException($"{typeof(T).Name} has no JSON form.");

    internal static JsonArray? Rules(IReadOnlyList<RunRule>? rules)
    {
        if (rules is null)
            return null;
        var array = new JsonArray();
        foreach (var rule in rules)
            array.Add(Option(rule, AnalyticsJson.Default.RunRule));
        return array;
    }

    /// <summary>
    /// An object of the given members, leaving out the null ones — an absent option takes the
    /// engine's default.
    /// </summary>
    internal static JsonObject Body(params (string Key, JsonNode? Value)[] members)
    {
        var body = new JsonObject();
        foreach (var (key, value) in members)
        {
            if (value is not null)
                body[key] = value;
        }
        return body;
    }
}
