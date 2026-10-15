using System.Text.Json;
using System.Text.Json.Nodes;
using Xunit;

namespace UAnalytics.Tests;

/// <summary>
/// The keys and names each C entry point reads, as the engine's own tests measure them from
/// the entry points (<c>src/ffi/request_keys.json</c>), and the same places read off a request
/// the client wrote. A place is written <c>$.limits.ucl</c>, <c>$.rules[]="SixTrend"</c> (a
/// name), <c>$.observation(truncation="time").end</c> (a key of one variant).
/// </summary>
internal static class RequestKeys
{
    internal static JsonObject EntryPoints { get; } = Load();

    private static JsonObject Load()
    {
        var path = Path.Combine(AppContext.BaseDirectory, "request_keys.json");
        Assert.True(File.Exists(path), $"{path} is not copied next to the tests");
        return JsonNode.Parse(File.ReadAllText(path))!["entry_points"]!.AsObject();
    }

    internal static JsonObject Schema(string entry)
        => EntryPoints[entry]?.AsObject() ?? throw new KeyNotFoundException($"the engine has no entry point {entry}");

    /// <summary>Every place a request to <paramref name="entry"/> can have.</summary>
    internal static HashSet<string> Places(string entry)
    {
        var places = new HashSet<string>();
        Expand(Schema(entry), "$", places);
        return places;
    }

    private static void Expand(JsonObject node, string path, HashSet<string> places)
    {
        if (node["properties"] is JsonObject properties)
        {
            foreach (var (key, value) in properties)
            {
                places.Add($"{path}.{key}");
                Expand(value!.AsObject(), $"{path}.{key}", places);
            }
        }
        if (node["variants"] is JsonObject variants)
        {
            var tag = node["tag"]!.GetValue<string>();
            foreach (var (name, variant) in variants)
            {
                var at = $"{path}({tag}=\"{name}\")";
                places.Add(at);
                Expand(variant!.AsObject(), at, places);
            }
        }
        if (node["items"] is JsonObject items)
            Expand(items, $"{path}[]", places);
        if (node["enum"] is JsonArray names)
        {
            foreach (var name in names)
                places.Add($"{path}=\"{name!.GetValue<string>()}\"");
        }
    }

    /// <summary>
    /// The places a written request has, into <paramref name="places"/>; anything the engine
    /// would not read — a key it does not have, a name it does not know, a shape it does not
    /// take — into <paramref name="problems"/>.
    /// </summary>
    internal static void Read(JsonObject node, JsonNode? value, string path, HashSet<string> places, List<string> problems)
    {
        if (node["properties"] is JsonObject properties)
        {
            if (value is not JsonObject o)
            {
                problems.Add($"{path}: the engine reads an object here, the client wrote {Show(value)}");
                return;
            }
            foreach (var (key, v) in o)
            {
                if (properties[key] is not JsonObject child)
                {
                    problems.Add($"{path}.{key}: the engine has no such key (it reads {string.Join(", ", properties.Select(p => p.Key))})");
                    continue;
                }
                places.Add($"{path}.{key}");
                Read(child, v, $"{path}.{key}", places, problems);
            }
        }
        else if (node["variants"] is JsonObject variants)
        {
            var tag = node["tag"]!.GetValue<string>();
            if (value is not JsonObject o || o[tag] is not JsonValue t || t.GetValueKind() != JsonValueKind.String)
            {
                problems.Add($"{path}: the engine reads an object named by \"{tag}\" here, the client wrote {Show(value)}");
                return;
            }
            var name = t.GetValue<string>();
            if (variants[name] is not JsonObject variant)
            {
                problems.Add($"{path}.{tag}: the engine has no \"{name}\"");
                return;
            }
            var at = $"{path}({tag}=\"{name}\")";
            places.Add(at);
            var rest = new JsonObject();
            foreach (var (k, v) in o)
            {
                if (k != tag)
                    rest[k] = v?.DeepClone();
            }
            Read(variant, rest, at, places, problems);
        }
        // A place may take a name or an array of other names (a named set, or its members spelt out).
        else if (node["items"] is JsonObject items && (value is JsonArray || node["enum"] is null))
        {
            if (value is not JsonArray a)
            {
                problems.Add($"{path}: the engine reads an array here, the client wrote {Show(value)}");
                return;
            }
            foreach (var item in a)
                Read(items, item, $"{path}[]", places, problems);
        }
        else if (node["enum"] is JsonArray names)
        {
            if (value is JsonValue s && s.GetValueKind() == JsonValueKind.String)
            {
                var name = s.GetValue<string>();
                if (names.Any(n => n!.GetValue<string>() == name))
                    places.Add($"{path}=\"{name}\"");
                else
                    problems.Add($"{path}: the engine has no \"{name}\" (it reads {string.Join(", ", names.Select(n => n!.GetValue<string>()))})");
            }
            else if (!(value is JsonValue n && n.GetValueKind() == JsonValueKind.Number && node["number"]?.GetValue<bool>() == true))
            {
                problems.Add($"{path}: the engine reads a name here, the client wrote {Show(value)}");
            }
        }
    }

    /// <summary>The names the engine reads at <paramref name="place"/> of a request to <paramref name="entry"/>.</summary>
    internal static string[] Names(string entry, string place)
        => [.. Places(entry)
            .Where(p => p.StartsWith($"{place}=\"", StringComparison.Ordinal))
            .Select(p => p[(place.Length + 2)..^1])
            .Order(StringComparer.Ordinal)];

    private static string Show(JsonNode? value) => value?.ToJsonString() ?? "null";
}
