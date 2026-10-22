using System.Text.RegularExpressions;

namespace LocalRag.Core.Indexing;

/// <summary>Minimal in-memory BM25 over chunk text. Catches exact terms (SKUs, error codes, names) that embeddings blur.</summary>
internal sealed partial class Bm25Index
{
    private const double K1 = 1.2, B = 0.75;

    [GeneratedRegex(@"[\p{L}\p{N}]+")]
    private static partial Regex TokenPattern();

    private sealed record Doc(string Id, Dictionary<string, int> Terms, int Length);

    private readonly List<Doc> _docs = [];
    private readonly Dictionary<string, int> _docFrequency = new();
    private double _avgLength;

    internal static IEnumerable<string> Tokenize(string text) =>
        TokenPattern().Matches(text).Select(m => m.Value.ToLowerInvariant());

    public void Rebuild(IEnumerable<(string Id, string Text)> items)
    {
        _docs.Clear();
        _docFrequency.Clear();
        foreach (var (id, text) in items)
        {
            var terms = new Dictionary<string, int>();
            var length = 0;
            foreach (var token in Tokenize(text))
            {
                terms[token] = terms.GetValueOrDefault(token) + 1;
                length++;
            }
            _docs.Add(new Doc(id, terms, length));
            foreach (var term in terms.Keys) _docFrequency[term] = _docFrequency.GetValueOrDefault(term) + 1;
        }
        _avgLength = _docs.Count == 0 ? 0 : _docs.Average(d => d.Length);
    }

    /// <summary>Returns the best-scoring document ids (score &gt; 0 only), highest first.</summary>
    public List<(string Id, double Score)> Search(string query, int top, Func<string, bool>? accept = null)
    {
        var queryTerms = Tokenize(query).Distinct().ToList();
        var n = _docs.Count;
        var results = new List<(string, double)>();
        if (n == 0 || queryTerms.Count == 0) return results;

        foreach (var doc in _docs)
        {
            if (accept is not null && !accept(doc.Id)) continue;
            var score = 0.0;
            foreach (var term in queryTerms)
            {
                if (!doc.Terms.TryGetValue(term, out var tf)) continue;
                var df = _docFrequency[term];
                var idf = Math.Log(1 + (n - df + 0.5) / (df + 0.5));
                score += idf * tf * (K1 + 1) / (tf + K1 * (1 - B + B * doc.Length / _avgLength));
            }
            if (score > 0) results.Add((doc.Id, score));
        }

        return results.OrderByDescending(r => r.Item2).Take(top).ToList();
    }
}
