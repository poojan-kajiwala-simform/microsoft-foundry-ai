using LocalRag.Core.Indexing;
using LocalRag.Core.Storage;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.Options;

namespace LocalRag.Core.Retrieval;

public sealed record RetrievedChunk(ChunkRecord Chunk, double Similarity, double FusedScore, int? VectorRank, int? KeywordRank);

/// <param name="TopSimilarity">Best cosine similarity among vector candidates; the confidence signal.</param>
/// <param name="Confident">False when the index is empty or <paramref name="TopSimilarity"/> is below the configured threshold.</param>
public sealed record RetrievalResult(IReadOnlyList<RetrievedChunk> Chunks, double TopSimilarity, bool Confident, bool IndexEmpty);

/// <summary>
/// Embed query → vector search (+ optional BM25) → reciprocal rank fusion → confidence gate.
/// Top-k alone always returns *something*; the gate is what lets the app say "I don't know".
/// </summary>
public sealed class Retriever(
    IEmbeddingGenerator<string, Embedding<float>> embedder,
    VectorIndex index,
    SqliteChunkRepository repository,
    IOptions<RagOptions> options)
{
    private const int RrfK = 60;
    private readonly RagOptions _options = options.Value;

    public async Task<RetrievalResult> RetrieveAsync(string query, string? sourcePrefix = null, CancellationToken ct = default)
    {
        if (!index.IsInitialized || index.Count == 0) return new RetrievalResult([], 0, false, IndexEmpty: true);

        // Embedding consistency: query and chunks must come from the same model.
        var configured = EmbeddingHelpers.ModelId(embedder, _options);
        var indexedWith = await repository.GetMetaAsync(SqliteChunkRepository.MetaEmbeddingModel, ct);
        if (indexedWith is not null && indexedWith != configured) throw new ReindexRequiredException(indexedWith, configured);

        var embedding = await embedder.GenerateAsync([query], cancellationToken: ct);
        var queryVector = EmbeddingHelpers.Normalize(embedding[0].Vector);
        if (queryVector.Length != index.Dimensions) throw new ReindexRequiredException($"{index.Dimensions}-dim vectors", configured);

        var vectorHits = await index.SearchAsync(queryVector, _options.CandidateCount, sourcePrefix, ct);
        var topSimilarity = vectorHits.Count == 0 ? 0 : vectorHits.Max(h => h.Similarity);

        var fused = new Dictionary<string, (ChunkRecord Chunk, double Score, int? VRank, int? KRank)>();
        for (var i = 0; i < vectorHits.Count; i++)
            fused[vectorHits[i].Chunk.Id] = (vectorHits[i].Chunk, 1.0 / (RrfK + i + 1), i + 1, null);

        if (_options.HybridSearch)
        {
            var keywordHits = index.KeywordSearch(query, _options.CandidateCount, sourcePrefix);
            for (var i = 0; i < keywordHits.Count; i++)
            {
                var (chunk, _) = keywordHits[i];
                var add = 1.0 / (RrfK + i + 1);
                fused[chunk.Id] = fused.TryGetValue(chunk.Id, out var e)
                    ? (e.Chunk, e.Score + add, e.VRank, i + 1)
                    : (chunk, add, null, i + 1);
            }
        }

        var ranked = fused.Values
            .OrderByDescending(e => e.Score)
            .Take(_options.TopK)
            .Select(e => new RetrievedChunk(
                e.Chunk,
                EmbeddingHelpers.Dot(queryVector, e.Chunk.Embedding.Span),
                e.Score, e.VRank, e.KRank))
            .ToList();

        return new RetrievalResult(ranked, topSimilarity, topSimilarity >= _options.MinSimilarity, IndexEmpty: false);
    }
}
