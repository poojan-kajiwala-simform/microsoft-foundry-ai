using LocalRag.Core.Storage;
using Microsoft.Extensions.VectorData;
using Microsoft.SemanticKernel.Connectors.InMemory;

namespace LocalRag.Core.Indexing;

public sealed record VectorHit(ChunkRecord Chunk, double Similarity);

/// <summary>
/// Search index over chunks: a <c>Microsoft.Extensions.VectorData</c> collection for vector search plus a BM25
/// index for keyword search. Swapping the in-memory connector for SQLite-vec, Qdrant, Azure AI Search, ... only
/// changes how <see cref="_store"/> is constructed. Persistence lives in <see cref="SqliteChunkRepository"/>.
/// </summary>
public sealed class VectorIndex
{
    private const string CollectionName = "chunks";

    private readonly object _gate = new();
    private readonly VectorStore _store = new InMemoryVectorStore();
    private readonly Dictionary<string, ChunkRecord> _byId = new();
    private readonly Dictionary<string, List<string>> _idsBySource = new();
    private readonly Bm25Index _bm25 = new();
    private bool _bm25Dirty;
    private VectorStoreCollection<string, ChunkRecord>? _collection;

    public int Dimensions { get; private set; }
    public int Count { get { lock (_gate) return _byId.Count; } }
    public bool IsInitialized => _collection is not null;

    /// <summary>(Re)creates an empty collection for vectors of the given size.</summary>
    public async Task InitializeAsync(int dimensions, CancellationToken ct = default)
    {
        var definition = new VectorStoreCollectionDefinition
        {
            Properties =
            [
                new VectorStoreKeyProperty(nameof(ChunkRecord.Id), typeof(string)),
                new VectorStoreDataProperty(nameof(ChunkRecord.Source), typeof(string)) { IsIndexed = true },
                new VectorStoreDataProperty(nameof(ChunkRecord.Section), typeof(string)),
                new VectorStoreDataProperty(nameof(ChunkRecord.ChunkIndex), typeof(int)),
                new VectorStoreDataProperty(nameof(ChunkRecord.Content), typeof(string)),
                new VectorStoreVectorProperty(nameof(ChunkRecord.Embedding), typeof(ReadOnlyMemory<float>), dimensions)
                {
                    DistanceFunction = DistanceFunction.CosineSimilarity,
                },
            ],
        };

        var collection = _store.GetCollection<string, ChunkRecord>(CollectionName, definition);
        await collection.EnsureCollectionDeletedAsync(ct);
        await collection.EnsureCollectionExistsAsync(ct);

        lock (_gate)
        {
            _collection = collection;
            Dimensions = dimensions;
            _byId.Clear();
            _idsBySource.Clear();
            _bm25Dirty = true;
        }
    }

    public async Task ClearAsync(CancellationToken ct = default)
    {
        if (_collection is null) return;
        await InitializeAsync(Dimensions, ct);
    }

    public async Task ReplaceDocumentAsync(string source, IEnumerable<ChunkRecord> chunks, CancellationToken ct = default)
    {
        var collection = _collection ?? throw new InvalidOperationException("Index not initialised.");
        var records = chunks.ToList();

        await RemoveDocumentAsync(source, ct);
        foreach (var record in records) await collection.UpsertAsync(record, ct);

        lock (_gate)
        {
            _idsBySource[source] = records.Select(r => r.Id).ToList();
            foreach (var record in records) _byId[record.Id] = record;
            _bm25Dirty = true;
        }
    }

    public async Task RemoveDocumentAsync(string source, CancellationToken ct = default)
    {
        if (_collection is null) return;
        List<string>? ids;
        lock (_gate)
        {
            if (!_idsBySource.Remove(source, out ids)) return;
            foreach (var id in ids) _byId.Remove(id);
            _bm25Dirty = true;
        }
        foreach (var id in ids) await _collection.DeleteAsync(id, ct);
    }

    /// <summary>Cosine-similarity search. <paramref name="sourcePrefix"/> is a metadata filter (exact file or folder prefix).</summary>
    public async Task<List<VectorHit>> SearchAsync(float[] query, int top, string? sourcePrefix = null, CancellationToken ct = default)
    {
        var collection = _collection;
        if (collection is null || Count == 0) return [];

        var options = new VectorSearchOptions<ChunkRecord>();
        if (!string.IsNullOrEmpty(sourcePrefix))
            options.Filter = r => r.Source.StartsWith(sourcePrefix);

        var hits = new List<VectorHit>();
        await foreach (var result in collection.SearchAsync(new ReadOnlyMemory<float>(query), top, options, ct))
            hits.Add(new VectorHit(result.Record, result.Score ?? 0));
        return hits;
    }

    public List<(ChunkRecord Chunk, double Score)> KeywordSearch(string query, int top, string? sourcePrefix = null)
    {
        lock (_gate)
        {
            if (_bm25Dirty)
            {
                _bm25.Rebuild(_byId.Values.Select(c => (c.Id, c.Content)));
                _bm25Dirty = false;
            }

            Func<string, bool>? accept = string.IsNullOrEmpty(sourcePrefix)
                ? null
                : id => _byId[id].Source.StartsWith(sourcePrefix, StringComparison.Ordinal);
            return _bm25.Search(query, top, accept).Select(r => (_byId[r.Id], r.Score)).ToList();
        }
    }
}
