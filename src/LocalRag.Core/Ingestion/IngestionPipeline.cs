using System.Diagnostics;
using System.Security.Cryptography;
using LocalRag.Core.Chunking;
using LocalRag.Core.Extraction;
using LocalRag.Core.Indexing;
using LocalRag.Core.Storage;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

namespace LocalRag.Core.Ingestion;

/// <summary>
/// Ingestion: parse → chunk → embed → store. Idempotent: files are keyed by relative path and a SHA-256 of their
/// bytes, so re-running only touches files that changed (or whose chunking/embedding settings changed), and
/// optionally removes documents whose source file disappeared. One bad file never aborts the run.
/// </summary>
public sealed class IngestionPipeline(
    IEmbeddingGenerator<string, Embedding<float>> embedder,
    SqliteChunkRepository repository,
    VectorIndex index,
    TextExtractor extractor,
    IChunker chunker,
    IOptions<RagOptions> options,
    ILogger<IngestionPipeline> logger)
{
    private readonly RagOptions _options = options.Value;
    private readonly SemaphoreSlim _runGate = new(1, 1);

    public string ConfigSignature =>
        $"{EmbeddingHelpers.ModelId(embedder, _options)}|{_options.Chunking}|{_options.ChunkSizeWords}|{_options.ChunkOverlapWords}";

    /// <summary>Syncs a folder: adds new files, updates changed ones, skips unchanged ones and removes vanished ones.</summary>
    public Task<IngestReport> SyncFolderAsync(string folder, CancellationToken ct = default)
    {
        var root = Path.GetFullPath(folder);
        var files = Directory.EnumerateFiles(root, "*", SearchOption.AllDirectories)
            .Where(TextExtractor.IsSupported)
            .Select(f => new SourceFile(Path.GetRelativePath(root, f).Replace('\\', '/'), f))
            .ToList();
        return IngestAsync(files, removeMissing: true, ct);
    }

    public async Task<IngestReport> IngestAsync(IReadOnlyList<SourceFile> files, bool removeMissing, CancellationToken ct = default)
    {
        if (!await _runGate.WaitAsync(0, ct)) throw new IngestionBusyException();
        try
        {
            return await RunAsync(files, removeMissing, ct);
        }
        finally
        {
            _runGate.Release();
        }
    }

    private async Task<IngestReport> RunAsync(IReadOnlyList<SourceFile> files, bool removeMissing, CancellationToken ct)
    {
        var total = Stopwatch.StartNew();
        var embeddingTime = TimeSpan.Zero;
        int added = 0, updated = 0, unchanged = 0, removed = 0, chunkCount = 0;
        var failures = new List<IngestFailure>();

        await repository.EnsureCreatedAsync(ct);
        var modelId = EmbeddingHelpers.ModelId(embedder, _options);
        var reindexed = await ResetIfEmbeddingModelChangedAsync(modelId, ct);
        var existing = (await repository.GetDocumentsAsync(ct)).ToDictionary(d => d.Source, StringComparer.Ordinal);

        foreach (var file in files)
        {
            ct.ThrowIfCancellationRequested();
            try
            {
                var info = new FileInfo(file.FullPath);
                if (info.Length > _options.MaxFileBytes)
                {
                    failures.Add(new(file.Source, $"File is {info.Length / 1024 / 1024} MB; the limit is {_options.MaxFileBytes / 1024 / 1024} MB."));
                    continue;
                }

                var hash = await HashAsync(file.FullPath, ct);
                existing.TryGetValue(file.Source, out var previous);
                if (previous is not null && previous.Hash == hash && previous.ConfigSignature == ConfigSignature)
                {
                    unchanged++;
                    continue;
                }

                string text;
                try { text = await extractor.ExtractTextAsync(file.FullPath, ct); }
                catch (Exception ex) when (ex is not OperationCanceledException)
                {
                    failures.Add(new(file.Source, $"Could not be parsed: {ex.Message}"));
                    continue;
                }

                if (string.IsNullOrWhiteSpace(text))
                {
                    failures.Add(new(file.Source, "No extractable text (scanned/image-only document?)."));
                    continue;
                }

                var pieces = chunker.Chunk(text);
                if (pieces.Count > _options.MaxChunksPerDocument)
                {
                    failures.Add(new(file.Source, $"Too large: {pieces.Count} chunks exceeds the limit of {_options.MaxChunksPerDocument}."));
                    continue;
                }

                var sw = Stopwatch.StartNew();
                var vectors = await EmbedAsync(pieces.Select(p => p.Content).ToList(), ct);
                embeddingTime += sw.Elapsed;

                await EnsureIndexReadyAsync(modelId, vectors[0].Length, ct);

                var records = pieces.Select((p, i) => new ChunkRecord
                {
                    Id = $"{file.Source}#{hash[..12]}#{p.Index}",
                    Source = file.Source,
                    Section = p.Section,
                    ChunkIndex = p.Index,
                    Content = p.Content,
                    Embedding = vectors[i],
                }).ToList();

                var document = new IndexedDocument(file.Source, hash, ConfigSignature, records.Count, DateTime.UtcNow);
                await repository.ReplaceDocumentAsync(document, records.Select(ToStored).ToList(), ct);
                await index.ReplaceDocumentAsync(file.Source, records, ct);

                chunkCount += records.Count;
                if (previous is null) added++; else updated++;
                logger.LogInformation("Indexed {Source}: {Chunks} chunks ({Kind}).", file.Source, records.Count, previous is null ? "new" : "updated");
            }
            catch (Exception ex) when (ex is not OperationCanceledException and not ModelNotReadyException)
            {
                logger.LogWarning(ex, "Failed to ingest {Source}.", file.Source);
                failures.Add(new(file.Source, ex.Message));
            }
        }

        if (removeMissing)
        {
            var present = files.Select(f => f.Source).ToHashSet(StringComparer.Ordinal);
            foreach (var gone in existing.Keys.Where(s => !present.Contains(s)))
            {
                await repository.DeleteDocumentAsync(gone, ct);
                await index.RemoveDocumentAsync(gone, ct);
                removed++;
                logger.LogInformation("Removed {Source} (source file no longer exists).", gone);
            }
        }

        return new IngestReport(added, updated, unchanged, removed, chunkCount, reindexed, failures, total.Elapsed, embeddingTime);
    }

    public async Task<bool> RemoveAsync(string source, CancellationToken ct = default)
    {
        // Serialised with ingestion runs so a delete cannot race a sync's view of the existing documents.
        await _runGate.WaitAsync(ct);
        try
        {
            var known = (await repository.GetDocumentsAsync(ct)).Any(d => d.Source == source);
            if (!known) return false;
            await repository.DeleteDocumentAsync(source, ct);
            await index.RemoveDocumentAsync(source, ct);
            return true;
        }
        finally { _runGate.Release(); }
    }

    /// <summary>Vectors from different embedding models are not comparable: if the model changed, drop everything and rebuild.</summary>
    private async Task<bool> ResetIfEmbeddingModelChangedAsync(string modelId, CancellationToken ct)
    {
        var indexedWith = await repository.GetMetaAsync(SqliteChunkRepository.MetaEmbeddingModel, ct);
        if (indexedWith is null || indexedWith == modelId) return false;

        logger.LogWarning("Embedding model changed ('{Old}' -> '{New}'); rebuilding the whole index.", indexedWith, modelId);
        await repository.ClearAsync(ct);
        await index.ClearAsync(ct);
        return true;
    }

    private async Task EnsureIndexReadyAsync(string modelId, int dimensions, CancellationToken ct)
    {
        if (!index.IsInitialized || index.Dimensions != dimensions)
        {
            if (index.IsInitialized && index.Count > 0)
                throw new InvalidOperationException($"Embedding dimension {dimensions} does not match the index ({index.Dimensions}).");
            await index.InitializeAsync(dimensions, ct);
        }
        await repository.SetMetaAsync(SqliteChunkRepository.MetaEmbeddingModel, modelId, ct);
        await repository.SetMetaAsync(SqliteChunkRepository.MetaDimensions, dimensions.ToString(), ct);
    }

    private async Task<List<float[]>> EmbedAsync(IReadOnlyList<string> texts, CancellationToken ct)
    {
        var result = new List<float[]>(texts.Count);
        for (var i = 0; i < texts.Count; i += _options.EmbeddingBatchSize)
        {
            var batch = texts.Skip(i).Take(_options.EmbeddingBatchSize).ToList();
            var embeddings = await embedder.GenerateAsync(batch, cancellationToken: ct);
            if (embeddings.Count != batch.Count)
                throw new InvalidOperationException($"Embedding model returned {embeddings.Count} vectors for {batch.Count} inputs.");
            result.AddRange(embeddings.Select(e => EmbeddingHelpers.Normalize(e.Vector)));
        }
        return result;
    }

    private static StoredChunk ToStored(ChunkRecord r) =>
        new(r.Id, r.Source, r.Section, r.ChunkIndex, r.Content, r.Embedding.ToArray());

    private static async Task<string> HashAsync(string path, CancellationToken ct)
    {
        await using var stream = File.OpenRead(path);
        return Convert.ToHexString(await SHA256.HashDataAsync(stream, ct)).ToLowerInvariant();
    }
}
