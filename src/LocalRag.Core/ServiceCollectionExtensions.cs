using LocalRag.Core.Chunking;
using LocalRag.Core.Extraction;
using LocalRag.Core.Generation;
using LocalRag.Core.Indexing;
using LocalRag.Core.Ingestion;
using LocalRag.Core.Retrieval;
using LocalRag.Core.Storage;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

namespace LocalRag.Core;

public static class ServiceCollectionExtensions
{
    /// <summary>
    /// Registers the provider-neutral RAG pipeline. The host must also register an
    /// <c>IChatClient</c> and an <c>IEmbeddingGenerator&lt;string, Embedding&lt;float&gt;&gt;</c> (Foundry Local, Azure OpenAI, Ollama, ...).
    /// </summary>
    public static IServiceCollection AddLocalRagCore(this IServiceCollection services, IConfiguration configuration)
    {
        services.AddOptions<RagOptions>()
            .Bind(configuration.GetSection(RagOptions.SectionName))
            .Validate(o => o.ChunkSizeWords > 20 && o.ChunkOverlapWords >= 0 && o.ChunkOverlapWords < o.ChunkSizeWords,
                "Rag:ChunkOverlapWords must be smaller than Rag:ChunkSizeWords (> 20).")
            .Validate(o => o.TopK > 0 && o.CandidateCount >= o.TopK, "Rag:CandidateCount must be >= Rag:TopK > 0.")
            .Validate(o => o.MinSimilarity is >= -1 and <= 1, "Rag:MinSimilarity must be a cosine value in [-1, 1].")
            .Validate(o => o.EmbeddingBatchSize > 0, "Rag:EmbeddingBatchSize must be > 0.")
            .ValidateOnStart();

        services.AddSingleton<SqliteChunkRepository>();
        services.AddSingleton<VectorIndex>();
        services.AddSingleton<TextExtractor>();
        services.AddSingleton<IChunker>(sp =>
        {
            var o = sp.GetRequiredService<IOptions<RagOptions>>().Value;
            return o.Chunking == ChunkingStrategy.Fixed
                ? new FixedSizeChunker(o.ChunkSizeWords, o.ChunkOverlapWords)
                : new StructureAwareChunker(o.ChunkSizeWords, o.ChunkOverlapWords);
        });
        services.AddSingleton<IngestionPipeline>();
        services.AddSingleton<Retriever>();
        services.AddSingleton<PromptBuilder>();
        services.AddSingleton<RagAnswerService>();
        services.AddHostedService<IndexHydrator>();
        return services;
    }
}

/// <summary>Reloads the persisted chunks into the in-memory index at startup. Needs no model, so it is instant and works offline.</summary>
internal sealed class IndexHydrator(SqliteChunkRepository repository, VectorIndex index, ILogger<IndexHydrator> logger) : IHostedService
{
    public async Task StartAsync(CancellationToken ct)
    {
        await repository.EnsureCreatedAsync(ct);
        var dimensions = await repository.GetMetaAsync(SqliteChunkRepository.MetaDimensions, ct);
        if (!int.TryParse(dimensions, out var dims)) return;

        await index.InitializeAsync(dims, ct);
        var byDocument = new Dictionary<string, List<ChunkRecord>>();
        await foreach (var c in repository.LoadAllChunksAsync(ct))
        {
            if (!byDocument.TryGetValue(c.Source, out var list)) byDocument[c.Source] = list = [];
            list.Add(new ChunkRecord { Id = c.Id, Source = c.Source, Section = c.Section, ChunkIndex = c.ChunkIndex, Content = c.Content, Embedding = c.Embedding });
        }
        foreach (var (source, records) in byDocument) await index.ReplaceDocumentAsync(source, records, ct);
        logger.LogInformation("Loaded {Chunks} chunks from {Documents} documents into the index ({Dims} dimensions).", index.Count, byDocument.Count, dims);
    }

    public Task StopAsync(CancellationToken ct) => Task.CompletedTask;
}
