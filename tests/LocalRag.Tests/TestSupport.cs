using LocalRag.Core;
using LocalRag.Core.Chunking;
using LocalRag.Core.Extraction;
using LocalRag.Core.Generation;
using LocalRag.Core.Indexing;
using LocalRag.Core.Ingestion;
using LocalRag.Core.Retrieval;
using LocalRag.Core.Storage;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.Extensions.Options;

namespace LocalRag.Tests;

/// <summary>Deterministic bag-of-words "embedding": texts sharing words are similar, unrelated texts are orthogonal. No model needed.</summary>
public sealed class HashEmbedder(string modelId = "fake-embed", int dims = 256) : IEmbeddingGenerator<string, Embedding<float>>
{
    private static readonly char[] Separators = [' ', '\n', '.', ',', '?', '#', '›', '-', ':', '[', ']', '(', ')', '"'];

    public int Calls { get; private set; }

    public Task<GeneratedEmbeddings<Embedding<float>>> GenerateAsync(IEnumerable<string> values, EmbeddingGenerationOptions? options = null, CancellationToken cancellationToken = default)
    {
        var result = new GeneratedEmbeddings<Embedding<float>>();
        foreach (var text in values)
        {
            Calls++;
            var v = new float[dims];
            foreach (var token in text.ToLowerInvariant().Split(Separators, StringSplitOptions.RemoveEmptyEntries))
            {
                if (token.Length < 3) continue; // ignore stop-ish words
                var h = 17;
                foreach (var c in token) h = unchecked(h * 31 + c);
                v[(h & int.MaxValue) % dims] += 1;
            }
            result.Add(new Embedding<float>(v));
        }
        return Task.FromResult(result);
    }

    public object? GetService(Type serviceType, object? serviceKey = null) =>
        serviceKey is null && serviceType == typeof(EmbeddingGeneratorMetadata) ? new EmbeddingGeneratorMetadata("fake", defaultModelId: modelId) : null;

    public void Dispose() { }
}

public sealed class ScriptedChat(params string[] fragments) : IChatClient
{
    public int Calls { get; private set; }
    public IList<ChatMessage>? LastMessages { get; private set; }

    public Task<ChatResponse> GetResponseAsync(IEnumerable<ChatMessage> messages, ChatOptions? options = null, CancellationToken cancellationToken = default) =>
        throw new NotSupportedException();

    public async IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(IEnumerable<ChatMessage> messages, ChatOptions? options = null,
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        Calls++;
        LastMessages = messages.ToList();
        foreach (var f in fragments) { await Task.Yield(); yield return new ChatResponseUpdate(ChatRole.Assistant, f); }
    }

    public object? GetService(Type serviceType, object? serviceKey = null) => null;
    public void Dispose() { }
}

/// <summary>A fully wired pipeline over a temp SQLite file and temp document folder.</summary>
public sealed class Rig : IDisposable
{
    public string Root { get; } = Path.Combine(Path.GetTempPath(), "localrag-tests-" + Guid.NewGuid().ToString("N"));
    public string Docs => Path.Combine(Root, "docs");
    public RagOptions Options { get; }
    public HashEmbedder Embedder { get; }
    public SqliteChunkRepository Repository { get; }
    public VectorIndex Index { get; } = new();
    public IngestionPipeline Ingestion { get; }
    public Retriever Retriever { get; }

    public Rig(Action<RagOptions>? configure = null, HashEmbedder? embedder = null, string? dbPath = null)
    {
        Directory.CreateDirectory(Docs);
        Options = new RagOptions { DatabasePath = dbPath ?? Path.Combine(Root, "rag.db"), MinSimilarity = 0.2, TopK = 3 };
        configure?.Invoke(Options);
        Embedder = embedder ?? new HashEmbedder();
        var opts = Microsoft.Extensions.Options.Options.Create(Options);
        Repository = new SqliteChunkRepository(opts);
        IChunker chunker = Options.Chunking == ChunkingStrategy.Fixed
            ? new FixedSizeChunker(Options.ChunkSizeWords, Options.ChunkOverlapWords)
            : new StructureAwareChunker(Options.ChunkSizeWords, Options.ChunkOverlapWords);
        Ingestion = new IngestionPipeline(Embedder, Repository, Index, new TextExtractor(), chunker, opts, NullLogger<IngestionPipeline>.Instance);
        Retriever = new Retriever(Embedder, Index, Repository, opts);
    }

    public RagAnswerService Answers(IChatClient chat) =>
        new(Retriever, new PromptBuilder(Microsoft.Extensions.Options.Options.Create(Options)), chat, Microsoft.Extensions.Options.Options.Create(Options));

    public string Write(string name, string content)
    {
        var path = Path.Combine(Docs, name);
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        File.WriteAllText(path, content);
        return path;
    }

    public void Dispose()
    {
        Microsoft.Data.Sqlite.SqliteConnection.ClearAllPools();
        try { Directory.Delete(Root, recursive: true); } catch { /* best effort */ }
    }
}
