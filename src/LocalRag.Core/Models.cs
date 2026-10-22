namespace LocalRag.Core;

/// <param name="Text">The exact chunk text that was given to the model, so a UI (or evaluator) can show/check what the citation points at.</param>
public sealed record Citation(int Index, string Source, string? Section, double Similarity, string Text);

public abstract record RagEvent;

/// <summary>Emitted once, before generation starts, with the chunks that made it into the prompt.</summary>
public sealed record RetrievedEvent(IReadOnlyList<Citation> Citations) : RagEvent;

/// <summary>One raw streamed fragment from the model (usually a sub-word token).</summary>
public sealed record TokenEvent(string Text) : RagEvent;

public sealed record DoneEvent(
    string Answer,
    IReadOnlyList<Citation> Retrieved,
    IReadOnlyList<int> CitedIndexes,
    string? AbstainReason,
    RagTimings Timings) : RagEvent;

public sealed record RagTimings(TimeSpan Retrieval, TimeSpan? FirstToken, TimeSpan Total, int StreamedFragments);

public static class AbstainReasons
{
    public const string NoDocuments = "no-documents";
    public const string LowConfidence = "low-confidence";
    public const string ModelDeclined = "model-declined";
}

public sealed record IngestFailure(string Source, string Reason);

public sealed record IngestReport(
    int Added,
    int Updated,
    int Unchanged,
    int Removed,
    int Chunks,
    bool Reindexed,
    IReadOnlyList<IngestFailure> Failures,
    TimeSpan Elapsed,
    TimeSpan EmbeddingTime);

public sealed record SourceFile(string Source, string FullPath);

public sealed record IndexedDocument(string Source, string Hash, string ConfigSignature, int ChunkCount, DateTime IndexedUtc);
