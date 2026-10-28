namespace LocalRag.Core;

public enum ChunkingStrategy
{
    /// <summary>Sliding window of N words with overlap. Ignores document structure.</summary>
    Fixed,

    /// <summary>Splits on headings/sections first, packs paragraphs within a section, falls back to a sliding window for oversized paragraphs.</summary>
    Structured,
}

/// <summary>Everything tunable, bound from the "Rag" configuration section.</summary>
public sealed class RagOptions
{
    public const string SectionName = "Rag";

    public string AppName { get; set; } = "LocalRagAssistant";

    // ---- Models (Foundry Local catalog aliases) ----
    public string ChatModelAlias { get; set; } = "phi-3.5-mini";
    public string EmbeddingModelAlias { get; set; } = "qwen3-embedding-0.6b";

    // ---- Chunking ----
    public ChunkingStrategy Chunking { get; set; } = ChunkingStrategy.Structured;
    /// <summary>Target chunk size in words (a word is ~1.3 tokens for English prose).</summary>
    public int ChunkSizeWords { get; set; } = 300;
    public int ChunkOverlapWords { get; set; } = 40;

    // ---- Retrieval ----
    /// <summary>Chunks that may reach the prompt (subject to <see cref="MaxContextTokens"/>).</summary>
    public int TopK { get; set; } = 4;
    /// <summary>Candidates pulled from each of the vector and keyword searches before fusion.</summary>
    public int CandidateCount { get; set; } = 20;
    /// <summary>Fuse BM25 keyword hits with vector hits (reciprocal rank fusion). Helps with IDs, codes and rare terms.</summary>
    public bool HybridSearch { get; set; } = true;
    /// <summary>
    /// If the best cosine similarity is below this, the app answers "I don't know" without calling the chat model.
    /// Model-specific: calibrate with the Eval project (it reports the score distribution for answerable vs. unanswerable questions).
    /// </summary>
    public double MinSimilarity { get; set; } = 0.40;

    /// <summary>
    /// Optional second way through the confidence gate: a BM25 score at or above this also counts as confident, so exact
    /// keyword queries are not refused when embeddings score them low. 0 (default) disables it; BM25 scores are
    /// corpus-dependent, so calibrate with the Eval project before enabling.
    /// </summary>
    public double MinKeywordScore { get; set; }

    // ---- Generation ----
    /// <summary>Token budget for retrieved context (estimated at ~4 chars/token). Keep well under the model's context window.</summary>
    public int MaxContextTokens { get; set; } = 1100;
    public int MaxAnswerTokens { get; set; } = 300;
    /// <summary>Total sequence length (prompt + answer) the chat model allocates its KV cache for. Must exceed context + answer budgets; lower it if model loading runs out of memory.</summary>
    public int MaxSequenceLength { get; set; } = 2048;
    public float Temperature { get; set; } = 0.1f;

    // ---- Ingestion / storage ----
    public string DocumentsFolder { get; set; } = "./Documents";
    public string DatabasePath { get; set; } = "data/rag.db";
    public long MaxFileBytes { get; set; } = 20 * 1024 * 1024;
    public int MaxChunksPerDocument { get; set; } = 2000;
    public int EmbeddingBatchSize { get; set; } = 16;

    // ---- Foundry Local specifics (read by LocalRag.Foundry only) ----
    /// <summary>Set false to load only the embedding model (ingestion/retrieval-only runs, or machines too small for the chat model).</summary>
    public bool LoadChatModel { get; set; } = true;

    /// <summary>Download/register hardware execution providers (GPU/NPU) at startup. Extra download; only helps on supported hardware.</summary>
    public bool RegisterExecutionProviders { get; set; }
}
