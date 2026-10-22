using System.Diagnostics;
using LocalRag.Core;
using LocalRag.Core.Chunking;
using LocalRag.Core.Extraction;
using LocalRag.Core.Generation;
using LocalRag.Core.Indexing;
using LocalRag.Core.Ingestion;
using LocalRag.Core.Retrieval;
using LocalRag.Core.Storage;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.AI.Evaluation;
using Microsoft.Extensions.AI.Evaluation.Quality;
using Microsoft.Extensions.Logging.Abstractions;

namespace LocalRag.Eval;

public sealed record GoldenQuestion(string Id, string Question, bool Answerable, string? ExpectedSource = null, string? Evidence = null, string[]? AnswerKeywords = null);

public sealed record GenRow(
    GoldenQuestion Q, string Answer, string Declined, double RetrievalMs, double? TtftMs, double TotalS,
    double? FragmentsPerSecond, bool? KeywordsOk, bool? CitedExpected, bool? Grounded, double? JudgeScore);

/// <summary>One complete pipeline (own SQLite file + index) so chunking/retrieval variants can be compared side by side on shared models.</summary>
public sealed class Stack
{
    public IChatClient Chat { get; }
    public IngestionPipeline Ingestion { get; }
    public Retriever Retriever { get; }
    public VectorIndex Index { get; } = new();
    private readonly RagOptions _options;
    private readonly PromptBuilder _prompts;

    public Stack(RagOptions options, IEmbeddingGenerator<string, Embedding<float>> embedder, IChatClient chat)
    {
        _options = options;
        Chat = chat;
        var opts = Microsoft.Extensions.Options.Options.Create(options);
        var repo = new SqliteChunkRepository(opts);
        IChunker chunker = options.Chunking == ChunkingStrategy.Fixed
            ? new FixedSizeChunker(options.ChunkSizeWords, options.ChunkOverlapWords)
            : new StructureAwareChunker(options.ChunkSizeWords, options.ChunkOverlapWords);
        Ingestion = new IngestionPipeline(embedder, repo, Index, new TextExtractor(), chunker, opts, NullLogger<IngestionPipeline>.Instance);
        Retriever = new Retriever(embedder, Index, repo, opts);
        _prompts = new PromptBuilder(opts);
    }

    public RagAnswerService Answers() =>
        new(Retriever, _prompts, Chat, Microsoft.Extensions.Options.Options.Create(_options));

}

public static class Groundedness
{
    private static readonly char[] Separators = [' ', '\n', '.', ',', ';', ':', '(', ')', '[', ']', '"', '?', '!'];

    /// <summary>
    /// Cheap, deterministic check: the answer cites at least one valid source, and most of its content words (numbers
    /// especially) appear in the retrieved context. Catches invented figures; does not understand paraphrase.
    /// </summary>
    public static bool Heuristic(string answer, IReadOnlyList<Citation> retrieved)
    {
        if (!System.Text.RegularExpressions.Regex.IsMatch(answer, @"\[\d+\]")) return false;
        var context = string.Join(' ', retrieved.Select(c => c.Text)).ToLowerInvariant();
        var words = answer.ToLowerInvariant().Split(Separators, StringSplitOptions.RemoveEmptyEntries)
            .Where(w => w.Length > 3 || w.Any(char.IsDigit)).Distinct().ToList();
        if (words.Count == 0) return false;
        return words.Count(w => context.Contains(w)) / (double)words.Count >= 0.6;
    }

    /// <summary>Microsoft.Extensions.AI.Evaluation groundedness (LLM-as-judge). Here the judge is the local chat model, so treat scores as indicative.</summary>
    public static async Task<double?> JudgeAsync(IChatClient judge, string question, string answer, string context)
    {
        try
        {
            var evaluator = new GroundednessEvaluator();
            var result = await evaluator.EvaluateAsync(
                [new ChatMessage(ChatRole.User, question)],
                new ChatResponse(new ChatMessage(ChatRole.Assistant, answer)),
                new ChatConfiguration(judge),
                additionalContext: [new GroundednessEvaluatorContext(context)]);
            return result.Get<NumericMetric>(GroundednessEvaluator.GroundednessMetricName).Value;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"  (judge failed: {ex.Message})");
            return null;
        }
    }
}

public static class Hardware
{
    /// <summary>Best-effort NVIDIA GPU name + VRAM via nvidia-smi; null when unavailable. Other GPUs/NPUs: use Task Manager.</summary>
    public static string? QueryNvidia()
    {
        try
        {
            using var p = Process.Start(new ProcessStartInfo("nvidia-smi", "--query-gpu=name,memory.used,memory.total --format=csv,noheader")
            { RedirectStandardOutput = true, RedirectStandardError = true, UseShellExecute = false, CreateNoWindow = true });
            if (p is null) return null;
            var output = p.StandardOutput.ReadToEnd().Trim();
            p.WaitForExit(3000);
            return p.ExitCode == 0 && output.Length > 0 ? output.Replace("\r", "").Replace("\n", "; ") : null;
        }
        catch { return null; }
    }
}
