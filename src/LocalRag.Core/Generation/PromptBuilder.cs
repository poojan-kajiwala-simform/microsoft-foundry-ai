using LocalRag.Core.Retrieval;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.Options;

namespace LocalRag.Core.Generation;

public sealed record BuiltPrompt(IList<ChatMessage> Messages, IReadOnlyList<RetrievedChunk> Included, int DroppedForBudget, int EstimatedContextTokens);

/// <summary>
/// Builds a grounded prompt that fits a token budget. Small local models have short usable context windows and
/// degrade when it's stuffed, so chunks are added in rank order until the budget is spent.
/// </summary>
public sealed class PromptBuilder(IOptions<RagOptions> options)
{
    public const string AbstainPhrase = "I don't have enough information in the provided documents to answer that.";

    private readonly RagOptions _options = options.Value;

    /// <summary>Rough estimate (~4 characters per token for English). Good enough for budgeting, not for billing.</summary>
    public static int EstimateTokens(string text) => (text.Length + 3) / 4;

    public BuiltPrompt Build(string question, IReadOnlyList<RetrievedChunk> chunks)
    {
        var budget = _options.MaxContextTokens;
        var included = new List<RetrievedChunk>();
        var used = 0;
        foreach (var chunk in chunks)
        {
            var cost = EstimateTokens(chunk.Chunk.Content) + 20; // + header overhead
            if (used + cost > budget && included.Count > 0) break;
            included.Add(chunk);
            used += cost;
        }

        var blocks = included.Select((c, i) =>
        {
            var label = c.Chunk.Section is null ? c.Chunk.Source : $"{c.Chunk.Source} › {c.Chunk.Section}";
            var body = c.Chunk.Content;
            // A single chunk bigger than the whole budget is truncated rather than silently blowing the context window.
            if (i == 0 && EstimateTokens(body) > budget) body = body[..(budget * 4)];
            return $"[{i + 1}] ({label})\n{body}";
        });

        var system = $"""
            You are a private knowledge assistant. Answer using ONLY the numbered context excerpts below.

            Rules:
            - After every claim, cite the excerpt(s) it came from like [1] or [2][3].
            - Do not use outside knowledge and do not guess.
            - If the excerpts do not contain the answer, reply exactly: "{AbstainPhrase}"
            - Be concise.

            Context:
            {string.Join("\n\n", blocks)}
            """;

        IList<ChatMessage> messages = [new(ChatRole.System, system), new(ChatRole.User, question)];
        return new BuiltPrompt(messages, included, chunks.Count - included.Count, used);
    }
}
